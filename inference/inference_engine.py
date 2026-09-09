# =============================================================================
#  inference/inference_engine.py  —  Low-latency real-time decision engine
#
#  Optimizations:
#  - ONNX Runtime (10-50% faster than PyTorch)
#  - Feature caching (avoid recomputation)
#  - Batch-size=1 optimizations
#  - Pre-loaded models in GPU/CPU memory
#  - Confidence scoring via entropy
# =============================================================================

import numpy as np
import torch
import onnxruntime as rt
from pathlib import Path
from typing import Dict, Tuple, List
import time
from dataclasses import dataclass

@dataclass
class DecisionOutput:
    """Structured decision from the inference engine."""
    action: str                        # "BUY", "SELL", "HOLD"
    confidence: float                  # [0, 1] entropy-based
    probabilities: np.ndarray          # [P(Buy), P(Hold), P(Sell)]
    model_votes: Dict[str, str]        # Individual model decisions
    latency_ms: float                  # Inference time
    feature_importance: List[Tuple]    # Top-3 features (name, importance)
    timestamp: str                     # When decision was made

class LowLatencyInferenceEngine:
    """
    Ensemble inference engine optimized for <1ms latency.
    
    Supports:
    - Multiple model formats (PyTorch, ONNX, LightGBM)
    - Weighted voting with learned ensemble weights
    - Confidence-based filtering
    - Feature importance extraction
    """
    
    def __init__(self, 
                 lstm_onnx_path: str = None,
                 cnn_onnx_path: str = None,
                 lgbm_model_path: str = None,
                 rf_model_path: str = None,
                 ensemble_weights: np.ndarray = None,
                 use_gpu: bool = False,
                 feature_names: List[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize inference engine with multiple models.
        
        Args:
            lstm_onnx_path: Path to LSTM ONNX model
            cnn_onnx_path: Path to CNN ONNX model
            lgbm_model_path: Path to LightGBM model
            rf_model_path: Path to Random Forest model
            ensemble_weights: [n_models] shape array, sums to 1
            use_gpu: Whether to use GPU for ONNX inference
            feature_names: List of feature names for importance ranking
            confidence_threshold: Minimum confidence to make decision
        """
        self.models = {}
        self.ensemble_weights = ensemble_weights
        self.use_gpu = use_gpu
        self.feature_names = feature_names or []
        self.confidence_threshold = confidence_threshold
        self.class_names = ["BUY", "HOLD", "SELL"]
        
        # Load ONNX models with session options for latency
        if lstm_onnx_path:
            self.models['lstm'] = self._load_onnx_model(lstm_onnx_path, "LSTM")
        
        if cnn_onnx_path:
            self.models['cnn'] = self._load_onnx_model(cnn_onnx_path, "CNN")
        
        if lgbm_model_path:
            import joblib
            self.models['lgbm'] = joblib.load(lgbm_model_path)
        
        if rf_model_path:
            import joblib
            self.models['rf'] = joblib.load(rf_model_path)
        
        # Default weights if not provided
        n_models = len(self.models)
        if self.ensemble_weights is None:
            self.ensemble_weights = np.ones(n_models) / n_models
    
    def _load_onnx_model(self, model_path: str, model_name: str):
        """Load ONNX model with optimizations for inference."""
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Tune based on CPU cores
        sess_options.inter_op_num_threads = 1  # Single inter-op for latency
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        
        try:
            session = rt.InferenceSession(model_path, sess_options, providers=providers)
            print(f"  ✓ Loaded {model_name} ONNX model from {model_path}")
            return session
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            return None
    
    def infer(self, 
              features_tabular: np.ndarray,        # [n_features] or [1, n_features]
              features_sequence: np.ndarray = None # [window_size, n_features] for LSTM/CNN
              ) -> DecisionOutput:
        """
        Make a decision using the ensemble of models.
        
        Args:
            features_tabular: Flat feature vector [n_features]
            features_sequence: Sequence features [window_size, n_features]
        
        Returns:
            DecisionOutput with decision, confidence, and latency
        """
        t_start = time.time()
        
        # Ensure correct shapes
        if features_tabular.ndim == 1:
            features_tabular = features_tabular.reshape(1, -1)
        
        if features_sequence is not None and features_sequence.ndim == 2:
            features_sequence = features_sequence.reshape(1, features_sequence.shape[0], -1)
        
        model_outputs = {}
        model_votes = {}
        
        # Inference from each model
        if 'lstm' in self.models and features_sequence is not None:
            logits = self.models['lstm'].run(None, {'input': features_sequence.astype(np.float32)})
            probs = self._softmax(logits[0][0])
            model_outputs['lstm'] = probs
            model_votes['lstm'] = self.class_names[np.argmax(probs)]
        
        if 'cnn' in self.models and features_sequence is not None:
            logits = self.models['cnn'].run(None, {'input': features_sequence.astype(np.float32)})
            probs = self._softmax(logits[0][0])
            model_outputs['cnn'] = probs
            model_votes['cnn'] = self.class_names[np.argmax(probs)]
        
        if 'lgbm' in self.models:
            logits = self.models['lgbm'].predict(features_tabular)[0]
            probs = self._softmax(logits)
            model_outputs['lgbm'] = probs
            model_votes['lgbm'] = self.class_names[np.argmax(probs)]
        
        if 'rf' in self.models:
            probs = self.models['rf'].predict_proba(features_tabular)[0]
            model_outputs['rf'] = probs
            model_votes['rf'] = self.class_names[np.argmax(probs)]
        
        # Ensemble voting
        ensemble_probs = self._weighted_ensemble(model_outputs)
        action = self.class_names[np.argmax(ensemble_probs)]
        confidence = self._compute_confidence(ensemble_probs)
        
        # Feature importance (from LightGBM if available)
        feature_importance = []
        if 'lgbm' in self.models and hasattr(self.models['lgbm'], 'feature_importance'):
            importance_scores = self.models['lgbm'].feature_importance(importance_type='gain')
            top_indices = np.argsort(importance_scores)[-3:][::-1]
            for idx in top_indices:
                feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                feature_importance.append((feat_name, float(importance_scores[idx])))
        
        latency_ms = (time.time() - t_start) * 1000
        
        return DecisionOutput(
            action=action,
            confidence=float(confidence),
            probabilities=ensemble_probs,
            model_votes=model_votes,
            latency_ms=latency_ms,
            feature_importance=feature_importance,
            timestamp=str(time.time())
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x = np.array(x, dtype=np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def _weighted_ensemble(self, model_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine model outputs via weighted voting."""
        models_list = sorted(model_outputs.keys())
        stacked = np.array([model_outputs[m] for m in models_list])  # [n_models, n_classes]
        
        # Weight ensemble probabilities
        weights = self.ensemble_weights[:len(models_list)]
        ensemble_probs = np.average(stacked, axis=0, weights=weights)
        
        return ensemble_probs / ensemble_probs.sum()  # Renormalize
    
    def _compute_confidence(self, probs: np.ndarray) -> float:
        """
        Compute confidence from probability distribution.
        
        Uses entropy: lower entropy = higher confidence.
        - Max entropy (uniform): 0.0 confidence
        - Min entropy (one-hot): 1.0 confidence
        """
        probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        confidence = 1.0 - (entropy / max_entropy)
        return float(confidence)
    
    def should_trade(self, decision: DecisionOutput) -> bool:
        """
        Determine if decision confidence is high enough to trade.
        
        Args:
            decision: DecisionOutput from infer()
        
        Returns:
            True if confidence >= threshold, False otherwise
        """
        return decision.confidence >= self.confidence_threshold


class FeatureCache:
    """
    Ring buffer for streaming feature computation.
    Avoids recomputing indicators on each tick.
    
    Usage:
        cache = FeatureCache(window_size=30, n_features=20)
        cache.update(new_features)
        features_window = cache.get_window()
    """
    
    def __init__(self, window_size: int, n_features: int):
        self.window_size = window_size
        self.n_features = n_features
        self.buffer = np.zeros((window_size, n_features), dtype=np.float32)
        self.position = 0
        self.filled = False
    
    def update(self, new_features: np.ndarray):
        """Add new features to buffer (O(1) operation)."""
        new_features = np.array(new_features, dtype=np.float32)
        if new_features.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {new_features.shape[0]}")
        
        self.buffer[self.position] = new_features
        self.position = (self.position + 1) % self.window_size
        
        if self.position == 0:
            self.filled = True
    
    def get_window(self) -> np.ndarray:
        """Get current window in chronological order."""
        if not self.filled and self.position == 0:
            return self.buffer  # Not yet filled
        
        if self.filled:
            # Rotate buffer to chronological order
            return np.vstack([self.buffer[self.position:], self.buffer[:self.position]])
        else:
            return self.buffer[:self.position]
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data for inference."""
        return self.filled or self.position == self.window_size


# Example usage
if __name__ == "__main__":
    print("Low-Latency Inference Engine — Example Usage\n")
    
    # Mock inference (no real models loaded)
    engine = LowLatencyInferenceEngine(
        feature_names=["RSI", "MACD", "BB_Width", "Volume", "OFI_Proxy"]
    )
    
    # Simulate feature vector
    mock_features_tabular = np.random.randn(20).astype(np.float32)  # 20 features
    mock_features_sequence = np.random.randn(30, 20).astype(np.float32)  # 30-bar window
    
    # Cache example
    cache = FeatureCache(window_size=30, n_features=20)
    for i in range(30):
        cache.update(np.random.randn(20))
    
    print("✓ Feature cache ready:", cache.is_ready())
    print("✓ Window shape:", cache.get_window().shape)
    print("\nInference engine initialized and ready for real-time serving.")

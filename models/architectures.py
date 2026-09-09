# =============================================================================
#  models/architectures.py  —  LSTM, 1D-CNN, and LightGBM model definitions
#
#  Design constraints (for FPGA portability later):
#  - LSTM: unidirectional only (bidirectional cannot run causally)
#  - CNN: causal padding or same padding with no future context
#  - All activations: ReLU (maps to DSP48 on FPGA)
#  - No batch norm (stateful; hard to pipeline on FPGA — use layer norm)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FEATURES, LABELS, LSTM as LSTM_CFG, CNN as CNN_CFG, LGBM as LGBM_CFG


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — LSTM
# ════════════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """
    3-layer unidirectional LSTM for sequence classification.

    Architecture
    ------------
    Input  : (batch, window_size, n_features)
    LSTM   : 3 layers, hidden_size=64, dropout between layers
    Output : (batch, n_classes)  via linear head

    FPGA note
    ---------
    This maps to hls4ml's LSTM support (unidirectional only).
    hidden_size=64 keeps BRAM usage within Arty A7 budget.
    bidirectional=False is non-negotiable for causal live inference.
    """

    def __init__(
        self,
        n_features:   int = len(FEATURES["feature_cols"]),
        hidden_size:  int = LSTM_CFG["hidden_size"],
        num_layers:   int = LSTM_CFG["num_layers"],
        n_classes:    int = LABELS["n_classes"],
        dropout:      float = LSTM_CFG["dropout"],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,        # MUST be False for live inference
        )

        # Layer normalisation after LSTM (more FPGA-friendly than BatchNorm)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window_size, n_features)

        Returns
        -------
        logits : (batch, n_classes)
        """
        lstm_out, _ = self.lstm(x)              # (batch, window, hidden)
        last_hidden  = lstm_out[:, -1, :]       # take last timestep only
        normed       = self.layer_norm(last_hidden)
        logits       = self.classifier(normed)
        return logits                            # raw logits (no softmax — handled by loss)

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — 1D CNN
# ════════════════════════════════════════════════════════════════════════════════

class CNNModel(nn.Module):
    """
    3-layer 1D CNN for temporal pattern recognition.

    Architecture
    ------------
    Input  : (batch, window_size, n_features)
    Conv1D : 3 layers with increasing channels [64, 128, 64]
             kernel_size=3, causal padding (no future context leakage)
    GAP    : Global Average Pooling → (batch, 64)
    Head   : Linear → n_classes

    FPGA note
    ---------
    Conv1D maps naturally to MAC arrays in hls4ml.
    Causal padding (pad left only) ensures no future data is seen.
    Depthwise separable convolution could further reduce DSP usage.
    """

    def __init__(
        self,
        n_features:  int   = len(FEATURES["feature_cols"]),
        channels:    list  = CNN_CFG["channels"],
        kernel_size: int   = CNN_CFG["kernel_size"],
        n_classes:   int   = LABELS["n_classes"],
        dropout:     float = CNN_CFG["dropout"],
    ):
        super().__init__()
        self.kernel_size = kernel_size

        # Build conv blocks
        in_ch  = n_features
        layers = []
        for out_ch in channels:
            layers.extend([
                # Causal padding: pad (kernel_size-1) on left, 0 on right
                # Implemented as nn.ConstantPad1d below
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv_layers = nn.ModuleList()
        self.pad_size    = kernel_size - 1
        in_ch = n_features
        for out_ch in channels:
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size))
            in_ch = out_ch

        self.dropout     = nn.Dropout(dropout)
        self.out_channels = channels[-1]

        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_channels // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window_size, n_features)

        Returns
        -------
        logits : (batch, n_classes)
        """
        # Conv1D expects (batch, channels, length)
        x = x.permute(0, 2, 1)                             # → (batch, n_features, window)

        for conv in self.conv_layers:
            # Causal padding: add (kernel-1) zeros to the left
            x = F.pad(x, (self.pad_size, 0))
            x = F.relu(conv(x))
            x = self.dropout(x)
                                                            # shape stays: (batch, ch, window)

        # Global Average Pooling over time dimension
        x = x.mean(dim=2)                                   # (batch, channels)

        logits = self.classifier(x)
        return logits

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — LightGBM
# ════════════════════════════════════════════════════════════════════════════════

class LightGBMModel:
    """
    Gradient-boosted tree ensemble for tabular feature classification.

    Why LightGBM
    ------------
    - Decision trees compile to FPGA LUTs — no multiply needed
    - Single-bar inference (no window) → lowest possible latency
    - Feature importance scores for ablation study
    - Interpretable predictions for guide presentation

    FPGA note
    ---------
    LightGBM trees can be converted to C if-else code via
    `m2cgen` library, then synthesised as FPGA lookup logic.
    Each tree becomes a cascade of threshold comparisons.
    """

    def __init__(
        self,
        n_estimators:       int   = LGBM_CFG["n_estimators"],
        max_depth:          int   = LGBM_CFG["max_depth"],
        num_leaves:         int   = LGBM_CFG["num_leaves"],
        learning_rate:      float = LGBM_CFG["learning_rate"],
        subsample:          float = LGBM_CFG["subsample"],
        colsample_bytree:   float = LGBM_CFG["colsample_bytree"],
        min_child_samples:  int   = LGBM_CFG["min_child_samples"],
        class_weight:       str   = LGBM_CFG["class_weight"],
        n_jobs:             int   = LGBM_CFG["n_jobs"],
        verbose:            int   = LGBM_CFG["verbose"],
        n_classes:          int   = LABELS["n_classes"],
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            class_weight=class_weight,
            objective="multiclass",
            num_class=n_classes,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42,
        )
        self.n_classes     = n_classes
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        feature_names: list = None,
    ) -> "LightGBMModel":
        """
        Train with early stopping on validation set.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        return self.model.predict_proba(X)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        Return feature importances sorted descending.
        Use this for ablation study and presentation.
        """
        import pandas as pd
        imp = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path: Path = None) -> None:
        import joblib
        path = path or LGBM_CFG["model_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[LightGBM] Saved to {path}")

    def load(self, path: Path = None) -> "LightGBMModel":
        import joblib
        path = path or LGBM_CFG["model_path"]
        self.model = joblib.load(path)
        print(f"[LightGBM] Loaded from {path}")
        return self


# ════════════════════════════════════════════════════════════════════════════════
#  MODEL 4 — Ensemble (soft voting over LSTM + CNN + LightGBM)
# ════════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Soft-voting ensemble over three models.
    Averages softmax probabilities; takes argmax.

    Usage (inference only — no training):
        ensemble = EnsemblePredictor(lstm, cnn, lgbm)
        proba    = ensemble.predict_proba(X_seq, X_tab)
        decision = ensemble.predict(X_seq, X_tab)
    """

    def __init__(
        self,
        lstm_model:  LSTMModel,
        cnn_model:   CNNModel,
        lgbm_model:  LightGBMModel,
        weights:     tuple = (1.0, 1.0, 1.0),   # relative voting weights
        device:      str   = "cpu",
    ):
        self.lstm   = lstm_model.to(device).eval()
        self.cnn    = cnn_model.to(device).eval()
        self.lgbm   = lgbm_model
        self.weights = np.array(weights) / sum(weights)
        self.device  = device

    @torch.no_grad()
    def predict_proba(
        self,
        X_seq: torch.Tensor,   # (batch, window, features) for LSTM & CNN
        X_tab: np.ndarray,     # (batch, features) for LightGBM
    ) -> np.ndarray:
        """
        Returns probability array of shape (batch, n_classes).
        """
        # LSTM probabilities
        lstm_logits = self.lstm(X_seq.to(self.device))
        lstm_proba  = F.softmax(lstm_logits, dim=-1).cpu().numpy()

        # CNN probabilities
        cnn_logits  = self.cnn(X_seq.to(self.device))
        cnn_proba   = F.softmax(cnn_logits,  dim=-1).cpu().numpy()

        # LightGBM probabilities
        lgbm_proba  = self.lgbm.predict_proba(X_tab)

        # Weighted average
        ensemble_proba = (
            self.weights[0] * lstm_proba +
            self.weights[1] * cnn_proba  +
            self.weights[2] * lgbm_proba
        )
        return ensemble_proba

    def predict(
        self,
        X_seq: torch.Tensor,
        X_tab: np.ndarray,
        confidence_threshold: float = 0.65,
    ) -> np.ndarray:
        """
        Returns class predictions.
        If max probability < confidence_threshold, outputs HOLD (1).
        """
        proba = self.predict_proba(X_seq, X_tab)
        preds = np.argmax(proba, axis=1)
        # Apply confidence gate — low-confidence → HOLD
        max_proba = proba.max(axis=1)
        preds[max_proba < confidence_threshold] = 1   # force HOLD
        return preds


# ── Quick model summary ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    n_feat = len(FEATURES["feature_cols"])
    print(f"Number of features: {n_feat}")

    # LSTM
    lstm = LSTMModel(n_features=n_feat)
    print(f"\nLSTM parameters: {lstm.get_n_params():,}")
    x = torch.randn(4, 30, n_feat)
    out = lstm(x)
    print(f"LSTM output shape: {out.shape}")   # (4, 3)

    # CNN
    cnn = CNNModel(n_features=n_feat)
    print(f"\nCNN parameters: {cnn.get_n_params():,}")
    out = cnn(x)
    print(f"CNN output shape: {out.shape}")    # (4, 3)

    # LightGBM (just init check — no training here)
    lgbm = LightGBMModel()
    print(f"\nLightGBM: {lgbm.model.n_estimators} estimators")

    print("\nAll model architecture checks passed.")

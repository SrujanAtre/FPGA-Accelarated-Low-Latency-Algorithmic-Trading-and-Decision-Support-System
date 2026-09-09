# =============================================================================
#  examples/full_inference_example.py  —  End-to-end inference demo
#
#  Shows how to use:
#  - Low-latency inference engine
#  - Feature caching for streaming
#  - Risk management
#  - Decision explainability
#  - Backtesting
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.inference_engine import LowLatencyInferenceEngine, FeatureCache
from backtest.backtester import HFTBacktester
from utils.risk_and_explainability import RiskManager, DecisionExplainer
from models.multi_timeframe_ensemble import MultiTimeframeEnsemble


def demo_inference_engine():
    """
    Demo 1: Real-time inference with low-latency feature cache.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Low-Latency Inference Engine")
    print("=" * 70)
    
    # Initialize engine (in real scenario, load trained models)
    engine = LowLatencyInferenceEngine(
        feature_names=["RSI", "MACD", "BB_Width", "Volume", "OFI_Proxy",
                      "VWAP", "ATR", "ROC", "Momentum", "OBV",
                      "VolumeZscore", "HLRatio", "CloseNorm",
                      "MACDSignal", "MACDHist", "BBUpper", "BBMid",
                      "BBLower", "BBPct", "VWAPDev"]
    )
    
    # Feature cache (ring buffer for O(1) updates)
    cache = FeatureCache(window_size=30, n_features=20)
    
    print("\nSimulating real-time tick stream (100 bars)...")
    
    for i in range(100):
        # Simulate incoming features from data feed
        new_features = np.random.randn(20).astype(np.float32)
        cache.update(new_features)
        
        if cache.is_ready() and i % 10 == 0:  # Infer every 10 ticks
            # Make decision
            decision = engine.infer(
                features_tabular=new_features,
                features_sequence=cache.get_window()
            )
            
            print(f"\nBar {i}:")
            print(f"  Action: {decision.action}")
            print(f"  Confidence: {decision.confidence:.2%}")
            print(f"  Latency: {decision.latency_ms:.2f} ms")
            if decision.feature_importance:
                print(f"  Top features: {decision.feature_importance[:2]}")
    
    print("\n✓ Inference demo complete")


def demo_risk_management():
    """
    Demo 2: Confidence-aware position sizing with Kelly Criterion.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Risk Management & Position Sizing")
    print("=" * 70)
    
    risk_mgr = RiskManager(
        initial_capital=100_000,
        kelly_fraction=0.25,
        max_risk_per_trade=0.02
    )
    
    # Simulate trading results from backtest
    risk_mgr.update_metrics(
        win_rate=0.55,
        avg_win=0.02,  # 2% avg win
        avg_loss=0.01  # 1% avg loss
    )
    
    print("\nHistorical Metrics:")
    print(f"  Win Rate: {risk_mgr.win_rate:.1%}")
    print(f"  Avg Win: {risk_mgr.avg_win:.2%}")
    print(f"  Avg Loss: {risk_mgr.avg_loss:.2%}")
    
    # Different scenarios
    scenarios = [
        ("Low confidence", 100.0, 0.4, "stable"),
        ("Moderate confidence", 100.0, 0.65, "trending"),
        ("High confidence", 100.0, 0.82, "trending"),
    ]
    
    for scenario_name, price, confidence, regime in scenarios:
        decision = risk_mgr.size_position(
            current_price=price,
            decision_confidence=confidence,
            market_regime=regime
        )
        
        print(f"\n{scenario_name} (Confidence: {confidence:.2%}, Regime: {regime}):")
        print(f"  Position Size: {decision.position_size_fraction:.4f} ({decision.position_size_fraction*100:.2f}% of capital)")
        print(f"  Capital at Risk: ₹{decision.position_size_fraction * risk_mgr.initial_capital:,.2f}")
        print(f"  Stop Loss: ₹{decision.recommended_stop_loss:.2f}")
        print(f"  Take Profit: ₹{decision.take_profit:.2f}")
        print(f"  Risk Level: {decision.risk_level.value}")
    
    print("\n✓ Risk management demo complete")


def demo_explainability():
    """
    Demo 3: Decision explainability with feature importance.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Decision Explainability & Audit Trail")
    print("=" * 70)
    
    explainer = DecisionExplainer([
        "RSI", "MACD", "BB_Width", "Volume", "OFI_Proxy"
    ])
    
    # Simulate a trading decision
    explanation = explainer.explain(
        decision_action="BUY",
        decision_confidence=0.78,
        model_votes={
            'lstm': 'BUY',
            'cnn': 'BUY',
            'lgbm': 'HOLD',
            'rf': 'BUY'
        },
        feature_importance=[
            ('RSI', 0.35),
            ('MACD', 0.25),
            ('Volume', 0.20),
            ('OFI_Proxy', 0.15),
            ('BB_Width', 0.05)
        ],
        current_price=100.0,
        market_regime="trending"
    )
    
    explainer.print_explanation(explanation)
    
    # Audit trail
    print("\nAudit Trail (last 3 decisions):")
    audit = explainer.get_audit_trail(limit=3)
    for i, decision_record in enumerate(audit, 1):
        print(f"  {i}. {decision_record['action']} @ {decision_record['confidence']:.2%} confidence")
    
    print("\n✓ Explainability demo complete")


def demo_backtesting():
    """
    Demo 4: Walk-forward backtesting with comprehensive metrics.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Walk-Forward Backtesting")
    print("=" * 70)
    
    # Generate mock data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.randn(1000) * 0.05),
        'Low': prices - np.abs(np.random.randn(1000) * 0.05),
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, 1000)
    }, index=dates)
    
    # Generate mock trading decisions
    decisions = np.random.choice([0, 1, 2], size=1000, p=[0.1, 0.7, 0.2])
    confidences = np.random.rand(1000)
    
    # Run backtest
    backtester = HFTBacktester(
        initial_capital=100_000,
        commission_bps=2.0,
        slippage_bps=1.0
    )
    
    results = backtester.backtest(df, decisions, confidences)
    backtester.print_results(results)
    
    print("✓ Backtesting demo complete")


def demo_multi_timeframe_ensemble():
    """
    Demo 5: Multi-timeframe ensemble (Capstone innovation!).
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Multi-Timeframe Ensemble (Capstone Innovation)")
    print("=" * 70)
    
    # Create ensemble
    ensemble = MultiTimeframeEnsemble(
        timeframes=["1m", "5m", "15m"],
        weights=np.array([0.4, 0.35, 0.25]),
        aggregation_type="weighted_voting"
    )
    
    # Mock data for each timeframe
    mock_data = {
        "1m": pd.DataFrame({
            'Open': np.random.randn(30) + 100,
            'High': np.random.randn(30) + 100.5,
            'Low': np.random.randn(30) + 99.5,
            'Close': np.random.randn(30) + 100,
            'Volume': np.random.randint(1000, 5000, 30)
        }),
        "5m": pd.DataFrame({
            'Open': np.random.randn(6) + 100,
            'High': np.random.randn(6) + 100.5,
            'Low': np.random.randn(6) + 99.5,
            'Close': np.random.randn(6) + 100,
            'Volume': np.random.randint(5000, 20000, 6)
        }),
        "15m": pd.DataFrame({
            'Open': np.random.randn(2) + 100,
            'High': np.random.randn(2) + 100.5,
            'Low': np.random.randn(2) + 99.5,
            'Close': np.random.randn(2) + 100,
            'Volume': np.random.randint(15000, 50000, 2)
        })
    }
    
    # Make prediction
    decision = ensemble.predict(mock_data)
    
    print(ensemble.explain_decision(decision))
    
    print("✓ Multi-timeframe ensemble demo complete")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HFT TRADING SYSTEM — COMPREHENSIVE DEMO")
    print("=" * 70)
    
    try:
        demo_inference_engine()
    except Exception as e:
        print(f"⚠ Inference demo error: {e}")
    
    try:
        demo_risk_management()
    except Exception as e:
        print(f"⚠ Risk management demo error: {e}")
    
    try:
        demo_explainability()
    except Exception as e:
        print(f"⚠ Explainability demo error: {e}")
    
    try:
        demo_backtesting()
    except Exception as e:
        print(f"⚠ Backtesting demo error: {e}")
    
    try:
        demo_multi_timeframe_ensemble()
    except Exception as e:
        print(f"⚠ Multi-timeframe demo error: {e}")
    
    print("\n" + "=" * 70)
    print("All demos completed! 🚀")
    print("=" * 70)

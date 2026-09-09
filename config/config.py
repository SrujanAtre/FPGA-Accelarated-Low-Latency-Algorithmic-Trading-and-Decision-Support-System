# =============================================================================
#  config/config.py  —  Central configuration for HFT Decision System
#  All hyperparameters, paths, and settings in ONE place.
#  Change here; everything else picks it up automatically.
# =============================================================================

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data settings ─────────────────────────────────────────────────────────────
DATA = {
    "ticker":       "^NSEI",          # NIFTY 50 index
    "interval":     "1m",             # 1-minute bars (yfinance: 1m,2m,5m,15m,30m,60m,1d)
    "period":       "60d",            # yfinance max for 1m is 60 days (free tier)
    "raw_dir":      ROOT / "data" / "raw",
    "processed_dir":ROOT / "data" / "processed",
    "raw_file":     ROOT / "data" / "raw" / "nifty_1m.csv",
    "processed_file": ROOT / "data" / "processed" / "features.csv",
}

# ── Feature settings ──────────────────────────────────────────────────────────
FEATURES = {
    # Price-derived (all incremental — O(1) per tick in live path)
    "rsi_period":        14,
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":       9,
    "bb_period":         20,       # Bollinger Bands
    "bb_std":            2.0,
    "roc_period":        5,        # Rate of Change
    "momentum_period":   5,

    # Volume-derived
    "vwap_session_reset": True,    # Reset VWAP at 09:15 IST each day

    # Microstructure — OFI approximated from volume delta (no L2 data needed)
    "use_ofi_proxy":     True,     # True = use volume-based OFI proxy

    # Final feature list used as model input
    # (computed in features/feature_engineering.py)
    "feature_cols": [
        # Time-based features (with cyclical encoding)
        "hour", "minute", "day_of_week",
        "hour_sin", "hour_cos",
        "minute_sin", "minute_cos",
        # Price-derived
        "rsi", "macd", "macd_signal_line", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
        "vwap", "vwap_dev",         # deviation from VWAP
        "obv", "obv_norm",
        "roc", "momentum",
        "ofi_proxy",                 # (bid-ask vol proxy)
        "volume_zscore",             # volume anomaly
        "atr",                       # Average True Range (no sqrt problem)
        "close_norm",                # normalised close price
        "hl_ratio",                  # (high-low)/close — volatility proxy
    ],
}

# ── Label settings ────────────────────────────────────────────────────────────
LABELS = {
    "lookahead_bars":    3,        # predict N bars ahead
    "buy_threshold":     0.0002,    # +0.1% → BUY
    "sell_threshold":   -0.0002,    # -0.1% → SELL
    # else → HOLD
    # Class mapping: BUY=0, HOLD=1, SELL=2
    "class_names":  ["BUY", "HOLD", "SELL"],
    "n_classes":    3,
}

# ── Data split ────────────────────────────────────────────────────────────────
SPLIT = {
    "train_frac":   0.70,
    "val_frac":     0.15,
    # test = remaining 0.15
    # ALWAYS time-based (no shuffle) to prevent data leakage
}

# ── Sequence model settings (LSTM / CNN) ──────────────────────────────────────
SEQUENCE = {
    "window_size":  30,            # 30 bars lookback for LSTM and CNN
}

# ── LSTM hyperparameters ──────────────────────────────────────────────────────
LSTM = {
    "hidden_size":      64,
    "num_layers":       3,
    "dropout":          0.2,
    "bidirectional":    False,     # MUST be False for causal live inference
    "batch_size":       64,
    "epochs":           2,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "patience":         10,        # early stopping patience
    "grad_clip":        1.0,
    "model_path":       ROOT / "models" / "saved" / "lstm.pt",
    "onnx_path":        ROOT / "models" / "saved" / "lstm.onnx",
}

# ── CNN hyperparameters ───────────────────────────────────────────────────────
CNN = {
    "channels":         [64, 128, 64],   # conv layer output channels
    "kernel_size":      3,
    "dropout":          0.2,
    "batch_size":       64,
    "epochs":           20,
    "lr":               3e-4,
    # "lr":               1e-3,
    "weight_decay":     1e-4,
    "patience":         5,
    "grad_clip":        1.0,
    "model_path":       ROOT / "models" / "saved" / "cnn.pt",
    "onnx_path":        ROOT / "models" / "saved" / "cnn.onnx",
}

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGBM = {
    "n_estimators":     500,
    "max_depth":        6,
    "num_leaves":       31,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples":20,
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "verbose":          -1,
    "model_path":       ROOT / "models" / "saved" / "lgbm.pkl",
}

# ── Training settings ─────────────────────────────────────────────────────────
TRAINING = {
    "device":           "cpu",     # change to "cuda" if GPU available
    "seed":             42,
    "use_smote":        False,     # oversample minority classes (alternative: class weights)
    "use_class_weights":True,      # recommended over SMOTE for time-series
    "log_dir":          ROOT / "training" / "logs",
    "checkpoint_dir":   ROOT / "models" / "saved",
}

# ── Evaluation settings ───────────────────────────────────────────────────────
EVAL = {
    "metrics": ["accuracy", "macro_f1", "precision", "recall", "roc_auc"],
    "report_dir": ROOT / "evaluation" / "reports",
    "plot_dir":   ROOT / "evaluation" / "plots",
    "backtest_initial_capital": 100_000,   # INR
    "transaction_cost_bps":     1,          # 1 basis point per trade
}

# ── Inference (Real-time serving) settings ────────────────────────────────────
INFERENCE = {
    "use_onnx":                 True,       # ONNX Runtime (faster than PyTorch)
    "use_gpu":                  False,      # GPU for ONNX inference
    "ensemble_weights":         None,       # Learned weights (auto-tuned in training)
    "confidence_threshold":     0.55,       # Min confidence to trade
    "batch_size":               1,          # Single-sample inference for low latency
    "feature_cache_size":       30,         # Ring buffer for streaming features
}

# ── Backtesting settings ──────────────────────────────────────────────────────
BACKTEST = {
    "initial_capital":          100_000,
    "commission_bps":           2.0,        # 2 basis points per trade
    "slippage_bps":             1.0,        # 1 basis point slippage
    "max_position_size":        0.1,        # 10% of capital max per position
    "walk_forward_window":      252,        # 1 trading year
    "walk_forward_step":        63,         # 1 quarter roll
}

# ── Risk Management settings ──────────────────────────────────────────────────
RISK = {
    "kelly_fraction":           0.25,       # Use 25% of full Kelly
    "max_risk_per_trade":       0.02,       # Max 2% capital at risk
    "stop_loss_pct":            0.02,       # 2% stop loss
    "take_profit_pct":          0.05,       # 5% take profit target
    "adaptive_sizing":          True,       # Size positions by confidence
}

# ── Multi-timeframe ensemble settings ──────────────────────────────────────
MULTI_TIMEFRAME = {
    "enabled":                  True,
    "timeframes":               ["1m", "5m", "15m"],
    "weights":                  [0.4, 0.35, 0.25],  # 1m most important
    "aggregate_method":         "weighted_voting",  # or "stacking"
}

# ── Ensure all directories exist ─────────────────────────────────────────────
def make_dirs():
    for key in ["raw_dir", "processed_dir"]:
        DATA[key].mkdir(parents=True, exist_ok=True)
    for key in ["log_dir", "checkpoint_dir"]:
        TRAINING[key].mkdir(parents=True, exist_ok=True)
    for key in ["report_dir", "plot_dir"]:
        EVAL[key].mkdir(parents=True, exist_ok=True)
    
    # Add inference, backtest, risk directories
    (ROOT / "inference").mkdir(parents=True, exist_ok=True)
    (ROOT / "backtest").mkdir(parents=True, exist_ok=True)

make_dirs()

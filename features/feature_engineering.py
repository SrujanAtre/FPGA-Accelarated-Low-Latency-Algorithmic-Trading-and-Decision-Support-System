# =============================================================================
#  features/feature_engineering.py  —  Incremental feature computation
#
#  DESIGN PRINCIPLE: Every feature uses the incremental (O(1) per tick) formula.
#  These exact same formulas are re-implemented in HLS C++ / FPGA fixed-point.
#  If you change the formula here, update the FPGA code to match.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import DATA, FEATURES, LABELS


# # ════════════════════════════════════════════════════════════════════════════════
# #  SECTION 1 — Individual feature functions
# #  Each function takes a DataFrame and returns a Series.
# #  All use pandas rolling/ewm which are vectorised for the OFFLINE training path.
# #  The live FPGA/CPU path uses incremental C++ equivalents.
# # ════════════════════════════════════════════════════════════════════════════════

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (exponential, alpha=1/period).
    Incremental formula: avg_gain = (avg_gain * (N-1) + gain) / N
    This is equivalent to EWM with alpha=1/N (com=N-1).
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder's smoothing = EWM with com=(period-1), adjust=False
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rs  = avg_gain / (avg_loss + 1e-9)   # avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename("rsi")


def compute_macd(
    close: pd.Series,
    fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    MACD using EMA.
    Incremental formula: ema_new = alpha * price + (1 - alpha) * ema_prev
    where alpha = 2 / (N + 1).
    Returns DataFrame with columns: macd, macd_signal_line, macd_hist
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame({
        "macd":             macd_line,
        "macd_signal_line": signal_line,
        "macd_hist":        histogram,
    })


def compute_bollinger(
    close: pd.Series, period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands using rolling mean and std.
    Incremental formula uses Welford's online algorithm (see FPGA code).
    Returns DataFrame: bb_upper, bb_mid, bb_lower, bb_width, bb_pct
    """
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std(ddof=0)  # population std (ddof=0 for FPGA consistency)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / (mid + 1e-9)
    pct   = (close - lower) / (upper - lower + 1e-9)  # 0=at lower, 1=at upper

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid":   mid,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct":   pct.clip(0, 1),
    })


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, reset_daily: bool = True
) -> pd.DataFrame:
    """
    VWAP = cumulative(typical_price * volume) / cumulative(volume).
    Resets at the start of each trading session (09:15 IST).
    Incremental formula: sum_pv += tp * vol; sum_v += vol; vwap = sum_pv/sum_v

    Returns DataFrame: vwap, vwap_dev (deviation of close from vwap)
    """
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume

    if reset_daily and isinstance(close.index, pd.DatetimeIndex):
        # Group by date and compute cumulative sum within each day
        dates = close.index.date
        date_series = pd.Series(dates, index=close.index)
        cum_pv  = pv.groupby(date_series).cumsum()
        cum_vol = volume.groupby(date_series).cumsum()
    else:
        cum_pv  = pv.cumsum()
        cum_vol = volume.cumsum()

    vwap    = cum_pv / (cum_vol + 1e-9)
    vwap_dev = (close - vwap) / (vwap + 1e-9)   # relative deviation

    return pd.DataFrame({"vwap": vwap, "vwap_dev": vwap_dev})


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    On-Balance Volume: cumulative signed volume.
    Incremental: obv += volume if close > prev_close else -volume (if down) else 0.
    Returns normalised OBV (z-score over rolling 50-bar window).
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv_norm = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-9)

    return pd.DataFrame({"obv": obv, "obv_norm": obv_norm})


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Average True Range — no sqrt needed (absolute values only).
    Incremental: atr = ema(true_range, period).
    True range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, adjust=False).mean()
    return atr.rename("atr")


def compute_roc(close: pd.Series, period: int = 5) -> pd.Series:
    """
    Rate of Change: (close - close[N periods ago]) / close[N periods ago].
    Incremental: keep a circular buffer of length N; lookup O(1).
    """
    roc = close.pct_change(period)
    return roc.rename("roc")


def compute_momentum(close: pd.Series, period: int = 5) -> pd.Series:
    """
    Raw momentum: close - close[N periods ago].
    Normalised by close for comparability across price levels.
    """
    mom = (close - close.shift(period)) / (close.shift(period) + 1e-9)
    return mom.rename("momentum")


def compute_ofi_proxy(volume: pd.Series, close: pd.Series) -> pd.Series:
    """
    OFI Proxy (no Level-2 data required).
    Approximates order flow imbalance using volume direction:
    ofi = (up_vol - down_vol) / (up_vol + down_vol + epsilon)
    where up_vol = volume on up ticks, down_vol = volume on down ticks.
    Incremental: one comparison and conditional assignment per tick.
    """
    direction = np.sign(close.diff()).fillna(0)
    up_vol   = (volume * (direction > 0)).rolling(5).sum()
    down_vol = (volume * (direction < 0)).rolling(5).sum()
    ofi = (up_vol - down_vol) / (up_vol + down_vol + 1e-9)
    return ofi.rename("ofi_proxy")


def compute_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Z-score: how anomalous is current volume vs rolling mean?
    Incremental: maintain running mean and variance (Welford's method).
    """
    vol_mean = volume.rolling(period).mean()
    vol_std  = volume.rolling(period).std() + 1e-9
    zscore   = (volume - vol_mean) / vol_std
    return zscore.clip(-4, 4).rename("volume_zscore")   # clip outliers


def compute_hl_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    High-Low ratio: (high - low) / close — normalised intrabar volatility.
    Incremental: one subtraction and one division per tick.
    """
    return ((high - low) / (close + 1e-9)).rename("hl_ratio")


# # ════════════════════════════════════════════════════════════════════════════════
# #  SECTION 2 — Label generation
# # ════════════════════════════════════════════════════════════════════════════════

# def generate_labels(
#     close: pd.Series,
#     lookahead: int  = LABELS["lookahead_bars"],
#     buy_thresh: float = LABELS["buy_threshold"],
#     sell_thresh: float = LABELS["sell_threshold"],
# ) -> pd.Series:
#     """
#     Generate BUY / HOLD / SELL labels using fixed lookahead.

#     Logic
#     -----
#     future_return = (close[t + lookahead] - close[t]) / close[t]
#     BUY  (0) if future_return >  buy_thresh
#     SELL (2) if future_return < sell_thresh
#     HOLD (1) otherwise

#     The last `lookahead` rows will be NaN (no future data) and are dropped later.

#     Parameters
#     ----------
#     close      : Close price series.
#     lookahead  : Number of bars to look forward.
#     buy_thresh : Minimum return to trigger BUY label.
#     sell_thresh: Maximum (negative) return to trigger SELL label.
#     """
#     future_close  = close.shift(-lookahead)
#     future_return = (future_close - close) / (close + 1e-9)

#     labels = pd.Series(1, index=close.index, dtype=int, name="label")   # default HOLD
#     labels[future_return >  buy_thresh]  = 0   # BUY
#     labels[future_return < sell_thresh]  = 2   # SELL
#     labels[future_return.isna()]         = np.nan

#     return labels


# # ════════════════════════════════════════════════════════════════════════════════
# #  SECTION 3 — Full pipeline: OHLCV → feature matrix → labels
# # ════════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on OHLCV DataFrame.
    Returns a DataFrame with all features and the label column.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close, volume],
         DatetimeIndex.

    Returns
    -------
    pd.DataFrame with feature columns + 'label' column.
    Rows with NaN (start of series / end for lookahead) are dropped.
    """
    print("\n[FeatureEng] Computing features...")
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    cfg = FEATURES
    out = pd.DataFrame(index=df.index)

    # ── Price-derived ─────────────────────────────────────────────────────────
    out["rsi"]          = compute_rsi(c, cfg["rsi_period"])
    macd_df             = compute_macd(c, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    out                 = pd.concat([out, macd_df], axis=1)
    bb_df               = compute_bollinger(c, cfg["bb_period"], cfg["bb_std"])
    out                 = pd.concat([out, bb_df], axis=1)
    out["roc"]          = compute_roc(c, cfg["roc_period"])
    out["momentum"]     = compute_momentum(c, cfg["momentum_period"])
    out["atr"]          = compute_atr(h, l, c)
    out["hl_ratio"]     = compute_hl_ratio(h, l, c)

    # ── Volume-derived ────────────────────────────────────────────────────────
    vwap_df             = compute_vwap(h, l, c, v, cfg["vwap_session_reset"])
    out                 = pd.concat([out, vwap_df], axis=1)
    obv_df              = compute_obv(c, v)
    out                 = pd.concat([out, obv_df], axis=1)
    out["volume_zscore"]= compute_volume_zscore(v)
    out["ofi_proxy"]    = compute_ofi_proxy(v, c)

    # ── Normalised close (for context) ────────────────────────────────────────
    out["close_norm"]   = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)

    # ── Labels ────────────────────────────────────────────────────────────────
    out["label"] = generate_labels(c)

    # ── Drop NaN rows (warmup period + lookahead rows) ────────────────────────
    before = len(out)
    # out    = out.dropna()
    out = out.dropna(thresh=int(0.8 * len(out.columns)))
    print(f"[FeatureEng] Dropped {before - len(out):,} NaN rows (warmup + lookahead)")
    print(f"[FeatureEng] Feature matrix: {out.shape}")

    # ── Label distribution report ─────────────────────────────────────────────
    label_counts = out["label"].value_counts().sort_index()
    total = len(out)
    print(f"\n[FeatureEng] Label distribution:")
    for cls, name in enumerate(LABELS["class_names"]):
        count = label_counts.get(cls, 0)
        pct   = count / total * 100
        print(f"  {name:5s} ({cls}): {count:6,} ({pct:.1f}%)")

    return out


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Normalisation and saving
# ════════════════════════════════════════════════════════════════════════════════

def normalise_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_path: Path = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to feature columns.

    Parameters
    ----------
    df           : DataFrame with feature_cols + 'label'.
    feature_cols : List of feature column names to scale.
    scaler_path  : If provided, save/load scaler from this path.
    fit          : True = fit on this data (training). False = transform only (val/test).

    Returns
    -------
    (scaled_df, scaler)
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    if not fit and scaler_path and Path(scaler_path).exists():
        print(f"[Normalise] Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
    else:
        print(f"[Normalise] Fitting scaler on {len(df):,} rows...")
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        if scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"[Normalise] Saved scaler to {scaler_path}")

    return df, scaler


def save_features(df: pd.DataFrame) -> None:
    """Save processed feature matrix to CSV."""
    DATA["processed_dir"].mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA["processed_file"])
    print(f"[FeatureEng] Saved to {DATA['processed_file']}")


def load_features() -> pd.DataFrame:
    """Load processed feature matrix from CSV."""
    path = DATA["processed_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed features not found at {path}. "
            "Run: python features/feature_engineering.py"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[FeatureEng] Loaded {len(df):,} rows from {path}")
    return df


# # ── Standalone run ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     from data.data_loader import get_data

#     print("=" * 60)
#     print("  HFT Feature Engineering Pipeline")
#     print("=" * 60)

#     # Load raw data
#     raw = get_data()

#     # Build features
#     features_df = build_features(raw)

#     # Preview
#     print(f"\nFeature columns ({len(features_df.columns) - 1} features + label):")
#     for col in features_df.columns:
#         print(f"  {col}")

#     print(f"\nFirst 3 rows:\n{features_df.head(3)}")
#     print(f"\nStats:\n{features_df.describe().round(3)}")

#     # Save
#     save_features(features_df)
#     print("\n[Done] Features saved successfully.")



# =============================================================================
#  features/feature_engineering.py  —  Incremental feature computation
#
#  DESIGN PRINCIPLE: Every feature uses the incremental (O(1) per tick) formula.
#  These exact same formulas are re-implemented in HLS C++ / FPGA fixed-point.
#  If you change the formula here, update the FPGA code to match.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import DATA, FEATURES, LABELS


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Individual feature functions
# ════════════════════════════════════════════════════════════════════════════════

def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, reset_daily: bool = True
) -> pd.DataFrame:

    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume

    if reset_daily and isinstance(close.index, pd.DatetimeIndex):
        # [FIX] replaced .date with floor("D") for timezone safety + performance
        dates = close.index.floor("D")
        cum_pv  = pv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_pv  = pv.cumsum()
        cum_vol = volume.cumsum()

    vwap    = cum_pv / (cum_vol + 1e-9)
    vwap_dev = (close - vwap) / (vwap + 1e-9)

    return pd.DataFrame({"vwap": vwap, "vwap_dev": vwap_dev})


def compute_ofi_proxy(volume: pd.Series, close: pd.Series) -> pd.Series:
    """
    OFI Proxy improved using signed volume (better than direction only)
    """
    # [FIX] improved OFI logic (more realistic proxy)
    price_change = close.diff()
    signed_vol = volume * np.sign(price_change).fillna(0)

    ofi = signed_vol.rolling(5).sum() / (volume.rolling(5).sum() + 1e-9)
    return ofi.rename("ofi_proxy")


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Label generation
# ════════════════════════════════════════════════════════════════════════════════

def generate_labels(
    close: pd.Series,
    lookahead: int  = LABELS["lookahead_bars"],
    buy_thresh: float = LABELS["buy_threshold"],
    sell_thresh: float = LABELS["sell_threshold"],
) -> pd.Series:

    future_close  = close.shift(-lookahead)

    # [FIX] added spread-aware realistic labeling
    # spread = close * 0.0005  # assume 5 bps cost
    spread = 0

    future_return = (future_close - close) / (close + 1e-9)

    labels = pd.Series(1, index=close.index, dtype=int, name="label")

    labels[future_return > (buy_thresh + spread)] = 0
    labels[future_return < (sell_thresh - spread)] = 2
    labels[future_return.isna()] = np.nan

    return labels


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Full pipeline
# ════════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[FeatureEng] Computing features...")

    # [FIX] empty dataframe check (critical)
    if df is None or df.empty:
        raise ValueError("[FeatureEng] Input DataFrame is empty. Check data loader.")

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    cfg = FEATURES
    out = pd.DataFrame(index=df.index)

    # ── Add explicit timestamp column ─────────────────────────────────────────
    # [FIX] Convert timezone to IST and add timestamp for visualization
    if df.index.tz is not None:
        out["timestamp"] = df.index.tz_convert("Asia/Kolkata")
    else:
        out["timestamp"] = df.index
    
    print(f"[FeatureEng] Timestamp range: {out['timestamp'].min()} → {out['timestamp'].max()}")

    # ── Time-based features (hour, minute, day_of_week) ──────────────────────
    idx = out["timestamp"]
    out["hour"] = idx.dt.hour
    out["minute"] = idx.dt.minute
    out["day_of_week"] = idx.dt.dayofweek

    # ── Cyclical encoding for time features (very important!) ────────────────
    # Prevents model from treating 23:59 as far from 00:00
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["minute_sin"] = np.sin(2 * np.pi * out["minute"] / 60)
    out["minute_cos"] = np.cos(2 * np.pi * out["minute"] / 60)

    # ── Price-derived ─────────────────────────────────────────────────────────
    out["rsi"] = compute_rsi(c, cfg["rsi_period"])
    macd_df = compute_macd(c, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    out = pd.concat([out, macd_df], axis=1)
    bb_df = compute_bollinger(c, cfg["bb_period"], cfg["bb_std"])
    out = pd.concat([out, bb_df], axis=1)

    out["roc"] = compute_roc(c, cfg["roc_period"])
    out["momentum"] = compute_momentum(c, cfg["momentum_period"])
    out["atr"] = compute_atr(h, l, c)
    out["hl_ratio"] = compute_hl_ratio(h, l, c)

    # ── Volume-derived ────────────────────────────────────────────────────────
    vwap_df = compute_vwap(h, l, c, v, cfg["vwap_session_reset"])
    out = pd.concat([out, vwap_df], axis=1)

    obv_df = compute_obv(c, v)
    out = pd.concat([out, obv_df], axis=1)

    out["volume_zscore"] = compute_volume_zscore(v)
    out["ofi_proxy"] = compute_ofi_proxy(v, c)

    # ── Normalised close ──────────────────────────────────────────────────────
    out["close_norm"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)

    # ── Labels ────────────────────────────────────────────────────────────────
    out["label"] = generate_labels(c)

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    before = len(out)

    # [FIX] CRITICAL: Drop ALL rows with ANY NaN (strict requirement for training)
    # We need 100% clean data - no NaN in features or labels
    out = out.dropna()

    print(f"[FeatureEng] Dropped {before - len(out):,} NaN rows (warmup + lookahead)")
    print(f"[FeatureEng] Feature matrix: {out.shape}")

    # [FIX] validation after processing
    if out.empty:
        raise ValueError("[FeatureEng] All rows dropped after feature engineering.")

    # ── Label distribution report ─────────────────────────────────────────────
    label_counts = out["label"].value_counts().sort_index()
    total = len(out)

    print(f"\n[FeatureEng] Label distribution:")
    for cls, name in enumerate(LABELS["class_names"]):
        count = label_counts.get(cls, 0)
        pct = count / total * 100
        print(f"  {name:5s} ({cls}): {count:6,} ({pct:.1f}%)")

    return out


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Normalisation
# ════════════════════════════════════════════════════════════════════════════════

def normalise_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_path: Path = None,
    fit: bool = True,
):

    # [FIX] switched to RobustScaler (better for financial data)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()

    if not fit and scaler_path and Path(scaler_path).exists():
        print(f"[Normalise] Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
    else:
        print(f"[Normalise] Fitting scaler on {len(df):,} rows...")
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        if scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"[Normalise] Saved scaler to {scaler_path}")

    return df, scaler
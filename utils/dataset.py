# =============================================================================
#  utils/dataset.py  —  PyTorch Dataset classes + train/val/test splitter
#  Handles both sequence data (LSTM/CNN) and tabular data (LightGBM).
# =============================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FEATURES, LABELS, SPLIT, SEQUENCE, TRAINING


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Time-based train/val/test split
# ════════════════════════════════════════════════════════════════════════════════

def time_split(
    df: pd.DataFrame,
    train_frac: float = SPLIT["train_frac"],
    val_frac:   float = SPLIT["val_frac"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by time (no shuffle — prevents data leakage).

    Parameters
    ----------
    df         : Full feature DataFrame (time-ordered).
    train_frac : Fraction for training set.
    val_frac   : Fraction for validation set.
    test_frac  : Remaining rows = test set.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    print(f"\n[Split] Total rows: {n:,}")
    print(f"  Train: {len(train):,} rows ({len(train)/n*100:.1f}%)")
    print(f"  Val:   {len(val):,} rows   ({len(val)/n*100:.1f}%)")
    print(f"  Test:  {len(test):,} rows   ({len(test)/n*100:.1f}%)")
    print(f"  Train: {train.index[0]} -> {train.index[-1]}")
    print(f"  Val:   {val.index[0]} -> {val.index[-1]}")
    print(f"  Test:  {test.index[0]} -> {test.index[-1]}")

    return train, val, test


def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced BUY/HOLD/SELL labels.
    Passed to CrossEntropyLoss(weight=...) during training.
    
    If data is heavily imbalanced (>90% single class), use uniform weights to avoid NaN issues.
    """
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(float)
    
    # Check for heavy imbalance
    max_pct = counts.max() / counts.sum()
    if max_pct > 0.90:
        # Heavily imbalanced - use uniform weights
        weights = np.ones(n_classes)
        print(f"[Weights] Heavy imbalance detected ({max_pct*100:.0f}%). Using uniform weights.")
    else:
        counts = np.where(counts == 0, 1, counts)           # avoid divide-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes        # normalise so mean=1
        
        # Clamp weights to avoid extreme values (prevents NaN)
        weights = np.clip(weights, 0.1, 10.0)
        weights = weights / weights.mean()  # Re-normalize after clamping
    
    print(f"[Weights] Class weights: BUY={weights[0]:.3f}  HOLD={weights[1]:.3f}  SELL={weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Sequence Dataset (LSTM / CNN)
# ════════════════════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    """
    Returns sliding windows of shape (window_size, n_features) as input
    and the label at the last timestep as target.

    Each sample:
        X : Tensor of shape (window_size, n_features)
        y : int label (0=BUY, 1=HOLD, 2=SELL)
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        feature_cols: list   = None,
        window_size:  int    = SEQUENCE["window_size"],
        label_col:    str    = "label",
    ):
        """
        Parameters
        ----------
        df           : DataFrame (already normalised) with features + label.
        feature_cols : List of feature column names to use as input.
        window_size  : Number of consecutive bars in each input window.
        label_col    : Column name for the target label.
        """
        if feature_cols is None:
            feature_cols = FEATURES["feature_cols"]

        # Validate columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        self.window_size = window_size
        self.n_features  = len(feature_cols)

        # Convert to numpy once (fast random access later)
        self.X_raw = df[feature_cols].values.astype(np.float32)
        self.y_raw = df[label_col].values.astype(np.int64)
        self.n_samples = len(self.X_raw) - window_size + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough rows ({len(df)}) for window_size={window_size}. "
                "Reduce window_size or get more data."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X_raw[idx : idx + self.window_size]      # (window, n_features)
        y = self.y_raw[idx + self.window_size - 1]         # label at last bar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Tabular Dataset (LightGBM uses numpy directly, but
#  we also provide a PyTorch version for consistency)
# ════════════════════════════════════════════════════════════════════════════════

class TabularDataset(Dataset):
    """
    Single-bar tabular dataset (no sliding window).
    Used for LightGBM (via get_tabular_arrays) and as a sanity-check Dataset.

    Each sample:
        X : Tensor of shape (n_features,)
        y : int label
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        feature_cols: list = None,
        label_col:    str  = "label",
    ):
        if feature_cols is None:
            feature_cols = FEATURES["feature_cols"]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_tabular_arrays(
    df: pd.DataFrame, feature_cols: list = None, label_col: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) as numpy arrays for direct use with LightGBM / sklearn.
    """
    if feature_cols is None:
        feature_cols = FEATURES["feature_cols"]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DataLoader factory
# ════════════════════════════════════════════════════════════════════════════════

def make_sequence_loaders(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list = None,
    window_size:  int  = SEQUENCE["window_size"],
    batch_size:   int  = 128,
    num_workers:  int  = 0,   # 0 = main process (safe on all platforms)
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for LSTM / CNN training.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    Only train_loader shuffles — val and test preserve time order.
    """
    if feature_cols is None:
        feature_cols = FEATURES["feature_cols"]

    train_ds = SequenceDataset(train_df, feature_cols, window_size)
    val_ds   = SequenceDataset(val_df,   feature_cols, window_size)
    test_ds  = SequenceDataset(test_df,  feature_cols, window_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    print(f"\n[DataLoader] Sequence loaders (window={window_size}, batch={batch_size}):")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")

    return train_loader, val_loader, test_loader


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset utilities with synthetic data...")

    # Create a dummy DataFrame
    n = 2000
    np.random.seed(42)
    dummy = pd.DataFrame(
        np.random.randn(n, len(FEATURES["feature_cols"])),
        columns=FEATURES["feature_cols"]
    )
    dummy["label"] = np.random.randint(0, 3, n)

    train, val, test = time_split(dummy)
    weights = compute_class_weights(train["label"].values)

    tl, vl, tsl = make_sequence_loaders(train, val, test, batch_size=32)
    X_batch, y_batch = next(iter(tl))

    print(f"\nBatch X shape: {X_batch.shape}")   # (32, 30, n_features)
    print(f"Batch y shape: {y_batch.shape}")     # (32,)
    print(f"Label values:  {y_batch[:8].tolist()}")
    print("\nAll dataset tests passed.")

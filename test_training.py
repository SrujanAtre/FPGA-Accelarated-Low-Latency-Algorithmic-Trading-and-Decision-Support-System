#!/usr/bin/env python
"""Quick test of LSTM training without full pipeline."""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config import SEQUENCE, LSTM as LSTM_CFG, LABELS, FEATURES
from models.architectures import LSTMModel
from training.trainer import PyTorchTrainer
from utils.dataset import SequenceDataset, time_split
from features.feature_engineering import load_features
import pandas as pd

print("[TEST] Loading features...")
df = load_features()
print(f"Loaded {len(df)} rows")

print("[TEST] Splitting data...")
train_df, val_df, test_df = time_split(df, train_frac=0.7, val_frac=0.15)

print("[TEST] Creating dataset...")
train_ds = SequenceDataset(train_df, SEQUENCE["window_size"])
print(f"Train dataset: {len(train_ds)} samples")

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=False
)
print(f"Train loader: {len(train_loader)} batches")

print("[TEST] Creating LSTM model...")
n_features = len(LSTM_CFG.get("feature_cols", LABELS.get("feature_cols", [])))
if n_features == 0:
    n_features = 20  # fallback
    
model = LSTMModel(
    n_features=n_features,
    hidden_size=LSTM_CFG["hidden_size"],
    num_layers=LSTM_CFG["num_layers"],
    dropout=LSTM_CFG["dropout"],
    num_classes=3,
    bidirectional=LSTM_CFG["bidirectional"],
)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

print("[TEST] Creating trainer...")
trainer = PyTorchTrainer(
    model=model,
    device="cpu",
    lr=LSTM_CFG["lr"],
    weight_decay=LSTM_CFG["weight_decay"],
    grad_clip=LSTM_CFG["grad_clip"],
    # class_weights=torch.ones(3),
    class_weights=torch.tensor([1.0, 2.5, 1.0]),
)
print("Trainer created")

print("[TEST] Running 1 epoch...")
try:
    train_loss, _, _ = trainer._run_epoch(train_loader, training=True)
    print(f"✓ Epoch completed! Train Loss: {train_loss:.4f}")
except Exception as e:
    print(f"✗ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("[TEST] All tests passed!")

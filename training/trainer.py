# =============================================================================
#  training/trainer.py  —  Training loops for LSTM, CNN, LightGBM
#  Features: early stopping, class weights, learning rate scheduling,
#  per-epoch logging, model checkpointing.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import time, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import LSTM as LSTM_CFG, CNN as CNN_CFG, LGBM as LGBM_CFG, TRAINING, LABELS
from models.architectures import LSTMModel, CNNModel, LightGBMModel



# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Shared utilities
# ════════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when validation metric stops improving.
    Saves the best model checkpoint.
    """

    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-4):
        """
        Parameters
        ----------
        patience : How many epochs to wait after last improvement.
        mode     : 'max' for F1/accuracy (higher=better), 'min' for loss.
        delta    : Minimum improvement to count as improvement.
        """
        self.patience    = patience
        self.mode        = mode
        self.delta       = delta
        self.best_score  = None
        self.counter     = 0
        self.should_stop = False
        self.best_epoch  = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Returns True if we should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            (score > self.best_score + self.delta) if self.mode == "max"
            else (score < self.best_score - self.delta)
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Fast macro-F1 without sklearn (used during training for speed)."""
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PyTorch model trainer (shared by LSTM and CNN)
# ════════════════════════════════════════════════════════════════════════════════

# class PyTorchTrainer:
#     """
#     Generic trainer for any PyTorch classification model.
#     Works for both LSTMModel and CNNModel.
#     """

#     def __init__(
#         self,
#         model:        nn.Module,
#         model_name:   str,
#         lr:           float,
#         weight_decay: float,
#         epochs:       int,
#         patience:     int,
#         grad_clip:    float,
#         model_path:   Path,
#         class_weights: Optional[torch.Tensor] = None,
#         device:       str = TRAINING["device"],
#     ):
#         self.model        = model.to(device)
#         self.model_name   = model_name
#         self.epochs       = epochs
#         self.grad_clip    = grad_clip
#         self.model_path   = Path(model_path)
#         self.device       = device

#         # Loss function with optional class weights
#         self.criterion = nn.CrossEntropyLoss(
#             weight=class_weights.to(device) if class_weights is not None else None
#         )

#         # AdamW with cosine LR schedule
#         self.optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#         self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer, T_max=epochs, eta_min=lr * 0.01
#         )

#         self.early_stop = EarlyStopping(patience=patience, mode="max")
#         self.history    = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

#     def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
#         """Run one epoch (train or eval). Returns (loss, all_preds, all_labels)."""
#         self.model.train() if training else self.model.eval()
#         total_loss = 0.0
#         all_preds, all_labels = [], []

#         context = torch.enable_grad() if training else torch.no_grad()
#         with context:
#             for X_batch, y_batch in loader:
#                 X_batch = X_batch.to(self.device)
#                 y_batch = y_batch.to(self.device)

#                 if training:
#                     self.optimizer.zero_grad()

#                 logits = self.model(X_batch)
#                 loss   = self.criterion(logits, y_batch)

#                 if training:
#                     loss.backward()
#                     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
#                     self.optimizer.step()

#                 total_loss  += loss.item() * len(y_batch)
#                 preds        = logits.argmax(dim=1).cpu().numpy()
#                 all_preds.append(preds)
#                 all_labels.append(y_batch.cpu().numpy())

#         avg_loss   = total_loss / len(loader.dataset)
#         all_preds  = np.concatenate(all_preds)
#         all_labels = np.concatenate(all_labels)
#         return avg_loss, all_preds, all_labels

#     def train(
#         self,
#         train_loader: DataLoader,
#         val_loader:   DataLoader,
#     ) -> dict:
#         """
#         Full training loop with early stopping.
#         Returns history dict with per-epoch metrics.
#         """
#         print(f"\n{'='*60}")
#         print(f"  Training {self.model_name}")
#         print(f"  Device: {self.device} | Epochs: {self.epochs} | "
#               f"Patience: {self.early_stop.patience}")
#         print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
#         print(f"{'='*60}")

#         self.model_path.parent.mkdir(parents=True, exist_ok=True)
#         best_val_f1 = 0.0
#         start_time  = time.time()

#         for epoch in range(1, self.epochs + 1):
#             # Train
#             train_loss, _, _ = self._run_epoch(train_loader, training=True)
#             self.scheduler.step()

#             # Validate
#             val_loss, val_preds, val_labels = self._run_epoch(val_loader, training=False)
#             val_acc = (val_preds == val_labels).mean()
#             val_f1  = macro_f1_score(val_labels, val_preds, n_classes=LABELS["n_classes"])

#             # Log
#             self.history["train_loss"].append(train_loss)
#             self.history["val_loss"].append(val_loss)
#             self.history["val_f1"].append(val_f1)
#             self.history["val_acc"].append(float(val_acc))

#             lr_now = self.optimizer.param_groups[0]["lr"]
#             print(
#                 f"Epoch {epoch:3d}/{self.epochs} | "
#                 f"Train Loss: {train_loss:.4f} | "
#                 f"Val Loss: {val_loss:.4f} | "
#                 f"Val F1: {val_f1:.4f} | "
#                 f"Val Acc: {val_acc:.4f} | "
#                 f"LR: {lr_now:.6f}"
#             )

#             # Save best model
#             if val_f1 > best_val_f1:
#                 best_val_f1 = val_f1
#                 torch.save(self.model.state_dict(), self.model_path)
#                 print(f"  ✓ Saved best model (F1={best_val_f1:.4f})")

#             # Early stopping
#             if self.early_stop(val_f1, epoch):
#                 print(f"\n[EarlyStopping] Stopping at epoch {epoch}. "
#                       f"Best F1={self.early_stop.best_score:.4f} at epoch {self.early_stop.best_epoch}")
#                 break

#         elapsed = time.time() - start_time
#         print(f"\nTraining complete in {elapsed/60:.1f} min | Best Val F1: {best_val_f1:.4f}")

#         # Save history
#         log_dir = TRAINING["log_dir"] / self.model_name
#         log_dir.mkdir(parents=True, exist_ok=True)
#         with open(log_dir / "history.json", "w") as f:
#             json.dump(self.history, f, indent=2)

#         return self.history

#     def load_best(self) -> nn.Module:
#         """Load the best saved checkpoint."""
#         self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
#         self.model.eval()
#         print(f"[{self.model_name}] Loaded best checkpoint from {self.model_path}")
#         return self.model

#     @torch.no_grad()
#     def evaluate(self, loader: DataLoader) -> dict:
#         """Evaluate on a DataLoader. Returns metrics dict."""
#         self.model.eval()
#         _, preds, labels = self._run_epoch(loader, training=False)
#         acc = (preds == labels).mean()
#         f1  = macro_f1_score(labels, preds, n_classes=LABELS["n_classes"])
#         return {
#             "accuracy": float(acc),
#             "macro_f1": float(f1),
#             "predictions": preds,
#             "labels": labels,
#         }


# # ════════════════════════════════════════════════════════════════════════════════
# #  SECTION 3 — LSTM trainer factory
# # ════════════════════════════════════════════════════════════════════════════════

def train_lstm(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    n_features:    int,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[LSTMModel, dict]:
    """
    Train the LSTM model. Returns (trained_model, history).
    """
    model   = LSTMModel(n_features=n_features)
    trainer = PyTorchTrainer(
        model=model,
        model_name="LSTM",
        lr=LSTM_CFG["lr"],
        weight_decay=LSTM_CFG["weight_decay"],
        epochs=LSTM_CFG["epochs"],
        patience=LSTM_CFG["patience"],
        grad_clip=LSTM_CFG["grad_clip"],
        model_path=LSTM_CFG["model_path"],
        class_weights=class_weights,
    )
    history = trainer.train(train_loader, val_loader)
    model   = trainer.load_best()
    return model, history


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — CNN trainer factory
# ════════════════════════════════════════════════════════════════════════════════

def train_cnn(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    n_features:    int,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[CNNModel, dict]:
    """
    Train the CNN model. Returns (trained_model, history).
    """
    model   = CNNModel(n_features=n_features)
    trainer = PyTorchTrainer(
        model=model,
        model_name="CNN",
        lr=CNN_CFG["lr"],
        weight_decay=CNN_CFG["weight_decay"],
        epochs=CNN_CFG["epochs"],
        patience=CNN_CFG["patience"],
        grad_clip=CNN_CFG["grad_clip"],
        model_path=CNN_CFG["model_path"],
        class_weights=class_weights,
    )
    history = trainer.train(train_loader, val_loader)
    model   = trainer.load_best()
    return model, history


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LightGBM trainer
# ════════════════════════════════════════════════════════════════════════════════

def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    feature_names: list = None,
) -> LightGBMModel:
    """
    Train the LightGBM model with early stopping.
    Returns trained LightGBMModel.
    """
    print(f"\n{'='*60}")
    print("  Training LightGBM")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(f"{'='*60}")

    lgbm = LightGBMModel()
    lgbm.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Save
    lgbm.save(LGBM_CFG["model_path"])

    # Feature importance
    imp = lgbm.feature_importance()
    print(f"\n[LightGBM] Top 10 features by importance (gain):")
    print(imp.head(10).to_string())

    return lgbm


# =============================================================================
#  training/trainer.py  —  Training loops for LSTM, CNN, LightGBM
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import time, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import LSTM as LSTM_CFG, CNN as CNN_CFG, LGBM as LGBM_CFG, TRAINING, LABELS
from models.architectures import LSTMModel, CNNModel, LightGBMModel


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Shared utilities
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PyTorch model trainer
# ════════════════════════════════════════════════════════════════════════════════

class PyTorchTrainer:

    def __init__(self, model, model_name, lr, weight_decay, epochs, patience, grad_clip, model_path, class_weights=None, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.model_path = model_path
        self.grad_clip = grad_clip
        self.epochs = epochs

        # [FIX] ensure class_weights dtype + device safety
        if class_weights is not None:
            class_weights = class_weights.to(device).float()

        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion = nn.CrossEntropyLoss(
                        weight=class_weights,
                        label_smoothing=0.05   # [FIX] prevents overconfidence
        )

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        self.early_stop = EarlyStopping(patience=patience, mode="max")

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_acc": [],
        }

    def _run_epoch(self, loader: DataLoader, training: bool):

        # [FIX] guard against empty dataset
        if len(loader.dataset) == 0:
            raise ValueError("[Trainer] DataLoader is empty.")

        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for X_batch, y_batch in loader:

                # [FIX] ensure correct dtype
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).long()

                if training:
                    self.optimizer.zero_grad()

                logits = self.model(X_batch)

                # [FIX] sanity check logits shape
                if logits.shape[0] != y_batch.shape[0]:
                    raise ValueError("[Trainer] Logits batch size mismatch.")

                loss = self.criterion(logits, y_batch)

                if training:
                    loss.backward()

                    # [FIX] prevent exploding gradients
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    self.optimizer.step()

                total_loss += loss.item() * len(y_batch)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y_batch.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)

        # [FIX] safe concatenation
        if len(all_preds) == 0:
            raise ValueError("[Trainer] No predictions generated.")

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        return avg_loss, all_preds, all_labels


    def train(self, train_loader: DataLoader, val_loader: DataLoader):

        print(f"\n{'='*60}")
        print(f"  Training {self.model_name}")
        print(f"  Device: {self.device} | Epochs: {self.epochs} | Patience: {self.early_stop.patience}")

        # [FIX] safe param count
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {param_count:,}")
        print(f"{'='*60}")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        best_val_f1 = 0.0
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):

            train_loss, _, _ = self._run_epoch(train_loader, training=True)

            self.scheduler.step()

            val_loss, val_preds, val_labels = self._run_epoch(val_loader, training=False)

            # [FIX] safe metric calculation
            if len(val_preds) == 0:
                raise ValueError("[Trainer] Validation predictions empty.")

            val_acc = (val_preds == val_labels).mean()
            val_f1 = macro_f1_score(val_labels, val_preds, n_classes=LABELS["n_classes"])

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["val_acc"].append(float(val_acc))

            lr_now = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {lr_now:.6f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  [OK] Saved best model (F1={best_val_f1:.4f})")

            if self.early_stop(val_f1, epoch):
                print(f"\n[EarlyStopping] Stopping at epoch {epoch}. "
                      f"Best F1={self.early_stop.best_score:.4f}")
                break

        elapsed = time.time() - start_time

        print(f"\nTraining complete in {elapsed/60:.1f} min | Best Val F1: {best_val_f1:.4f}")

        # [FIX] safe JSON save
        log_dir = TRAINING["log_dir"] / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


    def load_best(self):
        # [FIX] safe loading (handles GPU/CPU mismatch and missing files)
        if Path(self.model_path).exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"[{self.model_name}] Loaded best checkpoint from {self.model_path}")
        else:
            print(f"[{self.model_name}] No saved checkpoint found. Using current model.")
            self.model.eval()
        return self.model


    @torch.no_grad()
    def evaluate(self, loader: DataLoader):

        self.model.eval()

        _, preds, labels = self._run_epoch(loader, training=False)

        if len(preds) == 0:
            raise ValueError("[Trainer] Evaluation failed: empty predictions.")

        acc = (preds == labels).mean()
        f1 = macro_f1_score(labels, preds, n_classes=LABELS["n_classes"])

        return {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "predictions": preds,
            "labels": labels,
        }
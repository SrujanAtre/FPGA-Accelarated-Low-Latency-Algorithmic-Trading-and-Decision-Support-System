# =============================================================================
#  evaluation/evaluator.py  —  Full evaluation suite
#  ML metrics, per-class report, confusion matrix, ONNX export.
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json, sys
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
import matplotlib
matplotlib.use("Agg")     # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import LABELS, EVAL, LSTM as LSTM_CFG, CNN as CNN_CFG


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Classification metrics
# ════════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> dict:
    """
    Compute and print full classification metrics.

    Returns dict with:
      accuracy, macro_f1, precision (per class), recall (per class),
      f1 (per class), confusion_matrix
    """
    class_names = LABELS["class_names"]
    n_classes = len(class_names)
    all_labels = list(range(n_classes))  # [0, 1, 2]

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=all_labels)
    prec       = precision_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
    rec        = recall_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
    f1_per     = f1_score(y_true, y_pred, average=None, zero_division=0, labels=all_labels)
    cm         = confusion_matrix(y_true, y_pred, labels=all_labels)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, labels=all_labels)

    print(f"\n{'='*55}")
    print(f"  Evaluation Results — {model_name}")
    print(f"{'='*55}")
    print(f"  Overall Accuracy : {acc:.4f}")
    print(f"  Macro F1 Score   : {macro_f1:.4f}")
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':6s}  {'Precision':9s}  {'Recall':8s}  {'F1':6s}")
    print(f"  {'-'*40}")
    for i, name in enumerate(class_names):
        print(f"  {name:6s}  {prec[i]:.4f}      {rec[i]:.4f}    {f1_per[i]:.4f}")
    print(f"\n  Full Classification Report:")
    print(report_str)

    metrics = {
        "model":       model_name,
        "accuracy":    float(acc),
        "macro_f1":    float(macro_f1),
        "precision":   {class_names[i]: float(prec[i])   for i in range(len(class_names))},
        "recall":      {class_names[i]: float(rec[i])    for i in range(len(class_names))},
        "f1_per_class":{class_names[i]: float(f1_per[i]) for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
    }

    if save:
        _save_metrics(metrics, model_name)

    return metrics


def _save_metrics(metrics: dict, model_name: str) -> None:
    """Save metrics to JSON."""
    out_dir = EVAL["report_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name.lower()}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Saved] {path}")


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Confusion matrix plot
# ════════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    class_names = LABELS["class_names"]
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, cbar_kws={"label": "% of true class"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)

    plt.tight_layout()
    if save:
        out_dir = EVAL["plot_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model_name.lower()}_confusion.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {path}")
    plt.close()


def plot_training_history(history: dict, model_name: str = "Model", save: bool = True) -> None:
    """Plot training loss and validation F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="coral")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_f1"],  label="Val Macro-F1", color="green")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy",  color="purple")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title(f"{model_name} — Validation Metrics"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save:
        out_dir = EVAL["plot_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model_name.lower()}_training.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ONNX export
# ════════════════════════════════════════════════════════════════════════════════

def export_to_onnx(
    model:       nn.Module,
    model_name:  str,
    onnx_path:   Path,
    n_features:  int,
    window_size: int = 30,
    opset:       int = 18,
) -> None:
    """
    Export a trained PyTorch model to ONNX format.
    This ONNX file is consumed by:
      - hls4ml (FPGA deployment)
      - ONNX Runtime (optimised CPU deployment)

    Parameters
    ----------
    model      : Trained PyTorch model (LSTMModel or CNNModel).
    model_name : Name for logging ("LSTM" or "CNN").
    onnx_path  : Output path for the .onnx file.
    n_features : Number of input features.
    window_size: Input sequence length (for LSTM/CNN).
    opset      : ONNX opset version. Use 14 for hls4ml compatibility.
    """
    model.eval()
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # Dummy input matching the model's expected input shape
    dummy_input = torch.randn(1, window_size, n_features)

    print(f"\n[ONNX] Exporting {model_name} to {onnx_path}...")
    print(f"  Input shape: {list(dummy_input.shape)}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,    # fold constants for smaller graph
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model verified successfully")
        print(f"  ✓ Saved to {onnx_path}")
        # Uncomment below for detailed graph info
        # print(f"  Nodes in graph: {len(onnx_model.graph.node)}")
        # print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
        # print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

    except ImportError:
        print("  [Warning] onnx package not installed. Skipping verification.")
        print(f"  Install with: pip install onnx")
        print(f"  ✓ File saved to {onnx_path} (unverified)")


def verify_onnx_inference(
    onnx_path:  Path,
    n_features: int,
    window_size: int = 30,
) -> None:
    """
    Run a test inference with ONNX Runtime to verify the exported model works.
    """
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, window_size, n_features).astype(np.float32)
        out   = sess.run(None, {"input": dummy})
        print(f"  ✓ ONNX Runtime inference test passed. Output shape: {out[0].shape}")
    except ImportError:
        print("  [Warning] onnxruntime not installed. Skipping runtime test.")
        print("  Install with: pip install onnxruntime")


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Feature importance plot (for LightGBM + ablation)
# ════════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    importance: pd.Series,
    model_name: str = "LightGBM",
    top_n: int = 15,
    save: bool = True,
) -> None:
    """Plot top-N feature importances as a horizontal bar chart."""
    top = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(range(len(top)), top.values, color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=12, pad=10)
    ax.grid(axis="x", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        out_dir = EVAL["plot_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model_name.lower()}_feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Ensemble comparison summary
# ════════════════════════════════════════════════════════════════════════════════

def print_model_comparison(results: dict) -> None:
    """
    Print a summary table comparing all models.
    results = {model_name: metrics_dict}
    """
    print(f"\n{'='*65}")
    print("  Model Comparison Summary")
    print(f"{'='*65}")
    print(f"  {'Model':15s}  {'Accuracy':10s}  {'Macro F1':10s}  {'BUY F1':9s}  {'SELL F1':8s}")
    print(f"  {'-'*60}")

    for name, m in results.items():
        acc     = m.get("accuracy", 0)
        f1      = m.get("macro_f1", 0)
        buy_f1  = m.get("f1_per_class", {}).get("BUY",  0)
        sell_f1 = m.get("f1_per_class", {}).get("SELL", 0)
        print(f"  {name:15s}  {acc:.4f}      {f1:.4f}      {buy_f1:.4f}    {sell_f1:.4f}")

    print(f"{'='*65}")


def create_comparison_table(results: dict, save: bool = True) -> pd.DataFrame:
    """
    Create a comparison DataFrame from model results and save as CSV + plot.

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and metrics dicts as values.
        Format: {model_name: metrics_dict}
    save : bool
        Whether to save CSV and plot files.

    Returns
    -------
    pd.DataFrame
        Comparison table with model metrics.
    """
    comparison = []

    for model_name, metrics in results.items():
        f1_per_class = metrics.get("f1_per_class", {})
        comparison.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0),
            "Macro_F1": metrics.get("macro_f1", 0),
            "BUY_F1": f1_per_class.get("BUY", 0),
            "HOLD_F1": f1_per_class.get("HOLD", 0),
            "SELL_F1": f1_per_class.get("SELL", 0),
        })

    comparison_df = pd.DataFrame(comparison)

    print(f"\n{'='*65}")
    print("  Model Comparison Table")
    print(f"{'='*65}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*65}\n")

    if save:
        # Save CSV
        out_dir = EVAL["report_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"  [Saved] {csv_path}")

        # Create and save comparison plot
        plot_dir = EVAL["plot_dir"]
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        comparison_df.set_index("Model")[["Accuracy", "Macro_F1"]].plot(
            kind="bar", ax=ax, color=["steelblue", "coral"], edgecolor="white"
        )
        ax.set_title("Model Comparison — Accuracy vs Macro F1", fontsize=13, pad=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xlabel("Model", fontsize=11)
        ax.legend(title="Metric", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = plot_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  [Saved] {plot_path}")
        plt.close()

    return comparison_df

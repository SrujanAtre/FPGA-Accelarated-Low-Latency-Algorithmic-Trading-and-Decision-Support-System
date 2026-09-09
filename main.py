# =============================================================================
#  main.py  —  Master pipeline: Data → Features → Train → Evaluate → Export
#
#  Run this one script to execute the entire software pipeline.
#  Usage:
#      python main.py                    # full pipeline
#      python main.py --skip-download    # use cached data
#      python main.py --model lstm       # train only LSTM
#      python main.py --eval-only        # evaluate saved models
# =============================================================================

import argparse
import numpy as np
import torch
import sys, json, time
from pathlib import Path
import pandas as pd

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config import (
    FEATURES, LABELS, SEQUENCE, TRAINING,
    LSTM as LSTM_CFG, CNN as CNN_CFG, LGBM as LGBM_CFG
)
from data.data_loader import get_data
from features.feature_engineering import build_features, normalise_features, save_features, load_features
from utils.dataset import (
    time_split, compute_class_weights,
    make_sequence_loaders, get_tabular_arrays
)
from models.architectures import LSTMModel, CNNModel, LightGBMModel, EnsemblePredictor
from training.trainer import train_lstm, train_cnn, train_lgbm, PyTorchTrainer
from evaluation.evaluator import (
    compute_metrics, plot_confusion_matrix,
    plot_training_history, export_to_onnx, verify_onnx_inference,
    plot_feature_importance, print_model_comparison
)


def set_seed(seed: int = 42):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="HFT Decision System — Software Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Use cached raw data")
    parser.add_argument("--skip-features", action="store_true", help="Use cached feature CSV")
    parser.add_argument("--model",         type=str, default="all",
                        choices=["all", "lstm", "cnn", "lgbm"],
                        help="Which model(s) to train")
    parser.add_argument("--eval-only",     action="store_true", help="Skip training, evaluate saved models")
    parser.add_argument("--no-export",     action="store_true", help="Skip ONNX export")
    parser.add_argument("--backtest",      action="store_true", help="Run walk-forward backtesting")
    parser.add_argument("--inference",     action="store_true", help="Test real-time inference engine")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(TRAINING["seed"])

    print("\n" + "=" * 65)
    print("  FPGA-Accelerated HFT Decision System — Software Pipeline")
    print("  MIT World Peace University | Capstone 2025-26")
    print("=" * 65)
    t_start = time.time()

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — Data collection
    # ════════════════════════════════════════════════════════════════
    print("\n[STEP 1] Data Collection")
    raw_df = get_data()
    
    if len(raw_df) == 0:
        print("  ERROR: No valid data downloaded. Check ticker, interval, or market hours filter.")
        return
    
    print(f"  Raw data: {len(raw_df):,} bars | {raw_df.index[0]} -> {raw_df.index[-1]}")

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — Feature engineering
    # ════════════════════════════════════════════════════════════════
    print("\n[STEP 2] Feature Engineering")
    if args.skip_features and Path(str(FEATURES)).exists():
        features_df = load_features()
    else:
        features_df = build_features(raw_df)
        save_features(features_df)

    feature_cols = FEATURES["feature_cols"]
    # Validate all expected features are present
    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        print(f"  [Warning] Missing expected features: {missing}")
        print(f"  Using available features only.")
        feature_cols = [c for c in feature_cols if c in features_df.columns]

    n_features = len(feature_cols)
    print(f"  Features used: {n_features}")
    print(f"  Feature list: {feature_cols}")

    # ════════════════════════════════════════════════════════════════
    # STEP 3 — Train/Val/Test split + normalisation
    # ════════════════════════════════════════════════════════════════
    print("\n[STEP 3] Data Splitting & Normalisation")
    train_df, val_df, test_df = time_split(features_df)

    scaler_path = ROOT / "models" / "saved" / "scaler.pkl"
    train_df, scaler = normalise_features(train_df, feature_cols, scaler_path=scaler_path, fit=True)
    val_df,   _      = normalise_features(val_df,   feature_cols, scaler_path=scaler_path, fit=False)
    test_df,  _      = normalise_features(test_df,  feature_cols, scaler_path=scaler_path, fit=False)

    # Class weights (computed from training labels only)
    class_weights = compute_class_weights(train_df["label"].values)

    # ════════════════════════════════════════════════════════════════
    # STEP 4 — Model training
    # ════════════════════════════════════════════════════════════════
    print("\n[STEP 4] Model Training")

    all_results = {}
    train_model_lstm = (args.model in ["all", "lstm"])
    train_model_cnn  = (args.model in ["all", "cnn"])
    train_model_lgbm = (args.model in ["all", "lgbm"])

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if train_model_lstm and not args.eval_only:
        train_loader, val_loader, test_loader = make_sequence_loaders(
            train_df, val_df, test_df,
            feature_cols=feature_cols,
            window_size=SEQUENCE["window_size"],
            batch_size=LSTM_CFG["batch_size"],
        )
        lstm_model, lstm_history = train_lstm(
            train_loader, val_loader,
            n_features=n_features,
            class_weights=class_weights,
        )
        plot_training_history(lstm_history, model_name="LSTM")

    elif train_model_lstm and args.eval_only:
        lstm_model = LSTMModel(n_features=n_features)
        lstm_model.load_state_dict(torch.load(LSTM_CFG["model_path"], map_location="cpu"))
        lstm_model.eval()
        train_loader, val_loader, test_loader = make_sequence_loaders(
            train_df, val_df, test_df,
            feature_cols=feature_cols,
            window_size=SEQUENCE["window_size"],
            batch_size=LSTM_CFG["batch_size"],
        )
        print("[LSTM] Loaded saved model for evaluation.")

    # ── CNN ───────────────────────────────────────────────────────────────────
    if train_model_cnn and not args.eval_only:
        if not train_model_lstm:  # loaders not yet created
            train_loader, val_loader, test_loader = make_sequence_loaders(
                train_df, val_df, test_df,
                feature_cols=feature_cols,
                window_size=SEQUENCE["window_size"],
                batch_size=CNN_CFG["batch_size"],
            )
        cnn_model, cnn_history = train_cnn(
            train_loader, val_loader,
            n_features=n_features,
            class_weights=class_weights,
        )
        plot_training_history(cnn_history, model_name="CNN")

    elif train_model_cnn and args.eval_only:
        cnn_model = CNNModel(n_features=n_features)
        cnn_model.load_state_dict(torch.load(CNN_CFG["model_path"], map_location="cpu"))
        cnn_model.eval()
        if not train_model_lstm:
            train_loader, val_loader, test_loader = make_sequence_loaders(
                train_df, val_df, test_df,
                feature_cols=feature_cols,
                window_size=SEQUENCE["window_size"],
                batch_size=CNN_CFG["batch_size"],
            )
        print("[CNN] Loaded saved model for evaluation.")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if train_model_lgbm:
        X_train, y_train = get_tabular_arrays(train_df, feature_cols)
        X_val,   y_val   = get_tabular_arrays(val_df,   feature_cols)
        X_test,  y_test  = get_tabular_arrays(test_df,  feature_cols)

        if not args.eval_only:
            lgbm_model = train_lgbm(
                X_train, y_train, X_val, y_val,
                feature_names=feature_cols,
            )
        else:
            lgbm_model = LightGBMModel()
            lgbm_model.load(LGBM_CFG["model_path"])

    # ════════════════════════════════════════════════════════════════
    # STEP 5 — Evaluation on test set
    # ════════════════════════════════════════════════════════════════
    print("\n[STEP 5] Evaluation on Test Set")

    if train_model_lstm:
        lstm_trainer = PyTorchTrainer(
            model=lstm_model, model_name="LSTM",
            lr=LSTM_CFG["lr"], weight_decay=LSTM_CFG["weight_decay"],
            epochs=1, patience=1, grad_clip=1.0,
            model_path=LSTM_CFG["model_path"],
        )
        lstm_results = lstm_trainer.evaluate(test_loader)
        lstm_metrics = compute_metrics(
            lstm_results["labels"], lstm_results["predictions"], model_name="LSTM"
        )
        plot_confusion_matrix(lstm_results["labels"], lstm_results["predictions"], "LSTM")
        all_results["LSTM"] = lstm_metrics

    if train_model_cnn:
        cnn_trainer = PyTorchTrainer(
            model=cnn_model, model_name="CNN",
            lr=CNN_CFG["lr"], weight_decay=CNN_CFG["weight_decay"],
            epochs=1, patience=1, grad_clip=1.0,
            model_path=CNN_CFG["model_path"],
        )
        cnn_results = cnn_trainer.evaluate(test_loader)
        cnn_metrics = compute_metrics(
            cnn_results["labels"], cnn_results["predictions"], model_name="CNN"
        )
        plot_confusion_matrix(cnn_results["labels"], cnn_results["predictions"], "CNN")
        all_results["CNN"] = cnn_metrics

    if train_model_lgbm:
        lgbm_preds = lgbm_model.predict(X_test)
        lgbm_metrics = compute_metrics(y_test, lgbm_preds, model_name="LightGBM")
        plot_confusion_matrix(y_test, lgbm_preds, "LightGBM")
        all_results["LightGBM"] = lgbm_metrics

        # Feature importance plot
        imp = lgbm_model.feature_importance()
        plot_feature_importance(imp, model_name="LightGBM")

    # ── Ensemble (if all three models trained) ────────────────────────────────
    if train_model_lstm and train_model_cnn and train_model_lgbm:
        print("\n[Ensemble] Evaluating soft-voting ensemble on test set...")
        ensemble = EnsemblePredictor(lstm_model, cnn_model, lgbm_model, device=TRAINING["device"])

        # Build sequence batches for test set
        from torch.utils.data import DataLoader
        from utils.dataset import SequenceDataset
        test_seq_ds = SequenceDataset(test_df, feature_cols, SEQUENCE["window_size"])
        test_seq_loader = DataLoader(test_seq_ds, batch_size=256, shuffle=False)

        all_seq_X, all_true = [], []
        for X_b, y_b in test_seq_loader:
            all_seq_X.append(X_b)
            all_true.append(y_b.numpy())
        all_seq_X = torch.cat(all_seq_X)
        all_true  = np.concatenate(all_true)

        # Align tabular test data with sequence test data
        n_seq = len(test_seq_ds)
        X_test_aligned = X_test[-n_seq:]  # last n_seq rows match sequence dataset

        ens_preds = ensemble.predict(all_seq_X, X_test_aligned, confidence_threshold=0.65)
        ens_metrics = compute_metrics(all_true, ens_preds, model_name="Ensemble")
        plot_confusion_matrix(all_true, ens_preds, "Ensemble")
        all_results["Ensemble"] = ens_metrics

    # ── Model comparison table ────────────────────────────────────────────────
    if all_results:
        print_model_comparison(all_results)

    # ════════════════════════════════════════════════════════════════
    # STEP 6 — ONNX export
    # ════════════════════════════════════════════════════════════════
    if not args.no_export:
        print("\n[STEP 6] ONNX Export (for FPGA hls4ml + CPU ONNX Runtime)")

        if train_model_lstm:
            export_to_onnx(
                lstm_model, "LSTM", LSTM_CFG["onnx_path"],
                n_features=n_features, window_size=SEQUENCE["window_size"]
            )
            verify_onnx_inference(LSTM_CFG["onnx_path"], n_features, SEQUENCE["window_size"])

        if train_model_cnn:
            export_to_onnx(
                cnn_model, "CNN", CNN_CFG["onnx_path"],
                n_features=n_features, window_size=SEQUENCE["window_size"]
            )
            verify_onnx_inference(CNN_CFG["onnx_path"], n_features, SEQUENCE["window_size"])

        # [COMMENTED OUT] Verbose ONNX deployment notes
        # print("\n[ONNX Export Notes]")
        # print("  Next steps for FPGA deployment:")
        # print("  1. pip install hls4ml brevitas")
        # print("  2. hls4ml.converters.convert_from_onnx_model(onnx_model, hls_config)")
        # print("  3. Set hls_config precision: ap_fixed<16,6> for all layers")
        # print("  4. hls_model.build(csim=True, synth=True) in Vitis HLS")
        # print("\n  Next steps for CPU ONNX Runtime deployment:")
        # print("  1. pip install onnxruntime")
        # print("  2. sess = ort.InferenceSession(onnx_path)")
        # print("  3. output = sess.run(None, {'input': X_np})")

    # ════════════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Models saved in: {ROOT / 'models' / 'saved'}")
    print(f"  Plots saved in:  {ROOT / 'evaluation' / 'plots'}")
    print(f"  Reports saved in:{ROOT / 'evaluation' / 'reports'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

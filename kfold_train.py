"""
kfold_train.py — Stratified K-Fold cross-validation at the FILE level.

Why K-fold for sleep staging?
──────────────────────────────
With only 153 recordings, a single 80/20 split has HIGH variance: depending
on which 31 files land in the validation fold, macro-F1 can swing ±0.03.
5-fold CV trains 5 models each on ~122 files and validates on the other ~31,
cycling through all files.  Benefits:

  1. Every file contributes to training (no permanently held-out internal data).
  2. Average metrics across 5 folds give a much tighter performance estimate.
  3. Ensemble of 5 models (avg. softmax) consistently outperforms any single
     model for hard classes like N1 (variance reduction across folds).

Stratification: majority sleep stage per file → both train and val in each
fold have a balanced mix of recordings dominated by Wake, N2, etc.

Usage:
    python kfold_train.py          # trains 5 models, ~17 min × 5 = 85 min total
    python kfold_train.py --folds 3   # faster 3-fold run

Checkpoints saved to:
    checkpoints/fold_0/best_model.pt  + scaler.pkl
    checkpoints/fold_1/best_model.pt  + scaler.pkl
    ...

Run ensemble_external_validation.py afterwards for the holdout evaluation.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BATCH_SIZE, CHECKPOINT_DIR, DROPOUT, HIDDEN_SIZE, INPUT_SIZE,
    INTERNAL_DIR, K_FOLDS, NUM_CLASSES, NUM_LAYERS, NUM_WORKERS,
    PIN_MEMORY, RESULTS_DIR, SEED,
)
from model_development.data_utils import (
    compute_class_weights, discover_files, fit_scaler, load_recording,
    save_scaler,
)
from model_development.dataset import SleepWindowDataset
from model_development.evaluate import compute_metrics, get_predictions, plot_confusion_matrix, save_metrics
from model_development.model import BiLSTMSleepStager
from model_development.train import train


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_majority_labels(files):
    """Return majority sleep stage per file for stratification."""
    labels = []
    for f in files:
        try:
            df = load_recording(f)
            labels.append(int(df["label"].mode().iloc[0]))
        except Exception:
            labels.append(0)
    return labels


def run_kfold(n_splits: int = K_FOLDS) -> None:
    set_seed(SEED)

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    all_files = discover_files(INTERNAL_DIR)
    majority_labels = get_majority_labels(all_files)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(all_files, majority_labels))

    fold_results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold + 1, n_splits)
        logger.info("=" * 60)

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        logger.info("Train: %d files | Val: %d files",
                    len(train_files), len(val_files))

        fold_dir = CHECKPOINT_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Fit scaler and weights on THIS fold's training data only
        scaler = fit_scaler(train_files)
        weights = compute_class_weights(train_files)
        save_scaler(scaler, fold_dir / "scaler.pkl")

        train_ds = SleepWindowDataset(train_files, scaler)
        val_ds = SleepWindowDataset(val_files,   scaler)
        logger.info("Train samples: %s | Val samples: %s",
                    f"{len(train_ds):,}", f"{len(val_ds):,}")

        # sqrt-inverse sampler for stable gradients (see main.py for rationale)
        sqrt_w = np.sqrt(weights)
        sample_weights = torch.tensor(
            sqrt_w[train_ds._labels], dtype=torch.float32)
        sampler = WeightedRandomSampler(
            sample_weights, len(train_ds), replacement=True)

        use_pin = PIN_MEMORY and device.type in ["cuda", "mps"]
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=use_pin)
        val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=use_pin)

        model = BiLSTMSleepStager(
            input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT,
        )
        n_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
        logger.info("Model params: %s", f"{n_params:,}")

        # train() saves best_model.pt to fold_dir automatically
        model = train(model, train_loader, val_loader, weights, device,
                      checkpoint_dir=fold_dir)

        y_true, y_pred = get_predictions(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, split_name=f"fold_{fold}")
        plot_confusion_matrix(y_true, y_pred, split_name=f"fold_{fold}",
                              save_path=RESULTS_DIR / f"confusion_fold_{fold}.png")
        save_metrics(metrics, split_name=f"fold_{fold}")
        fold_results.append(metrics)

    # ── Summary across folds ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("K-FOLD SUMMARY  (K=%d)", n_splits)
    logger.info("=" * 60)

    from config import CLASS_NAMES
    macro_f1s = [m["macro_f1"] for m in fold_results]
    kappas = [m["kappa"] for m in fold_results]

    logger.info("Macro F1: %.4f ± %.4f", np.mean(macro_f1s), np.std(macro_f1s))
    logger.info("Kappa   : %.4f ± %.4f", np.mean(kappas),    np.std(kappas))
    logger.info("Per-class F1 (mean ± std):")
    for cls in CLASS_NAMES:
        scores = [m["per_cls_f1"][cls] for m in fold_results]
        logger.info("  %-6s  %.4f ± %.4f", cls,
                    np.mean(scores), np.std(scores))

    # Persist summary
    summary = {
        "n_folds":  n_splits,
        "macro_f1": {"mean": float(np.mean(macro_f1s)), "std": float(np.std(macro_f1s))},
        "kappa":    {"mean": float(np.mean(kappas)),    "std": float(np.std(kappas))},
        "per_cls_f1": {
            cls: {"mean": float(np.mean([m["per_cls_f1"][cls] for m in fold_results])),
                  "std":  float(np.std([m["per_cls_f1"][cls] for m in fold_results]))}
            for cls in CLASS_NAMES
        },
        "fold_results": [
            {"fold": i, "macro_f1": m["macro_f1"], "kappa": m["kappa"],
             "per_cls_f1": m["per_cls_f1"]}
            for i, m in enumerate(fold_results)
        ],
    }
    with open(RESULTS_DIR / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved → %s", RESULTS_DIR / "kfold_summary.json")
    logger.info(
        "Run  python ensemble_external_validation.py  for holdout evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=K_FOLDS,
                        help=f"Number of CV folds (default: {K_FOLDS})")
    args = parser.parse_args()
    run_kfold(n_splits=args.folds)

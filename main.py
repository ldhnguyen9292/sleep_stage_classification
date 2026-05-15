"""
main.py — Training entry point.

Run this script to:
  1. Discover and split internal CSV files (file-level, no epoch leakage).
  2. Fit the feature scaler on training data only.
  3. Build sliding-window datasets and DataLoaders.
  4. Initialise the BiLSTM model.
  5. Train with Focal Loss + WeightedRandomSampler, early stopping, LR decay.
  6. Evaluate on the internal validation split.
  7. Save the best checkpoint and scaler for external_validation.py.

Usage:
  python main.py

The 44 external CSV files in processed_data/external/ are NEVER touched
by this script.  Run external_validation.py once — and only once — after
development is fully complete.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    INTERNAL_DIR,
    NUM_CLASSES,
    NUM_LAYERS,
    NUM_WORKERS,
    PIN_MEMORY,
    RESULTS_DIR,
    SEED,
)
from model_development.data_utils import (
    compute_class_weights,
    discover_files,
    fit_scaler,
    save_scaler,
    split_files,
)
from model_development.dataset import SleepWindowDataset
from model_development.evaluate import compute_metrics, plot_confusion_matrix, save_metrics, get_predictions
from model_development.model import BiLSTMSleepStager
from model_development.train import train


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fix all random number generators for full reproducibility.

    torch.backends.cudnn.deterministic=True forces cuDNN to use only
    deterministic algorithms; combined with a fixed seed this makes GPU
    training bit-for-bit reproducible across runs (at a small speed cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # disable auto-tuner (non-deterministic)
    torch.backends.cudnn.benchmark = False


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(SEED)
    logger.info("Global seed set to %d", SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("GPU: %s  (VRAM: %.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.info(
            "No GPU detected — training on CPU (slower but fully functional)")

    # ── Discover & split files ────────────────────────────────────────────────
    all_files = discover_files(INTERNAL_DIR)
    train_files, val_files = split_files(all_files)

    # Persist the split so results are reproducible and auditable.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_record = {
        "train": [f.name for f in train_files],
        "val":   [f.name for f in val_files],
    }
    with open(RESULTS_DIR / "data_split.json", "w") as f:
        json.dump(split_record, f, indent=2)
    logger.info("Data split saved → %s", RESULTS_DIR / "data_split.json")

    # ── Scaler (fit on training epochs ONLY) ──────────────────────────────────
    scaler = fit_scaler(train_files)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, CHECKPOINT_DIR / "scaler.pkl")

    # ── Class weights (computed from training labels ONLY) ────────────────────
    class_weights = compute_class_weights(train_files)

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Building sliding-window datasets …")
    train_dataset = SleepWindowDataset(train_files, scaler)
    val_dataset = SleepWindowDataset(val_files,   scaler)

    logger.info(
        "Train samples: %s  |  Val samples: %s",
        f"{len(train_dataset):,}", f"{len(val_dataset):,}",
    )
    logger.info("Train label distribution: %s", train_dataset.label_counts)
    logger.info("Val   label distribution: %s", val_dataset.label_counts)

    # ── WeightedRandomSampler (sqrt-inverse frequency) ───────────────────────
    # sqrt(inverse_freq) is MILDER than pure inverse_freq:
    #   pure inverse:   N1 oversampled 3.3× vs N2  →  gradient noise from
    #                   over-representing hard/ambiguous N1 batches
    #   sqrt-inverse:   N1 oversampled 1.8× vs N2  →  still more than natural,
    #                   but gradient variance is lower and training is stable
    # The Focal Loss already provides an additional implicit re-weighting
    # based on per-sample difficulty, so combined emphasis is still adequate.
    sqrt_weights = np.sqrt(class_weights)
    sample_weights = torch.tensor(
        sqrt_weights[train_dataset._labels],
        dtype=torch.float32,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    use_pin = PIN_MEMORY and device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,          # replaces shuffle=True
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = BiLSTMSleepStager(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("BiLSTM parameters: %s", f"{n_params:,}")

    # ── Training ───────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    model = train(model, train_loader, val_loader, class_weights, device)

    # ── Internal validation evaluation ────────────────────────────────────────
    logger.info("Evaluating on internal validation split …")
    y_true, y_pred = get_predictions(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred, split_name="internal_validation")
    plot_confusion_matrix(y_true, y_pred, split_name="internal_validation")
    save_metrics(metrics, split_name="internal_validation")

    logger.info(
        "Done.  Internal val macro-F1=%.4f  kappa=%.4f",
        metrics["macro_f1"], metrics["kappa"],
    )
    logger.info(
        "Run  python external_validation.py  for holdout evaluation "
        "(do this only once, after all model development is complete)."
    )


if __name__ == "__main__":
    main()

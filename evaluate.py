"""
Evaluation: collecting predictions, computing metrics, and visualising
the confusion matrix.

Metrics rationale
─────────────────
● Macro F1-score
    Arithmetic mean of per-class F1 scores.  Every stage is weighted
    equally regardless of how many epochs it contains, so a model that
    correctly identifies rare N1 and N3 is rewarded as much as one that
    correctly identifies abundant N2.  This is the primary metric for
    comparative sleep staging studies (AASM-aligned).

● Cohen's Kappa
    Measures agreement between the model and the reference labels,
    corrected for chance.  Kappa > 0.6 is generally considered "substantial"
    agreement; > 0.8 is "almost perfect".  Kappa is more informative than
    raw accuracy on imbalanced datasets because it subtracts the expected
    agreement of a random classifier.

● Per-class F1
    Reveals which stages are most difficult (typically N1, which shares
    spectral characteristics with Wake and early REM).

● Normalised confusion matrix
    Rows: true stage.  Columns: predicted stage.
    Cell (i, j) = fraction of true-stage-i epochs predicted as stage j.
    Normalising by row makes error patterns legible across imbalanced classes.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from config import CLASS_NAMES, RESULTS_DIR

logger = logging.getLogger(__name__)


# ── Prediction collection ──────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over *loader* and return (y_true, y_pred) as integer arrays.

    The model is set to eval() mode to disable dropout.  torch.no_grad()
    prevents autograd from building a computation graph, halving memory use
    and increasing throughput during inference.
    """
    model.eval()
    all_preds:  list = []
    all_labels: list = []

    for x, y in loader:
        x      = x.to(device)
        logits = model(x)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

    return np.array(all_labels, dtype=np.int64), np.array(all_preds, dtype=np.int64)


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    split_name: str = "validation",
) -> dict:
    """
    Compute and log all evaluation metrics.

    Args:
        y_true:     Ground-truth integer labels.
        y_pred:     Model-predicted integer labels.
        split_name: Human-readable name printed in the log (e.g. 'external').

    Returns:
        Dict with keys: macro_f1, kappa, per_cls_f1, report.
    """
    macro_f1  = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    kappa     = cohen_kappa_score(y_true, y_pred)
    per_cls   = f1_score(y_true, y_pred, average=None,       zero_division=0,
                         labels=list(range(len(CLASS_NAMES))))
    report    = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    bar = "=" * 62
    logger.info("\n%s\n  %s RESULTS\n%s", bar, split_name.upper(), bar)
    logger.info("  Macro F1-score : %.4f", macro_f1)
    logger.info("  Cohen's Kappa  : %.4f", kappa)
    logger.info("  Per-class F1   :")
    for name, score in zip(CLASS_NAMES, per_cls):
        logger.info("    %-6s  %.4f", name, score)
    logger.info("\nClassification report:\n%s", report)

    return {
        "macro_f1":   macro_f1,
        "kappa":      kappa,
        "per_cls_f1": dict(zip(CLASS_NAMES, per_cls.tolist())),
        "report":     report,
    }


# ── Confusion matrix visualisation ────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    split_name: str           = "validation",
    save_path:  Optional[Path] = None,
) -> None:
    """
    Save a row-normalised confusion matrix heatmap as a PNG.

    Normalisation by true-class row reveals proportional error patterns
    and is legible even when class frequencies differ by an order of
    magnitude (which is typical in sleep staging datasets).
    """
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted stage")
    axes[0].set_ylabel("True stage")
    axes[0].set_title(f"Counts — {split_name}")

    # Right: row-normalised proportions
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xlabel("Predicted stage")
    axes[1].set_ylabel("True stage")
    axes[1].set_title(f"Row-normalised — {split_name}")

    plt.suptitle(f"Sleep stage confusion matrix — {split_name}", fontsize=13)
    plt.tight_layout()

    if save_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = RESULTS_DIR / f"confusion_matrix_{split_name}.png"

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


# ── Metrics persistence ────────────────────────────────────────────────────────

def save_metrics(metrics: dict, split_name: str) -> None:
    """Write a plain-text metrics summary to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{split_name}_metrics.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== {split_name.upper()} METRICS ===\n\n")
        f.write(metrics["report"])
        f.write(f"\nMacro F1 : {metrics['macro_f1']:.4f}\n")
        f.write(f"Kappa    : {metrics['kappa']:.4f}\n")
        f.write("\nPer-class F1:\n")
        for cls, score in metrics["per_cls_f1"].items():
            f.write(f"  {cls:<6}  {score:.4f}\n")
    logger.info("Metrics saved → %s", out_path)

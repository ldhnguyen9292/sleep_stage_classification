"""
Training loop — v3 changes:
  ● FocalLoss with label smoothing (ε=0.1) — handles noisy N1 boundaries
  ● CosineAnnealingWarmRestarts — prevents plateau-induced early convergence
  ● EMA-smoothed val F1 for early stopping — suppresses 0.70↔0.72 oscillation
  ● Weight decay in AdamW — replaces Adam + separate weight_decay

Why CosineAnnealingWarmRestarts?
  ReduceLROnPlateau stepped the LR from 1e-3 to ~6e-5 by epoch ~30, then the
  model was stuck.  CosineAnnealingWarmRestarts restores the LR after each cycle,
  allowing the optimiser to escape local minima that look like plateaus at low LR.
  T_0=10 + T_mult=2 gives cycles of 10, 20, 40, 80 epochs.

Why EMA for early stopping?
  val macro-F1 oscillated ±0.01 per epoch because the WeightedRandomSampler
  over-represents N1 (noisy class), creating high-variance validation error.
  EMA(alpha=0.15) smooths this into a stable trend — the model stops when the
  TREND stops improving, not when one noisy measurement dips.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    EMA_ALPHA,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_LR,
    NUM_CLASSES,
    PATIENCE,
    T_MULT,
    T_RESTART,
    USE_AMP,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


# ── Focal Loss with label smoothing ───────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with label smoothing.

    Label smoothing (ε=0.1) distributes ε/(C-1) probability to non-target
    classes, preventing the model from becoming overconfident.  Combined
    with the focal weighting (1-p_t)^γ, this is especially effective for
    N1 epochs at stage transitions where the "true" label is ambiguous.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = LABEL_SMOOTHING) -> None:
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)   # (B, C)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (C - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # CE with smooth labels — per sample
        ce_loss = -(smooth_targets * log_prob).sum(dim=1)  # (B,)

        # Focal weight based on raw (non-smoothed) probability of true class
        prob = torch.exp(log_prob)
        pt   = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        return (focal_weight * ce_loss).mean()


# ── Early stopping with EMA smoothing ─────────────────────────────────────────

class EarlyStopping:
    """
    Monitors EMA-smoothed validation metric.  Using EMA prevents premature
    stopping when raw metric oscillates due to high-variance mini-batches.
    """

    def __init__(
        self,
        patience:  int   = PATIENCE,
        mode:      str   = "max",
        min_delta: float = 5e-4,
        ema_alpha: float = EMA_ALPHA,
    ) -> None:
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.ema_alpha  = ema_alpha
        self.counter    = 0
        self.ema_score: Optional[float] = None
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict]  = None

    def step(self, raw_score: float, model: nn.Module) -> bool:
        # Update EMA
        if self.ema_score is None:
            self.ema_score = raw_score
        else:
            self.ema_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self.ema_score

        improved = (
            self.best_score is None
            or (self.mode == "max" and self.ema_score > self.best_score + self.min_delta)
            or (self.mode == "min" and self.ema_score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = self.ema_score
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Per-epoch helpers ──────────────────────────────────────────────────────────

def _train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    grad_scaler: Optional[GradScaler],
    use_amp:     bool,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool,
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_amp:
            with autocast("cuda"):
                logits = model(x); loss = criterion(logits, y)
        else:
            logits = model(x); loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return mean_loss, macro_f1


# ── Main training function ─────────────────────────────────────────────────────

def train(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    class_weights: np.ndarray,          # kept for API compat; not used in loss
    device:        torch.device,
    checkpoint_dir: Optional[Path] = None,
) -> nn.Module:
    """
    Train with Focal+LabelSmooth loss, CosineWarmRestarts, EMA early stopping.
    Saves best checkpoint to checkpoint_dir (defaults to CHECKPOINT_DIR).
    """
    save_dir = checkpoint_dir or CHECKPOINT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"

    use_amp     = USE_AMP and device.type == "cuda"
    grad_scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("AMP (FP16) enabled")

    criterion = FocalLoss(gamma=2.0, smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts: restores LR after each cycle, escaping plateaus.
    # eta_min ensures LR never collapses to 0 (which would freeze learning).
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_RESTART, T_mult=T_MULT, eta_min=MIN_LR,
    )
    stopper = EarlyStopping(patience=PATIENCE, mode="max")

    model.to(device)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss       = _train_one_epoch(model, train_loader, criterion, optimizer,
                                             device, grad_scaler, use_amp)
        val_loss, val_f1 = _eval_one_epoch(model, val_loader, criterion, device, use_amp)
        current_lr       = optimizer.param_groups[0]["lr"]

        scheduler.step()   # CosineWarmRestarts steps every epoch

        ema = stopper.ema_score if stopper.ema_score is not None else val_f1
        logger.info(
            "Epoch %3d/%d | loss %.4f/%.4f | val_f1=%.4f ema=%.4f | lr=%.2e",
            epoch, MAX_EPOCHS, train_loss, val_loss, val_f1, ema, current_lr,
        )

        if stopper.step(val_f1, model):
            logger.info(
                "Early stopping at epoch %d — best EMA val macro-F1 = %.4f",
                epoch, stopper.best_score,
            )
            break

    stopper.restore_best(model)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_macro_f1":     stopper.best_score,
            "config": {
                "hidden_size": model.lstm.hidden_size,
                "num_layers":  model.lstm.num_layers,
                "input_size":  model.lstm.input_size,
                "num_classes": model.classifier[-1].out_features,
            },
        },
        checkpoint_path,
    )
    logger.info("Checkpoint saved → %s  (EMA val F1=%.4f)", checkpoint_path, stopper.best_score)
    return model

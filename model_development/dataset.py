"""
PyTorch Dataset: sliding-window sleep staging samples.

Window design rationale
───────────────────────
A single 30-second epoch is an ambiguous classification unit in isolation.
The AASM scoring rules explicitly reference context: N2 is defined by the
presence of at least one sleep spindle or K-complex in the epoch *or* a
nearby epoch; REM continuity depends on the stage of adjacent epochs;
arousals and stage transitions are inherently temporal phenomena.

A 31-epoch sliding window (= 15.5 min of bilateral context) mirrors this
clinical reality.  The model sees:
  - 15 epochs of history  →  captures the build-up of delta power into N3
                             and the slow atonia onset before REM
  - the target epoch      →  the epoch whose label is predicted
  - 15 epochs of future   →  captures stage transitions and confirms
                             stage continuity (critical for REM scoring)

Why 31 (not 20 or 50)?
  ● One full sleep cycle is ~90 min (180 epochs).  31 epochs cover ~17 %
    of a cycle — enough to capture local stage dynamics without dwarfing
    the recurrent state with irrelevant distant context.
  ● Larger windows increase memory and computation quadratically for
    attention models; for BiLSTM the cost is linear and manageable.
  ● Published sequence-based sleep staging work (e.g., SeqSleepNet,
    AttnSleep) uses windows of 20–50 epochs; 31 is a principled midpoint.

Edge handling: SKIP (not pad)
─────────────────────────────
The first and last CENTER_IDX (= 15) epochs of each recording cannot be
window centres because they lack the required bilateral context.  We simply
skip them.  Alternatives are:
  ● Zero-padding: introduces artificial signal discontinuities at the window
    boundary that the LSTM would have to learn to ignore.
  ● Replicate-edge padding: biases the model toward start/end dynamics.
  ● Skipping: clean, lossless for the bulk of the recording.  A ~1200-epoch
    recording loses only 30 / 1200 = 2.5 % of its epochs.

Memory layout
─────────────
All windows are pre-computed and stacked into a (N, 31, 7) float32 array
at construction time.  For the full training set this is:
  153 files × ~1100 epochs × 31 × 7 × 4 bytes ≈ 145 MB  ← fits in RAM.
Pre-computing avoids per-batch file I/O and is faster than lazy loading
at the cost of a one-time startup overhead of a few seconds.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import CENTER_IDX, FEATURE_COLS, LABEL_COL, WINDOW_SIZE
from model_development.data_utils import load_recording


class SleepWindowDataset(Dataset):
    """
    Converts a list of PSG recording files into a flat collection of
    (window, centre_label) samples ready for batch training.

    Args:
        files:  Paths to CSV recordings (all from the same split).
        scaler: A **pre-fitted** StandardScaler.  This is deliberately not
                fitted inside the Dataset to enforce the contract that
                normalisation statistics come exclusively from training data.
    """

    def __init__(self, files: List[Path], scaler: StandardScaler) -> None:
        windows_list: List[np.ndarray] = []
        labels_list:  List[int] = []

        for path in files:
            df = load_recording(path)
            # (N, 7) float64→normalised
            features = scaler.transform(df[FEATURE_COLS].values)
            labels = df[LABEL_COL].values                        # (N,) int

            n_epochs = len(df)
            if n_epochs < WINDOW_SIZE:
                # Recording too short to form even a single window; skip.
                continue

            # Generate one window per valid centre epoch.
            # range(15, N-15) guarantees full [centre-15 … centre+15] context.
            for centre in range(CENTER_IDX, n_epochs - CENTER_IDX):
                window = features[centre -
                                  CENTER_IDX: centre + CENTER_IDX + 1]  # (31, 7)
                windows_list.append(window)
                labels_list.append(int(labels[centre]))

        if not windows_list:
            raise RuntimeError("Dataset is empty — no valid windows produced.")

        # Stack into contiguous arrays for fast __getitem__ indexing.
        self._windows = np.stack(windows_list, axis=0).astype(
            np.float32)  # (N, 31, 7)
        self._labels = np.array(
            labels_list, dtype=np.int64)              # (N,)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : FloatTensor  (WINDOW_SIZE=31, num_features=7)
            y : LongTensor   scalar — integer label of the centre epoch
        """
        x = torch.from_numpy(
            self._windows[idx])     # shares memory — zero-copy
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def label_counts(self) -> dict:
        """Label frequency dict — useful for debugging class balance."""
        unique, counts = np.unique(self._labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

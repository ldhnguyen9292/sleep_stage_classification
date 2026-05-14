"""
Data utilities: file discovery, loading, file-level splitting,
preprocessing pipeline, and class-weight computation.

Critical preprocessing fix: log transform before StandardScaler
───────────────────────────────────────────────────────────────
EEG band power, EOG variance, and EMG RMS are all strictly positive
physical quantities that follow log-normal distributions.  Their raw
skewness values are 4–11 (measured on this dataset), meaning the
StandardScaler alone produces highly skewed normalised features.

An LSTM trained on skewed inputs will have its gradient dominated by
outlier epochs (high-amplitude artefacts, large eye movements) rather
than learning the typical per-stage power profile.  After log transform:
  ● Skewness drops from ~5–11 → ~0.1–0.5
  ● Distributions become approximately Gaussian
  ● All gradient magnitudes are comparable across epochs and features

Implementation: sklearn Pipeline(log → StandardScaler) so the combined
transform can be saved, loaded, and applied as a single object.  All
downstream code that calls scaler.transform(X) requires no changes.

Data leakage contract: the Pipeline is fitted ONLY on training epochs.
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import (
    FEATURE_COLS, LABEL_COL, NUM_CLASSES, SEED, VAL_FRACTION,
)

logger = logging.getLogger(__name__)

# Columns that must exist in the raw CSV files.
# Engineered features (theta_ratio, etc.) are computed inside load_recording
# AFTER this validation — they are NOT expected to be in the CSV.
_RAW_CSV_COLS = [
    "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
    "eeg_beta_power",  "eeg_spindle_power", "eog_var", "emg_rms",
]

# EEG band columns used to compute total power for ratio features.
_EEG_BAND_COLS = [
    "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
    "eeg_beta_power",  "eeg_spindle_power",
]


# ── Feature engineering ────────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append three physiologically-motivated ratio features.

    Why ratios?
      N1 is a TRANSITION stage: its EEG profile is defined by RELATIVE
      changes from wakefulness (alpha reduction, theta emergence) rather
      than absolute power thresholds, which vary hugely across subjects
      and recording sessions.  Ratios are session-invariant and give the
      LSTM an explicit signal for these relative shifts.

      theta_ratio   — θ / Σ_band  rises sharply at Wake→N1 onset.
                      Distinguishes N1 from Wake (low theta) and N2
                      (lower theta_ratio than N1 due to spindle/delta rise).
      delta_ratio   — δ / Σ_band  dominates in N3; near-zero in Wake/REM.
                      Gives the model a clean slow-wave index without
                      needing to know the subject's absolute delta power.
      eog_emg_ratio — EOG_var / EMG_rms encodes the REM signature
                      (large rapid eye movements + muscle atonia) as a
                      single number, separating REM from Wake (high both)
                      and NREM (low EOG, moderate EMG).

    These are computed from RAW (pre-log) values so the ratios represent
    true proportions.  The log transform in the preprocessing pipeline
    then applies to all 10 features uniformly.
    """
    total_eeg = df[_EEG_BAND_COLS].sum(axis=1) + 1e-30
    df = df.copy()
    df["theta_ratio"]   = df["eeg_theta_power"] / total_eeg
    df["delta_ratio"]   = df["eeg_delta_power"]  / total_eeg
    df["eog_emg_ratio"] = df["eog_var"] / (df["emg_rms"] + 1e-30)
    return df


# ── Log transform ──────────────────────────────────────────────────────────────

def _safe_log(X: np.ndarray) -> np.ndarray:
    """
    Natural log with a floor at 1e-30 to guard against exact zeros.

    EEG power values are strictly positive in physics, but floating-point
    underflow or silent data corruption can produce zeros.  Clipping at
    1e-30 (log ≈ -69) is far below any real physiological value and
    prevents -inf from propagating into the scaler fit.
    """
    return np.log(np.maximum(X, 1e-30))


def _make_preprocessing_pipeline() -> Pipeline:
    """
    Returns an unfitted log+StandardScaler pipeline.

    Why not RobustScaler?
      RobustScaler clips the IQR range, which discards information at the
      tails.  Sleep staging performance depends on extreme values (e.g.,
      large delta power bursts in N3, high EMG in wake arousals).
      After log transform the distributions are close enough to Gaussian
      that StandardScaler is appropriate and preserves tail information.
    """
    return Pipeline([
        ("log",   FunctionTransformer(_safe_log, validate=False)),
        ("scale", StandardScaler()),
    ])


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    logger.info("Found %d CSV files in %s", len(files), directory)
    return files


# ── Single-file loading ────────────────────────────────────────────────────────

def load_recording(path: Path) -> pd.DataFrame:
    """
    Load and validate one PSG CSV.  Drops NaN rows, validates label range.
    Returns a clean DataFrame with a contiguous integer index.
    """
    df = pd.read_csv(path)

    # Validate only raw CSV columns; engineered features are computed below.
    missing = [c for c in _RAW_CSV_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    # Drop only on raw CSV columns (engineered features don't exist yet).
    df = df.dropna(subset=_RAW_CSV_COLS + [LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    bad_mask = ~df[LABEL_COL].between(0, NUM_CLASSES - 1)
    if bad_mask.any():
        logger.warning("%s: dropping %d out-of-range label rows", path.name, bad_mask.sum())
        df = df[~bad_mask].reset_index(drop=True)

    df = _engineer_features(df)
    return df


# ── File-level split ───────────────────────────────────────────────────────────

def split_files(
    files: List[Path],
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> Tuple[List[Path], List[Path]]:
    """
    Stratified file-level train/val split (stratification key: majority stage).
    Falls back to random split if any stratum is too small for stratification.
    """
    majority_labels: List[int] = []
    valid_files:     List[Path] = []

    for f in files:
        try:
            df = load_recording(f)
            majority_labels.append(int(df[LABEL_COL].mode().iloc[0]))
            valid_files.append(f)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    try:
        train_files, val_files = train_test_split(
            valid_files, test_size=val_fraction,
            stratify=majority_labels, random_state=seed,
        )
    except ValueError:
        logger.warning("Stratified split failed — falling back to random split.")
        train_files, val_files = train_test_split(
            valid_files, test_size=val_fraction, random_state=seed,
        )

    logger.info("Split → train: %d files  |  val: %d files", len(train_files), len(val_files))
    return sorted(train_files), sorted(val_files)


# ── Preprocessing pipeline (scaler) ───────────────────────────────────────────

def fit_scaler(train_files: List[Path]) -> Pipeline:
    """
    Fit the log+StandardScaler pipeline on concatenated training epochs.

    Fitted on training data ONLY to prevent leaking distribution statistics
    (mean, std) from validation or external test sets into the normalisation.
    """
    all_features: List[np.ndarray] = []
    for path in train_files:
        df = load_recording(path)
        all_features.append(df[FEATURE_COLS].values)

    X_train = np.concatenate(all_features, axis=0)   # (N_total_epochs, 7)
    pipeline = _make_preprocessing_pipeline()
    pipeline.fit(X_train)

    # Sanity-check: confirm log transform reduced skewness substantially
    from scipy.stats import skew as _skew
    X_log      = _safe_log(X_train)
    sk_raw     = [_skew(X_train[:, i]) for i in range(X_train.shape[1])]
    sk_log     = [_skew(X_log[:, i])   for i in range(X_log.shape[1])]
    logger.info(
        "Pipeline fitted on %d epochs | "
        "raw skewness [%.1f, %.1f] → log skewness [%.2f, %.2f]",
        X_train.shape[0],
        min(sk_raw), max(sk_raw),
        min(sk_log), max(sk_log),
    )
    return pipeline


def save_scaler(scaler: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Preprocessing pipeline saved → %s", path)


def load_scaler(path: Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Class weights ──────────────────────────────────────────────────────────────

def compute_class_weights(train_files: List[Path]) -> np.ndarray:
    """
    Inverse-frequency weights: weight_c = N / (C × n_c).

    N3 (~6.7%) and N1 (~11%) receive 4–5× more weight than N2 (~35%),
    preventing the model from defaulting to predicting N2 for everything.
    """
    counts: Counter = Counter()
    for path in train_files:
        df = load_recording(path)
        counts.update(df[LABEL_COL].tolist())

    total   = sum(counts.values())
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for cls in range(NUM_CLASSES):
        n_c = counts.get(cls, 0)
        weights[cls] = (total / (NUM_CLASSES * n_c)) if n_c > 0 else 0.0
        if n_c == 0:
            logger.warning("Class %d absent from training set.", cls)

    logger.info("Class weights: %s", {i: f"{w:.3f}" for i, w in enumerate(weights)})
    return weights

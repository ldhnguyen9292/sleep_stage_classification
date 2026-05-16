"""
Central configuration — all hyperparameters and paths in one place.

Changes from v1:
  - HIDDEN_SIZE 128 → 256  (more capacity, 16 GB VRAM has headroom)
  - BATCH_SIZE  128 → 512  (larger batches → stable gradients, faster epochs)
  - DROPOUT     0.3 → 0.35
  - PATIENCE    15  → 20   (give the model more time after log-transform fix)
  - LR_PATIENCE  5  →  7
  - USE_AMP = True          (FP16 mixed-precision on RTX 5060 Ti)
  - WEIGHT_DECAY added      (L2 regularisation in Adam)
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
INTERNAL_DIR = DATA_DIR / "internal"
EXTERNAL_DIR = DATA_DIR / "external"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Features ───────────────────────────────────────────────────────────────────
# All seven channels are physical quantities (power, variance, RMS) that are
# strictly positive and follow log-normal distributions in raw µV²/Hz space.
# A log transform is applied before StandardScaler inside data_utils.py.
FEATURE_COLS = [
    # ── Raw power / amplitude features ─────────────────────────────────────────
    "eeg_delta_power",    # <4 Hz slow oscillations — dominant in N3
    "eeg_theta_power",    # 4–8 Hz — N1 vertex waves, REM theta activity
    "eeg_alpha_power",    # 8–12 Hz — occipital alpha, eyes-closed wakefulness
    "eeg_beta_power",     # 12–30 Hz — cortical activation, Wake, arousals
    "eeg_spindle_power",  # 12–15 Hz spindles — hallmark of N2
    "eog_var",            # EOG variance — rapid eye movements → REM
    "emg_rms",            # Chin EMG — high in Wake, atonic in REM
    # ── Engineered ratio features (computed in data_utils.load_recording) ──────
    # Ratios capture RELATIVE band dominance, which is more stable across
    # recording sessions than absolute power.  After log-transform these become
    # linear combinations of the raw features and are especially helpful for
    # disambiguating N1, which is defined by RELATIVE changes (alpha drops,
    # theta rises) rather than absolute thresholds.
    "theta_ratio",    # θ / Σ_EEG  — key N1 marker (rising relative theta)
    "delta_ratio",    # δ / Σ_EEG  — key N3 marker (slow-wave dominance)
    "eog_emg_ratio",  # EOG / EMG  — REM: high EOG + low EMG; Wake: both high
]
LABEL_COL = "label"

# Integer label encoding already in CSVs (Sleep-EDF / AASM 5-stage):
#   0=Wake  1=N1  2=N2  3=N3  4=REM
NUM_CLASSES = 5
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# ── Sliding window ─────────────────────────────────────────────────────────────
WINDOW_SIZE = 31          # 31 epochs × 30 s = 15.5 min bilateral context
CENTER_IDX = WINDOW_SIZE // 2   # = 15

# ── Data split ─────────────────────────────────────────────────────────────────
VAL_FRACTION = 0.20   # file-level split; 20 % → internal validation

# ── Model ─────────────────────────────────────────────────────────────────────
INPUT_SIZE = len(FEATURE_COLS)   # 10  (7 raw + 3 engineered ratios)
HIDDEN_SIZE = 256   # per-direction units; 256 × 2 = 512 at classifier input
NUM_LAYERS = 2
DROPOUT = 0.35

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 5
PATIENCE = 20        # applied to EMA-smoothed val F1, not raw (see train.py)
# smoothing factor for early-stop metric (reduces oscillation)
EMA_ALPHA = 0.15
MIN_LR = 1e-6

# CosineAnnealingWarmRestarts — replaces ReduceLROnPlateau
# Cycle 1: 0→T_RESTART epochs, then doubles each restart.
# Warm restarts escape local minima that trap ReduceLROnPlateau.
T_RESTART = 10        # first cycle length (epochs)
T_MULT = 2         # each cycle is T_MULT× longer

# Label smoothing for FocalLoss — reduces overconfidence on noisy N1 boundaries
# (61 % of N1 runs are ≤2 epochs, many are at ambiguous stage transitions)
LABEL_SMOOTHING = 0.10

# K-Fold cross validation
K_FOLDS = 5               # stratified at file level

# ── Hardware ───────────────────────────────────────────────────────────────────
# AMP: mixed FP16/FP32 precision halves VRAM use and accelerates matrix ops
# on Tensor Cores.  Set False if you hit numerical instability warnings.
USE_AMP = True
NUM_WORKERS = 0    # keep 0 on Windows (multiprocessing pickling issues)
PIN_MEMORY = True

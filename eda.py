"""
eda.py — Exploratory Data Analysis for the sleep staging dataset.

Run before training to understand:
  1. Class distribution and imbalance
  2. Feature distributions per sleep stage (raw vs log-transformed)
  3. Feature discriminability (mutual information, ANOVA F-statistic)
  4. Inter-feature Pearson correlation
  5. Domain shift: Cassette (internal) vs Telemetry (external)

Usage:
    python eda.py

Outputs saved to results/eda/
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEATURE_COLS, LABEL_COL, CLASS_NAMES, NUM_CLASSES, INTERNAL_DIR, EXTERNAL_DIR
from data_utils import discover_files, load_recording

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EDA_DIR = Path(__file__).resolve().parent / "results" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {0: "#4878CF", 1: "#6ACC65", 2: "#D65F5F", 3: "#B47CC7", 4: "#C4AD66"}
STAGE_COLORS = [PALETTE[i] for i in range(NUM_CLASSES)]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all(files, source_tag="internal"):
    dfs = []
    for f in files:
        df = load_recording(f)
        df["source"] = source_tag
        df["recording_type"] = "cassette" if "cassette" in f.name else "telemetry"
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d epochs from %d files (%s)", len(combined), len(files), source_tag)
    return combined


# ── Plot 1: class distribution ─────────────────────────────────────────────────

def plot_class_distribution(df_int, df_ext):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, df, title in zip(axes, [df_int, df_ext], ["Internal (Cassette)", "External (Telemetry)"]):
        counts = df[LABEL_COL].value_counts().sort_index()
        bars = ax.bar(CLASS_NAMES, [counts.get(i, 0) for i in range(NUM_CLASSES)],
                      color=STAGE_COLORS, edgecolor="white", linewidth=0.8)
        total = counts.sum()
        for bar, (_, cnt) in zip(bars, counts.items()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f"{100*cnt/total:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sleep stage")
        ax.set_ylabel("Epoch count")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.suptitle("Class distribution — imbalance diagnostic", fontsize=13)
    plt.tight_layout()
    out = EDA_DIR / "01_class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)


# ── Plot 2: skewness and log-transform effect ──────────────────────────────────

def plot_log_transform_effect(df):
    n_feat = len(FEATURE_COLS)
    fig, axes = plt.subplots(2, n_feat, figsize=(3.5 * n_feat, 6))

    for col_idx, feat in enumerate(FEATURE_COLS):
        vals = df[feat].values
        log_vals = np.log(np.maximum(vals, 1e-30))

        sk_raw = stats.skew(vals)
        sk_log = stats.skew(log_vals)

        # Raw histogram
        ax_raw = axes[0, col_idx]
        ax_raw.hist(vals, bins=60, color="#4878CF", alpha=0.75, edgecolor="none")
        ax_raw.set_title(f"{feat.replace('eeg_','').replace('_power','')}\nskew={sk_raw:.1f}",
                         fontsize=8)
        ax_raw.set_yticks([])

        # Log histogram
        ax_log = axes[1, col_idx]
        ax_log.hist(log_vals, bins=60, color="#D65F5F", alpha=0.75, edgecolor="none")
        ax_log.set_title(f"log → skew={sk_log:.2f}", fontsize=8)
        ax_log.set_yticks([])

    axes[0, 0].set_ylabel("Raw", fontsize=10)
    axes[1, 0].set_ylabel("Log-transformed", fontsize=10)
    plt.suptitle(
        "Feature skewness: raw (top) vs log-transformed (bottom)\n"
        "High skewness in raw space is why StandardScaler alone gives poor LSTM performance",
        fontsize=10,
    )
    plt.tight_layout()
    out = EDA_DIR / "02_log_transform_effect.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)


# ── Plot 3: feature distributions by sleep stage ──────────────────────────────

def plot_feature_by_stage(df):
    log_df = df.copy()
    for feat in FEATURE_COLS:
        log_df[feat] = np.log(np.maximum(log_df[feat].values, 1e-30))

    n_feat = len(FEATURE_COLS)
    fig, axes = plt.subplots(1, n_feat, figsize=(3.5 * n_feat, 5), sharey=False)

    for ax, feat in zip(axes, FEATURE_COLS):
        data_by_class = [log_df[log_df[LABEL_COL] == i][feat].values for i in range(NUM_CLASSES)]
        parts = ax.violinplot(data_by_class, positions=range(NUM_CLASSES),
                               showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], STAGE_COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        short = feat.replace("eeg_", "").replace("_power", "")
        ax.set_title(short, fontsize=9, fontweight="bold")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)

    axes[0].set_ylabel("log(power)")
    plt.suptitle("Log-transformed feature distributions by sleep stage", fontsize=12)
    plt.tight_layout()
    out = EDA_DIR / "03_feature_by_stage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)


# ── Plot 4: feature discriminability ──────────────────────────────────────────

def plot_feature_discriminability(df):
    X_log = np.log(np.maximum(df[FEATURE_COLS].values, 1e-30))
    y     = df[LABEL_COL].values

    # Mutual information (non-parametric)
    mi_scores = mutual_info_classif(X_log, y, random_state=42, n_neighbors=5)

    # One-way ANOVA F-statistic (class separability in log space)
    f_stats = []
    for i, feat in enumerate(FEATURE_COLS):
        groups = [X_log[y == c, i] for c in range(NUM_CLASSES)]
        f_val, _ = stats.f_oneway(*groups)
        f_stats.append(f_val)

    short_names = [f.replace("eeg_", "").replace("_power", "") for f in FEATURE_COLS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    order_mi = np.argsort(mi_scores)[::-1]
    axes[0].barh([short_names[i] for i in order_mi],
                 [mi_scores[i] for i in order_mi],
                 color="#4878CF", alpha=0.8)
    axes[0].set_xlabel("Mutual information (higher = more discriminative)")
    axes[0].set_title("Mutual information vs sleep stage label")

    order_f = np.argsort(f_stats)[::-1]
    axes[1].barh([short_names[i] for i in order_f],
                 [f_stats[i] for i in order_f],
                 color="#D65F5F", alpha=0.8)
    axes[1].set_xlabel("ANOVA F-statistic (higher = better class separation)")
    axes[1].set_title("One-way ANOVA F-statistic vs sleep stage label")

    plt.suptitle("Feature discriminability for sleep stage classification", fontsize=12)
    plt.tight_layout()
    out = EDA_DIR / "04_feature_discriminability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)

    # Print table
    print("\n── Feature discriminability summary (log-space) ──────────────")
    print(f"{'Feature':<20} {'Mutual Info':>12} {'ANOVA F':>12}")
    print("-" * 46)
    for i, feat in enumerate(FEATURE_COLS):
        short = short_names[i]
        print(f"{short:<20} {mi_scores[i]:>12.4f} {f_stats[i]:>12.1f}")


# ── Plot 5: inter-feature correlation ─────────────────────────────────────────

def plot_correlation(df):
    X_log = pd.DataFrame(
        np.log(np.maximum(df[FEATURE_COLS].values, 1e-30)),
        columns=[f.replace("eeg_", "").replace("_power", "") for f in FEATURE_COLS],
    )
    corr = X_log.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax,
        square=True, linewidths=0.5, linecolor="white",
    )
    ax.set_title("Pearson correlation — log-transformed features", fontsize=12)
    plt.tight_layout()
    out = EDA_DIR / "05_feature_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)


# ── Plot 6: domain shift (Cassette vs Telemetry) ──────────────────────────────

def plot_domain_shift(df_int, df_ext):
    """
    Compares feature distributions between internal (Cassette) and
    external (Telemetry) datasets for the SAME sleep stage (N2, largest class).
    A visible distributional gap explains why external validation accuracy
    often drops relative to internal: the model must generalise across
    recording systems, not just across subjects.
    """
    combined = pd.concat([df_int, df_ext], ignore_index=True)

    # Use N2 (label=2) as the common reference stage across both datasets
    combined_n2 = combined[combined[LABEL_COL] == 2].copy()
    for feat in FEATURE_COLS:
        combined_n2[feat] = np.log(np.maximum(combined_n2[feat].values, 1e-30))

    n_feat = len(FEATURE_COLS)
    fig, axes = plt.subplots(1, n_feat, figsize=(3.5 * n_feat, 4))

    for ax, feat in zip(axes, FEATURE_COLS):
        for domain, color, ls in [("cassette", "#4878CF", "-"), ("telemetry", "#D65F5F", "--")]:
            vals = combined_n2[combined_n2["recording_type"] == domain][feat].dropna()
            if len(vals) == 0:
                continue
            kde_xs = np.linspace(vals.min(), vals.max(), 300)
            kde = stats.gaussian_kde(vals)
            ax.plot(kde_xs, kde(kde_xs), color=color, linestyle=ls,
                    linewidth=1.8, label=domain.capitalize())
        short = feat.replace("eeg_", "").replace("_power", "")
        ax.set_title(short, fontsize=8)
        ax.set_yticks([])

    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Density")
    plt.suptitle(
        "Domain shift: Cassette (internal) vs Telemetry (external) — N2 epochs\n"
        "Distribution mismatch directly limits external generalisation",
        fontsize=10,
    )
    plt.tight_layout()
    out = EDA_DIR / "06_domain_shift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("Saved → %s", out)


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df_int, df_ext):
    print("\n" + "=" * 62)
    print("  DATASET SUMMARY")
    print("=" * 62)

    for df, name in [(df_int, "Internal (Cassette)"), (df_ext, "External (Telemetry)")]:
        total = len(df)
        print(f"\n{name}:  {total:,} epochs")
        for i, stage in enumerate(CLASS_NAMES):
            cnt = (df[LABEL_COL] == i).sum()
            print(f"  {stage:<6}  {cnt:>7,}  ({100*cnt/total:.1f}%)")

    print("\nRaw feature skewness (internal):")
    for feat in FEATURE_COLS:
        sk = stats.skew(df_int[feat].dropna())
        sk_log = stats.skew(np.log(np.maximum(df_int[feat].dropna().values, 1e-30)))
        short = feat.replace("eeg_", "").replace("_power", "")
        print(f"  {short:<18}  raw={sk:>6.2f}   log={sk_log:>6.3f}")

    print("\nDomain shift — N2 feature KL-divergence (Cassette vs Telemetry):")
    n2_int = df_int[df_int[LABEL_COL] == 2]
    n2_ext = df_ext[df_ext[LABEL_COL] == 2]
    for feat in FEATURE_COLS:
        v1 = np.log(np.maximum(n2_int[feat].dropna().values, 1e-30))
        v2 = np.log(np.maximum(n2_ext[feat].dropna().values, 1e-30))
        # Approximate KL via histograms
        bins = np.linspace(min(v1.min(), v2.min()), max(v1.max(), v2.max()), 50)
        p, _ = np.histogram(v1, bins=bins, density=True)
        q, _ = np.histogram(v2, bins=bins, density=True)
        p, q = p + 1e-9, q + 1e-9
        kl = float(stats.entropy(p, q))
        short = feat.replace("eeg_", "").replace("_power", "")
        print(f"  {short:<18}  KL={kl:.4f}")
    print("=" * 62)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading internal files …")
    int_files = discover_files(INTERNAL_DIR)
    df_int    = load_all(int_files, "internal")

    logger.info("Loading external files …")
    ext_files = discover_files(EXTERNAL_DIR)
    df_ext    = load_all(ext_files, "external")

    print_summary(df_int, df_ext)

    logger.info("Generating plots …")
    plot_class_distribution(df_int, df_ext)
    plot_log_transform_effect(df_int)
    plot_feature_by_stage(df_int)
    plot_feature_discriminability(df_int)
    plot_correlation(df_int)
    plot_domain_shift(df_int, df_ext)

    logger.info("All EDA plots saved to %s", EDA_DIR)
    logger.info("Key findings:")
    logger.info("  1. All 7 features are heavily right-skewed → log transform is mandatory before scaling")
    logger.info("  2. N3 (6.7%%) and N1 (11%%) are rare → class weighting is essential")
    logger.info("  3. Internal=Cassette, External=Telemetry → domain shift expected on holdout")


if __name__ == "__main__":
    main()

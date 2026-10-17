#!/usr/bin/env python3
"""
=======================================================================
 05_publication_figures.py
 Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
 Purpose : Generate publication-quality figures (300 dpi, APA-style):
           1) Group means with 95% CI error bars
           2) Raincloud-style violin + box plot
           3) Pre-post change plot per group
 Usage   : python 05_publication_figures.py --input data_clean.csv \
             --outdir figures [--pre-col pre_score --post-col post_score]
 Author  : LICA-AI Research Team
=======================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {"control": "#4C72B0", "lica_ai": "#DD8452"}


def mean_ci(x: np.ndarray) -> tuple:
    """Mean and 95% normal-approximation CI."""
    x = x[~np.isnan(x)]
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * se


def fig_group_means(df, group_col, score_col, outdir: Path) -> Path:
    """Bar-less (Cleveland-style) mean plot with 95% CI."""
    fig, ax = plt.subplots(figsize=(5, 4))
    groups = sorted(df[group_col].astype(str).str.lower().str.strip().unique())
    for i, g in enumerate(groups):
        vals = pd.to_numeric(df.loc[df[group_col].astype(str).str.lower().str.strip() == g, score_col],
                             errors="coerce").to_numpy()
        m, ci = mean_ci(vals)
        ax.errorbar(i, m, yerr=ci, fmt="o", capsize=5, color=COLORS.get(g, "gray"),
                    markersize=7, linewidth=1.5)
        ax.annotate(f"M = {m:.1f}", (i, m), textcoords="offset points",
                    xytext=(12, -4), fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("_", " ").title() for g in groups])
    ax.set_ylabel("Writing Score")
    ax.set_xlabel("")
    ax.set_title("Group Means with 95% Confidence Intervals")
    out = outdir / "fig1_group_means.png"
    fig.savefig(out); plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_distribution(df, group_col, score_col, outdir: Path) -> Path:
    """Violin + box + jittered points per condition."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn unavailable; skipping distribution figure.")
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_df = df.copy()
    plot_df[group_col] = plot_df[group_col].astype(str).str.lower().str.strip()
    sns.violinplot(data=plot_df, x=group_col, y=score_col, ax=ax,
                   palette=COLORS, inner=None, cut=0)
    sns.boxplot(data=plot_df, x=group_col, y=score_col, ax=ax, width=0.25,
                showcaps=True, boxprops={"facecolor": "white", "alpha": 0.8})
    rng = np.random.default_rng(1)
    for i, g in enumerate(sorted(plot_df[group_col].unique())):
        vals = pd.to_numeric(plot_df.loc[plot_df[group_col] == g, score_col],
                             errors="coerce").dropna().to_numpy()
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12, alpha=0.55,
                   color=COLORS.get(g, "gray"), edgecolor="none", zorder=3)
    ax.set_xticklabels([t.get_text().replace("_", " ").title()
                        for t in ax.get_xticklabels()])
    ax.set_xlabel(""); ax.set_ylabel("Writing Score")
    ax.set_title("Score Distributions by Condition")
    out = outdir / "fig2_distributions.png"
    fig.savefig(out); plt.close(fig)
    logger.info("Saved %s", out)
    return out


def fig_pre_post(df, group_col, pre_col, post_col, outdir: Path) -> Path:
    """Paired pre-post lines (sample of participants) per group."""
    if pre_col not in df.columns or post_col not in df.columns:
        logger.warning("Pre/post columns not found; skipping change figure.")
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for g, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[pre_col, post_col])
        pre = pd.to_numeric(sub[pre_col], errors="coerce")
        post = pd.to_numeric(sub[post_col], errors="coerce")
        ax.plot([0, 1], [pre.mean(), post.mean()], "o-", color=COLORS.get(str(g).lower(), "gray"),
                linewidth=2.5, markersize=8, label=str(g).replace("_", " ").title())
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre-test", "Post-test"])
    ax.set_ylabel("Writing Score")
    ax.set_title("Mean Change from Pre-test to Post-test")
    ax.legend(frameon=False)
    out = outdir / "fig3_pre_post_change.png"
    fig.savefig(out); plt.close(fig)
    logger.info("Saved %s", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication figures for LICA-AI study.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--score-col", default="score_mean")
    parser.add_argument("--pre-col", default="pre_score")
    parser.add_argument("--post-col", default="post_score")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if args.score_col not in df.columns:
            raise KeyError(f"Score column '{args.score_col}' not found. "
                           f"Available: {list(df.columns)}")
        fig_group_means(df, args.group_col, args.score_col, outdir)
        fig_distribution(df, args.group_col, args.score_col, outdir)
        fig_pre_post(df, args.group_col, args.pre_col, args.post_col, outdir)
        logger.info("All requested figures generated in %s", outdir)
        return 0
    except Exception as exc:
        logger.error("Figure generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

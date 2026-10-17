#!/usr/bin/env python3
"""
=======================================================================
 04_effect_sizes_morris_d.py
 Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
 Purpose : Compute Morris (2008) pre-post standardized mean-change
           effect sizes (d_av and d_rm) and between-group differences,
           with bootstrap 95% confidence intervals.
 Usage   : python 04_effect_sizes_morris_d.py --input data_clean.csv \
             --pre-col pre_score --post-col post_score \
             --group-col group --output effect_sizes.csv
 Author  : LICA-AI Research Team
=======================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

RNG_SEED = 2024
N_BOOT = 5000


def morris_d(pre: np.ndarray, post: np.ndarray, variant: str = "av") -> float:
    """
    Morris (2008) standardized mean-change effect size for one group.

    d_av : pooled pre/post SD (averages the two variances)
    d_rm : SD of difference scores corrected for correlation
    """
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    mask = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[mask], post[mask]
    n = len(pre)
    if n < 3:
        raise ValueError("Need >= 3 complete pre/post pairs.")

    mean_change = post.mean() - pre.mean()
    s_pre, s_post = pre.std(ddof=1), post.std(ddof=1)
    s_av = np.sqrt((s_pre**2 + s_post**2) / 2)
    r = np.corrcoef(pre, post)[0, 1]
    if variant == "av":
        d = mean_change / s_av
    elif variant == "rm":
        s_diff = post.std(ddof=1) * np.sqrt(2 * (1 - r)) if np.isfinite(r) else np.nan
        d = mean_change / s_diff
    else:
        raise ValueError("variant must be 'av' or 'rm'")
    # small-sample correction (Morris, 2008)
    correction = 1 - 3 / (4 * n - 5)
    return d * correction


def bootstrap_ci(pre, post, variant, n_boot=N_BOOT, seed=RNG_SEED) -> tuple:
    """Percentile bootstrap CI for Morris d."""
    rng = np.random.default_rng(seed)
    mask = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[mask], post[mask]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pre), len(pre))
        try:
            stats.append(morris_d(pre[idx], post[idx], variant))
        except Exception:
            continue
    stats = np.asarray(stats)
    if stats.size == 0:
        return np.nan, np.nan
    return tuple(np.percentile(stats, [2.5, 97.5]))


def hedges_g_between(g1: np.ndarray, g2: np.ndarray) -> dict:
    """Hedges' g for between-group post-test comparison."""
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group needs >= 2 observations.")
    s_pooled = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    d = (g2.mean() - g1.mean()) / s_pooled
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return {"hedges_g": d * J, "cohens_d": d, "n1": n1, "n2": n2}


def main() -> int:
    parser = argparse.ArgumentParser(description="Morris (2008) effect sizes.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--pre-col", default="pre_score")
    parser.add_argument("--post-col", default="post_score")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--output", default="effect_sizes.csv")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        for col in [args.pre_col, args.post_col, args.group_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {args.input}")

        rows = []
        for g, sub in df.groupby(args.group_col):
            pre = pd.to_numeric(sub[args.pre_col], errors="coerce").to_numpy()
            post = pd.to_numeric(sub[args.post_col], errors="coerce").to_numpy()
            for variant in ("av", "rm"):
                d = morris_d(pre, post, variant)
                lo, hi = bootstrap_ci(pre, post, variant)
                rows.append({"group": g, "effect": f"morris_d_{variant}",
                             "value": round(d, 4), "ci_low": round(lo, 4),
                             "ci_high": round(hi, 4)})
                logger.info("%s | %s = %.3f [%.3f, %.3f]", g, variant, d, lo, hi)

        # Between-group contrast on post scores
        groups = df[args.group_col].astype(str).str.lower().str.strip().unique()
        if len(groups) == 2:
            g1, g2 = sorted(groups)
            es = hedges_g_between(
                pd.to_numeric(df.loc[df[args.group_col].str.lower().str.strip() == g1, args.post_col], errors="coerce").to_numpy(),
                pd.to_numeric(df.loc[df[args.group_col].str.lower().str.strip() == g2, args.post_col], errors="coerce").to_numpy(),
            )
            rows.append({"group": f"{g1}_vs_{g2}", "effect": "hedges_g_between",
                         "value": round(es["hedges_g"], 4), "ci_low": np.nan, "ci_high": np.nan})
            logger.info("Between-group Hedges' g = %.3f", es["hedges_g"])

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Saved effect sizes to %s", out)
        return 0
    except Exception as exc:
        logger.error("Effect-size analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

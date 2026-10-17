#!/usr/bin/env python3
"""
=======================================================================
 02_inter_rater_reliability_icc.py
 Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
 Purpose : Compute Inter-Rater Reliability via ICC(2,1) and ICC(3,1)
           (two-way random/fixed, single measures) with 95% CIs using
           pingouin (falls back to a manual Shrout-Fleiss ANOVA method).
 Usage   : python 02_inter_rater_reliability_icc.py --input data_clean.csv \
             --output icc_results.csv
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

RATERS = ["rater1", "rater2"]


def icc_manual(ratings: np.ndarray) -> dict:
    """Manual Shrout-Fleiss ICC(2,1)/ICC(3,1) for 2 raters with F-based CI."""
    ratings = ratings[~np.isnan(ratings).any(axis=1)]
    n, k = ratings.shape
    if n < 3:
        raise ValueError(f"Need >=3 complete dyads; got {n}")

    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ss_total = ((ratings - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    if ss_err < 0:
        ss_err = max(ss_err, 0.0)

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    icc21 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)
    icc31 = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)

    f = ms_rows / ms_err if ms_err > 0 else np.inf
    fl = f * (1 + (k - 1) * icc21) - (k - 1)  # lower F
    fu = f / (1 + (k - 1) * icc21)            # upper F
    try:
        from scipy import stats
        fl = max(fl, 1e-12)
        ci_lo = (fl - 1) / (fl + k - 1)
        ci_hi = (fu * k - 1) / (fu * k + k - 1)
    except Exception:
        ci_lo, ci_hi = np.nan, np.nan

    return {"n": n, "k": k, "ICC21": icc21, "ICC31": icc31,
            "ICC21_CI_low": ci_lo, "ICC21_CI_high": ci_hi}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inter-rater reliability (ICC) analysis.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="icc_results.csv")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        ratings = df[RATERS].to_numpy(dtype=float)
    except Exception as exc:
        logger.error("Could not load data: %s", exc)
        return 1

    result = pd.DataFrame()
    try:
        import pingouin as pg
        r = pg.intraclass_corr(
            data=df.melt(value_vars=RATERS, var_name="rater", value_name="score").dropna(),
            targets="__melt_index__" if False else None,
        ) if False else None
        # pingouin expects long data: targets/raters/ratings
        long = df.reset_index().melt(id_vars="index", value_vars=RATERS,
                                     var_name="rater", value_name="score").dropna()
        res = pg.intraclass_corr(data=long, targets="index", raters="rater", ratings="score")
        res = res[res["Type"].isin(["ICC2", "ICC3"])]
        result = res[["Type", "ICC", "CI95%"]].rename(columns={"Type": "method"})
        result.columns = ["method", "icc", "ci95"]
        logger.info("pingouin ICC results:\n%s", result.to_string(index=False))
    except ImportError:
        logger.warning("pingouin not installed; using manual Shrout-Fleiss method.")
    except Exception as exc:
        logger.warning("pingouin failed (%s); using manual method.", exc)

    if result.empty:
        manual = icc_manual(ratings)
        result = pd.DataFrame([{
            "method": "ICC(2,1) manual", "icc": manual["ICC21"],
            "ci95": f"[{manual['ICC21_CI_low']:.3f}, {manual['ICC21_CI_high']:.3f}]",
            "n": manual["n"],
        }, {
            "method": "ICC(3,1) manual", "icc": manual["ICC31"], "n": manual["n"],
        }])
        logger.info("Manual ICC results:\n%s", result.to_string(index=False))

    # Interpretation (Koo & Li, 2016)
    def interpret(v):
        if v < 0.50: return "poor"
        if v < 0.75: return "moderate"
        if v < 0.90: return "good"
        return "excellent"
    result["interpretation"] = result["icc"].apply(interpret)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    logger.info("Saved ICC results to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

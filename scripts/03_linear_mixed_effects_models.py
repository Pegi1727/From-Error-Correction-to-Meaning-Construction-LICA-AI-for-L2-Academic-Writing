#!/usr/bin/env python3
"""
=======================================================================
 03_linear_mixed_effects_models.py
 Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
 Purpose : Fit Linear Mixed-Effects Models of writing scores with
           condition (control vs LICA-AI) as a fixed effect and
           rater/student as random effects; report ANOVA tests and
           estimated marginal means.
 Usage   : python 03_linear_mixed_effects_models.py --input data_clean_long.csv \
             --output lmm_results.txt
 Author  : LICA-AI Research Team
=======================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_long_data(path: Path) -> pd.DataFrame:
    """Load tidy long data (one row per student x rater)."""
    df = pd.read_csv(path)
    required = {"student_id", "group", "rater", "score"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")
    df["group"] = df["group"].astype(str).str.strip().str.lower()
    df["rater"] = df["rater"].astype(str).str.strip().str.lower()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    if df["group"].nunique() < 2:
        raise ValueError("Both 'control' and 'lica_ai' groups must be present.")
    return df


def fit_models(df: pd.DataFrame):
    """Fit nested LMMs and return (model objects, lines of summary text)."""
    import statsmodels.formula.api as smf

    lines = []
    models = {}

    # M0: unconditional (null) model - variance decomposition
    m0 = smf.mixedlm("score ~ 1", df, groups=df["student_id"]).fit(reml=True)
    models["M0_null"] = m0
    lines.append("=" * 70)
    lines.append("M0: Unconditional model  score ~ 1 + (1 | student_id)")
    lines.append("=" * 70)
    lines.append(str(m0.summary()))

    # M1: condition fixed effect + rater random intercept + student random intercept
    m1 = smf.mixedlm(
        "score ~ C(group)", df,
        groups=df["student_id"],
        re_formula="1",  # random intercept for student
    ).fit(reml=True)
    models["M1_condition"] = m1
    lines.append("=" * 70)
    lines.append("M1: score ~ group + (1 | student_id)  [rater averaged via long data]")
    lines.append("=" * 70)
    lines.append(str(m1.summary()))

    # Likelihood-ratio style comparison (AIC/BIC)
    lines.append("=" * 70)
    lines.append("Model comparison")
    lines.append("=" * 70)
    comp = pd.DataFrame({
        "model": list(models),
        "AIC": [m.aic for m in models.values()],
        "BIC": [m.bic for m in models.values()],
        "logLik": [m.llf for m in models.values()],
    })
    lines.append(comp.to_string(index=False))
    return models, lines


def report_group_means(df: pd.DataFrame, lines: list) -> None:
    """Append descriptives and estimated marginal means by condition."""
    import numpy as np
    lines.append("=" * 70)
    lines.append("Estimated marginal means (model-based) and raw group means")
    lines.append("=" * 70)
    raw = df.groupby("group")["score"].agg(["count", "mean", "std", "sem"])
    lines.append(raw.to_string())
    for g in df["group"].unique():
        sub = df.loc[df["group"] == g, "score"]
        ci = 1.96 * sub.sem()
        lines.append(f"{g}: M = {sub.mean():.2f}, SE = {sub.sem():.2f}, "
                     f"95% CI [{sub.mean()-ci:.2f}, {sub.mean()+ci:.2f}], n = {sub.count()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LMM analysis for LICA-AI study.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="lmm_results.txt")
    args = parser.parse_args()

    try:
        df = load_long_data(Path(args.input))
        logger.info("Loaded %d long-format rows", len(df))
        models, lines = fit_models(df)
        report_group_means(df, lines)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved LMM results to %s", out)
        return 0
    except ImportError as exc:
        logger.error("Missing dependency (install via requirements.txt): %s", exc)
        return 1
    except Exception as exc:
        logger.error("LMM analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

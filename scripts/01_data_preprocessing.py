#!/usr/bin/env python3
"""
=======================================================================
 01_data_preprocessing.py
 Project : From Error Correction to Meaning Construction:
           Evaluating LICA-AI as a Real-Time Cognitive Scaffold
           for L2 Academic Writing
 Purpose : Clean, validate, and pre-process raw essay-score data
           (CSV/Excel) for downstream reliability, LMM, and effect-
           size analyses. Produces a tidy long-format analysis file.
 Usage   : python 01_data_preprocessing.py --input raw_scores.xlsx \
             --output data_clean.csv [--group-col group] \
             [--id-col student_id] [--sheet 0]
 Author  : LICA-AI Research Team
=======================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- #
# Logging configuration
# ---------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EXPECTED_RATERS = ["rater1", "rater2"]
EXPECTED_CONDITIONS = {"control", "lica_ai"}

VALID_SCORE_MIN, VALID_SCORE_MAX = 0.0, 100.0


def load_raw_data(path: Path, sheet) -> pd.DataFrame:
    """Load raw data from CSV or Excel with robust error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, sheet_name=sheet)
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read '{path}': {exc}") from exc


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, snake-case all column names."""
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    )
    return df


def validate_structure(df: pd.DataFrame, id_col: str, group_col: str) -> None:
    """Validate required columns and coding of key variables."""
    missing = [c for c in [id_col, group_col, *EXPECTED_RATERS] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    df[id_col] = df[id_col].astype(str).str.strip()
    if df[id_col].duplicated().any():
        dups = df.loc[df[id_col].duplicated(), id_col].tolist()
        logger.warning("Duplicated student IDs found: %s", dups)

    groups = set(df[group_col].dropna().astype(str).str.lower().str.strip())
    unmatched = groups - EXPECTED_CONDITIONS
    if unmatched:
        logger.warning("Unexpected group labels (kept as-is): %s", unmatched)
    df[group_col] = df[group_col].astype(str).str.lower().str.strip()


def audit_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Range-check, coerce to numeric, and report missingness."""
    for rater in EXPECTED_RATERS:
        df[rater] = pd.to_numeric(df[rater], errors="coerce")
        n_bad = ((df[rater] < VALID_SCORE_MIN) | (df[rater] > VALID_SCORE_MAX)).sum()
        if n_bad:
            logger.warning("%s: %d out-of-range value(s) set to NaN", rater, n_bad)
            df.loc[(df[rater] < VALID_SCORE_MIN) | (df[rater] > VALID_SCORE_MAX), rater] = np.nan
        logger.info("%s: n=%d valid, %d missing", rater, df[rater].notna().sum(), df[rater].isna().sum())
    return df


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of available rater scores; requires at least one rater."""
    df["score_mean"] = df[EXPECTED_RATERS].mean(axis=1)
    df["score_range"] = df[EXPECTED_RATERS].max(axis=1) - df[EXPECTED_RATERS].min(axis=1)
    df = df.dropna(subset=["score_mean"]).reset_index(drop=True)
    logger.info("Rows retained after composite scoring: %d", len(df))
    return df


def to_long_format(df: pd.DataFrame, id_col: str, group_col: str) -> pd.DataFrame:
    """Reshape to long (one row per student x rater) for ICC/LMM workflows."""
    long = df.melt(
        id_vars=[id_col, group_col],
        value_vars=EXPECTED_RATERS,
        var_name="rater",
        value_name="score",
    ).dropna(subset=["score"]).reset_index(drop=True)
    logger.info("Long-format rows: %d", len(long))
    return long


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-process LICA-AI raw score data.")
    parser.add_argument("--input", required=True, help="Raw data file (.csv/.xlsx)")
    parser.add_argument("--output", default="data_clean.csv", help="Clean wide CSV output")
    parser.add_argument("--output-long", default="data_clean_long.csv", help="Clean long CSV output")
    parser.add_argument("--id-col", default="student_id", help="Student ID column name")
    parser.add_argument("--group-col", default="group", help="Condition/group column name")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    args = parser.parse_args()

    try:
        df = load_raw_data(Path(args.input), args.sheet)
        logger.info("Loaded %d rows, %d columns", *df.shape)
        df = standardize_columns(df)
        validate_structure(df, args.id_col.lower(), args.group_col.lower())
        df = audit_scores(df)
        df = compute_composite(df)

        out_wide, out_long = Path(args.output), Path(args.output_long)
        out_wide.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_wide, index=False)
        to_long_format(df, args.id_col.lower(), args.group_col.lower()).to_csv(out_long, index=False)
        logger.info("Saved: %s and %s", out_wide, out_long)
        return 0
    except Exception as exc:
        logger.error("Pre-processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

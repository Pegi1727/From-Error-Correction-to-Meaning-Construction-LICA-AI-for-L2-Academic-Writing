#!/usr/bin/env Rscript
# ======================================================================
# 01_descriptive_and_normality.R
# Project : From Error Correction to Meaning Construction:
#           Evaluating LICA-AI as a Real-Time Cognitive Scaffold
#           for L2 Academic Writing
# Purpose : Descriptive statistics by condition and tests of normality
#           (Shapiro-Wilk) and homogeneity of variance (Levene).
# Usage   : Rscript 01_descriptive_and_normality.R --input data_clean.csv
#           (or source() with INPUT_PATH set below)
# Author  : LICA-AI Research Team
# ======================================================================

suppressPackageStartupMessages({
  library(psych)
  library(car)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) args[i + 1] else default
}
INPUT_PATH  <- get_arg("--input", "data_clean.csv")
OUTPUT_PATH <- get_arg("--output", "descriptives_normality.txt")
GROUP_COL   <- get_arg("--group", "group")
SCORE_COL   <- get_arg("--score", "score_mean")

main <- function() {
  if (!file.exists(INPUT_PATH)) {
    stop("Input file not found: ", INPUT_PATH, call. = FALSE)
  }
  df <- tryCatch(
    read.csv(INPUT_PATH, stringsAsFactors = FALSE),
    error = function(e) stop("Failed to read input: ", e$message, call. = FALSE)
  )

  if (!GROUP_COL %in% names(df) || !SCORE_COL %in% names(df)) {
    stop("Required columns not found: ", GROUP_COL, ", ", SCORE_COL, call. = FALSE)
  }
  df$score <- suppressWarnings(as.numeric(df[[SCORE_COL]]))
  df$group <- trimws(tolower(as.character(df[[GROUP_COL]])))
  df <- df[!is.na(df$score), , drop = FALSE]
  if (nrow(df) == 0) stop("No valid numeric scores found.", call. = FALSE)

  sink(OUTPUT_PATH)
  on.exit(sink(), add = TRUE)

  cat("=================================================================\n")
  cat(" DESCRIPTIVE STATISTICS BY CONDITION\n")
  cat("=================================================================\n")
  desc <- describeBy(df$score, group = df$group, mat = TRUE, digits = 3)
  print(desc)

  cat("\n=================================================================\n")
  cat(" NORMALITY: Shapiro-Wilk (per group)\n")
  cat("=================================================================\n")
  for (g in sort(unique(df$group))) {
    x <- df$score[df$group == g]
    if (length(x) >= 3 && length(x) <= 5000) {
      sw <- shapiro.test(x)
      cat(sprintf("%-12s W = %.4f, p = %.4f, n = %d\n", g, sw$statistic, sw$p.value, length(x)))
    } else {
      cat(sprintf("%-12s n = %d (out of Shapiro-Wilk valid range)\n", g, length(x)))
    }
  }

  cat("\n=================================================================\n")
  cat(" HOMOGENEITY OF VARIANCE: Levene's test\n")
  cat("=================================================================\n")
  lev <- tryCatch(leveneTest(score ~ group, data = df),
                  error = function(e) { cat("Levene test failed:", e$message, "\n"); NULL })
  if (!is.null(lev)) print(lev)

  cat("\n=================================================================\n")
  cat(" Q-Q CORRELATIONS (robust normality index)\n")
  cat("=================================================================\n")
  for (g in sort(unique(df$group))) {
    x <- df$score[df$group == g]
    if (length(x) >= 5) {
      qq <- qqnorm(x, plot.it = FALSE)
      cat(sprintf("%-12s r(QQ) = %.4f\n", g, cor(qq$x, qq$y)))
    }
  }
  message("Saved: ", normalizePath(OUTPUT_PATH))
  invisible(TRUE)
}

main()

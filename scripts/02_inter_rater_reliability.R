#!/usr/bin/env Rscript
# ======================================================================
# 02_inter_rater_reliability.R
# Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
# Purpose : Inter-rater reliability via ICC(2,1) / ICC(3,1) with 95% CIs
#           (irr / psych), plus Pearson/Spearman correlation between raters.
# Usage   : Rscript 02_inter_rater_reliability.R --input data_clean.csv
# Author  : LICA-AI Research Team
# ======================================================================

suppressPackageStartupMessages({
  library(irr)
  library(psych)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) args[i + 1] else default
}
INPUT_PATH  <- get_arg("--input", "data_clean.csv")
OUTPUT_PATH <- get_arg("--output", "icc_results.txt")

main <- function() {
  if (!file.exists(INPUT_PATH)) stop("Input file not found: ", INPUT_PATH, call. = FALSE)
  df <- read.csv(INPUT_PATH, stringsAsFactors = FALSE)

  raters <- intersect(c("rater1", "rater2"), names(df))
  if (length(raters) < 2) stop("Need rater1 and rater2 columns.", call. = FALSE)

  ratings <- as.data.frame(lapply(df[raters], function(x) suppressWarnings(as.numeric(x))))
  complete <- complete.cases(ratings)
  ratings_c <- ratings[complete, ]
  if (nrow(ratings_c) < 3) stop("Fewer than 3 complete rater dyads.", call. = FALSE)

  sink(OUTPUT_PATH)
  on.exit(sink(), add = TRUE)

  cat("=================================================================\n")
  cat(" INTER-RATER RELIABILITY (n complete dyads = ", nrow(ratings_c), ")\n", sep = "")
  cat("=================================================================\n")

  icc2 <- tryCatch(icc(ratings_c, model = "twoway", type = "agreement", unit = "single"),
                   error = function(e) { cat("ICC(2,1) failed:", e$message, "\n"); NULL })
  if (!is.null(icc2)) {
    cat("ICC(2,1) two-way random, absolute agreement, single measures:\n")
    print(icc2)
  }

  icc3 <- tryCatch(icc(ratings_c, model = "twoway", type = "consistency", unit = "single"),
                   error = function(e) { cat("ICC(3,1) failed:", e$message, "\n"); NULL })
  if (!is.null(icc3)) {
    cat("\nICC(3,1) two-way mixed, consistency, single measures:\n")
    print(icc3)
  }

  icc_full <- tryCatch(psych::ICC(ratings_c), error = function(e) NULL)
  if (!is.null(icc_full)) {
    cat("\nFull psych::ICC boundary estimates:\n")
    print(icc_full$results[, c("type", "ICC", "L", "U")])
  }

  cat("\n=================================================================\n")
  cat(" RATER AGREEMENT CORRELATIONS\n")
  cat("=================================================================\n")
  pear <- cor.test(ratings_c[[1]], ratings_c[[2]], method = "pearson")
  spear <- cor.test(ratings_c[[1]], ratings_c[[2]], method = "spearman", exact = FALSE)
  cat(sprintf("Pearson  r = %.3f, 95%% CI [%.3f, %.3f], p = %.4g\n",
              pear$estimate, pear$conf.int[1], pear$conf.int[2], pear$p.value))
  cat(sprintf("Spearman rho = %.3f, p = %.4g\n", spear$estimate, spear$p.value))

  cat("\nInterpretation (Koo & Li, 2016): <0.50 poor; 0.50-0.75 moderate; ",
      "0.75-0.90 good; >0.90 excellent\n")
  message("Saved: ", normalizePath(OUTPUT_PATH))
  invisible(TRUE)
}

main()

#!/usr/bin/env Rscript
# ======================================================================
# 03_lmer_mixed_effects_analysis.R
# Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
# Purpose : Linear mixed-effects analysis (lme4/lmerTest) of writing
#           scores: random-intercepts model, condition fixed effect,
#           model comparison, and estimated marginal means (emmeans).
# Usage   : Rscript 03_lmer_mixed_effects_analysis.R --input data_clean_long.csv
# Author  : LICA-AI Research Team
# ======================================================================

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) args[i + 1] else default
}
INPUT_PATH  <- get_arg("--input", "data_clean_long.csv")
OUTPUT_PATH <- get_arg("--output", "lmer_results.txt")

main <- function() {
  if (!file.exists(INPUT_PATH)) stop("Input file not found: ", INPUT_PATH, call. = FALSE)
  df <- read.csv(INPUT_PATH, stringsAsFactors = FALSE)

  required <- c("student_id", "group", "rater", "score")
  miss <- setdiff(required, names(df))
  if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "), call. = FALSE)

  df$score <- suppressWarnings(as.numeric(df$score))
  df$group <- factor(trimws(tolower(df$group)))
  df$rater <- factor(trimws(tolower(df$rater)))
  df$student_id <- factor(df$student_id)
  df <- df[complete.cases(df[, c("score", "group", "rater", "student_id")]), ]
  if (nlevels(droplevels(df$group)) < 2) stop("Both conditions required.", call. = FALSE)

  sink(OUTPUT_PATH)
  on.exit(sink(), add = TRUE)

  cat("=================================================================\n")
  cat(" M0: UNCONDITIONAL MODEL  score ~ 1 + (1 | student_id)\n")
  cat("=================================================================\n")
  m0 <- lmer(score ~ 1 + (1 | student_id), data = df, REML = TRUE)
  print(summary(m0))
  vc <- as.data.frame(VarCorr(m0))
  cat("\nVariance components & ICC (students):\n")
  print(vc)
  v_stu <- vc$vcov[vc$grp == "student_id"]
  cat(sprintf("ICC(student) = %.3f\n", v_stu / sum(vc$vcov)))

  cat("\n=================================================================\n")
  cat(" M1: CONDITION MODEL  score ~ group + rater + (1 | student_id)\n")
  cat("=================================================================\n")
  m1 <- lmer(score ~ group + rater + (1 | student_id), data = df, REML = FALSE)
  print(summary(m1))

  cat("\nSatterthwaite p-values (lmerTest):\n")
  print(anova(m1))

  cat("\n=================================================================\n")
  cat(" MODEL COMPARISON\n")
  cat("=================================================================\n")
  print(AIC(m0, m1))
  print(BIC(m0, m1))

  cat("\n=================================================================\n")
  cat(" ESTIMATED MARGINAL MEANS\n")
  cat("=================================================================\n")
  emm <- emmeans(m1, ~ group)
  print(emm)
  print(pairs(emm, adjust = "holm"))
  print(confint(emm))

  cat("\nResidual diagnostics (scaled skew / spread by group):\n")
  print(tapply(residuals(m1), df$group, function(x) c(sd = sd(x), mean = mean(x))))
  message("Saved: ", normalizePath(OUTPUT_PATH))
  invisible(TRUE)
}

main()

#!/usr/bin/env Rscript
# ======================================================================
# 04_effect_sizes_and_contrasts.R
# Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
# Purpose : Effect sizes: Morris (2008) pre-post standardized mean
#           change (d_av, d_rm), Hedges' g between groups, Welch t-test
#           and ANCOVA-style contrast on post scores.
# Usage   : Rscript 04_effect_sizes_and_contrasts.R --input data_clean.csv
# Author  : LICA-AI Research Team
# ======================================================================

suppressPackageStartupMessages({
  library(effectsize)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) args[i + 1] else default
}
INPUT_PATH  <- get_arg("--input", "data_clean.csv")
OUTPUT_PATH <- get_arg("--output", "effect_sizes_R.txt")
PRE_COL     <- get_arg("--pre", "pre_score")
POST_COL    <- get_arg("--post", "post_score")
GROUP_COL   <- get_arg("--group", "group")

morris_d <- function(pre, post, variant = c("av", "rm")) {
  variant <- match.arg(variant)
  ok <- complete.cases(pre, post)
  pre <- pre[ok]; post <- post[ok]
  n <- length(pre)
  if (n < 3) stop("Need >= 3 complete pre/post pairs.", call. = FALSE)
  change <- mean(post) - mean(pre)
  s_pre <- sd(pre); s_post <- sd(post)
  r <- suppressWarnings(cor(pre, post))
  if (!is.finite(r)) r <- 0
  d <- switch(variant,
    "av" = change / sqrt((s_pre^2 + s_post^2) / 2),
    "rm" = change / (s_post * sqrt(2 * (1 - r)))
  )
  correction <- 1 - 3 / (4 * n - 5)   # Morris small-sample correction
  d * correction
}

main <- function() {
  if (!file.exists(INPUT_PATH)) stop("Input file not found: ", INPUT_PATH, call. = FALSE)
  df <- read.csv(INPUT_PATH, stringsAsFactors = FALSE)

  sink(OUTPUT_PATH)
  on.exit(sink(), add = TRUE)

  cat("=================================================================\n")
  cat(" MORRIS (2008) PRE-POST STANDARDIZED MEAN CHANGE\n")
  cat("=================================================================\n")
  if (all(c(PRE_COL, POST_COL, GROUP_COL) %in% names(df))) {
    for (g in sort(unique(df[[GROUP_COL]]))) {
      sub <- df[df[[GROUP_COL]] == g, ]
      pre <- suppressWarnings(as.numeric(sub[[PRE_COL]]))
      post <- suppressWarnings(as.numeric(sub[[POST_COL]]))
      if (sum(complete.cases(pre, post)) < 3) next
      d_av <- morris_d(pre, post, "av")
      d_rm <- morris_d(pre, post, "rm")
      cat(sprintf("%-12s d_av = %.3f | d_rm = %.3f | n = %d\n",
                  g, d_av, d_rm, sum(complete.cases(pre, post))))
    }
  } else {
    cat("Pre/post columns not found; skipping Morris d.\n")
  }

  cat("\n=================================================================\n")
  cat(" BETWEEN-GROUP CONTRAST ON POST SCORES\n")
  cat("=================================================================\n")
  if (POST_COL %in% names(df) && GROUP_COL %in% names(df)) {
    df$post <- suppressWarnings(as.numeric(df[[POST_COL]]))
    df$grp <- factor(trimws(tolower(as.character(df[[GROUP_COL]]))))
    ok <- complete.cases(df$post, df$grp)
    if (nlevels(droplevels(df$grp[ok])) == 2) {
      tt <- t.test(post ~ grp, data = df[ok, ], var.equal = FALSE)
      print(tt)
      hg <- tryCatch(hedges_g(post ~ grp, data = df[ok, ]),
                     error = function(e) { cat("effectsize failed:", e$message, "\n"); NULL })
      if (!is.null(hg)) print(hg)
    } else {
      cat("Two groups required for between-group contrast.\n")
    }
  }

  cat("\n=================================================================\n")
  cat(" T-TEST AND COHEN'S D ON COMPOSITE SCORE\n")
  cat("=================================================================\n")
  if ("score_mean" %in% names(df)) {
    df$score <- suppressWarnings(as.numeric(df$score_mean))
    df$grp <- factor(trimws(tolower(as.character(df[[GROUP_COL]]))))
    ok <- complete.cases(df$score, df$grp)
    if (nlevels(droplevels(df$grp[ok])) == 2) {
      tt <- t.test(score ~ grp, data = df[ok, ], var.equal = FALSE)
      print(tt)
      print(tryCatch(cohens_d(score ~ grp, data = df[ok, ], pooled_sd = TRUE),
                     error = function(e) NULL))
    }
  }
  message("Saved: ", normalizePath(OUTPUT_PATH))
  invisible(TRUE)
}

main()

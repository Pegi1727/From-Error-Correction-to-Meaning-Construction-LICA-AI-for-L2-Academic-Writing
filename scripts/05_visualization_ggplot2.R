#!/usr/bin/env Rscript
# ======================================================================
# 05_visualization_ggplot2.R
# Project : LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing
# Purpose : Publication-quality ggplot2 figures (300 dpi):
#           1) Group means with 95% CI
#           2) Violin + boxplot distributions
#           3) Pre-post mean change
# Usage   : Rscript 05_visualization_ggplot2.R --input data_clean.csv --outdir figures
# Author  : LICA-AI Research Team
# ======================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (!is.na(i) && length(args) >= i + 1) args[i + 1] else default
}
INPUT_PATH <- get_arg("--input", "data_clean.csv")
OUT_DIR    <- get_arg("--outdir", "figures")
GROUP_COL  <- get_arg("--group", "group")
SCORE_COL  <- get_arg("--score", "score_mean")

theme_apa <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8)),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

main <- function() {
  if (!file.exists(INPUT_PATH)) stop("Input file not found: ", INPUT_PATH, call. = FALSE)
  df <- read.csv(INPUT_PATH, stringsAsFactors = FALSE)
  if (!GROUP_COL %in% names(df) || !SCORE_COL %in% names(df)) {
    stop("Required columns missing: ", GROUP_COL, ", ", SCORE_COL, call. = FALSE)
  }
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

  df$.grp <- factor(trimws(tolower(as.character(df[[GROUP_COL]]))))
  df$.score <- suppressWarnings(as.numeric(df[[SCORE_COL]]))
  df <- df[complete.cases(df$.score, df$.grp), ]

  palette <- c(control = "#4C72B0", lica_ai = "#DD8452")

  # ---- Figure 1: means with 95% CI ------------------------------------
  summ <- df %>%
    group_by(.grp) %>%
    summarise(mean = mean(.score), se = sd(.score) / sqrt(n()), n = n(), .groups = "drop") %>%
    mutate(lo = mean - 1.96 * se, hi = mean + 1.96 * se)
  p1 <- ggplot(summ, aes(x = .grp, y = mean, colour = .grp)) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.12, linewidth = 0.7) +
    geom_point(size = 3.2) +
    geom_hline(yintercept = mean(df$.score), linetype = "dashed", alpha = 0.4) +
    scale_colour_manual(values = palette) +
    labs(title = "Group Means with 95% CI", x = NULL, y = "Writing Score") +
    theme_apa
  ggsave(file.path(OUT_DIR, "fig1_group_means.png"), p1, width = 5.5, height = 4.2, dpi = 300)

  # ---- Figure 2: distributions ----------------------------------------
  p2 <- ggplot(df, aes(x = .grp, y = .score, fill = .grp)) +
    geom_violin(alpha = 0.55, colour = NA, trim = TRUE) +
    geom_boxplot(width = 0.22, fill = "white", outlier.shape = NA) +
    geom_jitter(width = 0.06, size = 1.2, alpha = 0.55, colour = "grey25") +
    scale_fill_manual(values = palette) +
    labs(title = "Score Distributions by Condition", x = NULL, y = "Writing Score") +
    theme_apa
  ggsave(file.path(OUT_DIR, "fig2_distributions.png"), p2, width = 6, height = 4.5, dpi = 300)

  # ---- Figure 3: pre-post change --------------------------------------
  if (all(c("pre_score", "post_score") %in% names(df))) {
    df$pre <- suppressWarnings(as.numeric(df$pre_score))
    df$post <- suppressWarnings(as.numeric(df$post_score))
    chg <- df %>%
      filter(complete.cases(pre, post)) %>%
      group_by(.grp) %>%
      summarise(pre = mean(pre), post = mean(post), .groups = "drop") %>%
      tidyr::pivot_longer(cols = c(pre, post), names_to = "time", values_to = "score") %>%
      mutate(time = factor(time, levels = c("pre", "post"), labels = c("Pre-test", "Post-test")))
    p3 <- ggplot(chg, aes(x = time, y = score, group = .grp, colour = .grp)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 3) +
      scale_colour_manual(values = palette, name = "Condition") +
      labs(title = "Mean Change from Pre-test to Post-test", x = NULL, y = "Writing Score") +
      theme_apa + theme(legend.position = "bottom")
    ggsave(file.path(OUT_DIR, "fig3_pre_post_change.png"), p3, width = 5.5, height = 4.5, dpi = 300)
  } else {
    message("pre_score/post_score not found; skipping Figure 3.")
  }

  message("Figures saved to: ", normalizePath(OUT_DIR))
  invisible(TRUE)
}

main()

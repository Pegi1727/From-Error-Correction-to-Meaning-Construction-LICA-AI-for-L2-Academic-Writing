# LICA-AI Analysis Scripts

**Project:** *From Error Correction to Meaning Construction: Evaluating LICA-AI as a Real-Time Cognitive Scaffold for L2 Academic Writing*

This package contains the complete, reproducible analysis pipeline for the study, implemented in parallel in **Python** and **R**.

## Directory Structure

```
lica_ai_scripts/
├── README.md
├── requirements.txt
├── scripts/
│   ├── python/
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_inter_rater_reliability_icc.py
│   │   ├── 03_linear_mixed_effects_models.py
│   │   ├── 04_effect_sizes_morris_d.py
│   │   └── 05_publication_figures.py
│   └── r/
│       ├── 01_descriptive_and_normality.R
│       ├── 02_inter_rater_reliability.R
│       ├── 03_lmer_mixed_effects_analysis.R
│       ├── 04_effect_sizes_and_contrasts.R
│       └── 05_visualization_ggplot2.R
```

## Requirements

### Python (3.9+)
```bash
pip install -r requirements.txt
```

### R (4.2+)
```r
install.packages(c("psych", "car", "irr", "lme4", "lmerTest",
                   "emmeans", "effectsize", "ggplot2", "dplyr", "tidyr"))
```

## Pipeline

### Python track
| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_data_preprocessing.py` | Clean/validate raw CSV/Excel scores; composite & long-format outputs |
| 2 | `02_inter_rater_reliability_icc.py` | ICC(2,1) & ICC(3,1) with 95% CIs (pingouin or manual Shrout–Fleiss) |
| 3 | `03_linear_mixed_effects_models.py` | LMMs (statsmodels): null vs condition model, AIC/BIC, EMMs |
| 4 | `04_effect_sizes_morris_d.py` | Morris (2008) d_av/d_rm with bootstrap CIs; Hedges' g between groups |
| 5 | `05_publication_figures.py` | 300-dpi APA-style figures: means+CI, violin/box, pre–post change |

### R track
| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_descriptive_and_normality.R` | Descriptives (psych), Shapiro–Wilk, Levene's test, QQ correlations |
| 2 | `02_inter_rater_reliability.R` | ICC(2,1)/ICC(3,1) via `irr`/`psych`; Pearson/Spearman agreement |
| 3 | `03_lmer_mixed_effects_analysis.R` | `lmerTest` models, variance components/ICC, `emmeans` contrasts |
| 4 | `04_effect_sizes_and_contrasts.R` | Morris d_av/d_rm, Hedges' g, Welch t-tests, Cohen's d |
| 5 | `05_visualization_ggplot2.R` | ggplot2 300-dpi publication figures with an APA theme |

## Example Usage

```bash
# Python pipeline
python scripts/python/01_data_preprocessing.py --input raw_scores.xlsx --output data_clean.csv --output-long data_clean_long.csv
python scripts/python/02_inter_rater_reliability_icc.py --input data_clean.csv
python scripts/python/03_linear_mixed_effects_models.py --input data_clean_long.csv
python scripts/python/04_effect_sizes_morris_d.py --input data_clean.csv
python scripts/python/05_publication_figures.py --input data_clean.csv --outdir figures

# R pipeline
Rscript scripts/r/01_descriptive_and_normality.R --input data_clean.csv
Rscript scripts/r/02_inter_rater_reliability.R --input data_clean.csv
Rscript scripts/r/03_lmer_mixed_effects_analysis.R --input data_clean_long.csv
Rscript scripts/r/04_effect_sizes_and_contrasts.R --input data_clean.csv
Rscript scripts/r/05_visualization_ggplot2.R --input data_clean.csv --outdir figures
```

## Data Schema

**Wide format** (`data_clean.csv`): `student_id, group, rater1, rater2, pre_score, post_score, score_mean, score_range`

**Long format** (`data_clean_long.csv`): `student_id, group, rater, score`

`group` labels: `control` | `lica_ai`. All scripts perform column/label validation, robust error handling with informative messages, and non-zero exit codes on failure.

## Notes

- ICC interpretation follows Koo & Li (2016): <0.50 poor; 0.50–0.75 moderate; 0.75–0.90 good; >0.90 excellent.
- Morris d uses the small-sample correction 1 − 3/(4n − 5); bootstrap CIs use 5,000 resamples (seed = 2024).
- All figures are exported at 300 dpi suitable for journal submission.

**Author:** LICA-AI Research Team

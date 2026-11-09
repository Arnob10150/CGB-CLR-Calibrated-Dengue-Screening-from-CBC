# CGB-CLR: Calibrated Dengue Screening from CBC

Audit-aware, calibrated LightGBM–XGBoost–CatBoost ensemble with conformal prediction for externally-validated dengue screening from complete blood counts (ICRPSET 2026).

## Overview

CGB-CLR represents the leukocyte differential as compositional data via a centred-log-ratio (CLR) transform, augments it with clinically motivated hematological ratios, and couples a calibrated gradient-boosting ensemble with conformal selective prediction. The framework is evaluated on a Jamalpur (Bangladesh) cohort under repeated 10×5 cross-validation with Nadeau–Bengio corrected significance testing, and externally validated on an independent Kalai/Joypurhat cohort.

## Repository structure

```
dengue-cgb-clr/
├── paper/            # Camera-ready manuscript (LaTeX source, compiled PDF, figures)
├── code/              # Analysis and modeling scripts
├── data/              # Jamalpur (primary) and Kalai (external validation) cohorts
├── results/           # Saved model outputs, metrics, and significance-test results (JSON)
└── README.md
```

### `code/`

- `dengue_pipeline.py`, `dengue_framework_final.py`, `dengue_framework_v2.py`, `dengue_boost.py`, `dengue_advanced.py`, `dengue_sweep.py` — model development and ablation scripts
- `dengue_results_final.py`, `dengue_results_final_v2.py` — definitive results pipeline (5×5 and 10×5 repeated cross-validation respectively)
- `evaluate_baselines.py` — baseline model comparison
- `dengue_external.py` — external validation on the Kalai/Joypurhat cohort
- `make_figures.py`, `make_figs_rev.py`, `make_arch.py` — figure generation (re-runs model fitting/evaluation to produce plots)
- `scratch_nbcorrection.py`, `scratch_morerepeats.py` — Nadeau–Bengio corrected significance testing at 5×5 and 10×5 fold counts
- `scratch_thresholds.py` — operating-point / threshold trade-off analysis
- `scratch_featureshift.py` — feature-matched internal vs. external AUC comparison (dataset-shift analysis)

### `data/`

- `jamalpur/` — primary cohort: Dengue Fever Hematological Dataset (Mendeley Data)
- `kalai_predictive_clinical_dataset.zip` — external validation cohort (Predictive Clinical Dataset for Dengue Fever, Mendeley Data)

### `results/`

Saved JSON outputs from the scripts above: cross-validation scores, hold-out metrics, significance tests (naive and Nadeau–Bengio corrected), external validation results, and threshold/operating-point tables.

### `paper/`

Camera-ready manuscript: `dengue_screening_paper.tex`, compiled `dengue_screening_paper.pdf`, and `figs/` (all figures referenced by the manuscript).

## Reproducing the results

Requires Python 3 with `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`.

```bash
cd code
python dengue_results_final_v2.py   # primary 10x5 CV results + significance tests
python dengue_external.py           # external validation
python scratch_nbcorrection.py      # Nadeau-Bengio corrected significance (5x5)
python scratch_morerepeats.py       # Nadeau-Bengio corrected significance (10x5)
python scratch_thresholds.py        # operating-point / threshold analysis
python scratch_featureshift.py      # feature-matched internal vs. external AUC
```

## Citation

Dataset sources:
- Nirob, M. A. S. et al. "Dengue fever hematological dataset: Clinical insights for improved diagnosis and patient management." *Mendeley Data*, ver. 2, 2025. doi: 10.17632/6fsrsk3mb8.2
- Mim, M. M. H. et al. "Predictive clinical dataset for dengue fever using vital signs and blood parameters." *Mendeley Data*, ver. 1, 2024. doi: 10.17632/xrsbyjs24t.1

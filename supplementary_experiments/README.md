# Supplementary Experiments for TabPFN-QSAR

This directory contains all supplementary experiment scripts conducted in response to reviewer comments during the peer-review process. Each subfolder corresponds to a specific reviewer concern and contains the relevant code.

**Note:** Datasets and pre-computed results are excluded from this repository. To reproduce the experiments, you will need the original datasets (k-OH QSAR, Tox21 SR-MMP, etc.) placed in the appropriate locations as referenced by each script.

## Overview

| # | Experiment | Reviewer Concern Addressed |
|---|-----------|--------------------------|
| 01 | Non-parametric Statistical Tests | Lack of formal hypothesis testing (Reviewer 4) |
| 02 | Descriptor Space Comparison | MACCS-only bias (Reviewers 2, 3, 4) |
| 03 | Dynamic Curve Stability | Breakpoint N* robustness (Reviewer 3) |
| 04 | SHAP Explanation Stability | Interpretability robustness (Reviewer 2) |
| 05 | Imbalanced Classification (F1) | TabPFGen underperformance analysis (Reviewer 4) |

## Requirements

- Python >= 3.9
- tabpfn >= 0.2.0
- tabpfn-extensions (for AutoTabPFNRegressor)
- scikit-learn, xgboost, catboost, scipy, pandas, numpy
- hyperopt (for hyperparameter tuning)
- rdkit (for descriptor generation)
- imbalanced-learn (for SMOTE)
- tabpfgen (for TabPFGen augmentation)

## Citation

If you use these supplementary experiments in your work, please cite the main TabPFN-QSAR paper.

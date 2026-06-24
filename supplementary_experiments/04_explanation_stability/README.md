# 04 - SHAP Explanation Stability Analysis

**Reviewer Concern:** "The interpretability stability analysis would benefit from one simple quantitative measure, such as SHAP top-k overlap, Spearman rank correlation of feature importance, or bootstrap confidence intervals under different missingness levels." (Reviewer 2)

## What Was Done

We implemented a dual-perturbation framework to quantitatively assess SHAP explanation stability under two realistic QSAR data challenges:

### 1. Bootstrap Resampling (Sampling Variability)
- 30 bootstrap iterations yielding 435 pairwise comparisons
- Metrics: Spearman rank correlation (global) and weighted Kendall tau (top-rank focused)

### 2. Missing-Value Robustness (MCAR)
- Artificially introduced missing values at 5%, 10%, and 15% rates
- No imputation performed (tests TabPFN inherent missing-value handling)
- Metrics: Spearman rho, weighted Kendall tau, Top-5 overlap

## Files

| File | Description |
|------|-------------|
| run_stability.py | Top-level entry point; supports bootstrap, missing, and both modes |
| explanation_stability/__init__.py | Package init |
| explanation_stability/bootstrap_analysis.py | Core analysis: bootstrap resampling, SHAP computation, stability metrics |
| explanation_stability/metrics.py | Stability metrics: Spearman, weighted Kendall tau, Top-K overlap |
| shap_missing_value(0).ipynb | Jupyter notebook for interactive missing-value SHAP analysis |

## Usage

```bash
# Bootstrap analysis only (default)
python run_stability.py

# Missing-rate perturbation only
python run_stability.py missing

# Both analyses
python run_stability.py both
```

Requires `categorized_maccs.xlsx` (the primary k-OH QSAR dataset) in the same directory.

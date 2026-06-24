# 03 - Dynamic Curve Stability Analysis

**Reviewer Concern:** "The robustness and generalizability of the estimated breakpoint N* remain unclear [...] noisy early-stage points may substantially affect the fitted parameters and shift the estimated N*." (Reviewer 3)

## What Was Done

We quantified the stability of the dynamic efficient-window breakpoints (N_start, N_end) by running the full piecewise-regression pipeline across 31 different random initial subsets (seeds 40-70), while keeping the train/test split fixed.

The pipeline consists of:
1. Select 200 initial samples using a given seed
2. Evaluate TabPFN RMSE at 11 subset sizes (100-200)
3. Fit an exponential or power-law decay curve to observed points
4. Extrapolate to N=1000
5. Apply three-segment piecewise linear regression to identify breakpoints (bp1=N_start, bp2=N_end)

## Files

| File | Description |
|------|-------------|
| stability_analysis.py | Main stability analysis: exponential/power-law fitting + piecewise regression across 31 seeds |
| protocal.py | Protocol/configuration for the DUC pipeline |

## Usage

```bash
python stability_analysis.py
```

Requires `categorized_maccs.xlsx` (the primary k-OH QSAR dataset) in the same directory.

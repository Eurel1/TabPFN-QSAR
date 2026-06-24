# 02 - Descriptor Space Comparison

**Reviewer Concerns:** "TabPFN performance appears to be highly representation-dependent [...] Could the authors show whether the observed advantages persist under alternative descriptor spaces?" (Reviewer 3); "The descriptor-space sensitivity analysis remains limited." (Reviewer 2); "By restricting all models to 166-bit MACCS keys [...] the study creates a structural asymmetry." (Reviewer 4)

## What Was Done

We compared model performance across four descriptor representations on the k-OH dataset (1089 compounds, 31 seeds):

| Descriptor | Dim | Generator Script | Benchmark Script |
|-----------|-----|-----------------|-----------------|
| MACCS Keys | 166 | - | model_compare_maccs.py |
| Morgan FP (1024-bit) | 1024 | generate_features.py | model_compare_morgan_1024.py |
| Morgan FP (2048-bit) | 2048 | generate_features.py | model_compare_morgan_2048.py |
| Morgan PCA-reduced (64/128/256) | 64-256 | generate_features_dimreduced.py | model_compare_morgan_dimreduced.py |
| RDKit PhysChem | ~200 | generate_features.py | model_compare_rdkit_physchem.py |

All baselines (XGBoost, CatBoost, SVR, MLP, Ridge) underwent hyperparameter tuning via Hyperopt (30 evaluations each). Conventional models were allowed to use their full native dimensionality; only TabPFN was constrained to 500 or fewer features.

## Files

| File | Description |
|------|-------------|
| generate_features.py | Generate Morgan 1024/2048 fingerprints and RDKit descriptors from SMILES |
| generate_features_dimreduced.py | Generate PCA-reduced Morgan fingerprints (64, 128, 256 dims) |
| model_compare_maccs.py | Baseline benchmark with MACCS keys (TabPFN + 5 baselines) |
| model_compare_morgan_1024.py | Benchmark with 1024-bit Morgan fingerprints |
| model_compare_morgan_2048.py | Benchmark with 2048-bit Morgan fingerprints |
| model_compare_morgan_dimreduced.py | Benchmark with PCA-reduced Morgan fingerprints |
| model_compare_rdkit_physchem.py | Benchmark with RDKit physicochemical descriptors |
| tabpfn_supplement_morgan.py | Supplementary TabPFN-specific analysis on Morgan features |

## Usage

```bash
# Step 1: Generate features (requires RDKit and original SMILES data)
python generate_features.py

# Step 2: Run benchmarks (each script runs 31 seeds x 5-6 models)
python model_compare_rdkit_physchem.py
```

# 05 - Imbalanced Classification: F1 Score Comparison

**Reviewer Concern:** "TabPFGen combined with TabPFN performs worse than TabPFN alone [...] In imbalanced classification tasks, TabPFGen F1 score is numerically inferior to established methods like SMOTE and GMM. The attribution of this underperformance to threshold sensitivity lacks robust calibration analysis." (Reviewer 4)

## What Was Done

We conducted a systematic comparison of three augmentation methods on the Tox21 SR-MMP imbalanced classification task (224 samples, 31 seeds):

| Method | Script | Approach |
|--------|--------|----------|
| TabPFGen | TabPFGen_aug_scaler.py | SGLD-based synthetic sample generation, target 500/class |
| SMOTE | smote_aug_scaler.py | SMOTE oversampling (k=5), target 500/class |
| GMM | gmm_aug_scaler.py | Gaussian Mixture Model sampling, target 500/class |

All methods include:
- StandardScaler normalization before augmentation
- 5-fold stratified CV for threshold calibration (sweep 0.01-0.99)
- Multi-seed evaluation (seeds 40-70) with 50/50 stratified train/test splits
- Metrics: Accuracy, Precision, Recall, F1, Best_F1 (threshold-tuned), ROC-AUC, PR-AUC

## Files

| File | Description |
|------|-------------|
| TabPFGen_aug_scaler.py | TabPFGen augmentation + threshold-tuned benchmark |
| smote_aug_scaler.py | SMOTE augmentation + threshold-tuned benchmark |
| gmm_aug_scaler.py | GMM augmentation + threshold-tuned benchmark |

## Usage

```bash
# Run each augmentation method independently
python TabPFGen_aug_scaler.py
python smote_aug_scaler.py
python gmm_aug_scaler.py
```

Requires `processed_tox21_data.xlsx` (Tox21 SR-MMP dataset with physicochemical descriptors) in the same directory.

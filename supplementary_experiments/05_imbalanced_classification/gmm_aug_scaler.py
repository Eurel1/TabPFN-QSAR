import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

# ==============================
# Helper: Best-threshold search (sweep 0.01 .. 0.99)
# ==============================
def find_best_threshold(y_true, y_prob):
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ==============================
# Helper: Fresh model factory
# ==============================
def create_model(name):
    if name == 'XGBoost':
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        )
    elif name == 'CatBoost':
        return CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=42, verbose=False
        )
    elif name == 'MLP':
        return MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=42, early_stopping=True
        )
    elif name == 'SVM':
        return SVC(
            kernel='rbf', C=1.0,
            probability=True, random_state=42
        )
    elif name == 'TabPFN':
        return TabPFNClassifier(device='cuda')
    else:
        raise ValueError(f"Unknown model: {name}")


# ==============================
# 1. Data loading
# ==============================
data = pd.read_excel('processed_tox21_data.xlsx')
X = data.drop('SR-MMP', axis=1).values
y = data['SR-MMP'].values


# ==============================
# 2. GMM augmentation
# ==============================
def gmm_resample(X, y, target_count=500, n_components=5, random_state=42):
    X_resampled, y_resampled = X.copy(), y.copy()
    unique_classes = np.unique(y)

    for cls in unique_classes:
        curr_X = X[y == cls]
        num_to_add = target_count - len(curr_X)

        if num_to_add > 0:
            gmm = GaussianMixture(
                n_components=min(n_components, len(curr_X)),
                covariance_type='full',
                random_state=random_state
            )
            gmm.fit(curr_X)
            X_new, _ = gmm.sample(num_to_add)

            X_resampled = np.vstack([X_resampled, X_new])
            y_resampled = np.concatenate([y_resampled, [cls] * num_to_add])

    return X_resampled, y_resampled


# ==============================
# 3. Model names
# ==============================
model_names = ['XGBoost', 'CatBoost', 'MLP', 'SVM', 'TabPFN']


# ==============================
# 4. Multi-seed benchmark with CV threshold tuning
# ==============================
seeds = range(40, 71)
n_cv_folds = 5
detailed_results = []

print(f"GMM + StandardScaler + {n_cv_folds}-Fold CV Threshold | seeds={len(seeds)}")

for seed in seeds:
    print(f"Seed {seed}...", end=' ')

    # 4.1 50/50 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # 4.2 StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4.3 GMM augmentation
    X_train_aug, y_train_aug = gmm_resample(
        X_train_scaled, y_train,
        target_count=500, random_state=seed
    )

    # 4.4 K-Fold CV for threshold search
    kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    for name in model_names:
        # ---- CV phase: collect best thresholds across folds ----
        cv_thresholds = []

        for train_idx, val_idx in kf.split(X_train_aug, y_train_aug):
            X_cv_train, X_cv_val = X_train_aug[train_idx], X_train_aug[val_idx]
            y_cv_train, y_cv_val = y_train_aug[train_idx], y_train_aug[val_idx]

            model = create_model(name)
            model.fit(X_cv_train, y_cv_train)
            y_cv_val_prob = model.predict_proba(X_cv_val)[:, 1]
            best_t, _ = find_best_threshold(y_cv_val, y_cv_val_prob)
            cv_thresholds.append(best_t)

        best_threshold = float(np.mean(cv_thresholds))
        cv_threshold_std = float(np.std(cv_thresholds, ddof=1))

        # ---- Retrain on full augmented set ----
        model = create_model(name)
        model.fit(X_train_aug, y_train_aug)

        # ---- Evaluate on test ----
        y_test_pred_default = model.predict(X_test_scaled)
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_test_pred_best = (y_test_prob >= best_threshold).astype(int)

        detailed_results.append({
            "Seed": seed,
            "Model": name,
            "Best_Threshold": round(best_threshold, 4),
            "CV_Threshold_Std": round(cv_threshold_std, 4),
            "Accuracy": accuracy_score(y_test, y_test_pred_default),
            "Precision": precision_score(y_test, y_test_pred_default, zero_division=0),
            "Recall": recall_score(y_test, y_test_pred_default, zero_division=0),
            "F1": f1_score(y_test, y_test_pred_default, zero_division=0),
            "Best_F1": f1_score(y_test, y_test_pred_best, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_test_prob),
            "PR-AUC": average_precision_score(y_test, y_test_prob),
        })

    print("done")


# ==============================
# 5. Build DataFrames
# ==============================
df_details = pd.DataFrame(detailed_results)


# ==============================
# 6. 95% CI
# ==============================
def get_ci95(data):
    n = len(data)
    if n < 2:
        return 0
    se = stats.sem(data)
    return se * stats.t.ppf((1 + 0.95) / 2., n - 1)


report_metrics = ["Accuracy", "Precision", "Recall", "F1", "Best_F1", "ROC-AUC", "PR-AUC"]

summary_table = []

for model_name in model_names:
    model_data = df_details[df_details['Model'] == model_name]

    res_row = {"Model": model_name}

    for m in report_metrics:
        mean_val = model_data[m].mean()
        ci_val = get_ci95(model_data[m])
        res_row[m] = f"{mean_val:.3f} +/- {ci_val:.3f}"

    # Mean best threshold
    mean_thresh = model_data["Best_Threshold"].mean()
    res_row["Best_Threshold (Mean)"] = f"{mean_thresh:.3f}"

    summary_table.append(res_row)

df_summary = pd.DataFrame(summary_table)


# ==============================
# 7. Output
# ==============================
print("\n" + "=" * 65)
print(f"GMM + StandardScaler + {n_cv_folds}-Fold CV Threshold (Mean +/- 95% CI)")
print("=" * 65)
print(df_summary.to_string(index=False))

output_file = "GMM_Scaled_with_TabPFN_ThresholdTuned.xlsx"

with pd.ExcelWriter(output_file) as writer:
    df_summary.to_excel(writer, sheet_name="Summary_95CI", index=False)
    df_details.to_excel(writer, sheet_name="Detailed_Seeds", index=False)

print(f"\nSaved to: {output_file}")

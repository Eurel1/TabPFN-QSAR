# ==============================
# GMM + StandardScaler + TabPFN Benchmark
# ==============================

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier  # ✅ 新增

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

# ==============================
# 1. 数据读取
# ==============================
data = pd.read_excel('processed_tox21_data.xlsx')
X = data.drop('SR-MMP', axis=1).values
y = data['SR-MMP'].values


# ==============================
# 2. GMM 增强函数
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
# 3. 初始化模型
# ==============================
models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        random_state=42, verbose=False
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=500,
        random_state=42, early_stopping=True
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0,
        probability=True, random_state=42
    ),

    # ✅ 新增 TabPFN
    'TabPFN': TabPFNClassifier(device='cuda')  # GPU改为 'cuda'
}


# ==============================
# 4. 多随机种子实验
# ==============================
seeds = range(40, 71)
detailed_results = []

print(f"开始实验 (GMM + 归一化 + TabPFN)，总计种子数: {len(seeds)}")

for seed in seeds:
    print(f"Running Seed: {seed}...", end='\r')

    # 1. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # 2. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. GMM 增强（在归一化空间）
    X_train_res, y_train_res = gmm_resample(
        X_train_scaled, y_train,
        target_count=500,
        random_state=seed
    )

    # 4. 训练与评估
    for name, model in models.items():

        # ⚠️ TabPFN 必须每次重新实例化（避免跨 seed 污染）
        if name == 'TabPFN':
            model = TabPFNClassifier(device='cuda')

        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        detailed_results.append({
            "Seed": seed,
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob)
        })


# ==============================
# 5. 转 DataFrame
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


metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

summary_table = []

for model_name in models.keys():
    model_data = df_details[df_details['Model'] == model_name]

    res_row = {"Model": model_name}

    for m in metrics:
        mean_val = model_data[m].mean()
        ci_val = get_ci95(model_data[m])
        res_row[m] = f"{mean_val:.3f} ± {ci_val:.3f}"

    summary_table.append(res_row)

df_summary = pd.DataFrame(summary_table)


# ==============================
# 7. 输出
# ==============================
print("\n" + "=" * 50)
print("GMM + StandardScaler + TabPFN 实验汇总 (Mean ± 95% CI)")
print("=" * 50)
print(df_summary.to_string(index=False))

output_file = "GMM_Scaled_with_TabPFN.xlsx"

with pd.ExcelWriter(output_file) as writer:
    df_summary.to_excel(writer, sheet_name="Summary_95CI", index=False)
    df_details.to_excel(writer, sheet_name="Detailed_Seeds", index=False)

print(f"\n所有数据已保存至: {output_file}")
# ==============================
# No Augmentation + StandardScaler + TabPFN Benchmark
# ==============================

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# 1. 读取数据
# ==============================
data = pd.read_excel('processed_tox21_data.xlsx')
X = data.drop('SR-MMP', axis=1).values
y = data['SR-MMP'].values

# ==============================
# 2. 初始化模型
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
    'TabPFN': TabPFNClassifier(device='cuda')  # GPU 可改 'cuda'
}

# ==============================
# 3. 多随机种子实验
# ==============================
seeds = range(40, 71)
detailed_records = []

print(f"开始多随机种子实验 (无数据增强 + 归一化 + TabPFN)，共 {len(seeds)} 个种子...")

for seed in seeds:
    print(f"正在处理随机种子: {seed}...", end='\r')

    # A. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,
        random_state=seed,
        stratify=y
    )

    # B. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # C. 模型训练与评估
    for name, model in models.items():

        # ⚠️ TabPFN 每个 seed 重新实例化（非常关键）
        if name == 'TabPFN':
            model = TabPFNClassifier(device='cuda')

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        detailed_records.append({
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
# 转 DataFrame
# ==============================
df_detailed = pd.DataFrame(detailed_records)

# ==============================
# 4. 95% CI
# ==============================
def get_ci95(data):
    n = len(data)
    if n < 2:
        return 0
    se = stats.sem(data)
    return se * stats.t.ppf((1 + 0.95) / 2., n - 1)


metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

summary_list = []

for model_name in models.keys():
    m_data = df_detailed[df_detailed['Model'] == model_name]

    res_row = {"Model": model_name}

    for m in metrics:
        mean_val = m_data[m].mean()
        ci_val = get_ci95(m_data[m])
        res_row[m] = f"{mean_val:.3f} ± {ci_val:.3f}"

    summary_list.append(res_row)

df_summary = pd.DataFrame(summary_list)

# ==============================
# 5. 输出
# ==============================
print("\n" + "=" * 60)
print("No Augmentation + StandardScaler 性能汇总 (Mean ± 95% CI)")
print("=" * 60)
print(df_summary.to_string(index=False))

output_file = "benchmark_results_no_augmentation_with_TabPFN.xlsx"

with pd.ExcelWriter(output_file) as writer:
    df_summary.to_excel(writer, sheet_name="Summary_95CI", index=False)
    df_detailed.to_excel(writer, sheet_name="Detailed_Seeds", index=False)

print(f"\n[完成] 结果已保存至: {output_file}")
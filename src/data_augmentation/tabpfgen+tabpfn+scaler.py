# ==============================
# TabPFN + TabPFGen + StandardScaler Benchmark
# ==============================

import pandas as pd
import numpy as np
from scipy import stats

from tabpfgen import TabPFGen
from tabpfn import TabPFNClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 2. 多随机种子
# ==============================
seeds = range(40, 71)
detailed_records = []

print(f"开始 TabPFN + TabPFGen + 标准化实验，总种子数: {len(seeds)}")

# ==============================
# 3. 主循环
# ==============================
for seed in seeds:
    print(f"Processing seed: {seed}...", end='\r')

    # A. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.5,
        random_state=seed,
        stratify=y
    )

    # B. 标准化（关键：只用训练集 fit）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # C. TabPFGen 数据增强（在归一化空间）
    generator = TabPFGen(n_sgld_steps=500)

    X_synth, y_synth, X_combined, y_combined = generator.balance_dataset(
        X_train_scaled, y_train,
        target_per_class=500
    )

    # D. TabPFN 模型
    model = TabPFNClassifier(device='cuda')  # GPU可改 'cuda'

    model.fit(X_combined, y_combined)

    # E. 预测（使用同样归一化空间）
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # F. 记录指标
    detailed_records.append({
        "Seed": seed,
        "Model": "TabPFN + TabPFGen (Scaled)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    })


# ==============================
# 4. 95% CI 计算
# ==============================
def get_ci95(data):
    n = len(data)
    if n < 2:
        return 0
    se = stats.sem(data)
    return se * stats.t.ppf((1 + 0.95) / 2., n - 1)


# ==============================
# 5. 汇总结果
# ==============================
df_detailed = pd.DataFrame(detailed_records)

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

summary_row = {"Model": "TabPFN + TabPFGen (Scaled)"}

for m in metrics:
    mean_val = df_detailed[m].mean()
    ci_val = get_ci95(df_detailed[m])
    summary_row[m] = f"{mean_val:.3f} ± {ci_val:.3f}"

df_summary = pd.DataFrame([summary_row])


# ==============================
# 6. 输出
# ==============================
print("\n" + "=" * 60)
print("TabPFN + TabPFGen (Scaled) 性能汇总 (Mean ± 95% CI)")
print("=" * 60)
print(df_summary.to_string(index=False))

file_name = "Results_TabPFN_TabPFGen_Scaled.xlsx"

with pd.ExcelWriter(file_name) as writer:
    df_summary.to_excel(writer, sheet_name="Summary_95CI", index=False)
    df_detailed.to_excel(writer, sheet_name="Detailed_Seeds", index=False)

print(f"\n[完成] 结果已保存至: {file_name}")
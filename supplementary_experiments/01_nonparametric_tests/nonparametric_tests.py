# ======================================================
# Non-parametric Tests: Friedman + Wilcoxon Signed-Rank
# Models: TabPFN, XGBoost, CatBoost, SVM (SVR)
# Metric: RMSE across 31 seeds (40-70)
# ======================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

# -------------------------------------------------------
# 1. Load and pivot data
# -------------------------------------------------------
df = pd.read_csv("model_comparison_results.csv")
target_models = ["TabPFN", "XGBoost", "CatBoost", "SVM"]
df_filtered = df[df["Model"].isin(target_models)]

# Pivot: rows=Seed, columns=Model, values=RMSE
pivot = df_filtered.pivot(index="Seed", columns="Model", values="RMSE")

print("=" * 72)
print("  非参数检验分析: Friedman + Wilcoxon 符号秩检验")
print("  模型: TabPFN, XGBoost, CatBoost, SVM")
print("  指标: RMSE | 种子数:", pivot.shape[0])
print("=" * 72)

# -------------------------------------------------------
# 2. Descriptive statistics
# -------------------------------------------------------
print("\n--- 描述统计 (RMSE mean ± std) ---")
for model in target_models:
    vals = pivot[model]
    print(f"  {model:<12s}: {vals.mean():.6f} ± {vals.std():.6f}")
print(f"\n  最佳模型(均值最小): {pivot.mean().idxmin()}")

# -------------------------------------------------------
# 3. Friedman test (global comparison)
# -------------------------------------------------------
print("\n--- Friedman 检验 (全局比较) ---")
stat, p_friedman = friedmanchisquare(
    pivot["TabPFN"], pivot["XGBoost"],
    pivot["CatBoost"], pivot["SVM"]
)
print(f"  chi2 = {stat:.4f}")
print(f"  p    = {p_friedman:.6f}")
if p_friedman < 0.05:
    print("  结论: 四模型间存在显著差异 (p < 0.05)")
else:
    print("  结论: 四模型间无显著差异 (p ≥ 0.05)")

# -------------------------------------------------------
# 4. Wilcoxon signed-rank test (pairwise)
# -------------------------------------------------------
print("\n--- Wilcoxon 符号秩检验 (成对比较) ---")

alpha = 0.05
n_comparisons = len(list(combinations(target_models, 2)))
alpha_corrected = alpha / n_comparisons

print(f"  Bonferroni 校正阈值: alpha = {alpha} / {n_comparisons} = {alpha_corrected:.4f}")

results = []
for m1, m2 in combinations(target_models, 2):
    stat_w, p_raw = wilcoxon(pivot[m1], pivot[m2], alternative="two-sided")
    sig_raw = "Y" if p_raw < alpha else "-"
    sig_corrected = "Y" if p_raw < alpha_corrected else "-"
    results.append({
        "Model A": m1, "Model B": m2,
        "Statistic": stat_w, "p_raw": p_raw,
        "sig(.05)": sig_raw, "sig(corr)": sig_corrected
    })

print(f"\n  {'Model A':<12s} {'Model B':<12s} {'Statistic':>10s} {'p_raw':>10s}  "
      f"{'sig(.05)':>10s} {'sig({:.4f})'.format(alpha_corrected):>12s}")
print("  " + "-" * 70)
for r in results:
    print(f"  {r['Model A']:<12s} {r['Model B']:<12s} {r['Statistic']:10.1f} "
          f"{r['p_raw']:10.6f}  {r['sig(.05)']:>10s} {r['sig(corr)']:>12s}")

# -------------------------------------------------------
# 5. Summary
# -------------------------------------------------------
print("\n--- 汇总 ---")
n_sig_raw = sum(1 for r in results if r["p_raw"] < alpha)
n_sig_corrected = sum(1 for r in results if r["p_raw"] < alpha_corrected)
print(f"  Friedman: p = {p_friedman:.6f} ({'显著' if p_friedman < 0.05 else '不显著'})")
print(f"  成对显著 (alpha=0.05):      {n_sig_raw}/{n_comparisons} 对")
print(f"  成对显著 (Bonferroni):  {n_sig_corrected}/{n_comparisons} 对")

# Rank means for Friedman
ranks = pivot.rank(axis=1, ascending=True)  # rank 1 = best (lowest RMSE)
rank_means = ranks.mean().sort_values()
print(f"\n  Friedman 平均秩次 (越小越好):")
for model in rank_means.index:
    print(f"    {model:<12s}: {rank_means[model]:.2f}")

print("\n" + "=" * 72)
print("  分析完成。")

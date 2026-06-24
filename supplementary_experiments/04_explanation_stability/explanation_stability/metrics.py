"""解释稳定性定量指标函数库

所有指标接口统一为:
    metric(shap_vals_a, shap_vals_b, feature_names, **kwargs) -> float

其中 shap_vals_a / shap_vals_b 为形状 (n_samples, n_features) 的 SHAP 值矩阵。
指标计算前会自动聚合为逐特征平均绝对 SHAP 值作为特征重要性。
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from typing import List


def _feature_importance(shap_vals):
    """从 SHAP 矩阵提取特征重要性：逐特征平均绝对值。"""
    return np.mean(np.abs(shap_vals), axis=0)


def _rank_array(values):
    """将数值数组转换为 rank（降序，最大值为 rank 1）。"""
    order = np.argsort(values)[::-1]
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def kendall_tau(shap_vals_a, shap_vals_b, feature_names):
    """特征重要性排序的 Kendall tau 相关系数。
    基于平均绝对 SHAP 值的排序。完全一致 = 1.0，完全相反 = -1.0。
    """
    imp_a = _feature_importance(shap_vals_a)
    imp_b = _feature_importance(shap_vals_b)
    tau, _ = kendalltau(imp_a, imp_b)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def spearman_r(shap_vals_a, shap_vals_b, feature_names):
    """特征重要性排序的 Spearman 秩相关系数。
    基于平均绝对 SHAP 值的排序。完全一致 = 1.0，完全相反 = -1.0。
    """
    imp_a = _feature_importance(shap_vals_a)
    imp_b = _feature_importance(shap_vals_b)
    rho, _ = spearmanr(imp_a, imp_b)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def weighted_kendall_tau(shap_vals_a, shap_vals_b, feature_names):
    """加权 Kendall tau：Top 特征差异惩罚更重。
    权重 w_i = 1 / (rank_i + 1)，排名靠前的特征不一致时指标下降更多。
    完全一致 = 1.0，完全相反趋近 -1.0。
    """
    imp_a = _feature_importance(shap_vals_a)
    imp_b = _feature_importance(shap_vals_b)
    ranks_a = _rank_array(imp_a)
    ranks_b = _rank_array(imp_b)

    n = len(imp_a)
    weights_a = 1.0 / (ranks_a + 1.0)
    weights_b = 1.0 / (ranks_b + 1.0)

    concordant = 0.0
    discordant = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            a_sgn = np.sign(imp_a[i] - imp_a[j])
            b_sgn = np.sign(imp_b[i] - imp_b[j])
            w = (weights_a[i] + weights_a[j] + weights_b[i] + weights_b[j]) / 4.0
            if a_sgn * b_sgn > 0:
                concordant += w
            elif a_sgn * b_sgn < 0:
                discordant += w

    total = concordant + discordant
    if total == 0:
        return 0.0
    return float((concordant - discordant) / total)


def top_k_overlap(shap_vals_a, shap_vals_b, feature_names, k=10):
    """Top-K 特征重要性集合的 Jaccard 重叠率。
    返回 |topK_a AND topK_b| / |topK_a OR topK_b|。
    k 超过特征数时自动截断为特征总数。
    """
    imp_a = _feature_importance(shap_vals_a)
    imp_b = _feature_importance(shap_vals_b)

    k = min(k, len(imp_a))
    top_a = set(np.argsort(imp_a)[::-1][:k])
    top_b = set(np.argsort(imp_b)[::-1][:k])

    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    if union == 0:
        return 1.0
    return float(intersection / union)


def rbo(shap_vals_a, shap_vals_b, feature_names, p=0.9):
    """Rank-Biased Overlap：对 Top 特征更敏感的排序重叠度。
    RBO(S, T, p) 衡量两个排序列表的一致性，参数 p 控制 Top 权重：
    p=0.9 时约 90% 权重集中在排名前 10% 的特征。取值 [0, 1]。
    参考: Webber et al. (2010)
    """
    imp_a = _feature_importance(shap_vals_a)
    imp_b = _feature_importance(shap_vals_b)

    order_a = np.argsort(imp_a)[::-1]
    order_b = np.argsort(imp_b)[::-1]

    n = len(order_a)
    set_a = set()
    set_b = set()
    rbo_val = 0.0

    for d in range(1, n + 1):
        set_a.add(order_a[d - 1])
        set_b.add(order_b[d - 1])
        overlap = len(set_a & set_b)
        rbo_val += (p ** (d - 1)) * (overlap / d)

    rbo_val *= (1.0 - p)
    return float(rbo_val)


def pearson_r(shap_vals_a, shap_vals_b, feature_names):
    """SHAP 值数值层面的 Pearson 相关系数。
    比较两组的逐特征平均 SHAP 值（带符号）。完全一致 = 1.0。
    """
    imp_a = np.mean(shap_vals_a, axis=0)
    imp_b = np.mean(shap_vals_b, axis=0)

    std_a, std_b = np.std(imp_a), np.std(imp_b)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0

    r, _ = pearsonr(imp_a, imp_b)
    if np.isnan(r):
        return 0.0
    return float(r)


def sign_agreement(shap_vals_a, shap_vals_b, feature_names):
    """特征 SHAP 方向一致性：符号相同的特征比例。
    取逐特征平均 SHAP 值的符号，取值范围 [0, 1]。
    """
    imp_a = np.mean(shap_vals_a, axis=0)
    imp_b = np.mean(shap_vals_b, axis=0)

    sign_a = np.sign(imp_a)
    sign_b = np.sign(imp_b)

    agreement = np.sum(sign_a == sign_b)
    return float(agreement / len(imp_a))


# 指标注册表，方便遍历
METRIC_FUNCTIONS = {
    "kendall_tau": kendall_tau,
    "spearman_r": spearman_r,
    "weighted_kendall_tau": weighted_kendall_tau,
    "pearson_r": pearson_r,
    "sign_agreement": sign_agreement,
}

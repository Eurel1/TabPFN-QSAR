"""Bootstrap驱动的解释稳定性分析

通过 Bootstrap 重采样生成 N 个不同训练集，
每次训练 TabPFN 并计算 SHAP 值，
然后对所有配对调用全套稳定性指标。
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Optional, Tuple

from .metrics import (
    kendall_tau,
    spearman_r,
    weighted_kendall_tau,
    top_k_overlap,
    rbo,
    pearson_r,
    sign_agreement,
)

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

DEFAULT_TOP_K_VALUES = [5, 10, 20]
DEFAULT_RBO_P = 0.9


# --------------------------------------------------------------------------
# Bootstrap 驱动
# --------------------------------------------------------------------------

def run_bootstrap_analysis(
    X,
    y,
    n_bootstrap=30,
    test_size=0.2,
    random_state=42,
    model_class=None,
    model_kwargs=None,
    top_k_values=None,
    device="cuda",
    model_fit_kwargs=None,
) -> Dict:
    """执行 Bootstrap 解释稳定性分析。

    返回字典：
        - shap_list: 每次 Bootstrap 的 SHAP 值列表
        - pairwise_results: 配对指标的字典，每项为 (n_pairs,) 数组
        - summary: 每个指标的 {mean, std, min, max}
        - feature_names: 特征名列表
    """
    if model_class is None:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNRegressor,
        )
        model_class = AutoTabPFNRegressor
    if model_kwargs is None:
        model_kwargs = {"device": device}
    if model_fit_kwargs is None:
        model_fit_kwargs = {}
    if top_k_values is None:
        top_k_values = DEFAULT_TOP_K_VALUES

    rng = np.random.RandomState(random_state)
    top_k_values = list(top_k_values)
    feature_names = list(X.columns)

    # 固定测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    n_train = len(X_train_full)

    print(f"训练集: {n_train}, 测试集: {len(X_test)}")
    print(f"Bootstrap 次数: {n_bootstrap}")

    # 存储每次 Bootstrap 的 SHAP 值
    shap_list = []

    for b in range(n_bootstrap):
        indices = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train_full.iloc[indices]
        y_boot = y_train_full.iloc[indices]

        # 去掉 Bootstrap 样本中的常数列
        constant_cols = X_boot.columns[X_boot.nunique() <= 1]
        X_boot = X_boot.drop(columns=constant_cols)
        X_test_sub = X_test.drop(columns=constant_cols, errors="ignore")

        try:
            model = model_class(**model_kwargs).fit(X_boot, y_boot, **model_fit_kwargs)
        except Exception as e:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}: 训练失败 ({e})，跳过")
            continue

        preds = model.predict(X_test_sub)
        rmse = mean_squared_error(y_test, preds, squared=False)

        # 计算 SHAP
        try:
            from tabpfn_extensions import interpretability
            shap_vals = interpretability.shap.get_shap_values(
                estimator=model,
                test_x=X_test_sub,
                attribute_names=list(X_test_sub.columns),
                algorithm="permutation",
            )
        except Exception as e:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}: SHAP 计算失败 ({e})，跳过")
            continue

        # 转换 Explanation 对象为 numpy 数组
        shap_vals_raw = np.array(shap_vals.values)

        # 补齐缺失特征的 SHAP 值为 0
        shap_full = np.zeros((len(y_test), len(feature_names)))
        for j, col in enumerate(X_test_sub.columns):
            idx = feature_names.index(col)
            shap_full[:, idx] = shap_vals_raw[:, j]

        shap_list.append(shap_full)
        print(f"  Bootstrap {b + 1}/{n_bootstrap}: RMSE={rmse:.4f}, 特征数={len(X_test_sub.columns)}")

    print(f"\n成功完成 {len(shap_list)}/{n_bootstrap} 次 Bootstrap\n")

    if len(shap_list) < 2:
        raise RuntimeError(f"只成功了 {len(shap_list)} 次 Bootstrap，不足以计算配对指标")

    # 计算配对指标
    pairwise_results = compute_pairwise_metrics(
        shap_list, feature_names, top_k_values=top_k_values
    )

    # 汇总
    summary = {}
    for metric_name, values in pairwise_results.items():
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "shap_list": shap_list,
        "pairwise_results": pairwise_results,
        "summary": summary,
        "feature_names": feature_names,
    }


# --------------------------------------------------------------------------
# 配对指标计算
# --------------------------------------------------------------------------

def compute_pairwise_metrics(
    shap_list,
    feature_names,
    top_k_values=None,
    rbo_p=DEFAULT_RBO_P,
) -> Dict[str, np.ndarray]:
    """对所有 SHAP 配对计算全套指标。

    返回字典: 指标名 -> (n_pairs,) 数组
    """
    if top_k_values is None:
        top_k_values = DEFAULT_TOP_K_VALUES

    n = len(shap_list)
    n_pairs = n * (n - 1) // 2

    results = defaultdict(lambda: np.zeros(n_pairs))

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            results["kendall_tau"][idx] = kendall_tau(
                shap_list[i], shap_list[j], feature_names
            )
            results["spearman_r"][idx] = spearman_r(
                shap_list[i], shap_list[j], feature_names
            )
            results["weighted_kendall_tau"][idx] = weighted_kendall_tau(
                shap_list[i], shap_list[j], feature_names
            )
            results["pearson_r"][idx] = pearson_r(
                shap_list[i], shap_list[j], feature_names
            )
            results["sign_agreement"][idx] = sign_agreement(
                shap_list[i], shap_list[j], feature_names
            )
            results["rbo"][idx] = rbo(
                shap_list[i], shap_list[j], feature_names, p=rbo_p
            )
            for k in top_k_values:
                key = f"top_{k}_overlap"
                results[key][idx] = top_k_overlap(
                    shap_list[i], shap_list[j], feature_names, k=k
                )
            idx += 1

    return dict(results)


# --------------------------------------------------------------------------
# 可视化报告
# --------------------------------------------------------------------------

def report_stability(
    results: Dict,
    output_dir: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """生成解释稳定性报告：控制台汇总表 + 箱线图 + 热力图。

    如果指定 output_dir，会保存 PNG 和 CSV。
    """
    summary = results["summary"]
    pairwise = results["pairwise_results"]

    # -------- 控制台汇总表 --------
    print("=" * 72)
    print("  解释稳定性评估报告 (Bootstrap 重采样)")
    print("=" * 72)
    print(f"  {'指标':<28s} {'均值':>8s} {'标准差':>8s} {'最小':>8s} {'最大':>8s}")
    print("  " + "-" * 64)

    ordered_keys = sorted(summary.keys())
    for key in ordered_keys:
        s = summary[key]
        display_name = _metric_display_name(key)
        print(f"  {display_name:<28s} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f}")

    print("=" * 72)
    print()

    # -------- 箱线图 --------
    fig, ax = plt.subplots(figsize=(12, 5))
    metric_labels = [_metric_display_name(k) for k in ordered_keys]
    data_to_plot = [pairwise[k] for k in ordered_keys]

    bp = ax.boxplot(data_to_plot, labels=metric_labels, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
              "#ffd92f", "#e5c494", "#b3b3b3", "#1f78b4", "#33a02c"]
    for patch, color in zip(bp["boxes"], colors[:len(ordered_keys)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_title("Explanation Stability Metrics (Boxplot across Bootstrap Pairs)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "stability_boxplot.png"), dpi=150, bbox_inches="tight")
        print(f"箱线图已保存: {output_dir}/stability_boxplot.png")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # -------- 热力图 --------
    fig2, ax2 = plt.subplots(figsize=(10, max(3, len(ordered_keys) * 0.5)))
    heatmap_data = np.array([pairwise[k] for k in ordered_keys])
    im = ax2.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax2.set_yticks(range(len(ordered_keys)))
    ax2.set_yticklabels(metric_labels, fontsize=9)
    ax2.set_xlabel(f"Bootstrap Pair Index (total {heatmap_data.shape[1]} pairs)")
    ax2.set_title("Pairwise Metric Heatmap", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Metric Value")
    plt.tight_layout()

    if output_dir:
        fig2.savefig(os.path.join(output_dir, "stability_heatmap.png"), dpi=150, bbox_inches="tight")
        print(f"热力图已保存: {output_dir}/stability_heatmap.png")

        # 导出 CSV
        df_out = pd.DataFrame(summary).T
        df_out.index.name = "metric"
        csv_path = os.path.join(output_dir, "stability_summary.csv")
        df_out.to_csv(csv_path, encoding="utf-8-sig")
        print(f"汇总表已保存: {csv_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig2)


def _metric_display_name(key: str) -> str:
    """指标内部 key 到显示名称的映射。"""
    mapping = {
        "kendall_tau": "Kendall tau",
        "spearman_r": "Spearman r",
        "weighted_kendall_tau": "Weighted Kendall tau",
        "pearson_r": "Pearson r (SHAP)",
        "sign_agreement": "Sign Agreement",
        "rbo": "RBO (p=0.9)",
        "top_5_overlap": "Top-5 Overlap",
        "top_10_overlap": "Top-10 Overlap",
        "top_20_overlap": "Top-20 Overlap",
    }
    return mapping.get(key, key)


# --------------------------------------------------------------------------
# 缺失率扰动分析
# --------------------------------------------------------------------------

def run_missing_rate_analysis(
    X,
    y,
    missing_rates=(0.0, 0.05, 0.10, 0.15),
    test_size=0.2,
    random_state=42,
    model_class=None,
    model_kwargs=None,
    top_k_values=None,
    device="cuda",
    model_fit_kwargs=None,
) -> Dict:
    """执行缺失率扰动下的解释稳定性分析。

    以 rate=0.0 为基线，逐缺失率比较 SHAP 解释一致性。

    参数:
        missing_rates: 缺失率列表（第一个应为 0.0 作为基线）
    返回:
        - per_rate: {rate: {"shap": array, "rmse": float}} 每个率的 SHAP 和 RMSE
        - metrics_vs_baseline: {rate: {metric_name: float}} 各率 vs 基线的指标
        - feature_names: 特征名列表
    """
    if model_class is None:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
            AutoTabPFNRegressor,
        )
        model_class = AutoTabPFNRegressor
    if model_kwargs is None:
        model_kwargs = {"device": device}
    if model_fit_kwargs is None:
        model_fit_kwargs = {}
    if top_k_values is None:
        top_k_values = DEFAULT_TOP_K_VALUES

    rng = np.random.RandomState(random_state)
    top_k_values = list(top_k_values)
    feature_names = list(X.columns)

    # 固定测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"缺失率: {list(missing_rates)}")

    per_rate = {}

    for rate in missing_rates:
        print(f"\n--- 缺失率 {rate*100:.0f}% ---")

        # 对训练集注入缺失
        if rate == 0.0:
            X_train_miss = X_train.copy()
        else:
            mask = rng.choice([True, False], size=X_train.shape,
                              p=[rate, 1 - rate])
            X_train_miss = pd.DataFrame(
                np.where(mask, np.nan, X_train),
                columns=X_train.columns,
                index=X_train.index,
            )

        # 对齐特征
        common_cols = [c for c in feature_names
                       if c in X_train_miss.columns]
        X_train_sub = X_train_miss[common_cols]
        X_test_sub = X_test[common_cols]

        try:
            model = model_class(**model_kwargs).fit(X_train_sub, y_train, **model_fit_kwargs)
        except Exception as e:
            print(f"  训练失败 ({e})，跳过")
            continue

        preds = model.predict(X_test_sub)
        rmse = mean_squared_error(y_test, preds, squared=False)
        print(f"  RMSE: {rmse:.4f}")

        # 计算 SHAP
        try:
            from tabpfn_extensions import interpretability
            shap_vals = interpretability.shap.get_shap_values(
                estimator=model,
                test_x=X_test_sub,
                attribute_names=list(X_test_sub.columns),
                algorithm="permutation",
            )
        except Exception as e:
            print(f"  SHAP 计算失败 ({e})，跳过")
            continue

        # 转换 Explanation 对象为 numpy 数组
        shap_vals_raw = np.array(shap_vals.values)

        # 补齐到完整特征空间
        shap_full = np.zeros((len(y_test), len(feature_names)))
        for j, col in enumerate(X_test_sub.columns):
            idx = feature_names.index(col)
            shap_full[:, idx] = shap_vals_raw[:, j]

        per_rate[rate] = {"shap": shap_full, "rmse": rmse}

    # 基线是 rate=0.0
    baseline_rate = 0.0
    if baseline_rate not in per_rate:
        raise RuntimeError("基线 (rate=0.0) 未成功计算")

    baseline_shap = per_rate[baseline_rate]["shap"]

    print(f"\n{'='*60}")
    print(f"  缺失率稳定性指标 (基线: rate=0.0)")
    print(f"{'='*60}")

    metrics_vs_baseline = {}
    for rate in missing_rates:
        if rate == baseline_rate or rate not in per_rate:
            continue
        row = {}
        shap_r = per_rate[rate]["shap"]
        row["kendall_tau"] = kendall_tau(baseline_shap, shap_r, feature_names)
        row["spearman_r"] = spearman_r(baseline_shap, shap_r, feature_names)
        row["weighted_kendall_tau"] = weighted_kendall_tau(
            baseline_shap, shap_r, feature_names
        )
        row["pearson_r"] = pearson_r(baseline_shap, shap_r, feature_names)
        row["sign_agreement"] = sign_agreement(
            baseline_shap, shap_r, feature_names
        )
        row["rbo"] = rbo(baseline_shap, shap_r, feature_names, p=DEFAULT_RBO_P)
        for k in top_k_values:
            row[f"top_{k}_overlap"] = top_k_overlap(
                baseline_shap, shap_r, feature_names, k=k
            )
        metrics_vs_baseline[rate] = row

        print(f"\n  缺失率 {rate*100:.0f}% (RMSE={per_rate[rate]['rmse']:.4f}):")
        for name in sorted(row):
            print(f"    {_metric_display_name(name):<28s} {row[name]:.4f}")

    return {
        "per_rate": per_rate,
        "metrics_vs_baseline": metrics_vs_baseline,
        "feature_names": feature_names,
    }


# --------------------------------------------------------------------------
# 缺失率可视化报告
# --------------------------------------------------------------------------

def report_missing_rate_stability(
    results: Dict,
    output_dir: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """生成缺失率稳定性报告：折线图展示各指标随缺失率的变化。

    如果指定 output_dir，会保存 PNG 和 CSV。
    """
    metrics_vs_baseline = results["metrics_vs_baseline"]
    rates = sorted(metrics_vs_baseline.keys())

    if len(rates) == 0:
        print("无可用缺失率数据")
        return

    # 收集所有指标名
    all_metrics = sorted(metrics_vs_baseline[rates[0]].keys())

    # -------- 折线图 --------
    fig, ax = plt.subplots(figsize=(10, 6))
    rate_pct = [r * 100 for r in rates]

    for name in all_metrics:
        values = [metrics_vs_baseline[r][name] for r in rates]
        ax.plot(rate_pct, values, marker="o", linewidth=2,
                label=_metric_display_name(name), markersize=5)

    ax.set_xlabel("Missing Rate (%)", fontsize=12)
    ax.set_ylabel("Stability vs Baseline (rate=0%)", fontsize=12)
    ax.set_title("Explanation Stability vs Missing Data Rate", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "missing_rate_stability.png"),
                    dpi=150, bbox_inches="tight")
        print(f"折线图已保存: {output_dir}/missing_rate_stability.png")

        # CSV
        df_rows = []
        for r in rates:
            row = {"missing_rate": r * 100}
            row.update(metrics_vs_baseline[r])
            df_rows.append(row)
        df_out = pd.DataFrame(df_rows)
        csv_path = os.path.join(output_dir, "missing_rate_stability.csv")
        df_out.to_csv(csv_path, encoding="utf-8-sig", index=False)
        print(f"CSV 已保存: {csv_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # -------- 控制台表格 --------
    print()
    print("=" * 90)
    print("  缺失率 vs 基线 (rate=0.0) 稳定性汇总")
    print("=" * 90)

    header = f"  {'缺失率':>8s}"
    for name in all_metrics[:5]:
        header += f" {_metric_display_name(name):>18s}"
    print(header)
    print("  " + "-" * 86)
    for r in rates:
        row = f"  {r*100:7.0f}%"
        for name in all_metrics[:5]:
            row += f" {metrics_vs_baseline[r][name]:18.4f}"
        print(row)

    if len(all_metrics) > 5:
        print()
        header2 = f"  {'缺失率':>8s}"
        for name in all_metrics[5:]:
            header2 += f" {_metric_display_name(name):>18s}"
        print(header2)
        print("  " + "-" * 86)
        for r in rates:
            row = f"  {r*100:7.0f}%"
            for name in all_metrics[5:]:
                row += f" {metrics_vs_baseline[r][name]:18.4f}"
            print(row)

    print("=" * 90)

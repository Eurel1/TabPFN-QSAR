"""解释稳定性定量评估 - 顶层入口脚本

支持两种扰动模式：
  - bootstrap: Bootstrap 重采样评估解释稳定性
  - missing:   缺失率扰动评估解释稳定性

用法:
  python run_stability.py            # 默认 bootstrap 模式
  python run_stability.py missing    # 缺失率模式
  python run_stability.py both       # 两种都跑
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from explanation_stability.bootstrap_analysis import (
    run_bootstrap_analysis,
    run_missing_rate_analysis,
    report_stability,
    report_missing_rate_stability,
)


# ==========================================================================
# 可配置参数
# ==========================================================================

DATA_PATH = os.path.join(PROJECT_ROOT, "categorized_maccs.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "stability_results")

N_BOOTSTRAP = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K_VALUES = [5, 10, 20]
DEVICE = "cuda"

# 缺失率分析参数
MISSING_RATES = [0.0, 0.05, 0.10, 0.15]


# ==========================================================================
# 主流程
# ==========================================================================

def load_data():
    """加载并预处理数据。"""
    maccs_df = pd.read_excel(DATA_PATH, sheet_name="Sheet1", header=0)
    X = maccs_df.filter(regex="MACCS_\\d+")
    y = maccs_df.iloc[:, 5]

    constant_cols = X.columns[X.nunique() == 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)
        print(f"  移除 {len(constant_cols)} 个常数列")

    print(f"  数据: {X.shape[0]} 样本 x {X.shape[1]} 特征")
    return X, y


def run_bootstrap(X, y):
    """Bootstrap 模式。"""
    print(f"\n{'='*72}")
    print("  模式: Bootstrap 重采样")
    print(f"{'='*72}\n")

    results = run_bootstrap_analysis(
        X, y,
        n_bootstrap=N_BOOTSTRAP,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        top_k_values=TOP_K_VALUES,
        device=DEVICE,
    )

    output = os.path.join(OUTPUT_DIR, "bootstrap")
    os.makedirs(output, exist_ok=True)
    report_stability(results, output_dir=output, show_plot=False)
    return results


def run_missing(X, y):
    """缺失率模式。"""
    print(f"\n{'='*72}")
    print("  模式: 缺失率扰动")
    print(f"{'='*72}\n")

    results = run_missing_rate_analysis(
        X, y,
        missing_rates=MISSING_RATES,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        top_k_values=TOP_K_VALUES,
        device=DEVICE,
    )

    output = os.path.join(OUTPUT_DIR, "missing_rate")
    os.makedirs(output, exist_ok=True)
    report_missing_rate_stability(results, output_dir=output, show_plot=False)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bootstrap"

    print("=" * 72)
    print("  解释稳定性定量评估")
    print("=" * 72)

    print("\n[1/2] 加载数据...")
    X, y = load_data()

    print("\n[2/2] 运行分析...")
    if mode == "bootstrap":
        run_bootstrap(X, y)
    elif mode == "missing":
        run_missing(X, y)
    elif mode == "both":
        run_bootstrap(X, y)
        run_missing(X, y)
    else:
        print(f"未知模式: {mode}")
        print("可用模式: bootstrap, missing, both")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  完成! 输出: {OUTPUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

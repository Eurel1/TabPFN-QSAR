# ==============================
# Fit Data Utility Curve + Detect Two Breakpoints
# For uploaded file: learning_curve_76.csv
# ==============================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ==============================
# 1. Read data
# ==============================
file_path = "learning_curve_42.csv"   # 修改为你的文件路径

df = pd.read_csv(file_path)

# 这里默认你的列名就是这两个
x_obs = df["Samples"].values.astype(float)
y_obs = df["RMSE"].values.astype(float)

# 按样本量排序，避免顺序问题
sort_idx = np.argsort(x_obs)
x_obs = x_obs[sort_idx]
y_obs = y_obs[sort_idx]

print("Data preview:")
print(df.head())
print(f"\nTotal points: {len(x_obs)}")
print(f"Sample range: {x_obs.min():.0f} - {x_obs.max():.0f}")


# ==============================
# 2. Candidate smooth functions
# ==============================
def exp_decay(n, a, b, c):
    """Exponential decay."""
    return a * np.exp(-b * n) + c

def power_law(n, a, b, c):
    """Power-law decay."""
    return a * np.power(n, -b) + c


# ==============================
# 3. Fit smooth curve
# ==============================
def fit_smooth_curve(x, y):
    """
    Fit exponential and power-law curves.
    Select the one with lower SSE.
    """
    candidates = []

    # exponential
    try:
        exp_p0 = [max(y) - min(y), 0.01, min(y)]
        exp_params, _ = curve_fit(
            exp_decay,
            x, y,
            p0=exp_p0,
            bounds=([0, 1e-6, 0], [np.inf, 10, np.inf]),
            maxfev=20000
        )
        y_fit = exp_decay(x, *exp_params)
        sse = np.sum((y - y_fit) ** 2)
        candidates.append(("exponential", exp_params, sse, exp_decay))
    except Exception as e:
        print("Exponential fit failed:", e)

    # power law
    try:
        pow_p0 = [10.0, 0.5, min(y)]
        pow_params, _ = curve_fit(
            power_law,
            x, y,
            p0=pow_p0,
            bounds=([0, 1e-6, 0], [np.inf, 10, np.inf]),
            maxfev=20000
        )
        y_fit = power_law(x, *pow_params)
        sse = np.sum((y - y_fit) ** 2)
        candidates.append(("power_law", pow_params, sse, power_law))
    except Exception as e:
        print("Power-law fit failed:", e)

    if not candidates:
        raise RuntimeError("No smooth fit succeeded.")

    best_name, best_params, best_sse, best_func = sorted(candidates, key=lambda z: z[2])[0]
    return best_name, best_params, best_sse, best_func


# ==============================
# 4. Piecewise helpers
# ==============================
def fit_line(x, y):
    coef = np.polyfit(x, y, deg=1)
    y_hat = np.polyval(coef, x)
    sse = np.sum((y - y_hat) ** 2)
    return coef, y_hat, sse


def three_segment_piecewise(x_grid, y_grid, min_points=20):
    """
    Search two breakpoints that minimize total SSE of
    three linear segments.
    """
    best_piecewise = None
    best_total_sse = np.inf

    for i in range(min_points, len(x_grid) - 2 * min_points):
        for j in range(i + min_points, len(x_grid) - min_points):
            x1, y1 = x_grid[:i], y_grid[:i]
            x2, y2 = x_grid[i:j], y_grid[i:j]
            x3, y3 = x_grid[j:], y_grid[j:]

            coef1, yhat1, sse1 = fit_line(x1, y1)
            coef2, yhat2, sse2 = fit_line(x2, y2)
            coef3, yhat3, sse3 = fit_line(x3, y3)

            total_sse = sse1 + sse2 + sse3

            if total_sse < best_total_sse:
                best_total_sse = total_sse
                best_piecewise = {
                    "i": i,
                    "j": j,
                    "coef1": coef1,
                    "coef2": coef2,
                    "coef3": coef3,
                    "x1": x1, "yhat1": yhat1,
                    "x2": x2, "yhat2": yhat2,
                    "x3": x3, "yhat3": yhat3,
                    "total_sse": total_sse
                }

    if best_piecewise is None:
        raise RuntimeError("Three-segment piecewise regression failed.")

    bp1 = float(x_grid[best_piecewise["i"]])
    bp2 = float(x_grid[best_piecewise["j"]])

    return best_piecewise, bp1, bp2


# ==============================
# 5. Smooth fit on observed data
# ==============================
fit_name, fit_params, fit_sse, fit_func = fit_smooth_curve(x_obs, y_obs)

print("\n===== Smooth Fit Result =====")
print(f"Best fit type: {fit_name}")
print(f"Parameters: {fit_params}")
print(f"SSE: {fit_sse:.6f}")

# 建立更密的横轴网格，用于平滑曲线与断点搜索
x_grid = np.arange(int(x_obs.min()), int(x_obs.max()) + 1)
y_grid = fit_func(x_grid, *fit_params)


# ==============================
# 6. Detect two breakpoints
# ==============================
# min_points 可按数据长度自动适配，避免太短区间
min_points = max(5, len(x_grid) // 20)

piecewise_result, bp1, bp2 = three_segment_piecewise(
    x_grid, y_grid, min_points=min_points
)

print("\n===== Three-Segment Breakpoints =====")
print(f"Breakpoint 1 = {bp1:.0f}")
print(f"Breakpoint 2 = {bp2:.0f}")
print(f"Piecewise total SSE = {piecewise_result['total_sse']:.6f}")


# ==============================
# 7. Save results
# ==============================
summary_df = pd.DataFrame([{
    "best_fit_type": fit_name,
    "param_a": fit_params[0],
    "param_b": fit_params[1],
    "param_c": fit_params[2],
    "fit_sse": fit_sse,
    "breakpoint_1": bp1,
    "breakpoint_2": bp2,
    "piecewise_total_sse": piecewise_result["total_sse"]
}])

summary_df.to_csv("duc_fit_breakpoints_summary.csv", index=False)
print("\nSaved summary to: duc_fit_breakpoints_summary.csv")


# ==============================
# 8. Plot
# ==============================
MAIN_COLOR = "#E69F00"   # 主拟合线
POINT_COLOR = "#E69F00"  # 数据点
BP_COLOR = "#D55E00"     # 断点

fig, ax = plt.subplots(figsize=(7.0, 5.2))

# observed data
ax.scatter(
    x_obs, y_obs,
    s=30, alpha=0.45, color=POINT_COLOR,
    label="Observed data", zorder=2
)

# smooth fitted curve
ax.plot(
    x_grid, y_grid,
    color=MAIN_COLOR, linewidth=3.0,
    label=f"Fitted {fit_name} curve", zorder=4
)

# optional: piecewise lines
ax.plot(
    piecewise_result["x1"], piecewise_result["yhat1"],
    linestyle="--", linewidth=1.5, color="#56B4E9",
    zorder=3
)
ax.plot(
    piecewise_result["x2"], piecewise_result["yhat2"],
    linestyle="--", linewidth=1.5, color="#009E73",
    zorder=3
)
ax.plot(
    piecewise_result["x3"], piecewise_result["yhat3"],
    linestyle="--", linewidth=1.5, color="#CC79A7",
    zorder=3
)

# breakpoints
ax.axvline(
    bp1, linestyle=":", linewidth=1.8, color=BP_COLOR,
    label=f"Breakpoint 1 (N={bp1:.0f})", zorder=1
)
ax.axvline(
    bp2, linestyle="-.", linewidth=1.8, color=BP_COLOR,
    label=f"Breakpoint 2 (N={bp2:.0f})", zorder=1
)

# breakpoint markers on smooth curve
ax.scatter(
    [bp1, bp2],
    [
        fit_func(bp1, *fit_params),
        fit_func(bp2, *fit_params)
    ],
    s=55, color=BP_COLOR, zorder=5
)

ax.set_xlabel("Sample size (N)")
ax.set_ylabel("RMSE")
ax.set_title("Data Utility Curve with Two Breakpoints")
ax.tick_params(direction="in", length=5, width=1)
ax.legend(frameon=False, loc="best")

fig.tight_layout()
fig.savefig("duc_fit_with_breakpoints.png", dpi=300, bbox_inches="tight")
plt.show()
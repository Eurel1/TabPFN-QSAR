# ==============================
# TabPFN-guided Data Utility Curve
# Efficient window detection by 3-segment piecewise regression
# ==============================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tabpfn import TabPFNRegressor


# ==============================
# 0. Global settings
# ==============================
RANDOM_STATE = 42
INITIAL_N = 200
TEST_SIZE = 0.2
REPEATS = 8
SUBSET_SIZES = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

MAX_SEARCH_N = 1000   # 外推上限


# ==============================
# 1. Read data and preprocess
# ==============================
df = pd.read_excel("categorized_maccs.xlsx")

X = df.filter(regex=r"MACCS_\d+").copy()
y = df["logkOH•"].copy()

constant_cols = X.columns[X.nunique() == 1]
X = X.drop(columns=constant_cols)

print(f"Original dataset size: {len(X)}")
print(f"Feature dimension after removing constant columns: {X.shape[1]}")


# ==============================
# 2. Fixed train/test split
# ==============================
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train_pool = X_train_pool.reset_index(drop=True)
y_train_pool = y_train_pool.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

if len(X_train_pool) < INITIAL_N:
    raise ValueError("Training pool is smaller than INITIAL_N.")


# ==============================
# 3. Simulate initial low-data scenario
# ==============================
rng = np.random.default_rng(RANDOM_STATE)
initial_indices = rng.choice(len(X_train_pool), size=INITIAL_N, replace=False)

X_init = X_train_pool.iloc[initial_indices].reset_index(drop=True)
y_init = y_train_pool.iloc[initial_indices].reset_index(drop=True)

print(f"Initial available sample size: {len(X_init)}")


# ==============================
# 4. Utility curve functions
# ==============================
def exp_decay(n, a, b, c):
    return a * np.exp(-b * n) + c

def power_law(n, a, b, c):
    return a * np.power(n, -b) + c


# ==============================
# 5. Repeated subsampling evaluation
# ==============================
results = []

for n in SUBSET_SIZES:
    rmse_list = []

    for rep in range(REPEATS):
        rep_rng = np.random.default_rng(RANDOM_STATE + rep + n)
        subset_idx = rep_rng.choice(len(X_init), size=n, replace=False)

        X_sub = X_init.iloc[subset_idx]
        y_sub = y_init.iloc[subset_idx]

        model = TabPFNRegressor()
        model.fit(X_sub, y_sub)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    results.append({
        "N": n,
        "RMSE_mean": np.mean(rmse_list),
        "RMSE_std": np.std(rmse_list, ddof=1) if len(rmse_list) > 1 else 0.0
    })

results_df = pd.DataFrame(results)
print("\nObserved utility curve:")
print(results_df)


# ==============================
# 6. Fit smooth extrapolation curve
# ==============================
x_obs = results_df["N"].values.astype(float)
y_obs = results_df["RMSE_mean"].values.astype(float)

exp_p0 = [max(y_obs) - min(y_obs), 0.01, min(y_obs)]
pow_p0 = [10.0, 0.5, min(y_obs)]

fit_candidates = []

try:
    exp_params, _ = curve_fit(
        exp_decay, x_obs, y_obs,
        p0=exp_p0,
        bounds=([0, 1e-6, 0], [np.inf, 10, np.inf]),
        maxfev=20000
    )
    exp_pred = exp_decay(x_obs, *exp_params)
    exp_sse = np.sum((y_obs - exp_pred) ** 2)
    fit_candidates.append(("exponential", exp_params, exp_sse))
except Exception as e:
    print("Exponential fit failed:", e)

try:
    pow_params, _ = curve_fit(
        power_law, x_obs, y_obs,
        p0=pow_p0,
        bounds=([0, 1e-6, 0], [np.inf, 10, np.inf]),
        maxfev=20000
    )
    pow_pred = power_law(x_obs, *pow_params)
    pow_sse = np.sum((y_obs - pow_pred) ** 2)
    fit_candidates.append(("power_law", pow_params, pow_sse))
except Exception as e:
    print("Power-law fit failed:", e)

if not fit_candidates:
    raise RuntimeError("No smooth fit succeeded.")

best_name, best_params, best_sse = sorted(fit_candidates, key=lambda x: x[2])[0]
fit_func = exp_decay if best_name == "exponential" else power_law

print(f"\nBest smooth fit: {best_name}")
print(f"Parameters: {best_params}")


# ==============================
# 7. Extrapolated curve
# ==============================
n_grid = np.arange(min(SUBSET_SIZES), MAX_SEARCH_N + 1)
rmse_grid = fit_func(n_grid, *best_params)


# ==============================
# 8. Three-segment piecewise regression
#    Search two breakpoints that minimize total SSE
# ==============================
def fit_line(x, y):
    coef = np.polyfit(x, y, deg=1)
    y_hat = np.polyval(coef, x)
    sse = np.sum((y - y_hat) ** 2)
    return coef, y_hat, sse

best_piecewise = None
best_total_sse = np.inf

# 为避免每段点数过少，要求每段至少有 min_points 个点
min_points = 20

for i in range(min_points, len(n_grid) - 2 * min_points):
    for j in range(i + min_points, len(n_grid) - min_points):
        x1, y1 = n_grid[:i], rmse_grid[:i]
        x2, y2 = n_grid[i:j], rmse_grid[i:j]
        x3, y3 = n_grid[j:], rmse_grid[j:]

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
                "x3": x3, "yhat3": yhat3
            }

i = best_piecewise["i"]
j = best_piecewise["j"]

N_start = int(n_grid[i])   # 第一断点：高效窗口起点
N_end = int(n_grid[j])     # 第二断点：高效窗口终点

print("\nEstimated efficient sample window by three-segment regression:")
print(f"N_start = {N_start}")
print(f"N_end   = {N_end}")


# ==============================
# 9. Optional retrospective validation
# ==============================
oracle_results = []
oracle_subset_sizes = sorted(set(SUBSET_SIZES + [250, 300, 400, 500, 600, 700, 800]))
oracle_subset_sizes = [n for n in oracle_subset_sizes if n <= len(X_train_pool)]

for n in oracle_subset_sizes:
    rmse_list = []

    for rep in range(REPEATS):
        rep_rng = np.random.default_rng(1000 + rep + n)
        subset_idx = rep_rng.choice(len(X_train_pool), size=n, replace=False)

        X_sub = X_train_pool.iloc[subset_idx]
        y_sub = y_train_pool.iloc[subset_idx]

        model = TabPFNRegressor()
        model.fit(X_sub, y_sub)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    oracle_results.append({
        "N": n,
        "RMSE_mean": np.mean(rmse_list),
        "RMSE_std": np.std(rmse_list, ddof=1) if len(rmse_list) > 1 else 0.0
    })

oracle_df = pd.DataFrame(oracle_results)


# ==============================
# 10. Save outputs
# ==============================
results_df.to_csv("observed_initial200_utility_curve.csv", index=False)
oracle_df.to_csv("oracle_full_pool_utility_curve.csv", index=False)

summary_df = pd.DataFrame([{
    "best_smooth_fit": best_name,
    "param_a": best_params[0],
    "param_b": best_params[1],
    "param_c": best_params[2],
    "N_start": N_start,
    "N_end": N_end
}])
summary_df.to_csv("piecewise_window_summary.csv", index=False)


# ==============================
# 11. Plot
# ==============================
plt.figure(figsize=(9, 6))

# observed points
plt.errorbar(
    results_df["N"], results_df["RMSE_mean"], yerr=results_df["RMSE_std"],
    fmt='o', capsize=4, label="Observed curve (initial 200)"
)

# smooth fitted + extrapolated curve
obs_max = max(SUBSET_SIZES)
mask_obs = n_grid <= obs_max
mask_ext = n_grid > obs_max

plt.plot(n_grid[mask_obs], rmse_grid[mask_obs], label=f"Fitted {best_name}")
plt.plot(n_grid[mask_ext], rmse_grid[mask_ext], '--', label="Extrapolated curve")

# piecewise segments
plt.plot(best_piecewise["x1"], best_piecewise["yhat1"], linewidth=2, label="Piecewise segment 1")
plt.plot(best_piecewise["x2"], best_piecewise["yhat2"], linewidth=2, label="Piecewise segment 2")
plt.plot(best_piecewise["x3"], best_piecewise["yhat3"], linewidth=2, label="Piecewise segment 3")

# optional oracle validation
plt.errorbar(
    oracle_df["N"], oracle_df["RMSE_mean"], yerr=oracle_df["RMSE_std"],
    fmt='s', alpha=0.7, capsize=3, label="Oracle curve (validation only)"
)

# efficient window
plt.axvspan(N_start, N_end, alpha=0.15, label="Efficient sample window")
plt.axvline(N_start, linestyle=':', label="Window start")
plt.axvline(N_end, linestyle=':', label="Window end")

plt.xlabel("Sample size (N)")
plt.ylabel("RMSE")
plt.title("TabPFN-guided Data Utility Curve with Three-Segment Window Detection")
plt.legend()
plt.tight_layout()
plt.savefig("tabpfn_piecewise_window.png", dpi=300)
plt.show()


# ==============================
# 12. Interpretation
# ==============================
print("\n===== Interpretation =====")
print(f"Using only the initial {INITIAL_N} samples, the efficient sample window is estimated as {N_start}–{N_end}.")
print("The first breakpoint marks the transition from rapid error reduction to diminishing returns.")
print("The second breakpoint marks the transition from diminishing returns to near-saturation.")
print("Use oracle_full_pool_utility_curve.csv only for retrospective validation.")
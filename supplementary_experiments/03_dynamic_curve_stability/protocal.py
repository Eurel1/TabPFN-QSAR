# ==============================
# TabPFN-guided Data Utility Curve
# Predicted DUC + True DUC
# Both curves use smooth fitting + 3-segment piecewise regression
# Piecewise lines removed from plots; only breakpoints retained
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
RANDOM_STATE = 41
INITIAL_N = 200
TEST_SIZE = 0.2
REPEATS = 8
SUBSET_SIZES = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

MAX_SEARCH_N = 1000   # extrapolation upper bound for predicted curve

# You can densify this if you want a smoother oracle DUC
ORACLE_EXTRA_SIZES = [250, 300, 400, 500, 600, 700, 800]
# Example for denser oracle curve:
# ORACLE_EXTRA_SIZES = list(range(220, 801, 20))

# Academic plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

# ==============================
# 0.1 Color settings
# ==============================
MAIN_COLOR = "#E69F00"    # main fitted curve: orange
BP_COLOR = "#D55E00"      # breakpoint lines and markers: dark orange
POINT_COLOR = "#E69F00"   # observed points: orange


# ==============================
# 1. Read data and preprocess
# ==============================
df = pd.read_excel("categorized_maccs.xlsx")

X = df.filter(regex=r"MACCS_\d+").copy()
y = df["logkOH•"].copy()

constant_cols = X.columns[X.nunique() == 1]
X = X.drop(columns=constant_cols)
stratify_column = df['chemical class-98']

print(f"Original dataset size: {len(X)}")
print(f"Feature dimension after removing constant columns: {X.shape[1]}")


# ==============================
# 2. Fixed train/test split
# ==============================
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_column
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
# 4. Utility curve candidate functions
# ==============================
def exp_decay(n, a, b, c):
    return a * np.exp(-b * n) + c

def power_law(n, a, b, c):
    return a * np.power(n, -b) + c


# ==============================
# 5. General helper functions
# ==============================
def evaluate_utility_curve(X_source, y_source, X_test, y_test, subset_sizes, repeats, seed_base):
    """
    Evaluate RMSE-based DUC by repeated random subsampling.
    """
    results = []

    for n in subset_sizes:
        rmse_list = []

        for rep in range(repeats):
            rep_rng = np.random.default_rng(seed_base + rep + n)
            subset_idx = rep_rng.choice(len(X_source), size=n, replace=False)

            X_sub = X_source.iloc[subset_idx]
            y_sub = y_source.iloc[subset_idx]

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

    return pd.DataFrame(results)


def fit_smooth_curve(x_obs, y_obs):
    """
    Fit exponential and power-law curves and choose the one with smaller SSE.
    """
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
        fit_candidates.append(("exponential", exp_params, exp_sse, exp_decay))
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
        fit_candidates.append(("power_law", pow_params, pow_sse, power_law))
    except Exception as e:
        print("Power-law fit failed:", e)

    if not fit_candidates:
        raise RuntimeError("No smooth fit succeeded.")

    best_name, best_params, best_sse, best_func = sorted(fit_candidates, key=lambda x: x[2])[0]
    return best_name, best_params, best_sse, best_func


def fit_line(x, y):
    coef = np.polyfit(x, y, deg=1)
    y_hat = np.polyval(coef, x)
    sse = np.sum((y - y_hat) ** 2)
    return coef, y_hat, sse


def three_segment_piecewise(x_grid, y_grid, min_points=20):
    """
    Search two breakpoints that minimize total SSE of three linear segments.
    The fitted segment lines are not plotted, but breakpoints are retained.
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

    bp1 = int(x_grid[best_piecewise["i"]])
    bp2 = int(x_grid[best_piecewise["j"]])

    return best_piecewise, bp1, bp2


# ==============================
# 6. Predicted DUC from initial 200
# ==============================
results_df = evaluate_utility_curve(
    X_source=X_init,
    y_source=y_init,
    X_test=X_test,
    y_test=y_test,
    subset_sizes=SUBSET_SIZES,
    repeats=REPEATS,
    seed_base=RANDOM_STATE
)

print("\nObserved utility curve from initial low-data regime:")
print(results_df)

x_pred_obs = results_df["N"].values.astype(float)
y_pred_obs = results_df["RMSE_mean"].values.astype(float)

pred_fit_name, pred_fit_params, pred_fit_sse, pred_fit_func = fit_smooth_curve(x_pred_obs, y_pred_obs)

n_grid_pred = np.arange(min(SUBSET_SIZES), MAX_SEARCH_N + 1)
rmse_grid_pred = pred_fit_func(n_grid_pred, *pred_fit_params)

pred_piecewise, predicted_bp1, predicted_bp2 = three_segment_piecewise(
    n_grid_pred, rmse_grid_pred, min_points=20
)

print("\nPredicted curve fitting result:")
print(f"Best smooth fit: {pred_fit_name}")
print(f"Parameters: {pred_fit_params}")
print(f"Predicted breakpoint 1 = {predicted_bp1}")
print(f"Predicted breakpoint 2 = {predicted_bp2}")


# ==============================
# 7. Oracle true DUC from full training pool
# ==============================
oracle_subset_sizes = sorted(set(SUBSET_SIZES + ORACLE_EXTRA_SIZES))
oracle_subset_sizes = [n for n in oracle_subset_sizes if n <= len(X_train_pool)]

oracle_df = evaluate_utility_curve(
    X_source=X_train_pool,
    y_source=y_train_pool,
    X_test=X_test,
    y_test=y_test,
    subset_sizes=oracle_subset_sizes,
    repeats=REPEATS,
    seed_base=1000
)

print("\nOracle utility curve from full training pool:")
print(oracle_df)

x_oracle_obs = oracle_df["N"].values.astype(float)
y_oracle_obs = oracle_df["RMSE_mean"].values.astype(float)

oracle_fit_name, oracle_fit_params, oracle_fit_sse, oracle_fit_func = fit_smooth_curve(
    x_oracle_obs, y_oracle_obs
)

# only fit within the observed oracle range
n_grid_oracle = np.arange(int(min(oracle_subset_sizes)), int(max(oracle_subset_sizes)) + 1)
rmse_grid_oracle = oracle_fit_func(n_grid_oracle, *oracle_fit_params)

# automatically adapt min_points to avoid failure when oracle grid is short
oracle_min_points = max(3, min(10, len(n_grid_oracle) // 6))

oracle_piecewise, oracle_bp1, oracle_bp2 = three_segment_piecewise(
    n_grid_oracle, rmse_grid_oracle, min_points=oracle_min_points
)

print("\nOracle curve fitting result:")
print(f"Best smooth fit: {oracle_fit_name}")
print(f"Parameters: {oracle_fit_params}")
print(f"Oracle breakpoint 1 = {oracle_bp1}")
print(f"Oracle breakpoint 2 = {oracle_bp2}")


# ==============================
# 8. Save outputs
# ==============================
results_df.to_csv("observed_initial200_utility_curve.csv", index=False)
oracle_df.to_csv("oracle_full_pool_utility_curve.csv", index=False)

summary_df = pd.DataFrame([{
    "predicted_best_smooth_fit": pred_fit_name,
    "predicted_param_a": pred_fit_params[0],
    "predicted_param_b": pred_fit_params[1],
    "predicted_param_c": pred_fit_params[2],
    "predicted_breakpoint_1": predicted_bp1,
    "predicted_breakpoint_2": predicted_bp2,
    "oracle_best_smooth_fit": oracle_fit_name,
    "oracle_param_a": oracle_fit_params[0],
    "oracle_param_b": oracle_fit_params[1],
    "oracle_param_c": oracle_fit_params[2],
    "oracle_breakpoint_1": oracle_bp1,
    "oracle_breakpoint_2": oracle_bp2
}])

summary_df.to_csv("duc_breakpoint_summary.csv", index=False)


# ==============================
# 9. Figure 1: Predicted DUC
# ==============================
fig, ax = plt.subplots(figsize=(6.8, 5.2))

# observed points
ax.scatter(
    results_df["N"], results_df["RMSE_mean"],
    s=28, alpha=0.35, color=POINT_COLOR,
    label="Observed data", zorder=2
)

# fitted + extrapolated curve
obs_max = max(SUBSET_SIZES)
mask_obs = n_grid_pred <= obs_max
mask_ext = n_grid_pred > obs_max

ax.plot(
    n_grid_pred[mask_obs], rmse_grid_pred[mask_obs],
    color=MAIN_COLOR, linewidth=3.0,
    label=f"Fitted {pred_fit_name} curve", zorder=4
)

ax.plot(
    n_grid_pred[mask_ext], rmse_grid_pred[mask_ext],
    color=MAIN_COLOR, linewidth=3.0, linestyle='--',
    label="Extrapolated curve", zorder=4
)

# breakpoints only
ax.axvline(
    predicted_bp1, linestyle=':', linewidth=1.6, color=BP_COLOR,
    label=f"Breakpoint 1 (N={predicted_bp1})", zorder=1
)
ax.axvline(
    predicted_bp2, linestyle='-.', linewidth=1.6, color=BP_COLOR,
    label=f"Breakpoint 2 (N={predicted_bp2})", zorder=1
)

# breakpoint markers on fitted curve
ax.scatter(
    [predicted_bp1, predicted_bp2],
    [
        pred_fit_func(predicted_bp1, *pred_fit_params),
        pred_fit_func(predicted_bp2, *pred_fit_params)
    ],
    s=50, color=BP_COLOR, zorder=5
)

ax.set_xlabel("Sample size (N)")
ax.set_ylabel("RMSE")
ax.set_title("Predicted Data Utility Curve")
ax.tick_params(direction='in', length=5, width=1)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig("Figure_predicted_data_utility_curve.png", bbox_inches="tight")
plt.show()


# ==============================
# 10. Figure 2: True DUC
# ==============================
fig, ax = plt.subplots(figsize=(6.8, 5.2))

# oracle observed points
ax.scatter(
    oracle_df["N"], oracle_df["RMSE_mean"],
    s=28, alpha=0.35, color=POINT_COLOR,
    label="Observed data", zorder=2
)

# fitted oracle curve
ax.plot(
    n_grid_oracle, rmse_grid_oracle,
    color=MAIN_COLOR, linewidth=3.0,
    label=f"Fitted {oracle_fit_name} curve", zorder=4
)

# breakpoints only
ax.axvline(
    oracle_bp1, linestyle=':', linewidth=1.6, color=BP_COLOR,
    label=f"Breakpoint 1 (N={oracle_bp1})", zorder=1
)
ax.axvline(
    oracle_bp2, linestyle='-.', linewidth=1.6, color=BP_COLOR,
    label=f"Breakpoint 2 (N={oracle_bp2})", zorder=1
)

# breakpoint markers
ax.scatter(
    [oracle_bp1, oracle_bp2],
    [
        oracle_fit_func(oracle_bp1, *oracle_fit_params),
        oracle_fit_func(oracle_bp2, *oracle_fit_params)
    ],
    s=50, color=BP_COLOR, zorder=5
)

ax.set_xlabel("Sample size (N)")
ax.set_ylabel("RMSE")
ax.set_title("True Data Utility Curve")
ax.tick_params(direction='in', length=5, width=1)
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig("Figure_true_data_utility_curve.png", bbox_inches="tight")
plt.show()


# ==============================
# 11. Interpretation
# ==============================
print("\n===== Interpretation =====")
print(f"Predicted DUC breakpoints: {predicted_bp1}, {predicted_bp2}")
print(f"True DUC breakpoints: {oracle_bp1}, {oracle_bp2}")
print("The predicted curve was inferred using only the initial low-data regime.")
print("The true curve was obtained retrospectively from the full training pool.")
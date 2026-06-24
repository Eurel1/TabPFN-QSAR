# ==============================
# TabPFN Supplement Script for Morgan 1024 & 2048
# Runs ONLY TabPFN (seeds 40-70), appends to existing benchmark results
# Run with: tabpfgen conda environment
# ==============================

import numpy as np
import pandas as pd
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

seeds = range(40, 71)

# ==============================
# 1. Morgan 1024
# ==============================
print("=" * 60)
print("Morgan 1024: Running TabPFN (31 seeds)...")
print("=" * 60)

df1024 = pd.read_excel("morgan_1024.xlsx")
X1024 = df1024.filter(regex=r"Morgan_\d+")
y1024 = df1024["logkOH•"]

# Drop constant features
constant_cols = X1024.columns[X1024.nunique() == 1]
X1024 = X1024.drop(columns=constant_cols)
print(f"Features: {X1024.shape[1]}")

tabpfn_results_1024 = []
for seed in seeds:
    print(f"  Seed: {seed}...", end="\r")
    X_train, X_test, y_train, y_test = train_test_split(
        X1024, y1024, test_size=0.2, random_state=seed
    )
    model = AutoTabPFNRegressor(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tabpfn_results_1024.append({
        "Seed": seed,
        "Model": "TabPFN",
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    })

# Append to existing details CSV
details_path_1024 = "morgan1024_benchmark_details.csv"
if os.path.exists(details_path_1024):
    existing_1024 = pd.read_csv(details_path_1024)
    # Remove old TabPFN rows if any
    existing_1024 = existing_1024[existing_1024["Model"] != "TabPFN"]
    combined_1024 = pd.concat([existing_1024, pd.DataFrame(tabpfn_results_1024)], ignore_index=True)
else:
    combined_1024 = pd.DataFrame(tabpfn_results_1024)
combined_1024.to_csv(details_path_1024, index=False)
print(f"\nMorgan 1024 details saved: {len(combined_1024)} rows")

# Regenerate summary
summary_1024 = combined_1024.groupby("Model").agg(
    RMSE=("RMSE", ["mean", "std"]),
    MAE=("MAE", ["mean", "std"]),
    R2=("R2", ["mean", "std"])
).reset_index()
summary_1024.columns = ["Model", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"]
# Save in compatible format with original
combined_1024.groupby("Model").agg({
    "RMSE": ["mean", "std"],
    "MAE": ["mean", "std"],
    "R2": ["mean", "std"]
}).to_csv("morgan1024_benchmark_summary.csv")
print("Morgan 1024 summary regenerated.\n")

# ==============================
# 2. Morgan 2048
# ==============================
print("=" * 60)
print("Morgan 2048: Running TabPFN (31 seeds)...")
print("=" * 60)

df2048 = pd.read_excel("morgan_2048.xlsx")
X2048 = df2048.filter(regex=r"Morgan_\d+")
y2048 = df2048["logkOH•"]

constant_cols = X2048.columns[X2048.nunique() == 1]
X2048 = X2048.drop(columns=constant_cols)
print(f"Features: {X2048.shape[1]}")

tabpfn_results_2048 = []
for seed in seeds:
    print(f"  Seed: {seed}...", end="\r")
    X_train, X_test, y_train, y_test = train_test_split(
        X2048, y2048, test_size=0.2, random_state=seed
    )
    model = AutoTabPFNRegressor(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tabpfn_results_2048.append({
        "Seed": seed,
        "Model": "TabPFN",
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    })

# Append to existing details CSV (or create new)
details_path_2048 = "morgan2048_benchmark_details.csv"
if os.path.exists(details_path_2048):
    existing_2048 = pd.read_csv(details_path_2048)
    existing_2048 = existing_2048[existing_2048["Model"] != "TabPFN"]
    combined_2048 = pd.concat([existing_2048, pd.DataFrame(tabpfn_results_2048)], ignore_index=True)
else:
    combined_2048 = pd.DataFrame(tabpfn_results_2048)
    print("  (Details CSV did not exist; created with TabPFN-only results)")
combined_2048.to_csv(details_path_2048, index=False)
print(f"\nMorgan 2048 details saved: {len(combined_2048)} rows")

# Regenerate summary
combined_2048.groupby("Model").agg({
    "RMSE": ["mean", "std"],
    "MAE": ["mean", "std"],
    "R2": ["mean", "std"]
}).to_csv("morgan2048_benchmark_summary.csv")
print("Morgan 2048 summary regenerated.\n")

# ==============================
# 3. Quick stats
# ==============================
print("=" * 60)
print("TabPFN Summary:")
print("=" * 60)
for label, res in [("Morgan 1024", tabpfn_results_1024), ("Morgan 2048", tabpfn_results_2048)]:
    df_r = pd.DataFrame(res)
    print(f"\n{label}:")
    print(f"  RMSE: {df_r['RMSE'].mean():.4f} +/- {df_r['RMSE'].std():.4f}")
    print(f"  MAE:  {df_r['MAE'].mean():.4f} +/- {df_r['MAE'].std():.4f}")
    print(f"  R2:   {df_r['R2'].mean():.4f} +/- {df_r['R2'].std():.4f}")

print("\nDone! TabPFN results appended to benchmark CSV files.")

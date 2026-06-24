# ==============================
# QSAR Benchmark Pipeline (Multi-Seed Version)
# Descriptor: PCA-Reduced Morgan Fingerprints (64, 128, 256 dims)
# Run with: tabpfgen conda environment
#   E:\miniconda\envs\tabpfgen\python.exe model_compare_morgan_dimreduced.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from hyperopt import fmin, tpe, hp, Trials

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# ==============================
# Configuration
# ==============================
PCA_DIMS = [256]
seeds = range(40, 71)
TUNING_EVALS = 30  # Reduce tuning evals for faster total runtime

for n_dim in PCA_DIMS:
    print("=" * 60)
    print(f"PCA {n_dim}: Loading data...")
    print("=" * 60)

    # ==============================
    # 1. Load data
    # ==============================
    fname = f"morgan_pca{n_dim}.xlsx"
    df = pd.read_excel(fname)
    y_col = [c for c in df.columns if "logkOH" in c][0]
    X = df.filter(regex=r"PCA_\d+")
    y = df[y_col]

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Fixed split for tuning
    X_train_init, X_temp, y_train_init, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val_init, X_test_init, y_val_init, y_test_init = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # ==============================
    # 2. Hyperparameter Tuning
    # ==============================
    best_params_dict = {}
    print("--- Starting Hyperparameter Optimization ---")

    # --- XGBoost Tuning ---
    space_xgb = {
        "n_estimators": hp.choice("n_estimators", [50, 100, 200, 300]),
        "max_depth": hp.choice("max_depth", [3, 5, 8, 12]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "subsample": hp.uniform("subsample", 0.6, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    }

    def objective_xgb(params):
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train_init, y_train_init)
        return np.sqrt(mean_squared_error(y_val_init, model.predict(X_val_init)))

    best_xgb_raw = fmin(fn=objective_xgb, space=space_xgb, algo=tpe.suggest, max_evals=TUNING_EVALS)
    best_params_dict["XGBoost"] = {
        "n_estimators": [50, 100, 200, 300][best_xgb_raw["n_estimators"]],
        "max_depth": [3, 5, 8, 12][best_xgb_raw["max_depth"]],
        "learning_rate": best_xgb_raw["learning_rate"],
        "subsample": best_xgb_raw["subsample"],
        "colsample_bytree": best_xgb_raw["colsample_bytree"],
    }

    # --- CatBoost Tuning ---
    space_cat = {
        "iterations": hp.choice("iterations", [100, 200, 300]),
        "depth": hp.choice("depth", [4, 6, 8]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 10),
    }

    def objective_cat(params):
        model = CatBoostRegressor(**params, verbose=False, random_seed=42)
        model.fit(X_train_init, y_train_init)
        return np.sqrt(mean_squared_error(y_val_init, model.predict(X_val_init)))

    best_cat_raw = fmin(fn=objective_cat, space=space_cat, algo=tpe.suggest, max_evals=TUNING_EVALS)
    best_params_dict["CatBoost"] = {
        "iterations": [100, 200, 300][best_cat_raw["iterations"]],
        "depth": [4, 6, 8][best_cat_raw["depth"]],
        "learning_rate": best_cat_raw["learning_rate"],
        "l2_leaf_reg": best_cat_raw["l2_leaf_reg"],
    }

    # --- SVR Tuning ---
    # PCA features are already standardized (near-zero mean), so use SimpleImputer only
    space_svr = {
        "C": hp.uniform("C", 0.1, 100),
        "epsilon": hp.uniform("epsilon", 0.01, 1),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(1)),
    }

    def objective_svr(params):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR(**params)),
        ])
        model.fit(X_train_init, y_train_init)
        return np.sqrt(mean_squared_error(y_val_init, model.predict(X_val_init)))

    best_params_dict["SVR"] = fmin(fn=objective_svr, space=space_svr, algo=tpe.suggest, max_evals=TUNING_EVALS)

    # --- MLP Tuning ---
    space_mlp = {
        "hidden_layer_sizes": hp.choice("hidden_layer_sizes", [(100,), (100, 50), (200, 100)]),
        "alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e-2)),
        "learning_rate_init": hp.uniform("learning_rate_init", 0.0005, 0.05),
    }

    def objective_mlp(params):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**params, max_iter=500, random_state=42)),
        ])
        model.fit(X_train_init, y_train_init)
        return np.sqrt(mean_squared_error(y_val_init, model.predict(X_val_init)))

    best_mlp_raw = fmin(fn=objective_mlp, space=space_mlp, algo=tpe.suggest, max_evals=TUNING_EVALS)
    best_params_dict["MLP"] = {
        "hidden_layer_sizes": [(100,), (100, 50), (200, 100)][best_mlp_raw["hidden_layer_sizes"]],
        "alpha": best_mlp_raw["alpha"],
        "learning_rate_init": best_mlp_raw["learning_rate_init"],
    }

    # --- Ridge Tuning ---
    space_ridge = {"alpha": hp.loguniform("alpha", np.log(0.001), np.log(100))}

    def objective_ridge(params):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(**params)),
        ])
        model.fit(X_train_init, y_train_init)
        return np.sqrt(mean_squared_error(y_val_init, model.predict(X_val_init)))

    best_params_dict["Ridge"] = fmin(fn=objective_ridge, space=space_ridge, algo=tpe.suggest, max_evals=TUNING_EVALS)

    print("Optimization Complete.\n")

    # ==============================
    # 3. Multi-Seed Benchmark (40-70)
    # ==============================
    all_results = []
    print(f"--- Starting Multi-Seed Benchmark (Seeds {seeds.start}-{seeds.stop - 1}) ---")

    for seed in seeds:
        print(f"  Seed: {seed}...", end="\r")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        models = {
            "TabPFN": AutoTabPFNRegressor(random_state=seed),
            "XGBoost": XGBRegressor(**best_params_dict["XGBoost"], random_state=seed),
            "CatBoost": CatBoostRegressor(**best_params_dict["CatBoost"], verbose=False, random_seed=seed),
            "SVR": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svr", SVR(**best_params_dict["SVR"])),
            ]),
            "MLP": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(**best_params_dict["MLP"], max_iter=500, random_state=seed)),
            ]),
            "Ridge": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(**best_params_dict["Ridge"])),
            ]),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_results.append({
                "Seed": seed,
                "Model": name,
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            })

    # ==============================
    # 4. Summarize & Save
    # ==============================
    results_df = pd.DataFrame(all_results)

    summary_df = results_df.groupby("Model").agg({
        "RMSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "R2": ["mean", "std"],
    }).reset_index()

    print(f"\n\nPCA {n_dim} - Final Benchmark Summary (Mean +/- Std):")
    print(summary_df.to_string())

    prefix = f"morgan_pca{n_dim}"
    results_df.to_csv(f"{prefix}_benchmark_details.csv", index=False)
    summary_df.to_csv(f"{prefix}_benchmark_summary.csv", index=False)
    pd.DataFrame(best_params_dict).T.to_csv(f"{prefix}_best_params.csv")

    print(f"\nPCA {n_dim} completed. Results saved to {prefix}_*.csv\n")

print("\n" + "=" * 60)
print("All PCA dimension benchmarks complete!")
print("=" * 60)

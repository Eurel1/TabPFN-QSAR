# ==============================
# Reproducible Missing Data Benchmark
# TabPFN vs AutoGluon vs XGBoost
# Multi-seed version
# ==============================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

from xgboost import XGBRegressor

from autogluon.tabular import TabularPredictor

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor


# ==============================
# Seeds for repeated experiments
# ==============================

SEEDS = list(range(40, 71))   # 40-70


# ==============================
# Load Dataset
# ==============================

maccs_df = pd.read_excel("categorized_maccs.xlsx")

X = maccs_df.filter(regex="MACCS_\d+")
y = maccs_df["logkOH•"]

constant_cols = X.columns[X.nunique() == 1]
X = X.drop(columns=constant_cols)


# ==============================
# Missing Data Generator
# ==============================

def create_missing_data(X, missing_rate, seed):

    rng = np.random.default_rng(seed)

    mask = rng.random(X.shape) < missing_rate

    X_missing = X.copy()
    X_missing[mask] = np.nan

    return X_missing


# ==============================
# XGBoost + MICE
# ==============================

def run_xgb_mice(X_train_missing, X_test_missing, y_train, y_test, seed):

    imputer = IterativeImputer(
        random_state=seed,
        max_iter=10
    )

    X_train_imp = imputer.fit_transform(X_train_missing)
    X_test_imp = imputer.transform(X_test_missing)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train_imp, y_train)

    pred = model.predict(X_test_imp)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return rmse


# ==============================
# XGBoost + missForest
# ==============================

def run_xgb_missforest(X_train_missing, X_test_missing, y_train, y_test, seed):

    rf_estimator = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1
    )

    imputer = IterativeImputer(
        estimator=rf_estimator,
        max_iter=10,
        random_state=seed
    )

    X_train_imp = imputer.fit_transform(X_train_missing)
    X_test_imp = imputer.transform(X_test_missing)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train_imp, y_train)

    pred = model.predict(X_test_imp)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return rmse


# ==============================
# TabPFN
# ==============================

def run_tabpfn(X_train_missing, X_test_missing, y_train, y_test, seed):

    model = AutoTabPFNRegressor(random_state=seed)

    model.fit(X_train_missing, y_train)

    pred = model.predict(X_test_missing)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return rmse


# ==============================
# AutoGluon
# ==============================

def run_autogluon(X_train_missing, X_test_missing, y_train, y_test):

    train_df = X_train_missing.copy()
    train_df["target"] = y_train

    predictor = TabularPredictor(
        label="target",
        problem_type="regression",
        eval_metric="rmse"
    ).fit(
        train_df,
        presets="medium_quality",
        verbosity=0
    )

    test_df = X_test_missing.copy()

    pred = predictor.predict(test_df)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return rmse


# ==============================
# Missing Rates
# ==============================

missing_rates = [0.0,0.1,0.2,0.3,0.4,0.5]


# ==============================
# Main Experiment
# ==============================

results = []

for seed in SEEDS:

    print(f"\nRunning seed {seed}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    for rate in missing_rates:

        print(f"  Missing rate {rate}")

        X_train_missing = create_missing_data(X_train, rate, seed)
        X_test_missing = create_missing_data(X_test, rate, seed)

        rmse_tabpfn = run_tabpfn(
            X_train_missing, X_test_missing, y_train, y_test, seed
        )

        rmse_autogluon = run_autogluon(
            X_train_missing, X_test_missing, y_train, y_test
        )

        rmse_mice = run_xgb_mice(
            X_train_missing, X_test_missing, y_train, y_test, seed
        )

        rmse_missforest = run_xgb_missforest(
            X_train_missing, X_test_missing, y_train, y_test, seed
        )

        results.append({
            "seed": seed,
            "missing_rate": rate,
            "TabPFN": rmse_tabpfn,
            "AutoGluon": rmse_autogluon,
            "XGB+MICE": rmse_mice,
            "XGB+missForest": rmse_missforest
        })


# ==============================
# Save full results
# ==============================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "missing_benchmark_all_runs.csv",
    index=False
)

print(results_df)


# ==============================
# Compute mean and std
# ==============================

summary = results_df.groupby("missing_rate").agg(["mean","std"])

summary.to_csv("missing_benchmark_summary.csv")

print(summary)
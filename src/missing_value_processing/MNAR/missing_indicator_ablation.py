import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator
from sklearn.metrics import mean_squared_error, r2_score

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor


# ==============================
# 1 读取数据
# ==============================
df = pd.read_excel("processed_dataset.xlsx", sheet_name="Sheet1")

le = LabelEncoder()
df['pollutant'] = le.fit_transform(df['pollutant'])

X = df.drop(columns=['pollutant', 'kobs'])
y_category = df['pollutant']
y_reg = df['kobs']


# ==============================
# 2 Missing Indicator
# ==============================
indicator = MissingIndicator(features="all")
X_indicator = indicator.fit_transform(X)

# 拼接原始特征 + indicator
X_with_indicator = np.hstack([X.values, X_indicator])


# ==============================
# 3 结果存储
# ==============================
rmse_base_list = []
r2_base_list = []

rmse_ind_list = []
r2_ind_list = []


# ==============================
# 4 多随机划分
# ==============================
for seed in range(40, 71):

    print(f"\n===== Random Seed {seed} =====")

    # stratified split
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
        X,
        y_reg,
        y_category,
        test_size=0.2,
        stratify=y_category,
        random_state=seed
    )

    X_train_ind, X_test_ind, _, _, _, _ = train_test_split(
        X_with_indicator,
        y_reg,
        y_category,
        test_size=0.2,
        stratify=y_category,
        random_state=seed
    )


    # ==============================
    # 标准化
    # ==============================
    scaler_base = StandardScaler()
    X_train_scaled = scaler_base.fit_transform(X_train)
    X_test_scaled = scaler_base.transform(X_test)

    scaler_ind = StandardScaler()
    X_train_ind_scaled = scaler_ind.fit_transform(X_train_ind)
    X_test_ind_scaled = scaler_ind.transform(X_test_ind)


    # ==============================
    # Baseline AutoTabPFN
    # ==============================
    model_base = AutoTabPFNRegressor(
        max_time=120,
        device='cuda',
        random_state=seed
    )

    model_base.fit(X_train_scaled, y_train)

    y_pred_base = model_base.predict(X_test_scaled)

    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    r2_base = r2_score(y_test, y_pred_base)

    rmse_base_list.append(rmse_base)
    r2_base_list.append(r2_base)


    # ==============================
    # AutoTabPFN + Missing Indicator
    # ==============================
    model_ind = AutoTabPFNRegressor(
        max_time=120,
        device='cuda',
        random_state=seed
    )

    model_ind.fit(X_train_ind_scaled, y_train)

    y_pred_ind = model_ind.predict(X_test_ind_scaled)

    rmse_ind = np.sqrt(mean_squared_error(y_test, y_pred_ind))
    r2_ind = r2_score(y_test, y_pred_ind)

    rmse_ind_list.append(rmse_ind)
    r2_ind_list.append(r2_ind)


# ==============================
# 5 统计结果
# ==============================
print("\n==============================")
print("Final Results (Seeds 40–70)")
print("==============================")

print("\nAutoTabPFN Baseline")
print(f"RMSE: {np.mean(rmse_base_list):.4f} ± {np.std(rmse_base_list):.4f}")
print(f"R²  : {np.mean(r2_base_list):.4f} ± {np.std(r2_base_list):.4f}")

print("\nAutoTabPFN + Missing Indicator")
print(f"RMSE: {np.mean(rmse_ind_list):.4f} ± {np.std(rmse_ind_list):.4f}")
print(f"R²  : {np.mean(r2_ind_list):.4f} ± {np.std(r2_ind_list):.4f}")
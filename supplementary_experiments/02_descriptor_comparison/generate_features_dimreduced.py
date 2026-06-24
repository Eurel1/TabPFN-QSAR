# ==============================
# Generate PCA-Reduced Morgan Fingerprint Features
# Reduces 2048-bit Morgan fingerprints via PCA
# Run with: python generate_features_dimreduced.py
# ==============================

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. Load 2048-bit Morgan fingerprints
# ==============================
df = pd.read_excel("morgan_2048.xlsx")
y_col = [c for c in df.columns if "logkOH" in c][0]
print(f"Target column: {y_col!r}")

X = df.filter(regex=r"Morgan_\d+")
y = df[y_col]

# Drop constant features
constant_cols = X.columns[X.nunique() == 1]
X = X.drop(columns=constant_cols)
print(f"Original features: {X.shape[1]} (dropped {len(constant_cols)} constant)")

# ==============================
# 2. PCA dimensionality reduction
# ==============================
# Standardize first (fingerprints are binary 0/1, but PCA benefits from scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to multiple target dimensions for comparison
pca_dims = [64, 128, 256]

for n_components in pca_dims:
    print(f"\n--- PCA to {n_components} components ---")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Cumulative explained variance: {explained_var:.4f}")

    # Create DataFrame with PCA column names
    pca_cols = [f"PCA_{i+1:04d}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca[y_col] = y.values

    fname = f"morgan_pca{n_components}.xlsx"
    df_pca.to_excel(fname, index=False)
    print(f"  Saved: {fname} ({df_pca.shape[1]} cols)")

print("\n=== PCA feature generation complete. ===")

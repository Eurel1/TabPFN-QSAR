import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. Load data
# ==============================
base = r"C:\Users\寒\Desktop\补充实验"
os.chdir(base)

df = pd.read_excel("original_smiles.xlsx")
smiles_list = df["SMILES"].tolist()
y_col = [c for c in df.columns if "logkOH" in c][0]
print(f"Target column: {y_col!r}")

# ==============================
# 2. Parse molecules (skip invalid SMILES)
# ==============================
mols = []
valid_indices = []
for i, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
        valid_indices.append(i)
    else:
        print(f"  [WARN] Invalid SMILES at row {i}: {smi}")

y_valid = df[y_col].values[valid_indices]
print(f"Valid molecules: {len(mols)} / {len(smiles_list)}")

# ==============================
# 3. RDKit Physicochemical Descriptors
# ==============================
print("\n--- Computing RDKit Physicochemical Descriptors ---")
desc_data = []
for mol in mols:
    row = {}
    for desc_name, desc_func in Descriptors.descList:
        try:
            val = desc_func(mol)
        except Exception:
            val = np.nan
        row[desc_name] = val
    desc_data.append(row)

desc_df = pd.DataFrame(desc_data)
desc_df = desc_df.dropna(axis=1, how="all")
print(f"  Descriptors: {desc_df.shape[1]} features")

out_physchem = desc_df.copy()
out_physchem[y_col] = y_valid
out_physchem.to_excel("rdkit_physchem.xlsx", index=False)
print(f"  Saved: rdkit_physchem.xlsx")

# ==============================
# 4. Morgan Fingerprints (1024-bit)
# ==============================
print("\n--- Computing Morgan 1024-bit Fingerprints ---")
nbits_1024 = 1024
fp_data_1024 = []
for mol in mols:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits_1024)
    arr = np.zeros((1,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_data_1024.append(arr)

fp_df_1024 = pd.DataFrame(
    fp_data_1024,
    columns=[f"Morgan_{i:04d}" for i in range(nbits_1024)]
)
print(f"  Morgan 1024: {fp_df_1024.shape[1]} features")

out_morgan1024 = fp_df_1024.copy()
out_morgan1024[y_col] = y_valid
out_morgan1024.to_excel("morgan_1024.xlsx", index=False)
print(f"  Saved: morgan_1024.xlsx")

# ==============================
# 5. Morgan Fingerprints (2048-bit)
# ==============================
print("\n--- Computing Morgan 2048-bit Fingerprints ---")
nbits_2048 = 2048
fp_data_2048 = []
for mol in mols:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits_2048)
    arr = np.zeros((1,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_data_2048.append(arr)

fp_df_2048 = pd.DataFrame(
    fp_data_2048,
    columns=[f"Morgan_{i:04d}" for i in range(nbits_2048)]
)
print(f"  Morgan 2048: {fp_df_2048.shape[1]} features")

out_morgan2048 = fp_df_2048.copy()
out_morgan2048[y_col] = y_valid
out_morgan2048.to_excel("morgan_2048.xlsx", index=False)
print(f"  Saved: morgan_2048.xlsx")

print("\n=== Feature generation complete. ===")

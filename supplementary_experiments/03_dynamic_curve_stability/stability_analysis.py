# ==============================
# 稳定性分析：动态有效区间 (N_start, N_end)
# 固定train/test分割, 仅变动初始子集种子 40-70
# ==============================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor

TRAIN_TEST_SEED = 42
INITIAL_SEEDS  = list(range(40, 71))
INITIAL_N      = 200
TEST_SIZE      = 0.2
REPEATS        = 8
SUBSET_SIZES   = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
MAX_SEARCH_N   = 1000

# ==============================
# 1. 读取数据
# ==============================
df = pd.read_excel("categorized_maccs.xlsx")
y_col = [c for c in df.columns if "logk" in c][0]
X = df.filter(regex=r"MACCS_\d+").copy()
y = df[y_col].copy()
constant_cols = X.columns[X.nunique() == 1]
X = X.drop(columns=constant_cols)

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=TRAIN_TEST_SEED,
    stratify=df["chemical class-98"]
)
X_train_pool = X_train_pool.reset_index(drop=True)
y_train_pool = y_train_pool.reset_index(drop=True)
X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)

print(f"数据: {len(X)}样本, {X.shape[1]}特征 | 训练池: {len(X_train_pool)}, 测试集: {len(X_test)}")
print(f"种子范围: {INITIAL_SEEDS[0]}-{INITIAL_SEEDS[-1]} (共{len(INITIAL_SEEDS)}个)")
print()

# ==============================
# 2. 曲线函数 & 工具
# ==============================
def exp_decay(n, a, b, c):   return a * np.exp(-b * n) + c
def power_law(n, a, b, c):   return a * np.power(n, -b) + c

def evaluate_duc(X_src, y_src, sizes, reps, seed_base):
    rows = []
    for n in sizes:
        rmses = []
        for rep in range(reps):
            rng = np.random.default_rng(int(seed_base) + rep + n)
            idx = rng.choice(len(X_src), size=n, replace=False)
            m = TabPFNRegressor()
            m.fit(X_src.iloc[idx], y_src.iloc[idx])
            pred = m.predict(X_test)
            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append({"N": n, "RMSE_mean": np.mean(rmses),
                     "RMSE_std": np.std(rmses, ddof=1) if len(rmses)>1 else 0.0})
    return pd.DataFrame(rows)

def fit_curve(x_obs, y_obs):
    cands = []
    for name, fn, p0 in [("exp", exp_decay,
                          [max(y_obs)-min(y_obs), 0.01, min(y_obs)]),
                         ("pow", power_law, [10.0, 0.5, min(y_obs)])]:
        try:
            params, _ = curve_fit(fn, x_obs, y_obs, p0=p0,
                                  bounds=([0,1e-6,0],[np.inf,10,np.inf]), maxfev=20000)
            sse = np.sum((y_obs - fn(x_obs, *params))**2)
            cands.append((name, params, sse, fn))
        except: pass
    if not cands: raise RuntimeError("拟合失败")
    best = sorted(cands, key=lambda x: x[2])[0]
    return best[0], best[1], best[3]

def fit_line(x, y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    A=np.vstack([x,np.ones_like(x)]).T
    s,i=np.linalg.lstsq(A,y,rcond=None)[0]
    yh=s*x+i; return (s,i), yh, np.sum((y-yh)**2)

def three_seg(n_grid, rmse_grid, min_pts=10):
    n_grid=np.asarray(n_grid); rmse_grid=np.asarray(rmse_grid)
    best,best_sse=None,np.inf
    for i in range(min_pts, len(n_grid)-2*min_pts):
        for j in range(i+min_pts, len(n_grid)-min_pts):
            _,_,s1=fit_line(n_grid[:i],rmse_grid[:i])
            _,_,s2=fit_line(n_grid[i:j],rmse_grid[i:j])
            _,_,s3=fit_line(n_grid[j:],rmse_grid[j:])
            if s1+s2+s3 < best_sse:
                best_sse=s1+s2+s3; best={"i":i,"j":j}
    return best["i"], best["j"]

# ==============================
# 3. 主循环
# ==============================
rows = []
for idx, seed in enumerate(INITIAL_SEEDS):
    # 初始子集
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(X_train_pool), size=INITIAL_N, replace=False)
    X_init = X_train_pool.iloc[init_idx].reset_index(drop=True)
    y_init = y_train_pool.iloc[init_idx].reset_index(drop=True)

    # DUC观测
    df_duc = evaluate_duc(X_init, y_init, SUBSET_SIZES, REPEATS, seed)
    xo = df_duc["N"].values.astype(float)
    yo = df_duc["RMSE_mean"].values.astype(float)

    # 平滑拟合 + 外推
    fn, fp, ff = fit_curve(xo, yo)
    ng = np.arange(min(SUBSET_SIZES), MAX_SEARCH_N+1)
    rg = ff(ng, *fp)

    # 三段分段回归
    mp = max(3, min(10, len(ng)//6))
    i, j = three_seg(ng, rg, min_pts=mp)
    bp1, bp2 = int(ng[i]), int(ng[j])

    rows.append({"seed":seed, "fit":fn,
                 "a":fp[0],"b":fp[1],"c":fp[2],
                 "bp1":bp1, "bp2":bp2, "width":bp2-bp1})
    print(f"  [{idx+1:2d}/{len(INITIAL_SEEDS)}] seed={seed:2d}  bp1={bp1:4d}  bp2={bp2:4d}  width={bp2-bp1:4d}")

df = pd.DataFrame(rows)
df.to_csv("stability_all_seeds.csv", index=False)

# ==============================
# 4. 统计汇总
# ==============================
print("\n" + "="*60)
print("稳定性分析结果 (N_start=Bp1, N_end=Bp2)")
print("="*60)
for label, col in [("断点1 (N_start)", "bp1"),
                    ("断点2 (N_end)",   "bp2"),
                    ("有效窗口宽度",     "width")]:
    v = df[col].values; n=len(v)
    m=np.mean(v); s=np.std(v,ddof=1)
    ci=sp_stats.t.interval(0.95, df=n-1, loc=m, scale=s/np.sqrt(n))
    print(f"  {label}:")
    print(f"    Mean±SD = {m:.1f} ± {s:.1f}")
    print(f"    95% CI  = [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"    Range   = [{np.min(v)}, {np.max(v)}] (跨度={np.max(v)-np.min(v)})")
    print(f"    CV      = {100*s/m:.1f}%")
    print()

# 拟合函数统计
from collections import Counter
fc = Counter(df["fit"])
print(f"  拟合函数选择: exp={fc.get('exp',0)}次, pow={fc.get('pow',0)}次")
print(f"\n详细数据: stability_all_seeds.csv")
print("="*60)

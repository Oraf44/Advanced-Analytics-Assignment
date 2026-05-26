"""
CLV Prediction Pipeline v3  — optimised for Spearman correlation
-----------------------------------------------------------------
Diagnostic finding: `frequency` alone achieves Spearman=0.384 (vs model v2=0.395)
meaning we need better feature *combinations*, not just more raw features.

Key changes vs v2:
  1. Interaction & ratio features that capture purchase velocity/cadence
  2. Single global log1p model (ranks ALL customers in one shot — no two-stage)
  3. Three-model ensemble: two-stage-log + global-log + tweedie (Spearman-tuned)
  4. Higher capacity: num_leaves=255, lr=0.02
  5. Regressor trained with MAPE objective (equivalent to log-scale MAE — good for ranking)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

DATA_DIR = "D:/KU Leuven - Academics/Advanced analytics & big data/Assignment 1/data/data"
TX_PATH    = f"{DATA_DIR}/transactions_2016_2017.csv"
TRAIN_PATH = f"{DATA_DIR}/customer_clv_train.csv"
TEST_PATH  = f"{DATA_DIR}/customer_clv_test.csv"
OUT_PATH   = f"{DATA_DIR}/submission.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD
# ──────────────────────────────────────────────────────────────────────────────
print("Loading data...")
tx    = pd.read_csv(TX_PATH, low_memory=False, parse_dates=["order_date", "pack_date"])
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

REF_DATE = tx["order_date"].max() + pd.Timedelta(days=1)   # 2018-01-01
print(f"  Transactions={tx.shape}  Train={train.shape}  Test={test.shape}")
print(f"  Ref={REF_DATE.date()}  zero-rate={(train['revenue_2018_2019']==0).mean():.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\nEngineering features...")

tx["is_returned"]  = tx["returned_to_shop_id"].notna().astype(int)
tx["order_year"]   = tx["order_date"].dt.year
tx["order_month"]  = tx["order_date"].dt.month
tx["order_weekday"]= tx["order_date"].dt.dayofweek

order_agg = (
    tx.groupby(["cust_id", "sale_id", "order_date"])
    .agg(
        order_revenue  = ("sale_revenue",          "sum"),
        order_discount = ("sale_discount_applied", "sum"),
        order_items    = ("prod_id",               "count"),
        order_returned = ("is_returned",           "max"),
        order_year     = ("order_year",            "first"),
    )
    .reset_index()
)

# ── 2a. Core RFM ──────────────────────────────────────────────────────────────
rfm = (
    order_agg.groupby("cust_id")
    .agg(
        recency             = ("order_date",    lambda x: (REF_DATE - x.max()).days),
        frequency           = ("sale_id",       "nunique"),
        monetary            = ("order_revenue", "sum"),
        avg_order_val       = ("order_revenue", "mean"),
        median_order_val    = ("order_revenue", "median"),
        max_order_val       = ("order_revenue", "max"),
        std_order_val       = ("order_revenue", "std"),
        total_items         = ("order_items",   "sum"),
        avg_items_per_order = ("order_items",   "mean"),
    )
    .reset_index()
)

# ── 2b. Temporal features ────────────────────────────────────────────────────
time_feats = (
    order_agg.groupby("cust_id")
    .agg(
        first_purchase_days = ("order_date", lambda x: (REF_DATE - x.min()).days),
        purchase_span_days  = ("order_date", lambda x: (x.max() - x.min()).days),
        n_active_months     = ("order_date", lambda x: x.dt.to_period("M").nunique()),
    )
    .reset_index()
)

# inter-purchase time
ipt = (
    order_agg.sort_values(["cust_id", "order_date"])
    .groupby("cust_id")["order_date"]
    .apply(lambda x: x.diff().dt.days.dropna())
)
ipt_stats = (
    ipt.groupby(level=0)
    .agg(avg_ipt="mean", std_ipt="std", min_ipt="min", max_ipt="max")
    .reset_index()
    .rename(columns={"level_0": "cust_id"})
)

# ── 2c. Year-split + quarterly features ──────────────────────────────────────
order_agg["order_quarter"] = order_agg["order_date"].dt.quarter

for yr in [2016, 2017]:
    mask = order_agg["order_year"] == yr
    sub  = order_agg[mask].groupby("cust_id").agg(
        **{f"revenue_{yr}": ("order_revenue", "sum")},
        **{f"orders_{yr}":  ("sale_id",       "nunique")},
        **{f"avg_val_{yr}": ("order_revenue", "mean")},
        **{f"returns_{yr}": ("order_returned","sum")},
    ).reset_index()
    rfm = rfm.merge(sub, on="cust_id", how="left")

# 2017 quarterly breakdown — captures within-year momentum
for q in [1, 2, 3, 4]:
    mask = (order_agg["order_year"] == 2017) & (order_agg["order_quarter"] == q)
    sub  = order_agg[mask].groupby("cust_id").agg(
        **{f"rev_2017_q{q}":    ("order_revenue", "sum")},
        **{f"orders_2017_q{q}": ("sale_id",       "nunique")},
    ).reset_index()
    rfm = rfm.merge(sub, on="cust_id", how="left")

for col in [c for c in rfm.columns if any(c.startswith(p) for p in
            ["revenue_", "orders_", "avg_val_", "returns_", "rev_2017_q", "orders_2017_q"])]:
    rfm[col] = rfm[col].fillna(0)

rfm["revenue_growth"] = (rfm["revenue_2017"] - rfm["revenue_2016"]) / (rfm["revenue_2016"].abs() + 1)
rfm["orders_growth"]  = (rfm["orders_2017"]  - rfm["orders_2016"])  / (rfm["orders_2016"]  + 1)
rfm["active_in_2017"] = (rfm["orders_2017"] > 0).astype(int)
rfm["active_in_2016"] = (rfm["orders_2016"] > 0).astype(int)
rfm["share_rev_2017"] = rfm["revenue_2017"] / (rfm["monetary"].abs() + 1)

# Within-2017 momentum: second half vs first half
rfm["rev_2017_h2"]       = rfm["rev_2017_q3"] + rfm["rev_2017_q4"]
rfm["rev_2017_h1"]       = rfm["rev_2017_q1"] + rfm["rev_2017_q2"]
rfm["rev_2017_h2h1_ratio"] = rfm["rev_2017_h2"] / (rfm["rev_2017_h1"] + 1)
rfm["orders_2017_h2"]    = rfm["orders_2017_q3"] + rfm["orders_2017_q4"]
rfm["active_q4_2017"]    = (rfm["orders_2017_q4"] > 0).astype(int)

# ── 2d. Recency-windowed features ────────────────────────────────────────────
cutoffs = {90: REF_DATE - pd.Timedelta(days=90),
           180: REF_DATE - pd.Timedelta(days=180),
           365: REF_DATE - pd.Timedelta(days=365)}

window_dfs = []
for days, cutoff in cutoffs.items():
    sub = order_agg[order_agg["order_date"] >= cutoff].groupby("cust_id").agg(
        **{f"revenue_last{days}d": ("order_revenue", "sum")},
        **{f"orders_last{days}d":  ("sale_id",       "nunique")},
        **{f"items_last{days}d":   ("order_items",   "sum")},
        **{f"avg_val_last{days}d": ("order_revenue", "mean")},
    ).reset_index()
    window_dfs.append(sub)

# ── 2e. Discount / return behaviour ──────────────────────────────────────────
disc_ret = (
    order_agg.groupby("cust_id")
    .agg(
        total_discount = ("order_discount", "sum"),
        avg_discount   = ("order_discount", "mean"),
        max_discount   = ("order_discount", "min"),
        return_rate    = ("order_returned", "mean"),
        n_returns      = ("order_returned", "sum"),
    )
    .reset_index()
)
disc_ret["discount_per_item"] = (
    disc_ret["total_discount"]
    / rfm.set_index("cust_id").loc[disc_ret["cust_id"], "total_items"].values
)

# ── 2f. Product preferences ───────────────────────────────────────────────────
prod_agg = (
    tx.groupby("cust_id")
    .agg(
        web_only_rate     = ("prod_web_only", "mean"),
        outlet_rate       = ("prod_outlet",   "mean"),
        n_unique_brands   = ("prod_brand",    "nunique"),
        n_unique_products = ("prod_id",       "nunique"),
        n_unique_sizes    = ("prod_size",     "nunique"),
        n_unique_colors   = ("prod_color",    "nunique"),
        avg_order_weekday = ("order_weekday", "mean"),
        std_order_month   = ("order_month",   "std"),
        avg_order_month   = ("order_month",   "mean"),
        share_q4          = ("order_month",   lambda x: (x >= 10).mean()),
    )
    .reset_index()
)

brand_counts = tx.groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")
brand_totals = brand_counts.groupby("cust_id")["cnt"].sum()
brand_counts["share"] = brand_counts["cnt"] / brand_counts["cust_id"].map(brand_totals)
brand_entropy   = (brand_counts.groupby("cust_id")["share"]
                   .apply(lambda p: -np.sum(p * np.log(p + 1e-12)))
                   .reset_index().rename(columns={"share": "brand_entropy"}))
top_brand_share = (brand_counts.groupby("cust_id")["share"]
                   .max().reset_index().rename(columns={"share": "top_brand_share"}))

cat_counts = tx.groupby(["cust_id","prod_type_1"]).size().reset_index(name="cnt")
cat_totals  = cat_counts.groupby("cust_id")["cnt"].sum()
cat_counts["share"] = cat_counts["cnt"] / cat_counts["cust_id"].map(cat_totals)
cat_pivot = (cat_counts.pivot_table(index="cust_id", columns="prod_type_1", values="share", fill_value=0)
             .reset_index())
cat_pivot.columns = ["cust_id"] + [f"cat_{c}" for c in cat_pivot.columns[1:]]

season_counts = tx.groupby(["cust_id","prod_season"]).size().reset_index(name="cnt")
season_totals  = season_counts.groupby("cust_id")["cnt"].sum()
season_counts["share"] = season_counts["cnt"] / season_counts["cust_id"].map(season_totals)
season_pivot = (season_counts.pivot_table(index="cust_id", columns="prod_season", values="share", fill_value=0)
                .reset_index())
season_pivot.columns = ["cust_id"] + [f"season_{c}" for c in season_pivot.columns[1:]]

avg_price = (
    tx.groupby("cust_id")
    .apply(lambda df: df["sale_revenue"].sum() / len(df) if len(df) > 0 else 0,
           include_groups=False)
    .reset_index().rename(columns={0: "avg_item_price"})
)

# ── 2g. Merge ─────────────────────────────────────────────────────────────────
print("Merging feature tables...")
all_feats = rfm.copy()
for df in ([time_feats, ipt_stats, disc_ret, prod_agg,
            brand_entropy, top_brand_share, cat_pivot, season_pivot, avg_price]
           + window_dfs):
    all_feats = all_feats.merge(df, on="cust_id", how="left")

for days in cutoffs:
    for col in [f"revenue_last{days}d", f"orders_last{days}d",
                f"items_last{days}d",   f"avg_val_last{days}d"]:
        all_feats[col] = all_feats[col].fillna(0)

# ── 2h. Engineered interaction / velocity features ───────────────────────────
# These directly capture what the diagnostic showed drives Spearman
# (frequency=0.384, avg_ipt=0.348, monetary=0.296)

all_feats["monetary_log"]      = np.log1p(all_feats["monetary"].clip(lower=0))
all_feats["frequency_log"]     = np.log1p(all_feats["frequency"])
all_feats["rev_2017_log"]      = np.log1p(all_feats["revenue_2017"].clip(lower=0))
all_feats["rev_last90d_log"]   = np.log1p(all_feats["revenue_last90d"].clip(lower=0))
all_feats["rev_last180d_log"]  = np.log1p(all_feats["revenue_last180d"].clip(lower=0))

# Purchase velocity: orders per month of customer lifetime
all_feats["purchase_rate_monthly"] = (
    all_feats["frequency"] / (all_feats["first_purchase_days"] / 30.0 + 1)
)

# Revenue velocity: total revenue per month of customer lifetime
all_feats["revenue_rate_monthly"] = (
    all_feats["monetary"] / (all_feats["first_purchase_days"] / 30.0 + 1)
)

# Recent velocity: revenue in last 90d per unique order
all_feats["rev_per_recent_order"] = (
    all_feats["revenue_last90d"] / (all_feats["orders_last90d"] + 1)
)

# Frequency × recency signal: frequent buyers who bought recently
all_feats["freq_x_recency_inv"]  = all_feats["frequency"] / (all_feats["recency"] + 1)
all_feats["freq_x_monetary_log"] = all_feats["frequency_log"] * all_feats["monetary_log"]

# 2017 intensity: revenue per order in 2017
all_feats["rev17_per_order"]  = all_feats["revenue_2017"] / (all_feats["orders_2017"] + 1)
all_feats["rev17_per_month"]  = all_feats["revenue_2017"] / 12.0  # 2017 has 12 months

# Recency normalised by inter-purchase time (customer's natural rhythm)
all_feats["recency_ipt_ratio"] = all_feats["recency"] / (all_feats["avg_ipt"] + 1)

# Share of revenue in last windows
all_feats["rev_last90d_share"]  = all_feats["revenue_last90d"]  / (all_feats["monetary"].abs() + 1)
all_feats["rev_last180d_share"] = all_feats["revenue_last180d"] / (all_feats["monetary"].abs() + 1)
all_feats["rev_last365d_share"] = all_feats["revenue_last365d"] / (all_feats["monetary"].abs() + 1)

# Discount intensity
all_feats["discount_rate"]      = all_feats["total_discount"].abs() / (all_feats["monetary"].abs() + 1)

# Orders per unique brand (loyalty)
all_feats["orders_per_brand"]   = all_feats["frequency"] / (all_feats["n_unique_brands"] + 1)

# ── Extra interaction / derived features around the dominant signals ──────────
# recency_ipt_ratio variants
ript = all_feats["recency_ipt_ratio"].clip(lower=0)
all_feats["log_recency_ipt"]    = np.log1p(ript)
all_feats["ript_sq"]            = ript ** 2
# "Has the customer skipped 1 / 2 / 3 full purchase cycles?"
all_feats["skipped_1_cycle"]    = (ript >= 1).astype(int)
all_feats["skipped_2_cycles"]   = (ript >= 2).astype(int)
all_feats["skipped_3_cycles"]   = (ript >= 3).astype(int)

# 2017 quarterly momentum features
all_feats["rev_2017_q4_log"]    = np.log1p(all_feats["rev_2017_q4"].clip(lower=0))
all_feats["active_q4_2017"]     = all_feats["active_q4_2017"]
all_feats["q4_to_q3_rev_ratio"] = all_feats["rev_2017_q4"] / (all_feats["rev_2017_q3"] + 1)

# Purchase intensity in most recent 90 days vs avg expectation
all_feats["recent_vs_expected"] = (
    all_feats["orders_last90d"] / (90 / (all_feats["avg_ipt"] + 1) + 0.01)
)

# Product-weighted signals: high-value items × recency
all_feats["max_val_x_freq_inv_rec"] = (
    all_feats["max_order_val"] * all_feats["frequency"] / (all_feats["recency"] + 1)
)

# Customer age in months (normalised frequency)
all_feats["freq_per_active_month"] = all_feats["frequency"] / (all_feats["n_active_months"] + 1)

print(f"  Feature matrix: {all_feats.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  MERGE WITH LABELS & SPLIT
# ──────────────────────────────────────────────────────────────────────────────
train_feat = train.merge(all_feats, on="cust_id", how="left")
test_feat  = test.merge(all_feats,  on="cust_id", how="left")

FEATURE_COLS = [c for c in all_feats.columns if c != "cust_id"]
print(f"  Total features: {len(FEATURE_COLS)}")

X      = train_feat[FEATURE_COLS].copy()
y      = train_feat["revenue_2018_2019"].values
X_test = test_feat[FEATURE_COLS].copy()

print("\nSplitting 80/20 (stratified by zero/nonzero)...")
from sklearn.model_selection import StratifiedShuffleSplit, KFold
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, val_idx = next(sss.split(X, (y > 0).astype(int)))
X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
y_tr, y_val = y[tr_idx], y[val_idx]
print(f"  Train={len(y_tr)} (pos={( y_tr>0).sum()})  Val={len(y_val)} (pos={(y_val>0).sum()})")

# ── 3a. Out-of-fold target encodings ─────────────────────────────────────────
print("  Adding OOF target encodings...")

# Get each customer's top brand, top_cat, top brand in 2017, top fine-grain category
top_brand_map = (
    tx.groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_brand"]
    .rename("top_brand")
)
# Top brand purchased ONLY in 2017 (most recent activity signal)
top_brand_2017_map = (
    tx[tx["order_year"] == 2017]
    .groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_brand"]
    .rename("top_brand_2017")
)
top_cat_map = (
    tx.groupby(["cust_id","prod_type_1"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_type_1"]
    .rename("top_cat")
)
# Fine-grained category (prod_type_3 — e.g. "high-top sneakers")
tx["prod_type_3_filled"] = tx["prod_type_3"].fillna("unknown")
top_cat3_map = (
    tx.groupby(["cust_id","prod_type_3_filled"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_type_3_filled"]
    .rename("top_cat3")
)
# Customer acquisition quarter cohort (first purchase year-quarter)
first_date_map = order_agg.groupby("cust_id")["order_date"].min()
cohort_map = (first_date_map.dt.to_period("Q").astype(str)).rename("cohort_q")

train_feat2 = (
    train_feat[["cust_id"]]
    .join(top_brand_map,      on="cust_id")
    .join(top_brand_2017_map, on="cust_id")
    .join(top_cat_map,        on="cust_id")
    .join(top_cat3_map,       on="cust_id")
    .join(cohort_map,         on="cust_id")
)
train_feat2["y_log"] = np.log1p(y.clip(min=0))

te_cols = ["top_brand", "top_brand_2017", "top_cat", "top_cat3", "cohort_q"]
kf      = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = train_feat2["y_log"].mean()

te_arrays = {c: np.zeros(len(y)) for c in te_cols}
te_means  = {}

for fold_tr, fold_val in kf.split(train_feat2):
    for col in te_cols:
        means = train_feat2.iloc[fold_tr].groupby(col)["y_log"].mean()
        te_arrays[col][fold_val] = train_feat2.iloc[fold_val][col].map(means).fillna(global_mean).values

# Full-data means for test set
test_feat2 = (
    test_feat[["cust_id"]]
    .join(top_brand_map,      on="cust_id")
    .join(top_brand_2017_map, on="cust_id")
    .join(top_cat_map,        on="cust_id")
    .join(top_cat3_map,       on="cust_id")
    .join(cohort_map,         on="cust_id")
)
te_test_arrays = {}
for col in te_cols:
    full_means = train_feat2.groupby(col)["y_log"].mean()
    te_means[col] = full_means
    te_test_arrays[col] = test_feat2[col].map(full_means).fillna(global_mean).values

# Append to feature matrices
for col in te_cols:
    X[f"te_{col}"]      = te_arrays[col]
    X_test[f"te_{col}"] = te_test_arrays[col]

FEATURE_COLS = list(X.columns)

# Rebuild train/val splits with new features
X_tr  = X.iloc[tr_idx]
X_val = X.iloc[val_idx]

# ──────────────────────────────────────────────────────────────────────────────
# 4.  SHARED LGBM HELPER
# ──────────────────────────────────────────────────────────────────────────────
BASE = dict(
    learning_rate    = 0.02,
    num_leaves       = 255,
    min_child_samples= 20,
    feature_fraction = 0.7,
    bagging_fraction = 0.7,
    bagging_freq     = 5,
    lambda_l1        = 0.05,
    lambda_l2        = 0.1,
    verbose          = -1,
    n_jobs           = -1,
    random_state     = 42,
)

# Model B uses lower lr for better convergence (dominant model in ensemble)
BASE_B = {**BASE, "learning_rate": 0.01, "num_leaves": 127,
          "feature_fraction": 0.75, "bagging_fraction": 0.75}

def train_lgb(params, Xtr, ytr, Xvl, yvl, rounds=4000, patience=150, log_every=400):
    ds_tr  = lgb.Dataset(Xtr, label=ytr)
    ds_val = lgb.Dataset(Xvl, label=yvl, reference=ds_tr)
    return lgb.train(
        params, ds_tr,
        num_boost_round = rounds,
        valid_sets      = [ds_val],
        callbacks       = [lgb.early_stopping(patience, verbose=False),
                           lgb.log_evaluation(log_every)],
    )

# ──────────────────────────────────────────────────────────────────────────────
# MODEL A: Two-stage (classifier + log-MAPE regressor)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Model A] Two-stage (classifier + MAPE-log regressor)...")

# A1: Classifier
y_tr_cls  = (y_tr  > 0).astype(int)
y_val_cls = (y_val > 0).astype(int)

clf = train_lgb(
    {**BASE, "objective": "binary", "metric": "binary_logloss"},
    X_tr, y_tr_cls, X_val, y_val_cls,
)
prob_val  = clf.predict(X_val)
prob_test = clf.predict(X_test)
print(f"  Classifier accuracy={(( prob_val>0.5)==y_val_cls).mean():.4f}  iter={clf.best_iteration}")

# A2: Regressor on positives — MAPE objective (= regression on log scale, rank-friendly)
pos_tr  = y_tr  > 0
pos_val = y_val > 0
reg = train_lgb(
    {**BASE, "objective": "mape", "metric": "mape",
     "min_child_samples": 10},   # MAPE needs smaller leaves on log scale
    X_tr[pos_tr],  y_tr[pos_tr],
    X_val[pos_val], y_val[pos_val],
)
print(f"  Regressor iter={reg.best_iteration}")

def pred_two_stage(prob, X):
    r = reg.predict(X)
    return prob * np.maximum(r, 0.0)

y_pred_A_val  = pred_two_stage(prob_val,  X_val)
y_pred_A_test = pred_two_stage(prob_test, X_test)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL B: Single global log1p regression on ALL customers
# — ranks zeros vs non-zeros + non-zero ordering in one shot
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Model B] Global log1p regression — 5-fold CV OOF + full refit for test...")

y_all_log = np.log1p(y.clip(min=0))
params_B  = {**BASE_B, "objective": "regression", "metric": "rmse"}

# 5-fold CV: get OOF predictions for ALL training customers
# (more robust validation; each customer predicted from a model that never saw them)
kf5 = KFold(n_splits=5, shuffle=True, random_state=7)
oof_B       = np.zeros(len(y))
fold_iters  = []
test_preds_folds = []

for fold_i, (f_tr, f_val) in enumerate(kf5.split(X)):
    Xf_tr, yf_tr = X.iloc[f_tr], y_all_log[f_tr]
    Xf_vl, yf_vl = X.iloc[f_val], y_all_log[f_val]

    m = train_lgb(params_B, Xf_tr, yf_tr, Xf_vl, yf_vl, log_every=9999)
    oof_B[f_val]   = m.predict(Xf_vl)
    test_preds_folds.append(m.predict(X_test))
    fold_iters.append(m.best_iteration)
    print(f"  Fold {fold_i+1}: iter={m.best_iteration}  "
          f"sp={spearmanr(y[f_val], np.expm1(np.maximum(m.predict(Xf_vl), 0))).statistic:.4f}")

avg_iter = int(np.mean(fold_iters))
print(f"  CV avg iter={avg_iter}")

# Avg test predictions from the 5 fold models
y_pred_B_test_cv = np.expm1(np.maximum(np.mean(test_preds_folds, axis=0), 0.0))

# Also refit on full data for comparison / final use
print(f"  Refitting on full data ({len(y)} rows) for {avg_iter} rounds...")
ds_all = lgb.Dataset(X, label=y_all_log)
glob_log_full = lgb.train(params_B, ds_all, num_boost_round=avg_iter,
                          callbacks=[lgb.log_evaluation(avg_iter)])

y_pred_B_val  = np.expm1(np.maximum(oof_B[val_idx], 0.0))   # OOF predictions for val set
y_pred_B_test = y_pred_B_test_cv                             # avg of 5 fold models

# ──────────────────────────────────────────────────────────────────────────────
# MODEL C: Tweedie regression on ALL customers
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Model C] Tweedie regression (all customers)...")

tweedie = train_lgb(
    {**BASE, "objective": "tweedie", "metric": "rmse",
     "tweedie_variance_power": 1.5},
    X_tr, y_tr, X_val, y_val,
)
print(f"  iter={tweedie.best_iteration}")

y_pred_C_val  = tweedie.predict(X_val)
y_pred_C_test = tweedie.predict(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE — grid search over (wA, wB, wC) maximising val Spearman
# ──────────────────────────────────────────────────────────────────────────────
print("\nTuning 3-model ensemble weights on val Spearman...")

best_sp, best_wa, best_wb = -1.0, 1/3, 1/3
for wa in np.arange(0.0, 1.01, 0.05):
    for wb in np.arange(0.0, 1.01 - wa, 0.05):
        wc = 1.0 - wa - wb
        blend = wa * y_pred_A_val + wb * y_pred_B_val + wc * y_pred_C_val
        sp    = spearmanr(y_val, blend).statistic
        if sp > best_sp:
            best_sp, best_wa, best_wb = sp, wa, wb

best_wc = 1.0 - best_wa - best_wb
print(f"  wA(two-stage)={best_wa:.2f}  wB(global-log)={best_wb:.2f}  wC(tweedie)={best_wc:.2f}")
print(f"  Ensemble Spearman={best_sp:.4f}")

y_pred_val  = best_wa * y_pred_A_val  + best_wb * y_pred_B_val  + best_wc * y_pred_C_val
y_pred_test = best_wa * y_pred_A_test + best_wb * y_pred_B_test + best_wc * y_pred_C_test

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Validation Metrics ===")
mae      = mean_absolute_error(y_val, y_pred_val)
spearman = spearmanr(y_val, y_pred_val).statistic
print(f"  MAE           : {mae:.4f}")
print(f"  Spearman      : {spearman:.4f}")

zero_mask = y_val == 0
nonz_mask = y_val >  0
print(f"  MAE zeros     : {mean_absolute_error(y_val[zero_mask], y_pred_val[zero_mask]):.4f}  (n={zero_mask.sum()})")
print(f"  MAE nonzeros  : {mean_absolute_error(y_val[nonz_mask], y_pred_val[nonz_mask]):.4f}  (n={nonz_mask.sum()})")
print(f"  Spearman nonz : {spearmanr(y_val[nonz_mask], y_pred_val[nonz_mask]).statistic:.4f}")

sp_A = spearmanr(y_val, y_pred_A_val).statistic
sp_B = spearmanr(y_val, y_pred_B_val).statistic
sp_C = spearmanr(y_val, y_pred_C_val).statistic
print(f"\n  Model A (two-stage MAPE) : {sp_A:.4f}")
print(f"  Model B (global log1p)   : {sp_B:.4f}")
print(f"  Model C (tweedie)        : {sp_C:.4f}")
print(f"  Ensemble                 : {spearman:.4f}")

# Feature importance
print("\nTop 25 features (global-log model, gain):")
imp = pd.Series(glob_log.feature_importance("gain"), index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(25).items():
    print(f"  {feat:<40} {val:,.0f}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "cust_id":           test["cust_id"].values,
    "revenue_2018_2019": np.round(y_pred_test, 4),
})
submission.to_csv(OUT_PATH, index=False)
print(f"\nSubmission saved -> {OUT_PATH}  ({len(submission)} rows)")
print(submission["revenue_2018_2019"].describe().to_string())

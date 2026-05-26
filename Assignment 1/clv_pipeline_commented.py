"""
CLV Prediction Pipeline v3  — optimised for Spearman correlation
-----------------------------------------------------------------
GOAL: Predict how much each customer will spend in 2018-2019, based on
their shopping behavior in 2016-2017.

Diagnostic finding: `frequency` alone achieves Spearman=0.384 (vs model v2=0.395)
meaning we need better feature *combinations*, not just more raw features.

Key changes vs v2:
  1. Interaction & ratio features that capture purchase velocity/cadence
  2. Single global log1p model (ranks ALL customers in one shot — no two-stage)
  3. Three-model ensemble: two-stage-log + global-log + tweedie (Spearman-tuned)
  4. Higher capacity: num_leaves=255, lr=0.02
  5. Regressor trained with MAPE objective (equivalent to log-scale MAE — good for ranking)
"""

# ── Library imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")                  # Suppress warning messages to keep output clean

import numpy as np                                  # Numerical operations (arrays, math)
import pandas as pd                                 # Data manipulation (tables, CSV reading)
from scipy.stats import spearmanr                   # Spearman rank correlation metric
from sklearn.model_selection import train_test_split # Splitting data into train/validation
from sklearn.metrics import mean_absolute_error      # MAE metric
import lightgbm as lgb                              # LightGBM: gradient boosting ML algorithm

# ── File paths ───────────────────────────────────────────────────────────────
DATA_DIR = "D:/KU Leuven - Academics/Advanced analytics & big data/Assignment 1/data/data"
TX_PATH    = f"{DATA_DIR}/transactions_2016_2017.csv"   # 344k rows of individual purchases
TRAIN_PATH = f"{DATA_DIR}/customer_clv_train.csv"       # 116k customers with known 2018-2019 revenue
TEST_PATH  = f"{DATA_DIR}/customer_clv_test.csv"        # 29k customers to predict
OUT_PATH   = f"{DATA_DIR}/submission.csv"               # Where predictions get saved

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
tx    = pd.read_csv(TX_PATH, low_memory=False, parse_dates=["order_date", "pack_date"])  # Read transactions, auto-parse dates
train = pd.read_csv(TRAIN_PATH)   # Read training labels (cust_id + revenue_2018_2019)
test  = pd.read_csv(TEST_PATH)    # Read test set (cust_id only, no revenue — that's what we predict)

# Reference date = one day after the last transaction (Jan 1, 2018)
# All recency calculations measure days backward from this date
REF_DATE = tx["order_date"].max() + pd.Timedelta(days=1)

print(f"  Transactions={tx.shape}  Train={train.shape}  Test={test.shape}")
print(f"  Ref={REF_DATE.date()}  zero-rate={(train['revenue_2018_2019']==0).mean():.3f}")
# zero-rate tells us 63.4% of training customers spent NOTHING in 2018-2019

# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
#     The raw data has one row per purchased item. We need one row per customer
#     with summary statistics describing their behavior.
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

# Create helper columns from the transaction data
tx["is_returned"]  = tx["returned_to_shop_id"].notna().astype(int)  # 1 if item was returned, 0 if not
tx["order_year"]   = tx["order_date"].dt.year                       # Extract year (2016 or 2017)
tx["order_month"]  = tx["order_date"].dt.month                      # Extract month (1-12)
tx["order_weekday"]= tx["order_date"].dt.dayofweek                  # Extract day of week (0=Mon, 6=Sun)

# First, aggregate from item-level to ORDER-level
# (one order can have multiple items — group them together)
order_agg = (
    tx.groupby(["cust_id", "sale_id", "order_date"])  # Group by customer + order + date
    .agg(
        order_revenue  = ("sale_revenue",          "sum"),   # Total revenue for this order
        order_discount = ("sale_discount_applied", "sum"),   # Total discount for this order
        order_items    = ("prod_id",               "count"), # Number of items in this order
        order_returned = ("is_returned",           "max"),   # 1 if ANY item in order was returned
        order_year     = ("order_year",            "first"), # Year of the order
    )
    .reset_index()
)

# ── 2a. Core RFM (Recency, Frequency, Monetary) ─────────────────────────────
# The classic trio of customer value analysis
rfm = (
    order_agg.groupby("cust_id")  # Now aggregate from order-level to CUSTOMER-level
    .agg(
        recency             = ("order_date",    lambda x: (REF_DATE - x.max()).days),  # Days since last purchase
        frequency           = ("sale_id",       "nunique"),   # Number of unique orders placed
        monetary            = ("order_revenue", "sum"),       # Total amount spent across all orders
        avg_order_val       = ("order_revenue", "mean"),      # Average order value
        median_order_val    = ("order_revenue", "median"),    # Median order value (less sensitive to outliers)
        max_order_val       = ("order_revenue", "max"),       # Largest single order
        std_order_val       = ("order_revenue", "std"),       # Variability in order values
        total_items         = ("order_items",   "sum"),       # Total items purchased
        avg_items_per_order = ("order_items",   "mean"),      # Average basket size
    )
    .reset_index()
)

# ── 2b. Temporal features ────────────────────────────────────────────────────
# How long has this customer been around? How spread out are their purchases?
time_feats = (
    order_agg.groupby("cust_id")
    .agg(
        first_purchase_days = ("order_date", lambda x: (REF_DATE - x.min()).days),     # Days since FIRST purchase (customer age)
        purchase_span_days  = ("order_date", lambda x: (x.max() - x.min()).days),      # Days between first and last purchase
        n_active_months     = ("order_date", lambda x: x.dt.to_period("M").nunique()), # Number of distinct months with a purchase
    )
    .reset_index()
)

# Inter-purchase time (IPT): the gap in days between consecutive orders
# This tells us the customer's natural buying rhythm
ipt = (
    order_agg.sort_values(["cust_id", "order_date"])    # Sort orders chronologically per customer
    .groupby("cust_id")["order_date"]
    .apply(lambda x: x.diff().dt.days.dropna())         # Calculate days between consecutive orders
)
ipt_stats = (
    ipt.groupby(level=0)
    .agg(avg_ipt="mean", std_ipt="std", min_ipt="min", max_ipt="max")  # Summary stats of gaps
    .reset_index()
    .rename(columns={"level_0": "cust_id"})
)

# ── 2c. Year-split + quarterly features ──────────────────────────────────────
# Compare 2016 vs 2017 behavior — is the customer spending more or less over time?
order_agg["order_quarter"] = order_agg["order_date"].dt.quarter  # Extract quarter (1-4)

for yr in [2016, 2017]:  # Create separate stats for each year
    mask = order_agg["order_year"] == yr
    sub  = order_agg[mask].groupby("cust_id").agg(
        **{f"revenue_{yr}": ("order_revenue", "sum")},    # Total revenue in that year
        **{f"orders_{yr}":  ("sale_id",       "nunique")}, # Number of orders in that year
        **{f"avg_val_{yr}": ("order_revenue", "mean")},   # Average order value in that year
        **{f"returns_{yr}": ("order_returned","sum")},     # Number of returns in that year
    ).reset_index()
    rfm = rfm.merge(sub, on="cust_id", how="left")  # Left join: keep all customers, fill missing with NaN

# Break down 2017 into quarters — captures within-year momentum
# If a customer was buying heavily in Q4 2017, they're more likely to continue into 2018
for q in [1, 2, 3, 4]:
    mask = (order_agg["order_year"] == 2017) & (order_agg["order_quarter"] == q)
    sub  = order_agg[mask].groupby("cust_id").agg(
        **{f"rev_2017_q{q}":    ("order_revenue", "sum")},    # Revenue in this quarter
        **{f"orders_2017_q{q}": ("sale_id",       "nunique")}, # Orders in this quarter
    ).reset_index()
    rfm = rfm.merge(sub, on="cust_id", how="left")

# Fill NaN with 0 for all year/quarter columns (NaN means no purchases in that period)
for col in [c for c in rfm.columns if any(c.startswith(p) for p in
            ["revenue_", "orders_", "avg_val_", "returns_", "rev_2017_q", "orders_2017_q"])]:
    rfm[col] = rfm[col].fillna(0)

# Derived growth features — is spending going up or down?
rfm["revenue_growth"] = (rfm["revenue_2017"] - rfm["revenue_2016"]) / (rfm["revenue_2016"].abs() + 1)  # Revenue change ratio
rfm["orders_growth"]  = (rfm["orders_2017"]  - rfm["orders_2016"])  / (rfm["orders_2016"]  + 1)        # Order count change ratio
rfm["active_in_2017"] = (rfm["orders_2017"] > 0).astype(int)   # Binary: did they buy in 2017?
rfm["active_in_2016"] = (rfm["orders_2016"] > 0).astype(int)   # Binary: did they buy in 2016?
rfm["share_rev_2017"] = rfm["revenue_2017"] / (rfm["monetary"].abs() + 1)  # What fraction of total spending was in 2017?

# Within-2017 momentum: did they spend more in the second half (Q3+Q4) vs first half (Q1+Q2)?
rfm["rev_2017_h2"]       = rfm["rev_2017_q3"] + rfm["rev_2017_q4"]                   # Second half 2017 revenue
rfm["rev_2017_h1"]       = rfm["rev_2017_q1"] + rfm["rev_2017_q2"]                   # First half 2017 revenue
rfm["rev_2017_h2h1_ratio"] = rfm["rev_2017_h2"] / (rfm["rev_2017_h1"] + 1)           # Ratio: >1 means accelerating
rfm["orders_2017_h2"]    = rfm["orders_2017_q3"] + rfm["orders_2017_q4"]              # Second half order count
rfm["active_q4_2017"]    = (rfm["orders_2017_q4"] > 0).astype(int)                   # Were they active in the most recent quarter?

# ── 2d. Recency-windowed features ────────────────────────────────────────────
# How much did they buy in the last 90 / 180 / 365 days?
# Recent behavior is a stronger predictor than old behavior
cutoffs = {90: REF_DATE - pd.Timedelta(days=90),    # Last 3 months
           180: REF_DATE - pd.Timedelta(days=180),  # Last 6 months
           365: REF_DATE - pd.Timedelta(days=365)}  # Last 12 months

window_dfs = []
for days, cutoff in cutoffs.items():
    sub = order_agg[order_agg["order_date"] >= cutoff].groupby("cust_id").agg(
        **{f"revenue_last{days}d": ("order_revenue", "sum")},    # Revenue in this window
        **{f"orders_last{days}d":  ("sale_id",       "nunique")}, # Orders in this window
        **{f"items_last{days}d":   ("order_items",   "sum")},    # Items in this window
        **{f"avg_val_last{days}d": ("order_revenue", "mean")},   # Avg order value in this window
    ).reset_index()
    window_dfs.append(sub)

# ── 2e. Discount / return behaviour ──────────────────────────────────────────
# Do they use lots of discounts? Do they return items often?
disc_ret = (
    order_agg.groupby("cust_id")
    .agg(
        total_discount = ("order_discount", "sum"),    # Total discount amount received
        avg_discount   = ("order_discount", "mean"),   # Average discount per order
        max_discount   = ("order_discount", "min"),    # Largest discount (note: discounts might be negative)
        return_rate    = ("order_returned", "mean"),    # Fraction of orders with a return
        n_returns      = ("order_returned", "sum"),     # Total number of orders with returns
    )
    .reset_index()
)
disc_ret["discount_per_item"] = (                      # Average discount per item purchased
    disc_ret["total_discount"]
    / rfm.set_index("cust_id").loc[disc_ret["cust_id"], "total_items"].values
)

# ── 2f. Product preferences ─────────────────────────────────────────────────
# What kinds of products does this customer buy?
prod_agg = (
    tx.groupby("cust_id")
    .agg(
        web_only_rate     = ("prod_web_only", "mean"),     # Fraction of items that are web-only products
        outlet_rate       = ("prod_outlet",   "mean"),     # Average outlet indicator (higher = more outlet purchases)
        n_unique_brands   = ("prod_brand",    "nunique"),  # How many different brands they buy from
        n_unique_products = ("prod_id",       "nunique"),  # How many different products they've bought
        n_unique_sizes    = ("prod_size",     "nunique"),  # How many different sizes (could indicate buying for family)
        n_unique_colors   = ("prod_color",    "nunique"),  # How many different colors
        avg_order_weekday = ("order_weekday", "mean"),     # Average day of week they order (0=Mon)
        std_order_month   = ("order_month",   "std"),      # How spread out their purchases are across months
        avg_order_month   = ("order_month",   "mean"),     # Average month of purchase
        share_q4          = ("order_month",   lambda x: (x >= 10).mean()),  # Fraction of purchases in Oct-Dec (holiday shopping)
    )
    .reset_index()
)

# Brand entropy: measures how spread out brand choices are
# Low entropy = loyal to one brand, High entropy = buys from many brands equally
brand_counts = tx.groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")        # Count purchases per brand per customer
brand_totals = brand_counts.groupby("cust_id")["cnt"].sum()                               # Total purchases per customer
brand_counts["share"] = brand_counts["cnt"] / brand_counts["cust_id"].map(brand_totals)   # Share of purchases per brand
brand_entropy   = (brand_counts.groupby("cust_id")["share"]
                   .apply(lambda p: -np.sum(p * np.log(p + 1e-12)))                       # Shannon entropy formula
                   .reset_index().rename(columns={"share": "brand_entropy"}))
top_brand_share = (brand_counts.groupby("cust_id")["share"]
                   .max().reset_index().rename(columns={"share": "top_brand_share"}))      # Share of their most-purchased brand

# Category mix: what fraction of purchases are men's, women's, boys', girls', etc.
# Creates one column per category (e.g., cat_men = 0.6 means 60% of purchases are men's shoes)
cat_counts = tx.groupby(["cust_id","prod_type_1"]).size().reset_index(name="cnt")
cat_totals  = cat_counts.groupby("cust_id")["cnt"].sum()
cat_counts["share"] = cat_counts["cnt"] / cat_counts["cust_id"].map(cat_totals)
cat_pivot = (cat_counts.pivot_table(index="cust_id", columns="prod_type_1", values="share", fill_value=0)
             .reset_index())
cat_pivot.columns = ["cust_id"] + [f"cat_{c}" for c in cat_pivot.columns[1:]]  # Rename columns like cat_men, cat_women

# Season mix: what fraction of purchases are from each collection (W14=Winter 2014, SS15=Spring/Summer 2015, etc.)
season_counts = tx.groupby(["cust_id","prod_season"]).size().reset_index(name="cnt")
season_totals  = season_counts.groupby("cust_id")["cnt"].sum()
season_counts["share"] = season_counts["cnt"] / season_counts["cust_id"].map(season_totals)
season_pivot = (season_counts.pivot_table(index="cust_id", columns="prod_season", values="share", fill_value=0)
                .reset_index())
season_pivot.columns = ["cust_id"] + [f"season_{c}" for c in season_pivot.columns[1:]]

# Average price per item (total revenue / total items)
avg_price = (
    tx.groupby("cust_id")
    .apply(lambda df: df["sale_revenue"].sum() / len(df) if len(df) > 0 else 0,
           include_groups=False)
    .reset_index().rename(columns={0: "avg_item_price"})
)

# ── 2g. Merge all feature tables into one big table ──────────────────────────
# Join everything together on cust_id so each customer has one row with all features
print("Merging feature tables...")
all_feats = rfm.copy()
for df in ([time_feats, ipt_stats, disc_ret, prod_agg,
            brand_entropy, top_brand_share, cat_pivot, season_pivot, avg_price]
           + window_dfs):
    all_feats = all_feats.merge(df, on="cust_id", how="left")

# Fill NaN for window features (customers with no purchases in that window get 0)
for days in cutoffs:
    for col in [f"revenue_last{days}d", f"orders_last{days}d",
                f"items_last{days}d",   f"avg_val_last{days}d"]:
        all_feats[col] = all_feats[col].fillna(0)

# ── 2h. Engineered interaction / velocity features ───────────────────────────
# These combine basic features into more powerful signals
# Diagnostic showed: frequency=0.384, avg_ipt=0.348, monetary=0.296 (individual Spearman with target)

# Log transforms: compress skewed distributions (few big spenders + many small spenders)
all_feats["monetary_log"]      = np.log1p(all_feats["monetary"].clip(lower=0))       # log(1 + monetary)
all_feats["frequency_log"]     = np.log1p(all_feats["frequency"])                    # log(1 + frequency)
all_feats["rev_2017_log"]      = np.log1p(all_feats["revenue_2017"].clip(lower=0))
all_feats["rev_last90d_log"]   = np.log1p(all_feats["revenue_last90d"].clip(lower=0))
all_feats["rev_last180d_log"]  = np.log1p(all_feats["revenue_last180d"].clip(lower=0))

# Purchase velocity: how many orders per month of being a customer
# A customer who placed 12 orders in 6 months is more active than one who placed 12 orders in 24 months
all_feats["purchase_rate_monthly"] = (
    all_feats["frequency"] / (all_feats["first_purchase_days"] / 30.0 + 1)
)

# Revenue velocity: how much revenue per month of being a customer
all_feats["revenue_rate_monthly"] = (
    all_feats["monetary"] / (all_feats["first_purchase_days"] / 30.0 + 1)
)

# Recent velocity: average revenue per order in the last 90 days
all_feats["rev_per_recent_order"] = (
    all_feats["revenue_last90d"] / (all_feats["orders_last90d"] + 1)
)

# Frequency × recency: identifies frequent buyers who also bought recently (the best customers)
all_feats["freq_x_recency_inv"]  = all_feats["frequency"] / (all_feats["recency"] + 1)
# Frequency × monetary: identifies customers who buy often AND spend a lot
all_feats["freq_x_monetary_log"] = all_feats["frequency_log"] * all_feats["monetary_log"]

# 2017 intensity: how much per order and per month in 2017
all_feats["rev17_per_order"]  = all_feats["revenue_2017"] / (all_feats["orders_2017"] + 1)
all_feats["rev17_per_month"]  = all_feats["revenue_2017"] / 12.0

# KEY FEATURE: Recency / average inter-purchase time
# If a customer buys every 30 days and their last purchase was 90 days ago, ratio = 3.0
# This means they've "skipped" 3 cycles — a strong signal they may not come back
all_feats["recency_ipt_ratio"] = all_feats["recency"] / (all_feats["avg_ipt"] + 1)

# Revenue concentration: what share of total spending happened recently?
all_feats["rev_last90d_share"]  = all_feats["revenue_last90d"]  / (all_feats["monetary"].abs() + 1)
all_feats["rev_last180d_share"] = all_feats["revenue_last180d"] / (all_feats["monetary"].abs() + 1)
all_feats["rev_last365d_share"] = all_feats["revenue_last365d"] / (all_feats["monetary"].abs() + 1)

# Discount intensity: what fraction of their total spend was discounted?
all_feats["discount_rate"]      = all_feats["total_discount"].abs() / (all_feats["monetary"].abs() + 1)

# Brand loyalty: orders per unique brand (high = loyal to few brands)
all_feats["orders_per_brand"]   = all_feats["frequency"] / (all_feats["n_unique_brands"] + 1)

# ── Extra interaction features around the dominant signal (recency_ipt_ratio) ──
ript = all_feats["recency_ipt_ratio"].clip(lower=0)
all_feats["log_recency_ipt"]    = np.log1p(ript)                          # Log version (less extreme values)
all_feats["ript_sq"]            = ript ** 2                                # Squared version (amplifies high values)
# Binary flags: has the customer skipped 1, 2, or 3 full purchase cycles?
all_feats["skipped_1_cycle"]    = (ript >= 1).astype(int)                  # Overdue by 1+ cycles
all_feats["skipped_2_cycles"]   = (ript >= 2).astype(int)                  # Overdue by 2+ cycles
all_feats["skipped_3_cycles"]   = (ript >= 3).astype(int)                  # Overdue by 3+ cycles

# 2017 Q4 features: the most recent quarter matters most
all_feats["rev_2017_q4_log"]    = np.log1p(all_feats["rev_2017_q4"].clip(lower=0))
all_feats["active_q4_2017"]     = all_feats["active_q4_2017"]
all_feats["q4_to_q3_rev_ratio"] = all_feats["rev_2017_q4"] / (all_feats["rev_2017_q3"] + 1)  # Q4 vs Q3 comparison

# Did the customer buy more or less than expected in the last 90 days?
# Compares actual orders to what their purchase rhythm would predict
all_feats["recent_vs_expected"] = (
    all_feats["orders_last90d"] / (90 / (all_feats["avg_ipt"] + 1) + 0.01)
)

# Combines max order value × frequency / recency — identifies high-value frequent recent buyers
all_feats["max_val_x_freq_inv_rec"] = (
    all_feats["max_order_val"] * all_feats["frequency"] / (all_feats["recency"] + 1)
)

# Orders per active month — normalised buying frequency
all_feats["freq_per_active_month"] = all_feats["frequency"] / (all_feats["n_active_months"] + 1)

print(f"  Feature matrix: {all_feats.shape}")  # Shows (number of customers, number of features)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  MERGE WITH LABELS & SPLIT INTO TRAIN/VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
# Attach the target variable (revenue_2018_2019) to the feature table
train_feat = train.merge(all_feats, on="cust_id", how="left")  # Train: has target
test_feat  = test.merge(all_feats,  on="cust_id", how="left")  # Test: no target

FEATURE_COLS = [c for c in all_feats.columns if c != "cust_id"]  # All columns except cust_id
print(f"  Total features: {len(FEATURE_COLS)}")

X      = train_feat[FEATURE_COLS].copy()       # Feature matrix for training (116k rows × N features)
y      = train_feat["revenue_2018_2019"].values # Target vector (what we're trying to predict)
X_test = test_feat[FEATURE_COLS].copy()         # Feature matrix for test (29k rows × N features)

# Split training data: 80% for training, 20% for validation (to check model performance)
# Stratified: both splits have the same proportion of zero-revenue customers (~63.4%)
print("\nSplitting 80/20 (stratified by zero/nonzero)...")
from sklearn.model_selection import StratifiedShuffleSplit, KFold
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)  # random_state=42 for reproducibility
tr_idx, val_idx = next(sss.split(X, (y > 0).astype(int)))  # Stratify by zero vs non-zero
X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]   # Training features / Validation features
y_tr, y_val = y[tr_idx], y[val_idx]               # Training target / Validation target
print(f"  Train={len(y_tr)} (pos={( y_tr>0).sum()})  Val={len(y_val)} (pos={(y_val>0).sum()})")

# ── 3a. Out-of-fold target encodings ─────────────────────────────────────────
# TARGET ENCODING: Replace categorical text (e.g., brand name "Nike") with the
# average revenue of customers who have that value.
# OUT-OF-FOLD: To prevent data leakage, we calculate averages using only OTHER
# customers (5-fold split), so a customer's own revenue doesn't influence their encoding.
print("  Adding OOF target encodings...")

# Find each customer's most-purchased brand (overall and in 2017 specifically)
top_brand_map = (
    tx.groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_brand"]
    .rename("top_brand")
)
# Most-purchased brand in 2017 only — more recent signal
top_brand_2017_map = (
    tx[tx["order_year"] == 2017]
    .groupby(["cust_id","prod_brand"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_brand"]
    .rename("top_brand_2017")
)
# Most-purchased category (men/women/boys/girls)
top_cat_map = (
    tx.groupby(["cust_id","prod_type_1"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_type_1"]
    .rename("top_cat")
)
# Most-purchased sub-category (e.g., "high-top sneakers", "dress boots")
tx["prod_type_3_filled"] = tx["prod_type_3"].fillna("unknown")  # Fill missing with "unknown"
top_cat3_map = (
    tx.groupby(["cust_id","prod_type_3_filled"]).size().reset_index(name="cnt")
    .sort_values("cnt", ascending=False).groupby("cust_id").first()["prod_type_3_filled"]
    .rename("top_cat3")
)
# Customer cohort: when did they first purchase? (e.g., "2016Q1", "2017Q3")
# Customers acquired in the same period may behave similarly
first_date_map = order_agg.groupby("cust_id")["order_date"].min()
cohort_map = (first_date_map.dt.to_period("Q").astype(str)).rename("cohort_q")

# Combine all categorical mappings into one table
train_feat2 = (
    train_feat[["cust_id"]]
    .join(top_brand_map,      on="cust_id")
    .join(top_brand_2017_map, on="cust_id")
    .join(top_cat_map,        on="cust_id")
    .join(top_cat3_map,       on="cust_id")
    .join(cohort_map,         on="cust_id")
)
train_feat2["y_log"] = np.log1p(y.clip(min=0))  # Log-transformed target for encoding

# 5-fold out-of-fold encoding
te_cols = ["top_brand", "top_brand_2017", "top_cat", "top_cat3", "cohort_q"]
kf      = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = train_feat2["y_log"].mean()  # Fallback value for unseen categories

te_arrays = {c: np.zeros(len(y)) for c in te_cols}  # Initialize arrays for encoded values
te_means  = {}

# For each fold: calculate means from 80% of data, assign to the other 20%
# This way no customer's own revenue influences their encoded value
for fold_tr, fold_val in kf.split(train_feat2):
    for col in te_cols:
        means = train_feat2.iloc[fold_tr].groupby(col)["y_log"].mean()                       # Mean revenue per category (from other folds)
        te_arrays[col][fold_val] = train_feat2.iloc[fold_val][col].map(means).fillna(global_mean).values  # Map to validation fold

# For test set: use means from ALL training data (no leakage concern for test)
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

# Add target-encoded features to the feature matrices
for col in te_cols:
    X[f"te_{col}"]      = te_arrays[col]       # Add to training features
    X_test[f"te_{col}"] = te_test_arrays[col]   # Add to test features

FEATURE_COLS = list(X.columns)  # Update feature list with new columns

# Rebuild train/val splits to include the new target-encoded features
X_tr  = X.iloc[tr_idx]
X_val = X.iloc[val_idx]

# ══════════════════════════════════════════════════════════════════════════════
# 4.  SHARED LGBM HELPER
#     LightGBM settings and a reusable training function
# ══════════════════════════════════════════════════════════════════════════════
BASE = dict(
    learning_rate    = 0.02,    # How much each tree adjusts predictions (smaller = more careful)
    num_leaves       = 255,     # Max leaves per tree (higher = more complex trees)
    min_child_samples= 20,      # Minimum samples in a leaf (prevents overfitting)
    feature_fraction = 0.7,     # Use 70% of features per tree (randomness reduces overfitting)
    bagging_fraction = 0.7,     # Use 70% of data per tree (more randomness)
    bagging_freq     = 5,       # Re-sample every 5 trees
    lambda_l1        = 0.05,    # L1 regularization (encourages sparse features)
    lambda_l2        = 0.1,     # L2 regularization (penalizes large weights)
    verbose          = -1,      # Suppress LightGBM output
    n_jobs           = -1,      # Use all CPU cores
    random_state     = 42,      # Reproducibility
)

# Model B gets its own settings: slower learning rate for finer convergence
BASE_B = {**BASE, "learning_rate": 0.01, "num_leaves": 127,
          "feature_fraction": 0.75, "bagging_fraction": 0.75}

def train_lgb(params, Xtr, ytr, Xvl, yvl, rounds=4000, patience=150, log_every=400):
    """Train a LightGBM model with early stopping.
    - Trains up to 4000 trees
    - Stops if validation score doesn't improve for 150 rounds (early stopping)
    - Returns the best model
    """
    ds_tr  = lgb.Dataset(Xtr, label=ytr)                          # Training data in LightGBM format
    ds_val = lgb.Dataset(Xvl, label=yvl, reference=ds_tr)         # Validation data
    return lgb.train(
        params, ds_tr,
        num_boost_round = rounds,                                  # Max number of trees
        valid_sets      = [ds_val],                                # Monitor validation performance
        callbacks       = [lgb.early_stopping(patience, verbose=False),  # Stop if no improvement
                           lgb.log_evaluation(log_every)],         # Print progress every N rounds
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: Two-stage (classifier + regressor)
#   Step 1: Predict probability of returning (will they buy anything at all?)
#   Step 2: Predict how much they'll spend IF they return
#   Final prediction = probability × predicted amount
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Model A] Two-stage (classifier + MAPE-log regressor)...")

# A1: CLASSIFIER — predicts P(revenue > 0)
y_tr_cls  = (y_tr  > 0).astype(int)   # Convert revenue to binary: 1 if they spent anything, 0 if not
y_val_cls = (y_val > 0).astype(int)

clf = train_lgb(
    {**BASE, "objective": "binary", "metric": "binary_logloss"},  # Binary classification
    X_tr, y_tr_cls, X_val, y_val_cls,
)
prob_val  = clf.predict(X_val)    # Probability of returning (0 to 1) for validation customers
prob_test = clf.predict(X_test)   # Probability of returning for test customers
print(f"  Classifier accuracy={(( prob_val>0.5)==y_val_cls).mean():.4f}  iter={clf.best_iteration}")

# A2: REGRESSOR — predicts revenue amount, trained ONLY on customers who actually spent something
# Uses MAPE objective (Mean Absolute Percentage Error) which is like regression on a log scale
# — good for ranking because it treats a €10 error on a €20 customer the same as a €100 error on a €200 customer
pos_tr  = y_tr  > 0   # Filter: only customers with positive revenue
pos_val = y_val > 0
reg = train_lgb(
    {**BASE, "objective": "mape", "metric": "mape",
     "min_child_samples": 10},
    X_tr[pos_tr],  y_tr[pos_tr],       # Train only on positive-revenue customers
    X_val[pos_val], y_val[pos_val],
)
print(f"  Regressor iter={reg.best_iteration}")

def pred_two_stage(prob, X):
    """Combine classifier probability with regressor prediction"""
    r = reg.predict(X)                   # Predicted amount if they return
    return prob * np.maximum(r, 0.0)     # probability × amount (floor at 0)

y_pred_A_val  = pred_two_stage(prob_val,  X_val)   # Model A predictions for validation set
y_pred_A_test = pred_two_stage(prob_test, X_test)  # Model A predictions for test set

# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: Global log1p regression
#   Instead of splitting into classifier + regressor, this model predicts
#   log(1 + revenue) for ALL customers in one shot.
#   This naturally handles zeros (log(1+0)=0) and large values.
#   Uses 5-fold cross-validation for robust estimates.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Model B] Global log1p regression — 5-fold CV OOF + full refit for test...")

y_all_log = np.log1p(y.clip(min=0))    # Transform target: log(1 + revenue). Compresses the scale.
params_B  = {**BASE_B, "objective": "regression", "metric": "rmse"}  # Standard regression with RMSE

# 5-FOLD CROSS-VALIDATION:
# Split data into 5 parts. Train on 4 parts, predict the 5th. Repeat 5 times.
# Every customer gets a prediction from a model that NEVER saw them during training.
# This gives more reliable validation scores than a single 80/20 split.
kf5 = KFold(n_splits=5, shuffle=True, random_state=7)
oof_B       = np.zeros(len(y))     # Out-of-fold predictions for all training customers
fold_iters  = []                    # Track how many trees each fold model used
test_preds_folds = []               # Each fold's predictions on the test set

for fold_i, (f_tr, f_val) in enumerate(kf5.split(X)):
    Xf_tr, yf_tr = X.iloc[f_tr], y_all_log[f_tr]     # This fold's training data
    Xf_vl, yf_vl = X.iloc[f_val], y_all_log[f_val]    # This fold's validation data

    m = train_lgb(params_B, Xf_tr, yf_tr, Xf_vl, yf_vl, log_every=9999)  # Train model
    oof_B[f_val]   = m.predict(Xf_vl)                  # Store OOF predictions
    test_preds_folds.append(m.predict(X_test))          # This fold's test predictions
    fold_iters.append(m.best_iteration)                 # How many trees this fold used
    print(f"  Fold {fold_i+1}: iter={m.best_iteration}  "
          f"sp={spearmanr(y[f_val], np.expm1(np.maximum(m.predict(Xf_vl), 0))).statistic:.4f}")

avg_iter = int(np.mean(fold_iters))   # Average number of trees across folds
print(f"  CV avg iter={avg_iter}")

# AVERAGE all 5 fold models' test predictions — reduces variance
y_pred_B_test_cv = np.expm1(np.maximum(np.mean(test_preds_folds, axis=0), 0.0))  # Convert back from log scale

# Also REFIT on ALL training data using the average iteration count
# This uses 100% of the data for the final model (no held-out validation)
print(f"  Refitting on full data ({len(y)} rows) for {avg_iter} rounds...")
ds_all = lgb.Dataset(X, label=y_all_log)
glob_log_full = lgb.train(params_B, ds_all, num_boost_round=avg_iter,
                          callbacks=[lgb.log_evaluation(avg_iter)])

y_pred_B_val  = np.expm1(np.maximum(oof_B[val_idx], 0.0))   # OOF predictions for validation set (back from log)
y_pred_B_test = y_pred_B_test_cv                              # Average of 5 fold models for test set

# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: Tweedie regression
#   Tweedie is a statistical distribution designed for data with:
#   - Lots of exact zeros (63% of customers)
#   - A long right tail (a few big spenders)
#   This is exactly the shape of customer revenue data.
#   tweedie_variance_power=1.5 is between Poisson (1) and Gamma (2)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Model C] Tweedie regression (all customers)...")

tweedie = train_lgb(
    {**BASE, "objective": "tweedie", "metric": "rmse",
     "tweedie_variance_power": 1.5},    # 1.5 = halfway between Poisson and Gamma
    X_tr, y_tr, X_val, y_val,
)
print(f"  iter={tweedie.best_iteration}")

y_pred_C_val  = tweedie.predict(X_val)    # Model C validation predictions
y_pred_C_test = tweedie.predict(X_test)   # Model C test predictions

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE — Find the best weighted combination of all 3 models
#   Try every combination of weights (wA, wB, wC) in steps of 5%
#   Pick the weights that maximize Spearman correlation on validation set
#   Example: final = 0.10 × ModelA + 0.80 × ModelB + 0.10 × ModelC
# ══════════════════════════════════════════════════════════════════════════════
print("\nTuning 3-model ensemble weights on val Spearman...")

best_sp, best_wa, best_wb = -1.0, 1/3, 1/3
for wa in np.arange(0.0, 1.01, 0.05):                       # Weight for Model A: 0%, 5%, 10%, ..., 100%
    for wb in np.arange(0.0, 1.01 - wa, 0.05):              # Weight for Model B: remaining range
        wc = 1.0 - wa - wb                                   # Weight for Model C: whatever is left (weights sum to 1)
        blend = wa * y_pred_A_val + wb * y_pred_B_val + wc * y_pred_C_val  # Weighted average
        sp    = spearmanr(y_val, blend).statistic             # Calculate Spearman correlation
        if sp > best_sp:                                      # Keep the best combination
            best_sp, best_wa, best_wb = sp, wa, wb

best_wc = 1.0 - best_wa - best_wb
print(f"  wA(two-stage)={best_wa:.2f}  wB(global-log)={best_wb:.2f}  wC(tweedie)={best_wc:.2f}")
print(f"  Ensemble Spearman={best_sp:.4f}")

# Apply the best weights to get final predictions
y_pred_val  = best_wa * y_pred_A_val  + best_wb * y_pred_B_val  + best_wc * y_pred_C_val   # Validation predictions
y_pred_test = best_wa * y_pred_A_test + best_wb * y_pred_B_test + best_wc * y_pred_C_test  # Test predictions

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — How well does the model perform on the held-out validation set?
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Validation Metrics ===")
mae      = mean_absolute_error(y_val, y_pred_val)      # Average prediction error in euros
spearman = spearmanr(y_val, y_pred_val).statistic       # How well does it rank customers?
print(f"  MAE           : {mae:.4f}")
print(f"  Spearman      : {spearman:.4f}")

# Break down performance by zero vs non-zero customers
zero_mask = y_val == 0    # Customers who didn't return
nonz_mask = y_val >  0    # Customers who did return
print(f"  MAE zeros     : {mean_absolute_error(y_val[zero_mask], y_pred_val[zero_mask]):.4f}  (n={zero_mask.sum()})")
print(f"  MAE nonzeros  : {mean_absolute_error(y_val[nonz_mask], y_pred_val[nonz_mask]):.4f}  (n={nonz_mask.sum()})")
print(f"  Spearman nonz : {spearmanr(y_val[nonz_mask], y_pred_val[nonz_mask]).statistic:.4f}")

# Compare individual model Spearman scores
sp_A = spearmanr(y_val, y_pred_A_val).statistic
sp_B = spearmanr(y_val, y_pred_B_val).statistic
sp_C = spearmanr(y_val, y_pred_C_val).statistic
print(f"\n  Model A (two-stage MAPE) : {sp_A:.4f}")
print(f"  Model B (global log1p)   : {sp_B:.4f}")
print(f"  Model C (tweedie)        : {sp_C:.4f}")
print(f"  Ensemble                 : {spearman:.4f}")

# FEATURE IMPORTANCE: which features does the model rely on most?
# "gain" = how much each feature reduces prediction error across all trees
print("\nTop 25 features (global-log model, gain):")
imp = pd.Series(glob_log_full.feature_importance("gain"), index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(25).items():
    print(f"  {feat:<40} {val:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE — Write predictions to submission.csv for leaderboard upload
# ══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    "cust_id":           test["cust_id"].values,           # Customer IDs from test set
    "revenue_2018_2019": np.round(y_pred_test, 4),         # Predicted revenue, rounded to 4 decimals
})
submission.to_csv(OUT_PATH, index=False)                    # Save as CSV
print(f"\nSubmission saved -> {OUT_PATH}  ({len(submission)} rows)")
print(submission["revenue_2018_2019"].describe().to_string())  # Summary statistics of predictions

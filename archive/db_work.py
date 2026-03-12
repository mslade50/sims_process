import sqlite3
import pandas as pd
from sim_inputs import name_replacements, course_id
# change the path if your DB lives elsewhere
with sqlite3.connect("dg_historical.db") as conn:
    player_rounds_df = pd.read_sql("SELECT * FROM player_rounds", conn)

# optional: parse round_date to datetime
player_rounds_df["round_date"] = pd.to_datetime(player_rounds_df["round_date"], errors="coerce")

# assumes `player_rounds_df` is already loaded
df = player_rounds_df.copy()
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
df = df.sort_values(["dg_id", "round_date", "year", "event_id", "round_num"], na_position="first")
df["prev_rows"] = df.groupby("dg_id").cumcount()
mask = (df["year"] >= 2023) & (df["prev_rows"] < 5) & df["tour"].eq("pga") & df["round_num"].isin([1, 2])
avg_sg_total = pd.to_numeric(df.loc[mask, "sg_total"], errors="coerce").mean()
avg_skill    = pd.to_numeric(df.loc[mask, "skill"],    errors="coerce").mean()
print("avg_sg_total (R1/R2):", avg_sg_total)
print("avg_skill (same rows):", avg_skill)


import numpy as np
import pandas as pd
from numba import njit

# assumes df already loaded with columns: dg_id, round_date, year, round_num, sg_total_adj
df = df.copy()
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["sg_total_adj"] = pd.to_numeric(df["sg_total_adj"], errors="coerce")
df = df.sort_values(["dg_id", "round_date", "event_id", "round_num"])

# Step 1: Create a temporary DataFrame to calculate field averages
temp_df = df[['player_name', 'tour', 'year', 'event_id', 'round_date', 'driving_acc']].copy()

# Step 2: Calculate the field average driving accuracy for each tournament
temp_df['field_avg_acc'] = temp_df.groupby(['tour', 'year', 'event_id'])['driving_acc'].transform('mean')

# Step 3: Calculate relative driving accuracy
temp_df['rel_acc'] = temp_df['driving_acc'] - temp_df['field_avg_acc']

# Step 4: Merge the relative driving accuracy back into the main DataFrame
df = df.merge(
    temp_df[['player_name', 'tour', 'year', 'event_id', 'round_date', 'rel_acc']],
    on=['player_name', 'tour', 'year', 'event_id', 'round_date'],
    how='left'
)


mask = df["round_num"].isin([1, 2])

# --- EMAs & SMA (shifted 1 to avoid lookahead) ---
def _roll_feats(g: pd.DataFrame) -> pd.DataFrame:
    x = g["sg_total_adj"]
    out = pd.DataFrame(index=g.index)
    out["ema11_adj"] = x.ewm(span=11, adjust=False, min_periods=1).mean().shift(1)
    out["ema21_adj"] = x.ewm(span=21, adjust=False, min_periods=1).mean().shift(1)
    out["sma50_adj"] = x.rolling(50, min_periods=1).mean().shift(1)
    return out

df.loc[mask, ["ema11_adj", "ema21_adj", "sma50_adj"]] = (
    df.loc[mask].groupby("dg_id", group_keys=False).apply(_roll_feats)
)

def _ema21_subcomponents(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("round_date")
    out = pd.DataFrame(index=g.index)
    for col, new in [
        ("sg_ott",  "ema21_ott"),
        ("sg_app",  "ema21_app"),
        ("sg_arg",  "ema21_arg"),
        ("sg_putt", "ema21_putt"),
        ("rel_dd", "ema21_rel_dd"),
        ("rel_acc", "ema21_rel_acc"),
    ]:
        x = pd.to_numeric(g[col], errors="coerce")
        ema = x.ewm(span=21, adjust=False, min_periods=1).mean().shift(1)  # avoid lookahead
        prior_cnt = x.notna().cumsum().shift(1, fill_value=0)             # require ≥9 prior vals
        out[new] = ema.where(prior_cnt >= 9, np.nan)
    return out

df.loc[mask, ["ema21_ott", "ema21_app", "ema21_arg", "ema21_putt", "ema21_rel_dd", "ema21_rel_acc"]] = (
    df.loc[mask].groupby("dg_id", group_keys=False).apply(_ema21_subcomponents)
)
# Train only on rows where the player already has ≥20 prior rounds in the DB
# === Train only on rows where player has ≥20 prior rounds; add 21-EMA subs ===
# === Two-step model ===
# Step 1: predict sg_total_adj from [ema11_adj, sma50_adj]
# Step 2: predict residuals from Step 1 using category EMAs [ema21_ott, ema21_app, ema21_arg, ema21_putt]
# - Train on 2017–2019, PGA, R1/R2, players with ≥20 prior rows, no intercept
# - Evaluate on 2019–2022, PGA, R1/R2

# === Three-step model ===
# Step 1: sg_total_adj ~ [ema11_adj, sma50_adj]
# Step 2: residual_1     ~ [ema21_ott, ema21_app, ema21_arg, ema21_putt]
# Step 3: residual_2     ~ [age, age^2]
# - Train on 2017–2019, PGA, R1/R2, players with ≥20 prior rows (no intercepts)
# - Evaluate on 2019–2022, PGA, R1/R2
# --- Add relative_dsle column, then build a 4-step model (adds Step 4: relative_dsle) ---

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# Ensure proper sorting/types
df = df.sort_values(["player_name", "round_date", "event_id", "round_num"]).copy()
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
df["age"] = pd.to_numeric(df.get("age"), errors="coerce")

# ---------- Build dsle and relative_dsle ----------
def _calc_dsle(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("round_date").copy()
    dsle = g["round_date"].diff().dt.days
    dsle = dsle.where(g["round_num"].eq(1), np.nan)  # only R1 gets a new dsle
    g["dsle"] = dsle.clip(upper=30)                  # cap at 30 days
    return g

tmp = (
    df[["player_name", "tour", "year", "event_id", "round_num", "round_date"]]
    .groupby("player_name", group_keys=False)
    .apply(_calc_dsle)
)

# forward-fill within player so R2 inherits the R1 dsle
tmp["dsle"] = tmp.groupby("player_name")["dsle"].ffill()
# merge back (drop existing if present)
df = df.drop(columns=["dsle"], errors="ignore").merge(
    tmp[["player_name", "tour", "year", "event_id", "round_num", "round_date", "dsle"]],
    on=["player_name", "tour", "year", "event_id", "round_num", "round_date"],
    how="left"
)

df["dsle"] = df["dsle"].fillna(30).astype(float)

# field-average (per event) and relative dsle
df["field_avg_dsle"] = df.groupby(["tour", "year", "event_id"])["dsle"].transform("mean")
df["relative_dsle"] = (df["dsle"] - df["field_avg_dsle"]).fillna(0.0)
# Zero-out relative_dsle for the first 3 months of 2017 (dataset warm-up)
mask_early_2017 = (df["round_date"] >= pd.Timestamp("2017-01-01")) & (df["round_date"] < pd.Timestamp("2017-04-01"))
df.loc[mask_early_2017, "relative_dsle"] = 0.0

# ---------- Modeling scope & prev_rows ----------
df = df.sort_values(["dg_id", "round_date", "event_id", "round_num"])
if "prev_rows" not in df.columns:
    df["prev_rows"] = df.groupby("dg_id").cumcount()

df_2=df.copy()

base = df.loc[df["tour"].eq("pga") & df["round_num"].isin([1, 2])].copy()
base["age_sq"] = base["age"] * base["age"]
TARGET = "sg_total_adj"

FEATS1 = ["ema11_adj", "sma50_adj"]
FEATS2 = ["ema21_ott", "ema21_app", "ema21_arg", "ema21_putt"]
FEATS3 = ["age", "age_sq"]
FEATS4 = ["relative_dsle"]

train_mask = base["year"].between(2017, 2019) & (base["prev_rows"] >= 20)

# ---------------- Step 1: TARGET ~ FEATS1 (no intercept) ----------------
train1 = base.loc[train_mask].dropna(subset=FEATS1 + [TARGET]).copy()
X1_tr = train1[FEATS1].to_numpy(); y_tr = train1[TARGET].to_numpy()
step1 = LinearRegression(fit_intercept=False).fit(X1_tr, y_tr)
ols1  = sm.OLS(y_tr, X1_tr).fit()
print("\n=== Step 1: sg_total_adj ~ ema11_adj + sma50_adj (no const) ===")
print(ols1.summary())

# ---------------- Step 2: residual_1 ~ FEATS2 (no intercept) ----------------
t2b = base.loc[train_mask].dropna(subset=FEATS1 + [TARGET]).copy()
t2b["pred_step1_tr"] = step1.predict(t2b[FEATS1].to_numpy())
t2b["resid_tr"] = t2b[TARGET] - t2b["pred_step1_tr"]
train2 = t2b.dropna(subset=FEATS2).copy()
X2_tr = train2[FEATS2].to_numpy(); y2_tr = train2["resid_tr"].to_numpy()
step2 = LinearRegression(fit_intercept=False).fit(X2_tr, y2_tr)
ols2  = sm.OLS(y2_tr, X2_tr).fit()
print("\n=== Step 2: residual_1 ~ ema21_ott + ema21_app + ema21_arg + ema21_putt (no const) ===")
print(ols2.summary())

# ---------------- Step 3: residual_2 ~ FEATS3 (no intercept) ----------------
t3b = t2b.copy()
t3b["pred_step2_tr"] = 0.0
idx2 = t3b[FEATS2].notna().all(axis=1)
if idx2.any():
    t3b.loc[idx2, "pred_step2_tr"] = step2.predict(t3b.loc[idx2, FEATS2].to_numpy())
t3b["resid2_tr"] = t3b[TARGET] - t3b["pred_step1_tr"] - t3b["pred_step2_tr"]
train3 = t3b.dropna(subset=FEATS3 + ["resid2_tr"]).copy()
X3_tr = train3[FEATS3].to_numpy(); y3_tr = train3["resid2_tr"].to_numpy()
step3 = LinearRegression(fit_intercept=False).fit(X3_tr, y3_tr)
ols3  = sm.OLS(y3_tr, X3_tr).fit()
print("\n=== Step 3: residual_2 ~ age + age_sq (no const) ===")
print(ols3.summary())

# ---------------- Step 4: residual_3 ~ FEATS4 (no intercept) ----------------
t4b = t3b.copy()
t4b["pred_step3_tr"] = 0.0
idx3 = t4b[FEATS3].notna().all(axis=1)
if idx3.any():
    t4b.loc[idx3, "pred_step3_tr"] = step3.predict(t4b.loc[idx3, FEATS3].to_numpy())
t4b["resid3_tr"] = t4b[TARGET] - t4b["pred_step1_tr"] - t4b["pred_step2_tr"] - t4b["pred_step3_tr"]

train4 = t4b.dropna(subset=FEATS4 + ["resid3_tr"]).copy()
X4_tr = train4[FEATS4].to_numpy(); y4_tr = train4["resid3_tr"].to_numpy()
step4 = LinearRegression(fit_intercept=False).fit(X4_tr, y4_tr)
ols4  = sm.OLS(y4_tr, X4_tr).fit()
print("\n=== Step 4: residual_3 ~ relative_dsle (no const) ===")
print(ols4.summary())

# ---------------- Predict / Evaluate (2019–2022, PGA R1/R2) ----------------
test = base.loc[base["year"].between(2019, 2022)].copy()

# Step 1
mask_f1 = test[FEATS1].notna().all(axis=1)
test1 = test.loc[mask_f1].copy()
test1["pred_step1"] = step1.predict(test1[FEATS1].to_numpy())

# Step 2
test1["pred_step2"] = 0.0
mask_f2 = test1[FEATS2].notna().all(axis=1)
if mask_f2.any():
    test1.loc[mask_f2, "pred_step2"] = step2.predict(test1.loc[mask_f2, FEATS2].to_numpy())

# Step 3
test1["pred_step3"] = 0.0
mask_f3 = test1[FEATS3].notna().all(axis=1)
if mask_f3.any():
    test1.loc[mask_f3, "pred_step3"] = step3.predict(test1.loc[mask_f3, FEATS3].to_numpy())

# Step 4
test1["pred_step4"] = 0.0
mask_f4 = test1[FEATS4].notna().all(axis=1)
if mask_f4.any():
    test1.loc[mask_f4, "pred_step4"] = step4.predict(test1.loc[mask_f4, FEATS4].to_numpy())

# Combined predictions
test1["pred_step12"]  = test1["pred_step1"] + test1["pred_step2"]
test1["pred_step123"] = test1["pred_step12"] + test1["pred_step3"]
test1["pred_step1234"]= test1["pred_step123"] + test1["pred_step4"]

ok = test1[TARGET].notna()
y_true = test1.loc[ok, TARGET].to_numpy()
p1  = test1.loc[ok, "pred_step1"].to_numpy()
p12 = test1.loc[ok, "pred_step12"].to_numpy()
p123= test1.loc[ok, "pred_step123"].to_numpy()
p1234=test1.loc[ok, "pred_step1234"].to_numpy()

print("\nStep 1 metrics:",   {"R2": r2_score(y_true, p1),   "RMSE": mean_squared_error(y_true, p1, squared=False),   "MAE": mean_absolute_error(y_true, p1)})
print("Step 1+2 metrics:",  {"R2": r2_score(y_true, p12),  "RMSE": mean_squared_error(y_true, p12, squared=False),  "MAE": mean_absolute_error(y_true, p12)})
print("Step 1-3 metrics:",  {"R2": r2_score(y_true, p123), "RMSE": mean_squared_error(y_true, p123, squared=False), "MAE": mean_absolute_error(y_true, p123)})
print("Step 1-4 metrics:",  {"R2": r2_score(y_true, p1234),"RMSE": mean_squared_error(y_true, p1234, squared=False),"MAE": mean_absolute_error(y_true, p1234)})


df=df_2


# ensure age_sq exists
df["age"] = pd.to_numeric(df["age"], errors="coerce")
if "age_sq" not in df.columns:
    df["age_sq"] = df["age"] * df["age"]

pred_mask = (pd.to_numeric(df["year"], errors="coerce") >= 2020) & df["tour"].eq("pga")
cols_needed = list(dict.fromkeys(FEATS1 + FEATS2 + FEATS3 + FEATS4))

future = df.loc[pred_mask, cols_needed].copy()

# start with NaNs (so carry-forward can detect "missing")
sg_pred_series = pd.Series(np.nan, index=future.index, dtype=float)

# Step 1 (required to use the model)
mask_f1 = future[FEATS1].notna().all(axis=1)
total = pd.Series(0.0, index=future.index, dtype=float)
total.loc[mask_f1] = step1.predict(future.loc[mask_f1, FEATS1].to_numpy())

# Step 2 (optional add-on)
mask_f2 = mask_f1 & future[FEATS2].notna().all(axis=1)
if mask_f2.any():
    total.loc[mask_f2] += step2.predict(future.loc[mask_f2, FEATS2].to_numpy())

# Step 3 (optional add-on)
mask_f3 = mask_f1 & future[FEATS3].notna().all(axis=1)
if mask_f3.any():
    total.loc[mask_f3] += step3.predict(future.loc[mask_f3, FEATS3].to_numpy())

# Step 4 (optional add-on)
mask_f4 = mask_f1 & future[FEATS4].notna().all(axis=1)
if mask_f4.any():
    total.loc[mask_f4] += step4.predict(future.loc[mask_f4, FEATS4].to_numpy())

# assign only where Step 1 was possible; leave others as NaN for now
sg_pred_series.loc[mask_f1] = total.loc[mask_f1]

# write back to df (preserve all rows)
df["sg_pred"] = df.get("sg_pred", np.nan)
df.loc[future.index, "sg_pred"] = sg_pred_series.values

# ---- carry R2 -> R3/R4 (override only when current sg_pred is NaN) ----
# ---- carry R2 -> R3/R4 (force override when R2 exists) ----
_keys = ["tour", "year", "event_id", "player_name"]
df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")
pred_mask = (pd.to_numeric(df["year"], errors="coerce") >= 2020) & df["tour"].eq("pga")

# map each player's R2 sg_pred within the event
r2_map = (
    df.loc[pred_mask & df["round_num"].eq(2), _keys + ["sg_pred"]]
      .dropna(subset=["sg_pred"])
      .rename(columns={"sg_pred": "sg_pred_r2"})
)

# merge R2 value back to all rows of that player's event
df = df.merge(r2_map, on=_keys, how="left")

# FORCE R3/R4 to use the R2 value whenever it exists
mask_r34_have_r2 = pred_mask & df["round_num"].isin([3, 4]) & df["sg_pred_r2"].notna()
df.loc[mask_r34_have_r2, "sg_pred"] = df.loc[mask_r34_have_r2, "sg_pred_r2"]

# final fallback AFTER carry-forward: fill remaining NaNs in 2023+ PGA with -2.05
df.loc[pred_mask & df["sg_pred"].isna(), "sg_pred"] = -2.45

# cleanup and recompute field means
df.drop(columns=["sg_pred_r2"], inplace=True)
df["field_pred"] = df.groupby(["tour", "year", "event_id", "round_num"])["sg_pred"].transform("mean")

df["round_num"] = pd.to_numeric(df["round_num"], errors="coerce")
df["skill"] = pd.to_numeric(df["skill"], errors="coerce")

conds = [
    df["round_num"].isin([1, 2]),
    df["round_num"].isin([3, 4]) & df["tour"].eq("pga") & df['cut'].eq("Cut"),
    df["round_num"].isin([3, 4]) & df["tour"].ne("pga") & df["skill"].notna() & df['cut'].eq("Cut"),
    df["round_num"].isin([3, 4]) & df["skill"].notna() & df['cut'].ne("Cut"),
]
choices = [
    df["skill"],
    df["field_pred"],
    df["skill"] + 0.4,
    df["skill"]
]

df["field_strength"] = np.select(conds, choices, default=np.nan)

# --- Build c_exp on df (efficient, no merge), then forward-fill within player-event-year ---

df = df.copy()
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
df["round_num"]  = pd.to_numeric(df["round_num"], errors="coerce")

# preserve original row order
df["_ord"] = np.arange(len(df))

# 1) Cumcount on PGA only by (player_name, course_num) ordered by date
df = df.sort_values(["player_name", "course_num", "round_date", "round_num"])
mask_pga = df["tour"].eq("pga")
df["c_exp"] = np.nan
df.loc[mask_pga, "c_exp"] = df.loc[mask_pga].groupby(["player_name", "course_num"]).cumcount()

# 2) Shift scale and apply only on R1
df["c_exp"] = df["c_exp"] - 6.5
df["c_exp"] = df["c_exp"].where(df["round_num"].eq(1))

# 3) Forward-fill within each player-event-year, ordered by date, then fill remaining with 0
df = df.sort_values(["player_name", "event_id", "year", "round_date", "round_num"])
df["c_exp"] = (
    df.groupby(["player_name", "event_id", "year"])["c_exp"]
      .ffill()
      .fillna(0)
)

# 4) Set to 0 for LIV
df.loc[df["tour"].eq("liv"), "c_exp"] = 0

# restore original row order
df = df.sort_values("_ord").drop(columns="_ord")

df = df[pd.to_numeric(df["year"], errors="coerce") >= 2020].copy()

df["sg_total"] = pd.to_numeric(df["sg_total"], errors="coerce")
df["field_strength"] = pd.to_numeric(df["field_strength"], errors="coerce")
df["sg_adj"] = df["sg_total"] + df["field_strength"]

# Replace column: drop old sg_total_adj and rename sg_adj -> sg_total_adj
df = df.drop(columns=["sg_total_adj"], errors="ignore")
df = df.rename(columns={"sg_adj": "sg_total_adj"})
# cutoff = pd.Timestamp("2025-10-22")
# df = df[pd.to_datetime(df["round_date"], errors="coerce").lt(cutoff)]


df = df.drop(columns=['course_name', 'course_par', 'gir', 'scrambling', 'great_shots', 'poor_shots','driving_acc', 'driving_dist', 'field_pred',
 'prox_fw', 'prox_rgh', 'event_name', 'score', 'start_hole', 'teetime', 'pro_time', 'skill', 'field_size','ema11_adj', 
 'ema21_adj', 'sma50_adj', 'ema21_ott','dsle','relative_dsle', 'ema21_app', 'ema21_arg', 'ema21_putt', 'ema21_rel_dd', 'ema21_rel_acc',], errors='ignore')

def _calc_dsle(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("round_date").copy()
    dsle = g["round_date"].diff().dt.days
    dsle = dsle.where(g["round_num"].eq(1), np.nan)   # only R1 starts a new dsle
    g["dsle"] = dsle.clip(upper=30)                   # cap at 30 days
    return g

tmp = (
    df[["player_name","tour","year","event_id","round_num","round_date"]]
      .groupby("player_name", group_keys=False)
      .apply(_calc_dsle)
)

# forward-fill within player so R2 inherits the R1 dsle (within each event)
tmp["dsle"] = tmp.groupby("player_name")["dsle"].ffill()

# merge back
df = df.drop(columns=["dsle"], errors="ignore").merge(
    tmp[["player_name","tour","year","event_id","round_num","round_date","dsle"]],
    on=["player_name","tour","year","event_id","round_num","round_date"],
    how="left"
)

df["dsle"] = pd.to_numeric(df["dsle"], errors="coerce").fillna(30.0)
df["field_avg_dsle"] = df.groupby(["tour","year","event_id"])["dsle"].transform("mean")
df["relative_dsle"] = (df["dsle"] - df["field_avg_dsle"]).fillna(0.0)
# Rebuild EMAs/SMAs for ALL 4 rounds and use field_strength/4 to adjust category inputs first

# ensure types
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
df["field_strength"] = pd.to_numeric(df["field_strength"], errors="coerce").fillna(0.0)
df["sg_total_adj"] = pd.to_numeric(df["sg_total_adj"], errors="coerce")

# --- Overall EMAs/SMAs on sg_total_adj (no mask; all rounds), shifted 1 to avoid lookahead ---
def _roll_feats_all(g: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(g["sg_total_adj"], errors="coerce")
    out = pd.DataFrame(index=g.index)
    out["ema11_adj"] = x.ewm(span=11, adjust=False, min_periods=1).mean().shift(1)
    out["ema21_adj"] = x.ewm(span=21, adjust=False, min_periods=1).mean().shift(1)
    out["sma50_adj"] = x.rolling(50, min_periods=1).mean().shift(1)
    return out

df[["ema11_adj","ema21_adj","sma50_adj"]] = (
    df.groupby("dg_id", group_keys=False).apply(_roll_feats_all)
)

# --- Category EMA(21) using adjusted inputs: raw + field_strength/4 (shifted 1; require ≥9 priors) ---
ADJ_COLS = [
    ("sg_ott",  "ema21_ott"),
    ("sg_app",  "ema21_app"),
    ("sg_arg",  "ema21_arg"),
    ("sg_putt", "ema21_putt"),
]
def _ema21_subcomponents_adj(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("round_date")
    fs_quarter = pd.to_numeric(g["field_strength"], errors="coerce").fillna(0.0) / 4.0
    out = pd.DataFrame(index=g.index)
    for raw_col, new_col in ADJ_COLS:
        x_raw = pd.to_numeric(g.get(raw_col), errors="coerce")
        x_adj = x_raw + fs_quarter
        ema = x_adj.ewm(span=21, adjust=False, min_periods=1).mean().shift(1)   # avoid lookahead
        prior_cnt = x_adj.notna().cumsum().shift(1, fill_value=0)               # require ≥9 prior vals
        out[new_col] = ema.where(prior_cnt >= 9, np.nan)
    return out

cols = [new for _, new in ADJ_COLS]
df[cols] = df.groupby("dg_id", group_keys=False).apply(_ema21_subcomponents_adj)

def _ema21_rel_simple(g):
    g = g.sort_values("round_date")
    out = pd.DataFrame(index=g.index)
    for raw_col, new_col in [("rel_dd","ema21_rel_dd"), ("rel_acc","ema21_rel_acc")]:
        x = pd.to_numeric(g.get(raw_col), errors="coerce")
        ema = x.ewm(span=21, adjust=False, min_periods=1).mean().shift(1)
        prior_cnt = x.notna().cumsum().shift(1, fill_value=0)
        out[new_col] = ema.where(prior_cnt >= 9, np.nan)
    return out

df[["ema21_rel_dd","ema21_rel_acc"]] = df.groupby("dg_id", group_keys=False).apply(_ema21_rel_simple)

# sort so cummax is chronological per player
df = df.sort_values(["dg_id","round_date","event_id","round_num"])

# cummax of the average(ema11_adj, sma50_adj) per player
df["max_skill"] = (
    ((pd.to_numeric(df["ema11_adj"], errors="coerce") + pd.to_numeric(df["sma50_adj"], errors="coerce")) / 2)
    .groupby(df["dg_id"]).cummax()
)

# cummax of sma50_adj per player
df["sma_50_max_skill"] = (
    pd.to_numeric(df["sma50_adj"], errors="coerce")
    .groupby(df["dg_id"]).cummax()
)

print(df.columns.tolist())

df_random_effects = pd.read_csv('course_random_effects_3.csv')

df = df[df['tour'] == 'pga']
df = df.merge(
    df_random_effects, 
    how='left', 
    on='course_num', 
    suffixes=("", "_courseRE")
)
df["course_adjustment"] = (
    df["ema21_putt"] * df["ema_21_sg_putt"] +
    df["ema21_app"]  * df["ema_21_sg_app"]  +
    df["ema21_rel_dd"]  * df["ema_21_rel_dd"]  +
    df["ema21_rel_acc"] * df["ema_21_rel_acc"] +
    df["ema21_arg"]  * df["ema_11_sg_arg"]
)

df["course_adjustment"] = df["course_adjustment"].fillna(0)

# ---- helper: clean event-frame before modeling (PGA only, no NaNs/Infs) ----
NUMERIC_REQ = [
    'ema11_adj','ema21_adj','sma50_adj',
    'ema21_ott','ema21_app','ema21_arg','ema21_putt',
    'age','age_sq','course_adjustment','relative_dsle','c_exp',
    'tourn_sg_avg_mcema'
]

def _prepare_event_frame_for_modeling(dfE: pd.DataFrame) -> pd.DataFrame:
    dfE = dfE.copy()
    # PGA only
    dfE = dfE[dfE['tour'].eq('pga')]

    # ensure numeric, drop NaN/Inf rows for required vars
    for c in NUMERIC_REQ:
        dfE[c] = pd.to_numeric(dfE[c], errors='coerce')
    dfE = dfE.replace([np.inf, -np.inf], np.nan).dropna(subset=NUMERIC_REQ)

    # bucket flag based on ema21_adj (>0.5 already set in _build_event_frame, but ensure it exists)
    if 'high_skill' not in dfE.columns:
        dfE['high_skill'] = (dfE['ema21_adj'] > 0.5).astype(int)

    return dfE
df.to_csv('field_adjusted_sg.csv')

import numpy as np, pandas as pd, statsmodels.api as sm, json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _skill_bucket_from_sma(x: float) -> str:
    x = pd.to_numeric(x, errors='coerce')
    if pd.isna(x):
        return 'MID'
    if x < -0.5:
        return 'LOW'
    elif x > 0.5:
        return 'HIGH'
    else:
        return 'MID'

def build_tavg_mcema_label(aligned_df: pd.DataFrame,
                           total_rounds: int = 4,
                           require_two_rounds: bool = True,
                           normalize_to_4: bool = False) -> pd.DataFrame:
    """
    Label using ONLY R1 & R2 actual rounds (no imputation for R3/R4).

    Returns the same columns as before, but:
      - 'rounds_played' now reflects R1/R2 rounds counted (1 or 2 unless filtered).
      - 'tourn_sg_sum_mcema' is the sum over R1/R2 only.
      - 'tourn_sg_avg_mcema' is the per-round average over the rounds present (R1/R2).
        If normalize_to_4=True, we scale the R1/R2 average to a 4-round equivalent
        by multiplying by (2/4)=0.5.

    Args:
        aligned_df: long rounds dataframe with columns:
            ['player_name','event_id','year','tour','round_num','round_date','sg_total_adj', ...]
        total_rounds: unused here (kept for signature compatibility).
        require_two_rounds: if True, only keep player-events with both R1 and R2 present.
        normalize_to_4: if True, convert the 2-round per-round avg to a 4-round equivalent.

    Returns:
        DataFrame with:
          ['player_name','event_id','year','tour','rounds_played',
           'tourn_sg_sum_mcema','tourn_sg_avg_mcema']
    """
    df = aligned_df.copy()
    df['round_date']   = pd.to_datetime(df['round_date'], errors='coerce')
    df['round_num']    = pd.to_numeric(df['round_num'], errors='coerce')
    df['sg_total_adj'] = pd.to_numeric(df['sg_total_adj'], errors='coerce')

    # Keep only R1 & R2
    r12 = (
        df.loc[df['round_num'].isin([1, 2])]
          .dropna(subset=['player_name','event_id','year','tour','round_num','sg_total_adj'])
          .sort_values(['player_name','event_id','year','tour','round_num','round_date'])
    )

    # Aggregate per player-event over rounds 1–2 only
    agg = (r12.groupby(['player_name','event_id','year','tour'], as_index=False)
              .agg(tourn_sg_sum_mcema=('sg_total_adj','sum'),
                   rounds_played=('sg_total_adj','size')))

    # Optionally enforce having both rounds
    if require_two_rounds:
        agg = agg.loc[agg['rounds_played'] >= 2].copy()

    # Per-round average over the available R1/R2 rounds
    # (No imputation; if only R1 present and require_two_rounds=False, it will be that single round)
    agg['tourn_sg_avg_mcema'] = agg['tourn_sg_sum_mcema'] / agg['rounds_played'].clip(lower=1)

    # Optional normalization to a "4-round equivalent" average
    # (Strictly speaking, a per-round average is invariant to # of rounds; this is only if you want
    # to scale *totals* conceptually to 4. Left here as an explicit option.)
    if normalize_to_4:
        # When using exactly 2 rounds, scaling average to 4-round average of totals/4
        # is equivalent to multiplying by 0.5 (sum/2 / 4 = sum / 8 = (avg/4)*2).
        # We just multiply the per-round average by (rounds_played/4).
        # With 2 rounds: factor = 2/4 = 0.5. With 1 round (if allowed): factor = 0.25.
        agg['tourn_sg_avg_mcema'] = agg['tourn_sg_avg_mcema'] * (agg['rounds_played'] / 4.0)

    # Return schema unchanged
    return agg[['player_name','event_id','year','tour','rounds_played',
                'tourn_sg_sum_mcema','tourn_sg_avg_mcema']]


# ---------- 1) Build event frame (R1 snapshot per event) ----------
def _build_event_frame_TAVG_mcema(df: pd.DataFrame) -> pd.DataFrame:
    tavg_lab = build_tavg_mcema_label(df, total_rounds=4)

    FEATS = [
        'ema11_adj','ema21_adj','sma50_adj',
        'max_skill','sma_50_max_skill',
        'ema21_ott','ema21_app','ema21_arg','ema21_putt',
        'age','age_sq','course_adjustment','relative_dsle','c_exp'
    ]
    pre_cols = ['player_name','event_id','year','tour','round_num','round_date'] + FEATS
    pre = (df.query('round_num == 1')[pre_cols]
             .sort_values('round_date')
             .groupby(['player_name','event_id','year','tour'], as_index=False)
             .tail(1)
             .rename(columns={'round_date':'event_date'}))

    dfE = pre.merge(tavg_lab, on=['player_name','event_id','year','tour'], how='inner')
    dfE = dfE.dropna(subset=['event_date','tourn_sg_avg_mcema']).copy()

    # 3-bucket split from sma50_adj
    dfE['skill_bucket'] = dfE['sma50_adj'].apply(_skill_bucket_from_sma)
    return dfE


def _fit_no_const(X, y):
    return sm.OLS(y, X, missing='drop', hasconst=False).fit()


# ---------- 2) Layered fit for TAVG ----------
def _layered_fit_TAVG(train_df, is_high: bool, label: str=""):
    y = train_df['tourn_sg_avg_mcema']

    # STEP 1: global form + caps
    step1_features = ['ema11_adj','ema21_adj','sma50_adj','sma_50_max_skill']
    X1 = train_df[step1_features]
    m1 = _fit_no_const(X1, y)
    pred1 = m1.predict(X1)

    # STEP 2: category EMAs
    r2 = y - pred1
    step2_features = ['ema21_ott','ema21_app','ema21_arg','ema21_putt']
    X2 = train_df[step2_features]
    m2 = _fit_no_const(X2, r2)
    pred2 = pred1 + m2.predict(X2)

    # STEP 3: age terms
    r3 = y - pred2
    X3 = train_df[['age','age_sq']].fillna(0.0)
    m3 = _fit_no_const(X3, r3)
    pred3 = pred2 + m3.predict(X3)

    # STEP 4: relative_dsle
    r4 = y - pred3
    X4 = train_df[['relative_dsle']].fillna(0.0)
    m4 = _fit_no_const(X4, r4)
    pred4 = pred3 + m4.predict(X4)

    # STEP 5: course_adjustment
    r5 = y - pred4
    X5 = train_df[['course_adjustment']].fillna(0.0)
    m5 = _fit_no_const(X5, r5)
    pred5 = pred4 + m5.predict(X5)

    # STEP 6: course experience
    r6 = y - pred5
    X6 = train_df[['c_exp']].fillna(0.0)
    m6 = _fit_no_const(X6, r6)
    pred6 = pred5 + m6.predict(X6)

    models = {'step1':m1,'step2':m2,'step3':m3,'step4':m4,'step5':m5,'step6':m6}
    featlist = {
        'step1': step1_features,
        'step2': step2_features,
        'step3': ['age','age_sq'],
        'step4': ['relative_dsle'],
        'step5': ['course_adjustment'],
        'step6': ['c_exp'],
    }
    return models, featlist


def _layered_predict_TAVG(test_df: pd.DataFrame, models, feats):
    X1 = test_df[feats['step1']]
    p1 = models['step1'].predict(X1)

    X2 = test_df[feats['step2']]
    p2 = p1 + models['step2'].predict(X2)

    X3 = test_df[feats['step3']].fillna(0.0)
    p3 = p2 + models['step3'].predict(X3)

    X4 = test_df[feats['step4']].fillna(0.0)
    p4 = p3 + models['step4'].predict(X4)

    X5 = test_df[feats['step5']].fillna(0.0)
    p5 = p4 + models['step5'].predict(X5)

    X6 = test_df[feats['step6']].fillna(0.0)
    p6 = p5 + models['step6'].predict(X6)

    return p1, p2, p3, p4, p5, p6


def _safe_hist_fit(y, x, name: str):
    x = pd.to_numeric(x, errors='coerce'); y = pd.to_numeric(y, errors='coerce')
    m = x.notna() & y.notna()
    if m.sum() < 2:
        return 0.0, 0.0, None
    X = sm.add_constant(x.loc[m], has_constant='add')
    mdl = sm.OLS(y.loc[m], X).fit()
    slope = float(mdl.params.get(name, 0.0))
    intercept = float(mdl.params.get('const', 0.0))
    return slope, intercept, mdl


# ---------- 3) Train w/ histories & 3 buckets ----------
def train_tavg_layered_with_histories(df: pd.DataFrame):
    dfE = _build_event_frame_TAVG_mcema(df).sort_values('event_date').reset_index(drop=True)
    dfE = _prepare_event_frame_for_modeling(dfE)
    if dfE.empty:
        raise ValueError("No PGA rows with complete features after filtering; cannot train.")

    outs = {}
    bucket_names = ['LOW','MID','HIGH']
    for bucket in bucket_names:
        sub = dfE[dfE['skill_bucket'].eq(bucket)].copy()
        if len(sub) < 5:
            print(f"[WARN] Skipping {bucket}: not enough rows ({len(sub)})")
            continue
        m, feats = _layered_fit_TAVG(sub, is_high=(bucket=="HIGH"), label=f"{bucket} Skill – TAVG-MCEMA")
        outs[bucket] = {'models': m, 'feats': feats}

    preds = []
    for bucket in bucket_names:
        if bucket not in outs: 
            continue
        m, feats = outs[bucket]['models'], outs[bucket]['feats']
        te = dfE[dfE['skill_bucket'].eq(bucket)].copy()
        a1,a2,a3,a4,a5,a6 = _layered_predict_TAVG(te, m, feats)
        te['pred_TAVG_step6'] = a6
        te['skill_bucket'] = bucket
        preds.append(te)

    if not preds:
        raise ValueError("No buckets trained; check thresholds/feature availability.")

    test_all = pd.concat(preds, axis=0).sort_values('event_date')

    # Overall metrics
    mask_eval = test_all[['tourn_sg_avg_mcema','pred_TAVG_step6']].notna().all(axis=1)
    y = test_all.loc[mask_eval, 'tourn_sg_avg_mcema'].to_numpy()
    p = test_all.loc[mask_eval, 'pred_TAVG_step6'].to_numpy()
    print("\n=== TAVG-MCEMA (3 buckets; FULL-SAMPLE / IN-SAMPLE) ===")
    print({'R2': r2_score(y, p),
           'RMSE': mean_squared_error(y, p, squared=False),
           'MAE': mean_absolute_error(y, p),
           'N': int(len(y))})

    # Per-bucket metrics
    for bucket in bucket_names:
        mb = mask_eval & test_all['skill_bucket'].eq(bucket).to_numpy()
        if mb.sum():
            yb = y[test_all['skill_bucket'].eq(bucket).loc[mask_eval].to_numpy()]
            pb = p[test_all['skill_bucket'].eq(bucket).loc[mask_eval].to_numpy()]
            print(f"  {bucket}: R2={r2_score(yb,pb):.3f}  RMSE={mean_squared_error(yb,pb,squared=False):.3f}  MAE={mean_absolute_error(yb,pb):.3f}  N={len(yb)}")

    # ----- Post-steps on ROUND residuals -----
    roundE = df.merge(
        test_all[['player_name','event_id','year','tour','pred_TAVG_step6','event_date']],
        on=['player_name','event_id','year','tour'], how='inner'
    )
    roundE['resid_vs_event'] = pd.to_numeric(roundE['sg_total_adj'], errors='coerce') - pd.to_numeric(roundE['pred_TAVG_step6'], errors='coerce')

    # Course history (R1 only)
    roundE = roundE.sort_values(['player_name','course_num','round_date'])
    roundE['course_resid_cumsum'] = roundE.groupby(['player_name','course_num'])['resid_vs_event'].cumsum()
    roundE['course_history'] = roundE.groupby(['player_name','course_num'])['course_resid_cumsum'].shift(1).fillna(0)
    roundE.loc[roundE['round_num'] != 1, 'course_history'] = np.nan
    roundE['course_history'] = (
        roundE.sort_values('round_date')
              .groupby(['player_name','event_id','year'])['course_history']
              .ffill().fillna(0)
    )

    # 3-bucket split for histories using sma50_adj
    roundE['skill_bucket'] = roundE['sma50_adj'].apply(_skill_bucket_from_sma)

    coef_course, intercept_course = {}, {}
    for bucket in bucket_names:
        cond = roundE['skill_bucket'].eq(bucket)
        mask = cond & roundE['round_num'].eq(1) & roundE['course_history'].notna()
        slope, intercept, model_c = _safe_hist_fit(
            roundE.loc[mask, 'resid_vs_event'],
            roundE.loc[mask, 'course_history'].rename('course_history'),
            'course_history'
        )
        coef_course[bucket] = slope
        intercept_course[bucket] = intercept
        if model_c is not None:
            print(model_c.summary())
        print(f"\nCourse history coef ({bucket}): {slope}  intercept: {intercept}")

    # Links (shared)
    links_ids = [2020046, 541, 2019006, 2018058, 2020102, 2019060, 2024134, 2023138, 2022122, 2019058, 100]
    mask_links = roundE['event_id'].isin(links_ids) & roundE['tour'].isin(['pga','euro'])
    links_df = roundE.loc[mask_links].copy().sort_values(['player_name','round_date'])
    if len(links_df):
        links_df['links_resid_cumsum'] = links_df.groupby('player_name')['resid_vs_event'].cumsum()
        links_df['links_history'] = links_df.groupby('player_name')['links_resid_cumsum'].shift(1).fillna(0)
        mL = links_df['round_num'].eq(1) & links_df['links_history'].notna()
        coef_links, intercept_links, model_links = _safe_hist_fit(
            links_df.loc[mL, 'resid_vs_event'],
            links_df.loc[mL, 'links_history'].rename('links_history'),
            'links_history'
        )
    else:
        coef_links, intercept_links, model_links = 0.0, 0.0, None
    print("\nLinks history coef (ALL):", coef_links, " intercept:", intercept_links)
    if model_links is not None:
        print(model_links.summary())

    # Majors (shared)
    major_ids = [535, 100, 14, 26, 33]
    mask_maj = (roundE['tour'] == 'pga') & (roundE['event_id'].isin(major_ids))
    maj_df = roundE.loc[mask_maj].copy().sort_values(['player_name','round_date'])
    if len(maj_df):
        maj_df['majors_resid_cumsum'] = maj_df.groupby('player_name')['resid_vs_event'].cumsum()
        maj_df['majors_history'] = maj_df.groupby('player_name')['majors_resid_cumsum'].shift(1).fillna(0)
        mM = maj_df['round_num'].eq(1) & maj_df['majors_history'].notna()
        coef_majors, intercept_majors, model_maj = _safe_hist_fit(
            maj_df.loc[mM, 'resid_vs_event'],
            maj_df.loc[mM, 'majors_history'].rename('majors_history'),
            'majors_history'
        )
    else:
        coef_majors, intercept_majors, model_maj = 0.0, 0.0, None
    print("\nMajors history coef (ALL):", coef_majors, " intercept:", intercept_majors)
    if model_maj is not None:
        print(model_maj.summary())

    # Save three buckets
    def _dump_bucket(path, models, feats, extras):
        out = {f'step{k}': mdl.params.to_dict() for k,(kk,mdl) in enumerate(models.items(), start=1)}
        out['features'] = feats
        out['extras']   = extras
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    if 'LOW' in outs:
        _dump_bucket("tavg_model_coeffs_low_skill.json",
                     outs['LOW']['models'], outs['LOW']['feats'],
                     {'course_history_coef':       coef_course.get('LOW', 0.0),
                      'course_history_intercept':  intercept_course.get('LOW', 0.0),
                      'links_history_coef':        coef_links,
                      'links_history_intercept':   intercept_links,
                      'majors_history_coef':       coef_majors,
                      'majors_history_intercept':  intercept_majors})

    if 'MID' in outs:
        _dump_bucket("tavg_model_coeffs_mid_skill.json",
                     outs['MID']['models'], outs['MID']['feats'],
                     {'course_history_coef':       coef_course.get('MID', 0.0),
                      'course_history_intercept':  intercept_course.get('MID', 0.0),
                      'links_history_coef':        coef_links,
                      'links_history_intercept':   intercept_links,
                      'majors_history_coef':       coef_majors,
                      'majors_history_intercept':  intercept_majors})

    if 'HIGH' in outs:
        _dump_bucket("tavg_model_coeffs_high_skill.json",
                     outs['HIGH']['models'], outs['HIGH']['feats'],
                     {'course_history_coef':       coef_course.get('HIGH', 0.0),
                      'course_history_intercept':  intercept_course.get('HIGH', 0.0),
                      'links_history_coef':        coef_links,
                      'links_history_intercept':   intercept_links,
                      'majors_history_coef':       coef_majors,
                      'majors_history_intercept':  intercept_majors})

    test_all[['player_name','event_id','year','tour','event_date',
              'tourn_sg_avg_mcema','pred_TAVG_step6','skill_bucket']].to_csv(
        'event_tavg_mcema_eval_step6.csv', index=False
    )
    return outs


# ---------- 4) Apply saved TAVG model to this week's field ----------
def apply_saved_tavg_to_field(df: pd.DataFrame, name_replacements: dict):
    df = df.sort_values(["dg_id","round_date"]).copy()
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    df["player_name_normalized"] = (
        df["player_name"].astype(str).str.strip().str.lower().replace(name_replacements)
    )

    try:
        player_df = pd.read_csv("field_updates.csv")
        field_names = (
            player_df["player_name"].astype(str).str.strip().str.lower()
            .replace(name_replacements).unique().tolist()
        )
    except Exception:
        field_names = None

    if field_names is not None:
        df = df[df["player_name_normalized"].isin(field_names)].copy()

    # Core features "now"
    df["sg_total_adj"]  = pd.to_numeric(df.get("sg_total_adj"), errors="coerce")
    df["field_strength"] = pd.to_numeric(df.get("field_strength"), errors="coerce").fillna(0.0)
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["age_sq"] = df["age"] ** 2

    def _roll_feats_now(g: pd.DataFrame) -> pd.DataFrame:
        x = pd.to_numeric(g["sg_total_adj"], errors="coerce")
        out = pd.DataFrame(index=g.index)
        out["ema11_adj"] = x.ewm(span=11, adjust=False, min_periods=1).mean()
        out["ema21_adj"] = x.ewm(span=21, adjust=False, min_periods=1).mean()
        out["sma50_adj"] = x.rolling(50, min_periods=1).mean()
        return out

    df[["ema11_adj","ema21_adj","sma50_adj"]] = (
        df.groupby("dg_id", group_keys=False).apply(_roll_feats_now)
    )

    ADJ_COLS = [("sg_ott","ema21_ott"), ("sg_app","ema21_app"),
                ("sg_arg","ema21_arg"), ("sg_putt","ema21_putt")]

    def _ema21_sub_now(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("round_date")
        fs_q = pd.to_numeric(g["field_strength"], errors="coerce").fillna(0.0) / 4.0
        out = pd.DataFrame(index=g.index)
        for raw_col, new_col in ADJ_COLS:
            x = pd.to_numeric(g.get(raw_col), errors="coerce") + fs_q
            ema = x.ewm(span=21, adjust=False, min_periods=1).mean()
            prior_cnt = x.notna().cumsum()
            out[new_col] = ema.where(prior_cnt >= 9, np.nan)
        return out

    df[["ema21_ott","ema21_app","ema21_arg","ema21_putt"]] = (
        df.groupby("dg_id", group_keys=False).apply(_ema21_sub_now)
    )

    # caps
    df = df.sort_values(["dg_id","round_date","event_id","round_num"])
    df["max_skill"] = (
        ((pd.to_numeric(df["ema11_adj"], errors="coerce") +
          pd.to_numeric(df["sma50_adj"], errors="coerce")) / 2.0)
        .groupby(df["dg_id"]).cummax()
    )
    df["sma_50_max_skill"] = (
        pd.to_numeric(df["sma50_adj"], errors="coerce").groupby(df["dg_id"]).cummax()
    )

    # dsle today + relative
    today = pd.Timestamp.today().normalize()
    last_round = (
        df.groupby("player_name", as_index=False)["round_date"].max()
          .rename(columns={"round_date":"last_round_date"})
    )
    last_round["dsle_today"] = (
        (today - last_round["last_round_date"]).dt.days.clip(lower=0, upper=30)
    ).fillna(30)

    lr = last_round.copy()
    lr["player_name_normalized"] = (
        lr["player_name"].astype(str).str.strip().str.lower().replace(name_replacements)
    )
    if field_names is not None:
        lr = lr[lr["player_name_normalized"].isin(field_names)]
    field_avg_today = lr["dsle_today"].mean()
    lr["relative_dsle"] = lr["dsle_today"] - field_avg_today

    latest = df.groupby("player_name", as_index=False).tail(1).copy()
    latest = latest.drop(columns=["relative_dsle", "dsle_today"], errors="ignore")
    latest = latest.merge(
        lr[["player_name", "dsle_today", "relative_dsle"]],
        on="player_name", how="left",
    )
    latest["dsle_today"]     = pd.to_numeric(latest["dsle_today"], errors="coerce").fillna(30.0)
    latest["relative_dsle"]  = pd.to_numeric(latest["relative_dsle"], errors="coerce").fillna(0.0)

    # Snapshot (3-bucket)
    snap = latest[['player_name','player_name_normalized','sma50_adj']].copy()
    snap['skill_bucket'] = snap['sma50_adj'].apply(_skill_bucket_from_sma)
    snap = snap.drop_duplicates(['player_name','player_name_normalized'], keep='last')
    snap.to_csv('field_bucket_snapshot.csv', index=False)

    for col in ["course_adjustment","c_exp"]:
        if col not in latest.columns:
            latest[col] = 0.0
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0.0)

    # Load bucket coeffs
    import json
    with open("tavg_model_coeffs_low_skill.json",'r') as fl:
        coeffs_low  = json.load(fl)
    with open("tavg_model_coeffs_mid_skill.json",'r') as fm:
        coeffs_mid  = json.load(fm)
    with open("tavg_model_coeffs_high_skill.json",'r') as fh:
        coeffs_high = json.load(fh)

    # Split by bucket
    latest['skill_bucket'] = latest['sma50_adj'].apply(_skill_bucket_from_sma)
    idx_low = latest['skill_bucket'].eq('LOW')
    idx_mid = latest['skill_bucket'].eq('MID')
    idx_high= latest['skill_bucket'].eq('HIGH')

    def _apply_bucket(lat: pd.DataFrame, coeffs: dict) -> pd.DataFrame:
        lat = lat.copy()
        preds = {}
        running = pd.Series(0.0, index=lat.index, dtype=float)

        def _step_contrib(step_key: str, feats: list) -> pd.Series:
            X = (lat.reindex(columns=feats)
                   .apply(pd.to_numeric, errors='coerce')
                   .fillna(0.0))
            beta = pd.Series(coeffs[step_key]).reindex(feats).fillna(0.0)
            return (X * beta).sum(axis=1)

        for k in [1,2,3,4,5,6]:
            skey = f"step{k}"
            feats = coeffs['features'][skey]
            contrib = _step_contrib(skey, feats)
            preds[f'pred_step{k}'] = running + contrib
            running = preds[f'pred_step{k}']

        export = lat[['player_name_normalized']].copy()
        for skey in ['step1', 'step2', 'step3', 'step4']:
            feats = coeffs['features'][skey]
            beta  = pd.Series(coeffs[skey]).reindex(feats).fillna(0.0)
            for f in feats:
                export[f] = pd.to_numeric(lat[f], errors='coerce').fillna(0.0) * float(beta[f])

        # (kept consistent with your pipeline)
        export['pred_TAVG_step6'] = preds['pred_step4']
        return export

    out_parts = []
    if idx_low.any():  out_parts.append(_apply_bucket(latest[idx_low],  coeffs_low))
    if idx_mid.any():  out_parts.append(_apply_bucket(latest[idx_mid],  coeffs_mid))
    if idx_high.any(): out_parts.append(_apply_bucket(latest[idx_high], coeffs_high))
    out = pd.concat(out_parts, axis=0) if out_parts else pd.DataFrame(columns=['player_name_normalized','pred_TAVG_step6'])

    used_names = out['player_name_normalized'].unique()
    latest_used = latest[latest['player_name_normalized'].isin(used_names)].copy()
    wanted_cols = ['player_name','player_name_normalized','round_date',
                   'event_id','year','tour','course_num','sg_total_adj','relative_dsle','sma50_adj']
    existing_cols = [c for c in wanted_cols if c in latest_used.columns]
    latest_used[existing_cols].sort_values('player_name_normalized').to_csv('latest_rows_used.csv', index=False)
    print(f"Wrote latest_rows_used.csv with {len(latest_used)} rows.")

    comp_cols = [c for c in out.columns if c not in ['player_name_normalized','pred_TAVG_step6']]
    cols = ['player_name_normalized','pred_TAVG_step6'] + comp_cols

    # Floor rule: if ema21_adj == 0 -> pred = -2.4 (unchanged)
    if 'ema21_adj' in out.columns:
        out['pred_TAVG_step6'] = np.where(
            pd.to_numeric(out['ema21_adj'], errors='coerce').fillna(0.0).eq(0.0),
            -2.4,
            out['pred_TAVG_step6']
        )

    out[cols].to_csv('latest_preds_new.csv', index=False)
    print("\nWrote latest_preds_new.csv with", len(out), "rows.")
    return out[cols]


# ===================== RUN (switch to df everywhere) =====================
# 1) Train TAVG-MCEMA layered model (no gapfactor) and fit post-steps (course/links/majors)
_ = train_tavg_layered_with_histories(df)

# 2) Apply the saved TAVG-MCEMA model to this week's field and write latest_preds_new.csv
_ = apply_saved_tavg_to_field(df, name_replacements)

# After training (if you still want the returned objects)
outs = train_tavg_layered_with_histories(df)
def dump_layered_ols_summaries(outs, to_file=None):
    """
    Print (and optionally save) the step-by-step OLS summaries
    for LOW and HIGH skill buckets of the TAVG layered model.
    """
    fh = open(to_file, "w", encoding="utf-8") if to_file else None

    def w(txt):
        print(txt)
        if fh:
            fh.write(txt + ("\n" if not txt.endswith("\n") else ""))

    for bucket in ["LOW", "HIGH"]:
        w(f"\n=== {bucket} Skill – TAVG (step-by-step) ===")
        for step in ["step1", "step2", "step3", "step4", "step5"]:
            mdl = outs[bucket]["models"][step]   # statsmodels OLSResults
            w(f"\n--- {bucket} {step} ---\n")
            w(mdl.summary().as_text())

    if fh:
        fh.close()
        print(f"\nWrote full summaries to {to_file}")

# Print to console
dump_layered_ols_summaries(outs)

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---------- 5) Weekly adjustments (3-bucket) ----------
def write_weekly_adjustments(df: pd.DataFrame,
                             name_replacements: dict,
                             course_id: int,
                             field_csv: str = "field_updates.csv",
                             preds_csv: str = "event_tavg_mcema_eval_step6.csv",
                             coef_low_json:  str = "tavg_model_coeffs_low_skill.json",
                             coef_mid_json:  str = "tavg_model_coeffs_mid_skill.json",
                             coef_high_json: str = "tavg_model_coeffs_high_skill.json",
                             bucket_snapshot_csv: str = "field_bucket_snapshot.csv",
                             out_csv: str = "weekly_adjustments.csv") -> pd.DataFrame:
    import json

    with open(coef_low_json, "r") as fl:
        L = json.load(fl).get("extras", {})
    with open(coef_mid_json, "r") as fm:
        M = json.load(fm).get("extras", {})
    with open(coef_high_json, "r") as fh:
        H = json.load(fh).get("extras", {})

    course_params = {
        "LOW":  {"coef": float(L.get("course_history_coef", 0.0)),
                 "const": float(L.get("course_history_intercept", 0.0))},
        "MID":  {"coef": float(M.get("course_history_coef", 0.0)),
                 "const": float(M.get("course_history_intercept", 0.0))},
        "HIGH": {"coef": float(H.get("course_history_coef", 0.0)),
                 "const": float(H.get("course_history_intercept", 0.0))}
    }
    # Shared links/majors
    links_coef   = float(H.get("links_history_coef",   M.get("links_history_coef",   L.get("links_history_coef",   0.0))))
    links_const  = float(H.get("links_history_intercept", M.get("links_history_intercept", L.get("links_history_intercept", 0.0))))
    majors_coef  = float(H.get("majors_history_coef",  M.get("majors_history_coef",  L.get("majors_history_coef",  0.0))))
    majors_const = float(H.get("majors_history_intercept", M.get("majors_history_intercept", L.get("majors_history_intercept", 0.0))))

    links_ids  = {2020046, 541, 2019006, 2018058, 2020102, 2019060, 2024134, 2023138, 2022122, 2019058, 100}
    major_ids  = {535, 100, 14, 26, 33}
    is_links_course  = course_id in links_ids
    is_majors_course = course_id in major_ids
    if not is_links_course:
        links_coef, links_const = 0.0, 0.0
    if not is_majors_course:
        majors_coef, majors_const = 0.0, 0.0

    # Field list
    fld = pd.read_csv(field_csv)
    fld["player_name_normalized"] = (
        fld["player_name"].astype(str).str.strip().str.lower().replace(name_replacements)
    )
    field_names = set(fld["player_name_normalized"].tolist())

    # Normalize & restrict
    df = df.copy()
    df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")
    df["player_name_normalized"] = (
        df["player_name"].astype(str).str.strip().str.lower().replace(name_replacements)
    )
    df = df[df["player_name_normalized"].isin(field_names)]

    # Load event predictions
    evpred = pd.read_csv(preds_csv)[["player_name","event_id","year","tour","pred_TAVG_step6"]]
    roundE = df.merge(evpred, on=["player_name","event_id","year","tour"], how="inner")
    roundE["resid_vs_event"] = (
        pd.to_numeric(roundE["sg_total_adj"], errors="coerce") -
        pd.to_numeric(roundE["pred_TAVG_step6"], errors="coerce")
    )

    # Cumsums (to prior)
    tmpC = roundE.loc[roundE["course_num"].eq(course_id)].sort_values(["player_name","round_date"]).copy()
    if len(tmpC):
        tmpC["course_hist_cumsum"] = tmpC.groupby("player_name")["resid_vs_event"].cumsum()
        course_hist = tmpC.groupby("player_name", as_index=False).agg(course_hist_cumsum=("course_hist_cumsum","last"))
    else:
        course_hist = pd.DataFrame(columns=["player_name","course_hist_cumsum"])

    tmpL = roundE.loc[roundE["event_id"].isin(links_ids) & roundE["tour"].isin(["pga","euro"])] \
                  .sort_values(["player_name","round_date"]).copy()
    if len(tmpL):
        tmpL["links_hist_cumsum"] = tmpL.groupby("player_name")["resid_vs_event"].cumsum()
        links_hist = tmpL.groupby("player_name", as_index=False).agg(links_hist_cumsum=("links_hist_cumsum","last"))
    else:
        links_hist = pd.DataFrame(columns=["player_name","links_hist_cumsum"])

    tmpM = roundE.loc[(roundE["tour"].eq("pga")) & (roundE["event_id"].isin(major_ids))] \
                 .sort_values(["player_name","round_date"]).copy()
    if len(tmpM):
        tmpM["majors_hist_cumsum"] = tmpM.groupby("player_name")["resid_vs_event"].cumsum()
        majors_hist = tmpM.groupby("player_name", as_index=False).agg(majors_hist_cumsum=("majors_hist_cumsum","last"))
    else:
        majors_hist = pd.DataFrame(columns=["player_name","majors_hist_cumsum"])

    # Latest + bucket snapshot
    latest = (
        df.sort_values("round_date")
          .groupby("player_name", as_index=False)
          .tail(1)[["player_name","player_name_normalized","sma50_adj"]]
          .copy()
    )
    latest["skill_bucket"] = latest["sma50_adj"].apply(_skill_bucket_from_sma)
    try:
        snap = (pd.read_csv(bucket_snapshot_csv)
                  .loc[:, ["player_name","player_name_normalized","skill_bucket"]]
                  .rename(columns={"skill_bucket": "skill_bucket_snap"})
                  .drop_duplicates(["player_name","player_name_normalized"], keep="last"))
        latest = latest.merge(snap, on=["player_name","player_name_normalized"], how="left")
        latest["skill_bucket"] = latest["skill_bucket_snap"].combine_first(latest["skill_bucket"])
        latest = latest.drop(columns=["skill_bucket_snap"])
    except Exception:
        pass

    out = (latest[['player_name','player_name_normalized','skill_bucket']]
           .merge(course_hist, on="player_name", how="left")
           .merge(links_hist,  on="player_name", how="left")
           .merge(majors_hist, on="player_name", how="left"))

    for col in ["course_hist_cumsum","links_hist_cumsum","majors_hist_cumsum"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["course_coef"]  = out["skill_bucket"].map(lambda b: course_params[b]["coef"])
    out["course_const"] = out["skill_bucket"].map(lambda b: course_params[b]["const"])
    out["links_coef"]   = links_coef
    out["links_const"]  = links_const
    out["majors_coef"]  = majors_coef
    out["majors_const"] = majors_const

    out["course_adj_total"] = out["course_const"] + out["course_coef"] * out["course_hist_cumsum"]
    out["links_adj_total"]  = out["links_const"]  + out["links_coef"]  * out["links_hist_cumsum"]
    out["majors_adj_total"] = out["majors_const"] + out["majors_coef"] * out["majors_hist_cumsum"]
    out["total_history_adj"] = out["course_adj_total"] + out["links_adj_total"] + out["majors_adj_total"]

    out["course_id"]        = course_id
    out["is_links_course"]  = bool(is_links_course)
    out["is_majors_course"] = bool(is_majors_course)

    cols = [
        "player_name","player_name_normalized","skill_bucket","course_id",
        "course_hist_cumsum","course_coef","course_const","course_adj_total",
        "links_hist_cumsum","links_coef","links_const","links_adj_total","is_links_course",
        "majors_hist_cumsum","majors_coef","majors_const","majors_adj_total","is_majors_course",
        "total_history_adj"
    ]
    out = out[cols].sort_values(["skill_bucket","player_name_normalized"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows for course_id={course_id}")
    return out


write_weekly_adjustments(df, name_replacements, course_id)
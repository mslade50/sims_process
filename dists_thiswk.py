"""
Filter per-player SG distributions down to this week's field, and
PRINT who is in the field but missing from the distributions.

Inputs (same folder):
  - sg_dist_player.csv          # from prior step
  - field_updates.csv           # must contain 'player_name'
  - sim_inputs.py               # provides name_replacements

Outputs:
  - this_week_dists.csv
  - this_week_missing_players.csv   # field players with no dist rows
"""

import pandas as pd
from sim_inputs import name_replacements

IN_DISTS = "sg_dist_player.csv"
IN_FIELD = "field_updates.csv"
OUT_DISTS = "this_week_dists.csv"
OUT_MISSING = "this_week_missing_players.csv"

# ---------- helpers ----------
def normalize_name(name: str) -> str:
    key = str(name).strip().lower()
    return name_replacements.get(key, key)

# ---------- load ----------
dist = pd.read_csv(IN_DISTS)
field = pd.read_csv(IN_FIELD)

# sanity checks
if "player_name" not in dist.columns:
    raise ValueError("sg_dist_player.csv missing 'player_name'")
if "player_name" not in field.columns:
    raise ValueError("field_updates.csv missing 'player_name'")

# ---------- normalize names on BOTH sides ----------
dist["player_key"] = dist["player_name"].apply(normalize_name)
field["player_key"] = field["player_name"].apply(normalize_name)

# ---------- compute sets & missing ----------
dist_keys = set(dist["player_key"].dropna().unique())
field_keys = set(field["player_key"].dropna().unique())
missing_keys = sorted(field_keys - dist_keys)

# build a friendly missing table using the field file (original display names)
missing_df = (
    field[field["player_key"].isin(missing_keys)]
    .loc[:, ["player_name", "player_key"]]
    .drop_duplicates()
    .rename(columns={"player_name": "player_name_original",
                     "player_key": "player_name_normalized"})
    .sort_values(["player_name_normalized", "player_name_original"])
    .reset_index(drop=True)
)

# save + print missing info
missing_df.to_csv(OUT_MISSING, index=False)
if missing_df.empty:
    print("[ok] Everyone in the field has a distribution row.")
else:
    print(f"[info] Missing players with no distributions: {len(missing_df)}")
    # print names to console
    for nm in missing_df["player_name_original"]:
        print(f"  - {nm}")
    print(f"[info] Full list saved to {OUT_MISSING}")

# ---------- filter to just this week's field ----------
week = dist[dist["player_key"].isin(field_keys)].copy()

# require expected columns to dedupe by strongest sample
if "category" not in week.columns:
    raise ValueError("sg_dist_player.csv missing 'category'")
if "n" not in week.columns:
    raise ValueError("sg_dist_player.csv missing 'n' (observation count)")

# If multiple raw names collapse into one normalized key, keep largest 'n'
idx = week.groupby(["player_key", "category"])["n"].idxmax()
week = week.loc[idx].copy()

# keep original & set canonical name to normalized
week["player_name_original"] = week["player_name"]
week["player_name"] = week["player_key"]
week = week.drop(columns=["player_key"]).reset_index(drop=True)

# stable sort
if "category_clean" in week.columns:
    week = week.sort_values(["player_name", "category_clean"]).reset_index(drop=True)
else:
    week = week.sort_values(["player_name", "category"]).reset_index(drop=True)

# ---------- save ----------
week.to_csv(OUT_DISTS, index=False)
num_players = week["player_name"].nunique()
num_rows = len(week)
print(f"[ok] wrote {OUT_DISTS} ({num_players} players, {num_rows} rows)")


"""
Apply course shape adjustments to this week's per-player distributions.

Inputs (same folder):
  - this_week_dists.csv
  - course_shape_adjustments_<course_id>.csv   # course_id from sim_inputs.py
  - sim_inputs.py  (must define: course_id)

Output:
  - this_week_dists_adjusted.csv
"""

"""
Apply course shape adjustments to this week's per-player distributions
AND run sanity checks to flag extreme transforms.

Inputs (same folder):
  - this_week_dists.csv
  - course_shape_adjustments_<course_id>.csv
  - sim_inputs.py  (provides: course_id)

Output:
  - this_week_dists_adjusted.csv
  - this_week_dists_adjusted_anomalies.csv  (only if any flagged)
"""

import os
import numpy as np
import pandas as pd
from sim_inputs import course_id

# ---------------- config ----------------
REF_MODE = "vs_tour"  # or: "vs_avg_course_unweighted", "vs_avg_course_weighted"

IN_THIS_WEEK = "this_week_dists.csv"
IN_COURSE_ADJ = f"course_shape_adjustments_{course_id}.csv"
OUT_ADJUSTED = "this_week_dists_adjusted.csv"
OUT_ANOMALIES = "this_week_dists_adjusted_anomalies.csv"

# Sanity thresholds (tune if needed)
MAX_ABS_MEAN_SHIFT = 0.5      # strokes; warn if |delta_mu| > 0.5
MIN_SIGMA_RATIO    = 0.70     # warn if < 0.70
MAX_SIGMA_RATIO    = 1.50     # warn if > 1.50
MAX_ABS_Q99        = 10.0     # warn if any adjusted q01..q99 outside [-10,10]
MIN_DF_T_WARN      = 8.0      # warn if Student-t df < 8 (very heavy tails)
MIN_STD_ALLOWED    = 1e-6     # guardrail for non-positive std

# ---------------- helpers ----------------
def choose_ref_cols(mode: str):
    if mode == "vs_tour":
        return ("delta_mu_vs_tour", "sigma_ratio_vs_tour", "tail_ratio_vs_tour")
    if mode == "vs_avg_course_unweighted":
        return ("delta_mu_vs_avg_course_unweighted",
                "sigma_ratio_vs_avg_course_unweighted",
                "tail_ratio_vs_avg_course_unweighted")
    if mode == "vs_avg_course_weighted":
        return ("delta_mu_vs_avg_course_weighted",
                "sigma_ratio_vs_avg_course_weighted",
                "tail_ratio_vs_avg_course_weighted")
    raise ValueError(f"Unknown REF_MODE: {mode}")

def excess_to_student_t_df(excess_kurtosis: float) -> float:
    """For Student-t: excess kurtosis = 6/(ν-4), so ν = 6/exk + 4. exk<=0 => Normal (ν=inf)."""
    if excess_kurtosis is None or excess_kurtosis <= 0:
        return float("inf")
    nu = 6.0 / excess_kurtosis + 4.0
    return float(np.clip(nu, 4.5, 200.0))
# ---------------- load ----------------
if not os.path.exists(IN_THIS_WEEK):
    raise FileNotFoundError(f"Missing {IN_THIS_WEEK}")

week = pd.read_csv(IN_THIS_WEEK)

# sanity (this week dists)
for col in ["player_name", "category", "mean", "std", "excess_kurtosis"]:
    if col not in week.columns:
        raise ValueError(f"{IN_THIS_WEEK} missing '{col}'")

# ensure category_clean for join
if "category_clean" not in week.columns:
    week["category_clean"] = week["category"].astype(str).str.replace("_adj", "", regex=False)

# ---------------- merge & transform ----------------
if os.path.exists(IN_COURSE_ADJ):
    course_adj = pd.read_csv(IN_COURSE_ADJ)
    delta_col, sigma_col, tail_col = choose_ref_cols(REF_MODE)
    for col in ["category", delta_col, sigma_col, tail_col]:
        if col not in course_adj.columns:
            raise ValueError(f"{IN_COURSE_ADJ} missing '{col}'")
    adj = course_adj[["category", delta_col, sigma_col, tail_col]].rename(
        columns={
            "category": "category_clean",
            delta_col: "delta_mu",
            sigma_col: "sigma_ratio",
            tail_col:  "tail_ratio_course"
        }
    )
    merged = week.merge(adj, on="category_clean", how="left")
else:
    print(f"[warn] {IN_COURSE_ADJ} not found — using identity adjustments (no course effect).")
    merged = week.copy()
    merged["delta_mu"] = 0.0
    merged["sigma_ratio"] = 1.0
    merged["tail_ratio_course"] = 0.0

# fill missing adjustments with no-ops (covers categories not listed in course file)
merged["delta_mu"] = merged["delta_mu"].fillna(0.0)
merged["sigma_ratio"] = merged["sigma_ratio"].fillna(1.0)
merged["tail_ratio_course"] = merged["tail_ratio_course"].fillna(0.0)

# location–scale warp
merged["mean_adj"] = merged["mean"] + merged["delta_mu"]
merged["std_adj"]  = merged["std"] * merged["sigma_ratio"]

# quantiles: q' = mu' + sigma_ratio * (q - mu)
qcols = [c for c in merged.columns if c.startswith("q") and c[1:].isdigit()]
for qc in qcols:
    merged[qc + "_adj"] = merged["mean_adj"] + merged["sigma_ratio"] * (merged[qc] - merged["mean"])

# heavier tails? convert target excess kurtosis to Student-t df
merged["target_exk"] = np.maximum(0.0, merged["excess_kurtosis"] + merged["tail_ratio_course"])
merged["df_t"] = merged["target_exk"].apply(excess_to_student_t_df)
merged["distribution"] = np.where(np.isfinite(merged["df_t"]), "student_t", "normal")

# ---------------- sanity checks ----------------
def check_row(r):
    reasons = []
    if abs(r["delta_mu"]) > MAX_ABS_MEAN_SHIFT:
        reasons.append(f"|delta_mu|>{MAX_ABS_MEAN_SHIFT:.2f}")
    if (r["sigma_ratio"] < MIN_SIGMA_RATIO) or (r["sigma_ratio"] > MAX_SIGMA_RATIO):
        reasons.append(f"sigma_ratio∉[{MIN_SIGMA_RATIO:.2f},{MAX_SIGMA_RATIO:.2f}]")
    if (r["std"] is not np.nan and r["std_adj"] <= MIN_STD_ALLOWED):
        reasons.append("std_adj<=0")
    qadj_vals = [r[c + "_adj"] for c in qcols if (c + "_adj") in merged.columns and pd.notna(r[c + "_adj"])]
    if len(qadj_vals):
        if (np.nanmin(qadj_vals) < -MAX_ABS_Q99) or (np.nanmax(qadj_vals) > MAX_ABS_Q99):
            reasons.append(f"adj_quantiles_outside_±{MAX_ABS_Q99}")
        ordered = [r[c + "_adj"] for c in sorted(qcols, key=lambda s: int(s[1:])) if pd.notna(r[c + "_adj"])]
        if any(ordered[i] > ordered[i+1] for i in range(len(ordered)-1)):
            reasons.append("adj_quantiles_not_monotonic")
    if np.isfinite(r["df_t"]) and r["df_t"] < MIN_DF_T_WARN:
        reasons.append(f"df_t<{MIN_DF_T_WARN:.0f}")
    return "; ".join(reasons) if reasons else ""

merged["sanity_flags"] = merged.apply(check_row, axis=1)
anoms = merged[merged["sanity_flags"] != ""].copy()

if not anoms.empty:
    anoms_cols = ["player_name", "category_clean", "n", "mean", "std",
                  "delta_mu", "sigma_ratio", "mean_adj", "std_adj", "df_t", "sanity_flags"]
    (anoms[anoms_cols]
     .sort_values(["category_clean", "sanity_flags", "player_name"])
     .to_csv(OUT_ANOMALIES, index=False))
    print(f"[warn] {len(anoms)} rows flagged. See {OUT_ANOMALIES}")
else:
    print("[ok] no anomalies flagged by current thresholds.")

# ---------------- per-category summary ----------------
summ = (merged.groupby("category_clean")
        .agg(n_rows=("player_name", "size"),
             mean_delta_mu=("delta_mu", "mean"),
             p95_abs_delta_mu=("delta_mu", lambda x: np.percentile(np.abs(x), 95)),
             mean_sigma_ratio=("sigma_ratio", "mean"),
             p95_sigma_ratio=("sigma_ratio", lambda x: np.percentile(x, 95)),
             p05_sigma_ratio=("sigma_ratio", lambda x: np.percentile(x, 5)),
             tail_share=("df_t", lambda s: float(np.mean(np.isfinite(s))))
             )
        .reset_index())
print("\nPer-category transform summary:")
for _, r in summ.iterrows():
    print(f"  {r['category_clean']}: rows={int(r['n_rows'])}, "
          f"d_mu={r['mean_delta_mu']:+.3f} (|d_mu| p95={r['p95_abs_delta_mu']:.3f}), "
          f"sig_ratio mean={r['mean_sigma_ratio']:.3f} (p05={r['p05_sigma_ratio']:.3f}, p95={r['p95_sigma_ratio']:.3f}), "
          f"student-t share={r['tail_share']:.2%}")

# ---------------- save adjusted dists ----------------
keep_cols = (
    ["player_name", "player_name_original"] if "player_name_original" in merged.columns else ["player_name"]
)
keep_cols += [
    "category", "category_clean", "n", "n_eff",
    "mean", "std", "skew", "excess_kurtosis",
    "delta_mu", "sigma_ratio", "tail_ratio_course",
    "mean_adj", "std_adj", "df_t", "distribution", "sanity_flags"
]
keep_cols += [c for c in merged.columns if c.endswith("_adj") and c.startswith("q")]

out = merged[keep_cols].copy()
out.to_csv(OUT_ADJUSTED, index=False)

print(f"\n[ok] course_id={course_id} -> wrote {OUT_ADJUSTED} "
      f"({out['player_name'].nunique()} players; {out['category_clean'].nunique()} categories)")

# ============================================================
# FILE SYNC: Push 'this_week_dists_adjusted.csv' to Repo & Backup
# ============================================================
import shutil

# Use the variable defined earlier in THIS script
FILE_TO_SYNC = OUT_ADJUSTED 

print("\n" + "="*60)
print(f"SYNCING INPUT FILE: {FILE_TO_SYNC}")
print("="*60)

# The destination folders you requested
sync_targets = [
    r"C:\Users\mckin\OneDrive\sims_process",
    r"C:\Users\mckin\OneDrive\etr-golf-sims"
]

# Get the absolute path of the source file to prevent "SameFileError"
src_abs = os.path.abspath(FILE_TO_SYNC)

for target_dir in sync_targets:
    try:
        # 1. Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # 2. Define the full destination path
        dest_path = os.path.join(target_dir, FILE_TO_SYNC)
        dest_abs = os.path.abspath(dest_path)

        # 3. Copy only if the source and destination are effectively different locations
        if src_abs.lower() != dest_abs.lower():
            shutil.copy2(FILE_TO_SYNC, dest_path)
            print(f"[ok] Copied/Overwrote to: {target_dir}")
        else:
            print(f"[skip] Script is running inside {target_dir} (file already there)")

    except Exception as e:
        print(f"[err] Could not copy to {target_dir}: {e}")
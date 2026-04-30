"""
Filter per-player AND per-team SG distributions to this week's field, apply
course-shape adjustments, and sync the _adjusted files to etr-golf-sims.

Team-aware extension of ~/OneDrive/dists_thiswk.py. Used for partner events
(e.g. Zurich Classic) where we need both:
  - per-player distributions for individual simulations
  - per-team distributions for alternate-shot / team-event simulations

Inputs (CWD):
  - sg_dist_player.csv                       # from cat_dists_team.py (or cat_dists_player.py)
  - sg_dist_team.csv                         # from cat_dists_team.py (optional)
  - field_updates.csv                        # field this week, includes partner.player_name
  - course_shape_adjustments_{course_id}.csv # optional; identity if missing
  - sim_inputs.py                            # name_replacements, course_id

Outputs (CWD, pushed to etr-golf-sims for _adjusted):
  - this_week_dists.csv
  - this_week_dists_adjusted.csv             # -> etr-golf-sims
  - this_week_team_dists.csv                 # only if sg_dist_team.csv exists
  - this_week_team_dists_adjusted.csv        # -> etr-golf-sims
  - this_week_missing_players.csv

Team warp math (team stats are averages of partner stats — single alt-shot round):
  mean_adj_team = team_mean + delta_mu
  std_adj_team  = team_std  * sigma_ratio
  q_adj_team    = mean_adj_team + sigma_ratio * (q - team_mean)
  target_exk    = max(0, team_excess_kurtosis + tail_ratio_course)
"""

import os
import shutil
import numpy as np
import pandas as pd
from sim_inputs import name_replacements, course_id

# ---------------- config ----------------
REF_MODE = "vs_tour"  # or: "vs_avg_course_unweighted", "vs_avg_course_weighted"

IN_DISTS_PLAYER = "sg_dist_player.csv"
IN_DISTS_TEAM   = "sg_dist_team.csv"
IN_FIELD        = "field_updates.csv"
IN_COURSE_ADJ   = f"course_shape_adjustments_{course_id}.csv"

OUT_WEEK_PLAYER        = "this_week_dists.csv"
OUT_ADJUSTED_PLAYER    = "this_week_dists_adjusted.csv"
OUT_WEEK_TEAM          = "this_week_team_dists.csv"
OUT_ADJUSTED_TEAM      = "this_week_team_dists_adjusted.csv"
OUT_MISSING            = "this_week_missing_players.csv"

SYNC_TARGETS = [r"C:\Users\mckin\OneDrive\etr-golf-sims"]

# Sanity thresholds
MAX_ABS_MEAN_SHIFT = 0.5
MIN_SIGMA_RATIO    = 0.70
MAX_SIGMA_RATIO    = 1.50
MAX_ABS_Q99        = 20.0   # teams span 2x (±20), players ±10
MIN_DF_T_WARN      = 8.0
MIN_STD_ALLOWED    = 1e-6


# ---------------- helpers ----------------
def normalize_name(name):
    key = str(name).strip().lower()
    return name_replacements.get(key, key)


def choose_ref_cols(mode):
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


def excess_to_student_t_df(excess_kurtosis):
    if excess_kurtosis is None or excess_kurtosis <= 0:
        return float("inf")
    nu = 6.0 / excess_kurtosis + 4.0
    return float(np.clip(nu, 4.5, 200.0))


def load_course_adj():
    """Return (adj_df, source_tag). adj_df has columns: category_clean, delta_mu,
    sigma_ratio, tail_ratio_course. Falls back to identity if file missing."""
    if not os.path.exists(IN_COURSE_ADJ):
        print(f"[warn] {IN_COURSE_ADJ} not found — using identity adjustments (no course effect).")
        return None, "identity"
    raw = pd.read_csv(IN_COURSE_ADJ)
    delta_col, sigma_col, tail_col = choose_ref_cols(REF_MODE)
    for col in ["category", delta_col, sigma_col, tail_col]:
        if col not in raw.columns:
            raise ValueError(f"{IN_COURSE_ADJ} missing '{col}'")
    adj = raw[["category", delta_col, sigma_col, tail_col]].rename(
        columns={
            "category": "category_clean",
            delta_col: "delta_mu",
            sigma_col: "sigma_ratio",
            tail_col:  "tail_ratio_course",
        }
    )
    return adj, REF_MODE


def sync_to_targets(file_path):
    src_abs = os.path.abspath(file_path)
    for target_dir in SYNC_TARGETS:
        try:
            os.makedirs(target_dir, exist_ok=True)
            dest_path = os.path.join(target_dir, os.path.basename(file_path))
            if src_abs.lower() != os.path.abspath(dest_path).lower():
                shutil.copy2(file_path, dest_path)
                print(f"[ok] Copied {os.path.basename(file_path)} -> {target_dir}")
            else:
                print(f"[skip] Already in {target_dir}")
        except Exception as e:
            print(f"[err] Could not copy {file_path} to {target_dir}: {e}")


# ============================================================
# STEP 1: Filter per-player dists to this week's field
# ============================================================
print("=" * 60)
print("STEP 1: Filter per-player distributions to field")
print("=" * 60)

dist = pd.read_csv(IN_DISTS_PLAYER)
field = pd.read_csv(IN_FIELD)

for col, name in [(dist, "sg_dist_player.csv"), (field, "field_updates.csv")]:
    if "player_name" not in col.columns:
        raise ValueError(f"{name} missing 'player_name'")

dist["player_key"] = dist["player_name"].apply(normalize_name)
field["player_key"] = field["player_name"].apply(normalize_name)

dist_keys = set(dist["player_key"].dropna().unique())
field_keys = set(field["player_key"].dropna().unique())
missing_keys = sorted(field_keys - dist_keys)

missing_df = (
    field[field["player_key"].isin(missing_keys)]
    .loc[:, ["player_name", "player_key"]]
    .drop_duplicates()
    .rename(columns={"player_name": "player_name_original",
                     "player_key": "player_name_normalized"})
    .sort_values(["player_name_normalized", "player_name_original"])
    .reset_index(drop=True)
)
missing_df.to_csv(OUT_MISSING, index=False)
if missing_df.empty:
    print("[ok] Everyone in the field has a distribution row.")
else:
    print(f"[info] Missing players with no distributions: {len(missing_df)}")
    for nm in missing_df["player_name_original"]:
        print(f"  - {nm}")
    print(f"[info] Full list saved to {OUT_MISSING}")

week = dist[dist["player_key"].isin(field_keys)].copy()
if "category" not in week.columns or "n" not in week.columns:
    raise ValueError("sg_dist_player.csv missing 'category' or 'n'")

idx = week.groupby(["player_key", "category"])["n"].idxmax()
week = week.loc[idx].copy()
week["player_name_original"] = week["player_name"]
week["player_name"] = week["player_key"]
week = week.drop(columns=["player_key"]).reset_index(drop=True)

if "category_clean" in week.columns:
    week = week.sort_values(["player_name", "category_clean"]).reset_index(drop=True)
else:
    week = week.sort_values(["player_name", "category"]).reset_index(drop=True)

week.to_csv(OUT_WEEK_PLAYER, index=False)
print(f"[ok] wrote {OUT_WEEK_PLAYER} "
      f"({week['player_name'].nunique()} players, {len(week)} rows)")


# ============================================================
# STEP 2: Apply course-shape adjustments to per-player dists
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Course-shape adjustments (per-player)")
print("=" * 60)

adj_df, adj_tag = load_course_adj()

if "category_clean" not in week.columns:
    week["category_clean"] = week["category"].astype(str).str.replace("_adj", "", regex=False)

if adj_df is not None:
    merged = week.merge(adj_df, on="category_clean", how="left")
else:
    merged = week.copy()
    merged["delta_mu"] = 0.0
    merged["sigma_ratio"] = 1.0
    merged["tail_ratio_course"] = 0.0

merged["delta_mu"] = merged["delta_mu"].fillna(0.0)
merged["sigma_ratio"] = merged["sigma_ratio"].fillna(1.0)
merged["tail_ratio_course"] = merged["tail_ratio_course"].fillna(0.0)

merged["mean_adj"] = merged["mean"] + merged["delta_mu"]
merged["std_adj"]  = merged["std"]  * merged["sigma_ratio"]

qcols = [c for c in merged.columns if c.startswith("q") and c[1:].isdigit()]
for qc in qcols:
    merged[qc + "_adj"] = merged["mean_adj"] + merged["sigma_ratio"] * (merged[qc] - merged["mean"])

merged["target_exk"] = np.maximum(0.0, merged["excess_kurtosis"] + merged["tail_ratio_course"])
merged["df_t"] = merged["target_exk"].apply(excess_to_student_t_df)
merged["distribution"] = np.where(np.isfinite(merged["df_t"]), "student_t", "normal")


def _check_row(r, qcols, merged_cols, abs_q_lim):
    reasons = []
    if abs(r["delta_mu"]) > MAX_ABS_MEAN_SHIFT:
        reasons.append(f"|delta_mu|>{MAX_ABS_MEAN_SHIFT:.2f}")
    if (r["sigma_ratio"] < MIN_SIGMA_RATIO) or (r["sigma_ratio"] > MAX_SIGMA_RATIO):
        reasons.append(f"sigma_ratio∉[{MIN_SIGMA_RATIO:.2f},{MAX_SIGMA_RATIO:.2f}]")
    if pd.notna(r["std_adj"]) and r["std_adj"] <= MIN_STD_ALLOWED:
        reasons.append("std_adj<=0")
    qadj_vals = [r[c + "_adj"] for c in qcols
                 if (c + "_adj") in merged_cols and pd.notna(r[c + "_adj"])]
    if qadj_vals:
        if np.nanmin(qadj_vals) < -abs_q_lim or np.nanmax(qadj_vals) > abs_q_lim:
            reasons.append(f"adj_quantiles_outside_±{abs_q_lim}")
        ordered = [r[c + "_adj"] for c in sorted(qcols, key=lambda s: int(s[1:]))
                   if pd.notna(r[c + "_adj"])]
        if any(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)):
            reasons.append("adj_quantiles_not_monotonic")
    if np.isfinite(r["df_t"]) and r["df_t"] < MIN_DF_T_WARN:
        reasons.append(f"df_t<{MIN_DF_T_WARN:.0f}")
    return "; ".join(reasons) if reasons else ""


merged["sanity_flags"] = merged.apply(
    lambda r: _check_row(r, qcols, merged.columns, MAX_ABS_Q99 / 2), axis=1
)
anoms = merged[merged["sanity_flags"] != ""].copy()
if not anoms.empty:
    print(f"[warn] {len(anoms)} per-player rows flagged (see 'sanity_flags' column)")
else:
    print("[ok] no per-player anomalies flagged")

keep_cols = ["player_name", "player_name_original"] if "player_name_original" in merged.columns else ["player_name"]
keep_cols += [
    "category", "category_clean", "n", "n_eff",
    "mean", "std", "skew", "excess_kurtosis",
    "delta_mu", "sigma_ratio", "tail_ratio_course",
    "mean_adj", "std_adj", "df_t", "distribution", "sanity_flags",
]
keep_cols += [c for c in merged.columns if c.endswith("_adj") and c.startswith("q")]
out = merged[[c for c in keep_cols if c in merged.columns]].copy()
out.to_csv(OUT_ADJUSTED_PLAYER, index=False)
print(f"[ok] course_id={course_id} ({adj_tag}) -> wrote {OUT_ADJUSTED_PLAYER} "
      f"({out['player_name'].nunique()} players)")


# ============================================================
# STEP 3: Filter + adjust per-team dists (if present)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Team distributions")
print("=" * 60)

team_adjusted_written = False

if os.path.exists(IN_DISTS_TEAM):
    team = pd.read_csv(IN_DISTS_TEAM)

    # sg_dist_team.csv is already built from field_updates.csv pairs, so the
    # "filter to field" step is essentially a copy — but we still emit the
    # unadjusted file for parity with the per-player pipeline.
    team.to_csv(OUT_WEEK_TEAM, index=False)
    print(f"[ok] wrote {OUT_WEEK_TEAM} "
          f"({team['team_name'].nunique()} teams, {len(team)} rows)")

    if "category_clean" not in team.columns:
        team["category_clean"] = team["category"].astype(str).str.replace("_adj", "", regex=False)

    # Merge course adjustments (same per-category table as per-player)
    if adj_df is not None:
        t_merged = team.merge(adj_df, on="category_clean", how="left")
    else:
        t_merged = team.copy()
        t_merged["delta_mu"] = 0.0
        t_merged["sigma_ratio"] = 1.0
        t_merged["tail_ratio_course"] = 0.0

    t_merged["delta_mu"] = t_merged["delta_mu"].fillna(0.0)
    t_merged["sigma_ratio"] = t_merged["sigma_ratio"].fillna(1.0)
    t_merged["tail_ratio_course"] = t_merged["tail_ratio_course"].fillna(0.0)

    # Team warp: team mean is the average of partner means (single alt-shot
    # round), so applying the per-category course shift once to the team mean
    # gives the same result as applying it to each partner and averaging.
    # std scales by sigma_ratio unchanged.
    t_merged["mean_adj"] = t_merged["mean"] + t_merged["delta_mu"]
    t_merged["std_adj"]  = t_merged["std"]  * t_merged["sigma_ratio"]

    t_qcols = [c for c in t_merged.columns if c.startswith("q") and c[1:].isdigit()]
    for qc in t_qcols:
        t_merged[qc + "_adj"] = (
            t_merged["mean_adj"]
            + t_merged["sigma_ratio"] * (t_merged[qc] - t_merged["mean"])
        )

    t_merged["target_exk"] = np.maximum(0.0, t_merged["excess_kurtosis"] + t_merged["tail_ratio_course"])
    t_merged["df_t"] = t_merged["target_exk"].apply(excess_to_student_t_df)
    t_merged["distribution"] = np.where(np.isfinite(t_merged["df_t"]), "student_t", "normal")

    t_merged["sanity_flags"] = t_merged.apply(
        lambda r: _check_row(r, t_qcols, t_merged.columns, MAX_ABS_Q99), axis=1
    )
    t_anoms = t_merged[t_merged["sanity_flags"] != ""]
    if not t_anoms.empty:
        print(f"[warn] {len(t_anoms)} team rows flagged (see 'sanity_flags' column)")
    else:
        print("[ok] no team anomalies flagged")

    team_keep = [
        "team_name", "player_a", "player_b",
        "category", "category_clean", "n", "n_a", "n_b",
        "mean", "std", "skew", "excess_kurtosis",
        "delta_mu", "sigma_ratio", "tail_ratio_course",
        "mean_adj", "std_adj", "df_t", "distribution", "sanity_flags",
    ]
    team_keep += [c for c in t_merged.columns if c.endswith("_adj") and c.startswith("q")]
    t_out = t_merged[[c for c in team_keep if c in t_merged.columns]].copy()
    t_out.to_csv(OUT_ADJUSTED_TEAM, index=False)
    team_adjusted_written = True
    print(f"[ok] course_id={course_id} ({adj_tag}) -> wrote {OUT_ADJUSTED_TEAM} "
          f"({t_out['team_name'].nunique()} teams)")
else:
    print(f"[info] {IN_DISTS_TEAM} not found — skipping team step. "
          f"(Run cat_dists_team.py first for team-event weeks.)")


# ============================================================
# STEP 4: Sync _adjusted files to etr-golf-sims
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Sync to etr-golf-sims")
print("=" * 60)

sync_to_targets(OUT_ADJUSTED_PLAYER)
if team_adjusted_written:
    sync_to_targets(OUT_ADJUSTED_TEAM)

print("\n[done]")

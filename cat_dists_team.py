"""
EMA-weighted per-player SG distributions + TEAM (partner) distributions.

Used for alternate-shot / team events (e.g. Zurich Classic). Produces the
standard per-player outputs AND a team distribution file by convolving each
partner pair's per-category histograms.

Outputs (CWD):
  - sg_dist_player.csv       (one row per player x category — same as cat_dists_player.py)
  - this_week_dists_v2.csv   (filtered to this week's field)
  - sg_dist_team.csv         (one row per team x category; mean/std/skew/kurt/quantiles/hist from convolved PMF)

Notes:
- Team pairs read from field_updates.csv ('player_name' + 'partner.player_name').
- Each team's distribution assumes the two partners' SG draws are independent,
  so the convolved PMF gives the distribution of their sum.
- Bins for team histograms span 2x the single-player range (default: -20..20 in 0.25 steps).
- Teams with a missing partner distribution are skipped and reported.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from sim_inputs import name_replacements
from sheet_config import load_config as _load_cfg
_cfg = _load_cfg()
tourney = _cfg["tourney"]

# --------------------------
# Config
# --------------------------
DB_PATH     = "C:/Users/mckin/OneDrive/dg_historical.db"
CATS        = ["sg_ott_adj", "sg_app_adj", "sg_arg_adj", "sg_putt_adj"]
Q_LIST      = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
HIST_BINS   = np.arange(-10.0, 10.25, 0.25)  # fixed 1/4-stroke bins
CLIP_TAILS  = True
CLIP_RANGE  = (-8.0, 8.0)                    # applied if CLIP_TAILS=True
MIN_OBS     = 12
EMA_SPAN    = 50                             # recency emphasis (≈ half-life ~17 rounds)
OUT_CSV     = "sg_dist_player.csv"

# Added "sg_total_adj" purely for consistency, though Section 4 now uses external file
cols_needed = [
    "player_name", "tour", "year", "round_date",
    "sg_ott_adj", "sg_app_adj", "sg_arg_adj", "sg_putt_adj", "sg_total_adj"
]

# --------------------------
# 1) Load & Preprocess
# --------------------------
_FA_SG_PATHS = [
    "field_adjusted_sg.csv",
    "C:/Users/mckin/OneDrive/field_adjusted_sg.csv",
]
_fa_sg_path = next((p for p in _FA_SG_PATHS if os.path.exists(p)), None)
if _fa_sg_path:
    df = pd.read_csv(_fa_sg_path, usecols=cols_needed, parse_dates=["round_date"])
    print(f"[ok] loaded {_fa_sg_path}")
else:
    raise FileNotFoundError("field_adjusted_sg.csv not found in current directory or OneDrive.")

# If year isn't present in the CSV, derive it from round_date
if "year" not in df.columns or df["year"].isna().all():
    df["year"] = df["round_date"].dt.year

# Apply same filters as the SQL query
df = df.loc[
    (df["tour"].eq("pga")) &
    (df["year"] >= 2019) &
    (df["sg_app_adj"].notna())
].copy()

# Ensure proper dtypes
df["round_date"] = pd.to_datetime(df["round_date"], errors="coerce")

# Process categories + sg_total_adj
for c in CATS + ["sg_total_adj"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if CLIP_TAILS:
        df[c] = df[c].clip(*CLIP_RANGE)

# Keep rows that have at least one category present
keep_mask = pd.Series(False, index=df.index)
for c in CATS:
    keep_mask |= df[c].notna()
df = df[keep_mask].copy()

# --------------------------
# 2) Helpers
# --------------------------
def ema_normalized_weights(n: int, span: int) -> np.ndarray:
    """
    Build normalized EMA weights for a sequence of length n (ascending order),
    where the most recent observation (index n-1) gets the highest weight.
    """
    if n <= 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    # weight for index i (0..n-1, oldest..newest): decay^(n-1-i)
    p = np.arange(n, dtype=float)
    w = decay ** (n - 1 - p)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.full(n, 1.0 / n)
    return w / w_sum

def weighted_stats(x: np.ndarray, w: np.ndarray) -> dict:
    """
    Weighted mean/std/skew/excess-kurtosis (population-style using normalized weights).
    Assumes w is normalized to sum to 1.
    """
    # Guard
    m = float(np.sum(w * x))
    xc = x - m
    m2 = float(np.sum(w * xc**2))
    std = np.sqrt(m2) if m2 > 0 else 0.0

    # Higher moments
    if std > 0:
        m3  = float(np.sum(w * xc**3))
        m4  = float(np.sum(w * xc**4))
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
        ex_kurt = (m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0
    else:
        skew = 0.0
        ex_kurt = 0.0

    return {"mean": m, "std": std, "skew": skew, "excess_kurtosis": ex_kurt}

def weighted_quantiles(x: np.ndarray, w: np.ndarray, probs: list[float]) -> dict:
    """
    Weighted quantiles for sorted (x,w).
    Returns dict mapping 'q01', 'q05', ... to float.
    """
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    out = {}
    for q in probs:
        # first index where cdf >= q
        idx = np.searchsorted(cdf, q, side="left")
        if idx >= len(xs):
            idx = len(xs) - 1
        out[f"q{int(q*100):02d}"] = float(xs[idx])
    return out

def summarize_player_cat(df_cat: pd.DataFrame, cat: str) -> dict | None:
    """
    Build EMA-weighted summary + histogram for a player's single category.
    df_cat must have columns [round_date, cat] with cat non-null.
    Returns dict with n (raw), n_eff (effective), stats, quantiles, and hist json.
    """
    s = df_cat.dropna(subset=[cat]).sort_values("round_date")
    n_raw = int(len(s))
    if n_raw < MIN_OBS:
        return None

    x = s[cat].to_numpy(dtype=float)
    w = ema_normalized_weights(n_raw, EMA_SPAN)

    # weighted statistics
    stats = weighted_stats(x, w)

    # weighted quantiles
    qdict = weighted_quantiles(x, w, Q_LIST)

    # weighted histogram (scale to "counts-like" by raw N for interpretability)
    counts, _ = np.histogram(x, bins=HIST_BINS, weights=w * n_raw)
    hist_json = json.dumps(counts.astype(float).tolist())

    # effective sample size
    w_eff = float((w.sum() ** 2) / np.sum(w**2))  # with normalized w, simplifies to 1/sum(w^2)

    out = {
        "n": n_raw,
        "n_eff": w_eff,
        **stats,
        **qdict,
        "hist_counts_json": hist_json
    }
    return out

def build_player_dist_ema(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    For each player and each adjusted SG category:
      - Apply EMA weights by round_date
      - Keep only if n_raw >= MIN_OBS in that category
      - Compute weighted summary + histogram
    """
    rows = []
    for cat in CATS:
        # Work only with rows that have this category present
        df_sub = df_in[["player_name", "round_date", cat]].dropna(subset=[cat]).copy()
        # Group per player
        for player, g in df_sub.groupby("player_name", sort=False):
            res = summarize_player_cat(g, cat)
            if res is None:
                continue
            rows.append({
                "player_name": player,
                "category": cat,
                "category_clean": cat.replace("_adj", ""),  # e.g., sg_app
                "ema_span": EMA_SPAN,
                "clipped": int(CLIP_TAILS),
                "bins_min": float(HIST_BINS[0]),
                "bins_max": float(HIST_BINS[-1]),
                "bins_step": float(HIST_BINS[1] - HIST_BINS[0]),
                **res
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["player_name", "category_clean"]).reset_index(drop=True)
    return out

# --------------------------
# 3) Build & save main distributions
# --------------------------
player_df = build_player_dist_ema(df)
player_df.to_csv(OUT_CSV, index=False)

n_players = player_df["player_name"].nunique() if not player_df.empty else 0
print(f"[ok] wrote {OUT_CSV} ({len(player_df)} rows, {n_players} players, EMA span={EMA_SPAN})")

# --------------------------
# 4) Field Check: 'Hot' players with insufficient history
# --------------------------
from scipy.stats import norm, t as student_t

OVERRIDE_CSV = "permanent_data/manual_dist_overrides.csv"

print("\n" + "="*60)
print("FIELD CHECK: Players with small sample but pred_step2 > 0")
print("="*60)

fit_filename = f"pre_course_fit_{tourney}.csv"

if os.path.exists(fit_filename):
    try:
        # Load the fit file
        fit_df = pd.read_csv(fit_filename)
        
        # Verify columns exist
        if "pred_step2" in fit_df.columns and "player_name" in fit_df.columns:
            
            # Normalize names for comparison
            fit_df["player_lower"] = fit_df["player_name"].astype(str).str.lower().str.strip()
            dist_players = set(player_df["player_name"].str.lower().str.strip()) if not player_df.empty else set()
            
            # Filter: 
            # 1. Player NOT in the main distribution list (meaning they failed MIN_OBS)
            # 2. Player HAS a positive pred_step2
            mask_missing = ~fit_df["player_lower"].isin(dist_players)
            mask_hot = fit_df["pred_step2"] > 0
            
            hot_prospects = fit_df[mask_missing & mask_hot].copy()
            
            if not hot_prospects.empty:
                print(f"Found {len(hot_prospects)} players fitting criteria:\n")
                
                # Load override names for cross-reference
                override_players = set()
                if os.path.exists(OVERRIDE_CSV):
                    ov = pd.read_csv(OVERRIDE_CSV)
                    ov = ov[~ov["player_name"].astype(str).str.startswith("_ANCHOR")]
                    override_players = set(ov["player_name"].astype(str).str.lower().str.strip())
                
                # Sort by prediction strength
                hot_prospects = hot_prospects.sort_values("pred_step2", ascending=False)
                
                for _, row in hot_prospects.iterrows():
                    n_rounds = row.get("sample", "N/A")
                    tag = " <-- MANUAL OVERRIDE ACTIVE" if row["player_lower"] in override_players else " ** NO OVERRIDE"
                    print(f"{row['player_name']}, {row['pred_step2']:.3f}, {n_rounds}{tag}")
            else:
                print("[info] No players found who are missing from dists but have pred_step2 > 0.")
        else:
            print(f"[err] {fit_filename} is missing required columns (player_name, pred_step2).")

    except Exception as e:
        print(f"[err] Error processing {fit_filename}: {e}")
else:
    print(f"[warn] {fit_filename} not found. Skipping field check.")


# --------------------------
# 5) Manual Overrides: inject synthetic distributions for profiled players
# --------------------------
"""
Loads manual_dist_overrides.csv and generates full distribution rows
(quantiles + histogram) from just (mean, std, skew, excess_kurtosis).

Only injects for player×category combos NOT already in player_df,
so real data always wins once a player crosses MIN_OBS.
"""

def synthetic_row_from_params(player_name: str, cat: str,
                              mu: float, sigma: float,
                              skew_val: float = 0.0,
                              ex_kurt: float = 0.0) -> dict:
    """
    Build a full sg_dist_player row from (mean, std, skew, excess_kurtosis).
    Uses Normal or Student-t to generate quantiles + histogram.
    """
    # Choose distribution based on excess kurtosis
    if ex_kurt > 0:
        # Student-t: excess_kurtosis = 6/(nu-4) → nu = 6/exk + 4
        nu = max(6.0 / ex_kurt + 4.0, 4.5)
        # Student-t with loc/scale
        dist = student_t(df=nu, loc=mu, scale=sigma * np.sqrt((nu - 2) / nu))
    else:
        dist = norm(loc=mu, scale=sigma)

    # Quantiles
    qdict = {}
    for q in Q_LIST:
        qdict[f"q{int(q*100):02d}"] = float(dist.ppf(q))

    # Histogram: evaluate PDF at bin centers, scale to look like N=MIN_OBS counts
    bin_centers = (HIST_BINS[:-1] + HIST_BINS[1:]) / 2.0
    pdf_vals = dist.pdf(bin_centers)
    # Scale so total "counts" ≈ MIN_OBS (matches the interpretability of real data)
    bin_width = float(HIST_BINS[1] - HIST_BINS[0])
    counts = pdf_vals * bin_width * MIN_OBS
    hist_json = json.dumps(counts.tolist())

    return {
        "player_name": player_name,
        "category": cat,
        "category_clean": cat.replace("_adj", ""),
        "ema_span": EMA_SPAN,
        "clipped": int(CLIP_TAILS),
        "bins_min": float(HIST_BINS[0]),
        "bins_max": float(HIST_BINS[-1]),
        "bins_step": float(HIST_BINS[1] - HIST_BINS[0]),
        "n": 0,               # flag: 0 = synthetic / manual override
        "n_eff": 0.0,
        "mean": mu,
        "std": sigma,
        "skew": skew_val,
        "excess_kurtosis": ex_kurt,
        **qdict,
        "hist_counts_json": hist_json,
    }


if os.path.exists(OVERRIDE_CSV):
    overrides = pd.read_csv(OVERRIDE_CSV)
    
    # Filter out _ANCHOR reference rows (those are just for the user's eyes)
    overrides = overrides[~overrides["player_name"].astype(str).str.startswith("_ANCHOR")]
    
    # Build set of existing (player, category) for fast lookup
    existing_keys = set()
    if not player_df.empty:
        existing_keys = set(
            zip(player_df["player_name"].str.lower().str.strip(),
                player_df["category"].str.strip())
        )
    
    injected = 0
    skipped_existing = 0
    override_rows = []
    
    for _, row in overrides.iterrows():
        pname = str(row["player_name"]).strip().lower()
        cat   = str(row["category"]).strip()
        
        # Skip if real data already covers this player×category
        if (pname, cat) in existing_keys:
            skipped_existing += 1
            continue
        
        mu      = float(row.get("mean", 0.0))
        sigma   = float(row.get("std", 1.15))       # default to ~tour avg vol
        skew_v  = float(row.get("skew", 0.0))
        ex_kurt = float(row.get("excess_kurtosis", 0.0))
        
        synth = synthetic_row_from_params(pname, cat, mu, sigma, skew_v, ex_kurt)
        override_rows.append(synth)
        injected += 1
    
    if override_rows:
        override_df = pd.DataFrame(override_rows)
        player_df = pd.concat([player_df, override_df], ignore_index=True)
        player_df = player_df.sort_values(["player_name", "category_clean"]).reset_index(drop=True)
        
        # Re-save with overrides included
        player_df.to_csv(OUT_CSV, index=False)
        print(f"\n[overrides] Injected {injected} synthetic rows, "
              f"skipped {skipped_existing} (real data exists)")
    else:
        print(f"\n[overrides] No new rows to inject "
              f"(skipped {skipped_existing} with existing real data)")
else:
    print(f"\n[info] No {OVERRIDE_CSV} found. Skipping manual overrides.")


# --------------------------
# 5b) Full-distribution clones
# --------------------------
# For players with no usable history (e.g. partner-event additions), clone
# another player's full distribution row — histogram, quantiles, moments —
# under a new (normalized, lowercase) name. Operates after the parametric
# overrides step, so a clone wins only when no real/override row exists yet.
DIST_CLONES = {
    # dest (normalized, lowercase) : source (raw player_name in sg_dist_player.csv)
    "olesen, jacob skov": "Bauchou, Zachary",
    "couvra, martin":     "Bauchou, Zachary",
}

clone_rows = []
existing_names = set(player_df["player_name"].str.lower().str.strip()) if not player_df.empty else set()
for dest, src in DIST_CLONES.items():
    if dest in existing_names:
        print(f"[clone] '{dest}' already has distributions — skipping")
        continue
    src_rows = player_df[player_df["player_name"] == src]
    if src_rows.empty:
        print(f"[clone] source '{src}' not found — cannot clone to '{dest}'")
        continue
    for _, r in src_rows.iterrows():
        new_row = r.copy()
        new_row["player_name"] = dest
        new_row["n"] = 0       # flag: 0 = synthetic (same convention as parametric overrides)
        clone_rows.append(new_row)

if clone_rows:
    player_df = pd.concat([player_df, pd.DataFrame(clone_rows)], ignore_index=True)
    player_df = player_df.sort_values(["player_name", "category_clean"]).reset_index(drop=True)
    player_df.to_csv(OUT_CSV, index=False)
    print(f"[clone] Cloned {len(clone_rows)} rows from source players "
          f"({len({r['player_name'] for r in clone_rows})} dest player(s))")


# --------------------------
# 6) Filter to this week's field → V2-ready output
# --------------------------
# Per-player V2 filter uses pre_course_fit as field source (aligned with
# cat_dists_player.py and new_sim). Section 7 below still reads field_updates.csv
# for the partner.player_name pairing info that pre_course_fit doesn't carry.
IN_FIELD = f"pre_course_fit_{tourney}.csv"
OUT_V2 = "this_week_dists_v2.csv"

if os.path.exists(IN_FIELD):
    field = pd.read_csv(IN_FIELD)

    # Normalize names on both sides
    def normalize_name(name):
        key = str(name).strip().lower()
        return name_replacements.get(key, key)

    player_df["player_key"] = player_df["player_name"].apply(normalize_name)
    field["player_key"] = field["player_name"].apply(normalize_name)

    field_keys = set(field["player_key"].dropna().unique())

    # Filter to field
    week = player_df[player_df["player_key"].isin(field_keys)].copy()

    # Dedup: if multiple raw names collapse to one key, keep largest n
    idx = week.groupby(["player_key", "category"])["n"].idxmax()
    week = week.loc[idx].copy()
    week["player_name"] = week["player_key"]  # canonical normalized name

    # Select V2-relevant columns only
    v2_cols = ["player_name", "category_clean", "mean", "std", "skew",
               "excess_kurtosis", "n", "n_eff"]
    week = week[v2_cols].sort_values(["player_name", "category_clean"]).reset_index(drop=True)

    # Missing players report
    dist_keys = set(week["player_name"].unique())
    missing = sorted(field_keys - dist_keys)
    if missing:
        print(f"\n[v2 field] {len(missing)} field players missing distributions:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    week.to_csv(OUT_V2, index=False)
    print(f"\n[ok] wrote {OUT_V2} ({week['player_name'].nunique()} players, {len(week)} rows)")
else:
    print(f"\n[warn] {IN_FIELD} not found. Skipping V2 field filter.")


# --------------------------
# 7) Team (Partner) Distributions — parametric average of per-category stats
# --------------------------
# Read field_updates.csv, pair each player with their partner, and build a
# team distribution per SG category. Treating the team as a single-round alt-shot
# entity: each team stat = simple average of the two partners' stats.
#   mean_team = (mean_A + mean_B) / 2
#   std_team  = (std_A  + std_B)  / 2
#   skew_team = (skew_A + skew_B) / 2
#   exk_team  = (exk_A  + exk_B)  / 2
# Quantiles + histogram are generated parametrically (Normal / Student-t) from
# the averaged params via synthetic_row_from_params (same function used for
# manual overrides).

TEAM_OUT_CSV = "sg_dist_team.csv"

print("\n" + "="*60)
print("TEAM DISTRIBUTIONS (parametric average of partner stats)")
print("="*60)

if os.path.exists(IN_FIELD):
    field_team = pd.read_csv(IN_FIELD)

    if "partner.player_name" not in field_team.columns:
        print(f"[team] {IN_FIELD} missing 'partner.player_name' column. Skipping team dists.")
    else:
        field_team["p1"] = field_team["player_name"].apply(normalize_name)
        field_team["p2"] = field_team["partner.player_name"].apply(normalize_name)

        team_pairs = set()
        for _, r in field_team.iterrows():
            p1, p2 = r.get("p1"), r.get("p2")
            if pd.isna(p1) or pd.isna(p2):
                continue
            p1, p2 = str(p1).strip(), str(p2).strip()
            if not p1 or not p2 or p1 == "nan" or p2 == "nan":
                continue
            team_pairs.add(tuple(sorted([p1, p2])))

        print(f"[team] {len(team_pairs)} unique teams in {IN_FIELD}")

        # Index per-player per-category rows for fast lookup
        key_col = "player_key" if "player_key" in player_df.columns else "player_name"
        pdf_idx = {}
        for _, row in player_df.iterrows():
            k = str(row[key_col]).strip().lower()
            pdf_idx[(k, row["category"])] = row

        team_rows = []
        missing = []

        for (a, b) in sorted(team_pairs):
            team_name = f"{a} + {b}"
            for cat in CATS:
                rowA = pdf_idx.get((a, cat))
                rowB = pdf_idx.get((b, cat))
                if rowA is None or rowB is None:
                    who = [p for p, r in [(a, rowA), (b, rowB)] if r is None]
                    missing.append((team_name, cat, who))
                    continue

                # Average per-category stats
                mean_team = (float(rowA["mean"]) + float(rowB["mean"])) / 2.0
                std_team  = (float(rowA["std"])  + float(rowB["std"]))  / 2.0
                skew_team = (float(rowA["skew"]) + float(rowB["skew"])) / 2.0
                exk_team  = (float(rowA["excess_kurtosis"]) + float(rowB["excess_kurtosis"])) / 2.0

                # Generate parametric distribution row (quantiles + histogram)
                # from the averaged params — reuses the override path so the
                # saved row matches the same schema as real/synthetic players.
                synth = synthetic_row_from_params(
                    team_name, cat, mean_team, std_team, skew_team, exk_team
                )

                nA = int(rowA.get("n", 0) or 0)
                nB = int(rowB.get("n", 0) or 0)
                n_team = min(nA, nB) if (nA > 0 and nB > 0) else MIN_OBS

                # Overwrite the defaults from synth with team-specific metadata
                synth.update({
                    "team_name": team_name,
                    "player_a": a,
                    "player_b": b,
                    "n_a": nA,
                    "n_b": nB,
                    "n": n_team,
                })
                synth.pop("player_name", None)   # team file uses team_name
                team_rows.append(synth)

        team_df = pd.DataFrame(team_rows)
        if not team_df.empty:
            # Move team_name / player_a / player_b / n_a / n_b to the front
            front = ["team_name", "player_a", "player_b", "category", "category_clean",
                     "ema_span", "clipped", "bins_min", "bins_max", "bins_step",
                     "n_a", "n_b", "n", "mean", "std", "skew", "excess_kurtosis"]
            q_cols = [c for c in team_df.columns if c.startswith("q") and c[1:].isdigit()]
            tail = ["hist_counts_json"]
            ordered = [c for c in front if c in team_df.columns] + q_cols + [c for c in tail if c in team_df.columns]
            team_df = team_df[ordered].sort_values(["team_name", "category_clean"]).reset_index(drop=True)
            team_df.to_csv(TEAM_OUT_CSV, index=False)
            print(f"[team] wrote {TEAM_OUT_CSV} "
                  f"({len(team_df)} rows, {team_df['team_name'].nunique()} teams)")
        else:
            print(f"[team] No team rows built.")

        if missing:
            missing_teams = sorted({m[0] for m in missing})
            print(f"[team] {len(missing)} (team × category) skipped across "
                  f"{len(missing_teams)} team(s) — missing player distribution:")
            for m in missing_teams[:15]:
                whos = sorted({w for row in missing if row[0] == m for w in row[2]})
                cats = sorted({row[1].replace("_adj", "") for row in missing if row[0] == m})
                print(f"  - {m}: missing {', '.join(whos)} in {','.join(cats)}")
            if len(missing_teams) > 15:
                print(f"  ... and {len(missing_teams) - 15} more teams")
else:
    print(f"[team] {IN_FIELD} not found. Skipping team distributions.")


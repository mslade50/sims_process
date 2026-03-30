"""
EMA-weighted per-player SG distributions (adjusted metrics), PGA >= 2019.

Output (CWD):
  - sg_dist_player.csv  (one row per player x category)

Notes:
- Uses sg_*_adj columns to avoid distortions.
- Per-player, per-category EMA weighting by round_date (more recent rounds carry more weight).
- Keeps only players with at least MIN_OBS observations *in that category*.
- Histogram counts are *EMA-weighted* and stored as JSON (float counts) using fixed 0.25-stroke bins.
- [NEW] Checks pre_course_fit_{tourney}.csv for "Hot" players (pred_step2 > 0) who missed the MIN_OBS cut.
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
# 6) Filter to this week's field → V2-ready output
# --------------------------
IN_FIELD = "field_updates.csv"
OUT_V2 = "this_week_dists_v2.csv"

# Fetch field from API if file doesn't exist
if not os.path.exists(IN_FIELD):
    print(f"\n[v2 field] {IN_FIELD} not found — fetching from DataGolf API...")
    try:
        from api_utils import fetch_field_updates
        _api_key = os.getenv('DATAGOLF_API_KEY', 'c05ee5fd8f2f3b14baab409bd83c')
        _field_df = fetch_field_updates(_api_key)
        if _field_df is not None and not _field_df.empty:
            _field_df.to_csv(IN_FIELD, index=False)
            print(f"[ok] wrote {IN_FIELD} ({len(_field_df)} players)")
        else:
            print("[warn] API returned empty field. Cannot build V2 dists.")
    except Exception as e:
        print(f"[warn] Failed to fetch field from API: {e}")

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


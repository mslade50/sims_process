"""
Unified Live Stats Engine — Processes rounds 1-4.

Replaces:  live_stats.py, live_stats_r2.py, live_stats_r3.py, live_stats_r4.py
Absorbs:   Prediction creation logic from rd_1_sd_multicourse_sim.py

Usage:
  Pre-event (create R1 predictions with weather):
    python live_stats_engine.py          (with round=0 in Google Sheet)

  After R1 (skill update + R2 predictions):
    python live_stats_engine.py          (with round=1 in Google Sheet)

  After R2 or R3 (skill update, then predictions if tee times available):
    python live_stats_engine.py          (with round=2 or 3 in Google Sheet)
    # If tee times aren't ready, run again later — script auto-detects.

  After R4 (record-keeping only):
    python live_stats_engine.py          (with round=4 in Google Sheet)

Pipeline Summary:
  Pre-event → model_predictions_r1.csv
  After R1  → r1_live_model.csv + model_predictions_r2.csv
  After R2  → r2_live_model.csv + model_predictions_r3.csv (auto if tee times ready)
  After R3  → r3_live_model.csv + model_predictions_r4.csv (auto if tee times ready)
  After R4  → r4_live_model.csv (record-keeping only)
"""

import argparse
import os
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from patsy import dmatrix
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _resolve_csv(filename):
    """Find a CSV in root first, then dashboard_data/ fallback."""
    if os.path.exists(filename):
        return filename
    alt = os.path.join("dashboard_data", filename)
    if os.path.exists(alt):
        print(f"  [resolve] {filename} not in root, using {alt}")
        return alt
    return filename  # let caller handle FileNotFoundError


from sim_inputs import (
    tourney, course_par, event_ids, wind_override, baseline_wind,
    dew_calculation,
    # R1 coefficients (4 skill-based buckets)
    coefficients_r1_high, coefficients_r1_midh, coefficients_r1_midl, coefficients_r1_low,
    # R2 coefficients (3 position-based buckets)
    coefficients_r2, coefficients_r2_6_30, coefficients_r2_30_up,
    # R3/R4 coefficients (3 position-based buckets, SG-only)
    coefficients_r3, coefficients_r3_mid, coefficients_r3_high,
    name_replacements,
)

from api_utils import (
    fetch_live_stats, fetch_field_updates,
    calculate_average_wind, compute_wind_factor, clean_names,
    fetch_player_decompositions,
    fetch_historical_hourly_wind, blend_wind_with_climo,
    get_round_dates, climo_weight_for_lead,
    fetch_realized_wind,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_KEY = "c05ee5fd8f2f3b14baab409bd83c"

# Upper cap on the R2 residual fed into the R2->R3 fix-layer cubic.
# Must match round_sim.py / new_sim.py and the Rust kernel (round_cascade.rs,
# cascade.rs).
RESID_FIX_CAP = 6.0

# Wind/dewpoint arrays indexed by round (1-based; index 0 unused)
# Populated at runtime from Google Sheet config (_apply_sheet_overrides)
WIND_ARRAYS = {1: [], 2: [], 3: [], 4: []}
DEW_ARRAYS = {1: [], 2: [], 3: [], 4: []}

# --- Bayesian wind blending: blend forecast with climatological prior ---
# Climo weight scales with actual lead time (days until round).
# Wind arrays are populated from Sheet at runtime (_apply_sheet_overrides),
# so blending happens there. This block pre-fetches climo for later use.
try:
    import pandas as _pd_coords
    from sim_inputs import course_id as _course_id
    from datetime import datetime as _dt
    _coords_csv = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")
    _coords_df = _pd_coords.read_csv(_coords_csv)
    _coords_row = _coords_df[_coords_df["course_id"] == _course_id]
    _CLIMO_WIND = None
    _ROUND_DATES = None
    if not _coords_row.empty:
        _lat = float(_coords_row["lat"].iloc[0])
        _lon = float(_coords_row["lon"].iloc[0])
        _CLIMO_WIND = fetch_historical_hourly_wind(_lat, _lon, _dt.now().month)
        _ROUND_DATES = get_round_dates()
        if _CLIMO_WIND:
            print(f"[weather] Climo wind prior loaded for blending")
except Exception as _e:
    _CLIMO_WIND = None
    _ROUND_DATES = None
    print(f"[weather] Climo blend skipped: {_e}")
# Per-round expected scoring baselines. Formerly imported from sim_inputs
# (score_adj_r1..r4); migrated to the Google Sheet (expected_score_r1..r4) and
# populated at runtime by _apply_sheet_overrides(). Defaults are placeholders.
SCORE_ADJS = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

# Multi-course expected scoring adjustments (populated from Google Sheet).
# Index 0 = first course_x encountered in API data, index 1 = second, etc.
# For single-course weeks, only index 0 is used.
COURSE_SCORE_ADJS = None  # Set by _apply_sheet_overrides; None = use SCORE_ADJS

# Name-keyed course score map: {api_course_code: expected_score}
# Built from course_codes + expected_score_1/2/3 in _apply_sheet_overrides().
COURSE_SCORE_MAP = {}  # e.g. {"PB": 68.7, "SG": 69.9}


# ══════════════════════════════════════════════════════════════════════════════
# Coefficient Routing
# ══════════════════════════════════════════════════════════════════════════════
# This section maps round numbers to their bucketing strategy and coefficient
# dictionaries. The logic differs by round (see architecture doc):
#   R1: 4 skill-based buckets keyed on pre-event 'pred' value
#   R2: 3 position-based buckets with residual + SG terms
#   R3/R4: 3 position-based buckets with SG-only terms (no residual)
#
# Each coefficient dict maps a SHORT KEY (used internally) to a multiplier.
# The _map_key_to_column function translates these to actual DataFrame columns.

R1_BUCKETS = [
    # (label, mask_func, coefficients)
    ("high",  lambda df: df["pred"] > 1,                                coefficients_r1_high),
    ("midh",  lambda df: (df["pred"] > 0.5) & (df["pred"] <= 1),       coefficients_r1_midh),
    ("midl",  lambda df: (df["pred"] > -0.5) & (df["pred"] <= 0.5),    coefficients_r1_midl),
    ("low",   lambda df: df["pred"] <= -0.5,                            coefficients_r1_low),
]

R2_BUCKETS = [
    ("top",   lambda df: df["position"] < 6,                                     coefficients_r2),
    ("mid",   lambda df: (df["position"] >= 6) & (df["position"] <= 30),          coefficients_r2_6_30),
    ("low",   lambda df: df["position"] > 30,                                     coefficients_r2_30_up),
]

R3_BUCKETS = [
    ("top",   lambda df: df["position"] < 6,                                     coefficients_r3),
    ("mid",   lambda df: (df["position"] >= 6) & (df["position"] <= 20),          coefficients_r3_mid),
    ("low",   lambda df: df["position"] > 20,                                     coefficients_r3_high),
]

# R4 reuses R3 coefficients intentionally — we're predicting R4 from R3 data,
# and there is no "R5" to predict from R4.  R4 live stats exist for backtesting.
R4_BUCKETS = R3_BUCKETS

ROUND_BUCKETS = {1: R1_BUCKETS, 2: R2_BUCKETS, 3: R3_BUCKETS, 4: R4_BUCKETS}

# Column mapping: coefficient key → actual DataFrame column name
_R1_COL_MAP = {"residual": "residual", "residual2": "residual2", "ott": "sg_ott", "putt": "sg_putt"}
_R2_COL_MAP = {
    "residual": "residual_capped", "residual2": "residual2", "residual3": "residual3",
    "avg_ott": "sg_ott_avg", "avg_putt": "sg_putt_avg", "avg_app": "sg_app_avg",
    "avg_arg": "sg_arg_avg", "delta_app": "sg_app_delta",
}
_R3R4_COL_MAP = {
    "sg_ott_avg": "sg_ott_avg", "sg_putt_avg": "sg_putt_avg",
    "sg_app_avg": "sg_app_avg", "sg_arg_avg": "sg_arg_avg",
    "avg_great_shots": "great_shots_avg",
    # LEVEL term: indicator column (1.0 for positions 6-10 into the round) so the
    # generic column*coeff application yields a flat add for those players only
    "pos_6_10": "pos_6_10",
}

COL_MAPS = {1: _R1_COL_MAP, 2: _R2_COL_MAP, 3: _R3R4_COL_MAP, 4: _R3R4_COL_MAP}


def _map_key_to_column(key, round_num):
    """Translate a coefficient dict key to the actual DataFrame column."""
    return COL_MAPS.get(round_num, {}).get(key)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Data Loading & Merging
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge(round_num):
    """
    Fetch live stats, field updates, and merge with prior predictions.
    
    Returns the fully merged DataFrame ready for residual computation.
    Each round has a different merge strategy (see comments inline).
    """
    # --- Fetch live data ---
    include_score = (round_num == 1)  # R1 needs score for satellite course proxy
    include_course = (round_num <= 3)  # Multi-course possible R1-R3
    teetime_col = f"r{round_num}_teetime"

    df = fetch_live_stats(round_num, API_KEY, include_score=include_score)
    if df is None:
        raise RuntimeError(f"Failed to fetch live stats for round {round_num}")

    field = fetch_field_updates(API_KEY, teetime_col=teetime_col, include_course=include_course)
    if field is not None:
        df = pd.merge(df, field, on="player_name", how="left")
    else:
        print(f"Warning: Field updates unavailable. Tee times / course info may be missing.")

    # --- Round-specific merges ---
    if round_num == 1:
        df = _merge_r1(df)
    elif round_num == 2:
        df = _merge_r2(df)
    elif round_num in (3, 4):
        df = _merge_r3r4(df, round_num)

    # --- Common cleanup ---
    # Remove WD/DQ
    df = df[~df["position"].astype(str).str.contains("WD|DQ", na=False)]

    # For R3+, drop players who were cut (no sg_total = didn't play this round)
    if round_num >= 3:
        before = len(df)
        df = df.dropna(subset=["sg_total"])
        cut_count = before - len(df)
        if cut_count > 0:
            print(f"  Dropped {cut_count} cut/inactive players (no R{round_num} data)")

    # Position: strip "T" prefix, convert to numeric
    # (api_utils no longer pre-converts position, so "T5" survives intact here)
    df["position"] = (
        df["position"].astype(str).str.replace("T", "", regex=False)
        .str.replace("CUT", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce").fillna(999).astype(int)
    )

    # Indicator for the pos_6_10 LEVEL term (coefficients_r3_mid): during the R3
    # run df["position"] is the standing after R3 = going INTO R4, which is the
    # population the term was fit on.
    df["pos_6_10"] = df["position"].between(6, 10).astype(float)

    return df


def _merge_r1(df):
    """
    R1 merge: load model_predictions_r1.csv (created pre-event).
    Source: live_stats.py lines 148-162
    """
    preds = pd.read_csv(_resolve_csv("model_predictions_r1.csv"))
    preds = clean_names(preds)
    preds["pred"] = preds["my_pred"]

    merge_cols = ["player_name", "wind_adj1", "dew_adj1", "pred"]
    merge_cols = [c for c in merge_cols if c in preds.columns]
    df = df.merge(preds[merge_cols], on="player_name", how="left")

    # Datetime conversion for spline
    df["r1_teetime"] = pd.to_datetime(df["r1_teetime"], errors="coerce")
    df["teetime_numeric"] = df["r1_teetime"].astype(np.int64)

    return df


def _merge_r2(df):
    """
    R2 merge: load R1 stats for averaging + R2 predictions.
    Source: live_stats_r2.py lines 65-95
    """
    # --- Load R1 model for stat averaging ---
    r1_model = pd.read_csv(_resolve_csv("r1_live_model.csv"))
    r1_model = clean_names(r1_model)
    r1_stats = ["great_shots", "poor_shots", "sg_app", "sg_arg", "sg_ott", "sg_putt"]
    # sg_adj_r1 (R1's ott/putt adjustment) comes along so _totals_r2 can undo it
    r1_keep = ["player_name"] + r1_stats + ["sg_adj_r1"]
    r1_keep = [c for c in r1_keep if c in r1_model.columns]
    r1_renamed = r1_model[r1_keep].copy()
    for c in r1_stats:
        if c in r1_renamed.columns:
            r1_renamed = r1_renamed.rename(columns={c: f"{c}_r1"})
    df = df.merge(r1_renamed, on="player_name", how="left")

    # --- Average R1 + R2 categories ---
    for cat in ["sg_app", "sg_arg", "sg_ott", "sg_putt"]:
        r1_col = f"{cat}_r1"
        if r1_col in df.columns and cat in df.columns:
            df[f"{cat}_avg"] = df[[cat, r1_col]].mean(axis=1)
    if "sg_app_r1" in df.columns and "sg_app" in df.columns:
        df["sg_app_delta"] = df["sg_app"] - df["sg_app_r1"]

    # --- Average great_shots across R1 + R2 ---
    if "great_shots_r1" in df.columns and "great_shots" in df.columns:
        df["great_shots_avg"] = df[["great_shots", "great_shots_r1"]].mean(axis=1)

    # --- Load R2 predictions (created by prior round's weather step) ---
    r2_preds = pd.read_csv(_resolve_csv("model_predictions_r2.csv"))
    r2_preds = clean_names(r2_preds)
    r2_merge = ["player_name", "r2_teetime", "wind_adj2", "dew_adj2", "my_pred2"]
    r2_merge = [c for c in r2_merge if c in r2_preds.columns]
    df = df.merge(r2_preds[r2_merge], on="player_name", how="left")
    df = df.rename(columns={"my_pred2": "updated_pred"})

    # Datetime for spline
    if "r2_teetime" in df.columns:
        df["r2_teetime"] = pd.to_datetime(df["r2_teetime"], errors="coerce")
        df["teetime_numeric"] = df["r2_teetime"].astype(np.int64)
    else:
        print("  [warn] r2_teetime not found in model_predictions_r2.csv -- spline will use index order")

    return df


def _merge_r3r4(df, round_num):
    """
    R3/R4 merge: load prior round model + current round predictions.
    Source: live_stats_r3.py lines 65-85 / live_stats_r4.py equivalent
    """
    prior_round = round_num - 1
    prior_file = f"r{prior_round}_live_model.csv"
    pred_file = _resolve_csv(f"model_predictions_r{round_num}.csv")

    # --- Prior round model (for carried-forward skill + adjustments) ---
    prior = pd.read_csv(_resolve_csv(prior_file))
    prior = clean_names(prior)
    prior_cols = [
        "player_name", f"updated_pred_r{round_num}",
        "sg_app_avg", "sg_ott_avg", "sg_arg_avg", "sg_putt_avg",
        "great_shots_avg",
        "tot_resid_adj", "tot_sg_adj",
    ]
    prior_cols = [c for c in prior_cols if c in prior.columns]
    df = df.merge(prior[prior_cols], on="player_name", how="left")

    # --- great_shots_avg ---
    # Loaded from prior round CSV (computed during R2 as avg of R1+R2).
    # If not present (e.g., first time running with this feature), fall back
    # to current round's great_shots.
    if "great_shots_avg" not in df.columns or df["great_shots_avg"].isna().all():
        if "great_shots" in df.columns:
            df["great_shots_avg"] = df["great_shots"].fillna(0)

    # --- Blend this round's category SG into the running averages ---
    # Source: live_stats_r3.py (0.66/0.34) and live_stats_r4.py (0.75/0.25);
    # this line was lost in the engine consolidation. Without it the sg_*_avg
    # columns stay frozen at the R2-era 0.5*(R1+R2) snapshot and the current
    # round's categories never reach the skill update. Matches the sim
    # cascades (round_sim.py:772-775 / round_cascade.rs use 0.66/0.34 for the
    # R3->R4 update). Where this round's category is missing (ShotLink gaps),
    # keep the prior average instead of letting NaN wipe the adjustment.
    blend_w = 0.66 if round_num == 3 else 0.75
    for sg in ["app", "ott", "arg", "putt"]:
        avg_col, cur_col = f"sg_{sg}_avg", f"sg_{sg}"
        if avg_col not in df.columns or cur_col not in df.columns:
            continue
        prior_avg = df[avg_col]
        cur = df[cur_col]
        blended = prior_avg * blend_w + cur * (1.0 - blend_w)
        df[avg_col] = blended.where(cur.notna() & prior_avg.notna(),
                                    prior_avg.where(cur.isna(), cur))

    # --- Current round predictions (wind/dew/teetime) ---
    if os.path.exists(pred_file):
        cur_preds = pd.read_csv(pred_file)
        cur_preds = clean_names(cur_preds)
        teetime_col = f"r{round_num}_teetime"
        cur_cols = ["player_name", f"wind_adj{round_num}", f"dew_adj{round_num}", teetime_col]
        cur_cols = [c for c in cur_cols if c in cur_preds.columns]
        # df may already carry r{N}_teetime from the live field merge. If the
        # prediction file also supplies it, drop the existing one first so the
        # merge doesn't collide into r{N}_teetime_x / _y — that collision left the
        # bare teetime_col absent, so teetime_numeric was never set and the spline
        # was silently skipped.
        if teetime_col in cur_cols and teetime_col in df.columns:
            df = df.drop(columns=[teetime_col])
        df = df.merge(cur_preds[cur_cols], on="player_name", how="left")
    else:
        print(f"Warning: {pred_file} not found. Weather adjustments will be zero.")

    # Datetime for spline
    teetime_col = f"r{round_num}_teetime"
    if teetime_col in df.columns:
        df[teetime_col] = pd.to_datetime(df[teetime_col], errors="coerce")
        df["teetime_numeric"] = df[teetime_col].astype(np.int64)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Residuals
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(df, round_num):
    """
    Compute sg_total_adj and residuals.
    
    The residual represents how much a player over/under-performed relative
    to our prediction after accounting for wind/dew benefits.
    """
    if round_num == 1:
        return _residuals_r1(df)
    elif round_num == 2:
        return _residuals_r2(df)
    else:
        return _residuals_r3r4(df, round_num)


def _residuals_r1(df):
    """Source: live_stats.py per-course processing block."""
    pred_avg = df["pred"].mean()
    df["sg_total_adj"] = df["sg_total"].fillna(0) + pred_avg

    wind_avg = df["wind_adj1"].mean()
    dew_avg = df["dew_adj1"].mean()
    df["player_wind_benefit"] = df["wind_adj1"] - wind_avg
    df["player_dew_benefit"] = df["dew_adj1"] - dew_avg

    df["residual"] = (
        df["sg_total_adj"] - df["pred"]
        + df["player_wind_benefit"] + df["dew_adj1"]
    )
    df["residual2"] = df["residual"] ** 2
    return df


def _residuals_r2(df):
    """Source: live_stats_r2.py lines 97-115"""
    pred_avg = df["updated_pred"].mean()
    df["sg_total_adj"] = df["sg_total"] + pred_avg
    df = df.dropna(subset=["updated_pred", "sg_total_adj"]).copy()

    wind_avg = df["wind_adj2"].mean() if "wind_adj2" in df.columns else 0
    dew_avg = df["dew_adj2"].mean() if "dew_adj2" in df.columns else 0
    df["player_wind_benefit"] = df.get("wind_adj2", 0) - wind_avg
    df["player_dew_benefit"] = df.get("dew_adj2", 0) - dew_avg

    df["residual"] = (
        df["sg_total_adj"] - df["updated_pred"]
        + df["player_wind_benefit"] + df["player_dew_benefit"]
    )
    # Cap the fix-layer input at +6: beyond the training support the cubic
    # explodes positive (a +9.7 residual otherwise earns ~+0.9 SG), while
    # empirically leaders at resid >= 6 keep mean-reverting ~-0.25 (2026-07-25
    # PGA backtest, n=382). Raw residual is kept for the weather spline/exports.
    df["residual_capped"] = df["residual"].clip(upper=RESID_FIX_CAP)
    df["residual2"] = df["residual_capped"] ** 2
    df["residual3"] = df["residual_capped"] ** 3
    return df


def _residuals_r3r4(df, round_num):
    """Source: live_stats_r3.py / live_stats_r4.py"""
    pred_col = f"updated_pred_r{round_num}"
    wind_col = f"wind_adj{round_num}"
    dew_col = f"dew_adj{round_num}"

    pred_avg = df[pred_col].mean()
    df["sg_total_adj"] = df["sg_total"] + pred_avg

    wind_avg = df[wind_col].mean() if wind_col in df.columns else 0
    dew_avg = df[dew_col].mean() if dew_col in df.columns else 0
    df["player_wind_benefit"] = df.get(wind_col, 0) - wind_avg
    df["player_dew_benefit"] = df.get(dew_col, 0) - dew_avg

    df["residual"] = (
        df["sg_total_adj"] - df[pred_col]
        + df["player_wind_benefit"] + df["player_dew_benefit"]
    )
    df["residual2"] = df["residual"] ** 2
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Weather Spline
# ══════════════════════════════════════════════════════════════════════════════

def fit_weather_spline(df, round_num):
    """
    Fit B-spline to residuals vs tee time to extract weather/wave signal.
    Source: Identical pattern in all 4 live_stats files.
    """
    teetime_col = f"r{round_num}_teetime"

    if "teetime_numeric" not in df.columns or df["teetime_numeric"].isna().all():
        print(f"  No tee time data for spline. Skipping.")
        df["weather_signal"] = 0.0
        df["residual_w_adj"] = df["residual"]
        return df

    valid = df.dropna(subset=["teetime_numeric", "residual"])
    if len(valid) < 10:
        df["weather_signal"] = 0.0
        df["residual_w_adj"] = df["residual"]
        return df

    try:
        X = dmatrix(
            "bs(teetime_numeric, df=4, degree=3, include_intercept=False)",
            {"teetime_numeric": valid["teetime_numeric"]},
            return_type="dataframe",
        )
        model = sm.OLS(valid["residual"].fillna(0), X).fit()
        df.loc[valid.index, "weather_signal"] = model.fittedvalues
        df["weather_signal"] = df["weather_signal"].fillna(0)
    except Exception as e:
        print(f"  Spline failed: {e}")
        df["weather_signal"] = 0.0

    df["residual_w_adj"] = df["residual"] - df["weather_signal"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Apply Coefficient Adjustments
# ══════════════════════════════════════════════════════════════════════════════

def apply_adjustments(df, round_num, has_shotlink=True):
    """
    Apply position/skill-bucket coefficient adjustments.
    
    For R1: Buckets are based on pred value (skill-based).
            OTT/putt adjustments only applied if ShotLink data exists.
    For R2: Buckets are based on leaderboard position.
    For R3/R4: Same as R2 but SG-only (no residual terms).
    """
    buckets = ROUND_BUCKETS[round_num]

    for label, mask_func, coeffs in buckets:
        mask = mask_func(df)
        for key, coeff in coeffs.items():
            col = _map_key_to_column(key, round_num)
            if col is None or col not in df.columns:
                continue

            # R1: skip OTT/putt if no ShotLink
            if round_num == 1 and key in ("ott", "putt") and not has_shotlink:
                continue

            adj_col = f"{key}_adj"
            if adj_col not in df.columns:
                df[adj_col] = 0.0
            df.loc[mask, adj_col] = df.loc[mask, col] * coeff

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Compute Totals & Updated Prediction
# ══════════════════════════════════════════════════════════════════════════════

def compute_totals(df, round_num):
    """Route to round-specific total computation."""
    if round_num == 1:
        return _totals_r1(df)
    elif round_num == 2:
        return _totals_r2(df)
    else:
        return _totals_r3r4(df, round_num)


def _totals_r1(df):
    """
    R1 totals and caps.
    Source: live_stats.py per-course block, post-coefficient section.
    """
    df["residual_adj"] = df.get("residual_adj", 0)
    df["residual2_adj"] = df.get("residual2_adj", 0)
    df["ott_adj"] = df.get("ott_adj", 0)
    df["putt_adj"] = df.get("putt_adj", 0)

    # Residual cap logic (from live_stats.py), plus a -0.5 floor: the concave
    # quadratics run away negative on blow-up rounds (low bucket at resid -10
    # would give -1.5), while empirically R1 resid < -6 mean-reverts -0.507
    # (PGA 2019-2026, n=578). Mirrors the R2 layer's floor.
    df["tot_resid_adj"] = df["residual_adj"] + df["residual2_adj"]
    df["tot_resid_adj"] = np.minimum(
        np.where(
            (df["tot_resid_adj"] > 0.2) & (df["residual"] < 0),
            0.2,
            df["tot_resid_adj"],
        ),
        0.5,
    ).clip(-0.5)

    # Persist R1's category-SG piece so the R2 run can undo it when the
    # avg-based category terms replace it (original live_stats_r2.py read
    # sg_adj_r1 from r1_live_model.csv for exactly this).
    df["sg_adj_r1"] = df["ott_adj"] + df["putt_adj"]

    df["total_adjustment"] = df["ott_adj"] + df["putt_adj"] + df["tot_resid_adj"]
    df["updated_pred"] = df["pred"] + df["total_adjustment"]

    return df


def _totals_r2(df):
    """
    R2 totals.
    Source: live_stats_r2.py lines 155-185
    """
    df["residual_adj"] = df.get("residual_adj", 0)
    df["residual2_adj"] = df.get("residual2_adj", 0)
    df["residual3_adj"] = df.get("residual3_adj", 0)

    # Residual total, clipped at -0.5
    df["tot_resid_adj"] = (
        df["residual_adj"] + df["residual2_adj"] + df["residual3_adj"]
    ).clip(lower=-0.5)

    # SG adjustment total (fillna handles multi-course ShotLink gaps
    # where delta_app may be NaN if a player lacks SG data from both courses)
    sg_cols = ["avg_arg_adj", "avg_putt_adj", "avg_ott_adj", "avg_app_adj", "delta_app_adj"]
    for c in sg_cols:
        df[c] = df.get(c, 0)
    df["tot_sg_adj"] = df[sg_cols].fillna(0).sum(axis=1)

    # Undo R1's OTT/PUTT adjustment: the avg-based category terms above
    # re-price the same signal from two rounds of data, so the R1-only
    # estimate must be replaced, not stacked on top. Original behavior
    # (live_stats_r2.py: `total_adjustment = sum(adj) - sg_adj_r1`) that was
    # lost in the engine consolidation; matches round_sim.py:750 /
    # new_sim.py:831 ("avoid double counting R1 part").
    if "sg_adj_r1" in df.columns:
        df["r1_sg_adj_undo"] = -df["sg_adj_r1"].fillna(0)
    else:
        df["r1_sg_adj_undo"] = 0.0

    # Total adjustment = clipped residual total + SG components + R1 undo.
    # Uses tot_resid_adj (not the raw residual components) so the -0.5 lower
    # clip actually reaches updated_pred_r3 — the raw components previously
    # bypassed it.
    adj_components = [
        "tot_resid_adj",
        "avg_ott_adj", "avg_putt_adj", "avg_app_adj", "avg_arg_adj", "delta_app_adj",
        "r1_sg_adj_undo",
    ]
    adj_components = [c for c in adj_components if c in df.columns]
    df["total_adjustment"] = df[adj_components].sum(axis=1)

    df["Score"] = df["round"] + course_par

    # Updated prediction for next round
    df["updated_pred_r3"] = df["updated_pred"] + df["total_adjustment"]

    if "r3_teetime" in df.columns:
        skill_avg = df.loc[df["r3_teetime"].notna(), "updated_pred_r3"].mean()
    else:
        skill_avg = df["updated_pred_r3"].mean()
    print(f"  Next round skill average: {skill_avg:.4f}" if not pd.isna(skill_avg) else "")

    return df


def _totals_r3r4(df, round_num):
    """
    R3/R4 totals: SG-only adjustments.
    Source: live_stats_r3.py lines 95-115
    
    Key logic: We UNDO the prior round's SG and residual adjustments,
    then apply this round's fresh adjustments.
    
    total_adjustment = fresh_adj - prior_sg - prior_resid
    so that Post = Pre + total_adjustment (consistent with R1/R2).
    """
    # This round's fresh SG adjustments
    adj_cols = ["sg_ott_avg_adj", "sg_putt_avg_adj", "sg_app_avg_adj", "sg_arg_avg_adj",
                "avg_great_shots_adj", "pos_6_10_adj"]
    adj_cols = [c for c in adj_cols if c in df.columns]
    fresh_adj = df[adj_cols].sum(axis=1) if adj_cols else 0

    # Prior round's adjustments to undo
    prior_sg = df.get("tot_sg_adj", pd.Series(0, index=df.index)).fillna(0)
    prior_resid = df.get("tot_resid_adj", pd.Series(0, index=df.index)).fillna(0)

    # Net total adjustment = fresh - prior (so Post = Pre + total_adjustment)
    df["total_adjustment"] = fresh_adj - prior_sg - prior_resid

    pred_col = f"updated_pred_r{round_num}"
    next_pred_col = f"updated_pred_r{round_num + 1}" if round_num < 4 else "updated_pred_final"

    df[next_pred_col] = df[pred_col] + df["total_adjustment"]

    df["Score"] = df["round"] + course_par
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Multi-Course Handling (R1 special logic)
# ══════════════════════════════════════════════════════════════════════════════

def process_multicourse(df, round_num):
    """
    Handle multi-course events by processing each course independently.
    
    Source: live_stats.py per-course loop (lines 125-280)
    
    For each course:
      - Detect ShotLink availability
      - Compute course-specific residuals
      - Fit course-specific weather spline
      - Apply adjustments (with/without ShotLink)
    
    Falls back to single-course processing if course_x is absent or has one value.
    """
    course_col = "course" if "course" in df.columns else "course_x" if "course_x" in df.columns else None

    # Single-course path (most common)
    if course_col is None or df[course_col].nunique() <= 1:
        df = compute_residuals(df, round_num)
        df = fit_weather_spline(df, round_num)
        has_shotlink = _detect_shotlink(df)
        df = apply_adjustments(df, round_num, has_shotlink=has_shotlink)
        df = compute_totals(df, round_num)
        return df

    # Multi-course path
    courses = [c for c in df[course_col].unique() if pd.notna(c)]
    print(f"Multi-course event detected: {courses}")
    frames = []

    for course_id in courses:
        chunk = df[df[course_col] == course_id].copy()
        print(f"  Processing course: {course_id} ({len(chunk)} players)")

        # ShotLink detection: satellite courses may not have granular SG data
        has_shotlink = _detect_shotlink(chunk)
        print(f"    ShotLink available: {has_shotlink}")

        # Satellite course proxy: if no sg_total, compute from raw score
        if not has_shotlink:
            if "score" in chunk.columns and chunk["score"].notna().any():
                avg_score = chunk["score"].mean()
                chunk["sg_total"] = avg_score - chunk["score"]
            chunk["sg_total"] = chunk["sg_total"].fillna(0)

        chunk = compute_residuals(chunk, round_num)
        chunk = fit_weather_spline(chunk, round_num)
        chunk = apply_adjustments(chunk, round_num, has_shotlink=has_shotlink)
        chunk = compute_totals(chunk, round_num)
        frames.append(chunk)

    return pd.concat(frames, ignore_index=True)


def _detect_shotlink(df):
    """Check if ShotLink data exists (sg_ott has non-zero values)."""
    if "sg_ott" not in df.columns:
        return False
    return df["sg_ott"].abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: R1 Leaderboard Gravity
# ══════════════════════════════════════════════════════════════════════════════

def apply_leaderboard_gravity(df):
    """
    R1-only: Small negative bumps for top-5 players with low predictions.
    Source: live_stats.py lines 285-288
    
    Only applies if updated_pred < 0.5 (strong players).
    Rationale: Top of leaderboard after R1 tends to regress slightly.
    """
    gravity = {1: -0.07, 2: -0.03, 3: -0.02, 4: -0.01, 5: -0.01}
    mask = df["updated_pred"] < 0.5
    df.loc[mask, "updated_pred"] += df.loc[mask, "position"].map(gravity).fillna(0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Prediction Creation (Weather Update)
# ══════════════════════════════════════════════════════════════════════════════

def create_next_round_predictions(round_num):
    """
    Create model_predictions_r{N+1}.csv from the live model output.
    
    This absorbs the prediction-creation logic from rd_1_sd_multicourse_sim.py.
    
    Steps:
      1. Load updated skill from r{N}_live_model.csv
      2. Load tee times for round N+1 from field updates API
      3. Compute wind/dew adjustments using forecast arrays
      4. Compute expected scores = skill + avg_weather - player_weather
      5. Save model_predictions_r{N+1}.csv
    
    Args:
        round_num: The round that just completed (1-4). Predictions are for N+1.
    """
    if round_num >= 4:
        print("R4 complete — no next round predictions needed.")
        return

    next_round = round_num + 1
    print(f"\n{'='*60}")
    print(f"  Creating predictions for Round {next_round}")
    print(f"{'='*60}")

    # --- Load skill from live model ---
    live_model = pd.read_csv(_resolve_csv(f"r{round_num}_live_model.csv"))
    live_model = clean_names(live_model)

    # Determine the skill column name
    if round_num == 1:
        skill_col = "updated_pred"
    else:
        skill_col = f"updated_pred_r{next_round}"

    if skill_col not in live_model.columns:
        raise ValueError(f"Column '{skill_col}' not found in r{round_num}_live_model.csv")

    # Build predictions DataFrame — skill + std_dev only.
    # Course assignment comes from the field updates API (next round's course),
    # NOT from the live model (which has the current round's course).
    keep_cols = ["player_name", skill_col]
    if "std_dev" in live_model.columns:
        keep_cols.append("std_dev")

    preds = live_model[[c for c in keep_cols if c in live_model.columns]].copy()

    # Standardize skill column name: my_pred for R1, my_pred{N} for R2+
    pred_name = "my_pred" if next_round == 1 else f"my_pred{next_round}"
    preds = preds.rename(columns={skill_col: pred_name})

    # --- Load tee times ---
    teetime_col = f"r{next_round}_teetime"
    field = fetch_field_updates(API_KEY, teetime_col=teetime_col, include_course=(next_round <= 3))
    if field is not None and teetime_col in field.columns:
        # Merge tee time + course for next round from field updates API
        merge_cols = ["player_name", teetime_col]
        if "course" in field.columns:
            merge_cols.append("course")
        preds = preds.merge(field[[c for c in merge_cols if c in field.columns]], on="player_name", how="left")

        # Drop players without tee times (cut / withdrawn after this round)
        before = len(preds)
        preds = preds.dropna(subset=[teetime_col])
        cut = before - len(preds)
        if cut > 0:
            print(f"  Dropped {cut} players without R{next_round} tee times (cut/inactive)")
    else:
        print(f"Warning: Tee times for R{next_round} not yet available.")
        print("Saving skill-only predictions. Run again once tee times are available.")
        # No tee times to filter on, so drop cut/WD players by live position
        # instead (position parses to 999 for "CUT"/missing). Otherwise this
        # fallback file re-admits the whole pre-cut field to weekend sims.
        if next_round >= 3 and "position" in live_model.columns:
            pos = pd.to_numeric(live_model["position"], errors="coerce").fillna(999)
            out_players = set(live_model.loc[pos >= 900, "player_name"])
            if out_players:
                before = len(preds)
                preds = preds[~preds["player_name"].isin(out_players)].copy()
                print(f"  Dropped {before - len(preds)} cut/WD players (no live position)")
        preds[f"scores_r{next_round}"] = preds[pred_name]  # Skill only, no weather
        _save_predictions(preds, next_round)
        return

    # --- Compute wind/dew adjustments ---
    wind_factor = compute_wind_factor(event_ids, wind_override, baseline_wind)
    wind_array = WIND_ARRAYS[next_round]
    dew_array = DEW_ARRAYS[next_round]

    wind_vals, dew_vals = [], []
    for _, row in preds.iterrows():
        tt = row.get(teetime_col)
        wind_vals.append(calculate_average_wind(tt, wind_array))
        dew_vals.append(calculate_average_wind(tt, dew_array))

    wind_col = f"wind_adj{next_round}"
    dew_col = f"dew_adj{next_round}"

    preds[f"wind_r{next_round}"] = wind_vals
    preds[f"dew_r{next_round}"] = dew_vals
    preds[wind_col] = preds[f"wind_r{next_round}"] * wind_factor
    preds[dew_col] = preds[f"dew_r{next_round}"] * dew_calculation

    # Center dew adjustment
    avg_dew = preds[dew_col].mean()
    preds[dew_col] = preds[dew_col] - avg_dew

    avg_wind = preds[wind_col].mean()

    # Expected SG: skill + field wind level - player wind cost - player dew cost.
    # Both wind_adj and dew_adj are SCORE-space (positive = costs strokes;
    # dew_calculation is negative so humid windows are a benefit), so both
    # subtract. round_sim's wx_lookup mirrors this weather term — lockstep.
    preds[f"scores_r{next_round}"] = (
        preds[pred_name] + avg_wind - preds[wind_col] - preds[dew_col]
    )

    # --- Diagnostics ---
    avg_wind_speed = preds[f"wind_r{next_round}"].mean()
    print(f"  Avg wind speed R{next_round}: {avg_wind_speed:.2f} mph")
    print(f"  Avg wind impact in SG: {avg_wind:.4f}")
    print(f"  Avg skill R{next_round}: {preds[pred_name].mean():.4f}")

    # Per-course expected scoring (multi-course aware)
    course_col = "course" if "course" in preds.columns else None

    # Primary path: API gave us next-round course directly + we have COURSE_SCORE_MAP
    if course_col and preds[course_col].nunique() > 1 and COURSE_SCORE_MAP:
        preds["course_score_adj"] = np.nan
        print(f"  Multi-course mapping:")
        for code, adj in COURSE_SCORE_MAP.items():
            mask = preds[course_col] == code
            n = mask.sum()
            if n > 0:
                course_skill = preds.loc[mask, pred_name].mean()
                course_wind = preds.loc[mask, wind_col].mean()
                exp_score = round(adj - course_skill + course_wind, 2)
                print(f"    {code} -> adj={adj}, players={n}, avg_skill={course_skill:.3f}, expected scoring={exp_score}")
                preds.loc[mask, "course_score_adj"] = adj
        # Warn about unmapped players
        unmapped = preds[course_col].notna() & preds["course_score_adj"].isna()
        if unmapped.any():
            unknown = preds.loc[unmapped, course_col].unique()
            print(f"  Warning: Unmapped course codes: {list(unknown)} — using fallback")
            preds.loc[unmapped, "course_score_adj"] = SCORE_ADJS.get(next_round, 0)

    # Fallback: no course from API, but live model had one + exactly 2 courses → flip
    elif len(COURSE_SCORE_MAP) == 2:
        live_course_col = "course" if "course" in live_model.columns else "course_x" if "course_x" in live_model.columns else None
        if live_course_col and live_course_col in live_model.columns:
            codes = list(COURSE_SCORE_MAP.keys())
            flip = {codes[0]: codes[1], codes[1]: codes[0]}
            # Map live model's current-round course to preds, then flip for next round
            course_lookup = live_model.set_index("player_name")[live_course_col]
            preds["course"] = preds["player_name"].map(course_lookup).map(flip)
            print(f"  Multi-course mapping (flipped from R{round_num}):")
            for code, adj in COURSE_SCORE_MAP.items():
                mask = preds["course"] == code
                n = mask.sum()
                if n > 0:
                    course_skill = preds.loc[mask, pred_name].mean()
                    print(f"    {code} -> adj={adj}, players={n}, avg_skill={course_skill:.3f}")
                    preds.loc[mask, "course_score_adj"] = adj
            print(f"  Warning: No next-round course from API — flipped R{round_num} assignments for R{next_round}")
        else:
            score_adj = SCORE_ADJS.get(next_round, 0)
            expected_scoring = round(score_adj - preds[pred_name].mean() + avg_wind, 2)
            print(f"  Expected scoring avg R{next_round}: {expected_scoring}")

    # Legacy fallback: positional COURSE_SCORE_ADJS (no COURSE_SCORE_MAP)
    elif (course_col and preds[course_col].nunique() > 1
          and COURSE_SCORE_ADJS and len(COURSE_SCORE_ADJS) > 1):
        courses_ordered = [c for c in preds[course_col].unique() if pd.notna(c)]
        print(f"  Multi-course mapping (order of appearance, legacy):")
        for i, cid in enumerate(courses_ordered):
            adj = COURSE_SCORE_ADJS[i] if i < len(COURSE_SCORE_ADJS) else COURSE_SCORE_ADJS[0]
            course_players = preds[preds[course_col] == cid]
            course_skill = course_players[pred_name].mean()
            course_wind = course_players[wind_col].mean()
            exp_score = round(adj - course_skill + course_wind, 2)
            print(f"    expected_score_{i+1} -> {cid}: adj={adj}, expected scoring={exp_score}")
            preds.loc[preds[course_col] == cid, "course_score_adj"] = adj

    else:
        score_adj = SCORE_ADJS.get(next_round, 0)
        expected_scoring = round(score_adj - preds[pred_name].mean() + avg_wind, 2)
        print(f"  Expected scoring avg R{next_round}: {expected_scoring}")

    _save_predictions(preds, next_round)


def _save_predictions(preds, next_round):
    """Save prediction file with standard naming."""
    filename = f"model_predictions_r{next_round}.csv"
    preds.to_csv(filename, index=False)
    print(f"  [ok] Saved {filename} ({len(preds)} players)")


def create_pre_event_predictions():
    """
    Pre-event: Create model_predictions_r1.csv from final predictions + R1 weather.
    
    Source: rd_1_sd_multicourse_sim.py lines 280-390
    
    This is the initial prediction file before any rounds have been played.
    """
    print(f"\n{'='*60}")
    print(f"  PRE-EVENT: Creating R1 predictions")
    print(f"{'='*60}")

    # Load base predictions
    preds = pd.read_csv(f"final_predictions_{tourney}.csv")
    preds = clean_names(preds)

    # Optionally merge pre_sim_summary pred (overrides my_pred)
    summary_file = f"pre_sim_summary_{tourney}.csv"
    if os.path.exists(summary_file):
        summary = pd.read_csv(summary_file, usecols=["player_name", "pred"])
        summary = clean_names(summary)
        preds = preds.drop(columns=["my_pred"], errors="ignore")
        preds = preds.merge(summary, on="player_name", how="left")
        preds = preds.rename(columns={"pred": "my_pred"})

    # Replace low-confidence predictions with DG decomposition
    dg_decomp = fetch_player_decompositions(API_KEY)
    if not dg_decomp.empty and 'dg_final_pred' in dg_decomp.columns:
        preds = preds.merge(dg_decomp[['player_name', 'dg_final_pred']], on='player_name', how='left')
        mask = pd.Series(False, index=preds.index)  # manual list only
        n_replaced = mask.sum()
        if n_replaced > 0:
            replaced = preds.loc[mask, ['player_name', 'my_pred', 'dg_final_pred']].copy()
            preds.loc[mask, 'my_pred'] = preds.loc[mask, 'dg_final_pred']
            print(f"  [DG decomp] Replaced {n_replaced} predictions (manual list) with DG decomposition:")
            for _, r in replaced.iterrows():
                print(f"    {r['player_name']}: {r['my_pred']:.3f} -> {r['dg_final_pred']:.3f}")
        else:
            print("  [DG decomp] No predictions below threshold needed replacement")
        preds = preds.drop(columns=['dg_final_pred'])

    # Load tee times
    teetime_col = "r1_teetime"
    if teetime_col not in preds.columns:
        field = fetch_field_updates(API_KEY, teetime_col=teetime_col, include_course=True)
        if field is not None:
            preds = preds.merge(field, on="player_name", how="left")

    # Compute wind factor
    wind_factor = compute_wind_factor(event_ids, wind_override, baseline_wind)

    # Compute wind/dew
    wind_vals, dew_vals = [], []
    for _, row in preds.iterrows():
        tt = row.get(teetime_col)
        wind_vals.append(calculate_average_wind(tt, WIND_ARRAYS[1]))
        dew_vals.append(calculate_average_wind(tt, DEW_ARRAYS[1]))

    preds["wind_r1"] = wind_vals
    preds["dew_r1"] = dew_vals
    preds["wind_adj1"] = preds["wind_r1"] * wind_factor
    preds["dew_adj1"] = preds["dew_r1"] * dew_calculation

    # Center dew
    avg_dew = preds["dew_adj1"].mean()
    preds["dew_adj1"] = preds["dew_adj1"] - avg_dew

    avg_wind = preds["wind_adj1"].mean()
    avg_skill = preds["my_pred"].mean()

    # Expected scoring
    expected_scoring = round(SCORE_ADJS[1] - avg_skill + avg_wind, 2)
    # dew_adj is SCORE-space like wind_adj (dew_calculation comes from
    # humidity.py's score regression: negative coef -> humid = lower scores
    # = easier), so it must be SUBTRACTED like wind: a player in the more
    # humid window gains expected SG. round_sim's wx_lookup mirrors this
    # exact weather term — keep them in lockstep.
    preds["scores_r1"] = preds["my_pred"] + avg_wind - preds["wind_adj1"] - preds["dew_adj1"]

    # --- Diagnostics ---
    print(f"  Players: {len(preds)}")
    print(f"  Avg wind speed: {preds['wind_r1'].mean():.2f} mph")
    print(f"  Avg wind impact (SG): {avg_wind:.4f}")
    print(f"  Avg skill: {avg_skill:.4f}")
    print(f"  Expected R1 scoring avg: {expected_scoring}")

    hi = preds.loc[preds["wind_adj1"].idxmax()]
    lo = preds.loc[preds["wind_adj1"].idxmin()]
    print(f"  Highest wind adj: {hi['player_name']} ({hi['wind_adj1']:.3f})")
    print(f"  Lowest wind adj:  {lo['player_name']} ({lo['wind_adj1']:.3f})")

    preds.to_csv("model_predictions_r1.csv", index=False)
    print(f"  [ok] Saved model_predictions_r1.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(df, round_num):
    """Export live model CSV, summary CSV, residual summary, and plot."""

    # --- Full model ---
    model_file = f"r{round_num}_live_model.csv"
    df.to_csv(model_file, index=False)
    df.to_csv(f"r{round_num}_live_model_{tourney}.csv", index=False)
    print(f"  [ok] Saved {model_file}")

    # --- Summary (round-specific columns) ---
    if round_num == 1:
        summary_cols = [
            "player_name", "residual", "weather_signal", "residual_w_adj",
            "tot_resid_adj", "total_adjustment", "updated_pred",
        ]
        # Include course_x if multi-course
        course_col = "course" if "course" in df.columns else "course_x" if "course_x" in df.columns else None
        if course_col:
            summary_cols.append(course_col)
    elif round_num == 2:
        summary_cols = [
            "player_name", "tot_resid_adj", "total_adjustment",
            "avg_ott_adj", "avg_putt_adj", "avg_app_adj", "avg_arg_adj",
            "delta_app_adj", "r1_sg_adj_undo", "updated_pred_r3",
        ]
    else:
        adj_cols = [f"{c}_adj_r{round_num}" for c in
                    ["sg_ott_avg", "sg_putt_avg", "sg_app_avg", "sg_arg_avg",
                     "avg_great_shots", "pos_6_10"]]
        next_pred = f"updated_pred_r{round_num + 1}" if round_num < 4 else "updated_pred_final"
        summary_cols = ["player_name"] + adj_cols + [f"updated_pred_r{round_num}", next_pred]

    existing = [c for c in summary_cols if c in df.columns]
    summary_file = f"r{round_num}_live_summary.csv"
    df[existing].to_csv(summary_file, index=False)
    print(f"  [ok] Saved {summary_file}")

    # --- Residual summary (appended across rounds) ---
    valid = df.dropna(subset=["sg_total_adj", "residual"])
    if not valid.empty:
        # Determine which prediction column to use for R²
        if round_num == 1:
            r2_pred_col = "pred"
        elif round_num == 2:
            r2_pred_col = "updated_pred"
        else:
            r2_pred_col = f"updated_pred_r{round_num}"

        if r2_pred_col in valid.columns:
            rmse = np.sqrt((valid["residual"] ** 2).mean())
            r2 = r2_score(valid["sg_total_adj"], valid[r2_pred_col])
            avg_res = valid["residual"].abs().mean()

            print(f"  RMSE: {rmse:.4f} | R²: {r2:.4f} | Avg |residual|: {avg_res:.4f}")

            row = pd.DataFrame([{
                "event_name": df["event_name"].iloc[0] if "event_name" in df.columns else "Unknown",
                "round_num": round_num,
                "average_residual": avg_res,
                "rmse": rmse,
                "r_squared": r2,
                "year": datetime.now().year,
            }])
            path = "residual_summary.csv"
            row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    # --- Spline Plot (saved as PDF for email attachment) ---
    teetime_col = f"r{round_num}_teetime"
    spline_pdf_path = None
    if teetime_col in df.columns and "weather_signal" in df.columns:
        try:
            plot_df = df.dropna(subset=[teetime_col, "weather_signal"]).sort_values(teetime_col)
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))

                course_col = "course" if "course" in df.columns and df["course"].nunique() > 1 else None
                if course_col:
                    for cid, group in plot_df.groupby(course_col):
                        ax.plot(group[teetime_col], group["weather_signal"], label=cid, linewidth=2)
                    ax.legend(title="Course")
                else:
                    ax.plot(plot_df[teetime_col], plot_df["weather_signal"], color="#1f77b4", linewidth=2)

                # Scatter individual residuals behind the spline
                ax.scatter(plot_df[teetime_col], plot_df["residual"], alpha=0.25, s=15, color="gray", zorder=1)

                ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
                ax.set_title(f"R{round_num} Spline-Smoothed Residual by Tee Time", fontsize=14, fontweight="bold")
                ax.set_xlabel("Tee Time")
                ax.set_ylabel("Smoothed Residual (SG)")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
                fig.autofmt_xdate(rotation=30)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                spline_pdf_path = f"r{round_num}_weather_spline.pdf"
                fig.savefig(spline_pdf_path, format="pdf", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  [ok] Saved {spline_pdf_path}")
        except Exception as e:
            print(f"  Plot failed: {e}")

    df["Score"] = df["round"] + course_par
    return df, spline_pdf_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Email Summary
# ══════════════════════════════════════════════════════════════════════════════

EMAIL_FROM = "mckinleyslade@gmail.com"
EMAIL_TO = ["mckinleyslade@gmail.com", "mckinley.slade@gmail.com"]


def _get_component_columns(df, round_num):
    """
    Return the list of adjustment component columns that exist for this round,
    in a display-friendly order with clean labels.
    """
    # Raw SG stats (always shown, same for all rounds)
    raw_sg = [
        ("sg_total", "SG Total"),
        ("sg_ott", "SG OTT"),
        ("sg_app", "SG APP"),
        ("sg_arg", "SG ARG"),
        ("sg_putt", "SG Putt"),
    ]

    if round_num == 1:
        adj_cols = [
            ("tot_resid_adj", "Residual"),
            ("ott_adj", "OTT Adj"),
            ("putt_adj", "Putt Adj"),
        ]
    elif round_num == 2:
        adj_cols = [
            ("tot_resid_adj", "Residual"),
            ("avg_ott_adj", "Avg OTT Adj"),
            ("avg_putt_adj", "Avg Putt Adj"),
            ("avg_app_adj", "Avg APP Adj"),
            ("avg_arg_adj", "Avg ARG Adj"),
            ("delta_app_adj", "Δ APP Adj"),
        ]
    else:  # R3/R4
        adj_cols = [
            ("sg_ott_avg_adj", "Avg OTT Adj"),
            ("sg_putt_avg_adj", "Avg Putt Adj"),
            ("sg_app_avg_adj", "Avg APP Adj"),
            ("sg_arg_avg_adj", "Avg ARG Adj"),
            ("avg_great_shots_adj", "Avg Great Shots Adj"),
            ("pos_6_10_adj", "Pos 6-10 Level Adj"),
        ]

    # Only return columns that actually exist in the DataFrame
    return (
        [(col, label) for col, label in raw_sg if col in df.columns],
        [(col, label) for col, label in adj_cols if col in df.columns],
    )


def _get_pred_columns(df, round_num):
    """Return (pre_adj_pred_col, post_adj_pred_col) for this round."""
    if round_num == 1:
        return "pred", "updated_pred"
    elif round_num == 2:
        return "updated_pred", "updated_pred_r3"
    elif round_num == 3:
        return "updated_pred_r3", "updated_pred_r4"
    else:
        return "updated_pred_r4", "updated_pred_final"


def build_email_html(df, round_num):
    """
    Build an HTML email showing the 5 largest positive and 5 largest negative
    skill adjustments with their component breakdown.
    """
    event_name = df["event_name"].iloc[0] if "event_name" in df.columns else tourney
    raw_sg_cols, adj_cols = _get_component_columns(df, round_num)
    pre_col, post_col = _get_pred_columns(df, round_num)

    # Ensure total_adjustment exists
    if "total_adjustment" not in df.columns:
        return "<p>No adjustment data available.</p>"

    valid = df.dropna(subset=["total_adjustment"]).copy()
    valid = valid.sort_values("total_adjustment", ascending=False)

    top5 = valid.head(5)
    bot5 = valid.tail(5)

    def _build_table(subset, title, color):
        """Build an HTML table for a subset of players."""
        rows = ""
        for _, row in subset.iterrows():
            name = row.get("player_name", "?").title()
            pre = row.get(pre_col, 0)
            post = row.get(post_col, 0)
            total = row.get("total_adjustment", 0)

            # Raw SG cells (neutral background)
            sg_cells = ""
            for col, _ in raw_sg_cols:
                val = row.get(col, 0)
                if pd.isna(val):
                    val = 0
                cell_color = "#d4edda" if val > 0.01 else "#f8d7da" if val < -0.01 else "#ffffff"
                sg_cells += f'<td style="padding:6px 10px; text-align:center; background:{cell_color};">{val:+.2f}</td>'

            # Adjustment component cells
            comp_cells = ""
            for col, _ in adj_cols:
                val = row.get(col, 0)
                if pd.isna(val):
                    val = 0
                cell_color = "#d4edda" if val > 0.01 else "#f8d7da" if val < -0.01 else "#ffffff"
                comp_cells += f'<td style="padding:6px 10px; text-align:center; background:{cell_color};">{val:+.3f}</td>'

            total_color = "#d4edda" if total > 0 else "#f8d7da"
            rows += f"""
            <tr>
                <td style="padding:6px 10px; font-weight:500;">{name}</td>
                <td style="padding:6px 10px; text-align:center;">{pre:+.3f}</td>
                {sg_cells}
                {comp_cells}
                <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{total_color};">{total:+.3f}</td>
                <td style="padding:6px 10px; text-align:center; font-weight:bold;">{post:+.3f}</td>
            </tr>"""

        sg_headers = "".join(
            f'<th style="padding:6px 10px; text-align:center; background:#495057; color:white;">{label}</th>'
            for _, label in raw_sg_cols
        )
        comp_headers = "".join(
            f'<th style="padding:6px 10px; text-align:center; background:#e9ecef;">{label}</th>'
            for _, label in adj_cols
        )

        return f"""
        <h3 style="color:{color}; margin:20px 0 8px 0;">{title}</h3>
        <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
            <tr style="background:#343a40; color:white;">
                <th style="padding:6px 10px; text-align:left;">Player</th>
                <th style="padding:6px 10px; text-align:center;">Pre-Adj</th>
                {sg_headers}
                {comp_headers}
                <th style="padding:6px 10px; text-align:center;">Total Adj</th>
                <th style="padding:6px 10px; text-align:center;">Post-Adj</th>
            </tr>
            {rows}
        </table>"""

    # Model fit stats
    rmse_str = ""
    if "residual" in valid.columns:
        rmse = np.sqrt((valid["residual"] ** 2).mean())
        avg_res = valid["residual"].abs().mean()
        rmse_str = f"<p style='color:#666; font-size:12px;'>RMSE: {rmse:.3f} | Avg |Residual|: {avg_res:.3f} | Players: {len(valid)}</p>"

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:900px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">R{round_num} Skill Update — {event_name.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{datetime.now().strftime('%B %d, %Y %I:%M %p')}</p>
        {rmse_str}
        {_build_table(top5, "⬆ Largest Positive Adjustments", "#28a745")}
        {_build_table(bot5, "⬇ Largest Negative Adjustments", "#dc3545")}
        <p style="color:#999; font-size:11px; margin-top:30px;">
            Pre-Adj = prediction entering this round | Post-Adj = updated prediction after R{round_num} data |
            Components vary by round
        </p>
    </body>
    </html>"""

    return html


def send_summary_email(df, round_num, spline_pdf_path=None):
    """
    Send skill update summary email via Gmail SMTP.
    Reads app password from GMAIL_APP_PASSWORD environment variable.
    Attaches weather spline PDF if available.
    """
    password = os.environ.get("EMAIL_PASSWORD")
    if not password:
        print("  [warn] GMAIL_APP_PASSWORD not set. Skipping email.")
        return

    try:
        from email.mime.application import MIMEApplication

        html = build_email_html(df, round_num)
        event_name = df["event_name"].iloc[0] if "event_name" in df.columns else tourney

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"R{round_num} Skill Update — {event_name.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        # HTML body
        msg.attach(MIMEText(html, "html"))

        # Attach spline PDF if it exists
        if spline_pdf_path and os.path.exists(spline_pdf_path):
            with open(spline_pdf_path, "rb") as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
                pdf_attachment.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(spline_pdf_path),
                )
                msg.attach(pdf_attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  [ok] Summary email sent")

    except Exception as e:
        print(f"  [warn] Email failed: {e}")
        print("    (Skill update still saved — email is non-blocking)")


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Points
# ══════════════════════════════════════════════════════════════════════════════

def run_skill_update(round_num, dry_run=False):
    """
    Process live stats for round N and update player skill.
    Outputs: r{N}_live_model.csv
    """
    print(f"\n{'='*60}")
    print(f"  SKILL UPDATE — Round {round_num} ({tourney})")
    print(f"{'='*60}")

    # Load and merge all data sources
    df = load_and_merge(round_num)
    print(f"  Players loaded: {len(df)}")

    # R1 uses multi-course processing; R2+ uses single path
    # (but both check for multi-course when course_x exists)
    if round_num == 1:
        df = process_multicourse(df, round_num)
        df = apply_leaderboard_gravity(df)
    else:
        # R2+ can also have multi-course (R2/R3), but the coefficient
        # application doesn't branch by ShotLink — just by position.
        # We still want course-specific residuals/splines though.
        course_col = "course" if "course" in df.columns else "course_x" if "course_x" in df.columns else None
        if course_col and df[course_col].nunique() > 1:
            frames = []
            for cid in df[course_col].unique():
                if pd.isna(cid):
                    continue
                chunk = df[df[course_col] == cid].copy()
                print(f"  Course: {cid} ({len(chunk)} players)")
                chunk = compute_residuals(chunk, round_num)
                chunk = fit_weather_spline(chunk, round_num)
                frames.append(chunk)
            df = pd.concat(frames, ignore_index=True)
            df = apply_adjustments(df, round_num)
            df = compute_totals(df, round_num)
        else:
            df = compute_residuals(df, round_num)
            df = fit_weather_spline(df, round_num)
            df = apply_adjustments(df, round_num)
            df = compute_totals(df, round_num)

    # Export
    df, spline_pdf_path = export_results(df, round_num)

    # Email summary
    if not dry_run:
        send_summary_email(df, round_num, spline_pdf_path=spline_pdf_path)
    else:
        print("  [dry-run] Skipping email")

    return df


def run_weather_update(round_num):
    """
    Add weather forecasts to create model_predictions_r{N+1}.csv.
    Called automatically after skill update. If tee times aren't available,
    saves skill-only predictions and prints a message to run again later.
    """
    create_next_round_predictions(round_num)



# ══════════════════════════════════════════════════════════════════════════════
# Config Loading (Google Sheet or CLI fallback)
# ══════════════════════════════════════════════════════════════════════════════

def refresh_dew_forecasts():
    """
    Fetch fresh dewpoint forecasts from Open-Meteo for all 4 round dates
    and write updated dew arrays to the sheet's dew grid columns.

    Uses same lat/lon + date resolution as humidity.py.
    """
    import requests
    from sheets_storage import get_spreadsheet

    try:
        from sim_inputs import course_id as _cid
    except ImportError:
        print("  [dew_refresh] Cannot import course_id from sim_inputs")
        return

    # Get coordinates
    coords_csv = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")
    if not os.path.exists(coords_csv):
        print("  [dew_refresh] course_coordinates.csv not found")
        return

    coords_df = pd.read_csv(coords_csv)
    coords_row = coords_df[coords_df["course_id"] == _cid]
    if coords_row.empty:
        print(f"  [dew_refresh] course_id {_cid} not found in coordinates")
        return

    lat = float(coords_row["lat"].iloc[0])
    lon = float(coords_row["lon"].iloc[0])

    # Get round dates from DataGolf
    round_dates = get_round_dates()
    if not round_dates or len(round_dates) < 4:
        print("  [dew_refresh] Could not resolve round dates")
        return

    start_date = round_dates[0].strftime("%Y-%m-%d")
    end_date = round_dates[3].strftime("%Y-%m-%d")

    # Fetch forecast
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "dewpoint_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        resp = requests.get(api_url, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        print(f"  [dew_refresh] Forecast fetch failed: {e}")
        return

    if "hourly" not in data or "dewpoint_2m" not in data["hourly"]:
        print("  [dew_refresh] No dewpoint data in response")
        return

    timestamps = data["hourly"]["time"]
    dewpoint_values = data["hourly"]["dewpoint_2m"]

    # Build per-round arrays (6 AM - 8 PM = 15 values)
    dew_by_round = {i: [] for i in range(1, 5)}
    date_strs = [d.strftime("%Y-%m-%d") for d in round_dates]

    import datetime as _dt_mod
    for time_str, dewpoint in zip(timestamps, dewpoint_values):
        dt_obj = _dt_mod.datetime.fromisoformat(time_str)
        if 6 <= dt_obj.hour <= 20:
            date_str = dt_obj.date().strftime("%Y-%m-%d")
            if date_str in date_strs:
                rd = date_strs.index(date_str) + 1
                dew_by_round[rd].append(round(dewpoint))

    # Write to sheet
    import gspread
    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet("round_config")

    DEW_COLS = {1: 7, 2: 11, 3: 15, 4: 19}
    DATA_START_ROW = 3

    cells_to_update = []
    for rd in range(1, 5):
        col = DEW_COLS[rd]
        values = dew_by_round.get(rd, [])
        if not values:
            continue
        values = values[:15]
        while len(values) < 15:
            values.append("")
        for i, val in enumerate(values):
            row = DATA_START_ROW + i
            cells_to_update.append(gspread.Cell(row=row, col=col, value=val))
        print(f"  [dew_refresh] R{rd}: {len([v for v in values if v != ''])} values -> col {col}")

    if cells_to_update:
        ws.update_cells(cells_to_update, value_input_option="USER_ENTERED")
        print(f"  [dew_refresh] Updated {len(cells_to_update)} dew cells on sheet")


def _read_dew_array_from_sheet(ws, round_num):
    """
    Read the dew array for a given round from the sheet grid.
    Returns list of floats (6AM-8PM values from rows 3-17).
    """
    DEW_COLS = {1: 7, 2: 11, 3: 15, 4: 19}
    col = DEW_COLS.get(round_num)
    if col is None:
        return []

    all_data = ws.get_all_values()
    values = []
    for row_i in range(2, min(17, len(all_data))):  # rows 3-17 (0-indexed)
        try:
            val = float(all_data[row_i][col - 1])  # 0-indexed columns
            values.append(val)
        except (ValueError, IndexError, TypeError):
            pass
    return values


def _write_param_to_sheet(ws, all_data, param_name, value):
    """
    Write a single value to the round_config A:B parameter area.
    Finds the param row by name; creates it at the bottom if missing.
    """
    # Find existing row (1-indexed for gspread)
    for i, row in enumerate(all_data):
        if row and row[0].strip().lower() == param_name.lower():
            ws.update_cell(i + 1, 2, str(value))
            print(f"  [actuals] Wrote {param_name}={value} to row {i + 1}")
            return
    # Not found — append at bottom
    next_row = len(all_data) + 1
    ws.update_cell(next_row, 1, param_name)
    ws.update_cell(next_row, 2, str(value))
    print(f"  [actuals] Created {param_name}={value} at row {next_row}")


def write_actuals_to_sheet(round_num):
    """
    Write realized weather actuals for a completed round to the config tab.

    Steps:
      1. Read current dew array for this round (= forecast dew before refresh)
      2. Refresh all dew forecasts (overwrites sheet)
      3. Read refreshed dew array (= realized dew)
      4. Read realized_wind, dewpoint_base, wind_override, dew_calculation from config
      5. Read base_score and field_adj from pre-tourney breakdown (cols V-W)
      6. Compute expected score; fetch actual scoring avg; compute delta
      7. Write row to sheet
    """
    from sheets_storage import get_spreadsheet
    from sheet_config import load_config

    try:
        spreadsheet = get_spreadsheet()
        ws = spreadsheet.worksheet("round_config")
    except Exception as e:
        print(f"  [actuals] Could not connect to sheet: {e}")
        return

    # 1. Read forecast dew (before refresh)
    forecast_dew_arr = _read_dew_array_from_sheet(ws, round_num)
    forecast_dew = sum(forecast_dew_arr) / len(forecast_dew_arr) if forecast_dew_arr else 0.0
    print(f"  [actuals] R{round_num} forecast dew (pre-refresh): {forecast_dew:.1f}F")

    # 2. Refresh dew forecasts
    try:
        refresh_dew_forecasts()
    except Exception as e:
        print(f"  [actuals] Dew refresh failed: {e}")

    # 3. Re-read sheet for realized dew (post-refresh)
    # Need to re-fetch worksheet data after the refresh wrote to it
    try:
        ws = spreadsheet.worksheet("round_config")
    except Exception:
        pass
    realized_dew_arr = _read_dew_array_from_sheet(ws, round_num)
    realized_dew = sum(realized_dew_arr) / len(realized_dew_arr) if realized_dew_arr else 0.0
    print(f"  [actuals] R{round_num} realized dew (post-refresh): {realized_dew:.1f}F")

    # 4. Read config values and sheet grid (need grid for both param write-back and base_score read)
    config = load_config()
    realized_wind = config.get(f"realized_wind_r{round_num}")
    dewpoint_base = config.get("dewpoint_base", 0.0) or 0.0
    wind_factor = config.get("wind_override", 0.0) or 0.0
    dew_calc = config.get("dew_calculation", 0.0) or 0.0

    all_data = ws.get_all_values()

    # 4b. Auto-fetch realized wind from Open-Meteo archive if not in sheet
    if realized_wind is None:
        print(f"  [actuals] No realized_wind_r{round_num} in sheet — auto-fetching from Open-Meteo")
        try:
            from sim_inputs import course_id as _cid
            coords_csv = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")
            coords_df = pd.read_csv(coords_csv)
            coords_row = coords_df[coords_df["course_id"] == _cid]
            if coords_row.empty:
                print(f"  [actuals] Course {_cid} not in course_coordinates.csv — skipping")
                return
            lat = float(coords_row["lat"].iloc[0])
            lon = float(coords_row["lon"].iloc[0])
            from datetime import timedelta
            round_dates = get_round_dates()
            round_date = round_dates[round_num - 1]
            # If round dates are in the future (Monday backfill), use previous week
            if round_date.date() > datetime.now().date():
                round_date = round_date - timedelta(days=7)
                print(f"  [actuals] Using previous week date: {round_date.strftime('%Y-%m-%d')}")
            date_str = round_date.strftime("%Y-%m-%d")
            realized_wind = fetch_realized_wind(lat, lon, date_str)
            if realized_wind is None:
                print(f"  [actuals] Could not fetch realized wind — skipping actuals row")
                return
            # Write back to sheet so it persists
            param_name = f"realized_wind_r{round_num}"
            _write_param_to_sheet(ws, all_data, param_name, realized_wind)
        except Exception as e:
            print(f"  [actuals] Auto-fetch realized wind failed: {e}")
            return

    # 5. Read base_score and field_adj from pre-tourney breakdown (cols V-W of the round's row)
    # Row = 2 + round_num (R1=row3, R2=row4, etc.), Col V=22, W=23
    row_idx = 2 + round_num - 1  # 0-indexed
    base_score = 0.0
    field_adj = 0.0
    try:
        base_score = float(all_data[row_idx][21])  # col V (0-indexed = 21)
        field_adj = float(all_data[row_idx][22])   # col W (0-indexed = 22)
    except (ValueError, IndexError, TypeError):
        print(f"  [actuals] Could not read base_score/field_adj from pre-tourney breakdown")

    # 6. Compute expected and actual
    wind_impact = realized_wind * wind_factor
    dew_impact = (realized_dew - dewpoint_base) * dew_calc if dewpoint_base else 0.0

    # base_score is already the absolute baseline (not relative to par)
    expected_score = base_score + field_adj + wind_impact + dew_impact

    # Fetch actual scoring average from DataGolf.
    # The live-tournament-stats endpoint has no absolute "score" field; the
    # per-round score-to-par is in "round" (stat_round). Convert to strokes with
    # course_par, matching the df["round"] + course_par pattern used elsewhere.
    actual_score = None
    try:
        live_df = fetch_live_stats(round_num, API_KEY)
        if live_df is not None and "round" in live_df.columns:
            to_par = pd.to_numeric(live_df["round"], errors="coerce").dropna()
            if not to_par.empty:
                actual_score = float(to_par.mean() + course_par)
    except Exception as e:
        print(f"  [actuals] Could not fetch live scores: {e}")

    delta = (actual_score - expected_score) if actual_score is not None else None

    print(f"  [actuals] R{round_num}: wind={realized_wind:.1f}, "
          f"expected={expected_score:.1f}, actual={actual_score if actual_score else 'N/A'}, "
          f"delta={delta if delta is not None else 'N/A'}")

    # 7. Write to sheet (row = 10 + round_num, cols U-AC)
    import gspread
    COL_U = 21
    write_row = 10 + round_num  # R1=11, R2=12, etc.

    cells = [
        gspread.Cell(row=write_row, col=COL_U, value=f"R{round_num}"),
        gspread.Cell(row=write_row, col=COL_U + 1, value=round(realized_wind, 1)),
        gspread.Cell(row=write_row, col=COL_U + 2, value=round(forecast_dew, 1)),
        gspread.Cell(row=write_row, col=COL_U + 3, value=round(realized_dew, 1)),
        gspread.Cell(row=write_row, col=COL_U + 4, value=round(wind_impact, 3)),
        gspread.Cell(row=write_row, col=COL_U + 5, value=round(dew_impact, 3)),
        gspread.Cell(row=write_row, col=COL_U + 6, value=round(expected_score, 1) if expected_score else ""),
        gspread.Cell(row=write_row, col=COL_U + 7, value=round(actual_score, 2) if actual_score else ""),
        gspread.Cell(row=write_row, col=COL_U + 8, value=round(delta, 2) if delta is not None else ""),
    ]
    ws.update_cells(cells, value_input_option="USER_ENTERED")
    print(f"  [actuals] Wrote R{round_num} actuals to row {write_row}")


def _apply_sheet_overrides(config):
    """
    Apply Google Sheet config values as runtime overrides.
    
    The sheet provides wind/dew arrays and scoring adjustments for the
    NEXT round. This patches the module-level arrays so the rest of the
    engine uses the sheet values seamlessly.
    """
    global WIND_ARRAYS, DEW_ARRAYS, SCORE_ADJS, COURSE_SCORE_ADJS, COURSE_SCORE_MAP, dew_calculation, wind_override

    round_num = config["round_num"]
    next_round = round_num + 1 if round_num < 4 else 4

    # Load per-round wind/dew arrays from sheet (written by humidity.py)
    for rnd in range(1, 5):
        wind_key = f"wind_r{rnd}"
        dew_key = f"dew_r{rnd}"
        if config.get(wind_key):
            WIND_ARRAYS[rnd] = config[wind_key]
        if config.get(dew_key):
            DEW_ARRAYS[rnd] = config[dew_key]

    # Override with the generic wind/dew for the next round (live-round override)
    if config.get("wind"):
        WIND_ARRAYS[next_round] = config["wind"]
        print(f"  Wind array for R{next_round} loaded from sheet ({len(config['wind'])} hours)")

    if config.get("dew"):
        DEW_ARRAYS[next_round] = config["dew"]
        print(f"  Dew array for R{next_round} loaded from sheet ({len(config['dew'])} hours)")

    # Apply Bayesian blending with climo prior to all populated wind arrays
    if _CLIMO_WIND and _ROUND_DATES:
        _blend_log = []
        for rnd in range(1, 5):
            if WIND_ARRAYS[rnd]:
                _rd = _ROUND_DATES[rnd - 1] if _ROUND_DATES and len(_ROUND_DATES) >= rnd else None
                WIND_ARRAYS[rnd], _w = blend_wind_with_climo(
                    WIND_ARRAYS[rnd], _CLIMO_WIND, round_date=_rd
                )
                _blend_log.append(f"R{rnd}={_w:.0%}")
        if _blend_log:
            print(f"  Wind blended with climo prior ({', '.join(_blend_log)})")

    # Per-round expected scoring baselines from sheet (replaces the former
    # score_adj_r1..r4 constants in sim_inputs). The per-course expected_score_1/2/3
    # block below still overrides the live (next) round's value.
    for rnd in range(1, 5):
        vals = config.get(f"expected_score_r{rnd}")
        if vals:
            SCORE_ADJS[rnd] = vals[0]

    # Build per-course scoring adjustments list
    # expected_score_1 = first course_x encountered in API data, etc.
    course_adjs = []
    for key in ["expected_score_1", "expected_score_2", "expected_score_3"]:
        val = config.get(key)
        if val is not None:
            course_adjs.append(val)

    if course_adjs:
        COURSE_SCORE_ADJS = course_adjs
        SCORE_ADJS[next_round] = course_adjs[0]  # Default/single-course fallback
        if len(course_adjs) > 1:
            print(f"  Multi-course score adjs: {course_adjs}")
        else:
            print(f"  Score adj R{next_round}: {course_adjs[0]}")

    # Build name-keyed map: API course_code → expected_score
    course_codes = config.get("course_codes", [])
    COURSE_SCORE_MAP = {}
    for i, code in enumerate(course_codes):
        if i < len(course_adjs):
            COURSE_SCORE_MAP[code] = course_adjs[i]

    if COURSE_SCORE_MAP and len(COURSE_SCORE_MAP) > 1:
        print(f"  Course score map: {COURSE_SCORE_MAP}")

    # Override dew/wind calculation factors if set in sheet.
    # Must assign THIS module's globals (declared above) — patching
    # sim_inputs alone does nothing here because the consumers read the
    # from-import copies bound at import time. Guard on non-zero: blank
    # sheet cells parse to 0.0 and must not clobber sim_inputs values
    # (dew_calculation=0.0 would silently zero every dew adjustment).
    if config.get("dew_calculation"):
        import sim_inputs
        sim_inputs.dew_calculation = config["dew_calculation"]
        dew_calculation = config["dew_calculation"]
        print(f"  Dew calculation factor: {config['dew_calculation']}")

    if config.get("wind_override"):
        import sim_inputs
        sim_inputs.wind_override = config["wind_override"]
        wind_override = config["wind_override"]
        print(f"  Wind override: {config['wind_override']}")


def main():
    """
    Entry point. Reads config from Google Sheet by default.
    Falls back to CLI args if --cli flag is passed or sheet read fails.
    
    The engine automatically determines what it can do:
      1. Always runs skill update first
      2. Then attempts weather/predictions for the next round
         - If tee times are available → creates full predictions
         - If tee times aren't available yet → saves skill-only, tells you to run again later
      3. R4 → skill update only (no next round)
      4. Round 0 → pre-event prediction creation
    
    Google Sheet mode (default — just hit run):
        python live_stats_engine.py
    
    CLI mode (fallback):
        python live_stats_engine.py --cli --round 2
    """
    parser = argparse.ArgumentParser(description="Unified Live Stats Engine")
    parser.add_argument("--cli", action="store_true",
                        help="Use CLI args instead of Google Sheet config")
    parser.add_argument("--round", type=int, choices=[0, 1, 2, 3, 4],
                        help="Round that just completed (0 = pre-event)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip email sending (skill update still runs)")

    args = parser.parse_args()

    # ── Google Sheet mode (default) ──────────────────────────────────────
    if not args.cli:
        try:
            from sheet_config import load_config
            config = load_config()
            _apply_sheet_overrides(config)
            round_num = config["round_num"]
        except Exception as e:
            print(f"\n[warn] Could not read Google Sheet: {e}")
            print("   Falling back to CLI args. Use --cli flag to suppress this.\n")
            if args.round is None:
                parser.error("Sheet unavailable and no --round provided.")
            round_num = args.round
    else:
        # ── CLI mode ─────────────────────────────────────────────────────
        if args.round is None:
            parser.error("--round is required in CLI mode")
        round_num = args.round

    # ── Dispatch ─────────────────────────────────────────────────────────
    if round_num == 0:
        create_pre_event_predictions()
        return

    # Step 1: Always run skill update
    run_skill_update(round_num, dry_run=args.dry_run)

    # Step 1b: Write actuals for the completed round
    try:
        write_actuals_to_sheet(round_num)
    except Exception as e:
        print(f"\n[warn] Actuals write failed: {e}")

    # Step 2: Attempt weather/predictions for next round
    if round_num < 4:
        print(f"\n  Attempting to create R{round_num + 1} predictions...")
        try:
            run_weather_update(round_num)
        except Exception as e:
            print(f"\n[warn] Weather update could not complete: {e}")
            print(f"   Skill update is saved. Run again once R{round_num + 1} tee times are available.")
    else:
        print("\n  R4 complete — no next round. Skill update saved for records.")

    # Step 3: Update expected scoring averages for remaining rounds
    if round_num < 4:
        try:
            update_expected_scores(round_num)
        except Exception as e:
            print(f"\n[warn] Expected score update failed: {e}")
            import traceback; traceback.print_exc()


def update_expected_scores(completed_round):
    """
    Recompute expected scoring averages for future rounds using current
    wind forecasts from the Sheet, and write them back to expected_score_r{N}.

    Formula: expected_score = baseline - field_strength + avg_wind * wind_factor

    Uses:
      - Baselines from scoring_baseline_{tourney}.csv (FINAL rows)
      - Field strength from final_predictions or pre_course_fit CSV
      - Wind factor from compute_wind_factor()
      - Per-round wind arrays from WIND_ARRAYS (already loaded from Sheet)
    """
    import os
    from api_utils import compute_wind_factor
    from scoring_baseline import _compute_field_avg_wind

    print(f"\n{'='*60}")
    print(f"  UPDATING EXPECTED SCORES (R{completed_round + 1}-R4)")
    print(f"{'='*60}")

    api_key = os.getenv("DATAGOLF_API_KEY")

    # 1. Load baselines from scoring_baseline CSV
    baseline_file = f"scoring_baseline_{tourney}.csv"
    if not os.path.exists(baseline_file):
        print(f"  {baseline_file} not found — cannot update expected scores")
        return

    baseline_df = pd.read_csv(baseline_file)
    finals = baseline_df[baseline_df["year"] == "FINAL"]
    if finals.empty:
        print(f"  No FINAL rows in {baseline_file}")
        return

    baselines = {}
    for _, row in finals.iterrows():
        rnd = int(row["round_num"])
        baselines[rnd] = float(row["baseline"])
    print(f"  Baselines: {baselines}")

    # 2. Field strength from predictions
    field_strength = 0.0
    for pred_file in [f"final_predictions_{tourney}.csv", f"pre_course_fit_{tourney}.csv"]:
        path = _resolve_csv(pred_file)
        if os.path.exists(path):
            pred_df = pd.read_csv(path)
            col = "my_pred" if "my_pred" in pred_df.columns else "pred"
            if col in pred_df.columns:
                field_strength = pred_df[col].mean()
                print(f"  Field strength: {field_strength:+.4f} (from {path})")
            break
    if field_strength == 0.0:
        print(f"  WARNING: No prediction file found — field_strength = 0.0")

    # 3. Wind factor
    wf = compute_wind_factor(event_ids, wind_override, baseline_wind)
    print(f"  Wind factor: {wf:.4f}")

    # 4. Cut adjustment for R3/R4
    try:
        from sim_inputs import CUT_LINE
        cut_adj = 0.25 if CUT_LINE < 120 else 0.0
    except ImportError:
        cut_adj = 0.25

    # 5. Compute for future rounds
    future_rounds = range(completed_round + 1, 5)
    adjusted = {}

    print(f"\n  {'Rnd':>3} | {'Baseline':>8} | {'Wind Avg':>8} | {'Wind Eff':>8} | {'Field':>8} | {'Expected':>8}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for rnd in future_rounds:
        base = baselines.get(rnd)
        if base is None:
            print(f"  R{rnd}: no baseline — skipping")
            continue

        wind_arr = WIND_ARRAYS.get(rnd, [])
        avg_wind = _compute_field_avg_wind(wind_arr, api_key, rnd) if wind_arr else 0.0
        wind_effect = avg_wind * wf

        fld = -field_strength
        if rnd >= 3:
            fld -= cut_adj

        expected = base + fld + wind_effect
        adjusted[rnd] = round(expected, 1)

        print(f"  R{rnd}  | {base:>8.2f} | {avg_wind:>8.1f} | {wind_effect:>+8.3f} | {fld:>+8.3f} | {adjusted[rnd]:>8.1f}")

    if not adjusted:
        print("  No rounds to update.")
        return

    # 6. Write to Sheet
    try:
        from sheets_storage import get_spreadsheet
        spreadsheet = get_spreadsheet()
        ws = spreadsheet.worksheet("round_config")

        all_values = ws.get("A:C")
        param_rows = {}
        for i, row in enumerate(all_values):
            if row and row[0].strip():
                param_rows[row[0].strip().lower()] = i + 1

        for rnd, value in adjusted.items():
            param_name = f"expected_score_r{rnd}"
            row_idx = param_rows.get(param_name)
            if row_idx:
                ws.update_cell(row_idx, 2, str(value))
                ws.update_cell(row_idx, 3, f"Updated by live_stats_engine (after R{completed_round})")
                print(f"  Wrote {param_name} = {value}")
            else:
                print(f"  WARNING: {param_name} row not found in Sheet")

        print(f"  Expected scores updated in Sheet.")
    except Exception as e:
        print(f"  WARNING: Could not write to Sheet: {e}")


if __name__ == "__main__":
    main()
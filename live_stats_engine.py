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

from sim_inputs import (
    tourney, course_par, event_ids, wind_override, baseline_wind,
    dew_calculation,
    wind_1, wind_2, wind_3, wind_4,
    dewpoint_1, dewpoint_2, dewpoint_3, dewpoint_4,
    score_adj_r1, score_adj_r2, score_adj_r3, score_adj_r4,
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
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_KEY = "c05ee5fd8f2f3b14baab409bd83c"

# Wind/dewpoint arrays indexed by round (1-based; index 0 unused)
WIND_ARRAYS = {1: wind_1, 2: wind_2, 3: wind_3, 4: wind_4}
DEW_ARRAYS = {1: dewpoint_1, 2: dewpoint_2, 3: dewpoint_3, 4: dewpoint_4}
SCORE_ADJS = {1: score_adj_r1, 2: score_adj_r2, 3: score_adj_r3, 4: score_adj_r4}

# Multi-course expected scoring adjustments (populated from Google Sheet).
# Index 0 = first course_x encountered in API data, index 1 = second, etc.
# For single-course weeks, only index 0 is used.
COURSE_SCORE_ADJS = None  # Set by _apply_sheet_overrides; None = use SCORE_ADJS


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
    "residual": "residual", "residual2": "residual2", "residual3": "residual3",
    "avg_ott": "sg_ott_avg", "avg_putt": "sg_putt_avg", "avg_app": "sg_app_avg",
    "avg_arg": "sg_arg_avg", "delta_app": "sg_app_delta",
}
_R3R4_COL_MAP = {
    "sg_ott_avg": "sg_ott_avg", "sg_putt_avg": "sg_putt_avg",
    "sg_app_avg": "sg_app_avg", "sg_arg_avg": "sg_arg_avg",
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

    return df


def _merge_r1(df):
    """
    R1 merge: load model_predictions_r1.csv (created pre-event).
    Source: live_stats.py lines 148-162
    """
    preds = pd.read_csv("model_predictions_r1.csv")
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
    r1_model = pd.read_csv("r1_live_model.csv")
    r1_model = clean_names(r1_model)
    r1_stats = ["great_shots", "poor_shots", "sg_app", "sg_arg", "sg_ott", "sg_putt"]
    r1_keep = ["player_name"] + r1_stats
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

    # --- Load R2 predictions (created by prior round's weather step) ---
    r2_preds = pd.read_csv("model_predictions_r2.csv")
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
        print("  ⚠️  r2_teetime not found in model_predictions_r2.csv — spline will use index order")

    return df


def _merge_r3r4(df, round_num):
    """
    R3/R4 merge: load prior round model + current round predictions.
    Source: live_stats_r3.py lines 65-85 / live_stats_r4.py equivalent
    """
    prior_round = round_num - 1
    prior_file = f"r{prior_round}_live_model.csv"
    pred_file = f"model_predictions_r{round_num}.csv"

    # --- Prior round model (for carried-forward skill + adjustments) ---
    prior = pd.read_csv(prior_file)
    prior = clean_names(prior)
    prior_cols = [
        "player_name", f"updated_pred_r{round_num}",
        "sg_app_avg", "sg_ott_avg", "sg_arg_avg", "sg_putt_avg",
        "tot_resid_adj", "tot_sg_adj",
    ]
    prior_cols = [c for c in prior_cols if c in prior.columns]
    df = df.merge(prior[prior_cols], on="player_name", how="left")

    # --- Current round predictions (wind/dew/teetime) ---
    if os.path.exists(pred_file):
        cur_preds = pd.read_csv(pred_file)
        cur_preds = clean_names(cur_preds)
        teetime_col = f"r{round_num}_teetime"
        cur_cols = ["player_name", f"wind_adj{round_num}", f"dew_adj{round_num}", teetime_col]
        cur_cols = [c for c in cur_cols if c in cur_preds.columns]
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
    df["residual2"] = df["residual"] ** 2
    df["residual3"] = df["residual"] ** 3
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

    # Residual cap logic (from live_stats.py)
    df["tot_resid_adj"] = df["residual_adj"] + df["residual2_adj"]
    df["tot_resid_adj"] = np.minimum(
        np.where(
            (df["tot_resid_adj"] > 0.2) & (df["residual"] < 0),
            0.2,
            df["tot_resid_adj"],
        ),
        0.5,
    )

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

    # Total adjustment = residual components + SG components
    # Explicitly listed to avoid catching sg_total_adj (which is raw data, not an adjustment)
    adj_components = [
        "residual_adj", "residual2_adj", "residual3_adj",
        "avg_ott_adj", "avg_putt_adj", "avg_app_adj", "avg_arg_adj", "delta_app_adj",
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
    adj_cols = ["sg_ott_avg_adj", "sg_putt_avg_adj", "sg_app_avg_adj", "sg_arg_avg_adj"]
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
    live_model = pd.read_csv(f"r{round_num}_live_model.csv")
    live_model = clean_names(live_model)

    # Determine the skill column name
    if round_num == 1:
        skill_col = "updated_pred"
    else:
        skill_col = f"updated_pred_r{next_round}"

    if skill_col not in live_model.columns:
        raise ValueError(f"Column '{skill_col}' not found in r{round_num}_live_model.csv")

    # Build predictions DataFrame
    keep_cols = ["player_name", skill_col]
    if "std_dev" in live_model.columns:
        keep_cols.append("std_dev")
    # Carry course_x through for multi-course sim support
    course_col = "course" if "course" in live_model.columns else "course_x" if "course_x" in live_model.columns else None
    if course_col:
        keep_cols.append(course_col)

    preds = live_model[[c for c in keep_cols if c in live_model.columns]].copy()

    # Standardize skill column name: my_pred for R1, my_pred{N} for R2+
    pred_name = "my_pred" if next_round == 1 else f"my_pred{next_round}"
    preds = preds.rename(columns={skill_col: pred_name})

    # --- Load tee times ---
    teetime_col = f"r{next_round}_teetime"
    field = fetch_field_updates(API_KEY, teetime_col=teetime_col, include_course=(next_round <= 3))
    if field is not None and teetime_col in field.columns:
        preds = preds.merge(field[["player_name", teetime_col]], on="player_name", how="left")

        # Drop players without tee times (cut / withdrawn after this round)
        before = len(preds)
        preds = preds.dropna(subset=[teetime_col])
        cut = before - len(preds)
        if cut > 0:
            print(f"  Dropped {cut} players without R{next_round} tee times (cut/inactive)")
    else:
        print(f"Warning: Tee times for R{next_round} not yet available.")
        print("Saving skill-only predictions. Run again once tee times are available.")
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

    # Expected score: skill + avg_field_wind - player_wind_advantage + player_dew
    preds[f"scores_r{next_round}"] = (
        preds[pred_name] + avg_wind - preds[wind_col] + preds[dew_col]
    )

    # --- Diagnostics ---
    avg_wind_speed = preds[f"wind_r{next_round}"].mean()
    print(f"  Avg wind speed R{next_round}: {avg_wind_speed:.2f} mph")
    print(f"  Avg wind impact in SG: {avg_wind:.4f}")
    print(f"  Avg skill R{next_round}: {preds[pred_name].mean():.4f}")

    # Per-course expected scoring (multi-course aware)
    course_col = "course" if "course" in preds.columns else "course_x" if "course_x" in preds.columns else None
    if course_col and preds[course_col].nunique() > 1 and COURSE_SCORE_ADJS and len(COURSE_SCORE_ADJS) > 1:
        courses_ordered = [c for c in preds[course_col].unique() if pd.notna(c)]
        print(f"  Multi-course mapping (order of appearance):")
        for i, cid in enumerate(courses_ordered):
            adj = COURSE_SCORE_ADJS[i] if i < len(COURSE_SCORE_ADJS) else COURSE_SCORE_ADJS[0]
            course_players = preds[preds[course_col] == cid]
            course_skill = course_players[pred_name].mean()
            course_wind = course_players[wind_col].mean()
            exp_score = round(adj - course_skill + course_wind, 2)
            print(f"    expected_score_{i+1} → {cid}: adj={adj}, expected scoring={exp_score}")
            # Store the course_x → score_adj mapping on the DataFrame
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
    print(f"  ✓ Saved {filename} ({len(preds)} players)")


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
        wind_vals.append(calculate_average_wind(tt, wind_1))
        dew_vals.append(calculate_average_wind(tt, dewpoint_1))

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
    expected_scoring = round(score_adj_r1 - avg_skill + avg_wind, 2)
    preds["scores_r1"] = preds["my_pred"] + avg_wind - preds["wind_adj1"] + preds["dew_adj1"]

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
    print(f"  ✓ Saved model_predictions_r1.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(df, round_num):
    """Export live model CSV, summary CSV, residual summary, and plot."""

    # --- Full model ---
    model_file = f"r{round_num}_live_model.csv"
    df.to_csv(model_file, index=False)
    df.to_csv(f"r{round_num}_live_model_{tourney}.csv", index=False)
    print(f"  ✓ Saved {model_file}")

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
            "delta_app_adj", "updated_pred_r3",
        ]
    else:
        adj_cols = [f"{c}_adj_r{round_num}" for c in
                    ["sg_ott_avg", "sg_putt_avg", "sg_app_avg", "sg_arg_avg"]]
        next_pred = f"updated_pred_r{round_num + 1}" if round_num < 4 else "updated_pred_final"
        summary_cols = ["player_name"] + adj_cols + [f"updated_pred_r{round_num}", next_pred]

    existing = [c for c in summary_cols if c in df.columns]
    summary_file = f"r{round_num}_live_summary.csv"
    df[existing].to_csv(summary_file, index=False)
    print(f"  ✓ Saved {summary_file}")

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
                print(f"  ✓ Saved {spline_pdf_path}")
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
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not password:
        print("  ⚠️  GMAIL_APP_PASSWORD not set. Skipping email.")
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

        print("  ✓ Summary email sent")

    except Exception as e:
        print(f"  ⚠️  Email failed: {e}")
        print("    (Skill update still saved — email is non-blocking)")


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Points
# ══════════════════════════════════════════════════════════════════════════════

def run_skill_update(round_num):
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
    send_summary_email(df, round_num, spline_pdf_path=spline_pdf_path)

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

def _apply_sheet_overrides(config):
    """
    Apply Google Sheet config values as runtime overrides.
    
    The sheet provides wind/dew arrays and scoring adjustments for the
    NEXT round. This patches the module-level arrays so the rest of the
    engine uses the sheet values seamlessly.
    """
    global WIND_ARRAYS, DEW_ARRAYS, SCORE_ADJS, COURSE_SCORE_ADJS, dew_calculation, wind_override

    round_num = config["round_num"]
    next_round = round_num + 1 if round_num < 4 else 4

    # Override wind/dew arrays for the next round
    if config.get("wind"):
        WIND_ARRAYS[next_round] = config["wind"]
        print(f"  → Wind array for R{next_round} loaded from sheet ({len(config['wind'])} hours)")

    if config.get("dew"):
        DEW_ARRAYS[next_round] = config["dew"]
        print(f"  → Dew array for R{next_round} loaded from sheet ({len(config['dew'])} hours)")

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
            print(f"  → Multi-course score adjs: {course_adjs} (mapped to courses in order of appearance)")
        else:
            print(f"  → Score adj R{next_round}: {course_adjs[0]}")

    # Override dew/wind calculation factors if set in sheet
    if config.get("dew_calculation") is not None:
        # Patch the imported value
        import sim_inputs
        sim_inputs.dew_calculation = config["dew_calculation"]
        print(f"  → Dew calculation factor: {config['dew_calculation']}")

    if config.get("wind_override") is not None:
        import sim_inputs
        sim_inputs.wind_override = config["wind_override"]
        print(f"  → Wind override: {config['wind_override']}")


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

    args = parser.parse_args()

    # ── Google Sheet mode (default) ──────────────────────────────────────
    if not args.cli:
        try:
            from sheet_config import load_config
            config = load_config()
            _apply_sheet_overrides(config)
            round_num = config["round_num"]
        except Exception as e:
            print(f"\n⚠️  Could not read Google Sheet: {e}")
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
    run_skill_update(round_num)

    # Step 2: Attempt weather/predictions for next round
    if round_num < 4:
        print(f"\n  Attempting to create R{round_num + 1} predictions...")
        try:
            run_weather_update(round_num)
        except Exception as e:
            print(f"\n⚠️  Weather update could not complete: {e}")
            print(f"   Skill update is saved. Run again once R{round_num + 1} tee times are available.")
    else:
        print("\n  R4 complete — no next round. Skill update saved for records.")


if __name__ == "__main__":
    main()
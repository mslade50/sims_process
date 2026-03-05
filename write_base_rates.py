"""
write_base_rates.py — Write a structured base-rates reference table to Google Sheets.

Reads from sim_inputs.py, wind_test.csv, scoring_baseline_{tourney}.csv,
dg_historical.db, and weather forecast arrays. Writes a "Base Rates" tab
with Category | Parameter | Base Rate | This Week | Delta | Notes.

Pipeline position:
  cat_dists_player.py → dists_thiswk.py → humidity.py → scoring_baseline.py → write_base_rates.py
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

import sim_inputs
from api_utils import calculate_average_wind
from sheets_storage import store_base_rates, get_spreadsheet

# ---------------------------------------------------------------------------
# Constants / defaults (tour-average base rates)
# ---------------------------------------------------------------------------
BASE_DEW_CALCULATION = -0.0254
BASE_BASELINE_WIND = 0.12
BASE_COURSE_WIND_EFFECT = 0.08
BASE_WIND_FACTOR_SIM = 0.15
BASE_WIND_SPEED = 12.2
BASE_DEW_AVG = 54.0
BASE_COURSE_PAR = 72
BASE_CAT_MULT = 1.00
BASE_CAT_SKEW = {
    "sg_ott": -0.93,
    "sg_app": -0.21,
    "sg_arg": -0.18,
    "sg_putt": -0.05,
}

DB_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "dg_historical.db")
WIND_TEST_PATH = os.path.join(os.path.dirname(__file__), "permanent_data", "wind_test.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delta(this_week, base):
    """Compute delta, return None if either side is non-numeric."""
    try:
        tw = float(this_week)
        br = float(base)
        return round(tw - br, 4)
    except (TypeError, ValueError):
        return ""


def _mult_delta(this_week, base=1.0):
    """For multipliers, show % deviation from 1.0."""
    try:
        tw = float(this_week)
        return f"{(tw - base) * 100:+.0f}%"
    except (TypeError, ValueError):
        return ""


def _avg_array(arr):
    """Mean of a numeric list/array."""
    if not arr:
        return 0.0
    return round(float(np.mean(arr)), 1)


def _parse_hour(tt):
    """Parse teetime like '9:45am' or '1:20pm' to decimal hour."""
    s = str(tt).strip().lower()
    if not s:
        return None
    for fmt in ["%I:%M%p", "%Y-%m-%d %H:%M"]:
        try:
            parsed = datetime.strptime(s, fmt)
            return parsed.hour + parsed.minute / 60.0
        except ValueError:
            continue
    return None


def _lookup_course_wind_effect(event_ids):
    """Look up course wind effect from permanent_data/wind_test.csv."""
    try:
        df = pd.read_csv(WIND_TEST_PATH)
        first_id = str(event_ids[0]).strip()
        match = df[
            df["event_ids"].apply(
                lambda x: first_id in [s.strip() for s in str(x).split(",")]
            )
        ]
        if match.empty:
            return BASE_COURSE_WIND_EFFECT, "default (no match)"
        return float(match["wind_effect_adj_score"].iloc[-1]), ""
    except FileNotFoundError:
        return BASE_COURSE_WIND_EFFECT, "wind_test.csv not found"


def _load_scoring_baseline(tourney):
    """Load scoring_baseline_{tourney}.csv — returns (full_df, error_note)."""
    path = f"scoring_baseline_{tourney}.csv"
    if not os.path.exists(path):
        return None, f"{path} not found"
    try:
        df = pd.read_csv(path)
        return df, ""
    except Exception as e:
        return None, str(e)


def _compute_am_pm_wind_dew(wind_arr, dew_arr):
    """
    Compute AM vs PM 5-hour averages for wind and dew.
    AM tee = 7:30 AM, PM tee = 12:30 PM.
    Returns (am_wind, pm_wind, am_dew, pm_dew).
    """
    am_tee = "7:30am"
    pm_tee = "12:30pm"
    am_wind = calculate_average_wind(am_tee, wind_arr) if wind_arr else 0.0
    pm_wind = calculate_average_wind(pm_tee, wind_arr) if wind_arr else 0.0
    am_dew = calculate_average_wind(am_tee, dew_arr) if dew_arr else 0.0
    pm_dew = calculate_average_wind(pm_tee, dew_arr) if dew_arr else 0.0
    return am_wind, pm_wind, am_dew, pm_dew


def _query_historical_am_pm(event_ids):
    """
    Query dg_historical.db for historical AM/PM scoring splits for R1/R2
    at THIS course.
    Returns dict: {1: am_minus_pm, 2: am_minus_pm} in raw strokes.
    Positive = AM scored higher (worse).
    """
    result = {1: 0.0, 2: 0.0}
    if not os.path.exists(DB_PATH):
        return result, "dg_historical.db not found"

    try:
        conn = sqlite3.connect(DB_PATH)
        first_id = int(event_ids[0])
        df = pd.read_sql_query(
            """
            SELECT round_num, teetime, score, course_par
            FROM player_rounds
            WHERE event_id = ? AND round_num <= 2
              AND teetime IS NOT NULL AND teetime != ''
            """,
            conn,
            params=(first_id,),
        )
        conn.close()
    except Exception as e:
        return result, str(e)

    if df.empty:
        return result, "no data"

    df["hour"] = df["teetime"].apply(_parse_hour)
    df = df.dropna(subset=["hour", "score"])

    if df.empty:
        return result, "no parseable teetimes"

    for rnd in [1, 2]:
        rnd_df = df[df["round_num"] == rnd]
        if rnd_df.empty:
            continue
        am = rnd_df[rnd_df["hour"] < 11.0]["score"]
        pm = rnd_df[rnd_df["hour"] >= 11.0]["score"]
        if len(am) > 5 and len(pm) > 5:
            result[rnd] = round(float(am.mean() - pm.mean()), 2)

    return result, ""


def _query_tour_wide_am_pm():
    """
    Compute tour-wide AM/PM scoring differential for R1/R2 across all
    PGA events since 2019. Returns {1: diff, 2: diff}.
    """
    result = {1: 0.0, 2: 0.0}
    if not os.path.exists(DB_PATH):
        return result, "dg_historical.db not found"

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT round_num, teetime, score
            FROM player_rounds
            WHERE year >= 2019 AND round_num <= 2
              AND teetime IS NOT NULL AND teetime != ''
              AND tour = 'pga'
            """,
            conn,
        )
        conn.close()
    except Exception as e:
        return result, str(e)

    if df.empty:
        return result, "no data"

    df["hour"] = df["teetime"].apply(_parse_hour)
    df = df.dropna(subset=["hour", "score"])

    for rnd in [1, 2]:
        rnd_df = df[df["round_num"] == rnd]
        am = rnd_df[rnd_df["hour"] < 11.0]["score"]
        pm = rnd_df[rnd_df["hour"] >= 11.0]["score"]
        if len(am) > 10 and len(pm) > 10:
            result[rnd] = round(float(am.mean() - pm.mean()), 2)

    return result, ""


def _query_tour_wide_scoring_std():
    """
    Compute tour-wide scoring standard deviation (mean of per-event-round stds)
    across all PGA events since 2019.
    """
    if not os.path.exists(DB_PATH):
        return 2.88, "dg_historical.db not found; using fallback"

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT event_id, year, round_num, score
            FROM player_rounds
            WHERE year >= 2019 AND tour = 'pga'
              AND score IS NOT NULL
            """,
            conn,
        )
        conn.close()
    except Exception as e:
        return 2.88, str(e)

    if df.empty:
        return 2.88, "no data"

    grouped = df.groupby(["event_id", "year", "round_num"])["score"].std().dropna()
    return round(float(grouped.mean()), 2), ""


# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------

def build_rows():
    """Assemble all base-rate rows."""
    rows = []

    def add(cat, param, base, this_week, notes=""):
        d = _delta(this_week, base)
        rows.append([cat, param, base, this_week, d, notes])

    def add_mult(cat, param, base, this_week, notes=""):
        d = _mult_delta(this_week, base)
        rows.append([cat, param, base, this_week, d, notes])

    # --- Weather Coefficients ---
    add("Weather Coeff", "dew_calculation", BASE_DEW_CALCULATION, round(sim_inputs.dew_calculation, 4))
    add("Weather Coeff", "baseline_wind", BASE_BASELINE_WIND, sim_inputs.baseline_wind)

    course_wind_eff, cw_note = _lookup_course_wind_effect(sim_inputs.event_ids)
    add("Weather Coeff", "course_wind_effect", BASE_COURSE_WIND_EFFECT, course_wind_eff, cw_note)

    blended = course_wind_eff * 0.4 + sim_inputs.baseline_wind * 0.6
    add("Weather Coeff", "blended_wind_factor", BASE_COURSE_WIND_EFFECT * 0.4 + BASE_BASELINE_WIND * 0.6,
        round(blended, 4), "course*0.4 + baseline*0.6")

    add("Weather Coeff", "wind_speed_base", BASE_WIND_SPEED, sim_inputs.wind_speed_base)

    # --- Per-Round Weather ---
    wind_arrays = {
        1: getattr(sim_inputs, "wind_1", []),
        2: getattr(sim_inputs, "wind_2", []),
        3: getattr(sim_inputs, "wind_3", []),
        4: getattr(sim_inputs, "wind_4", []),
    }
    dew_arrays = {
        1: getattr(sim_inputs, "dewpoint_1", []),
        2: getattr(sim_inputs, "dewpoint_2", []),
        3: getattr(sim_inputs, "dewpoint_3", []),
        4: getattr(sim_inputs, "dewpoint_4", []),
    }
    for rnd in range(1, 5):
        avg_w = _avg_array(wind_arrays[rnd])
        add("Per-Round Weather", f"avg_wind_r{rnd}", BASE_WIND_SPEED, avg_w, f"mean of wind_{rnd} array")
        avg_d = _avg_array(dew_arrays[rnd])
        add("Per-Round Weather", f"avg_dew_r{rnd}", BASE_DEW_AVG, avg_d, f"mean of dewpoint_{rnd} array")

    # --- AM/PM Scoring Split ---
    tour_wide_splits, tw_note = _query_tour_wide_am_pm()
    hist_splits, hist_note = _query_historical_am_pm(sim_inputs.event_ids)
    for rnd in [1, 2]:
        add("AM/PM Split", f"hist_am_pm_diff_r{rnd}", tour_wide_splits[rnd], hist_splits[rnd],
            f"AM-PM strokes (tour avg vs this course){'; ' + hist_note if hist_note else ''}")

    for rnd in [1, 2]:
        w_arr = wind_arrays[rnd]
        d_arr = dew_arrays[rnd]
        am_w, pm_w, am_d, pm_d = _compute_am_pm_wind_dew(w_arr, d_arr)
        add("AM/PM Split", f"this_wk_am_pm_wind_r{rnd}", tour_wide_splits[rnd], round(am_w - pm_w, 2),
            f"AM 5hr avg {am_w:.1f} - PM {pm_w:.1f} mph")
        add("AM/PM Split", f"this_wk_am_pm_dew_r{rnd}", tour_wide_splits[rnd], round(am_d - pm_d, 1),
            f"AM 5hr avg {am_d:.1f} - PM {pm_d:.1f} F")

    # --- Sim Core ---
    tour_std, std_note = _query_tour_wide_scoring_std()
    add("Sim Core", "STD_DEV", tour_std, sim_inputs.STD_DEV,
        f"tour avg since 2019{'; ' + std_note if std_note else ''}")

    # --- Cat Variance (V2) ---
    cat_mults = getattr(sim_inputs, "COURSE_CAT_MULTS", {})
    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        val = cat_mults.get(cat, 1.0)
        add_mult("Cat Variance (V2)", f"cat_mult_{cat.replace('sg_', '')}", BASE_CAT_MULT, val)

    # --- Cat Skew (V2) ---
    cat_skew = getattr(sim_inputs, "COURSE_CAT_SKEW", {})
    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        base_sk = BASE_CAT_SKEW.get(cat, 0.0)
        val = cat_skew.get(cat, base_sk)
        add("Cat Skew (V2)", f"cat_skew_{cat.replace('sg_', '')}", base_sk, val)

    # --- Scoring Baseline ---
    sb_df, sb_note = _load_scoring_baseline(sim_inputs.tourney)
    if sb_df is not None:
        finals = sb_df[sb_df["year"] == "FINAL"]
        detail = sb_df[sb_df["year"] != "FINAL"]

        for _, row in finals.iterrows():
            rnd = int(row["round_num"])
            this_week_val = row.get("baseline", sim_inputs.course_par)

            # Base rate = weighted avg of raw_avg for this round at this course
            rnd_detail = detail[detail["round_num"] == rnd].copy()
            rnd_detail["raw_avg"] = pd.to_numeric(rnd_detail["raw_avg"], errors="coerce")
            rnd_detail["weight"] = pd.to_numeric(rnd_detail["weight"], errors="coerce")
            rnd_detail = rnd_detail.dropna(subset=["raw_avg", "weight"])
            if not rnd_detail.empty:
                base_val = round(
                    (rnd_detail["raw_avg"] * rnd_detail["weight"]).sum() / rnd_detail["weight"].sum(), 2
                )
            else:
                base_val = sim_inputs.course_par
            add("Scoring Baseline", f"expected_score_r{rnd}", base_val, this_week_val)

        # Historical scoring std (weighted average across detail rows)
        if "score_std" in detail.columns and "weight" in detail.columns:
            detail_clean = detail.dropna(subset=["score_std", "weight"]).copy()
            if not detail_clean.empty:
                detail_clean["score_std"] = pd.to_numeric(detail_clean["score_std"], errors="coerce")
                detail_clean["weight"] = pd.to_numeric(detail_clean["weight"], errors="coerce")
                detail_clean = detail_clean.dropna(subset=["score_std", "weight"])
                if not detail_clean.empty:
                    weighted_std = (detail_clean["score_std"] * detail_clean["weight"]).sum() / detail_clean["weight"].sum()
                    add("Scoring Baseline", "hist_scoring_std", tour_std, round(weighted_std, 2))
    else:
        rows.append(["Scoring Baseline", "(unavailable)", "", "", "", sb_note])

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"  BASE RATES REFERENCE")
    print(f"  Tournament: {sim_inputs.tourney} | Event IDs: {sim_inputs.event_ids}")
    print(f"{'='*70}\n")

    rows = build_rows()
    print(f"  Built {len(rows)} rows")

    spreadsheet = get_spreadsheet()
    store_base_rates(rows, spreadsheet=spreadsheet)

    print(f"\n  Done — check 'Base Rates' tab in Google Sheets")


if __name__ == "__main__":
    main()

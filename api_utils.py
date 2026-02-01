"""
Shared DataGolf API utilities and helper functions.
Extracted from: live_stats.py, live_stats_r2.py, live_stats_r3.py, live_stats_r4.py,
                rd_1_sd_multicourse_sim.py

This module centralizes:
  - DataGolf API fetching (live stats + field updates)
  - Wind/dew calculation from hourly arrays
  - Wind factor computation from historical data
  - Name cleaning
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

from sim_inputs import name_replacements

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DATAGOLF_BASE = "https://feeds.datagolf.com"

ALL_STATS = [
    "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_bs", "sg_total",
    "distance", "accuracy", "gir", "prox_fw", "prox_rgh", "scrambling",
    "great_shots", "poor_shots",
]


# --------------------------------------------------------------------------
# DataGolf API Functions
# --------------------------------------------------------------------------

def fetch_live_stats(round_num, api_key, include_score=False):
    """
    Fetch live tournament stats from DataGolf.
    
    Source: Originally duplicated across all 4 live_stats files.
    
    Args:
        round_num: Which round to fetch (1-4)
        api_key: DataGolf API key
        include_score: If True, adds "score" to requested stats (needed for R1 
                       satellite course proxy when ShotLink unavailable)
    
    Returns:
        DataFrame with live stats + metadata columns, or None on failure.
    """
    stats = ALL_STATS + (["score"] if include_score else [])
    params = {
        "stats": ",".join(stats),
        "round": round_num,
        "display": "value",
        "file_format": "json",
        "key": api_key,
    }

    resp = requests.get(f"{DATAGOLF_BASE}/preds/live-tournament-stats", params=params)
    if resp.status_code != 200:
        print(f"Error fetching stats: {resp.status_code} - {resp.text}")
        return None

    data = resp.json()
    if "live_stats" not in data:
        print("No 'live_stats' field found in response.")
        return None

    df = pd.DataFrame(data["live_stats"])
    df["course_name"] = data.get("course_name")
    df["event_name"] = data.get("event_name")
    df["last_updated"] = data.get("last_updated")

    # Reorder so player_name is first
    cols = ["player_name"] + [c for c in df.columns if c != "player_name"]
    df = df[cols]

    # Standardize names
    df["player_name"] = df["player_name"].str.lower().replace(name_replacements)

    # Round numeric columns (position excluded — handled by engine after T-stripping)
    round_cols = [
        "accuracy", "distance", "gir", "great_shots", "poor_shots",
        "prox_fw", "prox_rgh", "round", "scrambling",
        "sg_app", "sg_arg", "sg_bs", "sg_ott", "sg_putt", "sg_t2g", "sg_total",
    ]
    for c in round_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return df


def fetch_field_updates(api_key, teetime_col="r1_teetime", include_course=False):
    """
    Fetch field updates (tee times, course assignments) from DataGolf.
    
    Source: Originally duplicated across all 4 live_stats files.
    
    Args:
        api_key: DataGolf API key
        teetime_col: Which round's tee time column to extract (e.g. 'r1_teetime')
        include_course: If True, also extracts the 'course' column (for multi-course events)
    
    Returns:
        DataFrame with player_name + requested columns, or None on failure.
    """
    params = {"tour": "pga", "file_format": "json", "key": api_key}
    resp = requests.get(f"{DATAGOLF_BASE}/field-updates", params=params)

    if resp.status_code != 200:
        print(f"Error fetching field updates: {resp.status_code} - {resp.text}")
        return None

    data = resp.json()
    if "field" not in data:
        print("No 'field' data found in response.")
        return None

    df = pd.DataFrame(data["field"])

    # Build list of columns to keep
    keep = ["player_name"]
    if teetime_col in df.columns:
        keep.append(teetime_col)
    if include_course and "course" in df.columns:
        keep.append("course")

    df = df[[c for c in keep if c in df.columns]].copy()
    df["player_name"] = df["player_name"].str.lower().replace(name_replacements)

    return df


# --------------------------------------------------------------------------
# Wind / Dew Calculation
# --------------------------------------------------------------------------

def calculate_average_wind(teetime, wind_data):
    """
    Calculate average wind/dew over a 5-hour window starting at tee time.
    
    Source: rd_1_sd_multicourse_sim.py lines 51-82 (identical logic in new_sim.py)
    
    Wind data is an array of hourly values starting at 6 AM.
    We interpolate to minute-level resolution and average over a 5-hour window.
    
    Args:
        teetime: Tee time as string or datetime. Supports formats:
                 '%Y-%m-%d %H:%M', '%I:%M%p', '%m/%d/%Y %H:%M'
        wind_data: List/array of hourly values (index 0 = 6 AM)
    
    Returns:
        Float: average value over the 5-hour window, or 0.0 on parse failure.
    """
    if pd.isnull(teetime):
        return 0.0

    # Parse tee time
    parsed = None
    if isinstance(teetime, datetime):
        parsed = teetime
    elif isinstance(teetime, pd.Timestamp):
        parsed = teetime.to_pydatetime()
    else:
        s = str(teetime).strip()
        if not s:
            return 0.0
        for fmt in ["%Y-%m-%d %H:%M", "%I:%M%p", "%m/%d/%Y %H:%M"]:
            try:
                parsed = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return 0.0

    # Convert to decimal hour relative to 6 AM
    tee_decimal = parsed.hour + parsed.minute / 60.0
    block_start = tee_decimal - 6  # Relative to wind_data index 0 = 6 AM
    block_end = block_start + 5.0  # 5-hour window

    # Minute-level interpolation
    sample_points = np.arange(block_start, block_end, 1 / 60.0)
    wind_samples = np.interp(sample_points, np.arange(len(wind_data)), wind_data)

    return float(np.mean(wind_samples))


def compute_wind_factor(event_ids, wind_override, baseline_wind):
    """
    Compute the wind adjustment factor (strokes per MPH) for this course.
    
    Source: rd_1_sd_multicourse_sim.py lines 280-295
    
    Uses historical wind effect data from wind_test.csv blended with baseline.
    Override takes precedence if non-zero.
    
    Args:
        event_ids: List of event IDs for this tournament
        wind_override: Manual override value (0 = use computed)
        baseline_wind: Default wind effect factor
    
    Returns:
        Float: wind calculation factor (strokes of SG impact per MPH)
    """
    if wind_override != 0:
        print(f"Wind effect per MPH (override): {wind_override}")
        return wind_override

    try:
        wind_test_df = pd.read_csv("wind_test.csv")
        first_event_id = str(event_ids[0]).strip()

        filtered = wind_test_df[
            wind_test_df["event_ids"].apply(
                lambda x: first_event_id in map(str.strip, str(x).split(","))
            )
        ]

        if filtered.empty:
            course_wind_effect = 0.08
        else:
            course_wind_effect = filtered["wind_effect_adj_score"].iloc[-1]
    except FileNotFoundError:
        print("Warning: wind_test.csv not found, using default wind effect 0.08")
        course_wind_effect = 0.08

    wind_calculation = course_wind_effect * 0.4 + baseline_wind * 0.6
    print(f"Wind effect per MPH: {wind_calculation:.4f}")
    return wind_calculation


# --------------------------------------------------------------------------
# Name Cleaning
# --------------------------------------------------------------------------

def clean_names(df):
    """Standardize player names to lowercase with replacement mapping."""
    df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
    df["player_name"] = df["player_name"].replace(name_replacements)
    return df
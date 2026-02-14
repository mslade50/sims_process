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

    # Extract round number from teetime_col (e.g. "r2_teetime" → 2)
    round_num = int(teetime_col.replace("r", "").replace("_teetime", ""))

    # The API returns a nested 'teetimes' list instead of flat r{N}_teetime columns.
    # Parse it to extract the requested round's tee time and course code.
    if "teetimes" in df.columns:
        def _extract_teetime(teetimes):
            if not isinstance(teetimes, list):
                return pd.Series({teetime_col: None, "course": None})
            for entry in teetimes:
                if entry.get("round_num") == round_num:
                    return pd.Series({teetime_col: entry.get("teetime"), "course": entry.get("course_code")})
            return pd.Series({teetime_col: None, "course": None})

        parsed = df["teetimes"].apply(_extract_teetime)
        df[teetime_col] = parsed[teetime_col]
        df["course"] = parsed["course"]

    # Build list of columns to keep
    keep = ["player_name"]
    if teetime_col in df.columns:
        keep.append(teetime_col)
    if include_course and "course" in df.columns:
        keep.append("course")

    df = df[[c for c in keep if c in df.columns]].copy()
    df["player_name"] = df["player_name"].str.lower().replace(name_replacements)

    # If tee time column is missing or all empty (API returns structure before
    # tee times are set, e.g. R3 before cut), fill with 10:00 AM default.
    # This gives every player identical wind/dew, effectively zeroing out
    # the player-vs-field weather differential (skill-only differentiation).
    if teetime_col not in df.columns or df[teetime_col].isna().all():
        today_str = datetime.now().strftime("%Y-%m-%d")
        default_teetime = f"{today_str} 10:00"
        df[teetime_col] = default_teetime
        print(f"  ⚠️  {teetime_col} unavailable — defaulting to {default_teetime}")

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


# --------------------------------------------------------------------------
# Historical Round-Level SG Data
# --------------------------------------------------------------------------

def fetch_historical_rounds(event_id, year=None, api_key=None):
    """
    Fetch round-level SG category data from DataGolf.

    Returns long-format DataFrame:
        player_name, dg_id, round_num, sg_ott, sg_app, sg_arg, sg_putt, sg_total
    """
    import os
    if api_key is None:
        api_key = os.getenv("DATAGOLF_API_KEY")
    if year is None:
        from datetime import datetime as _dt
        year = _dt.now().year

    params = {
        "tour": "pga",
        "event_id": str(event_id),
        "year": year,
        "file_format": "json",
        "key": api_key,
    }

    try:
        resp = requests.get(
            f"{DATAGOLF_BASE}/historical-raw-data/rounds",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching historical rounds: {e}")
        return pd.DataFrame()

    if not data:
        print(f"  No historical round data for event {event_id}")
        return pd.DataFrame()

    # Handle nested structure
    if isinstance(data, dict) and "scores" in data:
        scores_data = data["scores"]
        print(f"  Event: {data.get('event_name', 'Unknown')}")
    else:
        scores_data = data

    if not scores_data:
        return pd.DataFrame()

    # Debug: print keys from first player's first round dict
    first_player = scores_data[0] if scores_data else {}
    for rnd_key in ["round_1", "round_2", "round_3", "round_4"]:
        rnd_dict = first_player.get(rnd_key)
        if isinstance(rnd_dict, dict):
            print(f"  [debug] {rnd_key} keys: {list(rnd_dict.keys())}")
            break

    SG_FIELDS = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]

    rows = []
    for player in scores_data:
        player_name = player.get("player_name", "")
        dg_id = player.get("dg_id", None)

        for rnd_num in range(1, 5):
            rnd_key = f"round_{rnd_num}"
            rnd_dict = player.get(rnd_key)
            if not isinstance(rnd_dict, dict):
                continue

            row = {
                "player_name": player_name,
                "dg_id": dg_id,
                "round_num": rnd_num,
            }
            for sg in SG_FIELDS:
                val = rnd_dict.get(sg)
                row[sg] = float(val) if val is not None else None

            rows.append(row)

    if not rows:
        print("  No round-level data extracted")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = clean_names(df)
    print(f"  Fetched {len(df)} player-round rows ({df['player_name'].nunique()} players)")
    return df
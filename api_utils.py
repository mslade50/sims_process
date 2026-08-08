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

import os
import sqlite3
import unicodedata
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from sim_inputs import name_replacements
from img_shot_local import (
    read_player_rounds as read_local_img_player_rounds,
    read_player_shots as read_local_img_player_shots,
    resolve_db_path as resolve_img_shot_db_path,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DATAGOLF_BASE = "https://feeds.datagolf.com"
IMG_SHOT_DEFAULT_URL = "https://golf-shot-ingest.mckinleyslade.workers.dev"

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

# --------------------------------------------------------------------------
# IMG shot archive API
# --------------------------------------------------------------------------

def _img_shot_api_config(base_url=None, read_token=None):
    """Resolve the read-only IMG archive endpoint without exposing write auth."""
    url = (
        base_url
        or os.getenv("IMG_SHOT_API_URL")
        or os.getenv("IMG_SHOT_INGEST_URL")
        or IMG_SHOT_DEFAULT_URL
    ).strip().rstrip("/")
    if url.endswith("/v1/ingest"):
        url = url[:-len("/v1/ingest")]
    token = (read_token or os.getenv("IMG_SHOT_READ_TOKEN") or "").strip()
    return url, token


def _fetch_img_archive_json(path, params, base_url=None, read_token=None, timeout=30):
    url, token = _img_shot_api_config(base_url, read_token)
    if not token:
        print("  IMG shot archive disabled: set IMG_SHOT_READ_TOKEN")
        return None
    try:
        response = requests.get(
            f"{url}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        print(f"  IMG shot archive request failed: {exc}")
        return None
    if response.status_code != 200:
        print(f"  IMG shot archive error: HTTP {response.status_code}")
        return None
    try:
        payload = response.json()
    except ValueError:
        print("  IMG shot archive returned invalid JSON")
        return None
    if not payload.get("ok") or not isinstance(payload.get("rows"), list):
        print(f"  IMG shot archive returned an invalid payload: {payload.get('error', 'missing rows')}")
        return None
    return payload


def _canonical_img_name(value):
    """Match PGA first-last names to the sims' lowercase last-first contract."""
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", str(value or ""))
        if not unicodedata.combining(character)
    )
    text = " ".join(text.lower().strip().split())
    if not text:
        return text
    if "," in text:
        surname, given = text.split(",", 1)
        canonical = f"{surname.strip()}, {' '.join(given.split())}"
    else:
        parts = text.split()
        canonical = text if len(parts) < 2 else f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name_replacements.get(canonical, canonical)


def _use_local_img_archive(base_url=None, read_token=None, db_path=None):
    source = os.getenv("IMG_SHOT_SOURCE", "auto").strip().lower()
    if source == "api" or base_url is not None or read_token is not None:
        return False
    return resolve_img_shot_db_path(db_path).is_file()


def _img_rows_frame(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "player_name" in frame.columns:
        frame["player_name"] = frame["player_name"].map(_canonical_img_name)
    numeric = [
        "tournament_id", "round_no", "hole_no", "shot_no", "round_strokes",
        "holes_with_data", "holes_completed", "shot_events", "numbered_shots",
        "tee_shots", "green_origin_shots", "penalty_events", "ball_drop_events",
        "eagles_or_better", "birdies", "pars", "bogeys",
        "double_bogeys_or_worse", "sequence_gap_holes",
        "shot_distance_m", "shot_distance_yd", "distance_to_pin_m",
        "distance_to_pin_yd", "distance_to_pin_ft",
        "avg_tee_shot_distance_m", "avg_tee_shot_distance_yd",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fetch_img_player_rounds(event_id, round_num, base_url=None, read_token=None,
                            timeout=30, db_path=None, tour="pga", season=None,
                            event_key=None):
    """Fetch current player-round SG locally, with the Worker as a fallback."""
    if _use_local_img_archive(base_url, read_token, db_path):
        try:
            resolved, rows = read_local_img_player_rounds(
                event_id, round_num, db_path=db_path, tour=tour,
                season=season or datetime.now().year,
                event_key=event_key or os.getenv("IMG_SHOT_EVENT_KEY"),
            )
            frame = _img_rows_frame(rows)
            frame.attrs.update(event_key=resolved, source="local_sqlite")
            return frame
        except (FileNotFoundError, LookupError, OSError, sqlite3.Error) as exc:
            print(f"  IMG local archive unavailable: {exc}")
            if os.getenv("IMG_SHOT_SOURCE", "auto").strip().lower() == "local":
                return None
    payload = _fetch_img_archive_json(
        "/v1/player-rounds",
        {"event_id": int(event_id), "round": int(round_num)},
        base_url=base_url,
        read_token=read_token,
        timeout=timeout,
    )
    return None if payload is None else _img_rows_frame(payload["rows"])


def fetch_img_player_shots(event_id, round_num, base_url=None, read_token=None,
                           page_size=1000, max_pages=100, timeout=30,
                           db_path=None, tour="pga", season=None, event_key=None):
    """Fetch current shot-level SG locally, with Worker pagination as fallback."""
    if _use_local_img_archive(base_url, read_token, db_path):
        try:
            resolved, rows = read_local_img_player_shots(
                event_id, round_num, db_path=db_path, tour=tour,
                season=season or datetime.now().year,
                event_key=event_key or os.getenv("IMG_SHOT_EVENT_KEY"),
            )
            frame = _img_rows_frame(rows)
            frame.attrs.update(event_key=resolved, source="local_sqlite")
            return frame
        except (FileNotFoundError, LookupError, OSError, sqlite3.Error) as exc:
            print(f"  IMG local archive unavailable: {exc}")
            if os.getenv("IMG_SHOT_SOURCE", "auto").strip().lower() == "local":
                return None
    rows = []
    cursor = None
    for _ in range(max_pages):
        params = {"event_id": int(event_id), "round": int(round_num),
                  "limit": min(max(1, int(page_size)), 1000)}
        if cursor:
            params["cursor"] = cursor
        payload = _fetch_img_archive_json(
            "/v1/player-shots", params, base_url=base_url,
            read_token=read_token, timeout=timeout,
        )
        if payload is None:
            return None
        rows.extend(payload["rows"])
        cursor = payload.get("next_cursor")
        if not cursor:
            return _img_rows_frame(rows)
    print(f"  IMG shot archive exceeded {max_pages} pages; refusing partial data")
    return None


def fetch_field_updates(api_key, teetime_col="r1_teetime", include_course=False,
                        fill_missing_teetimes=True):
    """
    Fetch field updates (tee times, course assignments) from DataGolf.

    Source: Originally duplicated across all 4 live_stats files.

    Args:
        api_key: DataGolf API key
        teetime_col: Which round's tee time column to extract (e.g. 'r1_teetime')
        include_course: If True, also extracts the 'course' column (for multi-course events)
        fill_missing_teetimes: If False, skip the 10:00 AM default fill so callers
            can distinguish "tee times not posted yet" (all NaN) from real data.
            Needed by cut filtering, where a NaN tee time means the player is out.

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
    if fill_missing_teetimes and (teetime_col not in df.columns or df[teetime_col].isna().all()):
        today_str = datetime.now().strftime("%Y-%m-%d")
        default_teetime = f"{today_str} 10:00"
        df[teetime_col] = default_teetime
        print(f"  WARNING: {teetime_col} unavailable — defaulting to {default_teetime}")

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
        # Lives in permanent_data/ (survives weekly cleanup); root checked
        # second for backward compatibility.
        _wt_path = os.path.join("permanent_data", "wind_test.csv")
        if not os.path.exists(_wt_path):
            _wt_path = "wind_test.csv"
        wind_test_df = pd.read_csv(_wt_path)
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


def compute_dew_factor(course_id, baseline_dew_factor):
    """
    Per-course dew coefficient (strokes per degF of dewpoint deviation).

    Analog of compute_wind_factor: courses with enough weather history get
    an empirical-Bayes-shrunk course-specific slope from
    permanent_data/dew_test.csv (built by archive/dew_course_effects.py,
    clamped to [-0.06, 0] — tropical venues with a pinned dew range land
    at 0, i.e. dew adjustment off). Courses not in the CSV fall back to
    the sim_inputs baseline blend.

    Args:
        course_id: Course ID (same ID space as dg_historical course_num)
        baseline_dew_factor: Fallback coefficient (sim_inputs blend)

    Returns:
        Float: dew coefficient (negative = humid plays easier)
    """
    try:
        # Lives in permanent_data/ (survives weekly cleanup)
        _dt_path = os.path.join("permanent_data", "dew_test.csv")
        if not os.path.exists(_dt_path):
            print(f"Dew factor: {baseline_dew_factor:.4f} (baseline — dew_test.csv not found)")
            return baseline_dew_factor

        dew_test_df = pd.read_csv(_dt_path)
        row = dew_test_df[dew_test_df["course_num"] == int(course_id)]
        if row.empty:
            print(f"Dew factor: {baseline_dew_factor:.4f} (baseline — course {course_id} not in dew_test.csv)")
            return baseline_dew_factor

        dew_coef = float(row["dew_coef"].iloc[-1])
        print(f"Dew factor: {dew_coef:.4f} (course {course_id} — raw {float(row['raw_slope'].iloc[-1]):+.4f} "
              f"over {int(row['n_round_days'].iloc[-1])} round-days, "
              f"shrink_wt {float(row['shrink_wt'].iloc[-1]):.2f}; baseline {baseline_dew_factor:.4f})")
        return dew_coef
    except Exception as e:
        print(f"Dew factor: {baseline_dew_factor:.4f} (baseline — dew_test lookup failed: {e})")
        return baseline_dew_factor


# --------------------------------------------------------------------------
# Bayesian Wind Blending (climatology prior + forecast)
# --------------------------------------------------------------------------

# NWP forecast skill decays roughly linearly toward climatology at ~10-12 days.
# climo_weight = clamp(lead_days / 12, 0.05, 0.50)
# At 0 days: 5% climo.  At 3 days: 25%.  At 6 days: 50% (cap).
MIN_CLIMO_WT = 0.05
MAX_CLIMO_WT = 0.50
SKILL_HORIZON_DAYS = 12.0  # forecast = climatology beyond this


def climo_weight_for_lead(lead_days):
    """Compute climatology blend weight from lead time in days."""
    return max(MIN_CLIMO_WT, min(MAX_CLIMO_WT, lead_days / SKILL_HORIZON_DAYS))


def fetch_historical_hourly_wind(lat, lon, month, start_year=2019, end_year=2025):
    """
    Query Open-Meteo archive for hourly wind (6 AM–8 PM) across multiple years
    for a given month. Returns 15-element list (index 0 = 6 AM, 14 = 8 PM)
    or None on failure.
    """
    import os
    from requests.exceptions import ConnectTimeout, ConnectionError as ReqConnectionError
    all_frames = []
    for year in range(start_year, end_year + 1):
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={year}-{month:02d}-01"
            f"&end_date={year}-{month:02d}-28"
            f"&hourly=wind_speed_10m"
            f"&windspeed_unit=mph&timezone=auto"
        )
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if "hourly" not in data or "wind_speed_10m" not in data["hourly"]:
                continue
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["hourly"]["time"]),
                "wind": data["hourly"]["wind_speed_10m"],
            })
            df["hour"] = df["datetime"].dt.hour
            df = df[(df["hour"] >= 6) & (df["hour"] <= 20)]
            all_frames.append(df)
        except (ConnectTimeout, ReqConnectionError) as e:
            print(f"  [climo wind] {year} connect failed — host unreachable, skipping remaining years")
            break
        except Exception as e:
            print(f"  [climo wind] {year} fetch failed: {e}")
            continue

    if not all_frames:
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    hourly_avg = combined.groupby("hour")["wind"].mean()
    return [round(float(hourly_avg.get(h, 0.0)), 1) for h in range(6, 21)]


def blend_wind_with_climo(forecast_array, climo_array, lead_days=None, round_date=None):
    """
    Bayesian blend of forecast wind array with climatological prior.

    Climo weight scales with lead time:  lead_days / 12, clamped to [0.05, 0.50].
      ~12 hrs out -> ~5% climo   (fresh forecast, trust it)
      ~3 days out -> 25% climo   (typical R1 from Tuesday)
      ~6 days out -> 50% climo   (R4 from early week, cap)

    Args:
        forecast_array: list[float] — 15 hourly values (6 AM–8 PM)
        climo_array:    list[float] — 15 hourly base-rate values
        lead_days:      float — explicit days until the round (preferred)
        round_date:     datetime — round date; lead = round_date - now()

    Returns:
        (blended_array, w_climo) — blended 15-element list and the weight used
    """
    if not forecast_array or not climo_array:
        return forecast_array, 0.0

    # Determine lead time
    if lead_days is None and round_date is not None:
        delta = round_date - datetime.now()
        lead_days = max(delta.total_seconds() / 86400.0, 0.0)
    if lead_days is None:
        lead_days = 3.0  # safe fallback

    w_climo = climo_weight_for_lead(lead_days)
    w_fcst = 1.0 - w_climo
    n = min(len(forecast_array), len(climo_array))
    blended = [
        round(w_fcst * forecast_array[i] + w_climo * climo_array[i], 1)
        for i in range(n)
    ]
    if len(forecast_array) > n:
        blended.extend(forecast_array[n:])
    return blended, w_climo


# Models for the multi-model wind blend. AIFS is the model behind Windy's
# "AI" layer; AIGFS is NOAA's AI model. Both run 6-hourly natively and are
# interpolated to hourly by Open-Meteo. gfs_graphcast025 excluded (returns
# all-null as of July 2026).
AI_WIND_MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "ncep_aigfs025"]


def fetch_multimodel_wind(lat, lon, start_date, end_date, models=None,
                          timezone="auto"):
    """
    Fetch hourly wind (mph) from multiple forecast models and blend them.

    Multi-model mean of ECMWF IFS + ECMWF AIFS + NOAA AIGFS. A multi-model
    mean is robustly more accurate than any single member; rows where a
    model is missing use the mean of the models present.

    Args:
        lat, lon: Course coordinates
        start_date, end_date: YYYY-MM-DD strings (inclusive)
        models: Optional list of Open-Meteo model names (default AI_WIND_MODELS)
        timezone: Open-Meteo timezone param. Pass the SAME timezone as the
            forecast call whose timestamps you join against. ``auto`` resolves
            the course-local timezone from latitude/longitude.

    Returns:
        DataFrame with 'time' (datetime), one mph column per model, and
        'wind_blend' (row-wise mean, mph) — or None on failure.
    """
    models = models or AI_WIND_MODELS
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "wind_speed_10m",
                "models": ",".join(models),
                "windspeed_unit": "mph", "timezone": timezone,
                "start_date": start_date, "end_date": end_date,
            }, timeout=20)
        data = resp.json()
        if "hourly" not in data:
            print(f"  [multimodel wind] No hourly data in response: {data.get('reason', data)}")
            return None
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        wind_cols = [c for c in df.columns if c.startswith("wind_speed_10m")]
        if not wind_cols:
            return None
        df["wind_blend"] = df[wind_cols].mean(axis=1, skipna=True)
        return df
    except Exception as e:
        print(f"  [multimodel wind] Fetch failed: {e}")
        return None


def fetch_event_weather_forecast(lat, lon, round_dates, timezone="auto"):
    """
    Fetch hourly wind and dewpoint arrays for each tournament round.

    Wind is the per-hour mean of ECMWF IFS + ECMWF AIFS + NOAA AIGFS when
    available, with Open-Meteo best_match as a per-hour fallback. Dewpoint uses
    best_match. Arrays cover 6 AM through 8 PM (15 values), matching the Sheet
    and calculate_average_wind contracts.

    Args:
        lat, lon: Course coordinates.
        round_dates: Four datetime/date objects ordered R1 through R4.
        timezone: Timezone used for both forecast calls. ``auto`` resolves the
            course-local IANA timezone from latitude/longitude. It must be
            identical for both calls so model timestamps align.

    Returns:
        Dict with:
          rounds: {round_num: {"wind": [...], "dew": [...]}}
          timezone: resolved course-local IANA timezone
          provider: descriptive provider string
          ai_hours: number of hours supplied by the multi-model blend

        Returns None if the base forecast is unavailable or malformed.
    """
    if not round_dates or len(round_dates) < 4:
        print("  [event weather] Four round dates are required")
        return None

    date_strings = [
        d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        for d in round_dates[:4]
    ]
    start_date, end_date = date_strings[0], date_strings[3]

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "dewpoint_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "timezone": timezone,
                "start_date": start_date,
                "end_date": end_date,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly") or {}
        timestamps = hourly.get("time") or []
        dewpoints = hourly.get("dewpoint_2m") or []
        best_wind = hourly.get("wind_speed_10m") or []
        if not timestamps or not dewpoints or not best_wind:
            print(f"  [event weather] Missing hourly data: {data.get('reason', data)}")
            return None
    except Exception as e:
        print(f"  [event weather] Base forecast failed: {e}")
        return None

    multimodel = fetch_multimodel_wind(
        lat, lon, start_date, end_date, timezone=timezone
    )
    ai_by_time = {}
    if multimodel is not None and not multimodel.empty:
        for _, row in multimodel.iterrows():
            value = row.get("wind_blend")
            if pd.notna(value):
                key = pd.Timestamp(row["time"]).strftime("%Y-%m-%dT%H:%M")
                ai_by_time[key] = float(value)

    rounds = {rnd: {"wind": [], "dew": []} for rnd in range(1, 5)}
    ai_hours_by_round = {rnd: 0 for rnd in range(1, 5)}
    for time_str, dewpoint, fallback_wind in zip(
            timestamps, dewpoints, best_wind):
        try:
            dt = pd.Timestamp(time_str)
        except Exception:
            continue
        if not 6 <= dt.hour <= 20:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in date_strings:
            continue
        if dewpoint is None and fallback_wind is None:
            continue

        rnd = date_strings.index(date_str) + 1
        key = dt.strftime("%Y-%m-%dT%H:%M")
        wind = ai_by_time.get(key, fallback_wind)
        if key in ai_by_time:
            ai_hours_by_round[rnd] += 1
        if wind is not None and pd.notna(wind):
            rounds[rnd]["wind"].append(round(float(wind), 1))
        if dewpoint is not None and pd.notna(dewpoint):
            rounds[rnd]["dew"].append(round(float(dewpoint), 1))

    for values in rounds.values():
        values["wind"] = values["wind"][:15]
        values["dew"] = values["dew"][:15]

    return {
        "rounds": rounds,
        "timezone": data.get("timezone") or timezone,
        "provider": (
            "open-meteo:ecmwf_ifs+aifs+aigfs (best_match fallback)"
            if ai_by_time else "open-meteo:best_match (AI blend unavailable)"
        ),
        "ai_hours": len(ai_by_time),
        "ai_hours_by_round": ai_hours_by_round,
    }


def fetch_realized_wind(lat, lon, date_str):
    """
    Fetch observed average wind (mph) for a course on a given date from
    Open-Meteo archive. Averages hourly wind_speed_10m from 6 AM to 8 PM
    local time (standard golf window).

    Args:
        lat: Course latitude
        lon: Course longitude
        date_str: Date string in YYYY-MM-DD format

    Returns:
        float (avg wind mph) or None on failure
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=wind_speed_10m"
        f"&windspeed_unit=mph&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if "hourly" not in data or "wind_speed_10m" not in data["hourly"]:
            print(f"  [realized wind] No hourly data for {date_str}")
            return None
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "wind": data["hourly"]["wind_speed_10m"],
        })
        df["hour"] = df["datetime"].dt.hour
        golf_window = df[(df["hour"] >= 6) & (df["hour"] <= 20)]
        if golf_window.empty:
            return None
        avg = float(golf_window["wind"].mean())
        print(f"  [realized wind] {date_str}: {avg:.1f} mph (6AM-8PM avg)")
        return round(avg, 1)
    except Exception as e:
        print(f"  [realized wind] Fetch failed for {date_str}: {e}")
        return None


def get_round_dates():
    """
    Return [R1, R2, R3, R4] datetimes for the current week's tournament.
    PGA events are always Thursday–Sunday.  Rounds start ~7 AM local.
    """
    from datetime import timedelta
    today = datetime.now()
    # Thursday = weekday 3.  Find this week's Thursday.
    days_since_thu = (today.weekday() - 3) % 7
    thursday = (today - timedelta(days=days_since_thu)).replace(hour=7, minute=0, second=0, microsecond=0)
    # If we're past Sunday, jump to next week's Thursday
    sunday = thursday + timedelta(days=3)
    if today > sunday.replace(hour=23):
        thursday += timedelta(days=7)
    return [thursday + timedelta(days=i) for i in range(4)]


def sim_round_from_dow(now_et=None) -> int:
    """Round being priced, derived PURELY from the America/New_York weekday — the
    sims mirror of the scraper's board/build.py round_for_now(). Thu->2, Fri->3,
    Sat->4, Sun->4, Mon-Wed->1. Always 1..4, never None.

    Used as a warning-only divergence check against the Google Sheet round (the
    sheet stays the single human-controlled source so a past round can still be
    re-priced manually) and by push_odds_screen to scope the odds screen to the
    same round the board stamps. Must anchor on ET to match the board exactly."""
    from zoneinfo import ZoneInfo
    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    return {0: 1, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}[now_et.weekday()]


# --------------------------------------------------------------------------
# Name Cleaning
# --------------------------------------------------------------------------

def clean_names(df):
    """Standardize player names to lowercase with replacement mapping."""
    df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
    df["player_name"] = df["player_name"].replace(name_replacements)
    return df


def detect_completed_round(api_key):
    """
    Auto-detect which round just completed by checking live stats availability.

    Probes DataGolf live-tournament-stats for rounds 1-4. If round N returns
    >10 players with stats, that round is considered complete.

    Returns:
        int: 0 (pre-event/no data), 1-4 (last completed round)
    """
    last_complete = 0
    for round_num in range(1, 5):
        df = fetch_live_stats(round_num, api_key)
        if df is not None and len(df) > 10:
            last_complete = round_num
        else:
            break
    return last_complete


def fetch_player_decompositions(api_key):
    """
    Fetch player skill decompositions from DataGolf.

    Returns DataFrame with columns: player_name, dg_final_pred, dg_baseline_pred.
    dg_final_pred is strokes gained per round vs PGA Tour average.
    Returns empty DataFrame on failure.
    """
    params = {
        "tour": "pga",
        "file_format": "json",
        "key": api_key,
    }

    try:
        resp = requests.get(
            f"{DATAGOLF_BASE}/preds/player-decompositions",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning: Failed to fetch player decompositions: {e}")
        return pd.DataFrame()

    players = data.get("players", [])
    if not players:
        print("  Warning: No players in decomposition response")
        return pd.DataFrame()

    df = pd.DataFrame(players)

    # Extract the columns we need
    keep = {}
    if "player_name" in df.columns:
        keep["player_name"] = df["player_name"]
    else:
        print("  Warning: No player_name in decomposition data")
        return pd.DataFrame()

    if "final_pred" in df.columns:
        keep["dg_final_pred"] = pd.to_numeric(df["final_pred"], errors="coerce")
    if "baseline_pred" in df.columns:
        keep["dg_baseline_pred"] = pd.to_numeric(df["baseline_pred"], errors="coerce")

    result = pd.DataFrame(keep)
    result = clean_names(result)

    if "dg_final_pred" not in result.columns:
        print("  Warning: No final_pred in decomposition data")
        return pd.DataFrame()

    print(f"  Fetched {len(result)} player decompositions from DataGolf")
    return result


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


# --------------------------------------------------------------------------
# Historical Betting Odds (opening + closing lines, with outcomes)
# --------------------------------------------------------------------------

def fetch_historical_matchup_odds(year, book, event_id="all",
                                  market="tournament_matchups", api_key=None):
    """
    Fetch historical matchup odds (opening + closing lines + settled outcomes)
    from DataGolf's historical-odds/matchups feed.

    betcris and pinnacle do NOT fully overlap on matchup offerings, so callers
    should pull both books and union on (event_id, dg_id pair) coverage.

    Returns long-format DataFrame, one row per matchup:
        event_id, event_name, season, book, bet_type, open_time, close_time,
        p1_dg_id, p1_player_name, p1_open, p1_close, p1_outcome,
        p2_dg_id, p2_player_name, p2_open, p2_close, p2_outcome
    Odds columns are American (float). Outcomes: 1.0 win, 0.0 loss (NaN if void).
    """
    if api_key is None:
        api_key = os.getenv("DATAGOLF_API_KEY")

    params = {
        "tour": "pga",
        "event_id": str(event_id),
        "year": year,
        "market": market,
        "book": book,
        "odds_format": "american",
        "file_format": "json",
        "key": api_key,
    }

    # DataGolf allows 45 req/min across ALL endpoints; exceeding it triggers a
    # 5-min suspension returned as 403. Do NOT retry into it — every request made
    # while suspended resets the clock. Return a rate-limit signal so the caller
    # can stop immediately and resume after a clean cooldown. Pace under 45/min.
    try:
        resp = requests.get(
            f"{DATAGOLF_BASE}/historical-odds/matchups",
            params=params,
            timeout=45,
        )
        if resp.status_code in (403, 429):
            print(f"  Rate-limited ({resp.status_code}) on {book} event {event_id}.")
            return "RATE_LIMITED"
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching historical matchup odds ({book}, event {event_id}): {e}")
        return pd.DataFrame()

    # event_id="all" returns a list of event dicts; a single event returns one dict.
    events = data if isinstance(data, list) else [data]
    rows = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Book doesn't archive this market — API returns a message string
        if not isinstance(ev.get("odds"), list):
            continue
        meta = {
            "event_id": ev.get("event_id"),
            "event_name": ev.get("event_name"),
            "season": ev.get("season") or ev.get("year"),
            "book": ev.get("book", book),
        }
        for o in ev.get("odds", []) or []:
            row = dict(meta)
            row.update({
                "bet_type": o.get("bet_type"),
                "open_time": o.get("open_time"),
                "close_time": o.get("close_time"),
                "tie_rule": o.get("tie_rule"),
            })
            for side in ("p1", "p2"):
                row[f"{side}_dg_id"] = o.get(f"{side}_dg_id")
                row[f"{side}_player_name"] = o.get(f"{side}_player_name")
                row[f"{side}_open"] = pd.to_numeric(o.get(f"{side}_open"), errors="coerce")
                row[f"{side}_close"] = pd.to_numeric(o.get(f"{side}_close"), errors="coerce")
                row[f"{side}_outcome"] = o.get(f"{side}_outcome")
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  Fetched {len(df)} {market} rows from {book} "
              f"({df['event_id'].nunique()} events)")
    return df


def fetch_historical_outright_odds(year, book, event_id, market="win", api_key=None):
    """
    Fetch historical outright/finish-position odds (opening + closing lines)
    from DataGolf's historical-odds/outrights feed.

    market: win, top_5, top_10, top_20 (also mc, make_cut, frl upstream).
    Not every book archives every market (e.g. pinnacle only archives win) —
    the API returns a message string instead of a list for missing markets,
    which is handled here as an empty DataFrame.

    Returns long-format DataFrame, one row per player:
        event_id, event_name, season, book, market, dg_id, player_name,
        open_time, close_time, open_odds, close_odds, outcome, bet_outcome_numeric
    Odds columns are American (float).
    Returns "RATE_LIMITED" (str) on a 403/429 so callers can stop immediately
    (same contract as fetch_historical_matchup_odds).
    """
    if api_key is None:
        api_key = os.getenv("DATAGOLF_API_KEY")

    params = {
        "tour": "pga",
        "event_id": str(event_id),
        "year": year,
        "market": market,
        "book": book,
        "odds_format": "american",
        "file_format": "json",
        "key": api_key,
    }

    try:
        resp = requests.get(
            f"{DATAGOLF_BASE}/historical-odds/outrights",
            params=params,
            timeout=45,
        )
        if resp.status_code in (403, 429):
            print(f"  Rate-limited ({resp.status_code}) on {book} {market} event {event_id}.")
            return "RATE_LIMITED"
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching historical outright odds ({book}, {market}, event {event_id}): {e}")
        return pd.DataFrame()

    events = data if isinstance(data, list) else [data]
    rows = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        odds = ev.get("odds")
        if not isinstance(odds, list):
            # Book doesn't archive this market — API returns a message string
            continue
        meta = {
            "event_id": ev.get("event_id"),
            "event_name": ev.get("event_name"),
            "season": ev.get("season") or ev.get("year"),
            "book": ev.get("book", book),
            "market": ev.get("market", market),
        }
        for o in odds:
            if not isinstance(o, dict):
                continue
            row = dict(meta)
            row.update({
                "dg_id": o.get("dg_id"),
                "player_name": o.get("player_name"),
                "open_time": o.get("open_time"),
                "close_time": o.get("close_time"),
                "open_odds": pd.to_numeric(o.get("open_odds"), errors="coerce"),
                "close_odds": pd.to_numeric(o.get("close_odds"), errors="coerce"),
                "outcome": o.get("outcome"),
                "bet_outcome_numeric": o.get("bet_outcome_numeric"),
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  Fetched {len(df)} {market} outright rows from {book}")
    return df

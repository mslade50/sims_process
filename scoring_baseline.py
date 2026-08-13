"""
scoring_baseline.py — Automated scoring baseline computation.

Produces a weather-free, tour-average-field scoring baseline per round
by combining:
  - Historical scoring averages from dg_historical.db
  - Historical weather data (wind + dewpoint) from Visual Crossing API
  - Wind sensitivity coefficients from wind_test.csv
  - Field strength data from field_strength.db
  - Recency weighting across years

Usage:
    python scoring_baseline.py

Reads config from sim_inputs.py: event_ids, course_id, course_par,
start_yr, tourney, baseline_wind, dew_calculation.
"""

import os
import sys
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from api_utils import lookup_course_wind_effect

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration from sim_inputs
# ---------------------------------------------------------------------------
from sim_inputs import (
    event_ids,
    course_id,
    course_par,
    start_yr,
    tourney,
    baseline_wind,
    wind_override,
    CUT_LINE,
    STD_DEV,
)
try:
    from sim_inputs import lat_override, lon_override
except ImportError:
    lat_override = None
    lon_override = None
try:
    from sim_inputs import baseline_score_adj
except ImportError:
    baseline_score_adj = 0.0
try:
    from sim_inputs import tour_override
except ImportError:
    tour_override = 'pga'
try:
    from sim_inputs import historical_event_ids as _configured_historical_event_ids
except ImportError:
    _configured_historical_event_ids = []

# A populated historical_event_ids is a deliberate override (for example, using
# LIV history for a PGA event). In the normal case the course is the historical
# identity and event IDs are informational only: DataGolf reuses event IDs when
# playoff events change venues/names. Treating the current event ID as history
# silently mixed Liberty National/TPC Boston into the TPC Southwind baseline in
# 2026, which materially inflated the R1 estimate.
HISTORICAL_EVENT_FILTER_EXPLICIT = bool(_configured_historical_event_ids)
historical_event_ids = (
    list(_configured_historical_event_ids)
    if HISTORICAL_EVENT_FILTER_EXPLICIT
    else list(event_ids)
)
# Force EVENT-LEVEL (all venues) category variance instead of the course
# filter — for rotating-venue events at a course we have no data for.
try:
    from sim_inputs import variance_event_level_only
except ImportError:
    variance_event_level_only = False
# Hard-coded per-round field scoring averages (par-space absolute strokes)
# that override the computed baseline when set. dict {1:.., 2:.., 3:.., 4:..}.
try:
    from sim_inputs import expected_score_override
except ImportError:
    expected_score_override = None

# Paths
DG_HISTORICAL_DB = os.path.join(os.path.expanduser("~"), "OneDrive", "dg_historical.db")
FIELD_STRENGTH_DB = os.path.join(os.path.expanduser("~"), "OneDrive", "field_strength.db")
WIND_DATA_DB = os.path.join(os.path.dirname(__file__), "wind_data.db")
WIND_TEST_CSV = os.path.join(os.path.dirname(__file__), "permanent_data", "wind_test.csv")
COURSE_COORDS_CSV = os.path.join(os.path.dirname(__file__), "permanent_data", "course_coordinates.csv")

VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")

# Dewpoint discount factor (40% confidence)
DEW_DISCOUNT = 0.40


def _history_where(event_id_list, min_year, course_num=None,
                   restrict_event_ids=None):
    """Return a safe SQL WHERE fragment and parameters for venue history.

    Course is the primary identity. Event IDs are added only when the weekly
    config explicitly supplies ``historical_event_ids``. Callers that request
    event-level analysis by passing ``course_num=None`` always retain the event
    filter.
    """
    if restrict_event_ids is None:
        restrict_event_ids = HISTORICAL_EVENT_FILTER_EXPLICIT

    clauses = ["year >= ?", "tour = ?"]
    params = [min_year, tour_override]
    if course_num is not None:
        clauses.insert(0, "course_num = ?")
        params.insert(0, course_num)
    if course_num is None or restrict_event_ids:
        ids = list(event_id_list or [])
        if not ids:
            raise RuntimeError("Historical event filter requested with no event IDs")
        placeholders = ",".join("?" * len(ids))
        clauses.insert(0, f"event_id IN ({placeholders})")
        params = ids + params
    return " AND ".join(clauses), params


def validate_history_scope(event_id_list, min_year, course_num=None,
                           restrict_event_ids=None):
    """Fail loudly before a scoring baseline can combine ambiguous history."""
    course_num = course_id if course_num is None else course_num
    where, params = _history_where(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    conn = sqlite3.connect(DG_HISTORICAL_DB)
    events = pd.read_sql_query(
        f"""
        SELECT year, event_id, event_name, course_num,
               MIN(round_date) AS event_start
        FROM player_rounds
        WHERE {where}
        GROUP BY year, event_id, event_name, course_num
        ORDER BY year, event_start
        """,
        conn,
        params=params,
    )
    conn.close()

    if events.empty:
        is_explicit = (
            restrict_event_ids
            if restrict_event_ids is not None
            else HISTORICAL_EVENT_FILTER_EXPLICIT
        )
        extra = (f" and explicit event IDs {list(event_id_list)}"
                 if is_explicit else "")
        raise RuntimeError(
            f"No {tour_override} history found for course_num={course_num} "
            f"since {min_year}{extra}. Refusing to build a fallback baseline."
        )

    wrong_courses = (
        set(pd.to_numeric(events["course_num"], errors="coerce"))
        - {float(course_num)}
    )
    if wrong_courses:
        raise RuntimeError(
            f"Mixed venues entered the history scope for course_num={course_num}: "
            f"{sorted(wrong_courses)}"
        )

    per_year = events.groupby("year").size()
    ambiguous_years = per_year[per_year > 1].index.tolist()
    if ambiguous_years:
        details = events[events["year"].isin(ambiguous_years)][
            ["year", "event_id", "event_name", "event_start"]
        ].to_dict("records")
        raise RuntimeError(
            "Multiple tournaments at the same course in the same year would be "
            f"combined: {details}. Set historical_event_ids explicitly to select "
            "the intended event history."
        )

    ids = sorted(set(int(x) for x in events["event_id"]))
    years = sorted(set(int(x) for x in events["year"]))
    print(
        f"  History scope verified: course_num={course_num}, "
        f"years={years}, event_ids={ids}"
    )
    return events


# ===========================================================================
# Step 2: Course Coordinate Lookup
# ===========================================================================

def get_course_coordinates(cid):
    """Look up lat/lon for a course_id from course_coordinates.csv.

    If sim_inputs.lat_override/lon_override are set, they take precedence —
    used for partner events at courses we have no historical data for but
    are reusing another course_id for variance profiles.
    """
    if lat_override is not None and lon_override is not None:
        print(f"  Using lat/lon OVERRIDE from sim_inputs: {lat_override}, {lon_override}")
        return float(lat_override), float(lon_override)

    if not os.path.exists(COURSE_COORDS_CSV):
        print(f"  WARNING: {COURSE_COORDS_CSV} not found")
        return None, None

    df = pd.read_csv(COURSE_COORDS_CSV)
    match = df[df["course_id"] == cid]
    if match.empty:
        print(f"  WARNING: course_id {cid} not found in course_coordinates.csv")
        return None, None

    row = match.iloc[0]
    return float(row["lat"]), float(row["lon"])


# ===========================================================================
# Step 3: Tournament Date Lookup
# ===========================================================================

def get_tournament_dates(event_id_list, min_year, course_num=None,
                         restrict_event_ids=None):
    """
    Query dg_historical.db for tournament round dates.

    Returns DataFrame: year, round_num, round_date, player_count
    """
    conn = sqlite3.connect(DG_HISTORICAL_DB)
    course_num = course_id if course_num is None else course_num
    where, params = _history_where(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    query = f"""
        SELECT year, round_num,
               MIN(round_date) AS round_date,
               COUNT(*) AS player_count
        FROM player_rounds
        WHERE {where}
        GROUP BY year, round_num
        ORDER BY year, round_num
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    df["round_date"] = pd.to_datetime(df["round_date"])
    return df


# ===========================================================================
# Step 4: Weather Data Fetch / Update
# ===========================================================================

def _ensure_wind_db_schema(conn):
    """Add new columns to wind_data table if they don't exist (backward compatible)."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wind_data (
            date TEXT,
            time TEXT,
            wind_speed REAL,
            event TEXT
        )
    """)
    conn.commit()

    # Check existing columns
    cursor.execute("PRAGMA table_info(wind_data)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    for col, dtype in [("event_id", "INTEGER"), ("dewpoint", "REAL"), ("temp", "REAL")]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE wind_data ADD COLUMN {col} {dtype}")
            print(f"  [schema] Added column '{col}' to wind_data")
    conn.commit()


def _wind_data_exists(conn, date_str, event_tag):
    """Check if we already have hourly data for this date+event."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM wind_data WHERE date = ? AND event = ?",
        (date_str, event_tag),
    )
    return cursor.fetchone()[0] > 0


def fetch_visual_crossing(lat, lon, start_date, end_date):
    """
    Fetch hourly weather from Visual Crossing API.

    Returns list of dicts: {date, time, wind_speed, dewpoint, temp}
    """
    if not VISUAL_CROSSING_KEY:
        print("  WARNING: No VISUAL_CROSSING_API_KEY in .env — skipping fetch")
        return []

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services"
        f"/timeline/{lat},{lon}/{start_date}/{end_date}"
        f"?unitGroup=us&key={VISUAL_CROSSING_KEY}&include=hours"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching weather {start_date} to {end_date}: {e}")
        return []

    records = []
    for day in data.get("days", []):
        for hour in day.get("hours", []):
            records.append({
                "date": day["datetime"],
                "time": hour["datetime"],
                "wind_speed": hour.get("windspeed"),
                "dewpoint": hour.get("dew"),
                "temp": hour.get("temp"),
            })

    print(f"  [weather] Fetched {len(records)} hourly records for {start_date} to {end_date}")
    return records


def fetch_and_cache_weather(lat, lon, tournament_dates_df, event_tag):
    """
    For each tournament year, fetch weather if not cached in wind_data.db.

    Args:
        lat, lon: Course coordinates
        tournament_dates_df: DataFrame with year, round_num, round_date
        event_tag: Event name tag for wind_data.db
    """
    conn = sqlite3.connect(WIND_DATA_DB)
    _ensure_wind_db_schema(conn)

    # Group by year to get date range per tournament year
    for year, group in tournament_dates_df.groupby("year"):
        dates = group["round_date"].sort_values()
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")

        if _wind_data_exists(conn, start_date, event_tag):
            print(f"  [cache] Weather data present for {year} ({start_date} to {end_date})")
            continue

        if lat is None or lon is None:
            print(f"  [skip] No coordinates — cannot fetch weather for {year}")
            continue

        records = fetch_visual_crossing(lat, lon, start_date, end_date)
        if not records:
            print(f"  [fallback] Trying dg_historical fallback for {year}")
            _fallback_weather(conn, event_tag, year, group)
            continue

        # Write to DB
        first_event_id = event_ids[0] if event_ids else None
        rows = []
        for rec in records:
            rows.append({
                "date": rec["date"],
                "time": rec["time"],
                "wind_speed": rec["wind_speed"],
                "event": event_tag,
                "event_id": first_event_id,
                "dewpoint": rec["dewpoint"],
                "temp": rec["temp"],
            })
        pd.DataFrame(rows).to_sql("wind_data", conn, if_exists="append", index=False)
        conn.commit()
        print(f"  [cache] Stored {len(rows)} rows for {year}")

        # Rate limit: Visual Crossing free tier
        time.sleep(1)

    conn.close()


def _fallback_weather(conn, event_tag, year, group):
    """
    Fallback: use player_weather_data_test from dg_historical.db
    to populate approximate round-level weather data.
    """
    try:
        hist_conn = sqlite3.connect(DG_HISTORICAL_DB)
        query = """
            SELECT p.event_id, w.round_date,
                   AVG(w.wind) AS avg_wind,
                   AVG(w.dew) AS avg_dew,
                   AVG(w.temp) AS avg_temp
            FROM player_weather_data_test w
            JOIN player_rounds p
              ON p.player_name = w.player_name
             AND p.event_id = w.event_id
             AND p.round_date = w.round_date
            WHERE p.course_num = ? AND p.tour = ? AND p.year = ?
            GROUP BY p.event_id, w.round_date
            ORDER BY w.round_date
        """
        fallback_df = pd.read_sql_query(
            query, hist_conn, params=[course_id, tour_override, year]
        )
        hist_conn.close()

        if fallback_df.empty:
            print(
                f"  [fallback] No weather data for course_id={course_id}, "
                f"tour={tour_override}, year={year}"
            )
            return

        # Synthesize hourly records from daily averages (7am-7pm, one per hour)
        rows = []
        for _, row in fallback_df.iterrows():
            date_str = str(row["round_date"])[:10]
            for hour in range(7, 20):  # 7am to 7pm
                rows.append({
                    "date": date_str,
                    "time": f"{hour:02d}:00:00",
                    "wind_speed": row["avg_wind"],
                    "event": event_tag,
                    "event_id": int(row["event_id"]),
                    "dewpoint": row["avg_dew"],
                    "temp": row["avg_temp"],
                })

        if rows:
            pd.DataFrame(rows).to_sql("wind_data", conn, if_exists="append", index=False)
            conn.commit()
            print(f"  [fallback] Stored {len(rows)} synthetic hourly rows for {year}")

    except Exception as e:
        print(f"  [fallback] Error: {e}")


def _build_hourly_array(day_data, column):
    """
    Build an hourly array from wind_data.db rows for a single day.
    Returns array where index 0 = 6 AM, matching api_utils.calculate_average_wind().

    Covers hours 6-20 (6 AM to 8 PM) = 15 values, enough for the latest
    tee time (~1:30 PM) + 5-hour window = 6:30 PM.
    """
    # Parse hour from time string
    day_data = day_data.copy()
    day_data["hour"] = pd.to_datetime(
        day_data["time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour

    # Build dict: hour -> value
    hourly = dict(zip(day_data["hour"], day_data[column]))

    # Extract hours 6 through 20 inclusive (index 0 = 6 AM)
    arr = []
    for h in range(6, 21):
        arr.append(hourly.get(h, np.nan))

    # Interpolate any gaps
    arr = np.array(arr, dtype=float)
    if np.any(np.isnan(arr)) and not np.all(np.isnan(arr)):
        valid = ~np.isnan(arr)
        arr = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])

    return arr


def _player_window_average(teetime_str, hourly_array):
    """
    Compute 5-hour window average for one player's tee time.
    Same interpolation logic as api_utils.calculate_average_wind().

    Args:
        teetime_str: Tee time string (e.g. '8:15am', '1:05pm')
        hourly_array: Array where index 0 = 6 AM (from _build_hourly_array)

    Returns:
        Float average over the 5-hour window, or np.nan on parse failure.
    """
    if not teetime_str or pd.isna(teetime_str):
        return np.nan

    s = str(teetime_str).strip()
    if not s:
        return np.nan

    # Parse tee time — try multiple formats
    parsed = None
    for fmt in ["%I:%M%p", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"]:
        try:
            parsed = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        return np.nan

    # Convert to decimal hour relative to 6 AM (index 0 of hourly_array)
    tee_decimal = parsed.hour + parsed.minute / 60.0
    block_start = tee_decimal - 6.0   # e.g. 8:15 AM -> 2.25
    block_end = block_start + 5.0     # 5-hour playing window

    # Clamp to array bounds
    block_start = max(0.0, block_start)
    block_end = min(float(len(hourly_array) - 1), block_end)

    if block_start >= block_end:
        return np.nan

    # Minute-level interpolation (same as api_utils.calculate_average_wind)
    sample_points = np.arange(block_start, block_end, 1 / 60.0)
    samples = np.interp(sample_points, np.arange(len(hourly_array)), hourly_array)

    return float(np.mean(samples))


def _get_tee_times(event_id_list, min_year, course_num=None,
                   restrict_event_ids=None):
    """
    Pull all player tee times from dg_historical.db.

    Returns DataFrame: year, round_num, teetime
    """
    conn = sqlite3.connect(DG_HISTORICAL_DB)
    course_num = course_id if course_num is None else course_num
    where, params = _history_where(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    query = f"""
        SELECT year, round_num, teetime
        FROM player_rounds
        WHERE {where} AND teetime IS NOT NULL AND teetime != ''
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_weather_averages(event_tag, tournament_dates_df):
    """
    Compute field-average wind and dewpoint exposure per year/round,
    using each player's actual tee time and a 5-hour playing window
    with minute-level interpolation (same method as api_utils).

    Returns DataFrame: year, round_num, avg_wind, avg_dew
    """
    # Load hourly weather from wind_data.db
    conn = sqlite3.connect(WIND_DATA_DB)
    query = "SELECT * FROM wind_data WHERE event = ?"
    weather_df = pd.read_sql_query(query, conn, params=[event_tag])
    conn.close()

    if weather_df.empty:
        print("  WARNING: No weather data found in wind_data.db")
        return pd.DataFrame(columns=["year", "round_num", "avg_wind", "avg_dew"])

    weather_df["date"] = pd.to_datetime(weather_df["date"])

    # Load player tee times from dg_historical.db
    tee_df = _get_tee_times(historical_event_ids, start_yr, course_id)
    tee_counts = tee_df.groupby(["year", "round_num"]).size()

    results = []
    for _, trow in tournament_dates_df.iterrows():
        yr = int(trow["year"])
        rnd = int(trow["round_num"])
        rd_date = trow["round_date"]

        # Get hourly weather for this date
        rd_str = rd_date.strftime("%Y-%m-%d")
        day_data = weather_df[weather_df["date"].dt.strftime("%Y-%m-%d") == rd_str]

        if day_data.empty:
            results.append({"year": yr, "round_num": rnd, "avg_wind": np.nan, "avg_dew": np.nan})
            continue

        # Build hourly arrays (index 0 = 6 AM)
        wind_array = _build_hourly_array(day_data, "wind_speed")
        has_dew = "dewpoint" in day_data.columns and day_data["dewpoint"].notna().any()
        dew_array = _build_hourly_array(day_data, "dewpoint") if has_dew else None

        # Get tee times for this year/round
        round_tees = tee_df[(tee_df["year"] == yr) & (tee_df["round_num"] == rnd)]

        if round_tees.empty:
            # No tee times available — fall back to 7am-7pm flat average
            avg_wind = float(np.nanmean(wind_array))
            avg_dew = float(np.nanmean(dew_array)) if dew_array is not None else np.nan
            results.append({"year": yr, "round_num": rnd, "avg_wind": avg_wind, "avg_dew": avg_dew})
            print(f"  [weather] {yr} R{rnd}: no tee times — using flat 6am-8pm average")
            continue

        # Compute per-player 5-hour window averages, then take field mean
        player_winds = []
        player_dews = []
        for tt in round_tees["teetime"]:
            pw = _player_window_average(tt, wind_array)
            if not np.isnan(pw):
                player_winds.append(pw)
            if dew_array is not None:
                pd_val = _player_window_average(tt, dew_array)
                if not np.isnan(pd_val):
                    player_dews.append(pd_val)

        avg_wind = float(np.mean(player_winds)) if player_winds else np.nan
        avg_dew = float(np.mean(player_dews)) if player_dews else np.nan

        n_players = len(round_tees)
        n_parsed = len(player_winds)
        if n_parsed < n_players:
            print(f"  [weather] {yr} R{rnd}: {n_parsed}/{n_players} tee times parsed")

        results.append({"year": yr, "round_num": rnd, "avg_wind": avg_wind, "avg_dew": avg_dew})

    return pd.DataFrame(results)


# ===========================================================================
# Step 5: Wind Coefficient Lookup
# ===========================================================================

def get_wind_coefficient(event_id_list, bl_wind, w_override, course_num=None):
    """
    Compute blended wind factor from wind_test.csv.
    Same formula as api_utils.compute_wind_factor().

    Returns (wind_factor, course_wind_effect)
    """
    if w_override != 0:
        return w_override, w_override

    try:
        course_num = course_id if course_num is None else course_num
        course_wind_effect, source = lookup_course_wind_effect(
            course_id=course_num,
            event_ids=event_id_list,
            wind_test_path=WIND_TEST_CSV,
        )
    except FileNotFoundError:
        print(f"  WARNING: {WIND_TEST_CSV} not found — using default 0.08")
        course_wind_effect, source = 0.08, "default (wind_test.csv missing)"

    print(f"  Wind coefficient source: {source}")

    wind_factor = course_wind_effect * 0.4 + bl_wind * 0.6
    return wind_factor, course_wind_effect


# ===========================================================================
# Step 6: Field Strength Matching
# ===========================================================================

def get_field_strength(event_id_list, min_year, course_num=None,
                       restrict_event_ids=None):
    """
    Match tournament event names from dg_historical.db to field_strength.db.

    Returns dict: {year: mean_skill}
    """
    # Get event names per year from dg_historical
    hist_conn = sqlite3.connect(DG_HISTORICAL_DB)
    course_num = course_id if course_num is None else course_num
    where, params = _history_where(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    query = f"""
        SELECT DISTINCT year, event_name
        FROM player_rounds
        WHERE {where}
        ORDER BY year
    """
    event_names_df = pd.read_sql_query(query, hist_conn, params=params)
    hist_conn.close()

    # Load field strength DB
    fs_conn = sqlite3.connect(FIELD_STRENGTH_DB)
    fs_df = pd.read_sql_query("SELECT [Tournament Name], [Mean Skill], Year FROM field_strength", fs_conn)
    fs_conn.close()
    fs_df["Year"] = fs_df["Year"].astype(str)

    mean_skills = {}

    for _, row in event_names_df.iterrows():
        year = row["year"]
        event_name = row["event_name"]
        year_str = str(year)

        # Filter field strength to this year
        fs_year = fs_df[fs_df["Year"] == year_str]

        if fs_year.empty:
            print(f"  WARNING: No field strength data for year {year}")
            mean_skills[year] = 0.0
            continue

        # Try exact match first
        exact = fs_year[fs_year["Tournament Name"] == event_name]
        if not exact.empty:
            mean_skills[year] = float(exact.iloc[0]["Mean Skill"])
            continue

        # Substring match: check if event_name contains or is contained by FS name
        matched = False
        for _, fs_row in fs_year.iterrows():
            fs_name = fs_row["Tournament Name"]
            # Bidirectional substring match
            if (event_name.lower() in fs_name.lower()) or (fs_name.lower() in event_name.lower()):
                mean_skills[year] = float(fs_row["Mean Skill"])
                matched = True
                break

        if not matched:
            # Try matching first significant word (skip "The", "A")
            event_words = [w for w in event_name.split() if w.lower() not in ("the", "a", "an")]
            for _, fs_row in fs_year.iterrows():
                fs_name = fs_row["Tournament Name"]
                if event_words and event_words[0].lower() in fs_name.lower():
                    mean_skills[year] = float(fs_row["Mean Skill"])
                    matched = True
                    print(f"  [field_str] Fuzzy match: '{event_name}' -> '{fs_name}' ({year})")
                    break

        if not matched:
            print(f"  WARNING: No field strength match for '{event_name}' ({year}) — using 0.0")
            mean_skills[year] = 0.0

    return mean_skills


# ===========================================================================
# Step 7: Compute Weather-Free Baseline
# ===========================================================================

def get_scoring_averages(event_id_list, min_year, course_num=None,
                         restrict_event_ids=None):
    """
    Get raw average score per year/round from dg_historical.db.

    Returns DataFrame: year, round_num, avg_score, score_std, n_players
    """
    conn = sqlite3.connect(DG_HISTORICAL_DB)
    course_num = course_id if course_num is None else course_num
    where, params = _history_where(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    query = f"""
        SELECT year, round_num,
               AVG(score) AS avg_score,
               SUM(score * score) AS sum_sq,
               SUM(score) AS sum_score,
               COUNT(score) AS n_players
        FROM player_rounds
        WHERE {where}
        GROUP BY year, round_num
        ORDER BY year, round_num
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Compute std dev from sum of squares (SQLite lacks STDEV)
    # std = sqrt(E[X^2] - E[X]^2)
    df["score_std"] = np.sqrt(
        (df["sum_sq"] / df["n_players"]) - (df["sum_score"] / df["n_players"]) ** 2
    )
    df.drop(columns=["sum_sq", "sum_score"], inplace=True)
    return df


def detect_cuts(event_id_list, min_year, course_num=None,
                restrict_event_ids=None):
    """
    Detect standard and late cuts per year.

    Returns dict: {year: {round_num: {'has_cut': bool, 'flagged': bool, 'drop_pct': float}}}
    """
    dates_df = get_tournament_dates(
        event_id_list, min_year, course_num, restrict_event_ids
    )
    cuts = {}

    for year, group in dates_df.groupby("year"):
        counts = dict(zip(group["round_num"], group["player_count"]))
        r1 = counts.get(1, 0)
        r2 = counts.get(2, 0)
        r3 = counts.get(3, 0)
        r4 = counts.get(4, 0)

        # Standard cut: R3 much fewer than R2
        r2_to_r3_drop = r2 - r3
        r2_to_r3_pct = r2_to_r3_drop / r2 if r2 > 0 else 0
        has_standard_cut = r2_to_r3_pct >= 0.20

        # Late cut: R4 much fewer than R2 but R3 wasn't cut
        r2_to_r4_drop = r2 - r4
        r2_to_r4_pct = r2_to_r4_drop / r2 if r2 > 0 else 0
        has_late_cut = (r2_to_r4_pct >= 0.20) and not has_standard_cut

        # Flagged: ambiguous cuts
        flag_r3 = (r2_to_r3_drop > 4) and (r2_to_r3_pct < 0.20)
        flag_r4 = (r2_to_r4_drop > 4) and not has_standard_cut and not has_late_cut

        year_cuts = {}
        for rnd in [1, 2, 3, 4]:
            if rnd <= 2:
                year_cuts[rnd] = {"has_cut": False, "flagged": False, "drop_pct": 0}
            elif rnd == 3:
                if has_standard_cut:
                    year_cuts[rnd] = {"has_cut": True, "flagged": False, "drop_pct": r2_to_r3_pct}
                elif flag_r3:
                    year_cuts[rnd] = {"has_cut": True, "flagged": True, "drop_pct": r2_to_r3_pct}
                else:
                    year_cuts[rnd] = {"has_cut": False, "flagged": False, "drop_pct": r2_to_r3_pct}
            elif rnd == 4:
                if has_standard_cut or has_late_cut:
                    year_cuts[rnd] = {"has_cut": True, "flagged": False, "drop_pct": r2_to_r4_pct}
                elif flag_r4:
                    year_cuts[rnd] = {"has_cut": True, "flagged": True, "drop_pct": r2_to_r4_pct}
                else:
                    year_cuts[rnd] = {"has_cut": False, "flagged": False, "drop_pct": r2_to_r4_pct}

        cuts[year] = year_cuts

    return cuts


def compute_baselines(scoring_df, weather_df, mean_skills, cuts, wind_factor):
    """
    Compute weather-free baseline for each year/round.

    Returns DataFrame with all detail columns.
    """
    # Merge scoring with weather
    merged = scoring_df.merge(weather_df, on=["year", "round_num"], how="left")

    # Compute dewpoint norm (historical average across all year-rounds)
    dew_norm = weather_df["avg_dew"].mean()
    if pd.isna(dew_norm):
        dew_norm = 60.0  # reasonable default
        print(f"  WARNING: No dewpoint data — using default norm {dew_norm}")

    results = []
    for _, row in merged.iterrows():
        year = int(row["year"])
        rnd = int(row["round_num"])
        raw_avg = row["avg_score"]
        wind_avg = row["avg_wind"] if pd.notna(row["avg_wind"]) else 0.0
        dew_avg = row["avg_dew"] if pd.notna(row["avg_dew"]) else dew_norm

        # Field strength adjustment
        ms = mean_skills.get(year, 0.0)
        cut_info = cuts.get(year, {}).get(rnd, {"has_cut": False, "flagged": False})
        if cut_info["has_cut"]:
            field_adj = ms + 0.25
        else:
            field_adj = ms

        # Weather effects (wind only — dew excluded to avoid double-counting with score_adj)
        wind_effect = wind_avg * wind_factor
        dew_delta = dew_avg - dew_norm  # kept for reference only

        # THE FORMULA: remove wind + normalize for field strength
        baseline = raw_avg + field_adj - wind_effect

        # Pull score_std and n_players from the scoring_df row
        score_std = row.get("score_std", np.nan)
        n_players = row.get("n_players", 0)

        results.append({
            "year": year,
            "round_num": rnd,
            "raw_avg": round(raw_avg, 2),
            "field_strength": round(ms, 3),
            "field_adj": round(field_adj, 3),
            "cut_applied": cut_info["has_cut"],
            "cut_flagged": cut_info["flagged"],
            "wind_avg_mph": round(wind_avg, 1),
            "wind_adj": round(-wind_effect, 2),
            "dew_avg": round(dew_avg, 1),
            "dew_delta": round(dew_delta, 1),
            "baseline": round(baseline, 2),
            "score_std": round(float(score_std), 2) if pd.notna(score_std) else np.nan,
            "n_players": int(n_players) if pd.notna(n_players) else 0,
        })

    return pd.DataFrame(results), dew_norm


# ===========================================================================
# Step 8: Recency-Weighted Average
# ===========================================================================

def compute_recency_weights(years):
    """
    Mild linear recency weighting: most recent year ~15% more than oldest.

    weight(year) = 1 + 0.025 * (year - min_year)
    """
    if not years:
        return {}

    sorted_years = sorted(set(years))
    min_yr = sorted_years[0]

    raw_weights = {y: 1.0 + 0.025 * (y - min_yr) for y in sorted_years}
    total = sum(raw_weights.values())

    return {y: w / total for y, w in raw_weights.items()}


def compute_final_estimates(detail_df):
    """
    Compute recency-weighted per-round estimates from detail rows.

    Returns dict: {round_num: weighted_baseline}
    """
    years = sorted(detail_df["year"].unique())
    weights = compute_recency_weights(years)

    # Add weight column
    detail_df = detail_df.copy()
    detail_df["weight"] = detail_df["year"].map(weights)

    final = {}
    for rnd in sorted(detail_df["round_num"].unique()):
        rnd_data = detail_df[detail_df["round_num"] == rnd]
        if rnd_data.empty:
            continue
        weighted_avg = (rnd_data["baseline"] * rnd_data["weight"]).sum() / rnd_data["weight"].sum()
        final[rnd] = round(weighted_avg, 2)

    return final, weights


# ===========================================================================
# Step 8b: Historical Comparisons & Regression
# ===========================================================================

def compute_historical_round_averages(detail_df):
    """
    Compute recency-weighted per-round averages of raw scoring avg,
    field strength, wind speed, and scoring std dev.

    Returns dict: {round_num: {hist_avg, hist_field, hist_wind, hist_std}}
    """
    years = sorted(detail_df["year"].unique())
    weights = compute_recency_weights(years)

    df = detail_df.copy()
    df["weight"] = df["year"].map(weights)

    result = {}
    for rnd in sorted(df["round_num"].unique()):
        rnd_data = df[df["round_num"] == rnd]
        if rnd_data.empty:
            continue
        w = rnd_data["weight"]
        w_sum = w.sum()

        hist_avg = (rnd_data["raw_avg"] * w).sum() / w_sum
        hist_field = (rnd_data["field_strength"] * w).sum() / w_sum

        wind_vals = rnd_data["wind_avg_mph"].fillna(0)
        hist_wind = (wind_vals * w).sum() / w_sum

        std_vals = rnd_data["score_std"].fillna(0)
        hist_std = (std_vals * w).sum() / w_sum

        result[rnd] = {
            "hist_avg": round(hist_avg, 2),
            "hist_field": round(hist_field, 3),
            "hist_wind": round(hist_wind, 1),
            "hist_std": round(hist_std, 2),
        }

    return result


def compute_wind_variance_regression(detail_df):
    """
    OLS regression of score_std ~ wind_avg_mph across all year-round observations.

    Returns (intercept, slope, r_squared, n_obs) or None if insufficient data.
    """
    df = detail_df.dropna(subset=["score_std", "wind_avg_mph"]).copy()
    if len(df) < 3:
        return None

    x = df["wind_avg_mph"].values
    y = df["score_std"].values

    # OLS: y = a + b*x
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

    return (round(intercept, 2), round(slope, 4), round(r_squared, 3), n)


# ===========================================================================
# Step 9: Output
# ===========================================================================

def print_report(detail_df, final_estimates, weights, wind_factor, course_wind_effect,
                 dew_norm, c_par):
    """Print the full console report."""
    first_eid = event_ids[0] if event_ids else "?"
    tname = tourney.title()

    print()
    print("=" * 80)
    print(f"  SCORING BASELINE: {tname} (event_id={first_eid}, par={c_par})")
    print(f"  Wind Factor: {wind_factor:.4f} (course={course_wind_effect:.2f}, baseline={baseline_wind:.2f})")
    print(f"  Dew excluded from baseline (handled by score_adj)")
    print("=" * 80)
    print()

    # Header
    header = (
        f"{'Year':>4} | {'Rnd':>3} | {'Raw Avg':>7} | {'Fld Str':>7} | {'Fld Adj':>7} | "
        f"{'Wind mph':>8} | {'Wind Adj':>8} | {'Dew dF':>7} | "
        f"{'Baseline':>8} | {'Weight':>6} | {'Notes':>10}"
    )
    print(header)
    print("-" * len(header))

    for _, row in detail_df.iterrows():
        yr = int(row["year"])
        wt = weights.get(yr, 0)
        notes = ""
        if row.get("cut_flagged", False):
            notes = "CUT?"
        elif row.get("cut_applied", False) and int(row["round_num"]) >= 3:
            notes = "+0.25 cut"

        print(
            f"{yr:>4} | {int(row['round_num']):>3} | {row['raw_avg']:>7.2f} | "
            f"{row['field_strength']:>7.3f} | {row['field_adj']:>7.3f} | "
            f"{row['wind_avg_mph']:>8.1f} | {row['wind_adj']:>8.2f} | "
            f"{row['dew_delta']:>+7.1f} | "
            f"{row['baseline']:>8.2f} | {wt:>6.3f} | {notes:>10}"
        )

    print()
    print("=" * 80)
    print("FINAL ESTIMATES (recency-weighted average):")
    parts = []
    for rnd in sorted(final_estimates.keys()):
        parts.append(f"  R{rnd}: {final_estimates[rnd]:.2f}")
    print("  ".join(parts))
    print("=" * 80)

    # Score adj suggestion (relative to par)
    print()
    print("SUGGESTED score_adj VALUES (baseline - par):")
    for rnd in sorted(final_estimates.keys()):
        adj = final_estimates[rnd] - c_par
        print(f"  score_adj_r{rnd} = {adj:+.2f}")

    # Dew reference summary (excluded from baseline — handled by score_adj)
    print()
    print("=" * 80)
    print("DEW REFERENCE (excluded from baseline — handled by score_adj):")
    dew_deltas = detail_df["dew_delta"].dropna()
    if not dew_deltas.empty:
        max_idx = dew_deltas.abs().idxmax()
        max_row = detail_df.loc[max_idx]
        print(f"  Dew norm: {dew_norm:.1f}F")
        print(
            f"  Largest delta: {max_row['dew_delta']:+.1f}F "
            f"({int(max_row['year'])} R{int(max_row['round_num'])})"
        )
    else:
        print("  No dewpoint data available")
    print("=" * 80)

    # Flag any ambiguous cuts
    flagged = detail_df[detail_df["cut_flagged"] == True]
    if not flagged.empty:
        print()
        for _, frow in flagged.iterrows():
            print(
                f"  WARNING: {int(frow['year'])} R{int(frow['round_num'])}: "
                f"Ambiguous cut detected. Applied +0.25 — review."
            )


def save_csv(detail_df, final_estimates, weights):
    """Save detail + summary to CSV."""
    df = detail_df.copy()
    df["weight"] = df["year"].map(weights)

    # Add summary rows
    for rnd, val in final_estimates.items():
        summary_row = {
            "year": "FINAL",
            "round_num": rnd,
            "raw_avg": "",
            "field_strength": "",
            "field_adj": "",
            "cut_applied": "",
            "cut_flagged": "",
            "wind_avg_mph": "",
            "wind_adj": "",
            "dew_avg": "",
            "dew_delta": "",
            "baseline": val,
            "weight": "",
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    output_path = f"scoring_baseline_{tourney}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  CSV saved: {output_path}")


def _to_native(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def save_to_sheets(detail_df, final_estimates, weights):
    """Write results to Google Sheets 'Scoring Baseline' tab."""
    try:
        from sheets_storage import get_spreadsheet, _get_or_create_tab
    except ImportError:
        print("  WARNING: sheets_storage not available — skipping Sheets write")
        return

    TAB_NAME = "Scoring Baseline"
    HEADERS = [
        "Year", "Round", "Raw Avg", "Field Strength", "Field Adj",
        "Cut Applied", "Wind Avg (mph)", "Wind Adj", "Dew Delta (F)",
        "Baseline", "Weight",
    ]

    try:
        spreadsheet = get_spreadsheet()
        ws = _get_or_create_tab(spreadsheet, TAB_NAME, HEADERS)

        # Clear existing data (keep headers)
        ws.clear()
        ws.append_row(HEADERS, value_input_option="RAW")

        # Detail rows
        rows = []
        for _, row in detail_df.iterrows():
            yr = int(row["year"])
            wt = float(weights.get(yr, 0))
            rows.append([
                yr,
                int(row["round_num"]),
                round(float(row["raw_avg"]), 2),
                round(float(row["field_strength"]), 3),
                round(float(row["field_adj"]), 3),
                "Yes" if row.get("cut_applied") else "",
                round(float(row["wind_avg_mph"]), 1),
                round(float(row["wind_adj"]), 2),
                round(float(row["dew_delta"]), 1),
                round(float(row["baseline"]), 2),
                round(wt, 4),
            ])

        # Summary rows
        rows.append([])  # blank separator
        for rnd in sorted(final_estimates.keys()):
            rows.append(["FINAL", int(rnd), "", "", "", "", "", "", "", "",
                         float(final_estimates[rnd]), ""])

        ws.append_rows(rows, value_input_option="USER_ENTERED")
        print(f"  [sheets] Wrote {len(rows)} rows to '{TAB_NAME}' tab")

    except Exception as e:
        print(f"  WARNING: Sheets write failed: {e}")


# ===========================================================================
# Write Estimates to round_config (backup fallback for nightly sim)
# ===========================================================================

def _compute_field_avg_wind(wind_arr, api_key, round_num):
    """
    Compute field-average wind exposure for a round using actual tee times
    and 5-hour playing windows.

    Fetches tee times from DataGolf field-updates API, then for each player
    computes their 5-hour wind average using calculate_average_wind().
    Falls back to simple array average if tee times unavailable.

    Args:
        wind_arr: list of hourly wind values (index 0 = 6 AM, 15 elements)
        api_key: DataGolf API key
        round_num: which round's tee times to fetch (1-4)

    Returns:
        float: field-average wind exposure in mph
    """
    from api_utils import fetch_field_updates, calculate_average_wind

    if not wind_arr:
        return 0.0

    # Try fetching tee times for this round
    teetime_col = f"r{round_num}_teetime"
    try:
        field_df = fetch_field_updates(api_key, teetime_col=teetime_col)
    except Exception as e:
        print(f"    [wind] Could not fetch R{round_num} tee times: {e}")
        field_df = None

    if field_df is not None and teetime_col in field_df.columns:
        tee_times = field_df[teetime_col].dropna()
        if len(tee_times) > 10:
            # Compute per-player 5-hour window average, then field mean
            player_winds = []
            for tt in tee_times:
                pw = calculate_average_wind(tt, wind_arr)
                if pw > 0:
                    player_winds.append(pw)

            if player_winds:
                avg = sum(player_winds) / len(player_winds)
                print(f"    [wind] R{round_num}: {len(player_winds)} players, "
                      f"field avg wind = {avg:.1f} mph (tee-time weighted)")
                return avg

    # Fallback: simple average of hourly wind array
    avg = sum(wind_arr) / len(wind_arr)
    print(f"    [wind] R{round_num}: using simple avg = {avg:.1f} mph (no tee times)")
    return avg


def compute_sg_variance_analysis(event_id_list, min_year, course_num=None):
    """
    SG-based variance analysis using trailing 50-round player std devs
    to separate course-driven variance from field-composition variance.

    Returns dict with:
      - 'rows': list of per-year dicts (year, qual_players, actual_std, field_exp_std, global_avg, vs_expected, vs_global)
      - 'global_avg_std': float
      - 'weighted_row': dict (weighted average across years)
    """
    current_year = datetime.now().year
    conn = sqlite3.connect(DG_HISTORICAL_DB)

    # Step 1: Global baseline — per-round std dev of sg_total_adj across ALL PGA events, last 5 years
    global_query = """
        SELECT event_id, year, round_num,
               SUM(sg_total_adj * sg_total_adj) AS sum_sq,
               SUM(sg_total_adj) AS sum_val,
               COUNT(sg_total_adj) AS n
        FROM player_rounds
        WHERE tour = 'pga' AND year >= ?
          AND sg_total_adj IS NOT NULL
        GROUP BY event_id, year, round_num
    """
    global_df = pd.read_sql_query(global_query, conn, params=[current_year - 5])
    # Compute std dev per group: sqrt(E[X^2] - E[X]^2)
    global_df["std"] = np.sqrt(
        (global_df["sum_sq"] / global_df["n"]) - (global_df["sum_val"] / global_df["n"]) ** 2
    )
    global_avg_std = float(global_df["std"].mean())
    print(f"  [sg_var] Global avg round std dev (PGA, {current_year-5}-{current_year}): {global_avg_std:.3f}")

    # Step 2: Get player-year combos at this course
    course_num = course_id if course_num is None else course_num
    history_where, history_params = _history_where(
        event_id_list, min_year, course_num
    )
    player_year_query = f"""
        SELECT DISTINCT player_name, year, MIN(round_date) AS event_start
        FROM player_rounds
        WHERE {history_where}
        GROUP BY player_name, year
    """
    player_years = pd.read_sql_query(
        player_year_query, conn, params=history_params
    )
    player_years["event_start"] = pd.to_datetime(player_years["event_start"])
    print(f"  [sg_var] Found {len(player_years)} player-year combos at this course (tour='{tour_override}')")

    if player_years.empty:
        conn.close()
        return None

    # Step 3: Get ALL rounds for those players (all tours)
    unique_players = player_years["player_name"].unique().tolist()
    # Batch query to avoid SQLite parameter limits
    all_rounds = []
    batch_size = 500
    for i in range(0, len(unique_players), batch_size):
        batch = unique_players[i:i + batch_size]
        ph = ",".join("?" * len(batch))
        batch_query = f"""
            SELECT player_name, round_date, sg_total_adj
            FROM player_rounds
            WHERE player_name IN ({ph}) AND sg_total_adj IS NOT NULL
            ORDER BY player_name, round_date
        """
        batch_df = pd.read_sql_query(batch_query, conn, params=batch)
        all_rounds.append(batch_df)

    rounds_df = pd.concat(all_rounds, ignore_index=True)
    rounds_df["round_date"] = pd.to_datetime(rounds_df["round_date"])
    print(f"  [sg_var] Loaded {len(rounds_df)} total rounds for {len(unique_players)} players")

    # Step 4: Get sg_total_adj at this event per player-year (for actual std dev)
    event_sg_query = f"""
        SELECT player_name, year, sg_total_adj
        FROM player_rounds
        WHERE {history_where}
          AND sg_total_adj IS NOT NULL
    """
    event_sg = pd.read_sql_query(
        event_sg_query, conn, params=history_params
    )
    conn.close()

    # Step 4: Compute trailing 50-round std dev per player-year
    trailing_stds = {}
    for _, py in player_years.iterrows():
        pname = py["player_name"]
        yr = py["year"]
        estart = py["event_start"]

        player_data = rounds_df[
            (rounds_df["player_name"] == pname) &
            (rounds_df["round_date"] < estart)
        ].sort_values("round_date")

        if len(player_data) < 50:
            continue

        last_50 = player_data.tail(50)["sg_total_adj"]
        trailing_stds[(pname, yr)] = float(last_50.std())

    print(f"  [sg_var] {len(trailing_stds)} player-years with 50+ trailing rounds "
          f"(dropped {len(player_years) - len(trailing_stds)})")

    # Step 5: Per-year results
    years_in_data = sorted(player_years["year"].unique())
    result_rows = []
    for yr in years_in_data:
        # Qualified players this year
        qual_keys = [(p, y) for (p, y) in trailing_stds.keys() if y == yr]
        qual_players = [k[0] for k in qual_keys]
        n_qual = len(qual_players)

        if n_qual < 10:
            continue

        # Actual std dev: sg_total_adj at this event, qualified players only, all rounds pooled
        yr_event = event_sg[
            (event_sg["year"] == yr) &
            (event_sg["player_name"].isin(qual_players))
        ]
        if yr_event.empty:
            continue
        actual_std = float(yr_event["sg_total_adj"].std())

        # Field expected std dev: mean of qualified players' trailing 50-round std devs
        field_exp_std = float(np.mean([trailing_stds[k] for k in qual_keys]))

        vs_expected = actual_std - field_exp_std
        vs_global = actual_std - global_avg_std

        result_rows.append({
            "year": yr,
            "qual_players": n_qual,
            "actual_std": round(actual_std, 3),
            "field_exp_std": round(field_exp_std, 3),
            "global_avg": round(global_avg_std, 3),
            "vs_expected": round(vs_expected, 3),
            "vs_global": round(vs_global, 3),
        })

    if not result_rows:
        print("  [sg_var] No years with enough qualified players")
        return None

    # Step 6: Recency-weighted average
    row_years = [r["year"] for r in result_rows]
    weights = compute_recency_weights(row_years)

    w_actual = sum(r["actual_std"] * weights[r["year"]] for r in result_rows)
    w_field_exp = sum(r["field_exp_std"] * weights[r["year"]] for r in result_rows)
    w_vs_expected = sum(r["vs_expected"] * weights[r["year"]] for r in result_rows)
    w_vs_global = sum(r["vs_global"] * weights[r["year"]] for r in result_rows)

    weighted_row = {
        "actual_std": round(w_actual, 3),
        "field_exp_std": round(w_field_exp, 3),
        "global_avg": round(global_avg_std, 3),
        "vs_expected": round(w_vs_expected, 3),
        "vs_global": round(w_vs_global, 3),
    }

    # Print summary
    print(f"\n  SG Variance Analysis:")
    print(f"  {'Year':>4}  {'Qual':>4}  {'Actual':>7}  {'FldExp':>7}  {'Global':>7}  {'vsExp':>7}  {'vsGlb':>7}")
    for r in result_rows:
        print(f"  {r['year']:>4}  {r['qual_players']:>4}  {r['actual_std']:>7.3f}  "
              f"{r['field_exp_std']:>7.3f}  {r['global_avg']:>7.3f}  "
              f"{r['vs_expected']:>+7.3f}  {r['vs_global']:>+7.3f}")
    print(f"  {'WtAvg':>4}  {'':>4}  {weighted_row['actual_std']:>7.3f}  "
          f"{weighted_row['field_exp_std']:>7.3f}  {weighted_row['global_avg']:>7.3f}  "
          f"{weighted_row['vs_expected']:>+7.3f}  {weighted_row['vs_global']:>+7.3f}")

    return {
        "rows": result_rows,
        "global_avg_std": global_avg_std,
        "weighted_row": weighted_row,
    }


# Winsorize the extreme left tail of category SG distributions before computing
# variance/skew. The worst ~0.5% of rounds in a PGA field are frequently posted
# by non-PGA-regular players (Monday qualifiers, sponsor invites, KFT call-ups)
# whose blow-ups distort the std and (especially) the skew. Capping rather than
# dropping keeps the sample size and removes only the outliers' leverage.
LEFT_TRIM_PCT = 0.005


def _winsorize_left(values, pct=LEFT_TRIM_PCT):
    """Cap values below the `pct` quantile at that quantile (left tail only).

    Self-disables on small samples (fewer than 1/pct points) so per-player
    trailing windows are untouched and only the large pooled distributions
    (event field, tour-wide baseline) are affected. Returns a 1-D float array.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if pct <= 0 or arr.size < 1.0 / pct:
        return arr
    floor = np.quantile(arr, pct)
    return np.clip(arr, floor, None)


def compute_sg_category_variance_analysis(event_id_list, min_year, course_id=None):
    """
    Per-category SG variance analysis — computes course variance multipliers
    AND skewness for OTT, APP, ARG, PUTT.

    Returns dict with:
      - 'category_mults': {sg_ott: float, ...}  (weighted avg actual/field_exp)
      - 'category_skew': {sg_ott: float, ...}   (recency-weighted at-course skewness)
      - 'baseline_skew': {sg_ott: float, ...}   (tour-wide baseline skewness)
      - 'per_year': list of per-year dicts with per-category breakdowns
      - 'weighted': dict with weighted averages per category
    """
    from scipy import stats as sp_stats

    CAT_COLS = ["sg_ott_adj", "sg_app_adj", "sg_arg_adj", "sg_putt_adj"]
    CAT_NAMES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

    current_year = datetime.now().year
    conn = sqlite3.connect(DG_HISTORICAL_DB)

    # Step 1: Global baseline per category (std devs via aggregation)
    cat_select = ", ".join(
        f"SUM({c} * {c}) AS sum_sq_{c}, SUM({c}) AS sum_val_{c}, COUNT({c}) AS n_{c}"
        for c in CAT_COLS
    )
    global_query = f"""
        SELECT event_id, year, round_num, {cat_select}
        FROM player_rounds
        WHERE tour = 'pga' AND year >= ?
          AND sg_total_adj IS NOT NULL
        GROUP BY event_id, year, round_num
    """
    global_df = pd.read_sql_query(global_query, conn, params=[current_year - 5])

    global_cat_stds = {}
    for c in CAT_COLS:
        mask = global_df[f"n_{c}"] > 5
        gf = global_df[mask]
        stds = np.sqrt(
            (gf[f"sum_sq_{c}"] / gf[f"n_{c}"]) - (gf[f"sum_val_{c}"] / gf[f"n_{c}"]) ** 2
        )
        global_cat_stds[c] = float(stds.mean())

    # Step 1b: Tour-wide baseline skewness (needs raw data, not aggregated)
    baseline_skew_query = f"""
        SELECT {', '.join(CAT_COLS)}
        FROM player_rounds
        WHERE tour = 'pga' AND year >= ?
          AND sg_total_adj IS NOT NULL
    """
    baseline_raw = pd.read_sql_query(baseline_skew_query, conn, params=[current_year - 5])
    baseline_skew = {}
    for c, name in zip(CAT_COLS, CAT_NAMES):
        vals = _winsorize_left(baseline_raw[c].values)
        baseline_skew[name] = round(float(sp_stats.skew(vals)), 2)

    print(f"\n  [sg_cat_var] Global baseline (PGA, {current_year-5}-{current_year}):")
    for c, name in zip(CAT_COLS, CAT_NAMES):
        print(f"    {name}: std={global_cat_stds[c]:.3f}  skew={baseline_skew[name]:+.2f}")

    # Step 2: Player-year combos at this course (same as total analysis)
    analysis_where = None
    analysis_params = None

    # Course filter: restrict the at-course stats to the SPECIFIC course being
    # played this week. Event IDs are deliberately ignored unless the weekly
    # config supplied an explicit historical override. Falls back to event-level
    # (all venues) if this course has no history at the event — e.g. a debut
    # venue — so we never silently emit nothing (which would leave stale values
    # on the sheet).
    if course_id is not None:
        course_where, course_params = _history_where(
            event_id_list, min_year, course_id
        )
        n_course = conn.execute(
            f"SELECT COUNT(*) FROM player_rounds WHERE {course_where}",
            course_params,
        ).fetchone()[0]
        if n_course == 0:
            print(f"  [sg_cat_var] WARNING: course_id {course_id} has no rounds "
                  f"since {min_year} — falling back "
                  f"to EVENT-LEVEL (all venues).")
        else:
            analysis_where, analysis_params = course_where, course_params
            print(f"  [sg_cat_var] Course filter ON: course_num={course_id} "
                  f"({n_course} same-course rounds).")

    if analysis_where is None:
        analysis_where, analysis_params = _history_where(
            event_id_list, min_year, None, restrict_event_ids=True
        )

    player_year_query = f"""
        SELECT DISTINCT player_name, year, MIN(round_date) AS event_start
        FROM player_rounds
        WHERE {analysis_where}
        GROUP BY player_name, year
    """
    player_years = pd.read_sql_query(
        player_year_query, conn, params=analysis_params
    )
    player_years["event_start"] = pd.to_datetime(player_years["event_start"])
    print(f"  [sg_cat_var] Found {len(player_years)} player-year combos at this course (tour='{tour_override}')")

    if player_years.empty:
        conn.close()
        return None

    # Step 3: Get ALL rounds for trailing std computation
    unique_players = player_years["player_name"].unique().tolist()
    cat_cols_str = ", ".join(CAT_COLS)
    all_rounds = []
    batch_size = 500
    for i in range(0, len(unique_players), batch_size):
        batch = unique_players[i:i + batch_size]
        ph = ",".join("?" * len(batch))
        batch_query = f"""
            SELECT player_name, round_date, {cat_cols_str}
            FROM player_rounds
            WHERE player_name IN ({ph}) AND sg_total_adj IS NOT NULL
            ORDER BY player_name, round_date
        """
        batch_df = pd.read_sql_query(batch_query, conn, params=batch)
        all_rounds.append(batch_df)

    rounds_df = pd.concat(all_rounds, ignore_index=True)
    rounds_df["round_date"] = pd.to_datetime(rounds_df["round_date"])

    # Step 4: Event SG categories at this course
    event_sg_query = f"""
        SELECT player_name, year, {cat_cols_str}
        FROM player_rounds
        WHERE {analysis_where}
          AND sg_total_adj IS NOT NULL
    """
    event_sg = pd.read_sql_query(
        event_sg_query, conn, params=analysis_params
    )
    conn.close()

    # Step 5: Trailing 50-round std per category per player-year
    trailing_cat_stds = {}  # (player, year) -> {col: std}
    for _, py in player_years.iterrows():
        pname = py["player_name"]
        yr = py["year"]
        estart = py["event_start"]

        player_data = rounds_df[
            (rounds_df["player_name"] == pname) &
            (rounds_df["round_date"] < estart)
        ].sort_values("round_date")

        if len(player_data) < 50:
            continue

        last_50 = player_data.tail(50)
        cat_stds = {}
        for c in CAT_COLS:
            valid = last_50[c].dropna()
            cat_stds[c] = float(valid.std()) if len(valid) >= 30 else None
        trailing_cat_stds[(pname, yr)] = cat_stds

    print(f"  [sg_cat_var] {len(trailing_cat_stds)} player-years with 50+ trailing rounds")

    # Step 6: Per-year results
    years_in_data = sorted(player_years["year"].unique())
    result_rows = []
    for yr in years_in_data:
        qual_keys = [(p, y) for (p, y) in trailing_cat_stds.keys() if y == yr]
        qual_players = [k[0] for k in qual_keys]
        n_qual = len(qual_players)

        if n_qual < 10:
            continue

        yr_event = event_sg[
            (event_sg["year"] == yr) &
            (event_sg["player_name"].isin(qual_players))
        ]
        if yr_event.empty:
            continue

        row_data = {"year": yr, "qual_players": n_qual}
        for c, name in zip(CAT_COLS, CAT_NAMES):
            valid_event = yr_event[c].dropna()
            # Cap the worst ~0.5% (fringe-player blow-ups) before measuring the
            # event-field std and skew. field_exp_std (per-player trailing std)
            # is left as-is — that is each player's own genuine volatility.
            ve = _winsorize_left(valid_event.values)
            actual_std = float(np.std(ve, ddof=1)) if len(ve) >= 10 else None

            field_stds = [trailing_cat_stds[k][c] for k in qual_keys
                          if trailing_cat_stds[k][c] is not None]
            field_exp_std = float(np.mean(field_stds)) if field_stds else None

            row_data[f"{name}_actual"] = round(actual_std, 3) if actual_std else None
            row_data[f"{name}_field_exp"] = round(field_exp_std, 3) if field_exp_std else None
            if actual_std and field_exp_std and field_exp_std > 0:
                row_data[f"{name}_mult"] = round(actual_std / field_exp_std, 2)
            else:
                row_data[f"{name}_mult"] = None

            # Per-year skewness at this course (winsorized to limit blow-up leverage)
            if len(ve) >= 20:
                row_data[f"{name}_skew"] = round(float(sp_stats.skew(ve)), 2)
            else:
                row_data[f"{name}_skew"] = None

        result_rows.append(row_data)

    if not result_rows:
        print("  [sg_cat_var] No years with enough qualified players")
        return None

    # Step 7: Recency-weighted averages
    row_years = [r["year"] for r in result_rows]
    weights = compute_recency_weights(row_years)

    weighted = {}
    category_mults = {}
    category_skew = {}
    for name in CAT_NAMES:
        w_actual = 0.0
        w_field = 0.0
        w_total = 0.0
        w_skew = 0.0
        w_skew_total = 0.0
        for r in result_rows:
            w = weights[r["year"]]
            if r[f"{name}_actual"] is not None and r[f"{name}_field_exp"] is not None:
                w_actual += r[f"{name}_actual"] * w
                w_field += r[f"{name}_field_exp"] * w
                w_total += w
            if r.get(f"{name}_skew") is not None:
                w_skew += r[f"{name}_skew"] * w
                w_skew_total += w

        if w_total > 0:
            w_actual /= w_total
            w_field /= w_total
            w_actual *= sum(weights.values())
            w_field *= sum(weights.values())

        weighted[f"{name}_actual"] = round(w_actual, 3)
        weighted[f"{name}_field_exp"] = round(w_field, 3)
        if w_field > 0:
            category_mults[name] = round(w_actual / w_field, 2)
        else:
            category_mults[name] = 1.0

        if w_skew_total > 0:
            w_skew /= w_skew_total
            w_skew *= sum(weights.values())
        category_skew[name] = round(w_skew, 2)

    # Print detailed "see the work" table — variance multipliers
    print(f"\n  Per-Category SG Variance Analysis:")
    print(f"  {'Year':>4}  {'Qual':>4}  ", end="")
    for name in CAT_NAMES:
        short = name.replace("sg_", "").upper()
        print(f"{'Act':>6} {'Exp':>6} {'Mult':>5}  ", end="")
    print()
    print(f"  {'':>4}  {'':>4}  ", end="")
    for name in CAT_NAMES:
        short = name.replace("sg_", "").upper()
        print(f"{'---'+short+'---':>17}  ", end="")
    print()

    for r in result_rows:
        print(f"  {r['year']:>4}  {r['qual_players']:>4}  ", end="")
        for name in CAT_NAMES:
            a = r[f"{name}_actual"]
            f = r[f"{name}_field_exp"]
            m = r[f"{name}_mult"]
            print(f"{a if a else 'N/A':>6} {f if f else 'N/A':>6} {m if m else 'N/A':>5}  ", end="")
        print()

    print(f"  {'WtAvg':>4}  {'':>4}  ", end="")
    for name in CAT_NAMES:
        a = weighted[f"{name}_actual"]
        f = weighted[f"{name}_field_exp"]
        m = category_mults[name]
        print(f"{a:>6} {f:>6} {m:>5}  ", end="")
    print()

    # Print skewness table
    print(f"\n  Per-Category Skewness (negative = left-skewed / blowup-prone):")
    print(f"  {'Year':>4}  {'Qual':>4}  ", end="")
    for name in CAT_NAMES:
        short = name.replace("sg_", "").upper()
        print(f"{short:>7}  ", end="")
    print()
    for r in result_rows:
        print(f"  {r['year']:>4}  {r['qual_players']:>4}  ", end="")
        for name in CAT_NAMES:
            s = r.get(f"{name}_skew")
            val = f"{s:>+7}" if s is not None else f"{'N/A':>7}"
            print(f"{val}  ", end="")
        print()
    print(f"  {'WtAvg':>4}  {'':>4}  ", end="")
    for name in CAT_NAMES:
        print(f"{category_skew[name]:>+7}  ", end="")
    print()
    print(f"  {'Base':>4}  {'':>4}  ", end="")
    for name in CAT_NAMES:
        print(f"{baseline_skew[name]:>+7}  ", end="")
    print("  (tour-wide)")

    # Print config blocks
    print(f"\n  COURSE_CAT_MULTS = {{")
    for name in CAT_NAMES:
        print(f"      '{name}': {category_mults[name]},")
    print(f"  }}")
    print(f"  COURSE_CAT_SKEW = {{")
    for name in CAT_NAMES:
        print(f"      '{name}': {category_skew[name]},")
    print(f"  }}")

    return {
        "category_mults": category_mults,
        "category_skew": category_skew,
        "baseline_skew": baseline_skew,
        "per_year": result_rows,
        "weighted": weighted,
    }


# Earliest year reliably present in dg_historical.db — floor for the "venue's own
# older history" lookup used when an intermittent host has no in-window playing.
PRE_WINDOW_MIN_YEAR = 2017

# Rotating-major event_ids whose host venue changes year to year (PGA=33,
# US Open=26, Open Championship=100). For these the per-category variance almost
# always falls back to event-level (the venue rarely has in-window history), so
# blending the venue's own pre-window same-course profile is ON by default.
# The Masters (14) is intentionally excluded: it never leaves Augusta National,
# so it always has in-window same-course data and never falls back — the blend
# would never fire anyway.
ROTATING_MAJOR_EVENT_IDS = {26, 33, 100}

# Pre-window same-course variance blend. Default ON for rotating majors, OFF
# otherwise; override with --blend-prewindow-course / --no-blend-prewindow-course.
# Set in main() from argparse; mirrors the _USE_PYTHON flag pattern.
_BLEND_PREWINDOW_COURSE = any(eid in ROTATING_MAJOR_EVENT_IDS for eid in event_ids)


def _course_has_rounds_since(event_id_list, course_id, min_year):
    """True if this exact venue has any rounds at the event within [min_year, now]."""
    if course_id is None:
        return True
    conn = sqlite3.connect(DG_HISTORICAL_DB)
    where, params = _history_where(event_id_list, min_year, course_id)
    n = conn.execute(
        f"SELECT COUNT(*) FROM player_rounds WHERE {where}",
        params,
    ).fetchone()[0]
    conn.close()
    return n > 0


def _maybe_blend_prewindow_course(event_result, event_id_list, course_id):
    """50/50-blend the venue's own pre-window history into an event-level result.

    The per-category analysis falls back to EVENT-LEVEL (all venues since start_yr)
    when this exact venue has no playing within the standard window. Rotating
    majors (US Open, Open Championship) revisit venues on long cycles, so a
    venue's only same-course data can predate start_yr (e.g. Shinnecock last
    hosted in 2018). Equal-weighting the venue's own variance/skew profile against
    the all-venues event profile preserves course-specific structure instead of
    discarding it.

    OPT-IN: only runs when --blend-prewindow-course is passed (_BLEND_PREWINDOW_COURSE).
    Default behaviour keeps the plain event-level values; on weeks where a blend is
    possible it prints a one-line note so the option is discoverable.

    No-op (returns event_result unchanged) when: the flag is off, no course filter,
    the venue already has in-window data (no fallback occurred), or no pre-window
    same-course data exists.
    """
    if event_result is None or course_id is None:
        return event_result
    # Only relevant if the main result actually fell back to event-level.
    if _course_has_rounds_since(event_id_list, course_id, start_yr):
        return event_result
    # Require genuine pre-window same-course history; otherwise the pre-window
    # call would itself fall back to event-level and we'd blend a result with
    # itself (true debut venue). Keep the event-level values in that case.
    if not _course_has_rounds_since(event_id_list, course_id, PRE_WINDOW_MIN_YEAR):
        return event_result
    # Fell back to event-level AND this venue has its own pre-window history: an
    # intermittent-host (rotating major) week. Blending is opt-in — surface the
    # option but default to event-level.
    if not _BLEND_PREWINDOW_COURSE:
        print(f"  [sg_cat_var] NOTE: course {course_id} has same-course history "
              f"before {start_yr} that could be blended in. Using event-level "
              f"values (default). Re-run with --blend-prewindow-course to "
              f"50/50-blend this venue's own variance/skew profile.")
        return event_result
    # Same-venue history that predates the standard window.
    pre = compute_sg_category_variance_analysis(
        event_id_list, min_year=PRE_WINDOW_MIN_YEAR, course_id=course_id)
    if pre is None:
        print(f"  [sg_cat_var] No pre-{start_yr} same-course history for course "
              f"{course_id} — keeping event-level values.")
        return event_result

    pre_years = sorted({r["year"] for r in pre["per_year"]})
    CAT_NAMES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    print(f"\n  [sg_cat_var] 50/50 BLEND: same-course {pre_years} vs "
          f"event-level (all venues since {start_yr})")
    print(f"  {'Cat':>8} | {'Course':>7} | {'Event':>7} | {'Blend':>7}   "
          f"skew crse/evt->bln")
    blended_mults, blended_skew = {}, {}
    for name in CAT_NAMES:
        mc = pre["category_mults"].get(name, 1.0)
        me = event_result["category_mults"].get(name, 1.0)
        sc = pre["category_skew"].get(name, 0.0)
        se = event_result["category_skew"].get(name, 0.0)
        blended_mults[name] = round(0.5 * mc + 0.5 * me, 2)
        blended_skew[name] = round(0.5 * sc + 0.5 * se, 2)
        print(f"  {name:>8} | {mc:>7} | {me:>7} | {blended_mults[name]:>7}   "
              f"{sc:+.2f}/{se:+.2f}->{blended_skew[name]:+.2f}")

    event_result["category_mults"] = blended_mults
    event_result["category_skew"] = blended_skew
    event_result["blend_note"] = (
        f"50/50 blend: course {pre_years} + event-level since {start_yr}")
    return event_result


def _write_details_tab(spreadsheet, table_rows, regression, sg_result,
                       sg_cat_result, field_strength, wind_factor, cut_adj):
    """
    Write analysis tables to the Details tab (moved from round_config cols E-S).
    Clear + rewrite each time.
    """
    import gspread
    try:
        from sheets_storage import _get_or_create_tab, TAB_DETAILS, DETAILS_HEADERS
    except ImportError:
        print("  WARNING: sheets_storage import failed — skipping Details tab")
        return

    try:
        # Details tab uses freeform sections with varying column widths,
        # so skip header validation (we clear + rewrite each time anyway)
        try:
            ws = spreadsheet.worksheet(TAB_DETAILS)
        except Exception:
            ws = spreadsheet.add_worksheet(title=TAB_DETAILS, rows=1000, cols=15)
        ws.clear()

        rows = []

        # Section 1: Expected scoring breakdown
        rows.append(["Scoring Breakdown", "Round", "Baseline", "Expected", "Hist Avg",
                      "Hist Field", "ThisWk Fld", "Hist Wind", "ThisWk Wind", "Hist StdDev", "Pred StdDev"])
        for trow in table_rows:
            rows.append([
                "", trow["round"], trow["baseline"], trow["expected"],
                trow["hist_avg"], trow["hist_field"], trow["this_wk_field"],
                trow["hist_wind"], trow["this_wk_wind"], trow["hist_std"],
                trow["pred_std"],
            ])

        # Summary
        summary_parts = [f"Field: {field_strength:+.4f} SG"]
        summary_parts.append(f"Wind Factor: {wind_factor:.4f}")
        if cut_adj > 0:
            summary_parts.append(f"Cut adj R3/R4: -{cut_adj:.2f}")
        rows.append(["", " | ".join(summary_parts)])

        rows.append([])  # blank separator

        # Section 2: Wind-Variance Regression
        if regression is not None:
            intercept, slope, r_sq, n_obs = regression
            rows.append(["Wind-Variance Regression", f"StdDev = {intercept:.2f} + {slope:.4f} * Wind_mph"])
            rows.append(["", f"R-sq: {r_sq:.3f} | N: {n_obs} obs"])
            pred_stds = [t["pred_std"] for t in table_rows if t["pred_std"] != ""]
            if pred_stds:
                suggested = round(np.mean(pred_stds), 2)
                rows.append(["", f"Suggested STD_DEV: {suggested} (current: {STD_DEV})"])
            rows.append([])

        # Section 3: SG Variance Analysis
        if sg_result is not None:
            sg_data_rows = sg_result["rows"]
            sg_weighted = sg_result["weighted_row"]
            rows.append(["SG Variance Analysis", "Year", "Qual Players", "Actual StdDev",
                          "Field Exp Std", "Global Avg", "vs Expected", "vs Global"])
            for sr in sg_data_rows:
                rows.append([
                    "", int(sr["year"]), int(sr["qual_players"]), float(sr["actual_std"]),
                    float(sr["field_exp_std"]), float(sr["global_avg"]),
                    f"{sr['vs_expected']:+.3f}", f"{sr['vs_global']:+.3f}",
                ])
            rows.append([
                "", "Wt Avg", "", float(sg_weighted["actual_std"]),
                float(sg_weighted["field_exp_std"]), float(sg_weighted["global_avg"]),
                f"{sg_weighted['vs_expected']:+.3f}", f"{sg_weighted['vs_global']:+.3f}",
            ])
            rows.append(["", "vs Expected < 0 = course compresses variance; > 0 = course amplifies"])
            rows.append([])

        # Section 4: Per-Category SG Variance
        CAT_NAMES_DISP = ["OTT", "APP", "ARG", "PUTT"]
        CAT_NAMES_KEY = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

        if sg_cat_result is not None:
            cat_data_rows = sg_cat_result["per_year"]
            cat_weighted = sg_cat_result["weighted"]
            cat_mults = sg_cat_result["category_mults"]

            # Header
            cat_headers = ["Per-Category Variance", "Year", "Qual"]
            for disp in CAT_NAMES_DISP:
                cat_headers.extend([f"{disp} Act", f"{disp} Exp", f"{disp} Mult"])
            rows.append(cat_headers)

            for cr in cat_data_rows:
                vals = ["", int(cr["year"]), int(cr["qual_players"])]
                for name in CAT_NAMES_KEY:
                    vals.append(cr.get(f"{name}_actual", ""))
                    vals.append(cr.get(f"{name}_field_exp", ""))
                    m = cr.get(f"{name}_mult")
                    vals.append(m if m is not None else "")
                rows.append(vals)

            # Weighted average
            wt_vals = ["", "Wt Avg", ""]
            for name in CAT_NAMES_KEY:
                wt_vals.append(cat_weighted.get(f"{name}_actual", ""))
                wt_vals.append(cat_weighted.get(f"{name}_field_exp", ""))
                wt_vals.append(cat_mults.get(name, ""))
            rows.append(wt_vals)

            # COURSE_CAT_MULTS summary
            mults_str = "  |  ".join(
                f"{name.replace('sg_', '').upper()}: {cat_mults[name]}"
                for name in CAT_NAMES_KEY
            )
            rows.append(["", f"COURSE_CAT_MULTS: {mults_str}"])

            # Skewness
            cat_skew = sg_cat_result.get("category_skew", {})
            base_skew = sg_cat_result.get("baseline_skew", {})
            rows.append([])
            skew_h = ["Skewness", ""]
            for disp in CAT_NAMES_DISP:
                skew_h.extend([f"{disp} Skew", "", ""])
            rows.append(skew_h)

            course_skew_vals = ["", "Course (wt avg)"]
            for name in CAT_NAMES_KEY:
                course_skew_vals.extend([cat_skew.get(name, ""), "", ""])
            rows.append(course_skew_vals)

            base_skew_vals = ["", "Tour baseline"]
            for name in CAT_NAMES_KEY:
                base_skew_vals.extend([base_skew.get(name, ""), "", ""])
            rows.append(base_skew_vals)

            skew_str = "  |  ".join(
                f"{name.replace('sg_', '').upper()}: {cat_skew.get(name, 0)}"
                for name in CAT_NAMES_KEY
            )
            rows.append(["", f"COURSE_CAT_SKEW: {skew_str}"])

        # Write all rows
        if rows:
            ws.append_rows(rows, value_input_option="USER_ENTERED")
        print(f"  [Details] Wrote {len(rows)} rows to Details tab")

    except Exception as e:
        print(f"  WARNING: Details tab write failed: {e}")


def write_estimates_to_round_config(final_estimates, wind_factor, detail_df=None):
    """
    Write per-round expected scoring averages to the round_config tab as
    expected_score_r1 through expected_score_r4. On a single-course pre-event
    slate, also synchronize expected_score_1 to the new R1 estimate.

    Adjusts the de-skilled/de-weathered baselines for THIS WEEK by adding back:
      - Field strength: mean of our model's 'my_pred' values
      - Per-round wind: tee-time-weighted field avg wind * wind_factor

    Formula: expected_score = baseline - field_strength + avg_wind * wind_factor

    Also writes an expanded breakdown table (10 cols) with historical comparisons,
    and a wind-variance regression summary below.
    """
    import os
    import gspread
    try:
        from sheets_storage import get_spreadsheet
    except ImportError as e:
        print(f"  WARNING: Import failed ({e}) — skipping round_config write")
        return

    api_key = os.getenv("DATAGOLF_API_KEY")

    # --- Field strength from pre-tournament course-fit predictions ---
    field_strength = 0.0
    pred_file = f"pre_course_fit_{tourney}.csv"
    if os.path.exists(pred_file):
        try:
            pred_df = pd.read_csv(pred_file)
            if "pred" in pred_df.columns:
                field_strength = pred_df["pred"].mean()
                print(f"  Field strength (mean pred): {field_strength:+.4f} SG/round "
                      f"({len(pred_df)} players from {pred_file})")
        except Exception as e:
            print(f"  WARNING: Could not read {pred_file}: {e}")
    else:
        print(f"  WARNING: {pred_file} not found — field strength = 0.0")
        print(f"  (Run pre-tournament pipeline first for accurate estimates)")

    # --- Read per-round wind from Sheet (rows 18-21, col F = comma-separated) ---
    round_wind_arrays = {}
    try:
        spreadsheet_tmp = get_spreadsheet()
        ws_tmp = spreadsheet_tmp.worksheet("round_config")
        all_data = ws_tmp.get_all_values()
        # Parse E-H rows 18-21 for wind_r1-r4 comma-separated values
        for row_idx in range(17, min(21, len(all_data))):
            row = all_data[row_idx]
            if len(row) >= 6 and row[4].strip().startswith("wind_r"):
                rnd_str = row[4].strip().replace("wind_r", "")
                try:
                    rnd_num = int(rnd_str)
                    vals = [float(x.strip()) for x in row[5].split(",") if x.strip()]
                    round_wind_arrays[rnd_num] = vals
                    print(f"  Read wind_r{rnd_num}: {len(vals)} values from Sheet")
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"  WARNING: Could not read wind from Sheet: {e}")

    # --- Compute tee-time-weighted wind exposure per round ---
    print(f"\n  Computing tee-time-weighted wind per round...")
    round_wind_avgs = {}
    for rnd in sorted(final_estimates.keys()):
        wind_arr = round_wind_arrays.get(rnd, [])
        round_wind_avgs[rnd] = _compute_field_avg_wind(wind_arr, api_key, rnd)

    # --- Historical comparisons (if detail_df provided) ---
    hist_avgs = {}
    regression = None
    sg_result = None
    sg_cat_result = None
    if detail_df is not None:
        hist_avgs = compute_historical_round_averages(detail_df)
        regression = compute_wind_variance_regression(detail_df)
        sg_result = compute_sg_variance_analysis(
            historical_event_ids, min_year=start_yr, course_num=course_id
        )
        _var_course_id = None if variance_event_level_only else course_id
        if variance_event_level_only:
            print(f"  [sg_cat_var] variance_event_level_only=True — pooling event "
                  f"{historical_event_ids} across all venues (course filter OFF).")
        sg_cat_result = compute_sg_category_variance_analysis(historical_event_ids, min_year=start_yr, course_id=_var_course_id)
        sg_cat_result = _maybe_blend_prewindow_course(sg_cat_result, historical_event_ids, _var_course_id)

    # --- Cut adjustment for R3/R4 ---
    cut_adj = 0.25 if CUT_LINE < 120 else 0.0

    # --- Compute adjusted expected scores ---
    adjusted = {}
    table_rows = []  # for Sheet breakdown
    print(f"\n  {'Rnd':>3} | {'Baseline':>8} | {'Expected':>8} | {'Hist Avg':>8} | "
          f"{'ThisWk Fld':>9} | {'Hist Wind':>9} | {'ThisWk Wnd':>10}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
    for rnd in sorted(final_estimates.keys()):
        base = final_estimates[rnd]
        avg_wind = round_wind_avgs.get(rnd, 0.0)
        wind_effect = avg_wind * wind_factor
        expected = base - field_strength + wind_effect + baseline_score_adj
        adjusted[rnd] = round(expected, 1)

        # This-week field adjustment (negative = stronger field)
        this_wk_field = -field_strength
        if rnd >= 3 and cut_adj > 0:
            this_wk_field -= cut_adj

        # Historical values
        h = hist_avgs.get(rnd, {})
        hist_avg = h.get("hist_avg", "")
        hist_field = h.get("hist_field", "")
        hist_wind = h.get("hist_wind", "")
        hist_std = h.get("hist_std", "")

        # Predicted std dev from regression
        pred_std = ""
        if regression is not None:
            intercept, slope, _, _ = regression
            pred_std = round(intercept + slope * avg_wind, 2)

        print(f"  R{rnd}  | {base:>8.2f} | {expected:>8.1f} | "
              f"{hist_avg if hist_avg != '' else 'N/A':>8} | "
              f"{this_wk_field:>+9.3f} | "
              f"{hist_wind if hist_wind != '' else 'N/A':>9} | {avg_wind:>10.1f}")

        table_rows.append({
            "round": f"R{rnd}",
            "baseline": round(base, 2),
            "expected": adjusted[rnd],
            "hist_avg": hist_avg,
            "hist_field": hist_field,
            "this_wk_field": round(this_wk_field, 3),
            "hist_wind": hist_wind,
            "this_wk_wind": round(avg_wind, 1),
            "hist_std": hist_std,
            "pred_std": pred_std,
        })

    # --- Write to round_config tab ---
    try:
        spreadsheet = get_spreadsheet()
        ws = spreadsheet.worksheet("round_config")

        # 1) Write expected_score_r1-r4 parameter values (cols A-B)
        all_values = ws.get("A:B")
        param_rows = {}
        for i, row in enumerate(all_values):
            if row and row[0].strip():
                param_rows[row[0].strip().lower()] = i + 1

        # Hard-coded per-round scores (sim_inputs.expected_score_override) take
        # precedence over the computed baseline — used when the venue has no
        # usable history (e.g. a rotating-venue major we have no data for).
        scores_to_write = expected_score_override if expected_score_override else adjusted
        _score_note = ("Hard-coded (sim_inputs.expected_score_override)"
                       if expected_score_override
                       else "Pre-tourney estimate (set by scoring_baseline.py)")
        updated_params = []
        for rnd in sorted(scores_to_write.keys()):
            param_name = f"expected_score_r{rnd}"
            value = str(scores_to_write[rnd])
            row_idx = param_rows.get(param_name)

            if row_idx is None:
                next_row = len(all_values) + 1
                ws.update_cell(next_row, 1, param_name)
                ws.update_cell(next_row, 2, value)
                ws.update_cell(next_row, 3, _score_note)
                all_values.append([param_name, value])
                updated_params.append(f"{param_name}={value} (new row)")
            else:
                ws.update_cell(row_idx, 2, value)
                updated_params.append(f"{param_name}={value}")

        # On a single-course pre-event slate, expected_score_1 is the live
        # absolute scoring average consumed by live_stats_engine/round_sim.
        # Keep it synchronized with the newly fitted R1 estimate. For a
        # multi-course slate expected_score_1 remains course-specific, so the
        # operator-managed course values must be preserved.
        course_pars_value = ""
        course_pars_row = param_rows.get("course_pars")
        if course_pars_row:
            course_pars_value = str(ws.cell(course_pars_row, 2).value or "")
        course_par_values = [
            value.strip() for value in course_pars_value.split(",") if value.strip()
        ]
        is_single_course = len(course_par_values) <= 1
        if is_single_course and 1 in scores_to_write:
            param_name = "expected_score_1"
            value = str(scores_to_write[1])
            row_idx = param_rows.get(param_name)
            if row_idx is None:
                next_row = len(all_values) + 1
                ws.update_cell(next_row, 1, param_name)
                ws.update_cell(next_row, 2, value)
                ws.update_cell(
                    next_row,
                    3,
                    "Live pre-R1 score synchronized by scoring_baseline.py",
                )
                all_values.append([param_name, value])
                updated_params.append(f"{param_name}={value} (new row)")
            else:
                ws.update_cell(row_idx, 2, value)
                updated_params.append(f"{param_name}={value}")

        # 1b) Write suggested_var_mult (just below expected_score_r4)
        BASELINE_STD = 2.75
        if sg_result is not None:
            var_mult = round(sg_result["weighted_row"]["actual_std"] / BASELINE_STD, 3)
            param_name = "suggested_var_mult"
            row_idx = param_rows.get(param_name)
            if row_idx is None:
                next_row = len(all_values) + 1
                ws.update_cell(next_row, 1, param_name)
                ws.update_cell(next_row, 2, str(var_mult))
                ws.update_cell(next_row, 3, f"Wt avg actual SG std / {BASELINE_STD} baseline")
                all_values.append([param_name, str(var_mult)])
                updated_params.append(f"{param_name}={var_mult} (new row)")
            else:
                ws.update_cell(row_idx, 2, str(var_mult))
                updated_params.append(f"{param_name}={var_mult}")

        # 1c) Write wind_factor to wind_override (user can override later)
        wo_row = param_rows.get("wind_override")
        if wo_row:
            ws.update_cell(wo_row, 2, str(round(wind_factor, 4)))
            updated_params.append(f"wind_override={wind_factor:.4f}")
        else:
            print("  WARNING: 'wind_override' row not found in Sheet")

        # 2) Write cat_mult_* and cat_skew_* param values to column B
        cells = []

        if sg_cat_result is not None:
            cat_mults = sg_cat_result["category_mults"]
            cat_skew = sg_cat_result.get("category_skew", {})

            for cat_key in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
                short = cat_key.replace("sg_", "")
                mult_row = param_rows.get(f"cat_mult_{short}")
                if mult_row:
                    cells.append(gspread.Cell(row=mult_row, col=2,
                                             value=str(cat_mults.get(cat_key, 1.0))))
                skew_row_idx = param_rows.get(f"cat_skew_{short}")
                if skew_row_idx:
                    cells.append(gspread.Cell(row=skew_row_idx, col=2,
                                             value=str(cat_skew.get(cat_key, 0))))
            blend_note = sg_cat_result.get("blend_note")
            updated_params.append(
                f"cat_mult/skew auto-set from {len(sg_cat_result['per_year'])} year(s)"
                + (f" [{blend_note}]" if blend_note else "")
            )

        # 3) Clear old analysis tables from cols E-S, rows 22-64
        COL_E = 5
        for r in range(22, 65):
            for c in range(COL_E, COL_E + 15):
                cells.append(gspread.Cell(row=r, col=c, value=""))

        # 4) Write pre-tournament expected score breakdown to cols T-Z, rows 2-7
        COL_T = 20  # col T (section label)
        COL_U = 21  # col U
        # Section label
        cells.append(gspread.Cell(row=2, col=COL_T, value="PRE-TOURNEY EXPECTED"))

        # Read dew info from sheet for dew impact computation
        dew_calc = 0.0
        dew_base = 0.0
        dc_row = param_rows.get("dew_calculation")
        if dc_row:
            try:
                dc_val = ws.cell(dc_row, 2).value
                dew_calc = float(dc_val) if dc_val else 0.0
            except (ValueError, TypeError):
                pass
        db_row = param_rows.get("dewpoint_base")
        if db_row:
            try:
                db_val = ws.cell(db_row, 2).value
                dew_base = float(db_val) if db_val else 0.0
            except (ValueError, TypeError):
                pass

        # Read dew arrays from sheet for average dew per round
        all_data = ws.get_all_values()
        dew_cols = {1: 7, 2: 11, 3: 15, 4: 19}  # round -> column (1-indexed)

        # Header row (row 2)
        breakdown_headers = ["Round", "Base Score", "Field Adj", "Wind Impact", "Dew Impact", "= Expected"]
        for ci, h in enumerate(breakdown_headers):
            cells.append(gspread.Cell(row=2, col=COL_U + ci, value=h))

        # Data rows (rows 3-6)
        for rnd in sorted(adjusted.keys()):
            row_num = 2 + rnd  # R1->row3, R2->row4, etc.
            base = final_estimates[rnd]
            this_wk_field = -field_strength
            if rnd >= 3 and cut_adj > 0:
                this_wk_field -= cut_adj
            avg_wind = round_wind_avgs.get(rnd, 0.0)
            wind_impact = avg_wind * wind_factor

            # Compute dew impact: read avg dew for this round from sheet grid
            avg_dew_rd = 0.0
            dew_col_idx = dew_cols.get(rnd)
            if dew_col_idx and len(all_data) >= 17:
                dew_vals = []
                for row_i in range(2, min(17, len(all_data))):  # rows 3-17 (0-indexed 2-16)
                    try:
                        v = float(all_data[row_i][dew_col_idx - 1])  # 0-indexed
                        dew_vals.append(v)
                    except (ValueError, IndexError, TypeError):
                        pass
                if dew_vals:
                    avg_dew_rd = sum(dew_vals) / len(dew_vals)
            dew_impact = (avg_dew_rd - dew_base) * dew_calc if dew_base else 0.0

            expected = round(base + this_wk_field + wind_impact + dew_impact, 1)

            vals = [
                f"R{rnd}",
                round(base, 2),
                round(this_wk_field, 3),
                round(wind_impact, 3),
                round(dew_impact, 3),
                expected,
            ]
            for ci, val in enumerate(vals):
                cells.append(gspread.Cell(row=row_num, col=COL_U + ci, value=val))

        # Coefficients reference row (row 7)
        cells.append(gspread.Cell(row=7, col=COL_U, value="Coefficients:"))
        cells.append(gspread.Cell(row=7, col=COL_U + 1, value=f"wind_factor={wind_factor:.4f}"))
        cells.append(gspread.Cell(row=7, col=COL_U + 2, value=f"dew_coeff={dew_calc:.4f}"))
        cells.append(gspread.Cell(row=7, col=COL_U + 3, value=f"dewpoint_base={dew_base:.1f}"))

        # 5) Actuals section label (row 10)
        cells.append(gspread.Cell(row=10, col=COL_T, value="ACTUALS"))

        # Actuals header (row 10, cols U-AC)
        actuals_headers = ["Round", "Realized Wind", "Forecast Dew", "Realized Dew",
                           "Wind Impact", "Dew Impact", "Expected", "Actual", "Delta"]
        for ci, h in enumerate(actuals_headers):
            cells.append(gspread.Cell(row=10, col=COL_U + ci, value=h))

        ws.update_cells(cells, value_input_option="USER_ENTERED")

        print(f"\n  [round_config] Wrote backup fallbacks: {', '.join(updated_params)}")
        print(f"  [round_config] Wrote pre-tourney breakdown to cols U-Z, rows 2-7")
        print(f"  [round_config] Cleared old analysis tables from cols E-S, rows 22-64")

        # 6) Write analysis tables to Details tab
        _write_details_tab(spreadsheet, table_rows, regression, sg_result,
                           sg_cat_result, field_strength, wind_factor, cut_adj)

    except Exception as e:
        print(f"  WARNING: Could not write to round_config: {e}")


# ===========================================================================
# Interactive Coefficient Approval
# ===========================================================================

def prompt_coefficient_approval(wind_factor, course_wind_effect):
    """Show coefficient review and optionally allow override."""
    print()
    print("=" * 50)
    print("  COEFFICIENT REVIEW")
    print("=" * 50)
    print(f"  Course wind sensitivity (wind_test.csv): {course_wind_effect:.4f}")
    print(f"  Baseline wind (sim_inputs):              {baseline_wind:.4f}")
    print(f"  Blended wind factor:                     {wind_factor:.4f}")
    print(f"  Dew: excluded from baseline (handled by score_adj)")
    print("=" * 50)

    try:
        choice = input("  Proceed? [Y/n/override]: ").strip().lower()
    except EOFError:
        choice = "y"

    if choice == "n":
        print("  Aborted.")
        sys.exit(0)
    elif choice == "override":
        try:
            wind_factor = float(input("  Enter wind factor override: "))
        except (ValueError, EOFError):
            print("  Invalid — keeping original")

    return wind_factor


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    global _BLEND_PREWINDOW_COURSE
    import argparse
    parser = argparse.ArgumentParser(
        description="Scoring baselines + expected scores + variance profiles.")
    blend_grp = parser.add_mutually_exclusive_group()
    blend_grp.add_argument(
        "--blend-prewindow-course", dest="blend_prewindow_course",
        action="store_true",
        help=(f"Force ON: 50/50-blend this venue's own pre-{start_yr} same-course "
              f"variance/skew when the category analysis falls back to event-level "
              f"(e.g. Shinnecock 2018). Default ON for rotating majors "
              f"{sorted(ROTATING_MAJOR_EVENT_IDS)}, OFF otherwise."))
    blend_grp.add_argument(
        "--no-blend-prewindow-course", dest="blend_prewindow_course",
        action="store_false",
        help="Force OFF: always use plain event-level variance (no pre-window blend).")
    parser.set_defaults(blend_prewindow_course=_BLEND_PREWINDOW_COURSE)
    args = parser.parse_args()

    _BLEND_PREWINDOW_COURSE = args.blend_prewindow_course

    print(f"\n{'='*80}")
    print(f"  SCORING BASELINE PIPELINE")
    print(f"  Tournament: {tourney} | Event IDs: {event_ids} | Course ID: {course_id}")
    print(f"  Par: {course_par} | Start Year: {start_yr}")
    print(f"  Pre-window same-course blend: "
          f"{'ON (50/50)' if _BLEND_PREWINDOW_COURSE else 'OFF (event-level)'}")
    print(f"{'='*80}\n")

    # Step 2: Get coordinates
    print("[1/8] Looking up course coordinates...")
    lat, lon = get_course_coordinates(course_id)
    if lat is not None:
        print(f"  Course {course_id}: ({lat}, {lon})")
    else:
        print("  No coordinates found — weather fetch will use fallback only")

    # Step 3: Get tournament dates
    print("\n[2/8] Looking up tournament dates...")
    validate_history_scope(historical_event_ids, start_yr, course_id)
    dates_df = get_tournament_dates(historical_event_ids, start_yr, course_id)
    years = sorted(dates_df["year"].unique())
    print(f"  Found data for {len(years)} years: {years}")

    # Step 4: Fetch/cache weather
    print("\n[3/8] Fetching/caching weather data...")
    event_tag = tourney
    fetch_and_cache_weather(lat, lon, dates_df, event_tag)

    # Load weather averages
    print("\n[4/8] Computing weather averages (7am-7pm)...")
    weather_df = load_weather_averages(event_tag, dates_df)
    if not weather_df.empty:
        print(f"  Weather data for {len(weather_df)} year-rounds")
    else:
        print("  WARNING: No weather averages computed")

    # Step 5: Wind coefficient
    print("\n[5/8] Computing wind coefficient...")
    wind_factor, course_wind_effect = get_wind_coefficient(
        event_ids, baseline_wind, wind_override, course_id
    )
    print(f"  Wind factor: {wind_factor:.4f} (course={course_wind_effect:.4f})")

    # Interactive approval
    wind_factor = prompt_coefficient_approval(wind_factor, course_wind_effect)

    # Step 6: Field strength
    print("\n[6/8] Matching field strength data...")
    mean_skills = get_field_strength(historical_event_ids, start_yr, course_id)
    for yr in sorted(mean_skills.keys()):
        print(f"  {yr}: Mean Skill = {mean_skills[yr]:+.3f}")

    # Step 7: Compute baselines
    print("\n[7/8] Computing scoring baselines...")
    scoring_df = get_scoring_averages(historical_event_ids, start_yr, course_id)
    cuts = detect_cuts(historical_event_ids, start_yr, course_id)
    detail_df, dew_norm = compute_baselines(scoring_df, weather_df, mean_skills, cuts, wind_factor)

    # Step 8: Recency-weighted average
    final_estimates, weights = compute_final_estimates(detail_df)

    # Step 9: Output
    print_report(detail_df, final_estimates, weights, wind_factor, course_wind_effect,
                 dew_norm, course_par)
    save_csv(detail_df, final_estimates, weights)
    save_to_sheets(detail_df, final_estimates, weights)
    write_estimates_to_round_config(final_estimates, wind_factor, detail_df=detail_df)

    print("\n  Done.")


if __name__ == "__main__":
    main()

"""
Google Sheets Config Reader for Live Stats Engine.

Reads round configuration from a Google Sheet so you can update
wind/dew forecasts, round number, and scoring adjustments from
your phone — no terminal access needed.

Sheet layout (tab: "round_config"):
    Column A: Parameter name
    Column B: Value
    Column C: Note (ignored by code)

    Expected parameters:
        round             Integer (0-4). 0 = pre-event. 1-4 = round just completed.
                          (e.g., set to 1 after R1 finishes to predict R2)
        expected_score_1  Numeric. Scoring adjustment for 1st course encountered in API data.
        expected_score_2  Numeric. 2nd course (multi-course only). Leave blank for single-course.
        expected_score_3  Numeric. 3rd course (multi-course only). Leave blank for single-course.
        wind              Comma-separated hourly wind array (6AM onward).
        dew               Comma-separated hourly dewpoint array (6AM onward).
        wind_paste        (Optional) Alternate wind array (e.g. for second forecast).
        dew_paste          (Optional) Alternate dew array.
        dew_calculation   Numeric. Dew effect factor. Falls back to sim_inputs if blank.
        wind_override     Numeric. 0 = use computed. Falls back to sim_inputs if blank.

        # NEW fields for tournament sim:
        course_codes      Comma-separated course codes from API (e.g. "TS" or "PB,SG").
                          Auto-populated by update_sheet_courses.py.
        course_pars       Comma-separated par values matching course_codes order.
        course_lat_lon    Course latitude/longitude as "lat,lon".
        course_timezone   IANA timezone resolved from the course coordinates.
        expected_score_r1 Expected scoring avg for R1. Pre-tourney estimate (backup fallback).
        expected_score_r2 Expected scoring avg for R2. Multi-course: comma-separated.
        expected_score_r3 Expected scoring avg for R3.
        expected_score_r4 Expected scoring avg for R4.
        wind_r1           Hourly wind for R1. Authoritative; set by humidity.py.
        wind_r2           Hourly wind for R2. Authoritative; set by humidity.py.
        wind_r3           Hourly wind for R3. Authoritative; set by humidity.py.
        wind_r4           Hourly wind for R4. Authoritative; set by humidity.py.
        dew_r1            Hourly dew for R1. Authoritative; set by humidity.py.
        dew_r2            Hourly dew for R2. Authoritative; set by humidity.py.
        dew_r3            Hourly dew for R3. Authoritative; set by humidity.py.
        dew_r4            Hourly dew for R4. Authoritative; set by humidity.py.
        wind / dew        LIVE-round quick-override channel: during live play
                          (round > 0) a non-empty value REPLACES the next
                          round's wind_r{N}/dew_r{N} in the engine. Ignored at
                          round 0 and blanked by reset_for_new_week() — a
                          leftover from last week must never reach a new
                          event's R1 (2026-08-12).

Authentication:
    Place credentials.json (Google service account key) in the project root.
    Share the Google Sheet with the service account email from credentials.json.

Usage:
    from sheet_config import load_config
    config = load_config()
    # config['round_num'] → 2
    # config['wind'] → [5, 5, 5, 5, ...]
    # config['dew'] → [36, 36, 38, ...]
    # config['course_codes'] → ['TS'] or ['PB', 'SG']
    # config['course_pars'] → [72] or [72, 72]
"""

import os
import json
import time
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv  

# Load environment variables
load_dotenv() 

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update these to match your setup
# ══════════════════════════════════════════════════════════════════════════════

# Google Sheet name (the title shown in the browser tab)
SHEET_NAME = "golf_sims"

# Tab within the sheet
TAB_NAME = "round_config"

# Path to service account credentials
# Looks in project root first, then current directory
CREDENTIALS_PATHS = [
    "credentials.json",
    os.path.join(os.path.dirname(__file__), "credentials.json"),
]

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive',
]

_RETRYABLE_SHEET_STATUS_CODES = {429, 500, 502, 503, 504}
# Four retries wait 75 seconds in total, long enough to cross the Sheets
# per-user, per-minute quota window that can be exhausted by the live pipeline.
_SHEET_RETRY_DELAYS_SECONDS = (5, 10, 20, 40)


# ══════════════════════════════════════════════════════════════════════════════
# Sheet Reader
# ══════════════════════════════════════════════════════════════════════════════

def _find_credentials():
    """Locate credentials.json file."""
    for path in CREDENTIALS_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "credentials.json not found. Place your Google service account key "
        "in the project root directory."
    )


def _sheet_error_code(exc):
    """Return an HTTP-like status code from a gspread/requests exception."""
    code = getattr(exc, "code", None)
    if code is None:
        error = getattr(exc, "error", None)
        if isinstance(error, dict):
            code = error.get("code")
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _retry_sheet_request(operation, label, delays=_SHEET_RETRY_DELAYS_SECONDS):
    """Retry transient Sheets API failures, honoring Retry-After when present."""
    attempts = len(delays) + 1
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            code = _sheet_error_code(exc)
            if code not in _RETRYABLE_SHEET_STATUS_CODES or attempt == attempts - 1:
                raise

            wait = float(delays[attempt])
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                try:
                    retry_after = float(headers.get("Retry-After", 0))
                    wait = min(max(wait, retry_after), 120.0)
                except (TypeError, ValueError):
                    pass

            print(
                f"[sheet_config] {label} failed with HTTP {code}; "
                f"retry in {wait:g}s ({attempt + 2}/{attempts})"
            )
            time.sleep(wait)


def _connect_sheet():
    """Authenticate and return the worksheet."""
    # Try environment variable first (GitHub Actions or .env file)
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    
    if creds_json:
        # Load credentials from environment variable
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    else:
        # Fallback to credentials.json file (backwards compatibility)
        creds_path = _find_credentials()
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    
    def open_worksheet():
        client = gspread.authorize(creds)
        spreadsheet = client.open(SHEET_NAME)
        return spreadsheet.worksheet(TAB_NAME)

    return _retry_sheet_request(open_worksheet, "connect")


def _parse_array(value_str):
    """
    Parse a comma-separated string into a list of floats.
    Handles both '5,5,5,5' and '5, 5, 5, 5' formats.
    Returns empty list if parsing fails.
    """
    if not value_str or str(value_str).strip() == "":
        return []
    try:
        return [float(x.strip()) for x in str(value_str).split(",") if x.strip()]
    except (ValueError, TypeError):
        print(f"  Warning: Could not parse array: '{value_str}'")
        return []


def _parse_numeric(value_str, default=None):
    """Parse a numeric value, returning default if blank or invalid."""
    if value_str is None or str(value_str).strip() == "":
        return default
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return default


def _parse_string_array(value_str):
    """
    Parse a comma-separated string into a list of strings.
    Returns empty list if blank.
    """
    if not value_str or str(value_str).strip() == "":
        return []
    return [x.strip() for x in str(value_str).split(",") if x.strip()]


def _parse_multi_numeric(value_str):
    """
    Parse a comma-separated string into a list of floats.
    Returns empty list if blank, single value if one number.
    """
    if not value_str or str(value_str).strip() == "":
        return []
    try:
        return [float(x.strip()) for x in str(value_str).split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def _parse_course_lat_lon(value_str):
    """Parse and range-check a ``lat,lon`` Sheet value."""
    if not value_str or str(value_str).strip() == "":
        return None, None
    try:
        parts = [float(part.strip()) for part in str(value_str).split(",")]
    except (TypeError, ValueError):
        return None, None
    if len(parts) != 2:
        return None, None
    latitude, longitude = parts
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None, None
    return latitude, longitude


_SUMMARY_PRINTED = False  # full config block prints once per process; one-liner after


def get_param(name, default=None):
    """Read a single round_config parameter (Column A name -> Column B value).

    Generic accessor for flags like 'maker_enabled' without extending
    load_config (which only parses the known sim fields). Returns the trimmed
    string value, or `default` if the row is absent/blank. Raises if the sheet
    can't be reached — callers needing fail-soft behavior should wrap the call.
    """
    ws = _connect_sheet()
    key = str(name).strip().lower()
    for row in ws.get("A:B")[1:]:  # skip header
        if row and row[0].strip().lower() == key:
            val = row[1].strip() if len(row) > 1 else ""
            return val if val != "" else default
    return default


def load_config(verbose=True):
    """
    Read the round_config tab and return a config dictionary.

    Returns:
        dict with keys:
            round_num:        int (0-4) — round that just completed
            wind:             list[float] — hourly wind array for NEXT round
            dew:              list[float] — hourly dewpoint array for NEXT round
            expected_score_1: float — scoring adjustment, course 1
            expected_score_2: float or None — scoring adjustment, course 2
            expected_score_3: float or None — scoring adjustment, course 3
            dew_calculation:  float or None — overrides sim_inputs if set
            wind_override:    float or None — overrides sim_inputs if set
            pre_event:        bool — True if round_num is 0

            # NEW for tournament sim:
            course_codes:     list[str] — course codes from API (e.g. ['TS'] or ['PB','SG'])
            course_pars:      list[float] — par values matching course_codes order
            expected_score_r1: list[float] — expected scoring avg for R1 (pre-tourney estimate)
            expected_score_r2: list[float] — expected scoring avg for R2 (per-course if multi)
            expected_score_r3: list[float] — expected scoring avg for R3
            expected_score_r4: list[float] — expected scoring avg for R4
            wind_r1:          list[float] — hourly wind for R1 (authoritative, from humidity.py)
            wind_r2:          list[float] — hourly wind for R2 (authoritative, from humidity.py)
            wind_r3:          list[float] — hourly wind for R3 (authoritative, from humidity.py)
            wind_r4:          list[float] — hourly wind for R4 (authoritative, from humidity.py)
            dew_r1:           list[float] — hourly dew for R1 (authoritative, from humidity.py)
            dew_r2:           list[float] — hourly dew for R2 (authoritative, from humidity.py)
            dew_r3:           list[float] — hourly dew for R3 (authoritative, from humidity.py)
            dew_r4:           list[float] — hourly dew for R4 (authoritative, from humidity.py)
            wind / dew:       list[float] — live-round override; replaces the NEXT
                              round's array only when round > 0 (see
                              live_stats_engine._apply_sheet_overrides)
    """
    print("Reading config from Google Sheet...")
    ws = _connect_sheet()

    # Read all rows from columns A and B
    # Returns list of lists: [['Parameter', 'Value'], ['round', '1'], ...]
    all_values = _retry_sheet_request(
        lambda: ws.get("A:B"),
        "read round_config",
    )

    # Build param → value dict (skip header row)
    params = {}
    for row in all_values[1:]:  # skip header
        if len(row) >= 2 and row[0].strip():
            params[row[0].strip().lower()] = row[1].strip() if len(row) > 1 else ""
        elif len(row) == 1 and row[0].strip():
            params[row[0].strip().lower()] = ""

    # --- Parse individual fields ---
    round_num = int(_parse_numeric(params.get("round"), default=0))

    wind = _parse_array(params.get("wind", ""))
    dew = _parse_array(params.get("dew", ""))

    # Scoring adjustments (course 1 required, 2 & 3 optional for multi-course)
    expected_score_1 = _parse_numeric(params.get("expected_score_1"), default=None)
    expected_score_2 = _parse_numeric(params.get("expected_score_2"), default=None)
    expected_score_3 = _parse_numeric(params.get("expected_score_3"), default=None)

    # Weather overrides (default 0.0 — no sim_inputs fallback)
    dew_calculation = _parse_numeric(params.get("dew_calculation"), default=0.0)
    wind_override = _parse_numeric(params.get("wind_override"), default=0.0)

    # --- V2 sim fields ---
    tourney = params.get("tourney", "").strip()
    event_id = int(_parse_numeric(params.get("event_id"), default=0))
    course_id = int(_parse_numeric(params.get("course_id"), default=0))
    cut_line = int(_parse_numeric(params.get("cut_line"), default=50))
    use_10_shot_rule = bool(int(_parse_numeric(params.get("use_10_shot_rule"), default=1)))
    simulations = int(_parse_numeric(params.get("simulations"), default=100000))
    std_dev = _parse_numeric(params.get("std_dev"), default=3.02)

    # Per-category course variance multipliers (default 1.0 = no scaling)
    course_cat_mults = {
        "sg_ott": _parse_numeric(params.get("cat_mult_ott"), default=1.0),
        "sg_app": _parse_numeric(params.get("cat_mult_app"), default=1.0),
        "sg_arg": _parse_numeric(params.get("cat_mult_arg"), default=1.0),
        "sg_putt": _parse_numeric(params.get("cat_mult_putt"), default=1.0),
    }

    # Per-category course skewness (None = use tour-wide baseline in V2)
    _raw_skew = {
        "sg_ott": _parse_numeric(params.get("cat_skew_ott"), default=None),
        "sg_app": _parse_numeric(params.get("cat_skew_app"), default=None),
        "sg_arg": _parse_numeric(params.get("cat_skew_arg"), default=None),
        "sg_putt": _parse_numeric(params.get("cat_skew_putt"), default=None),
    }
    # Only include non-None values; empty dict triggers tour-wide baseline
    course_cat_skew = {k: v for k, v in _raw_skew.items() if v is not None}

    # --- NEW fields for tournament sim ---
    course_codes = _parse_string_array(params.get("course_codes", ""))
    course_pars = _parse_multi_numeric(params.get("course_pars", ""))
    course_latitude, course_longitude = _parse_course_lat_lon(
        params.get("course_lat_lon", "")
    )
    course_timezone = params.get("course_timezone", "").strip() or None

    # Per-round expected scoring averages (can be multi-value for multi-course)
    expected_score_r1 = _parse_multi_numeric(params.get("expected_score_r1", ""))
    expected_score_r2 = _parse_multi_numeric(params.get("expected_score_r2", ""))
    expected_score_r3 = _parse_multi_numeric(params.get("expected_score_r3", ""))
    expected_score_r4 = _parse_multi_numeric(params.get("expected_score_r4", ""))

    # Per-round wind/dew arrays (empty = use default 'wind'/'dew')
    wind_r1 = _parse_array(params.get("wind_r1", ""))
    wind_r2 = _parse_array(params.get("wind_r2", ""))
    wind_r3 = _parse_array(params.get("wind_r3", ""))
    wind_r4 = _parse_array(params.get("wind_r4", ""))
    dew_r1 = _parse_array(params.get("dew_r1", ""))
    dew_r2 = _parse_array(params.get("dew_r2", ""))
    dew_r3 = _parse_array(params.get("dew_r3", ""))
    dew_r4 = _parse_array(params.get("dew_r4", ""))

    # Manual boosts: "player, name:0.35, other, player:-0.10"
    _raw_boosts = params.get("manual_boosts", "").strip()
    manual_boosts = {}
    if _raw_boosts:
        for chunk in _raw_boosts.split("|"):
            chunk = chunk.strip()
            if ":" not in chunk:
                continue
            # Last colon separates name from value (names can contain commas)
            idx = chunk.rfind(":")
            name = chunk[:idx].strip().lower()
            try:
                manual_boosts[name] = float(chunk[idx+1:].strip())
            except ValueError:
                pass

    # DG override players: "player, name | other, player"
    _raw_overrides = params.get("dg_override_players", "").strip()
    dg_override_players = []
    if _raw_overrides:
        dg_override_players = [p.strip().lower() for p in _raw_overrides.split("|") if p.strip()]

    # Archetype boosts: archetype_boost_<type> → float SG adjustment
    ARCHETYPE_KEYS = [
        "stud", "ball_striker", "long_wild", "long_accurate",
        "short_accurate", "short_game_specialist", "elite_putter",
        "low_skill", "balanced",
    ]
    archetype_boosts = {}
    for key in ARCHETYPE_KEYS:
        val = _parse_numeric(params.get(f"archetype_boost_{key}"), default=None)
        if val is not None and val != 0:
            # Convert key back to display label: "ball_striker" → "Ball Striker"
            label = key.replace("_", " ").title()
            if label == "Short Game Specialist":
                label = "Short Game Specialist"  # already correct from title()
            archetype_boosts[label] = val

    # Realized wind (manual entry) and dewpoint base (from humidity.py)
    realized_wind_r1 = _parse_numeric(params.get("realized_wind_r1"), default=None)
    realized_wind_r2 = _parse_numeric(params.get("realized_wind_r2"), default=None)
    realized_wind_r3 = _parse_numeric(params.get("realized_wind_r3"), default=None)
    realized_wind_r4 = _parse_numeric(params.get("realized_wind_r4"), default=None)
    dewpoint_base = _parse_numeric(params.get("dewpoint_base"), default=None)

    config = {
        "round_num": round_num,
        "pre_event": round_num == 0,
        "wind": wind,
        "dew": dew,
        "expected_score_1": expected_score_1,
        "expected_score_2": expected_score_2,
        "expected_score_3": expected_score_3,
        "dew_calculation": dew_calculation,
        "wind_override": wind_override,
        # Course / tournament fields
        "course_codes": course_codes,
        "course_pars": course_pars,
        "course_latitude": course_latitude,
        "course_longitude": course_longitude,
        "course_timezone": course_timezone,
        "expected_score_r1": expected_score_r1,
        "expected_score_r2": expected_score_r2,
        "expected_score_r3": expected_score_r3,
        "expected_score_r4": expected_score_r4,
        "wind_r1": wind_r1,
        "wind_r2": wind_r2,
        "wind_r3": wind_r3,
        "wind_r4": wind_r4,
        "dew_r1": dew_r1,
        "dew_r2": dew_r2,
        "dew_r3": dew_r3,
        "dew_r4": dew_r4,
        # V2 sim fields
        "tourney": tourney,
        "event_id": event_id,
        "course_id": course_id,
        "cut_line": cut_line,
        "use_10_shot_rule": use_10_shot_rule,
        "simulations": simulations,
        "std_dev": std_dev,
        "course_cat_mults": course_cat_mults,
        "course_cat_skew": course_cat_skew,
        # Realized weather + dewpoint base
        "realized_wind_r1": realized_wind_r1,
        "realized_wind_r2": realized_wind_r2,
        "realized_wind_r3": realized_wind_r3,
        "realized_wind_r4": realized_wind_r4,
        "dewpoint_base": dewpoint_base,
        # Player adjustments
        "manual_boosts": manual_boosts,
        "dg_override_players": dg_override_players,
        "archetype_boosts": archetype_boosts,
    }

    # --- Print summary (full block ONCE per process; a one-liner thereafter so
    # repeated load_config() calls within a run don't re-spam the whole config) ---
    global _SUMMARY_PRINTED
    if verbose and not _SUMMARY_PRINTED:
        _SUMMARY_PRINTED = True
        print(f"  Round:    {round_num} ({'pre-event' if round_num == 0 else f'R{round_num} complete'})")
        if tourney:
            print(f"  Tourney:  {tourney} (event_id={event_id}, course_id={course_id})")
        print(f"  Wind:     {len(wind)} hours -> {wind[:5]}{'...' if len(wind) > 5 else ''}")
        print(f"  Dew:      {len(dew)} hours -> {dew[:5]}{'...' if len(dew) > 5 else ''}")
        print(f"  Score adj: {expected_score_1}"
              + (f" / {expected_score_2}" if expected_score_2 is not None else "")
              + (f" / {expected_score_3}" if expected_score_3 is not None else ""))
        if course_codes:
            print(f"  Courses:  {course_codes} (pars: {course_pars})")
        if expected_score_r1:
            print(f"  Exp R1:   {expected_score_r1}")
        if expected_score_r2:
            print(f"  Exp R2:   {expected_score_r2}")
        if expected_score_r3:
            print(f"  Exp R3:   {expected_score_r3}")
        if expected_score_r4:
            print(f"  Exp R4:   {expected_score_r4}")
        print(f"  Sim:      {simulations} sims, std_dev={std_dev}, cut={cut_line}, wind_coeff={wind_override}")
        print(f"  Cat mults: " + " | ".join(f"{k.replace('sg_','').upper()}={v}" for k, v in course_cat_mults.items()))
        if course_cat_skew:
            print(f"  Cat skew:  " + " | ".join(f"{k.replace('sg_','').upper()}={v}" for k, v in course_cat_skew.items()))
        if archetype_boosts:
            print(f"  Arch boost: " + " | ".join(f"{n}: {v:+.2f}" for n, v in archetype_boosts.items()))
        if manual_boosts:
            print(f"  Boosts:    " + " | ".join(f"{n}: {v:+.2f}" for n, v in manual_boosts.items()))
        if dg_override_players:
            print(f"  DG overrides: {', '.join(dg_override_players)}")
    elif verbose:
        print(f"  Config: {tourney} R{round_num} (event {event_id}) [summary already shown above]")

    return config


# ══════════════════════════════════════════════════════════════════════════════
# Sheet Writer (for nightly backup)
# ══════════════════════════════════════════════════════════════════════════════

def write_primary_fields(round_num, expected_score_1, wind_str, dew_str):
    """
    Write primary config fields to the round_config tab.

    Used by nightly_round_sim.py to update the Sheet from per-round fallback
    values so that live_stats_engine.py and round_sim.py read correct config.

    Args:
        round_num: int (0-4) — round that just completed
        expected_score_1: float — scoring avg for primary course
        wind_str: str — comma-separated hourly wind array
        dew_str: str — comma-separated hourly dew array
    """
    ws = _connect_sheet()

    # Read all rows to find parameter positions
    all_values = ws.get("A:B")

    # Build param_name -> row_index mapping (1-indexed for gspread)
    param_rows = {}
    for i, row in enumerate(all_values):
        if row and row[0].strip():
            param_rows[row[0].strip().lower()] = i + 1  # gspread is 1-indexed

    updates = {
        "round": str(round_num),
        "expected_score_1": str(expected_score_1),
        "wind": wind_str,
        "dew": dew_str,
    }

    cells_updated = []
    for param, value in updates.items():
        row_idx = param_rows.get(param)
        if row_idx is None:
            print(f"  WARNING: '{param}' not found in Sheet, skipping")
            continue
        ws.update_cell(row_idx, 2, value)  # Column B = 2
        cells_updated.append(param)

    print(f"  Updated Sheet primary fields: {', '.join(cells_updated)}")


def reset_for_new_week():
    """
    Reset round_config tab for a new tournament week.

    Reads per-week config from sim_inputs.py and writes it to the Sheet:
      round       -> 0
      tourney     -> sim_inputs.tourney
      event_id    -> sim_inputs.event_ids[0]
      course_id   -> sim_inputs.course_id
      cut_line    -> sim_inputs.CUT_LINE  (if defined)
      course_pars -> sim_inputs.course_par (if defined; comma-joined when list)

    Call this at the start of each week before running humidity.py /
    scoring_baseline.py / etc.
    """
    from sim_inputs import tourney, event_ids, course_id

    ws = _connect_sheet()
    all_values = ws.get("A:B")

    param_rows = {}
    for i, row in enumerate(all_values):
        if row and row[0].strip():
            param_rows[row[0].strip().lower()] = i + 1

    updates = {
        "round": "0",
        "tourney": tourney,
        "event_id": str(event_ids[0]),
        "course_id": str(course_id),
    }

    # Optional: push cut_line and course_pars when present in sim_inputs so
    # they don't carry over from the prior week.
    try:
        from sim_inputs import CUT_LINE
        updates["cut_line"] = str(int(CUT_LINE))
    except ImportError:
        pass
    try:
        from sim_inputs import course_par
        if isinstance(course_par, (list, tuple)):
            updates["course_pars"] = ",".join(str(int(p)) for p in course_par)
        else:
            updates["course_pars"] = str(int(course_par))
    except ImportError:
        pass

    # Prior-week realized weather must not survive the reset: a leftover
    # realized_wind_r{N} skips the fresh Open-Meteo fetch, corrupting the
    # actuals delta the expected-score difficulty feedback consumes.
    for rnd in range(1, 5):
        for prefix in ("realized_wind_r", "realized_dew_r"):
            if f"{prefix}{rnd}" in param_rows:
                updates[f"{prefix}{rnd}"] = ""

    # The generic wind/dew rows are the LIVE-round quick-override channel; a
    # leftover from last week's live play must not survive into the new week
    # (2026-08-12: a stale ~3-4mph generic row would have replaced the real
    # 7-9mph R1 forecast at round 0 — belt-and-braces with the round_num > 0
    # guard in live_stats_engine._apply_sheet_overrides).
    for key in ("wind", "dew"):
        if key in param_rows:
            updates[key] = ""

    cells_updated = []
    for param, value in updates.items():
        row_idx = param_rows.get(param)
        if row_idx is None:
            print(f"  WARNING: '{param}' not found in Sheet, adding at bottom")
            next_row = len(all_values) + 1
            ws.update_cell(next_row, 1, param)
            ws.update_cell(next_row, 2, value)
            cells_updated.append(f"{param} (new row)")
        else:
            ws.update_cell(row_idx, 2, value)
            cells_updated.append(param)

    # Clear the per-round actuals block (rows 11-14, cols U-AC): the new
    # week's rows are written round by round, and the difficulty feedback
    # reads deltas from here — stale rows from last week must never be
    # readable as this week's evidence. Skipped on a mid-tournament rerun of
    # the SAME event (config touch-up), where the rows are current evidence.
    prior = {
        row[0].strip().lower(): (row[1] if len(row) > 1 else "")
        for row in all_values
        if row and row[0].strip()
    }
    try:
        same_event_mid_round = (
            str(prior.get("event_id", "")).strip() == str(event_ids[0])
            and float(prior.get("round") or 0) >= 1
        )
    except (TypeError, ValueError):
        same_event_mid_round = False
    if same_event_mid_round:
        print("  [reset] mid-tournament rerun — keeping this week's actuals block")
    else:
        try:
            ws.batch_clear(["U11:AC14"])
            cells_updated.append("actuals block (U11:AC14)")
        except Exception as exc:
            print(f"  WARNING: could not clear actuals block: {exc}")

    print(f"  [reset] Sheet updated for new week: {', '.join(cells_updated)}")
    summary_extra = []
    if "cut_line" in updates:
        summary_extra.append(f"cut_line={updates['cut_line']}")
    if "course_pars" in updates:
        summary_extra.append(f"course_pars={updates['course_pars']}")
    extra_str = (", " + ", ".join(summary_extra)) if summary_extra else ""
    print(f"  [reset] tourney={tourney}, event_id={event_ids[0]}, course_id={course_id}, round=0{extra_str}")


def write_sim_config(cat_mults=None, cat_skew=None):
    """
    Write category variance multipliers and/or skewness to round_config tab.

    Used by scoring_baseline.py to auto-populate cat_mult_* and cat_skew_*
    rows after computing per-category variance analysis.

    Args:
        cat_mults: dict like {'sg_ott': 1.27, 'sg_app': 1.12, ...}
        cat_skew:  dict like {'sg_ott': -1.14, 'sg_app': -0.38, ...}
    """
    ws = _connect_sheet()
    all_values = ws.get("A:B")

    param_rows = {}
    for i, row in enumerate(all_values):
        if row and row[0].strip():
            param_rows[row[0].strip().lower()] = i + 1

    cells_updated = []

    if cat_mults:
        for cat_key, value in cat_mults.items():
            short = cat_key.replace("sg_", "")
            param = f"cat_mult_{short}"
            row_idx = param_rows.get(param)
            if row_idx:
                ws.update_cell(row_idx, 2, str(round(value, 3)))
                cells_updated.append(f"{param}={value:.3f}")
            else:
                print(f"  WARNING: '{param}' not found in Sheet, skipping")

    if cat_skew:
        for cat_key, value in cat_skew.items():
            short = cat_key.replace("sg_", "")
            param = f"cat_skew_{short}"
            row_idx = param_rows.get(param)
            if row_idx:
                ws.update_cell(row_idx, 2, str(round(value, 3)))
                cells_updated.append(f"{param}={value:.3f}")
            else:
                print(f"  WARNING: '{param}' not found in Sheet, skipping")

    if cells_updated:
        print(f"  Updated Sheet sim config: {', '.join(cells_updated)}")
    else:
        print("  No sim config fields to update")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = load_config()
    print("\nFull config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

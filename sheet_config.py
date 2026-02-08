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

Authentication:
    Place credentials.json (Google service account key) in the project root.
    Share the Google Sheet with the service account email from credentials.json.

Usage:
    from sheet_config import load_config
    config = load_config()
    # config['round_num'] → 2
    # config['wind'] → [5, 5, 5, 5, ...]
    # config['dew'] → [36, 36, 38, ...]
"""

import os
import json
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
    
    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_NAME)
    worksheet = spreadsheet.worksheet(TAB_NAME)
    return worksheet


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


def load_config():
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
    """
    print("📊 Reading config from Google Sheet...")
    ws = _connect_sheet()

    # Read all rows from columns A and B
    # Returns list of lists: [['Parameter', 'Value'], ['round', '1'], ...]
    all_values = ws.get("A:B")

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
    expected_score_1 = _parse_numeric(params.get("expected_score_1"), default=0)
    expected_score_2 = _parse_numeric(params.get("expected_score_2"), default=None)
    expected_score_3 = _parse_numeric(params.get("expected_score_3"), default=None)

    # Optional overrides (fall back to sim_inputs if not set)
    dew_calculation = _parse_numeric(params.get("dew_calculation"), default=None)
    wind_override = _parse_numeric(params.get("wind_override"), default=None)

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
    }

    # --- Print summary ---
    print(f"  Round:    {round_num} ({'pre-event' if round_num == 0 else f'R{round_num} complete'})")
    print(f"  Wind:     {len(wind)} hours → {wind[:5]}{'...' if len(wind) > 5 else ''}")
    print(f"  Dew:      {len(dew)} hours → {dew[:5]}{'...' if len(dew) > 5 else ''}")
    print(f"  Score adj: {expected_score_1}"
          + (f" / {expected_score_2}" if expected_score_2 is not None else "")
          + (f" / {expected_score_3}" if expected_score_3 is not None else ""))

    return config


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = load_config()
    print("\nFull config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
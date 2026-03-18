"""
sheets_storage.py — Google Sheets storage for golf sim bet records.

Writes bet records to three tabs in the "golf_sims" spreadsheet:
  - Tournament Matchups   (from new_sim.py → combined_df)
  - Finish Positions       (from new_sim.py → combined_finish_df)
  - Round Matchups         (from round_sim.py → combined df)

Auth:
  Uses the same credential pattern as sheet_config.py:
  - GOOGLE_CREDS_JSON env var (for GitHub Actions / CI)
  - credentials.json file (for local dev)

Usage:
  from sheets_storage import (
      is_valid_run_time,
      store_tournament_matchups,
      store_finish_positions,
      store_round_matchups,
  )
"""

import os
import json
import uuid
import time
import tempfile
from datetime import datetime

import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SHEET_NAME = "golf_sims"

# Write-capable scopes (superset of sheet_config's readonly scopes)
WRITE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CREDENTIALS_PATHS = [
    "credentials.json",
    os.path.join(os.path.dirname(__file__), "credentials.json"),
]

# Tab names
TAB_TOURNAMENT_MU = "Tournament Matchups"
TAB_FINISH_POS = "Finish Positions"
TAB_ROUND_MU = "Round Matchups"
TAB_BASE_RATES = "Base Rates"
TAB_LIVE = "Live"
TAB_DETAILS = "Details"

DETAILS_HEADERS = ["Section", "Label", "Value", "Notes"]

# Parquet ledger (local-only, not committed to repo)
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "permanent_data", "bet_ledger.parquet")

# Dedup key columns for the ledger
LEDGER_DEDUP_COLS = ["event_id", "bet_type", "round", "bet_on", "opponent", "bookmaker"]


# ══════════════════════════════════════════════════════════════════════════════
# Headers — define column order for each tab
# ══════════════════════════════════════════════════════════════════════════════

TOURNAMENT_MU_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id",
    "player_1", "player_2", "dg_id_p1", "dg_id_p2",
    "bookmaker", "ties_rule",
    "p1_odds", "p2_odds", "fair_p1", "fair_p2",
    "edge_p1", "edge_p2",
    "bet_on", "edge_on", "pred_on", "pred_against", "sample_on",
    "half_shot_p1", "half_shot_p2",
    "wind_on", "wind_diff", "wx_edge",
    "result", "units_won",
]

FINISH_POS_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id",
    "player_name", "dg_id",
    "market_type", "sportsbook",
    "decimal_odds", "american_odds", "my_fair",
    "sim_prob", "edge", "kelly_stake",
    "my_pred", "sample",
    "result", "actual_finish", "units_won",
]

BASE_RATES_HEADERS = ["category", "parameter", "base_rate", "this_week", "delta", "notes"]

ROUND_MU_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id", "round",
    "player_1", "player_2", "dg_id_p1", "dg_id_p2",
    "bookmaker", "ties_rule",
    "p1_odds", "p2_odds", "fair_p1", "fair_p2",
    "edge_p1", "edge_p2",
    "bet_on", "edge_on", "pred_on", "pred_against", "sample_on",
    "half_shot_p1", "half_shot_p2",
    "wx_edge",
    "result", "p1_round_score", "p2_round_score", "units_won",
]


# ══════════════════════════════════════════════════════════════════════════════
# Auth & Connection
# ══════════════════════════════════════════════════════════════════════════════

def _get_credentials():
    """Build Google credentials from env var or file."""
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        return Credentials.from_service_account_info(creds_dict, scopes=WRITE_SCOPES)

    for path in CREDENTIALS_PATHS:
        if os.path.exists(path):
            return Credentials.from_service_account_file(path, scopes=WRITE_SCOPES)

    raise FileNotFoundError(
        "No Google credentials found. Set GOOGLE_CREDS_JSON env var or place "
        "credentials.json in the project root."
    )


def _connect_sheets(max_retries=3, base_delay=5):
    """Authenticate and return the gspread Spreadsheet object.

    Retries on transient errors (503, 429, network) with exponential backoff.
    """
    creds = _get_credentials()
    client = gspread.authorize(creds)
    for attempt in range(1, max_retries + 1):
        try:
            return client.open(SHEET_NAME)
        except gspread.exceptions.APIError as e:
            code = e.response.status_code if hasattr(e, 'response') else None
            if code in (503, 429, 500) and attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"  [storage] Google API {code}, retrying in {wait}s ({attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                raise



# ══════════════════════════════════════════════════════════════════════════════
# Connection Pooling
# ══════════════════════════════════════════════════════════════════════════════

_SPREADSHEET_CACHE = None


def get_spreadsheet():
    """Return a cached gspread Spreadsheet object (single auth per process)."""
    global _SPREADSHEET_CACHE
    if _SPREADSHEET_CACHE is None:
        _SPREADSHEET_CACHE = _connect_sheets()
        print("  [storage] Authenticated with Google Sheets (cached)")
    return _SPREADSHEET_CACHE


# ══════════════════════════════════════════════════════════════════════════════
# Tab Management
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_create_tab(spreadsheet, tab_name, headers):
    """
    Get an existing tab or create it with header row.
    If the tab exists but its header row is missing columns from `headers`,
    prints a warning so the mismatch is caught early.
    Returns the gspread Worksheet.
    """
    try:
        ws = spreadsheet.worksheet(tab_name)
        # Check for header drift — warn if existing tab is missing expected columns
        existing_headers = ws.row_values(1)
        missing = [h for h in headers if h not in existing_headers]
        if missing:
            print(f"  [storage] WARNING: '{tab_name}' header is missing columns: {missing}")
            print(f"  [storage]   Expected {len(headers)} cols, found {len(existing_headers)}")
            print(f"  [storage]   Run fix_tournament_mu_headers.py to repair")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(
            title=tab_name, rows=1000, cols=len(headers)
        )
        ws.append_row(headers, value_input_option="RAW")
        print(f"  [storage] Created tab '{tab_name}' with {len(headers)} columns")
    return ws


def _append_rows(ws, rows):
    """Append a list of row-lists to the worksheet."""
    if not rows:
        return
    ws.append_rows(rows, value_input_option="USER_ENTERED")


# ══════════════════════════════════════════════════════════════════════════════
# Time Gate
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_run_time():
    """
    Only store runs executed after 3 PM EST on Monday of tournament week.
    Monday before 3 PM is pre-field — too early for reliable odds.
    All other times (Tue-Sun) are valid, including Sunday for R4 storage.
    """
    try:
        import pytz
        est = pytz.timezone("US/Eastern")
    except ImportError:
        from zoneinfo import ZoneInfo
        est = ZoneInfo("US/Eastern")

    now = datetime.now(est)

    # Monday (weekday 0) before 3 PM → too early
    if now.weekday() == 0 and now.hour < 15:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now_est_iso():
    """Current timestamp in EST, ISO 8601 format."""
    try:
        import pytz
        est = pytz.timezone("US/Eastern")
    except ImportError:
        from zoneinfo import ZoneInfo
        est = ZoneInfo("US/Eastern")
    return datetime.now(est).strftime("%Y-%m-%d %H:%M:%S")


def _safe(val, round_digits=None):
    """
    Safely convert a value for Sheets storage.
    Handles NaN, None, numpy types → clean Python primitives.
    """
    if val is None:
        return ""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return ""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return ""
        if round_digits is not None:
            return round(val, round_digits)
        return val
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if round_digits is not None and isinstance(val, float):
        return round(val, round_digits)
    return val


def _get(row, *col_names, default=""):
    """
    Get the first matching column value from a DataFrame row (Series).
    Tries each col_name in order; returns default if none found or all NaN.
    """
    for col in col_names:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                return v
    return default


# ══════════════════════════════════════════════════════════════════════════════
# Store: Tournament Matchups
# ══════════════════════════════════════════════════════════════════════════════

def store_tournament_matchups(combined_df, tourney, event_id, dg_id_lookup=None, spreadsheet=None):
    """
    Write tournament matchup rows to the "Tournament Matchups" tab.

    Args:
        combined_df:    Filtered matchup DataFrame from new_sim.py
                        Expected columns: Player 1, Player 2, Bookmaker, Ties,
                        P1 Odds, P2 Odds, Fair_p1, Fair_p2, edge_p1, edge_p2,
                        edge_on, bet_on, pred_on, pred_against, sample_on
                        Optional: half_shot_p1, half_shot_p2, wind_on, wind_diff
        tourney:        Tournament name string (e.g. 'farmers')
        event_id:       DataGolf event ID (e.g. '014')
        dg_id_lookup:   Optional dict {player_name: dg_id} for grading joins
    """
    if combined_df is None or combined_df.empty:
        print("  [storage] No tournament matchups to store.")
        return

    year = datetime.now().year
    ts = _now_est_iso()
    dg = dg_id_lookup or {}

    rows = []
    for _, r in combined_df.iterrows():
        p1 = str(_get(r, "Player 1")).lower().strip()
        p2 = str(_get(r, "Player 2")).lower().strip()

        rows.append([
            ts,                                         # run_timestamp
            tourney,                                    # event_name
            year,                                       # year
            str(event_id),                              # event_id
            p1,                                         # player_1
            p2,                                         # player_2
            _safe(dg.get(p1, "")),                      # dg_id_p1
            _safe(dg.get(p2, "")),                      # dg_id_p2
            _safe(_get(r, "Bookmaker")),                # bookmaker
            _safe(_get(r, "Ties")),                     # ties_rule
            _safe(_get(r, "P1 Odds")),                  # p1_odds
            _safe(_get(r, "P2 Odds")),                  # p2_odds
            _safe(_get(r, "Fair_p1")),                  # fair_p1
            _safe(_get(r, "Fair_p2")),                  # fair_p2
            _safe(_get(r, "edge_p1"), round_digits=1),  # edge_p1
            _safe(_get(r, "edge_p2"), round_digits=1),  # edge_p2
            _safe(_get(r, "bet_on")),                   # bet_on
            _safe(_get(r, "edge_on"), round_digits=1),  # edge_on
            _safe(_get(r, "pred_on"), round_digits=2),  # pred_on
            _safe(_get(r, "pred_against"), round_digits=2),  # pred_against
            _safe(_get(r, "sample_on")),                # sample_on
            _safe(_get(r, "half_shot_p1"), round_digits=1),  # half_shot_p1
            _safe(_get(r, "half_shot_p2"), round_digits=1),  # half_shot_p2
            _safe(_get(r, "wind_on"), round_digits=2),  # wind_on
            _safe(_get(r, "wind_diff"), round_digits=2), # wind_diff
            _safe(_get(r, "wx_edge"), round_digits=1),  # wx_edge
            "",                                         # result (grading)
            "",                                         # units_won (grading)
        ])

    if spreadsheet is None:
        spreadsheet = get_spreadsheet()
    ws = _get_or_create_tab(spreadsheet, TAB_TOURNAMENT_MU, TOURNAMENT_MU_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} tournament matchup rows to '{TAB_TOURNAMENT_MU}'")

    # Parquet write-through
    _ledger_write_tournament_matchups(combined_df, tourney, event_id, dg_id_lookup, ts=ts)


# ══════════════════════════════════════════════════════════════════════════════
# Store: Finish Positions
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sim_prob(row):
    """
    Extract the correct simulation probability based on market_type.
    Handles both pre-rename (market_type) and post-rename (market) column names.
    """
    market = _get(row, "market_type", "market", default="")
    if market == "win":
        return _get(row, "simulated_win_prob", default=None)
    elif market == "top_5":
        return _get(row, "top_5", default=None)
    elif market == "top_10":
        return _get(row, "top_10", default=None)
    elif market == "top_20":
        return _get(row, "top_20", default=None)
    return None


def store_finish_positions(combined_finish_df, tourney, event_id, dg_id_lookup=None, spreadsheet=None, tab_name=None):
    """
    Write finish position bet rows to a Google Sheets tab.

    Args:
        combined_finish_df: Finish position DataFrame from new_sim.py
                            Expected columns: player_name, market_type (or market),
                            decimal_odds, american_odds, my_fair, edge, stake,
                            sample, my_pred, bookmaker (or book/sportsbook)
                            Plus sim probs: simulated_win_prob, top_5, top_10, top_20
        tourney:            Tournament name
        event_id:           DataGolf event ID
        dg_id_lookup:       Optional dict {player_name: dg_id}
        tab_name:           Override tab name (default: "Finish Positions")
    """
    tab = tab_name or TAB_FINISH_POS
    if combined_finish_df is None or combined_finish_df.empty:
        print("  [storage] No finish position bets to store.")
        return

    # Filter out tiny stakes (< $1) before writing
    stake_col = None
    for c in ["stake", "kelly_stake"]:
        if c in combined_finish_df.columns:
            stake_col = c
            break
    if stake_col:
        before = len(combined_finish_df)
        combined_finish_df = combined_finish_df[
            pd.to_numeric(combined_finish_df[stake_col], errors="coerce").fillna(0) >= 1.0
        ]
        dropped = before - len(combined_finish_df)
        if dropped:
            print(f"  [storage] Filtered out {dropped} finish position rows with stake < $1.00")
        if combined_finish_df.empty:
            print("  [storage] No finish position bets remaining after stake filter.")
            return

    year = datetime.now().year
    ts = _now_est_iso()
    dg = dg_id_lookup or {}

    rows = []
    for _, r in combined_finish_df.iterrows():
        player = str(_get(r, "player_name")).lower().strip()
        sim_prob = _extract_sim_prob(r)

        rows.append([
            ts,                                                 # run_timestamp
            tourney,                                            # event_name
            year,                                               # year
            str(event_id),                                      # event_id
            player,                                             # player_name
            _safe(dg.get(player, "")),                          # dg_id
            _safe(_get(r, "market_type", "market")),            # market_type
            _safe(_get(r, "bookmaker", "book", "sportsbook")),  # sportsbook
            _safe(_get(r, "decimal_odds"), round_digits=2),     # decimal_odds
            _safe(_get(r, "american_odds")),                    # american_odds
            _safe(_get(r, "my_fair")),                          # my_fair
            _safe(sim_prob, round_digits=4),                    # sim_prob
            _safe(_get(r, "edge"), round_digits=1),             # edge
            _safe(_get(r, "stake", "kelly_stake"), round_digits=2),  # kelly_stake
            _safe(_get(r, "my_pred"), round_digits=2),          # my_pred
            _safe(_get(r, "sample")),                           # sample
            "",                                                 # result (grading)
            "",                                                 # actual_finish (grading)
            "",                                                 # units_won (grading)
        ])

    if spreadsheet is None:
        spreadsheet = get_spreadsheet()
    ws = _get_or_create_tab(spreadsheet, tab, FINISH_POS_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} finish position rows to '{tab}'")

    # Parquet write-through
    if tab == TAB_FINISH_POS:
        _ledger_write_finish_positions(combined_finish_df, tourney, event_id, dg_id_lookup, ts=ts)
    elif tab == TAB_LIVE:
        _ledger_write_finish_positions(combined_finish_df, tourney, event_id, dg_id_lookup, ts=ts, bet_type="finish_position_live")


# ══════════════════════════════════════════════════════════════════════════════
# Store: Round Matchups
# ══════════════════════════════════════════════════════════════════════════════

def store_round_matchups(combined_df, sim_round, tourney, event_id, dg_id_lookup=None, spreadsheet=None):
    """
    Write round matchup rows to the "Round Matchups" tab.

    Args:
        combined_df:    Filtered round matchup DataFrame from round_sim.py
                        Expected columns: Player 1, Player 2, Bookmaker, Ties,
                        P1 Odds, P2 Odds, Fair_p1, Fair_p2, edge_p1, edge_p2,
                        edge_on, bet_on, p1_pred, p2_pred, pred_on, sample_on
                        Optional: half_shot_p1, half_shot_p2
        sim_round:      Round number (1-4)
        tourney:        Tournament name
        event_id:       DataGolf event ID
        dg_id_lookup:   Optional dict {player_name: dg_id}
    """
    if combined_df is None or combined_df.empty:
        print("  [storage] No round matchups to store.")
        return

    year = datetime.now().year
    ts = _now_est_iso()
    dg = dg_id_lookup or {}

    rows = []
    for _, r in combined_df.iterrows():
        p1 = str(_get(r, "Player 1")).lower().strip()
        p2 = str(_get(r, "Player 2")).lower().strip()

        rows.append([
            ts,                                         # run_timestamp
            tourney,                                    # event_name
            year,                                       # year
            str(event_id),                              # event_id
            int(sim_round),                             # round
            p1,                                         # player_1
            p2,                                         # player_2
            _safe(dg.get(p1, "")),                      # dg_id_p1
            _safe(dg.get(p2, "")),                      # dg_id_p2
            _safe(_get(r, "Bookmaker")),                # bookmaker
            _safe(_get(r, "Ties")),                     # ties_rule
            _safe(_get(r, "P1 Odds")),                  # p1_odds
            _safe(_get(r, "P2 Odds")),                  # p2_odds
            _safe(_get(r, "Fair_p1")),                  # fair_p1
            _safe(_get(r, "Fair_p2")),                  # fair_p2
            _safe(_get(r, "edge_p1"), round_digits=1),  # edge_p1
            _safe(_get(r, "edge_p2"), round_digits=1),  # edge_p2
            _safe(_get(r, "bet_on")),                   # bet_on
            _safe(_get(r, "edge_on"), round_digits=1),  # edge_on
            _safe(_get(r, "pred_on", "p1_pred"), round_digits=2),     # pred_on
            _safe(_get(r, "pred_against", "p2_pred"), round_digits=2),# pred_against
            _safe(_get(r, "sample_on")),                # sample_on
            _safe(_get(r, "half_shot_p1"), round_digits=1),  # half_shot_p1
            _safe(_get(r, "half_shot_p2"), round_digits=1),  # half_shot_p2
            _safe(_get(r, "wx_edge"), round_digits=1),  # wx_edge
            "",                                         # result (grading)
            "",                                         # p1_round_score (grading)
            "",                                         # p2_round_score (grading)
            "",                                         # units_won (grading)
        ])

    if spreadsheet is None:
        spreadsheet = get_spreadsheet()
    ws = _get_or_create_tab(spreadsheet, TAB_ROUND_MU, ROUND_MU_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} R{sim_round} matchup rows to '{TAB_ROUND_MU}'")

    # Parquet write-through
    _ledger_write_round_matchups(combined_df, sim_round, tourney, event_id, dg_id_lookup, ts=ts)





# ══════════════════════════════════════════════════════════════════════════════
# Store: Base Rates
# ══════════════════════════════════════════════════════════════════════════════

def store_base_rates(rows, spreadsheet=None):
    """
    Write base rates reference table to the "Base Rates" tab.
    Overwrites (clear + rewrite) since this is a point-in-time snapshot.

    Args:
        rows: list of lists, each [category, parameter, base_rate, this_week, delta, notes]
    """
    if not rows:
        print("  [storage] No base rates rows to store.")
        return

    if spreadsheet is None:
        spreadsheet = get_spreadsheet()
    ws = _get_or_create_tab(spreadsheet, TAB_BASE_RATES, BASE_RATES_HEADERS)

    # Clear and rewrite (snapshot reference)
    ws.clear()
    ws.append_row(BASE_RATES_HEADERS, value_input_option="RAW")

    # Sanitize all values
    clean_rows = []
    for row in rows:
        clean_rows.append([_safe(v, round_digits=4) if isinstance(v, float) else _safe(v) for v in row])

    ws.append_rows(clean_rows, value_input_option="USER_ENTERED")
    print(f"  [storage] Wrote {len(clean_rows)} base rates rows to '{TAB_BASE_RATES}'")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: Build dg_id lookup from pre_course_fit file
# ══════════════════════════════════════════════════════════════════════════════

def load_dg_id_lookup(tourney, name_replacements=None):
    """
    Build a player_name → dg_id dict from pre_course_fit_{tourney}.csv.
    Falls back gracefully if file or column doesn't exist.

    Args:
        tourney:           Tournament name (to find the file)
        name_replacements: Optional dict for normalizing player names
    """
    path = f"pre_course_fit_{tourney}.csv"
    if not os.path.exists(path):
        print(f"  [storage] {path} not found — dg_id will be empty")
        return {}

    try:
        df = pd.read_csv(path)
        if "dg_id" not in df.columns:
            print(f"  [storage] {path} has no 'dg_id' column — dg_id will be empty")
            return {}

        df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
        if name_replacements:
            df["player_name"] = df["player_name"].replace(name_replacements)

        lookup = dict(zip(df["player_name"], df["dg_id"]))
        print(f"  [storage] Loaded {len(lookup)} dg_id mappings from {path}")
        return lookup
    except Exception as e:
        print(f"  [storage] Error loading dg_id from {path}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Parquet Bet Ledger — Local Write-Through
# ══════════════════════════════════════════════════════════════════════════════

# Book categorization (duplicated from grade_bets for independence)
_SHARP_BOOKS = ["pinnacle", "betonline", "betcris", "bet online", "bookmaker"]
_RETAIL_BOOKS = ["fanduel", "draftkings", "caesars", "dk", "fd", "czr", "betmgm", "mgm"]


def _categorize_book(book_name):
    """Categorize a sportsbook as sharp, retail, or other."""
    if not book_name:
        return "other"
    b = str(book_name).lower().strip()
    for s in _SHARP_BOOKS:
        if s in b:
            return "sharp"
    for r in _RETAIL_BOOKS:
        if r in b:
            return "retail"
    return "other"


def _empty_ledger_record():
    """Return a dict with all ledger columns set to defaults."""
    return {
        "bet_id": "",
        "run_timestamp": "",
        "event_name": "",
        "year": 0,
        "event_id": "",
        "bet_type": "",
        "round": 0,
        "bet_on": "",
        "opponent": "",
        "dg_id_bet_on": "",
        "dg_id_opponent": "",
        "bookmaker": "",
        "book_odds": np.nan,
        "fair_odds": np.nan,
        "edge": np.nan,
        "pred_on": np.nan,
        "sample_on": np.nan,
        "kelly_stake": np.nan,
        "half_shot": np.nan,
        "wind_on": np.nan,
        "wind_diff": np.nan,
        "wx_edge": np.nan,
        "book_category": "",
        "result": "",
        "actual_finish": "",
        "p1_round_score": np.nan,
        "p2_round_score": np.nan,
        "units_wagered": np.nan,
        "units_won": np.nan,
        "graded_at": "",
        "margin": np.nan,
        "margin_bucket": "",
        "is_unlucky_push": False,
        "bet_on_sg_ott": np.nan,
        "bet_on_sg_app": np.nan,
        "bet_on_sg_arg": np.nan,
        "bet_on_sg_putt": np.nan,
        "opp_sg_ott": np.nan,
        "opp_sg_app": np.nan,
        "opp_sg_arg": np.nan,
        "opp_sg_putt": np.nan,
    }


def _append_to_ledger(records):
    """
    Append records to the Parquet ledger, deduplicating by LEDGER_DEDUP_COLS.
    Atomic write via temp file + os.replace().
    """
    if not records:
        return

    new_df = pd.DataFrame(records)

    # Coerce run_timestamp to datetime to match existing ledger dtype
    if "run_timestamp" in new_df.columns:
        new_df["run_timestamp"] = pd.to_datetime(new_df["run_timestamp"], errors="coerce")

    # Ensure permanent_data dir exists
    ledger_dir = os.path.dirname(LEDGER_PATH)
    os.makedirs(ledger_dir, exist_ok=True)

    if os.path.exists(LEDGER_PATH):
        try:
            existing = pd.read_parquet(LEDGER_PATH)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    # Normalize dedup columns for matching
    for col in LEDGER_DEDUP_COLS:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.lower().str.strip()

    # Keep first occurrence (existing rows win over new)
    combined = combined.drop_duplicates(subset=LEDGER_DEDUP_COLS, keep="first")

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=ledger_dir)
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, LEDGER_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    new_count = len(combined) - len(existing)
    print(f"  [ledger] {new_count} new rows added ({len(combined)} total in ledger)")


def _ledger_write_tournament_matchups(combined_df, tourney, event_id, dg_id_lookup, ts):
    """Build ledger records from tournament matchup DataFrame and append."""
    try:
        if combined_df is None or combined_df.empty:
            return
        dg = dg_id_lookup or {}
        year = datetime.now().year
        records = []
        for _, r in combined_df.iterrows():
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            bet_on = str(_get(r, "bet_on")).lower().strip()
            opponent = p2 if bet_on == p1 else p1

            if bet_on == p1:
                book_odds = _get(r, "P1 Odds", default=np.nan)
                fair_odds = _get(r, "Fair_p1", default=np.nan)
                hs = _get(r, "half_shot_p1", default=np.nan)
            else:
                book_odds = _get(r, "P2 Odds", default=np.nan)
                fair_odds = _get(r, "Fair_p2", default=np.nan)
                hs = _get(r, "half_shot_p2", default=np.nan)

            bookmaker = str(_get(r, "Bookmaker", default=""))

            rec = _empty_ledger_record()
            rec.update({
                "bet_id": str(uuid.uuid4()),
                "run_timestamp": ts,
                "event_name": tourney,
                "year": year,
                "event_id": str(event_id),
                "bet_type": "tournament_matchup",
                "round": 0,
                "bet_on": bet_on,
                "opponent": opponent,
                "dg_id_bet_on": str(dg.get(bet_on, "")),
                "dg_id_opponent": str(dg.get(opponent, "")),
                "bookmaker": bookmaker,
                "book_odds": _safe_float(book_odds),
                "fair_odds": _safe_float(fair_odds),
                "edge": _safe_float(_get(r, "edge_on", default=np.nan)),
                "pred_on": _safe_float(_get(r, "pred_on", default=np.nan)),
                "sample_on": _safe_float(_get(r, "sample_on", default=np.nan)),
                "half_shot": _safe_float(hs),
                "wind_on": _safe_float(_get(r, "wind_on", default=np.nan)),
                "wind_diff": _safe_float(_get(r, "wind_diff", default=np.nan)),
                "wx_edge": _safe_float(_get(r, "wx_edge", default=np.nan)),
                "book_category": _categorize_book(bookmaker),
            })
            records.append(rec)

        _append_to_ledger(records)
    except Exception as e:
        print(f"  [ledger] Warning: tournament matchup write failed: {e}")


def _ledger_write_finish_positions(combined_finish_df, tourney, event_id, dg_id_lookup, ts, bet_type="finish_position"):
    """Build ledger records from finish position DataFrame and append."""
    try:
        if combined_finish_df is None or combined_finish_df.empty:
            return
        dg = dg_id_lookup or {}
        year = datetime.now().year
        records = []
        for _, r in combined_finish_df.iterrows():
            player = str(_get(r, "player_name")).lower().strip()
            market = str(_get(r, "market_type", "market", default=""))
            bookmaker = str(_get(r, "bookmaker", "book", "sportsbook", default=""))

            rec = _empty_ledger_record()
            rec.update({
                "bet_id": str(uuid.uuid4()),
                "run_timestamp": ts,
                "event_name": tourney,
                "year": year,
                "event_id": str(event_id),
                "bet_type": bet_type,
                "round": 0,
                "bet_on": player,
                "opponent": market,
                "dg_id_bet_on": str(dg.get(player, "")),
                "dg_id_opponent": "",
                "bookmaker": bookmaker,
                "book_odds": _safe_float(_get(r, "american_odds", default=np.nan)),
                "fair_odds": _safe_float(_get(r, "my_fair", default=np.nan)),
                "edge": _safe_float(_get(r, "edge", default=np.nan)),
                "pred_on": _safe_float(_get(r, "my_pred", default=np.nan)),
                "sample_on": _safe_float(_get(r, "sample", default=np.nan)),
                "kelly_stake": _safe_float(_get(r, "stake", "kelly_stake", default=np.nan)),
                "book_category": _categorize_book(bookmaker),
            })
            records.append(rec)

        _append_to_ledger(records)
    except Exception as e:
        print(f"  [ledger] Warning: finish position write failed: {e}")


def _ledger_write_round_matchups(combined_df, sim_round, tourney, event_id, dg_id_lookup, ts):
    """Build ledger records from round matchup DataFrame and append."""
    try:
        if combined_df is None or combined_df.empty:
            return
        dg = dg_id_lookup or {}
        year = datetime.now().year
        records = []
        for _, r in combined_df.iterrows():
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            bet_on = str(_get(r, "bet_on")).lower().strip()
            opponent = p2 if bet_on == p1 else p1

            if bet_on == p1:
                book_odds = _get(r, "P1 Odds", default=np.nan)
                fair_odds = _get(r, "Fair_p1", default=np.nan)
                hs = _get(r, "half_shot_p1", default=np.nan)
            else:
                book_odds = _get(r, "P2 Odds", default=np.nan)
                fair_odds = _get(r, "Fair_p2", default=np.nan)
                hs = _get(r, "half_shot_p2", default=np.nan)

            bookmaker = str(_get(r, "Bookmaker", default=""))

            rec = _empty_ledger_record()
            rec.update({
                "bet_id": str(uuid.uuid4()),
                "run_timestamp": ts,
                "event_name": tourney,
                "year": year,
                "event_id": str(event_id),
                "bet_type": "round_matchup",
                "round": int(sim_round),
                "bet_on": bet_on,
                "opponent": opponent,
                "dg_id_bet_on": str(dg.get(bet_on, "")),
                "dg_id_opponent": str(dg.get(opponent, "")),
                "bookmaker": bookmaker,
                "book_odds": _safe_float(book_odds),
                "fair_odds": _safe_float(fair_odds),
                "edge": _safe_float(_get(r, "edge_on", default=np.nan)),
                "pred_on": _safe_float(_get(r, "pred_on", "p1_pred", default=np.nan)),
                "sample_on": _safe_float(_get(r, "sample_on", default=np.nan)),
                "half_shot": _safe_float(hs),
                "wx_edge": _safe_float(_get(r, "wx_edge", default=np.nan)),
                "book_category": _categorize_book(bookmaker),
            })
            records.append(rec)

        _append_to_ledger(records)
    except Exception as e:
        print(f"  [ledger] Warning: round matchup write failed: {e}")


def _safe_float(val):
    """Convert value to float, returning NaN for failures."""
    if val is None or val == "":
        return np.nan
    try:
        f = float(val)
        return f if not np.isinf(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def update_ledger_grades(graded_bets):
    """
    Update the Parquet ledger with grading results.
    Matches on LEDGER_DEDUP_COLS. Updates: result, units_wagered, units_won,
    actual_finish, p1_round_score, p2_round_score, graded_at.

    Args:
        graded_bets: list of dicts from grade_bets.py, each with:
            bet_type, bet_on, opponent (or player_1/player_2), bookmaker,
            round, event_id, result, units_wagered, units_won, etc.
    """
    if not graded_bets or not os.path.exists(LEDGER_PATH):
        if not os.path.exists(LEDGER_PATH):
            print("  [ledger] No ledger file found — skipping grade update")
        return

    try:
        ledger = pd.read_parquet(LEDGER_PATH)
    except Exception as e:
        print(f"  [ledger] Error reading ledger: {e}")
        return

    if ledger.empty:
        print("  [ledger] Ledger is empty — nothing to grade")
        return

    # Normalize dedup cols in ledger
    for col in LEDGER_DEDUP_COLS:
        if col in ledger.columns:
            ledger[col] = ledger[col].astype(str).str.lower().str.strip()

    graded_ts = _now_est_iso()
    updated = 0

    for bet in graded_bets:
        result = bet.get("result", "")
        if not result or result in ("no_data", "unknown", "duplicate"):
            continue

        # Build match key from graded bet
        bt = str(bet.get("bet_type", "")).lower().strip()
        bet_on = str(bet.get("bet_on", bet.get("player_name", ""))).lower().strip()
        bookmaker = str(bet.get("bookmaker", "")).lower().strip()
        event_id = str(bet.get("event_id", "")).lower().strip()

        # Determine round
        rd = bet.get("round", "0")
        if str(rd).lower() == "tournament":
            rd = "0"
        rd = str(rd).lower().strip()

        # Determine opponent
        if bt == "finish_position":
            opponent = str(bet.get("market_type", bet.get("opponent", ""))).lower().strip()
        else:
            opponent = str(bet.get("opponent", "")).lower().strip()
            if not opponent:
                # Derive from player_1/player_2
                p1 = str(bet.get("player_1", "")).lower().strip()
                p2 = str(bet.get("player_2", "")).lower().strip()
                opponent = p2 if bet_on == p1 else p1

        # Find matching row(s)
        mask = (
            (ledger["event_id"] == event_id) &
            (ledger["bet_type"] == bt) &
            (ledger["bet_on"] == bet_on) &
            (ledger["opponent"] == opponent) &
            (ledger["bookmaker"] == bookmaker) &
            (ledger["round"].astype(str) == rd)
        )

        matches = ledger.index[mask]
        if len(matches) == 0:
            continue

        idx = matches[0]
        ledger.at[idx, "result"] = result
        ledger.at[idx, "units_wagered"] = _safe_float(bet.get("units_wagered"))
        ledger.at[idx, "units_won"] = _safe_float(bet.get("units_won"))
        ledger.at[idx, "graded_at"] = graded_ts

        if "actual_finish" in bet:
            ledger.at[idx, "actual_finish"] = str(bet["actual_finish"])
        if "p1_score" in bet:
            ledger.at[idx, "p1_round_score"] = _safe_float(bet["p1_score"])
        if "p2_score" in bet:
            ledger.at[idx, "p2_round_score"] = _safe_float(bet["p2_score"])

        # Margin + SG attribution fields (always write, even None → NaN clears old bad values)
        margin = bet.get("margin")
        margin_val = _safe_float(margin) if margin is not None else np.nan
        ledger.at[idx, "margin"] = margin_val
        if not np.isnan(margin_val):
            abs_m = abs(margin_val)
            sign = "+" if margin_val > 0 else ("-" if margin_val < 0 else "")
            if abs_m < 1.5:
                ledger.at[idx, "margin_bucket"] = f"{sign}1"
            elif abs_m < 2.5:
                ledger.at[idx, "margin_bucket"] = f"{sign}2"
            else:
                ledger.at[idx, "margin_bucket"] = f"{sign}3+"
        else:
            ledger.at[idx, "margin_bucket"] = ""

        # Unlucky push: push at +130 or better odds (high EV missed)
        if result == "push":
            book_odds_val = _safe_float(ledger.at[idx, "book_odds"]) if "book_odds" in ledger.columns else np.nan
            if not np.isnan(book_odds_val) and book_odds_val >= 130:
                ledger.at[idx, "is_unlucky_push"] = True

        for sg_col in ["bet_on_sg_ott", "bet_on_sg_app", "bet_on_sg_arg", "bet_on_sg_putt",
                        "opp_sg_ott", "opp_sg_app", "opp_sg_arg", "opp_sg_putt"]:
            if sg_col in bet and bet[sg_col] is not None:
                ledger.at[idx, sg_col] = _safe_float(bet[sg_col])

        updated += 1

    if updated > 0:
        # Atomic write
        ledger_dir = os.path.dirname(LEDGER_PATH)
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=ledger_dir)
        os.close(fd)
        try:
            ledger.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, LEDGER_PATH)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    print(f"  [ledger] Graded {updated} / {len(graded_bets)} bets in ledger")


def query_ledger(**filters):
    """
    Query the Parquet ledger with optional filters.

    Keyword args (all optional):
        event: str — filter by event_name (substring match)
        bet_type: str — exact match on bet_type
        book: str — filter by bookmaker (substring match)
        min_edge: float — minimum edge
        graded: bool — if True, only graded (result != "")
        year: int — filter by year

    Returns:
        pd.DataFrame (empty if no ledger or no matches)
    """
    if not os.path.exists(LEDGER_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(LEDGER_PATH)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "event" in filters and filters["event"]:
        df = df[df["event_name"].str.contains(str(filters["event"]), case=False, na=False)]
    if "bet_type" in filters and filters["bet_type"]:
        df = df[df["bet_type"] == filters["bet_type"]]
    if "book" in filters and filters["book"]:
        df = df[df["bookmaker"].str.contains(str(filters["book"]), case=False, na=False)]
    if "min_edge" in filters and filters["min_edge"] is not None:
        df = df[df["edge"] >= float(filters["min_edge"])]
    if "graded" in filters and filters["graded"]:
        df = df[df["result"].astype(str).str.strip() != ""]
    if "year" in filters and filters["year"]:
        df = df[df["year"] == int(filters["year"])]

    return df


def cleanup_small_finish_bets(min_stake=1.0):
    """One-time cleanup: delete Finish Positions rows with kelly_stake < min_stake.

    Deletes rows bottom-to-top to preserve row indices.
    Rate-limited at 0.5s per delete to avoid Google API quota issues.
    """
    spreadsheet = get_spreadsheet()
    ws = spreadsheet.worksheet(TAB_FINISH_POS)
    rows = ws.get_all_values()

    if len(rows) <= 1:
        print("  [cleanup] No data rows in Finish Positions tab.")
        return

    headers = rows[0]
    try:
        stake_idx = headers.index("kelly_stake")
    except ValueError:
        print("  [cleanup] 'kelly_stake' column not found in headers.")
        return

    # Find rows to delete (1-indexed, header is row 1)
    to_delete = []
    for i, row in enumerate(rows[1:], start=2):  # start=2 because row 1 is header
        try:
            stake = float(row[stake_idx]) if row[stake_idx] else 0.0
        except (ValueError, IndexError):
            continue
        if stake < min_stake:
            to_delete.append(i)

    if not to_delete:
        print(f"  [cleanup] No rows with kelly_stake < ${min_stake:.2f}")
        return

    print(f"  [cleanup] Deleting {len(to_delete)} rows with kelly_stake < ${min_stake:.2f} ...")

    # Delete bottom-to-top to preserve indices
    for row_idx in reversed(to_delete):
        ws.delete_rows(row_idx)
        time.sleep(0.5)

    print(f"  [cleanup] Done. Removed {len(to_delete)} rows.")
"""
sheets_storage.py — Google Sheets + Drive storage for golf sim bet records.

Writes bet records to four tabs in the "golf_sims" spreadsheet:
  - Tournament Matchups   (from new_sim.py → combined_df)
  - Finish Positions       (from new_sim.py → combined_finish_df)
  - Round Matchups         (from round_sim.py → combined df)
  - Sharp Filtered Bets    (from both scripts, sharp-only, deduped)

Also uploads raw sim CSVs to a Google Drive folder.

Auth:
  Uses the same credential pattern as sheet_config.py:
  - GOOGLE_CREDS_JSON env var (for GitHub Actions / CI)
  - credentials.json file (for local dev)
  Scopes are widened to read/write (spreadsheets + drive.file).

Usage:
  from sheets_storage import (
      is_valid_run_time,
      store_tournament_matchups,
      store_finish_positions,
      store_round_matchups,
      store_sharp_filtered,
      upload_csv_to_drive,
  )
"""

import os
import json
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
TAB_SHARP = "Sharp Filtered Bets"
TAB_ALL_FILTERED = "All Filtered Bets"

# Drive folder root
DRIVE_ROOT_FOLDER = "golf_sim_outputs"


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
    "wind_on", "wind_diff",
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

ROUND_MU_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id", "round",
    "player_1", "player_2", "dg_id_p1", "dg_id_p2",
    "bookmaker", "ties_rule",
    "p1_odds", "p2_odds", "fair_p1", "fair_p2",
    "edge_p1", "edge_p2",
    "bet_on", "edge_on", "pred_on", "pred_against", "sample_on",
    "half_shot_p1", "half_shot_p2",
    "result", "p1_round_score", "p2_round_score", "units_won",
]

SHARP_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id",
    "bet_type", "round",
    "bet_on", "opponent", "bookmaker",
    "book_odds", "fair_odds", "edge",
    "pred_on", "sample_on",
    "kelly_stake", "half_shot",
    "result", "units_won",
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


def _connect_sheets():
    """Authenticate and return the gspread Spreadsheet object."""
    creds = _get_credentials()
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME)


def _connect_drive():
    """Build a Google Drive API service for CSV uploads."""
    from googleapiclient.discovery import build
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds)


# ══════════════════════════════════════════════════════════════════════════════
# Tab Management
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_create_tab(spreadsheet, tab_name, headers):
    """
    Get an existing tab or create it with header row.
    Returns the gspread Worksheet.
    """
    try:
        ws = spreadsheet.worksheet(tab_name)
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
    Sunday is too early (previous week). Monday before 3 PM is pre-field.
    All other times are valid.
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
    # Sunday (weekday 6) → previous week's tournament
    if now.weekday() == 6:
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

def store_tournament_matchups(combined_df, tourney, event_id, dg_id_lookup=None):
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
            "",                                         # result (grading)
            "",                                         # units_won (grading)
        ])

    spreadsheet = _connect_sheets()
    ws = _get_or_create_tab(spreadsheet, TAB_TOURNAMENT_MU, TOURNAMENT_MU_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} tournament matchup rows to '{TAB_TOURNAMENT_MU}'")


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


def store_finish_positions(combined_finish_df, tourney, event_id, dg_id_lookup=None):
    """
    Write finish position bet rows to the "Finish Positions" tab.

    Args:
        combined_finish_df: Finish position DataFrame from new_sim.py
                            Expected columns: player_name, market_type (or market),
                            decimal_odds, american_odds, my_fair, edge, stake,
                            sample, my_pred, bookmaker (or book/sportsbook)
                            Plus sim probs: simulated_win_prob, top_5, top_10, top_20
        tourney:            Tournament name
        event_id:           DataGolf event ID
        dg_id_lookup:       Optional dict {player_name: dg_id}
    """
    if combined_finish_df is None or combined_finish_df.empty:
        print("  [storage] No finish position bets to store.")
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

    spreadsheet = _connect_sheets()
    ws = _get_or_create_tab(spreadsheet, TAB_FINISH_POS, FINISH_POS_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} finish position rows to '{TAB_FINISH_POS}'")


# ══════════════════════════════════════════════════════════════════════════════
# Store: Round Matchups
# ══════════════════════════════════════════════════════════════════════════════

def store_round_matchups(combined_df, sim_round, tourney, event_id, dg_id_lookup=None):
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
            "",                                         # result (grading)
            "",                                         # p1_round_score (grading)
            "",                                         # p2_round_score (grading)
            "",                                         # units_won (grading)
        ])

    spreadsheet = _connect_sheets()
    ws = _get_or_create_tab(spreadsheet, TAB_ROUND_MU, ROUND_MU_HEADERS)
    _append_rows(ws, rows)
    print(f"  [storage] Wrote {len(rows)} R{sim_round} matchup rows to '{TAB_ROUND_MU}'")


# ══════════════════════════════════════════════════════════════════════════════
# Store: Sharp Filtered Bets (unified across bet types)
# ══════════════════════════════════════════════════════════════════════════════

def store_sharp_filtered(tourney, event_id,
                         sharp_matchups=None,
                         sharp_rounds=None, sim_round=None,
                         sharp_finishes=None):
    """
    Write sharp-filtered bets to the unified "Sharp Filtered Bets" tab.
    Accepts any combination of the three bet types.

    Args:
        tourney:          Tournament name
        event_id:         DataGolf event ID
        sharp_matchups:   Sharp tournament matchup DataFrame (new_sim.py → sharp_df)
        sharp_rounds:     Sharp round matchup DataFrame (round_sim.py → sharp)
        sim_round:        Round number for round matchups (1-4)
        sharp_finishes:   Finish position DataFrame (new_sim.py → combined_finish_df
                          or a sharp-filtered subset)
    """
    year = datetime.now().year
    ts = _now_est_iso()
    rows = []

    # --- Tournament matchups ---
    # --- Tournament matchups (pred > 0.75 and sample > 20 only) ---
    if sharp_matchups is not None and not sharp_matchups.empty:
        for _, r in sharp_matchups.iterrows():
            pred_val = float(_get(r, "pred_on") or 0)
            sample_val = float(_get(r, "sample_on") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            bet_player = _safe(_get(r, "bet_on"))
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            opponent = p2 if str(bet_player).lower().strip() == p1 else p1

            # Book odds for the bet side
            if str(bet_player).lower().strip() == p1:
                book_odds = _safe(_get(r, "P1 Odds"))
                fair_odds = _safe(_get(r, "Fair_p1"))
                hs = _safe(_get(r, "half_shot_p1"), round_digits=1)
            else:
                book_odds = _safe(_get(r, "P2 Odds"))
                fair_odds = _safe(_get(r, "Fair_p2"))
                hs = _safe(_get(r, "half_shot_p2"), round_digits=1)

            rows.append([
                ts, tourney, year, str(event_id),
                "tournament_matchup",                       # bet_type
                0,                                          # round (0 = tournament)
                _safe(bet_player),                          # bet_on
                opponent,                                   # opponent
                _safe(_get(r, "Bookmaker")),                # bookmaker
                book_odds,                                  # book_odds
                fair_odds,                                  # fair_odds
                _safe(_get(r, "edge_on"), round_digits=1),  # edge
                _safe(_get(r, "pred_on"), round_digits=2),  # pred_on
                _safe(_get(r, "sample_on")),                # sample_on
                "",                                         # kelly_stake (flat bet for MU)
                hs,                                         # half_shot
                "",                                         # result (grading)
                "",                                         # units_won (grading)
            ])

    # --- Round matchups (pred > 0.75 and sample > 20 only) ---
    if sharp_rounds is not None and not sharp_rounds.empty:
        rd = sim_round or 0
        for _, r in sharp_rounds.iterrows():
            pred_val = float(_get(r, "pred_on") or 0)
            sample_val = float(_get(r, "sample_on") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            bet_player = _safe(_get(r, "bet_on"))
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            opponent = p2 if str(bet_player).lower().strip() == p1 else p1

            if str(bet_player).lower().strip() == p1:
                book_odds = _safe(_get(r, "P1 Odds"))
                fair_odds = _safe(_get(r, "Fair_p1"))
                hs = _safe(_get(r, "half_shot_p1"), round_digits=1)
            else:
                book_odds = _safe(_get(r, "P2 Odds"))
                fair_odds = _safe(_get(r, "Fair_p2"))
                hs = _safe(_get(r, "half_shot_p2"), round_digits=1)

            rows.append([
                ts, tourney, year, str(event_id),
                "round_matchup",                            # bet_type
                int(rd),                                    # round
                _safe(bet_player),                          # bet_on
                opponent,                                   # opponent
                _safe(_get(r, "Bookmaker")),                # bookmaker
                book_odds,                                  # book_odds
                fair_odds,                                  # fair_odds
                _safe(_get(r, "edge_on"), round_digits=1),  # edge
                _safe(_get(r, "pred_on"), round_digits=2),  # pred_on
                _safe(_get(r, "sample_on")),                # sample_on
                "",                                         # kelly_stake (flat bet)
                hs,                                         # half_shot
                "",                                         # result (grading)
                "",                                         # units_won (grading)
            ])

    # --- Finish positions (pred > 0.75 and sample > 20 only) ---
    if sharp_finishes is not None and not sharp_finishes.empty:
        for _, r in sharp_finishes.iterrows():
            pred_val = float(_get(r, "my_pred") or 0)
            sample_val = float(_get(r, "sample") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            player = str(_get(r, "player_name")).lower().strip()
            market = _safe(_get(r, "market_type", "market"))

            rows.append([
                ts, tourney, year, str(event_id),
                "finish_position",                                   # bet_type
                0,                                                   # round (0 = tournament)
                player,                                              # bet_on
                market,                                              # opponent (market_type)
                _safe(_get(r, "bookmaker", "book", "sportsbook")),   # bookmaker
                _safe(_get(r, "american_odds")),                     # book_odds
                _safe(_get(r, "my_fair")),                           # fair_odds
                _safe(_get(r, "edge"), round_digits=1),              # edge
                _safe(_get(r, "my_pred"), round_digits=2),           # pred_on
                _safe(_get(r, "sample")),                            # sample_on
                _safe(_get(r, "stake", "kelly_stake"), round_digits=2),  # kelly_stake
                "",                                                  # half_shot (N/A)
                "",                                                  # result (grading)
                "",                                                  # units_won (grading)
            ])

    if not rows:
        print("  [storage] No sharp filtered bets to store.")
        return

    spreadsheet = _connect_sheets()
    ws = _get_or_create_tab(spreadsheet, TAB_SHARP, SHARP_HEADERS)
    _append_rows(ws, rows)

    # Summary counts by type
    type_counts = {}
    for row in rows:
        bt = row[4]  # bet_type column
        type_counts[bt] = type_counts.get(bt, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
    print(f"  [storage] Wrote {len(rows)} sharp filtered rows ({summary}) to '{TAB_SHARP}'")

def store_all_filtered(tourney, event_id,
                       all_matchups=None,
                       all_rounds=None, sim_round=None,
                       all_finishes=None):
    """
    Write filtered bets from ALL books to the "All Filtered Bets" tab.
    Same schema as Sharp Filtered, but includes every bookmaker.
    Gate: pred > 0.75 and sample > 20.
    """
    year = datetime.now().year
    ts = _now_est_iso()
    rows = []

    # --- Tournament matchups (all books, pred > 0.75 and sample > 20) ---
    if all_matchups is not None and not all_matchups.empty:
        for _, r in all_matchups.iterrows():
            pred_val = float(_get(r, "pred_on") or 0)
            sample_val = float(_get(r, "sample_on") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            bet_player = _safe(_get(r, "bet_on"))
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            opponent = p2 if str(bet_player).lower().strip() == p1 else p1

            if str(bet_player).lower().strip() == p1:
                book_odds = _safe(_get(r, "P1 Odds"))
                fair_odds = _safe(_get(r, "Fair_p1"))
                hs = _safe(_get(r, "half_shot_p1"), round_digits=1)
            else:
                book_odds = _safe(_get(r, "P2 Odds"))
                fair_odds = _safe(_get(r, "Fair_p2"))
                hs = _safe(_get(r, "half_shot_p2"), round_digits=1)

            rows.append([
                ts, tourney, year, str(event_id),
                "tournament_matchup", 0,
                _safe(bet_player), opponent,
                _safe(_get(r, "Bookmaker")),
                book_odds, fair_odds,
                _safe(_get(r, "edge_on"), round_digits=1),
                _safe(_get(r, "pred_on"), round_digits=2),
                _safe(_get(r, "sample_on")),
                "", hs, "", "",
            ])

    # --- Round matchups (all books, pred > 0.75 and sample > 20) ---
    if all_rounds is not None and not all_rounds.empty:
        rd = sim_round or 0
        for _, r in all_rounds.iterrows():
            pred_val = float(_get(r, "pred_on") or 0)
            sample_val = float(_get(r, "sample_on") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            bet_player = _safe(_get(r, "bet_on"))
            p1 = str(_get(r, "Player 1")).lower().strip()
            p2 = str(_get(r, "Player 2")).lower().strip()
            opponent = p2 if str(bet_player).lower().strip() == p1 else p1

            if str(bet_player).lower().strip() == p1:
                book_odds = _safe(_get(r, "P1 Odds"))
                fair_odds = _safe(_get(r, "Fair_p1"))
                hs = _safe(_get(r, "half_shot_p1"), round_digits=1)
            else:
                book_odds = _safe(_get(r, "P2 Odds"))
                fair_odds = _safe(_get(r, "Fair_p2"))
                hs = _safe(_get(r, "half_shot_p2"), round_digits=1)

            rows.append([
                ts, tourney, year, str(event_id),
                "round_matchup", int(rd),
                _safe(bet_player), opponent,
                _safe(_get(r, "Bookmaker")),
                book_odds, fair_odds,
                _safe(_get(r, "edge_on"), round_digits=1),
                _safe(_get(r, "pred_on"), round_digits=2),
                _safe(_get(r, "sample_on")),
                "", hs, "", "",
            ])

    # --- Finish positions (all books, pred > 0.75 and sample > 20) ---
    if all_finishes is not None and not all_finishes.empty:
        for _, r in all_finishes.iterrows():
            pred_val = float(_get(r, "my_pred") or 0)
            sample_val = float(_get(r, "sample") or 0)
            if pred_val <= 0.75 or sample_val <= 20:
                continue

            player = str(_get(r, "player_name")).lower().strip()
            market = _safe(_get(r, "market_type", "market"))

            rows.append([
                ts, tourney, year, str(event_id),
                "finish_position", 0,
                player, market,
                _safe(_get(r, "bookmaker", "book", "sportsbook")),
                _safe(_get(r, "american_odds")),
                _safe(_get(r, "my_fair")),
                _safe(_get(r, "edge"), round_digits=1),
                _safe(_get(r, "my_pred"), round_digits=2),
                _safe(_get(r, "sample")),
                _safe(_get(r, "stake", "kelly_stake"), round_digits=2),
                "", "", "",
            ])

    if not rows:
        print("  [storage] No all-book filtered bets to store.")
        return

    spreadsheet = _connect_sheets()
    ws = _get_or_create_tab(spreadsheet, TAB_ALL_FILTERED, SHARP_HEADERS)
    _append_rows(ws, rows)

    type_counts = {}
    for row in rows:
        bt = row[4]
        type_counts[bt] = type_counts.get(bt, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
    print(f"  [storage] Wrote {len(rows)} all-book filtered rows ({summary}) to '{TAB_ALL_FILTERED}'")
    
# ══════════════════════════════════════════════════════════════════════════════
# Google Drive: CSV Upload
# ══════════════════════════════════════════════════════════════════════════════

def _find_or_create_drive_folder(service, folder_name, parent_name=DRIVE_ROOT_FOLDER):
    """
    Find or create a folder in Drive. Creates parent if needed.
    Returns the folder ID.
    """
    def _find_folder(name, parent_id=None):
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        results = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def _create_folder(name, parent_id=None):
        body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            body["parents"] = [parent_id]
        folder = service.files().create(body=body, fields="id").execute()
        return folder["id"]

    # Find or create root folder
    root_id = _find_folder(parent_name)
    if not root_id:
        root_id = _create_folder(parent_name)
        print(f"  [storage] Created Drive folder: {parent_name}")

    # Find or create event subfolder
    event_id = _find_folder(folder_name, parent_id=root_id)
    if not event_id:
        event_id = _create_folder(folder_name, parent_id=root_id)
        print(f"  [storage] Created Drive folder: {parent_name}/{folder_name}")

    return event_id


def upload_csv_to_drive(df, filename, folder_name):
    """
    Upload a DataFrame as a CSV to a Google Drive folder.

    Args:
        df:          DataFrame to upload
        filename:    Target filename (e.g. 'simulated_probs_1430.csv')
        folder_name: Subfolder name under golf_sim_outputs/ (e.g. 'farmers_2026')
    """
    from googleapiclient.http import MediaInMemoryUpload

    if df is None or df.empty:
        print(f"  [storage] Skipping empty upload: {filename}")
        return

    service = _connect_drive()
    folder_id = _find_or_create_drive_folder(service, folder_name)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    media = MediaInMemoryUpload(csv_bytes, mimetype="text/csv")

    service.files().create(
        body={
            "name": filename,
            "parents": [folder_id],
            "mimeType": "text/csv",
        },
        media_body=media,
    ).execute()

    print(f"  [storage] Uploaded {filename} to Drive ({len(csv_bytes):,} bytes)")


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
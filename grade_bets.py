"""
grade_bets.py - Bet Results Tracker

Grades betting results by:
1. Reading ungraded bets from Google Sheets
2. Deduplicating bets (same event/player/market/book)
3. Fetching tournament results from DataGolf historical-raw-data/rounds API
4. Grading each bet with dead heat adjustments for finish positions
5. Writing results back to sheets + detailed results tabs
6. Updating rolling summary with performance metrics

Usage:
    python grade_bets.py                    # Grade last week's event (auto-detect)
    python grade_bets.py --event-id 123     # Grade specific event
    python grade_bets.py --dry-run          # Preview without writing

Outputs:
    Updates Google Sheets tabs:
    - Tournament Matchups, Round Matchups, Finish Positions (grades individual bets)
    - Matchups Results - Sharp/Retail/Other (detailed results by book category)
    - Finish Results - Sharp/Retail/Other (detailed results by book category)
    - Bet Results Summary (rolling performance metrics)
"""

import os
import argparse
import smtplib
from datetime import datetime, timedelta
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import numpy as np
import requests
import gspread
from dotenv import load_dotenv

load_dotenv()

from sim_inputs import name_replacements
from sheets_storage import get_spreadsheet, update_ledger_grades

# Email config
EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_KEY = os.getenv("DATAGOLF_API_KEY")
DATAGOLF_BASE = "https://feeds.datagolf.com"

# Tab names - source bets
TAB_TOURNAMENT_MU = "Tournament Matchups"
TAB_FINISH_POS = "Finish Positions"
TAB_ROUND_MU = "Round Matchups"
TAB_ROUND_3BALL = "Round 3-Balls"
TAB_LIVE = "Live"  # live in-tournament finish positions (bet_type=finish_position_live)
TAB_SCORE_EDGES = "Score Edges"  # round score O/U (bet_type=score_bet)

# Tab names - results output
TAB_RESULTS_SUMMARY = "Bet Results Summary"

# Book categorization
SHARP_BOOKS = ["pinnacle", "betonline", "betcris", "bet online", "bookmaker"]
RETAIL_BOOKS = ["fanduel", "draftkings", "caesars", "dk", "fd", "czr", "betmgm", "mgm"]

# Books that pay ties IN FULL on finish positions (no dead-heat reduction):
# the exchanges settle top-N as a binary, and Bovada pays ties in full by rule.
FULL_TIE_PAYOUT_BOOKS = ("kalshi", "novig", "bovada")

# Bet sizing assumptions
FLAT_BET_SIZE = 1.0  # 1 unit for matchups (flat betting)
UNIT_SIZE = 200.0    # $200 = 1 unit for finish position sizing
EXCLUDED_RESULT_PREFIX = "excluded_"

# Pred value buckets
PRED_BUCKETS = [
    (1.0, float('inf'), ">1"),
    (0.75, 1.0, "0.75-1"),
    (0.5, 0.75, "0.5-0.75"),
    (0.25, 0.5, "0.25-0.5"),
    (-0.25, 0.25, "-0.25-0.25"),
    (float('-inf'), -0.25, "<-0.25"),
]


def excluded_result_mask(results):
    """Identify audit-preserved bets that must never be graded or reported."""
    return results.fillna("").astype(str).str.lower().str.strip().str.startswith(
        EXCLUDED_RESULT_PREFIX
    )


def categorize_book(book_name):
    """Categorize a sportsbook as sharp, retail, or other."""
    if not book_name:
        return "other"
    book_lower = str(book_name).lower().strip()

    for sharp in SHARP_BOOKS:
        if sharp in book_lower:
            return "sharp"

    for retail in RETAIL_BOOKS:
        if retail in book_lower:
            return "retail"

    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# DataGolf API - Results Fetching
# ══════════════════════════════════════════════════════════════════════════════

def fetch_historical_results(event_id, year=None, tour="pga"):
    """
    Fetch historical tournament results from DataGolf historical-raw-data/rounds endpoint.

    Returns DataFrame with:
    - player_name: normalized player name
    - fin_text: finish position text (e.g., "1", "T5", "CUT")
    - fin_num: numeric finish position (for sorting)
    - round_1, round_2, round_3, round_4: round scores
    - total: total strokes
    """
    if year is None:
        year = datetime.now().year

    params = {
        "tour": tour,
        "event_id": str(event_id),
        "year": year,
        "file_format": "json",
        "key": API_KEY,
    }

    try:
        resp = requests.get(f"{DATAGOLF_BASE}/historical-raw-data/rounds", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"  Warning: No historical data for event {event_id}")
            return pd.DataFrame()

        # Extract player scores from nested structure
        if isinstance(data, dict) and "scores" in data:
            scores_data = data["scores"]
            print(f"  Event: {data.get('event_name', 'Unknown')}")
        else:
            scores_data = data

        if not scores_data:
            print(f"  Warning: No scores data for event {event_id}")
            return pd.DataFrame()

        df = pd.DataFrame(scores_data)

        # Debug: print column names
        print(f"  API columns: {list(df.columns)}")

        # Normalize player names
        if "player_name" in df.columns:
            df["player_name"] = df["player_name"].str.lower().str.strip().replace(name_replacements)

        # Parse fin_text to get numeric position
        if "fin_text" in df.columns:
            df["fin_num"] = df["fin_text"].apply(parse_finish_position)
        elif "position" in df.columns:
            df["fin_text"] = df["position"].astype(str)
            df["fin_num"] = df["position"].apply(parse_finish_position)

        # Extract round scores and SG categories from nested dicts
        # API returns round_1/2/3/4 as dicts with 'score', 'sg_ott', 'sg_app', etc.
        SG_CATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]
        for r in range(1, 5):
            col_name = f"round_{r}"
            if col_name in df.columns:
                # Check if it's a dict (nested structure)
                first_val = df[col_name].iloc[0] if len(df) > 0 else None
                if isinstance(first_val, dict):
                    # Extract SG categories before overwriting with score
                    for sg in SG_CATS:
                        df[f"r{r}_{sg}"] = df[col_name].apply(
                            lambda x, s=sg: x.get(s) if isinstance(x, dict) else None
                        )
                    # Extract 'score' from each dict
                    df[col_name] = df[col_name].apply(
                        lambda x: x.get('score') if isinstance(x, dict) else x
                    )

        # Calculate total if not present
        if "total" not in df.columns:
            round_cols = [f"round_{r}" for r in range(1, 5) if f"round_{r}" in df.columns]
            if round_cols:
                # Convert to numeric first
                for col in round_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df["total"] = df[round_cols].sum(axis=1, skipna=True)

        return df

    except Exception as e:
        print(f"  Error fetching historical results: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def parse_finish_position(pos_text):
    """Parse finish position text to numeric value."""
    if pd.isna(pos_text):
        return 999

    pos_str = str(pos_text).strip().upper()

    # Handle special cases
    if pos_str in ["CUT", "MC", "MDF"]:
        return 999
    if pos_str in ["WD", "W/D", "DQ"]:
        return 998

    # Remove T for ties
    pos_str = pos_str.replace("T", "").replace("=", "")

    try:
        return int(pos_str)
    except:
        return 999


def fetch_event_list(tour="pga"):
    """
    Fetch list of recent PGA events from DataGolf.

    Returns list of dicts with event_id, event_name, calendar_year, etc.
    """
    params = {
        "tour": tour,
        "file_format": "json",
        "key": API_KEY,
    }

    try:
        resp = requests.get(f"{DATAGOLF_BASE}/historical-raw-data/event-list", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Error fetching event list: {e}")
        return []


def get_last_completed_event():
    """
    Find the most recently completed PGA event.

    Returns (event_id, event_name, year) or (None, None, None) if not found.
    """
    events = fetch_event_list()
    if not events:
        return None, None, None

    current_year = datetime.now().year

    for event in events:
        if event.get("calendar_year") == current_year:
            event_id = event.get("event_id")
            event_name = event.get("event_name")

            # Try to fetch results to verify it's complete
            results = fetch_historical_results(event_id, current_year)
            if not results.empty and len(results) > 10:
                return event_id, event_name, current_year

    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Read Ungraded Bets from Sheets
# ══════════════════════════════════════════════════════════════════════════════

def read_sheet_as_df(spreadsheet, tab_name, max_retries=3):
    """Read a sheet tab into a DataFrame with rate-limit retry."""
    import time
    for attempt in range(max_retries):
        try:
            ws = spreadsheet.worksheet(tab_name)
            data = ws.get_all_values()
            if len(data) < 2:
                return pd.DataFrame()

            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        except gspread.exceptions.WorksheetNotFound:
            print(f"  Tab '{tab_name}' not found")
            return pd.DataFrame()
        except gspread.exceptions.APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Rate limited reading '{tab_name}', waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    print(f"  Failed to read '{tab_name}' after {max_retries} retries")
    return pd.DataFrame()


def get_ungraded_bets(spreadsheet, tab_name, event_id=None, regrade=False):
    """
    Get ungraded bets from a sheet tab.

    Ungraded = result column is empty.
    If regrade=True, returns ALL bets (including already-graded) for re-grading.
    Optionally filter by event_id.

    Returns DataFrame with row indices for updating.
    """
    df = read_sheet_as_df(spreadsheet, tab_name)

    if df.empty:
        return df

    # Find result column
    result_col = None
    for col in ["result", "Result"]:
        if col in df.columns:
            result_col = col
            break

    if result_col is None:
        print(f"  Warning: No 'result' column in {tab_name}")
        return pd.DataFrame()

    if regrade:
        # Return all bets except duplicates and audit-preserved exclusions.
        result = df[result_col].fillna("").astype(str).str.lower().str.strip()
        ungraded = df[(result != "duplicate") & ~excluded_result_mask(result)].copy()
    else:
        # Filter to ungraded (empty result)
        ungraded = df[df[result_col].astype(str).str.strip() == ""].copy()

    # Filter by event_id if specified
    if event_id is not None:
        event_col = None
        for col in ["event_id", "Event ID"]:
            if col in ungraded.columns:
                event_col = col
                break

        if event_col:
            ungraded = ungraded[ungraded[event_col].astype(str) == str(event_id)]

    # Add original row index (1-indexed, accounting for header)
    ungraded["_sheet_row"] = ungraded.index + 2

    return ungraded


def get_all_event_ids(spreadsheet):
    """Get all unique event IDs and names from bet tabs."""
    events = {}
    for tab_name in [TAB_TOURNAMENT_MU, TAB_ROUND_MU, TAB_FINISH_POS]:
        df = read_sheet_as_df(spreadsheet, tab_name)
        if df.empty:
            continue
        if "event_id" in df.columns and "event_name" in df.columns:
            for _, row in df[["event_id", "event_name"]].drop_duplicates().iterrows():
                eid = str(row["event_id"]).strip()
                ename = str(row["event_name"]).strip()
                if eid and eid != "":
                    events[eid] = ename
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_bets(df, bet_type):
    """
    Deduplicate bets by key columns.

    Keeps the FIRST occurrence (earliest timestamp).

    Returns:
        unique_df: DataFrame with unique bets
        duplicate_rows: List of sheet row numbers to mark as duplicates
    """
    if df.empty:
        return df, []

    # Define key columns based on bet type
    if bet_type == "round_matchup":
        key_cols = [
            "event_id", "player_1", "player_2", "bet_on", "bookmaker", "round",
            "p1_line", "p2_line", "line_verified",
        ]
    elif bet_type == "round_3ball":
        key_cols = ["event_id", "player_1", "player_2", "player_3", "bet_on", "bookmaker", "round"]
    elif bet_type == "tournament_matchup":
        key_cols = [
            "event_id", "player_1", "player_2", "bet_on", "bookmaker",
            "p1_line", "p2_line", "line_verified",
        ]
    elif bet_type.startswith("finish_position"):
        key_cols = ["event_id", "player_name", "market_type", "sportsbook"]
    elif bet_type == "score_bet":
        key_cols = ["event_id", "player", "round", "book", "best_side"]
    elif bet_type == "sharp":
        key_cols = ["event_id", "bet_type", "bet_on", "opponent", "bookmaker", "round"]
    else:
        key_cols = ["event_id", "bet_on", "bookmaker"]

    # Normalize key columns for matching
    df = df.copy()
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    available_keys = [c for c in key_cols if c in df.columns]

    if not available_keys:
        return df, []

    if "run_timestamp" in df.columns:
        df = df.sort_values("run_timestamp")

    df["_is_dup"] = df.duplicated(subset=available_keys, keep="first")

    unique_df = df[~df["_is_dup"]].copy()
    duplicate_rows = df[df["_is_dup"]]["_sheet_row"].tolist()

    unique_df = unique_df.drop(columns=["_is_dup"])

    return unique_df, duplicate_rows


# ══════════════════════════════════════════════════════════════════════════════
# Dead Heat Calculation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_dead_heat_factor(actual_finish, market_threshold, results_df):
    """
    Calculate dead heat factor for finish position bets.

    If multiple players tie at a position that straddles the market threshold,
    the payout is reduced proportionally.

    Args:
        actual_finish: The player's finish position (numeric)
        market_threshold: The market cutoff (e.g., 5 for top_5)
        results_df: DataFrame with all player results

    Returns:
        (num_tied, dead_heat_factor)
        - num_tied: Number of players tied at that position
        - dead_heat_factor: Proportion of payout (0 to 1)
    """
    if actual_finish > market_threshold:
        # Clearly outside the money
        return 1, 0.0

    if "fin_num" not in results_df.columns:
        return 1, 1.0

    # Count players tied at this position
    num_tied = len(results_df[results_df["fin_num"] == actual_finish])

    if num_tied <= 1:
        # No tie, full payout
        return 1, 1.0

    # Calculate how many "paying spots" remain at this position
    # If tied for 5th in top-5, only 1 spot remaining (5th place)
    # If tied for 4th in top-5, 2 spots remaining (4th and 5th)
    spots_remaining = market_threshold - actual_finish + 1

    # Dead heat factor = min(spots_remaining, num_tied) / num_tied
    spots_paid = min(spots_remaining, num_tied)
    dead_heat_factor = spots_paid / num_tied

    return num_tied, dead_heat_factor


# ══════════════════════════════════════════════════════════════════════════════
# Grading Logic
# ══════════════════════════════════════════════════════════════════════════════

def _ties_lose(row):
    """True when this matchup was priced/stored under a ties-LOSE book rule.

    The pricing layer treats 'separate bet offered' as a 3-way market where a
    tie loses the stake (new_sim/round_sim/reprice_core: use_tl = Ties ==
    'separate bet offered'); grading previously settled every tie as push
    regardless, overstating P&L at those books (2026-08 audit, −13.84u).
    """
    v = str(row.get("ties_rule", row.get("Ties", ""))).lower()
    return "separate" in v


def _truthy(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _opposite_spread(value):
    """Return the reciprocal line without letting malformed audit data crash grading."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return -number if np.isfinite(number) else np.nan


def _selected_matchup_spread(row):
    """Return (selected_line, error) for a supported matchup contract.

    Legacy BetCRIS/Bookmaker rows are unknowable because the old scraper threw
    away the source Line column. They stay untouched for audit and are not
    graded. New rows must explicitly prove either straight or reciprocal +/-0.5.
    """
    book = str(row.get("bookmaker", row.get("Bookmaker", ""))).lower().strip()
    verified = _truthy(row.get("line_verified", False))
    is_betcris = "betcris" in book or "bookmaker" in book
    if is_betcris and not verified:
        return None, "unknown BetCRIS handicap provenance"

    def parse(value):
        if value is None or str(value).strip().lower() in ("", "nan", "none"):
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    p1_line = parse(row.get("p1_line", row.get("P1 Line", 0.0)))
    p2_line = parse(row.get("p2_line", row.get("P2 Line", 0.0)))
    if p1_line is None or p2_line is None:
        return None, "unparseable matchup handicap"
    straight = abs(p1_line) <= 1e-9 and abs(p2_line) <= 1e-9
    half = (
        abs(abs(p1_line) - 0.5) <= 1e-9
        and abs(abs(p2_line) - 0.5) <= 1e-9
        and abs(p1_line + p2_line) <= 1e-9
    )
    if not (straight or (verified and half)):
        return None, "unsupported or unverified matchup handicap"

    p1 = str(row.get("player_1", row.get("Player 1", ""))).lower().strip()
    bet_on = str(row.get("bet_on", "")).lower().strip()
    return (p1_line if bet_on == p1 else p2_line), None


def _settle_two_way_matchup(winner, bet_on, spread_line, ties_lose, units_wagered, units_to_win):
    if winner == "tie":
        if spread_line > 0:
            return "win", units_to_win
        if spread_line < 0 or ties_lose:
            return "loss", -units_wagered
        return "push", 0.0
    if winner == bet_on:
        return "win", units_to_win
    return "loss", -units_wagered


def grade_round_matchup(row, results_df):
    """
    Grade a round matchup bet.

    Returns dict with all grading details.
    """
    p1 = str(row.get("player_1", "")).lower().strip()
    p2 = str(row.get("player_2", "")).lower().strip()
    bet_on = str(row.get("bet_on", "")).lower().strip()
    round_num = str(row.get("round", "1")).strip()
    spread_line, spread_error = _selected_matchup_spread(row)
    if spread_error:
        return {
            "result": "no_data", "p1_score": "", "p2_score": "",
            "units_wagered": 0, "units_won": 0,
            "notes": spread_error, "skip_write": True,
        }

    # Handle round number
    try:
        round_num = int(round_num)
    except:
        round_num = 1

    # Get round column name - try multiple formats
    score_col = None
    for col_format in [f"round_{round_num}", f"r{round_num}", f"rd{round_num}", f"round{round_num}"]:
        if col_format in results_df.columns:
            score_col = col_format
            break

    if score_col is None:
        return {
            "result": "no_data",
            "p1_score": "",
            "p2_score": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": f"No round {round_num} column found"
        }

    # Find players
    p1_data = results_df[results_df["player_name"] == p1]
    p2_data = results_df[results_df["player_name"] == p2]

    if p1_data.empty:
        return {
            "result": "no_data",
            "p1_score": "",
            "p2_score": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": f"Player not found: {p1}"
        }

    if p2_data.empty:
        return {
            "result": "no_data",
            "p1_score": "",
            "p2_score": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": f"Player not found: {p2}"
        }

    p1_score = p1_data[score_col].iloc[0]
    p2_score = p2_data[score_col].iloc[0]

    if pd.isna(p1_score) or pd.isna(p2_score):
        return {
            "result": "no_data",
            "p1_score": str(p1_score) if not pd.isna(p1_score) else "",
            "p2_score": str(p2_score) if not pd.isna(p2_score) else "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": "Missing round score"
        }

    try:
        p1_score = float(p1_score)
        p2_score = float(p2_score)
    except:
        return {
            "result": "no_data",
            "p1_score": "",
            "p2_score": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": "Could not parse scores"
        }

    # Determine winner (lower score wins)
    if p1_score < p2_score:
        winner = p1
    elif p2_score < p1_score:
        winner = p2
    else:
        winner = "tie"

    # Get odds for bet
    if bet_on == p1:
        book_odds = row.get("p1_odds")
    else:
        book_odds = row.get("p2_odds")

    # Calculate units wagered based on odds
    # If odds >= +100: Risk 1 unit to win (odds/100) units
    # If odds < -100: Risk (abs(odds)/100) units to win 1 unit
    units_wagered = FLAT_BET_SIZE
    units_to_win = FLAT_BET_SIZE

    if book_odds and str(book_odds).strip():
        try:
            odds = float(book_odds)
            if odds >= 100:
                # Plus money: risk 1 to win more
                units_wagered = 1.0
                units_to_win = odds / 100
            elif odds <= -100:
                # Minus money: risk more to win 1
                units_wagered = abs(odds) / 100
                units_to_win = 1.0
            else:
                # Edge case (odds between -100 and +100, shouldn't happen)
                units_wagered = 1.0
                units_to_win = 1.0
        except:
            units_wagered = 1.0
            units_to_win = 1.0

    result, units_won = _settle_two_way_matchup(
        winner, bet_on, spread_line, _ties_lose(row), units_wagered, units_to_win
    )

    # Margin: positive = our player won by N strokes (lower score is better)
    opponent = p2 if bet_on == p1 else p1
    opp_score = p2_score if bet_on == p1 else p1_score
    bet_on_score = p1_score if bet_on == p1 else p2_score
    margin = opp_score - bet_on_score  # positive = we won by this many

    # SG category lookups
    sg_fields = {}
    bet_on_data = results_df[results_df["player_name"] == bet_on]
    opp_data = results_df[results_df["player_name"] == opponent]
    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        col = f"r{round_num}_{cat}"
        sg_fields[f"bet_on_{cat}"] = float(bet_on_data[col].iloc[0]) if not bet_on_data.empty and col in bet_on_data.columns and pd.notna(bet_on_data[col].iloc[0]) else None
        sg_fields[f"opp_{cat}"] = float(opp_data[col].iloc[0]) if not opp_data.empty and col in opp_data.columns and pd.notna(opp_data[col].iloc[0]) else None

    return {
        "result": result,
        "p1_score": int(p1_score),
        "p2_score": int(p2_score),
        "units_wagered": round(units_wagered, 3),
        "units_won": round(units_won, 3),
        "spread_line": spread_line,
        "margin": margin,
        **sg_fields,
        "notes": ""
    }


def grade_round_3ball(row, results_df):
    """Grade a round 3-ball bet (bet_on to have the lowest score in the threesome).

    3-balls settle by DEAD HEAT: if bet_on ties for the low with N players, the bet
    is a 1/N fractional win at full odds (the rest of the stake loses) —
    units_won = units_wagered * (decimal_odds * (1/N) - 1). A sole low is a full win
    (N=1); not the low is a full loss. Mirrors grade_round_matchup for score lookup.
    """
    players = [str(row.get(f"player_{i}", "")).lower().strip() for i in (1, 2, 3)]
    bet_on = str(row.get("bet_on", "")).lower().strip()
    round_num = str(row.get("round", "1")).strip()
    try:
        round_num = int(round_num)
    except Exception:
        round_num = 1

    score_col = None
    for col_format in [f"round_{round_num}", f"r{round_num}", f"rd{round_num}", f"round{round_num}"]:
        if col_format in results_df.columns:
            score_col = col_format
            break

    base = {"result": "no_data", "p1_score": "", "p2_score": "", "p3_score": "",
            "units_wagered": FLAT_BET_SIZE, "units_won": 0,
            "num_tied": "", "dead_heat_factor": "", "margin": None}
    if score_col is None:
        return {**base, "notes": f"No round {round_num} column found"}

    scores = []
    for p in players:
        pdata = results_df[results_df["player_name"] == p]
        if pdata.empty or pd.isna(pdata[score_col].iloc[0]):
            return {**base, "notes": f"Missing round score for {p}"}
        try:
            scores.append(float(pdata[score_col].iloc[0]))
        except Exception:
            return {**base, "notes": "Could not parse scores"}

    if bet_on not in players:
        return {**base, "notes": f"bet_on {bet_on} not in threesome"}

    # Book odds on the bet_on side -> unit sizing + decimal odds for dead-heat math
    si = players.index(bet_on) + 1
    book_odds = row.get(f"p{si}_odds")
    units_wagered, units_to_win, decimal_odds = 1.0, 1.0, 2.0
    if book_odds is not None and str(book_odds).strip():
        try:
            odds = float(book_odds)
            if odds >= 100:
                units_wagered, units_to_win, decimal_odds = 1.0, odds / 100, odds / 100 + 1
            elif odds <= -100:
                units_wagered, units_to_win, decimal_odds = abs(odds) / 100, 1.0, 100 / abs(odds) + 1
        except Exception:
            pass

    low = min(scores)
    num_tied = sum(1 for s in scores if s == low)
    bet_score = scores[si - 1]

    if bet_score > low:
        result, units_won, dh_factor = "loss", -units_wagered, 0.0
    elif num_tied == 1:
        result, units_won, dh_factor = "win", units_to_win, 1.0
    else:
        dh_factor = 1.0 / num_tied
        units_won = units_wagered * (decimal_odds * dh_factor - 1)
        result = "dead_heat"

    opp_scores = [scores[i] for i in range(3) if players[i] != bet_on]
    margin = min(opp_scores) - bet_score  # positive = bet_on ahead of the best opponent

    return {
        "result": result,
        "p1_score": int(scores[0]), "p2_score": int(scores[1]), "p3_score": int(scores[2]),
        "units_wagered": round(units_wagered, 3),
        "units_won": round(units_won, 3),
        "num_tied": num_tied,
        "dead_heat_factor": round(dh_factor, 4),
        "margin": margin,
        "notes": "",
    }


def grade_tournament_matchup(row, results_df):
    """
    Grade a tournament matchup bet (72-hole totals or best finish).

    Returns dict with all grading details.
    """
    p1 = str(row.get("player_1", "")).lower().strip()
    p2 = str(row.get("player_2", "")).lower().strip()
    bet_on = str(row.get("bet_on", "")).lower().strip()
    spread_line, spread_error = _selected_matchup_spread(row)
    if spread_error:
        return {
            "result": "no_data", "p1_finish": "", "p2_finish": "",
            "units_wagered": 0, "units_won": 0,
            "notes": spread_error, "skip_write": True,
        }

    p1_data = results_df[results_df["player_name"] == p1]
    p2_data = results_df[results_df["player_name"] == p2]

    if p1_data.empty:
        return {
            "result": "no_data",
            "p1_finish": "",
            "p2_finish": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": f"Player not found: {p1}"
        }

    if p2_data.empty:
        return {
            "result": "no_data",
            "p1_finish": "",
            "p2_finish": "",
            "units_wagered": FLAT_BET_SIZE,
            "units_won": 0,
            "notes": f"Player not found: {p2}"
        }

    # Use fin_num for comparison (lower is better)
    p1_fin = p1_data["fin_num"].iloc[0] if "fin_num" in p1_data.columns else 999
    p2_fin = p2_data["fin_num"].iloc[0] if "fin_num" in p2_data.columns else 999

    p1_fin_text = p1_data["fin_text"].iloc[0] if "fin_text" in p1_data.columns else str(p1_fin)
    p2_fin_text = p2_data["fin_text"].iloc[0] if "fin_text" in p2_data.columns else str(p2_fin)

    # Determine winner (lower finish position wins)
    # WD (998) = auto-loss: player who withdrew loses to anyone who didn't
    # Both WD = push
    # Both MC (999) = compare R1+R2 strokes; lower total wins, tie = push
    if p1_fin == 998 and p2_fin == 998:
        winner = "tie"
    elif p1_fin == 998 and p2_fin != 998:
        winner = p2
    elif p2_fin == 998 and p1_fin != 998:
        winner = p1
    elif p1_fin == 999 and p2_fin == 999:
        # Both missed cut — compare R1+R2 scores (lower total wins)
        p1_r1 = pd.to_numeric(p1_data["round_1"].iloc[0], errors="coerce") if "round_1" in p1_data.columns else np.nan
        p1_r2 = pd.to_numeric(p1_data["round_2"].iloc[0], errors="coerce") if "round_2" in p1_data.columns else np.nan
        p2_r1 = pd.to_numeric(p2_data["round_1"].iloc[0], errors="coerce") if "round_1" in p2_data.columns else np.nan
        p2_r2 = pd.to_numeric(p2_data["round_2"].iloc[0], errors="coerce") if "round_2" in p2_data.columns else np.nan
        p1_total = (p1_r1 if not np.isnan(p1_r1) else 0) + (p1_r2 if not np.isnan(p1_r2) else 0)
        p2_total = (p2_r1 if not np.isnan(p2_r1) else 0) + (p2_r2 if not np.isnan(p2_r2) else 0)
        has_scores = not (np.isnan(p1_r1) and np.isnan(p1_r2)) and not (np.isnan(p2_r1) and np.isnan(p2_r2))
        if not has_scores:
            winner = "tie"  # no round data → push
        elif p1_total < p2_total:
            winner = p1
        elif p2_total < p1_total:
            winner = p2
        else:
            winner = "tie"
    elif p1_fin < p2_fin:
        winner = p1
    elif p2_fin < p1_fin:
        winner = p2
    else:
        winner = "tie"

    # Get odds
    if bet_on == p1:
        book_odds = row.get("p1_odds")
    else:
        book_odds = row.get("p2_odds")

    # Calculate units wagered based on odds
    # If odds >= +100: Risk 1 unit to win (odds/100) units
    # If odds < -100: Risk (abs(odds)/100) units to win 1 unit
    units_wagered = FLAT_BET_SIZE
    units_to_win = FLAT_BET_SIZE

    if book_odds and str(book_odds).strip():
        try:
            odds = float(book_odds)
            if odds >= 100:
                # Plus money: risk 1 to win more
                units_wagered = 1.0
                units_to_win = odds / 100
            elif odds <= -100:
                # Minus money: risk more to win 1
                units_wagered = abs(odds) / 100
                units_to_win = 1.0
            else:
                # Edge case (odds between -100 and +100, shouldn't happen)
                units_wagered = 1.0
                units_to_win = 1.0
        except:
            units_wagered = 1.0
            units_to_win = 1.0

    result, units_won = _settle_two_way_matchup(
        winner, bet_on, spread_line, _ties_lose(row), units_wagered, units_to_win
    )

    # Margin: compare strokes over the SAME rounds both players completed
    # If one MC'd and other didn't, margin is incomparable (set None)
    opponent = p2 if bet_on == p1 else p1
    margin = None
    bet_on_mc = (p1_fin >= 999) if bet_on == p1 else (p2_fin >= 999)
    opp_mc = (p2_fin >= 999) if bet_on == p1 else (p1_fin >= 999)
    if bet_on_mc != opp_mc:
        # One MC'd, one didn't — not comparable
        margin = None
    else:
        # Both same status: compare R1+R2 if both MC, full total if both made cut
        rounds_to_sum = [1, 2] if bet_on_mc else [1, 2, 3, 4]
        bet_on_sum = 0.0
        opp_sum = 0.0
        has_data = True
        for r in rounds_to_sum:
            col = f"round_{r}"
            if col not in results_df.columns:
                has_data = False
                break
            b_val = pd.to_numeric(p1_data[col].iloc[0] if bet_on == p1 else p2_data[col].iloc[0], errors="coerce")
            o_val = pd.to_numeric(p2_data[col].iloc[0] if bet_on == p1 else p1_data[col].iloc[0], errors="coerce")
            if pd.isna(b_val) or pd.isna(o_val):
                has_data = False
                break
            bet_on_sum += b_val
            opp_sum += o_val
        if has_data:
            margin = opp_sum - bet_on_sum

    # SG categories: sum across all rounds played
    sg_fields = {}
    bet_on_data = results_df[results_df["player_name"] == bet_on]
    opp_data = results_df[results_df["player_name"] == opponent]
    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        bet_on_sum = 0.0
        opp_sum = 0.0
        has_bet_on = False
        has_opp = False
        for r in range(1, 5):
            col = f"r{r}_{cat}"
            if col in results_df.columns:
                if not bet_on_data.empty and pd.notna(bet_on_data[col].iloc[0]):
                    bet_on_sum += float(bet_on_data[col].iloc[0])
                    has_bet_on = True
                if not opp_data.empty and pd.notna(opp_data[col].iloc[0]):
                    opp_sum += float(opp_data[col].iloc[0])
                    has_opp = True
        sg_fields[f"bet_on_{cat}"] = round(bet_on_sum, 3) if has_bet_on else None
        sg_fields[f"opp_{cat}"] = round(opp_sum, 3) if has_opp else None

    return {
        "result": result,
        "p1_finish": p1_fin_text,
        "p2_finish": p2_fin_text,
        "units_wagered": round(units_wagered, 3),
        "units_won": round(units_won, 3),
        "spread_line": spread_line,
        "margin": margin,
        **sg_fields,
        "notes": ""
    }


def grade_finish_position(row, results_df):
    """
    Grade a finish position bet (win, top_5, top_10, top_20) with dead heat adjustment.

    Returns dict with all grading details.
    """
    player = str(row.get("player_name", "")).lower().strip()
    market = str(row.get("market_type", "")).lower().strip().replace(" ", "_")

    # Exchange (Kalshi/NoVig) NO bets are encoded with a "_no" suffix on the
    # market_type (e.g. "top_20_no" = bet the player finishes OUTSIDE the top 20).
    # Strip it to recover the base market + side.
    side = "no" if market.endswith("_no") else "yes"
    base_market = market[:-3] if side == "no" else market

    player_data = results_df[results_df["player_name"] == player]

    if player_data.empty:
        return {
            "result": "no_data",
            "actual_finish": "",
            "num_tied": "",
            "dead_heat_factor": "",
            "units_wagered": 0,
            "units_won": 0,
            "notes": f"Player not found: {player}"
        }

    # Get finish position
    actual_pos = player_data["fin_num"].iloc[0] if "fin_num" in player_data.columns else 999
    fin_text = player_data["fin_text"].iloc[0] if "fin_text" in player_data.columns else str(actual_pos)

    # Market thresholds
    market_thresholds = {
        "win": 1,
        "winner": 1,
        "top_5": 5,
        "top_10": 10,
        "top_20": 20,
        "top5": 5,
        "top10": 10,
        "top20": 20,
    }

    threshold = market_thresholds.get(base_market, 1)

    # Get kelly stake (stored as raw dollars, convert to units: $200 = 1 unit)
    kelly_stake = row.get("kelly_stake", "")
    if kelly_stake and str(kelly_stake).strip():
        try:
            stake_dollars = float(kelly_stake)
            stake = stake_dollars / UNIT_SIZE
        except:
            stake = FLAT_BET_SIZE
    else:
        stake = FLAT_BET_SIZE

    units_wagered = stake

    # Get decimal odds
    decimal_odds = row.get("decimal_odds", "")
    if not decimal_odds or not str(decimal_odds).strip():
        # Try to convert from american
        american = row.get("american_odds", row.get("odds", ""))
        if american and str(american).strip():
            try:
                am = float(american)
                if am > 0:
                    decimal_odds = am / 100 + 1
                else:
                    decimal_odds = 100 / abs(am) + 1
            except:
                decimal_odds = 2.0
        else:
            decimal_odds = 2.0
    else:
        try:
            decimal_odds = float(decimal_odds)
        except:
            decimal_odds = 2.0

    # Settlement. Books that pay ties IN FULL — exchanges (Kalshi/NoVig) and
    # Bovada — and every NO-side bet, which only exists on exchanges, settle
    # with NO dead-heat: a tie at the threshold counts as finishing INSIDE.
    # YES wins inside the threshold, NO wins outside.
    # All other sportsbook YES bets keep standard dead-heat settlement.
    book = str(row.get("sportsbook", row.get("bookmaker", ""))).lower().strip()
    pays_ties_full = book in FULL_TIE_PAYOUT_BOOKS

    if side == "no" or pays_ties_full:
        inside = actual_pos <= threshold
        won = inside if side == "yes" else not inside
        result = "win" if won else "loss"
        units_won = units_wagered * (decimal_odds - 1) if won else -units_wagered
        num_tied, dead_heat_factor = "", 1.0
    else:
        # Calculate dead heat factor
        num_tied, dead_heat_factor = calculate_dead_heat_factor(actual_pos, threshold, results_df)

        if actual_pos > threshold:
            # Loss - finished outside threshold
            result = "loss"
            units_won = -units_wagered
        elif dead_heat_factor == 1.0:
            # Full win - no dead heat
            result = "win"
            units_won = units_wagered * (decimal_odds - 1)
        elif dead_heat_factor > 0:
            # Partial win due to dead heat
            result = "win_dh"  # Dead heat win
            # Standard dead-heat settlement: stake splits into a winning fraction
            # (factor) settled at full odds and a losing fraction (1 - factor) that
            # is LOST, not pushed. Net P&L = stake * (decimal_odds * factor - 1).
            units_won = units_wagered * (decimal_odds * dead_heat_factor - 1)
        else:
            # Dead heat resulted in effective loss (rare edge case)
            result = "loss"
            units_won = -units_wagered

    # SG categories: sum bet_on player's categories across all rounds
    sg_fields = {}
    for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
        total = 0.0
        has_data = False
        for r in range(1, 5):
            col = f"r{r}_{cat}"
            if col in results_df.columns and not player_data.empty and pd.notna(player_data[col].iloc[0]):
                total += float(player_data[col].iloc[0])
                has_data = True
        sg_fields[f"bet_on_{cat}"] = round(total, 3) if has_data else None
        sg_fields[f"opp_{cat}"] = None  # No opponent for finish positions

    return {
        "result": result,
        "actual_finish": fin_text,
        "num_tied": num_tied,
        "dead_heat_factor": round(dead_heat_factor, 3),
        "units_wagered": round(units_wagered, 3),
        "units_won": round(units_won, 3),
        "margin": None,
        **sg_fields,
        "notes": ""
    }


def grade_sharp_bet(row, results_df):
    """
    Grade a sharp filtered bet (unified format).

    Returns dict with grading details.
    """
    bet_type = str(row.get("bet_type", "")).lower().strip()

    if bet_type == "round_matchup":
        p1 = str(row.get("bet_on", "")).lower().strip()
        p2 = str(row.get("opponent", "")).lower().strip()
        round_num = row.get("round", 1)

        mock_row = {
            "player_1": p1,
            "player_2": p2,
            "bet_on": p1,
            "round": round_num,
            "p1_odds": row.get("book_odds"),
            "p1_line": row.get("spread_line", 0.0),
            "p2_line": _opposite_spread(row.get("spread_line", 0.0)),
            "line_verified": row.get("line_verified", False),
            "bookmaker": row.get("bookmaker", ""),
        }
        return grade_round_matchup(mock_row, results_df)

    elif bet_type == "tournament_matchup":
        p1 = str(row.get("bet_on", "")).lower().strip()
        p2 = str(row.get("opponent", "")).lower().strip()

        mock_row = {
            "player_1": p1,
            "player_2": p2,
            "bet_on": p1,
            "p1_odds": row.get("book_odds"),
            "p1_line": row.get("spread_line", 0.0),
            "p2_line": _opposite_spread(row.get("spread_line", 0.0)),
            "line_verified": row.get("line_verified", False),
            "bookmaker": row.get("bookmaker", ""),
        }
        return grade_tournament_matchup(mock_row, results_df)

    elif bet_type == "finish_position":
        player = str(row.get("bet_on", "")).lower().strip()
        market = str(row.get("opponent", "")).lower().strip()

        mock_row = {
            "player_name": player,
            "market_type": market,
            "kelly_stake": row.get("kelly_stake", FLAT_BET_SIZE),
            "decimal_odds": None,
        }

        # Convert american to decimal
        american = row.get("book_odds")
        if american and str(american).strip():
            try:
                am = float(american)
                if am > 0:
                    mock_row["decimal_odds"] = am / 100 + 1
                else:
                    mock_row["decimal_odds"] = 100 / abs(am) + 1
            except:
                pass

        return grade_finish_position(mock_row, results_df)

    return {
        "result": "unknown",
        "units_wagered": 0,
        "units_won": 0,
        "notes": f"Unknown bet type: {bet_type}"
    }


def grade_score_bet(row, results_df):
    """
    Grade a round score bet (over/under a line).

    Restored from b5562d5 (removed by the 6b606d1 production-copy promotion;
    the family then ran ungraded for 18 weeks). The bet is on a player's
    actual round score vs a line (e.g., 70.5); best_side is 'Over' or 'Under'.
    """
    player = str(row.get("player", "")).lower().strip()
    round_num = str(row.get("round", "1")).strip()
    best_side = str(row.get("best_side", "")).strip()

    try:
        line = float(row.get("line", 0))
    except (ValueError, TypeError):
        return {"result": "no_data", "actual_score": "", "units_wagered": FLAT_BET_SIZE, "units_won": 0}

    try:
        round_num = int(round_num)
    except (ValueError, TypeError):
        round_num = 1

    # Get the book odds for the side we bet
    if best_side.lower() == "under":
        odds_str = row.get("mkt_under", "")
    else:
        odds_str = row.get("mkt_over", "")

    # Convert american odds to decimal
    decimal_odds = 1.0
    try:
        am = float(odds_str)
        if am > 0:
            decimal_odds = am / 100 + 1
        else:
            decimal_odds = 100 / abs(am) + 1
    except (ValueError, TypeError):
        pass

    # Find round score column
    score_col = None
    for col_format in [f"round_{round_num}", f"r{round_num}", f"rd{round_num}", f"round{round_num}"]:
        if col_format in results_df.columns:
            score_col = col_format
            break

    if score_col is None:
        return {"result": "no_data", "actual_score": "", "units_wagered": FLAT_BET_SIZE, "units_won": 0}

    # Find player
    player_data = results_df[results_df["player_name"] == player]
    if player_data.empty:
        return {"result": "no_data", "actual_score": "", "units_wagered": FLAT_BET_SIZE, "units_won": 0}

    actual_score = player_data[score_col].iloc[0]
    if pd.isna(actual_score):
        return {"result": "no_data", "actual_score": "", "units_wagered": FLAT_BET_SIZE, "units_won": 0}

    try:
        actual_score = float(actual_score)
    except (ValueError, TypeError):
        return {"result": "no_data", "actual_score": "", "units_wagered": FLAT_BET_SIZE, "units_won": 0}

    # Grade: compare actual score to line
    if actual_score > line:
        outcome = "Over"
    elif actual_score < line:
        outcome = "Under"
    else:
        outcome = "push"

    if outcome == "push":
        result = "push"
        units_won = 0.0
    elif outcome.lower() == best_side.lower():
        result = "win"
        units_won = round(FLAT_BET_SIZE * (decimal_odds - 1), 3)
    else:
        result = "loss"
        units_won = round(-FLAT_BET_SIZE, 3)

    return {
        "result": result,
        "actual_score": actual_score,
        "units_wagered": FLAT_BET_SIZE,
        "units_won": units_won,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Write Results to Sheets
# ══════════════════════════════════════════════════════════════════════════════

def write_grades_to_sheet(spreadsheet, tab_name, grades, headers_list):
    """
    Write graded results back to source sheet.

    grades: list of dicts with {row: int, result: str, units_won: float, ...}
    """
    try:
        ws = spreadsheet.worksheet(tab_name)
        data = ws.get_all_values()
        headers = data[0] if data else []

        cells_to_update = []

        for grade in grades:
            row = grade["row"]

            if "result" in headers and "result" in grade:
                col = headers.index("result") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["result"])))

            if "units_won" in headers and "units_won" in grade:
                col = headers.index("units_won") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["units_won"])))

            if "p1_round_score" in headers and "p1_score" in grade:
                col = headers.index("p1_round_score") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["p1_score"])))

            if "p2_round_score" in headers and "p2_score" in grade:
                col = headers.index("p2_round_score") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["p2_score"])))

            if "p3_round_score" in headers and "p3_score" in grade:
                col = headers.index("p3_round_score") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["p3_score"])))

            if "actual_finish" in headers and "actual_finish" in grade:
                col = headers.index("actual_finish") + 1
                cells_to_update.append(gspread.Cell(row, col, str(grade["actual_finish"])))

        if cells_to_update:
            ws.update_cells(cells_to_update, value_input_option="USER_ENTERED")
            print(f"  Updated {len(grades)} rows in '{tab_name}'")

    except Exception as e:
        print(f"  Error writing to {tab_name}: {e}")




# ══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_SUMMARY_HEADERS = [
    "event_name", "event_id", "year", "week_of",
    # Overall
    "total_bets", "wins", "losses", "pushes", "no_data",
    "win_rate", "units_wagered", "units_won", "roi",
    # By bet type
    "round_mu_bets", "round_mu_units", "round_mu_roi",
    "tourn_mu_bets", "tourn_mu_units", "tourn_mu_roi",
    "finish_bets", "finish_units", "finish_roi",
    # By edge bucket
    "edge_3_5_bets", "edge_3_5_roi",
    "edge_5_8_bets", "edge_5_8_roi",
    "edge_8_plus_bets", "edge_8_plus_roi",
    # By book category
    "sharp_bets", "sharp_roi",
    "retail_bets", "retail_roi",
    "other_bets", "other_roi",
    # By pred value bucket
    "pred_gt1_bets", "pred_gt1_roi",
    "pred_075_1_bets", "pred_075_1_roi",
    "pred_05_075_bets", "pred_05_075_roi",
    "pred_025_05_bets", "pred_025_05_roi",
    "pred_neg025_025_bets", "pred_neg025_025_roi",
    "pred_lt_neg025_bets", "pred_lt_neg025_roi",
]


def calculate_performance_metrics(all_graded_bets, event_name, event_id, year):
    """
    Calculate comprehensive performance metrics.

    Returns dict with all metrics for the summary tab.
    """
    if not all_graded_bets:
        return None

    df = pd.DataFrame(all_graded_bets)
    df = df[~excluded_result_mask(df["result"])].copy()

    if df.empty:
        return None

    # Basic counts
    total = len(df)
    wins = len(df[df["result"].isin(["win", "win_dh"])])
    losses = len(df[df["result"] == "loss"])
    pushes = len(df[df["result"] == "push"])
    no_data = len(df[df["result"].isin(["no_data", "unknown"])])

    # Units - use actual wagered amounts
    df["units_won_num"] = pd.to_numeric(df["units_won"], errors="coerce").fillna(0)
    df["units_wagered_num"] = pd.to_numeric(df["units_wagered"], errors="coerce").fillna(1)

    # Only count wagered for bets that resolved (not no_data)
    resolved = df[~df["result"].isin(["no_data", "unknown", "duplicate"])]
    units_won = resolved["units_won_num"].sum()
    units_wagered = resolved["units_wagered_num"].sum()

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    roi = units_won / units_wagered * 100 if units_wagered > 0 else 0

    metrics = {
        "event_name": event_name,
        "event_id": str(event_id),
        "year": year,
        "week_of": datetime.now().strftime("%Y-%m-%d"),
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "no_data": no_data,
        "win_rate": round(win_rate, 1),
        "units_wagered": round(units_wagered, 2),
        "units_won": round(units_won, 2),
        "roi": round(roi, 1),
    }

    # By bet type
    for bt, prefix in [("round_matchup", "round_mu"), ("tournament_matchup", "tourn_mu"), ("finish_position", "finish")]:
        subset = resolved[resolved["bet_type"] == bt]
        count = len(subset)
        wagered = subset["units_wagered_num"].sum()
        units = subset["units_won_num"].sum()
        bt_roi = units / wagered * 100 if wagered > 0 else 0
        metrics[f"{prefix}_bets"] = count
        metrics[f"{prefix}_units"] = round(units, 2)
        metrics[f"{prefix}_roi"] = round(bt_roi, 1)

    # By edge bucket
    if "edge" in resolved.columns:
        resolved = resolved.copy()
        resolved["edge_num"] = pd.to_numeric(resolved["edge"], errors="coerce").fillna(0)

        for low, high, label in [(3, 5, "edge_3_5"), (5, 8, "edge_5_8"), (8, 100, "edge_8_plus")]:
            subset = resolved[(resolved["edge_num"] >= low) & (resolved["edge_num"] < high)]
            count = len(subset)
            wagered = subset["units_wagered_num"].sum()
            units = subset["units_won_num"].sum()
            bucket_roi = units / wagered * 100 if wagered > 0 else 0
            metrics[f"{label}_bets"] = count
            metrics[f"{label}_roi"] = round(bucket_roi, 1)

    # By book category
    if "book_category" in resolved.columns:
        for cat in ["sharp", "retail", "other"]:
            subset = resolved[resolved["book_category"] == cat]
            count = len(subset)
            wagered = subset["units_wagered_num"].sum()
            units = subset["units_won_num"].sum()
            cat_roi = units / wagered * 100 if wagered > 0 else 0
            metrics[f"{cat}_bets"] = count
            metrics[f"{cat}_roi"] = round(cat_roi, 1)

    # By pred value bucket
    if "pred_value" in resolved.columns:
        resolved = resolved.copy()
        resolved["pred_num"] = pd.to_numeric(resolved["pred_value"], errors="coerce")

        bucket_labels = ["pred_gt1", "pred_075_1", "pred_05_075", "pred_025_05", "pred_neg025_025", "pred_lt_neg025"]
        for (low, high, _), label in zip(PRED_BUCKETS, bucket_labels):
            subset = resolved[(resolved["pred_num"] >= low) & (resolved["pred_num"] < high)]
            count = len(subset)
            wagered = subset["units_wagered_num"].sum()
            units = subset["units_won_num"].sum()
            bucket_roi = units / wagered * 100 if wagered > 0 else 0
            metrics[f"{label}_bets"] = count
            metrics[f"{label}_roi"] = round(bucket_roi, 1)

    return metrics


def write_summary_row(spreadsheet, metrics):
    """Write (upsert) a summary row to the Bet Results Summary tab.

    If a row for this event_id already exists, it is replaced in-place.
    Otherwise a new row is appended.
    """
    try:
        try:
            ws = spreadsheet.worksheet(TAB_RESULTS_SUMMARY)
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title=TAB_RESULTS_SUMMARY, rows=1000, cols=len(RESULTS_SUMMARY_HEADERS))
            ws.append_row(RESULTS_SUMMARY_HEADERS, value_input_option="RAW")
            print(f"  Created '{TAB_RESULTS_SUMMARY}' tab")

        row = [metrics.get(h, "") for h in RESULTS_SUMMARY_HEADERS]

        # Upsert: find ALL existing rows by event_id, delete them, then append
        event_id = str(metrics.get("event_id", "")).strip()
        rows_to_delete = []
        if event_id:
            all_values = ws.get_all_values()
            headers = all_values[0] if all_values else []
            eid_col = headers.index("event_id") if "event_id" in headers else None
            if eid_col is not None:
                for i, r in enumerate(all_values[1:], start=2):
                    if str(r[eid_col]).strip() == event_id:
                        rows_to_delete.append(i)

        if rows_to_delete:
            # Delete from bottom up so indices stay valid
            for row_idx in sorted(rows_to_delete, reverse=True):
                ws.delete_rows(row_idx)
            ws.append_row(row, value_input_option="USER_ENTERED")
            print(f"  Replaced {len(rows_to_delete)} summary row(s) for {metrics.get('event_name')}")
        else:
            ws.append_row(row, value_input_option="USER_ENTERED")
            print(f"  Added summary row for {metrics.get('event_name')}")

    except Exception as e:
        print(f"  Error writing summary: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Email Summary Report
# ══════════════════════════════════════════════════════════════════════════════

def build_results_email_html(metrics, graded_bets, event_name, filter_label=None, clv_stats=None):
    """Build HTML email with betting results summary including detailed breakdowns.

    Args:
        metrics: Performance metrics dict
        graded_bets: List of graded bet dicts
        event_name: Tournament name
        filter_label: Optional label for filtered results (e.g., "Pred > 0.75")
        clv_stats: Optional dict from clv.clv_summary_stats() — adds a
            closing-line-value strip under the summary box
    """
    total = metrics.get("total_bets", 0)
    wins = metrics.get("wins", 0)
    losses = metrics.get("losses", 0)
    pushes = metrics.get("pushes", 0)
    units_won = metrics.get("units_won", 0)
    roi = metrics.get("roi", 0)
    win_rate = metrics.get("win_rate", 0)

    # Convert graded_bets to DataFrame for analysis
    if graded_bets:
        df = pd.DataFrame(graded_bets)
        df["units_won_num"] = pd.to_numeric(df["units_won"], errors="coerce").fillna(0)
        df["units_wagered_num"] = pd.to_numeric(df["units_wagered"], errors="coerce").fillna(1)
        df["pred_num"] = pd.to_numeric(df.get("pred_value", ""), errors="coerce")
        resolved = df[
            ~df["result"].isin(["no_data", "unknown", "duplicate"])
            & ~excluded_result_mask(df["result"])
        ]

        # Calculate aggregate $ PnL: all bet types now in units, multiply by $200/unit
        matchup_types = ["round_matchup", "tournament_matchup"]
        matchup_pnl = resolved[resolved["bet_type"].isin(matchup_types)]["units_won_num"].sum() * UNIT_SIZE
        finish_pnl = resolved[resolved["bet_type"] == "finish_position"]["units_won_num"].sum() * UNIT_SIZE
        aggregate_pnl = matchup_pnl + finish_pnl
    else:
        df = pd.DataFrame()
        resolved = pd.DataFrame()
        aggregate_pnl = 0

    roi_color = "#28a745" if aggregate_pnl >= 0 else "#dc3545"
    roi_bg = "#d4edda" if aggregate_pnl >= 0 else "#f8d7da"

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: Results by Round (Matchups) and Market (Finish Positions)
    # ═══════════════════════════════════════════════════════════════════════

    # Round matchups breakdown
    round_rows = ""
    if not resolved.empty:
        matchups = resolved[resolved["bet_type"].isin(["round_matchup", "tournament_matchup"])]
        for round_val, label in [("1", "R1"), ("2", "R2"), ("3", "R3"), ("4", "R4"), ("tournament", "Tournament")]:
            subset = matchups[matchups["round"].astype(str) == round_val] if "round" in matchups.columns else pd.DataFrame()
            if len(subset) > 0:
                bets = len(subset)
                won = subset["units_won_num"].sum()
                wagered = subset["units_wagered_num"].sum()
                r_roi = won / wagered * 100 if wagered > 0 else 0
                color = "#28a745" if won >= 0 else "#dc3545"
                round_rows += f"""
                <tr>
                    <td style="padding:6px 10px;">{label}</td>
                    <td style="padding:6px 10px; text-align:center;">{bets}</td>
                    <td style="padding:6px 10px; text-align:center; color:{color}; font-weight:600;">{won:+.2f}</td>
                    <td style="padding:6px 10px; text-align:center; color:{color};">{r_roi:+.1f}%</td>
                </tr>"""

    # Finish positions by market
    market_rows = ""
    if not resolved.empty:
        finish = resolved[resolved["bet_type"] == "finish_position"]
        for market, label in [("win", "Outright"), ("top_5", "Top 5"), ("top_10", "Top 10"), ("top_20", "Top 20")]:
            subset = finish[finish["market_type"].astype(str).str.lower().str.replace(" ", "_") == market] if "market_type" in finish.columns else pd.DataFrame()
            if len(subset) > 0:
                bets = len(subset)
                won = subset["units_won_num"].sum()
                wagered = subset["units_wagered_num"].sum()
                m_roi = won / wagered * 100 if wagered > 0 else 0
                color = "#28a745" if won >= 0 else "#dc3545"
                market_rows += f"""
                <tr>
                    <td style="padding:6px 10px;">{label}</td>
                    <td style="padding:6px 10px; text-align:center;">{bets}</td>
                    <td style="padding:6px 10px; text-align:center; color:{color}; font-weight:600;">${won:+.2f}</td>
                    <td style="padding:6px 10px; text-align:center; color:{color};">{m_roi:+.1f}%</td>
                </tr>"""

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: Book Breakdown for Matchups
    # ═══════════════════════════════════════════════════════════════════════

    book_matchup_rows = ""
    if not resolved.empty:
        matchups = resolved[resolved["bet_type"].isin(["round_matchup", "tournament_matchup"])]
        books = matchups["bookmaker"].dropna().unique() if "bookmaker" in matchups.columns else []

        for book in sorted(books):
            if not book:
                continue
            book_data = matchups[matchups["bookmaker"] == book]
            row_cells = [f'<td style="padding:6px 8px; font-weight:600;">{book.title()}</td>']

            total_won = 0
            total_wagered = 0

            for round_val in ["1", "2", "3", "4", "tournament"]:
                subset = book_data[book_data["round"].astype(str) == round_val] if "round" in book_data.columns else pd.DataFrame()
                if len(subset) > 0:
                    won = subset["units_won_num"].sum()
                    total_won += won
                    total_wagered += subset["units_wagered_num"].sum()
                    color = "#28a745" if won >= 0 else "#dc3545"
                    row_cells.append(f'<td style="padding:6px 8px; text-align:center; color:{color};">{won:+.2f}</td>')
                else:
                    row_cells.append('<td style="padding:6px 8px; text-align:center; color:#999;">-</td>')

            # Cumulative ROI
            cum_roi = total_won / total_wagered * 100 if total_wagered > 0 else 0
            color = "#28a745" if cum_roi >= 0 else "#dc3545"
            row_cells.append(f'<td style="padding:6px 8px; text-align:center; color:{color}; font-weight:600;">{cum_roi:+.1f}%</td>')

            book_matchup_rows += f"<tr>{''.join(row_cells)}</tr>"

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: Book Breakdown for Finish Positions
    # ═══════════════════════════════════════════════════════════════════════

    book_finish_rows = ""
    if not resolved.empty:
        finish = resolved[resolved["bet_type"] == "finish_position"]
        books = finish["bookmaker"].dropna().unique() if "bookmaker" in finish.columns else []

        for book in sorted(books):
            if not book:
                continue
            book_data = finish[finish["bookmaker"] == book]
            row_cells = [f'<td style="padding:6px 8px; font-weight:600;">{book.title()}</td>']

            total_won = 0
            total_wagered = 0

            for market in ["win", "top_5", "top_10", "top_20"]:
                subset = book_data[book_data["market_type"].astype(str).str.lower().str.replace(" ", "_") == market] if "market_type" in book_data.columns else pd.DataFrame()
                if len(subset) > 0:
                    won = subset["units_won_num"].sum()
                    total_won += won
                    total_wagered += subset["units_wagered_num"].sum()
                    color = "#28a745" if won >= 0 else "#dc3545"
                    row_cells.append(f'<td style="padding:6px 8px; text-align:center; color:{color};">${won:+.2f}</td>')
                else:
                    row_cells.append('<td style="padding:6px 8px; text-align:center; color:#999;">-</td>')

            # Cumulative ROI
            cum_roi = total_won / total_wagered * 100 if total_wagered > 0 else 0
            color = "#28a745" if cum_roi >= 0 else "#dc3545"
            row_cells.append(f'<td style="padding:6px 8px; text-align:center; color:{color}; font-weight:600;">{cum_roi:+.1f}%</td>')

            book_finish_rows += f"<tr>{''.join(row_cells)}</tr>"

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: Pred Bucket by Market
    # ═══════════════════════════════════════════════════════════════════════

    pred_market_rows = ""
    if not resolved.empty and "pred_num" in resolved.columns:
        pred_buckets = [
            (1.0, float('inf'), ">1"),
            (0.75, 1.0, "0.75-1"),
            (0.5, 0.75, "0.5-0.75"),
            (0.25, 0.5, "0.25-0.5"),
            (-0.25, 0.25, "-0.25-0.25"),
            (float('-inf'), -0.25, "<-0.25"),
        ]

        # Markets to include
        markets = [
            ("round_matchup", "1", "R1"),
            ("round_matchup", "2", "R2"),
            ("round_matchup", "3", "R3"),
            ("round_matchup", "4", "R4"),
            ("tournament_matchup", "tournament", "Tourn"),
            ("finish_position", "win", "Win"),
            ("finish_position", "top_5", "T5"),
            ("finish_position", "top_10", "T10"),
            ("finish_position", "top_20", "T20"),
        ]

        for low, high, bucket_label in pred_buckets:
            bucket_subset = resolved[(resolved["pred_num"] >= low) & (resolved["pred_num"] < high)]
            if len(bucket_subset) == 0:
                continue

            row_cells = [f'<td style="padding:6px 8px; font-weight:600;">{bucket_label}</td>']

            for bet_type, market_val, _ in markets:
                if bet_type == "finish_position":
                    if "market_type" in bucket_subset.columns:
                        subset = bucket_subset[(bucket_subset["bet_type"] == bet_type) &
                                              (bucket_subset["market_type"].astype(str).str.lower().str.replace(" ", "_") == market_val)]
                    else:
                        subset = pd.DataFrame()
                else:
                    if "round" in bucket_subset.columns:
                        subset = bucket_subset[(bucket_subset["bet_type"] == bet_type) &
                                              (bucket_subset["round"].astype(str) == market_val)]
                    else:
                        subset = pd.DataFrame()

                if len(subset) > 0:
                    won = subset["units_won_num"].sum()
                    wagered = subset["units_wagered_num"].sum()
                    r_roi = won / wagered * 100 if wagered > 0 else 0
                    color = "#28a745" if r_roi >= 0 else "#dc3545"
                    row_cells.append(f'<td style="padding:6px 8px; text-align:center; color:{color};">{r_roi:+.0f}%</td>')
                else:
                    row_cells.append('<td style="padding:6px 8px; text-align:center; color:#999;">-</td>')

            pred_market_rows += f"<tr>{''.join(row_cells)}</tr>"

    # ═══════════════════════════════════════════════════════════════════════
    # Top Wins and Losses
    # ═══════════════════════════════════════════════════════════════════════

    top_wins_html = ""
    top_losses_html = ""

    if not df.empty:
        top_wins = df[df["result"].isin(["win", "win_dh"])].nlargest(5, "units_won_num")
        for _, row in top_wins.iterrows():
            # Get player name - check multiple possible columns
            player = ""
            for col in ["player_name", "bet_on", "player_1"]:
                if col in row and row.get(col):
                    player = str(row.get(col, "")).strip()
                    if player:
                        break
            player = player.title() if player else "Unknown"

            # Get market/bet type description
            bt = row.get("bet_type", "")
            if bt == "finish_position":
                market = str(row.get("market_type", "")).replace("_", " ").title()
                desc = market if market else "Finish"
            elif bt == "round_matchup":
                rd = row.get("round", "")
                desc = f"R{rd} Matchup" if rd else "Round Matchup"
            elif bt == "tournament_matchup":
                desc = "Tournament Matchup"
            else:
                desc = str(bt).replace("_", " ").title()

            units = row["units_won_num"]
            top_wins_html += f"""
            <tr>
                <td style="padding:6px 10px;">{player}</td>
                <td style="padding:6px 10px;">{desc}</td>
                <td style="padding:6px 10px; text-align:right; color:#28a745; font-weight:600;">+${units:.0f}</td>
            </tr>"""

        top_losses = df[df["result"] == "loss"].nsmallest(5, "units_won_num")
        for _, row in top_losses.iterrows():
            # Get player name - check multiple possible columns
            player = ""
            for col in ["player_name", "bet_on", "player_1"]:
                if col in row and row.get(col):
                    player = str(row.get(col, "")).strip()
                    if player:
                        break
            player = player.title() if player else "Unknown"

            # Get market/bet type description
            bt = row.get("bet_type", "")
            if bt == "finish_position":
                market = str(row.get("market_type", "")).replace("_", " ").title()
                desc = market if market else "Finish"
            elif bt == "round_matchup":
                rd = row.get("round", "")
                desc = f"R{rd} Matchup" if rd else "Round Matchup"
            elif bt == "tournament_matchup":
                desc = "Tournament Matchup"
            else:
                desc = str(bt).replace("_", " ").title()

            units = row["units_won_num"]
            top_losses_html += f"""
            <tr>
                <td style="padding:6px 10px;">{player}</td>
                <td style="padding:6px 10px;">{desc}</td>
                <td style="padding:6px 10px; text-align:right; color:#dc3545; font-weight:600;">-${abs(units):.0f}</td>
            </tr>"""

    # ═══════════════════════════════════════════════════════════════════════
    # CLV strip (closing line value vs DataGolf's archived closes)
    # ═══════════════════════════════════════════════════════════════════════

    clv_section = ""
    if clv_stats:
        sharp_avg = clv_stats.get("sharp_avg_clv")
        sharp_color = "#28a745" if (sharp_avg or 0) >= 0 else "#dc3545"
        sharp_val = f"{sharp_avg:+.2f}pp" if sharp_avg is not None else "n/a"
        overall_color = "#28a745" if clv_stats["avg_clv"] >= 0 else "#dc3545"

        fam_cells = ""
        for fam, label in [("matchup", "Matchups"), ("finish", "Finish Pos")]:
            avg = clv_stats.get(f"{fam}_avg_clv")
            if avg is None:
                continue
            fam_color = "#28a745" if avg >= 0 else "#dc3545"
            fam_cells += f"""
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:22px; font-weight:700; color:{fam_color};">{avg:+.2f}pp</div>
                        <div style="color:#666; font-size:12px;">{label} (n={clv_stats.get(f'{fam}_n', 0)})</div>
                    </div>"""

        clv_section = f"""
            <!-- CLV vs Close -->
            <div style="background:#f0f4f8; border-radius:8px; padding:14px 20px; margin-bottom:24px;">
                <div style="color:#2c5282; font-weight:600; font-size:13px; margin-bottom:6px;">
                    CLOSING LINE VALUE &nbsp;<span style="color:#999; font-weight:400;">(implied close − implied bet; positive = beat the close)</span>
                </div>
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:22px; font-weight:700; color:{sharp_color};">{sharp_val}</div>
                        <div style="color:#666; font-size:12px;">vs Sharp Close (n={clv_stats.get('sharp_n', 0)})</div>
                    </div>
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:22px; font-weight:700; color:{overall_color};">{clv_stats['avg_clv']:+.2f}pp</div>
                        <div style="color:#666; font-size:12px;">All Books (n={clv_stats['n']})</div>
                    </div>
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:22px; font-weight:700; color:#333;">{clv_stats['pos_pct']:.0f}%</div>
                        <div style="color:#666; font-size:12px;">Beat Close</div>
                    </div>
                    {fam_cells}
                </div>
            </div>"""

    # ═══════════════════════════════════════════════════════════════════════
    # Build HTML
    # ═══════════════════════════════════════════════════════════════════════

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:900px; margin:0 auto; padding:20px; background:#f5f5f5;">
        <div style="background:white; border-radius:8px; padding:24px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">

            <h1 style="margin:0 0 4px 0; color:#1a1a1a;">Betting Results: {event_name}</h1>
            {f'<p style="color:#2c5282; font-weight:600; margin:0 0 8px 0;">Filter: {filter_label}</p>' if filter_label else ''}
            <p style="color:#666; margin:0 0 24px 0;">{datetime.now().strftime('%B %d, %Y')}</p>

            <!-- Overall Summary Box -->
            <div style="background:{roi_bg}; border-radius:8px; padding:20px; margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:32px; font-weight:700; color:{roi_color};">${aggregate_pnl:+,.0f}</div>
                        <div style="color:#666; font-size:13px;">$ Won</div>
                    </div>
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:32px; font-weight:700; color:{roi_color};">{roi:+.1f}%</div>
                        <div style="color:#666; font-size:13px;">ROI</div>
                    </div>
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:32px; font-weight:700; color:#333;">{wins}-{losses}-{pushes}</div>
                        <div style="color:#666; font-size:13px;">W-L-P</div>
                    </div>
                    <div style="text-align:center; padding:10px 20px;">
                        <div style="font-size:32px; font-weight:700; color:#333;">{win_rate:.1f}%</div>
                        <div style="color:#666; font-size:13px;">Win Rate</div>
                    </div>
                </div>
            </div>
            {clv_section}
            <!-- Results by Round/Market -->
            <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:24px;">
                <div style="flex:1; min-width:300px;">
                    <h3 style="color:#2c5282; margin:0 0 12px 0; border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
                        Matchups by Round
                    </h3>
                    <table style="width:100%; border-collapse:collapse; font-size:13px;">
                        <tr style="background:#f7fafc;">
                            <th style="padding:8px 10px; text-align:left;">Round</th>
                            <th style="padding:8px 10px; text-align:center;">Bets</th>
                            <th style="padding:8px 10px; text-align:center;">Units</th>
                            <th style="padding:8px 10px; text-align:center;">ROI</th>
                        </tr>
                        {round_rows if round_rows else '<tr><td colspan="4" style="padding:10px; color:#999;">No matchup bets</td></tr>'}
                    </table>
                </div>
                <div style="flex:1; min-width:300px;">
                    <h3 style="color:#2c5282; margin:0 0 12px 0; border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
                        Finish Positions by Market
                    </h3>
                    <table style="width:100%; border-collapse:collapse; font-size:13px;">
                        <tr style="background:#f7fafc;">
                            <th style="padding:8px 10px; text-align:left;">Market</th>
                            <th style="padding:8px 10px; text-align:center;">Bets</th>
                            <th style="padding:8px 10px; text-align:center;">$</th>
                            <th style="padding:8px 10px; text-align:center;">ROI</th>
                        </tr>
                        {market_rows if market_rows else '<tr><td colspan="4" style="padding:10px; color:#999;">No finish bets</td></tr>'}
                    </table>
                </div>
            </div>

            <!-- Book Breakdown: Matchups -->
            <h3 style="color:#2c5282; margin:24px 0 12px 0; border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
                Matchups by Sportsbook
            </h3>
            <table style="width:100%; border-collapse:collapse; font-size:12px; margin-bottom:20px;">
                <tr style="background:#f7fafc;">
                    <th style="padding:8px 8px; text-align:left;">Book</th>
                    <th style="padding:8px 8px; text-align:center;">R1</th>
                    <th style="padding:8px 8px; text-align:center;">R2</th>
                    <th style="padding:8px 8px; text-align:center;">R3</th>
                    <th style="padding:8px 8px; text-align:center;">R4</th>
                    <th style="padding:8px 8px; text-align:center;">Tourn</th>
                    <th style="padding:8px 8px; text-align:center;">ROI</th>
                </tr>
                {book_matchup_rows if book_matchup_rows else '<tr><td colspan="7" style="padding:10px; color:#999;">No matchup data</td></tr>'}
            </table>

            <!-- Book Breakdown: Finish Positions -->
            <h3 style="color:#2c5282; margin:24px 0 12px 0; border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
                Finish Positions by Sportsbook
            </h3>
            <table style="width:100%; border-collapse:collapse; font-size:12px; margin-bottom:20px;">
                <tr style="background:#f7fafc;">
                    <th style="padding:8px 8px; text-align:left;">Book</th>
                    <th style="padding:8px 8px; text-align:center;">Win</th>
                    <th style="padding:8px 8px; text-align:center;">Top 5</th>
                    <th style="padding:8px 8px; text-align:center;">Top 10</th>
                    <th style="padding:8px 8px; text-align:center;">Top 20</th>
                    <th style="padding:8px 8px; text-align:center;">ROI</th>
                </tr>
                {book_finish_rows if book_finish_rows else '<tr><td colspan="6" style="padding:10px; color:#999;">No finish data</td></tr>'}
            </table>

            <!-- Pred Bucket by Market -->
            <h3 style="color:#2c5282; margin:24px 0 12px 0; border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
                ROI by Pred Value Bucket
            </h3>
            <table style="width:100%; border-collapse:collapse; font-size:11px; margin-bottom:20px;">
                <tr style="background:#f7fafc;">
                    <th style="padding:6px 8px; text-align:left;">Pred</th>
                    <th style="padding:6px 8px; text-align:center;">R1</th>
                    <th style="padding:6px 8px; text-align:center;">R2</th>
                    <th style="padding:6px 8px; text-align:center;">R3</th>
                    <th style="padding:6px 8px; text-align:center;">R4</th>
                    <th style="padding:6px 8px; text-align:center;">Tourn</th>
                    <th style="padding:6px 8px; text-align:center;">Win</th>
                    <th style="padding:6px 8px; text-align:center;">T5</th>
                    <th style="padding:6px 8px; text-align:center;">T10</th>
                    <th style="padding:6px 8px; text-align:center;">T20</th>
                </tr>
                {pred_market_rows if pred_market_rows else '<tr><td colspan="10" style="padding:10px; color:#999;">No pred data</td></tr>'}
            </table>

            <!-- Top Wins and Losses -->
            <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:24px;">
                <div style="flex:1; min-width:250px;">
                    <h3 style="color:#28a745; margin:0 0 12px 0;">Top Wins</h3>
                    <table style="width:100%; border-collapse:collapse; font-size:13px; background:#f8fff8; border-radius:6px;">
                        {top_wins_html if top_wins_html else '<tr><td style="padding:10px; color:#666;">No wins</td></tr>'}
                    </table>
                </div>
                <div style="flex:1; min-width:250px;">
                    <h3 style="color:#dc3545; margin:0 0 12px 0;">Top Losses</h3>
                    <table style="width:100%; border-collapse:collapse; font-size:13px; background:#fff8f8; border-radius:6px;">
                        {top_losses_html if top_losses_html else '<tr><td style="padding:10px; color:#666;">No losses</td></tr>'}
                    </table>
                </div>
            </div>

            <p style="color:#999; font-size:11px; margin-top:30px; text-align:center;">
                Generated by Golf Sim Bet Tracker | {total} total bets graded
            </p>
        </div>
    </body>
    </html>
    """

    return html


def send_results_email(metrics, graded_bets, event_name, filter_label=None, clv_stats=None):
    """Send betting results email summary.

    Args:
        metrics: Performance metrics dict
        graded_bets: List of graded bet dicts
        event_name: Tournament name
        filter_label: Optional filter description for subject line
        clv_stats: Optional CLV summary dict (from clv.clv_summary_stats)
    """
    if not EMAIL_PASSWORD:
        print("  Warning: EMAIL_PASSWORD not set. Skipping email.")
        return False

    if not EMAIL_TO or not EMAIL_TO[0]:
        print("  Warning: EMAIL_RECIPIENTS not set. Skipping email.")
        return False

    try:
        html = build_results_email_html(metrics, graded_bets, event_name, filter_label, clv_stats=clv_stats)

        units_won = metrics.get("units_won", 0)
        emoji = "+" if units_won >= 0 else ""

        # Build subject line
        if filter_label:
            subject = f"Betting Results: {event_name} [{filter_label}] ({emoji}{units_won:.2f})"
        else:
            subject = f"Betting Results: {event_name} ({emoji}{units_won:.2f} units)"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        label_str = f" ({filter_label})" if filter_label else ""
        print(f"  Results email{label_str} sent successfully")
        return True

    except Exception as e:
        print(f"  Warning: Email failed: {e}")
        return False


def send_filtered_results_email(all_graded_bets, event_name, event_id, year, pred_threshold=0.75):
    """Send a filtered email for bets above a pred threshold.

    Args:
        all_graded_bets: List of all graded bet dicts
        event_name: Tournament name
        event_id: Event ID
        year: Year
        pred_threshold: Minimum pred value to include (default 0.75)
    """
    # Filter bets by pred value
    filtered_bets = []
    for bet in all_graded_bets:
        pred_val = bet.get("pred_value", "")
        if pred_val:
            try:
                pred_num = float(pred_val)
                if pred_num > pred_threshold:
                    filtered_bets.append(bet)
            except (ValueError, TypeError):
                pass

    if not filtered_bets:
        print(f"  No bets with pred > {pred_threshold} found, skipping filtered email")
        return False

    # Calculate metrics for filtered subset
    metrics = calculate_performance_metrics(filtered_bets, event_name, event_id, year)
    if not metrics:
        return False

    filter_label = f"Pred > {pred_threshold}"
    print(f"  Sending filtered email ({len(filtered_bets)} bets with pred > {pred_threshold})...")
    return send_results_email(metrics, filtered_bets, event_name, filter_label)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Grade betting results")
    parser.add_argument("--event-id", type=str, help="Specific event ID to grade")
    parser.add_argument("--event-name", type=str, help="Event name (for display)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--no-email", action="store_true", help="Skip sending email report")
    parser.add_argument("--no-clv", action="store_true", help="Skip closing line value annotation")
    parser.add_argument("--regrade", action="store_true", help="Re-grade already-graded bets (overwrites existing grades)")
    parser.add_argument("--all-events", action="store_true", help="Grade all events in sheets (use with --regrade to re-grade everything)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BET RESULTS GRADER")
    print("="*60)

    if args.regrade:
        print("  MODE: Re-grade (overwriting existing grades)")

    # Connect to sheets early (needed for --all-events discovery)
    print("\n  Connecting to Google Sheets...")
    spreadsheet = get_spreadsheet()

    # Build list of (event_id, event_name, year) to process
    events_to_grade = []
    if args.all_events:
        print("\n  Discovering all events from sheets...")
        all_events = get_all_event_ids(spreadsheet)
        if not all_events:
            print("  No events found in sheets.")
            return
        for eid, ename in sorted(all_events.items()):
            events_to_grade.append((eid, ename, datetime.now().year))
        print(f"  Found {len(events_to_grade)} events: {[e[1] for e in events_to_grade]}")
    elif args.event_id:
        event_id = args.event_id
        event_name = args.event_name or f"Event {event_id}"
        year = datetime.now().year
        events_to_grade.append((event_id, event_name, year))
    else:
        print("\n  Auto-detecting last completed event...")
        event_id, event_name, year = get_last_completed_event()
        if not event_id:
            print("  Could not find last completed event. Use --event-id to specify.")
            return
        events_to_grade.append((event_id, event_name, year))

    grand_total_graded = []
    clv_stats = None

    for evt_idx, (event_id, event_name, year) in enumerate(events_to_grade):
        if evt_idx > 0:
            import time; time.sleep(15)  # throttle between events for Sheets rate limit
        print(f"\n{'='*60}")
        print(f"  Event: {event_name}")
        print(f"  Event ID: {event_id}")
        print(f"  Year: {year}")

        # Fetch results using historical endpoint
        print("\n  Fetching tournament results from historical-raw-data/rounds...")
        results_df = fetch_historical_results(event_id, year)

        if results_df.empty:
            print("  No results data available. Skipping.")
            continue

        print(f"  Loaded {len(results_df)} players from results")

        # Debug: show sample of data
        if "fin_text" in results_df.columns:
            print(f"  Sample finishes: {results_df['fin_text'].head(10).tolist()}")
        if "round_1" in results_df.columns:
            print(f"  Round 1 scores available: Yes")

        all_graded_bets = []

        # Process each bet tab
        tabs_to_process = [
            (TAB_ROUND_MU, "round_matchup", grade_round_matchup),
            (TAB_ROUND_3BALL, "round_3ball", grade_round_3ball),
            (TAB_TOURNAMENT_MU, "tournament_matchup", grade_tournament_matchup),
            (TAB_FINISH_POS, "finish_position", grade_finish_position),
            ("Test Sim", "finish_position_v2", grade_finish_position),
            # Live in-tournament finish positions. Same grader as Finish Positions
            # (identical headers); deduplicate_bets collapses repriced snapshots to
            # one bet per (event, player, market, book) and marks the rest duplicate.
            (TAB_LIVE, "finish_position_live", grade_finish_position),
            (TAB_SCORE_EDGES, "score_bet", grade_score_bet),
        ]

        for tab_idx, (tab_name, bet_type, grade_fn) in enumerate(tabs_to_process):
            if tab_idx > 0:
                import time; time.sleep(5)  # avoid Sheets rate limit
            print(f"\n  Processing {tab_name}...")

            ungraded = get_ungraded_bets(spreadsheet, tab_name, event_id, regrade=args.regrade)

            if ungraded.empty:
                print(f"    No {'bets' if args.regrade else 'ungraded bets'} found")
                continue

            label = "bets to re-grade" if args.regrade else "ungraded bets"
            print(f"    Found {len(ungraded)} {label}")

            unique, duplicates = deduplicate_bets(ungraded, bet_type)

            if duplicates:
                print(f"    Marking {len(duplicates)} duplicates")

            print(f"    Grading {len(unique)} unique bets")

            grades = []
            for _, row in unique.iterrows():
                row_dict = row.to_dict()
                sheet_row = row_dict["_sheet_row"]

                grade_result = grade_fn(row_dict, results_df)
                if grade_result.get("skip_write"):
                    print(
                        f"    Leaving row {sheet_row} ungraded: "
                        f"{grade_result.get('notes', 'unknown matchup contract')}"
                    )
                    continue

                # Get bookmaker for categorization
                bookmaker = row_dict.get("bookmaker", row_dict.get("sportsbook", ""))
                book_category = categorize_book(bookmaker)

                # Get pred value
                if bet_type.startswith("finish_position"):
                    pred_value = row_dict.get("my_pred", "")
                    sample = row_dict.get("sample", "")
                else:
                    pred_value = row_dict.get("pred_on", "")
                    sample = row_dict.get("sample_on", "")

                grade = {
                    "row": sheet_row,
                    "result": grade_result["result"],
                    "units_wagered": grade_result.get("units_wagered", FLAT_BET_SIZE),
                    "units_won": grade_result.get("units_won", 0),
                    "bet_type": bet_type,
                    "edge": row_dict.get("edge_on", row_dict.get("edge", "")),
                    "bookmaker": bookmaker,
                    "book_category": book_category,
                    "pred_value": pred_value,
                    "sample": sample,
                }

                # Add margin + SG attribution fields
                grade["margin"] = grade_result.get("margin")
                for sg_field in ["bet_on_sg_ott", "bet_on_sg_app", "bet_on_sg_arg", "bet_on_sg_putt",
                                  "opp_sg_ott", "opp_sg_app", "opp_sg_arg", "opp_sg_putt"]:
                    grade[sg_field] = grade_result.get(sg_field)

                # Add type-specific fields
                if bet_type == "round_matchup":
                    grade["round"] = row_dict.get("round", "")
                    grade["p1_score"] = grade_result.get("p1_score", "")
                    grade["p2_score"] = grade_result.get("p2_score", "")
                    grade["bet_on"] = row_dict.get("bet_on", "")
                    grade["player_1"] = row_dict.get("player_1", "")
                    grade["player_2"] = row_dict.get("player_2", "")
                    grade["book_odds"] = row_dict.get("p1_odds", "") if str(row_dict.get("bet_on", "")).lower() == str(row_dict.get("player_1", "")).lower() else row_dict.get("p2_odds", "")
                    grade["spread_line"] = grade_result.get("spread_line", 0.0)

                elif bet_type == "round_3ball":
                    grade["round"] = row_dict.get("round", "")
                    grade["p1_score"] = grade_result.get("p1_score", "")
                    grade["p2_score"] = grade_result.get("p2_score", "")
                    grade["p3_score"] = grade_result.get("p3_score", "")
                    grade["num_tied"] = grade_result.get("num_tied", "")
                    grade["dead_heat_factor"] = grade_result.get("dead_heat_factor", "")
                    _p3b = [str(row_dict.get(f"player_{i}", "")).lower().strip() for i in (1, 2, 3)]
                    _beton3 = str(row_dict.get("bet_on", "")).lower().strip()
                    grade["bet_on"] = row_dict.get("bet_on", "")
                    grade["player_1"] = row_dict.get("player_1", "")
                    grade["player_2"] = row_dict.get("player_2", "")
                    grade["player_3"] = row_dict.get("player_3", "")
                    # Must match the ledger's opponent encoding ("oppA + oppB", sorted),
                    # else update_ledger_grades won't find the row.
                    grade["opponent"] = " + ".join(sorted(p for p in _p3b if p != _beton3))
                    _si3 = _p3b.index(_beton3) + 1 if _beton3 in _p3b else 1
                    grade["book_odds"] = row_dict.get(f"p{_si3}_odds", "")

                elif bet_type == "tournament_matchup":
                    grade["round"] = "tournament"
                    grade["p1_finish"] = grade_result.get("p1_finish", "")
                    grade["p2_finish"] = grade_result.get("p2_finish", "")
                    grade["bet_on"] = row_dict.get("bet_on", "")
                    grade["player_1"] = row_dict.get("player_1", "")
                    grade["player_2"] = row_dict.get("player_2", "")
                    grade["book_odds"] = row_dict.get("p1_odds", "") if str(row_dict.get("bet_on", "")).lower() == str(row_dict.get("player_1", "")).lower() else row_dict.get("p2_odds", "")
                    grade["spread_line"] = grade_result.get("spread_line", 0.0)

                elif bet_type.startswith("finish_position"):
                    grade["market_type"] = row_dict.get("market_type", "")
                    grade["actual_finish"] = grade_result.get("actual_finish", "")
                    grade["num_tied"] = grade_result.get("num_tied", "")
                    grade["dead_heat_factor"] = grade_result.get("dead_heat_factor", "")
                    grade["player_name"] = row_dict.get("player_name", "")

                elif bet_type == "score_bet":
                    _side = str(row_dict.get("best_side", "")).lower()
                    grade["round"] = row_dict.get("round", "")
                    grade["line"] = row_dict.get("line", "")
                    grade["best_side"] = row_dict.get("best_side", "")
                    grade["actual_score"] = grade_result.get("actual_score", "")
                    grade["bet_on"] = str(row_dict.get("player", "")).lower().strip()
                    # Ledger opponent encoding for score bets: side_line
                    grade["opponent"] = f"{_side}_{row_dict.get('line', '')}"
                    grade["book_odds"] = (row_dict.get("mkt_under", "") if _side == "under"
                                          else row_dict.get("mkt_over", ""))
                    grade["fair_odds"] = (row_dict.get("fair_under", "") if _side == "under"
                                          else row_dict.get("fair_over", ""))
                    grade["edge"] = row_dict.get("best_edge", row_dict.get("edge", ""))
                    grade["bookmaker"] = row_dict.get("book", bookmaker)
                    grade["book_category"] = categorize_book(grade["bookmaker"])

                grades.append(grade)
                if not bet_type.endswith("_v2"):
                    all_graded_bets.append(grade)

            # Mark duplicates (skip on regrade — they're already marked)
            if not args.regrade:
                for dup_row in duplicates:
                    grades.append({"row": dup_row, "result": "duplicate", "units_won": 0})

            # Write grades back to source sheet
            if not args.dry_run and grades:
                write_grades_to_sheet(spreadsheet, tab_name, grades, [])
            elif args.dry_run:
                print(f"    [DRY RUN] Would update {len(grades)} rows")

        # Insert score bets into the ledger (store_score_edges historically
        # didn't dual-write, so grade time is where these rows first exist;
        # _append_to_ledger dedups on the ledger key, so re-runs and the
        # store-time dual-write added 2026-08 can't double-insert).
        score_bets_for_ledger = [
            b for b in all_graded_bets
            if b.get("bet_type") == "score_bet"
            and b.get("result") not in ("duplicate", "no_data", "unknown")
        ]
        if score_bets_for_ledger and not args.dry_run:
            print(f"\n  Adding {len(score_bets_for_ledger)} score bets to ledger...")
            from sheets_storage import _append_to_ledger
            ledger_rows = []
            for b in score_bets_for_ledger:
                def _f(v):
                    try:
                        return float(str(v).replace("+", ""))
                    except (ValueError, TypeError):
                        return np.nan
                ledger_rows.append({
                    "run_timestamp": datetime.now().isoformat(),
                    "event_name": event_name,
                    "year": year,
                    "event_id": str(event_id),
                    "bet_type": "score_bet",
                    "round": str(b.get("round", "")),
                    "bet_on": str(b.get("bet_on", "")).lower().strip(),
                    "opponent": str(b.get("opponent", "")).lower().strip(),
                    "bookmaker": str(b.get("bookmaker", "")).lower().strip(),
                    "book_odds": _f(b.get("book_odds", "")),
                    "fair_odds": _f(b.get("fair_odds", "")),
                    "edge": _f(b.get("edge", "")),
                    "book_category": b.get("book_category", ""),
                    "result": b.get("result", ""),
                    "units_wagered": b.get("units_wagered", FLAT_BET_SIZE),
                    "units_won": b.get("units_won", 0),
                    "graded_at": datetime.now().isoformat(),
                })
            _append_to_ledger(ledger_rows)

        # Update Parquet ledger with grades (bet types already written at store time)
        non_score_graded = [b for b in all_graded_bets if b.get("bet_type") != "score_bet"]
        if non_score_graded and not args.dry_run:
            print("\n  Updating Parquet ledger...")
            for bet in non_score_graded:
                bet["event_id"] = str(event_id)
            update_ledger_grades(non_score_graded)

        # Calculate and write summary
        if all_graded_bets and not args.dry_run:
            print("\n  Calculating performance metrics...")
            metrics = calculate_performance_metrics(all_graded_bets, event_name, event_id, year)
            if metrics:
                write_summary_row(spreadsheet, metrics)

                print("\n  RESULTS SUMMARY")
                print("  " + "-"*40)
                print(f"  Total bets:     {metrics['total_bets']}")
                print(f"  Graded:         {metrics['wins'] + metrics['losses'] + metrics['pushes']}")
                print(f"  No data:        {metrics['no_data']}")
                print(f"  Record:         {metrics['wins']}-{metrics['losses']}-{metrics['pushes']}")
                print(f"  Win rate:       {metrics['win_rate']}%")
                print(f"  Units wagered:  {metrics['units_wagered']:.2f}")
                print(f"  Units won:      {metrics['units_won']:+.2f}")
                print(f"  ROI:            {metrics['roi']:+.1f}%")

        elif args.dry_run and all_graded_bets:
            print("\n  Calculating performance metrics (dry run)...")
            metrics = calculate_performance_metrics(all_graded_bets, event_name, event_id, year)
            if metrics:
                print("\n  PREVIEW RESULTS SUMMARY")
                print("  " + "-"*40)
                print(f"  Total bets:     {metrics['total_bets']}")
                print(f"  Graded:         {metrics['wins'] + metrics['losses'] + metrics['pushes']}")
                print(f"  No data:        {metrics['no_data']}")
                print(f"  Record:         {metrics['wins']}-{metrics['losses']}-{metrics['pushes']}")
                print(f"  Win rate:       {metrics['win_rate']}%")
                print(f"  Units wagered:  {metrics['units_wagered']:.2f}")
                print(f"  Units won:      {metrics['units_won']:+.2f}")
                print(f"  ROI:            {metrics['roi']:+.1f}%")
                print("  [DRY RUN] Would write summary, results tabs, and send email")

        # Closing line value — annotate open/close odds + CLV columns from
        # DataGolf's odds archive (sheet tabs + ledger). Re-runnable via
        # `python clv.py --event-id N --write`.
        if not args.dry_run and not args.no_clv:
            print("\n  Computing closing line value (CLV)...")
            try:
                from clv import annotate_event_clv, clv_summary_stats
                clv_df = annotate_event_clv(spreadsheet, event_id, year)
                clv_stats = clv_summary_stats(clv_df)
            except Exception as e:
                print(f"  Warning: CLV annotation failed: {e}")

        grand_total_graded.extend(all_graded_bets)

    # Skip email for multi-event re-grades
    if len(events_to_grade) == 1 and grand_total_graded and not args.dry_run:
        if not args.no_email:
            print("\n  Sending email reports...")
            send_results_email(metrics, grand_total_graded, event_name, clv_stats=clv_stats)
            send_filtered_results_email(grand_total_graded, event_name, event_id, year, pred_threshold=0.75)
        else:
            print("\n  Email report skipped (--no-email)")
    elif len(events_to_grade) > 1:
        print(f"\n  Re-graded {len(grand_total_graded)} bets across {len(events_to_grade)} events (email skipped for bulk re-grade)")

    print("\n" + "="*60)
    print("  Done.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

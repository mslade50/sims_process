"""Abstracted data access for the dashboard.

All functions return DataFrames (empty if file missing).
Reads CSV/Parquet files from the project root.
"""

import os
import sys
import glob

import pandas as pd
import numpy as np

from dashboard.config import PROJECT_ROOT, PERMANENT_DATA, LEDGER_PATH, SG_DIAGNOSTIC_PATH

# Staging directory for Render deployment (committed data copies)
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard_data")

# Add project root to path so we can import sim_inputs
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_path(filename, subdir=None):
    """Resolve a data file path: check project root (local dev) first, then dashboard_data/ (Render).

    Args:
        filename: The file name (e.g., "model_predictions_r1.csv").
        subdir: Optional subdirectory under PROJECT_ROOT (e.g., tournament folder name).
    """
    # 1. Check subdirectory under root (e.g., tournament folder)
    if subdir:
        path = os.path.join(PROJECT_ROOT, subdir, filename)
        if os.path.exists(path):
            return path
    # 2. Check project root directly
    path = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(path):
        return path
    # 3. Fall back to dashboard_data/ (Render deployment)
    return os.path.join(DASHBOARD_DATA, filename)


def _read_csv_safe(path, **kwargs):
    """Read a CSV, returning empty DataFrame if missing or corrupt."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _read_parquet_safe(path):
    """Read a Parquet file, returning empty DataFrame if missing or corrupt."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# ── Tournament Config ────────────────────────────────────────────────────────

def get_tournament_config():
    """Import sim_inputs attributes into a dict. Returns empty dict on failure."""
    try:
        import sim_inputs
        attrs = [
            "tourney", "course_id", "course_name", "course_par", "event_ids",
            "num_sims", "player_var", "baseline_wind", "wind_override",
            "baseline_dew", "dew_calculation",
            "score_adj_r1", "score_adj_r2", "score_adj_r3", "score_adj_r4",
            "PAR", "STD_DEV", "SIMULATIONS", "CUT_LINE",
        ]
        config = {}
        for attr in attrs:
            if hasattr(sim_inputs, attr):
                config[attr] = getattr(sim_inputs, attr)
        return config
    except Exception:
        return {}


# ── Bet Ledger (Google Sheets) ────────────────────────────────────────────────

# Cache for Sheets data (avoid re-fetching on every filter change)
_SHEETS_CACHE = {"data": None, "timestamp": 0}
_SHEETS_CACHE_TTL = 300  # 5 minutes

# Tab names (same as sheets_storage.py)
_TAB_TOURNAMENT_MU = "Tournament Matchups"
_TAB_ROUND_MU = "Round Matchups"
_TAB_FINISH_POS = "Finish Positions"

_TOURNAMENT_MU_HEADERS = [
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

_ROUND_MU_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id", "round",
    "player_1", "player_2", "dg_id_p1", "dg_id_p2",
    "bookmaker", "ties_rule",
    "p1_odds", "p2_odds", "fair_p1", "fair_p2",
    "edge_p1", "edge_p2",
    "bet_on", "edge_on", "pred_on", "pred_against", "sample_on",
    "half_shot_p1", "half_shot_p2",
    "result", "p1_round_score", "p2_round_score", "units_won",
]

_FINISH_POS_HEADERS = [
    "run_timestamp", "event_name", "year", "event_id",
    "player_name", "dg_id",
    "market_type", "sportsbook",
    "decimal_odds", "american_odds", "my_fair",
    "sim_prob", "edge", "kelly_stake",
    "my_pred", "sample",
    "result", "actual_finish", "units_won",
]


def _read_sheets_tab(spreadsheet, tab_name, headers):
    """Read a Google Sheets tab into a DataFrame."""
    try:
        ws = spreadsheet.worksheet(tab_name)
        rows = ws.get_all_values()
        if len(rows) <= 1:
            return pd.DataFrame(columns=headers)
        # Use first row as header, rest as data
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        return pd.DataFrame(columns=headers)


def _normalize_tab(tab_df, bet_type):
    """Normalize an individual Sheets tab to a common schema."""
    n = pd.DataFrame()
    n["run_timestamp"] = tab_df.get("run_timestamp", "")
    n["event_name"] = tab_df.get("event_name", "")
    n["year"] = tab_df.get("year", "")
    n["event_id"] = tab_df.get("event_id", "")
    n["bet_type"] = bet_type
    n["round"] = tab_df.get("round", "0")

    # bet_on: matchup tabs have "bet_on", finish pos has "player_name"
    n["bet_on"] = tab_df.get("bet_on", tab_df.get("player_name", ""))

    # opponent: for matchups, opponent is the other player (not bet_on)
    if "player_1" in tab_df.columns and "player_2" in tab_df.columns:
        bet_on_col = tab_df.get("bet_on", pd.Series(dtype=str))
        n["opponent"] = np.where(
            bet_on_col.astype(str).str.strip().str.lower()
            == tab_df["player_2"].astype(str).str.strip().str.lower(),
            tab_df["player_1"],
            tab_df["player_2"],
        )
    elif "market_type" in tab_df.columns:
        n["opponent"] = tab_df["market_type"]
    else:
        n["opponent"] = ""

    # bookmaker: matchups have "bookmaker", finish pos has "sportsbook"
    n["bookmaker"] = tab_df.get("bookmaker", tab_df.get("sportsbook", ""))

    # odds: matchups have p1/p2_odds (pick the side matching bet_on), finish pos has american_odds
    if "p1_odds" in tab_df.columns and "player_1" in tab_df.columns:
        bet_on_col = tab_df.get("bet_on", pd.Series(dtype=str))
        is_p1 = (
            bet_on_col.astype(str).str.strip().str.lower()
            == tab_df["player_1"].astype(str).str.strip().str.lower()
        )
        n["book_odds"] = np.where(is_p1, tab_df["p1_odds"], tab_df.get("p2_odds", ""))
        n["fair_odds"] = np.where(is_p1, tab_df.get("fair_p1", ""), tab_df.get("fair_p2", ""))
    else:
        n["book_odds"] = tab_df.get("american_odds", "")
        n["fair_odds"] = tab_df.get("my_fair", "")

    # edge: matchups have edge_on, finish pos has edge
    n["edge"] = tab_df.get("edge_on", tab_df.get("edge", ""))

    n["pred_on"] = tab_df.get("pred_on", tab_df.get("my_pred", ""))
    n["sample_on"] = tab_df.get("sample_on", tab_df.get("sample", ""))
    n["kelly_stake"] = tab_df.get("kelly_stake", "")

    # graded fields
    n["result"] = tab_df.get("result", "")
    n["units_won"] = tab_df.get("units_won", "")

    return n


def _load_all_bets_from_sheets():
    """Load all bet data from Google Sheets, with caching.

    Reads from individual tabs (Tournament MU, Round MU, Finish Pos) which
    contain graded results from grade_bets.py, then normalizes to a common schema.
    """
    import time
    now = time.time()

    if _SHEETS_CACHE["data"] is not None and (now - _SHEETS_CACHE["timestamp"]) < _SHEETS_CACHE_TTL:
        return _SHEETS_CACHE["data"]

    try:
        from sheets_storage import get_spreadsheet
        spreadsheet = get_spreadsheet()
    except Exception as e:
        print(f"  [dashboard] Sheets auth failed: {e}, falling back to local Parquet")
        df = _read_parquet_safe(LEDGER_PATH)
        return df

    frames = []

    # Read individual tabs (these have graded results from grade_bets.py)
    for tab, headers, bet_type in [
        (_TAB_TOURNAMENT_MU, _TOURNAMENT_MU_HEADERS, "tournament_matchup"),
        (_TAB_ROUND_MU, _ROUND_MU_HEADERS, "round_matchup"),
        (_TAB_FINISH_POS, _FINISH_POS_HEADERS, "finish_position"),
    ]:
        tab_df = _read_sheets_tab(spreadsheet, tab, headers)
        if not tab_df.empty:
            frames.append(_normalize_tab(tab_df, bet_type))

    if not frames:
        _SHEETS_CACHE["data"] = pd.DataFrame()
        _SHEETS_CACHE["timestamp"] = now
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Dedup: keep FIRST occurrence by (event_id, bet_type, round, bet_on, opponent, bookmaker)
    # Sort by timestamp first to match grade_bets.py's "first write wins" semantics.
    if "run_timestamp" in df.columns:
        df = df.sort_values("run_timestamp", na_position="last")
    dedup_cols = ["event_id", "bet_type", "round", "bet_on", "opponent", "bookmaker"]
    existing = [c for c in dedup_cols if c in df.columns]
    if existing:
        for col in existing:
            df[col] = df[col].astype(str).str.lower().str.strip()
        df = df.drop_duplicates(subset=existing, keep="first")

    # Convert numeric columns
    for col in ["edge", "pred_on", "sample_on", "units_won", "book_odds", "fair_odds", "kelly_stake"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # Filter out tiny finish position bets (kelly_stake < $1)
    if "kelly_stake" in df.columns:
        small_finish = (df["bet_type"] == "finish_position") & (df["kelly_stake"] < 1.0)
        n_dropped = small_finish.sum()
        if n_dropped > 0:
            df = df[~small_finish]
            print(f"  [dashboard] Filtered out {n_dropped} finish position bets with kelly_stake < $1")

    _SHEETS_CACHE["data"] = df
    _SHEETS_CACHE["timestamp"] = now
    print(f"  [dashboard] Loaded {len(df)} bets from Google Sheets ({len(df[df['result'].astype(str).str.strip() != ''])} graded)")
    return df


def get_bet_ledger(**filters):
    """Query bet data from Google Sheets with optional filters.

    Keyword args: event, bet_type, book, min_edge, graded, year, books (list).
    """
    df = _load_all_bets_from_sheets()
    if df.empty:
        return df

    if filters.get("event"):
        df = df[df["event_name"].str.contains(str(filters["event"]), case=False, na=False)]
    if filters.get("bet_type"):
        df = df[df["bet_type"] == filters["bet_type"]]
    if filters.get("book"):
        df = df[df["bookmaker"].str.contains(str(filters["book"]), case=False, na=False)]
    if filters.get("books"):
        df = df[df["bookmaker"].str.lower().isin([b.lower() for b in filters["books"]])]
    if filters.get("min_edge") is not None:
        df = df[df["edge"] >= float(filters["min_edge"])]
    if filters.get("graded"):
        df = df[df["result"].astype(str).str.strip() != ""]
    if filters.get("year"):
        df = df[df["year"] == int(filters["year"])]

    return df


# ── Model Predictions ────────────────────────────────────────────────────────

def get_model_predictions(round_num):
    """Read model_predictions_r{N}.csv."""
    path = _resolve_path(f"model_predictions_r{round_num}.csv")
    return _read_csv_safe(path)


def get_player_scoring_params(round_num):
    """Extract player expected scores and std_dev from model predictions."""
    df = get_model_predictions(round_num)
    if df.empty:
        return pd.DataFrame()

    # Column names vary by round
    pred_col = f"my_pred{round_num}" if f"my_pred{round_num}" in df.columns else "my_pred"
    score_col = f"scores_r{round_num}" if f"scores_r{round_num}" in df.columns else None

    cols = ["player_name"]
    if pred_col in df.columns:
        cols.append(pred_col)
    if score_col and score_col in df.columns:
        cols.append(score_col)

    result = df[cols].copy() if all(c in df.columns for c in cols) else df[["player_name"]].copy()

    # Rename for consistency
    if pred_col in result.columns:
        result = result.rename(columns={pred_col: "my_pred"})
    if score_col and score_col in result.columns:
        result = result.rename(columns={score_col: "expected_score"})

    return result


# ── Live Model ───────────────────────────────────────────────────────────────

def get_live_model(round_num):
    """Read r{N}_live_model.csv."""
    path = _resolve_path(f"r{round_num}_live_model.csv")
    return _read_csv_safe(path)


# ── Matchups (Google Sheets) ──────────────────────────────────────────────────

_MATCHUP_CACHE = {
    "round": {"data": None, "timestamp": 0},
    "tournament": {"data": None, "timestamp": 0},
}


def _load_matchups_from_sheets(tab_name, headers):
    """Load matchup data from a Google Sheets tab, with caching."""
    import time
    cache_key = "tournament" if "Tournament" in tab_name else "round"
    now = time.time()

    if (_MATCHUP_CACHE[cache_key]["data"] is not None
            and (now - _MATCHUP_CACHE[cache_key]["timestamp"]) < _SHEETS_CACHE_TTL):
        return _MATCHUP_CACHE[cache_key]["data"]

    try:
        from sheets_storage import get_spreadsheet
        spreadsheet = get_spreadsheet()
    except Exception as e:
        print(f"  [dashboard] Sheets auth failed for matchups: {e}")
        return pd.DataFrame()

    df = _read_sheets_tab(spreadsheet, tab_name, headers)
    if df.empty:
        _MATCHUP_CACHE[cache_key]["data"] = df
        _MATCHUP_CACHE[cache_key]["timestamp"] = now
        return df

    # Convert numeric columns
    for col in ["edge_on", "edge_p1", "edge_p2", "pred_on", "pred_against",
                "sample_on", "p1_odds", "p2_odds", "fair_p1", "fair_p2",
                "half_shot_p1", "half_shot_p2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _MATCHUP_CACHE[cache_key]["data"] = df
    _MATCHUP_CACHE[cache_key]["timestamp"] = now
    return df


def get_matchups(round_num, tourney=None):
    """Get matchup data from Google Sheets, filtered to latest timestamp rows.

    For numeric rounds (1-4): reads "Round Matchups" tab.
    For "tourn": reads "Tournament Matchups" tab.
    """
    is_tournament = str(round_num).lower() in ("tourn", "tournament")

    if is_tournament:
        df = _load_matchups_from_sheets(_TAB_TOURNAMENT_MU, _TOURNAMENT_MU_HEADERS)
        if df.empty:
            return df
        # Filter to current event
        if tourney is None:
            config = get_tournament_config()
            tourney = config.get("tourney", "")
        if tourney:
            df = df[df["event_name"].str.lower().str.strip() == tourney.lower()]
    else:
        df = _load_matchups_from_sheets(_TAB_ROUND_MU, _ROUND_MU_HEADERS)
        if df.empty:
            return df
        # Filter to requested round
        df = df[df["round"].astype(str).str.strip() == str(round_num)]

    if df.empty:
        return df

    # Keep only the latest timestamp rows
    latest_ts = df["run_timestamp"].max()
    df = df[df["run_timestamp"] == latest_ts].copy()

    # Normalize column names for display (Bookmaker capitalization)
    if "bookmaker" in df.columns and "Bookmaker" not in df.columns:
        df = df.rename(columns={"bookmaker": "Bookmaker"})

    if "ties_rule" in df.columns:
        df = df.rename(columns={"ties_rule": "Ties"})

    # Build bet_against, bet_on_odds, bet_against_odds, fair from player perspective
    if "player_1" in df.columns and "player_2" in df.columns and "bet_on" in df.columns:
        is_p2 = (
            df["bet_on"].astype(str).str.strip().str.lower()
            == df["player_2"].astype(str).str.strip().str.lower()
        )
        df["bet_against"] = np.where(is_p2, df["player_1"], df["player_2"])

        if "p1_odds" in df.columns and "p2_odds" in df.columns:
            df["bet_on_odds"] = np.where(is_p2, df["p2_odds"], df["p1_odds"])
            df["bet_against_odds"] = np.where(is_p2, df["p1_odds"], df["p2_odds"])

        if "fair_p1" in df.columns and "fair_p2" in df.columns:
            df["fair"] = np.where(is_p2, df["fair_p2"], df["fair_p1"])

    # Drop raw columns no longer needed for display
    drop_cols = [
        "player_1", "player_2", "p1_odds", "p2_odds",
        "fair_p1", "fair_p2", "half_shot_p1", "half_shot_p2",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ── Outrights ────────────────────────────────────────────────────────────────

def get_simulated_probs():
    """Read simulated_probs_live.csv."""
    path = _resolve_path("simulated_probs_live.csv")
    return _read_csv_safe(path)


def get_outright_edges(tourney=None):
    """Read outright_win_edges.csv from tournament folder."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    path = _resolve_path("outright_win_edges.csv", subdir=tourney)
    return _read_csv_safe(path)


def get_devig_fades(tourney=None):
    """Read betonline_devig_fades.csv from tournament folder."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    path = _resolve_path("betonline_devig_fades.csv", subdir=tourney)
    return _read_csv_safe(path)


def get_finish_equity(tourney=None):
    """Read finish_equity_live_{tourney}.csv."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    # Local: finish_equity_live_{tourney}.csv; Render: finish_equity_live.csv
    path = os.path.join(PROJECT_ROOT, f"finish_equity_live_{tourney}.csv")
    if os.path.exists(path):
        return _read_csv_safe(path)
    return _read_csv_safe(_resolve_path("finish_equity_live.csv"))


# ── Pre-Tournament Outrights ────────────────────────────────────────────────

def get_simulated_probs_pre():
    """Read simulated_probs.csv — prefer v2 output if available."""
    config = get_tournament_config()
    tourney = config.get("tourney", "")
    if tourney:
        v2_path = os.path.join(PROJECT_ROOT, tourney, "v2", "simulated_probs.csv")
        if os.path.exists(v2_path):
            return _read_csv_safe(v2_path)
    path = _resolve_path("simulated_probs.csv")
    return _read_csv_safe(path)


def get_rank_probs_pre(tourney=None):
    """Read rank_probs_updated_{tourney}.parquet — prefer v2 output if available."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    # Prefer v2 output
    v2_path = os.path.join(PROJECT_ROOT, tourney, "v2", f"rank_probs_updated_{tourney}.parquet")
    if os.path.exists(v2_path):
        return _read_parquet_safe(v2_path)
    # Fallback: root then dashboard_data
    path = os.path.join(PROJECT_ROOT, f"rank_probs_updated_{tourney}.parquet")
    if os.path.exists(path):
        return _read_parquet_safe(path)
    return _read_parquet_safe(os.path.join(DASHBOARD_DATA, "rank_probs_pre.parquet"))


def get_rank_probs_live(tourney=None):
    """Read rank_probs_live_{tourney}.parquet (live finish distributions)."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    # Local: rank_probs_live_{tourney}.parquet; Render: rank_probs_live.parquet
    path = os.path.join(PROJECT_ROOT, f"rank_probs_live_{tourney}.parquet")
    if os.path.exists(path):
        return _read_parquet_safe(path)
    return _read_parquet_safe(os.path.join(DASHBOARD_DATA, "rank_probs_live.parquet"))


def get_finish_equity_pre(tourney=None):
    """Read finish_equity_{tourney}.csv — prefer v2 output if available."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        return pd.DataFrame()
    # Prefer v2 output
    v2_path = os.path.join(PROJECT_ROOT, tourney, "v2", f"finish_equity_{tourney}.csv")
    if os.path.exists(v2_path):
        return _read_csv_safe(v2_path)
    # Fallback: root then dashboard_data
    path = os.path.join(PROJECT_ROOT, f"finish_equity_{tourney}.csv")
    if os.path.exists(path):
        return _read_csv_safe(path)
    return _read_csv_safe(_resolve_path("finish_equity_pre.csv"))


def get_fair_card(round_num, tourney=None):
    """Read fair_card_r{N}.csv from tournament folder or dashboard_data/."""
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")
    if not tourney:
        # No tourney config — try dashboard_data/ directly
        return _read_csv_safe(_resolve_path(f"fair_card_r{round_num}.csv"))
    path = _resolve_path(f"fair_card_r{round_num}.csv", subdir=tourney)
    return _read_csv_safe(path)


# ── V2 SG Category Distributions ─────────────────────────────────────────

def get_v2_dists():
    """Read this_week_dists_v2.csv (per-player SG category distributions)."""
    path = _resolve_path("this_week_dists_v2.csv")
    return _read_csv_safe(path)


def get_sg_dist_player():
    """Read sg_dist_player.csv (full historical per-player SG distributions with histograms)."""
    path = _resolve_path("sg_dist_player.csv")
    return _read_csv_safe(path)



# ── SG Diagnostics ───────────────────────────────────────────────────────────

def get_sg_diagnostics():
    """Read sg_diagnostic.parquet from permanent_data/ or dashboard_data/."""
    if os.path.exists(SG_DIAGNOSTIC_PATH):
        return _read_parquet_safe(SG_DIAGNOSTIC_PATH)
    return _read_parquet_safe(os.path.join(DASHBOARD_DATA, "sg_diagnostic.parquet"))


# ── Available Rounds ─────────────────────────────────────────────────────────

def get_available_rounds():
    """Scan which model_predictions_r{N}.csv files exist. Returns list of ints."""
    rounds = []
    for n in range(1, 5):
        path = _resolve_path(f"model_predictions_r{n}.csv")
        if os.path.exists(path):
            rounds.append(n)
    return rounds


def get_available_live_model_rounds():
    """Scan which r{N}_live_model.csv files exist. Returns list of ints."""
    rounds = []
    for n in range(1, 5):
        path = _resolve_path(f"r{n}_live_model.csv")
        if os.path.exists(path):
            rounds.append(n)
    return rounds


def get_available_matchup_rounds(tourney=None):
    """Check which rounds have matchup data in Google Sheets for the current event.

    Returns list of round identifiers — ints for round-specific (1-4),
    plus "tourn" if tournament-level matchups exist.
    """
    if tourney is None:
        config = get_tournament_config()
        tourney = config.get("tourney", "")

    rounds = []

    # Check Round Matchups tab
    round_df = _load_matchups_from_sheets(_TAB_ROUND_MU, _ROUND_MU_HEADERS)
    if not round_df.empty and tourney:
        event_df = round_df[round_df["event_name"].str.lower().str.strip() == tourney.lower()]
        if not event_df.empty:
            round_vals = event_df["round"].astype(str).str.strip().unique()
            for rv in sorted(round_vals):
                try:
                    rounds.append(int(rv))
                except ValueError:
                    pass

    # Check Tournament Matchups tab
    tourn_df = _load_matchups_from_sheets(_TAB_TOURNAMENT_MU, _TOURNAMENT_MU_HEADERS)
    if not tourn_df.empty and tourney:
        event_df = tourn_df[tourn_df["event_name"].str.lower().str.strip() == tourney.lower()]
        if not event_df.empty:
            rounds.append("tourn")

    return rounds


def get_luck_attribution():
    """Read luck/attribution data from bet_ledger.parquet (local) or dashboard_data/ (Render).

    Returns graded bets with margin and SG category columns.
    """
    # Prefer local ledger (has latest data)
    df = _read_parquet_safe(LEDGER_PATH)
    if df.empty:
        df = _read_parquet_safe(os.path.join(DASHBOARD_DATA, "luck_attribution.parquet"))
    if df.empty:
        return pd.DataFrame()

    # Filter to graded bets only
    if "result" in df.columns:
        df = df[df["result"].astype(str).str.strip().isin(["win", "loss", "push", "win_dh"])]

    # Coerce numeric columns
    numeric_cols = [
        "margin", "book_odds", "edge", "units_wagered", "units_won",
        "bet_on_sg_ott", "bet_on_sg_app", "bet_on_sg_arg", "bet_on_sg_putt",
        "opp_sg_ott", "opp_sg_app", "opp_sg_arg", "opp_sg_putt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dedup: same matchup across multiple books counts once for luck analysis
    dedup_cols = ["event_id", "bet_type", "round", "bet_on", "opponent"]
    existing = [c for c in dedup_cols if c in df.columns]
    if existing:
        df = df.drop_duplicates(subset=existing, keep="first")

    return df


HISTORICAL_DISTS = os.path.join(PERMANENT_DATA, "historical_dists")


def get_historical_events():
    """Scan permanent_data/historical_dists/ for archived events.

    Returns list of (event_id, tourney_name) tuples sorted by event_id descending.
    """
    if not os.path.isdir(HISTORICAL_DISTS):
        return []
    events = []
    for name in os.listdir(HISTORICAL_DISTS):
        path = os.path.join(HISTORICAL_DISTS, name)
        if not os.path.isdir(path):
            continue
        # Format: {event_id}_{tourney}
        parts = name.split("_", 1)
        if len(parts) == 2:
            try:
                event_id = int(parts[0])
                events.append((event_id, parts[1]))
            except ValueError:
                continue
    events.sort(key=lambda x: x[0], reverse=True)
    return events


def get_historical_rank_probs(event_id, tourney, mode="pre"):
    """Read archived rank_probs parquet for a historical event.

    Args:
        event_id: Numeric event ID.
        tourney: Tournament name slug.
        mode: "pre" or "live".

    Returns DataFrame (empty if missing).
    """
    path = os.path.join(HISTORICAL_DISTS, f"{event_id}_{tourney}", f"rank_probs_{mode}.parquet")
    return _read_parquet_safe(path)


def get_file_mtime(filename):
    """Get modification time of a file. Returns None if missing."""
    path = _resolve_path(filename)
    if os.path.exists(path):
        return os.path.getmtime(path)
    return None


# ── Weather Data ─────────────────────────────────────────────────────────────

def get_weather_impact_players():
    """Read weather_impact_{tourney}.csv (per-player wind/dew adjustments from sim).

    Local: weather_impact_{tourney}.csv; Render: weather_impact.csv in dashboard_data/.
    """
    config = get_tournament_config()
    tourney = config.get("tourney", "")
    if tourney:
        path = os.path.join(PROJECT_ROOT, f"weather_impact_{tourney}.csv")
        if os.path.exists(path):
            return _read_csv_safe(path)
    return _read_csv_safe(_resolve_path("weather_impact.csv"))


_WEATHER_CACHE = {"data": None, "timestamp": 0}


def get_weather_forecast():
    """Read wind/dew hourly arrays from Google Sheet round_config tab.

    Returns dict with wind_r1..wind_r4, dew_r1..dew_r4 arrays,
    plus wind_override, dew_calculation scalars.
    Falls back to sim_inputs for wind_speed_base.
    """
    import time
    now = time.time()

    if _WEATHER_CACHE["data"] is not None and (now - _WEATHER_CACHE["timestamp"]) < _SHEETS_CACHE_TTL:
        return _WEATHER_CACHE["data"]

    result = {}
    try:
        from sheet_config import load_config
        config = load_config()
        for r in range(1, 5):
            result[f"wind_r{r}"] = config.get(f"wind_r{r}", [])
            result[f"dew_r{r}"] = config.get(f"dew_r{r}", [])
        result["wind_override"] = config.get("wind_override", 0.0)
        result["dew_calculation"] = config.get("dew_calculation", 0.0)
    except Exception as e:
        print(f"  [dashboard] Sheet config failed: {e}")

    # wind_speed_base from sim_inputs
    try:
        import sim_inputs
        result["wind_speed_base"] = getattr(sim_inputs, "wind_speed_base", None)
        result["baseline_wind"] = getattr(sim_inputs, "baseline_wind", None)
        result["baseline_dew"] = getattr(sim_inputs, "baseline_dew", None)
    except Exception:
        pass

    _WEATHER_CACHE["data"] = result
    _WEATHER_CACHE["timestamp"] = now
    return result


def get_weather_matchup_edges():
    """Join weather_impact_players with BetOnline tournament matchups.

    Computes weather differential for each matched pair.
    Returns df sorted by absolute differential descending.
    """
    weather = get_weather_impact_players()
    if weather.empty:
        return pd.DataFrame()

    # Get tournament matchups (raw, before get_matchups drops player columns)
    tourn_df = _load_matchups_from_sheets(_TAB_TOURNAMENT_MU, _TOURNAMENT_MU_HEADERS)
    if tourn_df.empty:
        return pd.DataFrame()

    # Filter to current event, latest timestamp, BetOnline only
    config = get_tournament_config()
    tourney = config.get("tourney", "")
    if tourney:
        tourn_df = tourn_df[tourn_df["event_name"].str.lower().str.strip() == tourney.lower()]
    if tourn_df.empty:
        return pd.DataFrame()

    latest_ts = tourn_df["run_timestamp"].max()
    tourn_df = tourn_df[tourn_df["run_timestamp"] == latest_ts].copy()

    # Filter to BetOnline
    if "bookmaker" in tourn_df.columns:
        tourn_df = tourn_df[tourn_df["bookmaker"].str.lower().str.contains("betonline", na=False)]
    if tourn_df.empty:
        return pd.DataFrame()

    # Compute total weather advantage per player
    adv_cols = []
    for col in ["wind_adv_r1_2", "dew_adv_r1_2"]:
        if col in weather.columns:
            adv_cols.append(col)
    if not adv_cols:
        return pd.DataFrame()

    weather["total_weather_adv"] = weather[adv_cols].sum(axis=1)
    weather_lookup = weather.set_index("player_name")["total_weather_adv"].to_dict()

    rows = []
    for _, mu in tourn_df.iterrows():
        p1 = str(mu.get("player_1", "")).strip().lower()
        p2 = str(mu.get("player_2", "")).strip().lower()
        w1 = weather_lookup.get(p1)
        w2 = weather_lookup.get(p2)
        if w1 is None or w2 is None:
            continue
        diff = w1 - w2
        rows.append({
            "player_1": p1,
            "player_2": p2,
            "p1_weather_adv": round(w1, 3),
            "p2_weather_adv": round(w2, 3),
            "differential": round(diff, 3),
            "favored": p1 if diff > 0 else p2,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["abs_diff"] = result["differential"].abs()
    result = result.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"])
    return result


# ── Market Regression Diagnostics ────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "dg_historical.db")


def get_mkt_regress_diagnostics():
    """Combine historical mkt_regress files with actual tournament SG from db.

    Returns DataFrame with one row per player-event:
        player_name, event_name, event_id, pred, my_pred_regressed,
        comp_c_adj, tot_rgrs, mkt_adj, mu_adj, actual_sg,
        error_raw, error_regressed, regress_helped
    Returns empty DataFrame if no data available.
    """
    import sqlite3

    # Find all mkt_regress_*.csv files
    pattern = os.path.join(PROJECT_ROOT, "mkt_regress_*.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize columns across old/new format
            keep = {}
            keep["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
            keep["event_id"] = df["event_id"] if "event_id" in df.columns else np.nan
            keep["event_name"] = df["event_name"] if "event_name" in df.columns else os.path.basename(f)
            keep["pred"] = pd.to_numeric(df["pred"], errors="coerce")
            # New format uses my_pred_final, old uses my_pred_3
            if "my_pred_final" in df.columns:
                keep["my_pred_regressed"] = pd.to_numeric(df["my_pred_final"], errors="coerce")
            elif "my_pred_3" in df.columns:
                keep["my_pred_regressed"] = pd.to_numeric(df["my_pred_3"], errors="coerce")
            keep["tot_rgrs"] = pd.to_numeric(df.get("tot_rgrs", 0), errors="coerce")
            keep["mkt_adj"] = pd.to_numeric(df.get("mkt_adj", 0), errors="coerce")
            keep["mu_adj"] = pd.to_numeric(df.get("mu_adj", 0), errors="coerce")
            keep["comp_c_adj"] = pd.to_numeric(df.get("comp_c_adj", df.get("c_adj", 0)), errors="coerce")
            frames.append(pd.DataFrame(keep))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["event_id", "pred", "my_pred_regressed"])
    combined["event_id"] = combined["event_id"].astype(int)

    # Fetch actuals from db
    if not os.path.exists(_DB_PATH):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(_DB_PATH)
        event_ids = combined["event_id"].unique().tolist()
        placeholders = ",".join("?" * len(event_ids))
        actuals = pd.read_sql_query(
            f"""
            SELECT player_name, event_id,
                   AVG(sg_total) as actual_sg,
                   COUNT(*) as rounds_played
            FROM player_rounds
            WHERE event_id IN ({placeholders}) AND year = 2026
            GROUP BY player_name, event_id
            """,
            conn,
            params=event_ids,
        )
        conn.close()
    except Exception:
        return pd.DataFrame()

    if actuals.empty:
        return pd.DataFrame()

    actuals["player_name"] = actuals["player_name"].astype(str).str.lower().str.strip()

    merged = combined.merge(actuals, on=["player_name", "event_id"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    # Compute errors (positive = predicted too high / too optimistic)
    merged["error_raw"] = merged["pred"] - merged["actual_sg"]
    merged["error_regressed"] = merged["my_pred_regressed"] - merged["actual_sg"]
    merged["regress_helped"] = merged["error_regressed"].abs() < merged["error_raw"].abs()

    return merged

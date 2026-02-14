"""
sg_diagnostic.py - Post-Event SG Prediction Diagnostic

Compares Monte Carlo per-category SG predictions against actual round-level
SG from DataGolf. Classifies players into archetypes (splitting OTT by
distance vs accuracy), stores results in a persistent Parquet, and emails
a diagnostic report.

Usage:
    python sg_diagnostic.py                     # Current tourney (from sim_inputs)
    python sg_diagnostic.py --event-id 123      # Override event ID
    python sg_diagnostic.py --no-email          # Skip email, just store + print
    python sg_diagnostic.py --report            # Accumulated cross-event report
"""

import os
import sys
import argparse
import tempfile
import smtplib
import sqlite3
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from sim_inputs import tourney, event_ids, name_replacements
from api_utils import fetch_historical_rounds, clean_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "dg_historical.db")
DIAGNOSTIC_PATH = os.path.join(
    os.path.dirname(__file__), "permanent_data", "sg_diagnostic.parquet"
)
DIAGNOSTIC_DEDUP_COLS = ["event_id", "player_name", "round", "category"]
SG_CATS = ["ott", "app", "arg", "putt", "total"]

EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# ---------------------------------------------------------------------------
# 2b. load_predictions
# ---------------------------------------------------------------------------
def load_predictions(tourney_name):
    """
    Read avg_expected_cat_sg_{tourney}.csv and reshape from wide to long.

    Input cols:  player_name, r1_ott_mean, r1_app_mean, ..., r4_total_mean
    Output cols: player_name, round (1-4), category, predicted_sg
    """
    path = os.path.join(
        os.path.dirname(__file__), f"avg_expected_cat_sg_{tourney_name}.csv"
    )
    if not os.path.exists(path):
        print(f"  Prediction file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
    df["player_name"] = df["player_name"].replace(name_replacements)

    rows = []
    for _, row in df.iterrows():
        player = row["player_name"]
        for rnd in range(1, 5):
            for cat in SG_CATS:
                col = f"r{rnd}_{cat}_mean"
                if col in row.index and pd.notna(row[col]):
                    rows.append(
                        {
                            "player_name": player,
                            "round": rnd,
                            "category": cat,
                            "predicted_sg": float(row[col]),
                        }
                    )

    result = pd.DataFrame(rows)
    print(
        f"  Loaded predictions: {result['player_name'].nunique()} players, "
        f"{len(result)} rows"
    )
    return result


# ---------------------------------------------------------------------------
# 2c. fetch_actuals
# ---------------------------------------------------------------------------
def _fetch_actuals_from_db(event_id, year=None):
    """
    Primary: pull ADJUSTED SG actuals from dg_historical.db.
    Uses sg_*_adj columns which are field-strength adjusted -- comparable
    to our predictions (which are built from adjusted historical SG dists).

    Returns long-format DataFrame or empty DataFrame if event not in DB.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), False

    if year is None:
        year = datetime.now().year

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT player_name, dg_id, round_num,
                   sg_ott_adj, sg_app_adj, sg_arg_adj, sg_putt_adj, sg_total_adj
            FROM player_rounds
            WHERE event_id = ? AND year = ?
              AND sg_ott_adj IS NOT NULL
            ORDER BY player_name, round_num
            """,
            conn,
            params=(int(event_id), int(year)),
        )
        conn.close()
    except Exception as e:
        print(f"  DB query error: {e}")
        return pd.DataFrame(), False

    if df.empty:
        return pd.DataFrame(), False

    # Normalize names
    df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
    df["player_name"] = df["player_name"].replace(name_replacements)

    # Melt to long format
    rows = []
    for _, row in df.iterrows():
        for adj_col, cat in [
            ("sg_ott_adj", "ott"),
            ("sg_app_adj", "app"),
            ("sg_arg_adj", "arg"),
            ("sg_putt_adj", "putt"),
            ("sg_total_adj", "total"),
        ]:
            val = row.get(adj_col)
            if pd.isna(val):
                continue
            rows.append(
                {
                    "player_name": row["player_name"],
                    "dg_id": row.get("dg_id"),
                    "round": int(row["round_num"]),
                    "category": cat,
                    "actual_sg": float(val),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        print(
            f"  DB actuals (adjusted): {result['player_name'].nunique()} players, "
            f"{result['round'].nunique()} rounds, {len(result)} rows"
        )
    return result, True


def _fetch_actuals_live(max_round=4):
    """
    Fallback: fetch actuals from live-tournament-stats API (works mid-event).
    WARNING: Returns RAW (unadjusted) SG -- not directly comparable to
    predictions which are based on adjusted distributions. Absolute bias
    values are NOT meaningful; only relative player/archetype differences
    may be informative.

    Returns long-format DataFrame matching fetch_actuals() output.
    """
    from api_utils import fetch_live_stats
    api_key = os.getenv("DATAGOLF_API_KEY")

    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]
    rows = []

    for rnd in range(1, max_round + 1):
        df = fetch_live_stats(rnd, api_key)
        if df is None or df.empty:
            print(f"    R{rnd}: no data")
            continue

        # Check if SG data is present (not all NaN)
        has_sg = any(col in df.columns and df[col].notna().any() for col in sg_cols)
        if not has_sg:
            print(f"    R{rnd}: no SG data")
            continue

        count = 0
        for _, row in df.iterrows():
            for sg_col in sg_cols:
                val = row.get(sg_col)
                if pd.isna(val):
                    continue
                cat = sg_col.replace("sg_", "")
                rows.append(
                    {
                        "player_name": row["player_name"],
                        "dg_id": row.get("dg_id"),
                        "round": rnd,
                        "category": cat,
                        "actual_sg": float(val),
                    }
                )
                count += 1
        print(f"    R{rnd}: {count // len(sg_cols)} players with SG data")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    print(
        f"  Live actuals (RAW -- not field-adjusted): "
        f"{result['player_name'].nunique()} players, "
        f"{result['round'].nunique()} rounds, {len(result)} rows"
    )
    return result


def fetch_actuals(event_id, year=None):
    """
    Fetch actual SG data for comparison against predictions.

    Priority order:
    1. dg_historical.db (adjusted SG) -- apples-to-apples with predictions
    2. DataGolf historical-raw-data/rounds API (raw SG)
    3. Live-tournament-stats API (raw SG, works mid-event)

    Returns: (DataFrame, is_adjusted: bool)
    """
    # 1. Try database first (adjusted SG -- best for comparison)
    db_result, used_db = _fetch_actuals_from_db(event_id, year=year)
    if not db_result.empty:
        return db_result, True

    # 2. Try historical API (completed events, raw SG)
    df = fetch_historical_rounds(event_id, year=year)
    if not df.empty:
        sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]
        rows = []
        for _, row in df.iterrows():
            for sg_col in sg_cols:
                val = row.get(sg_col)
                if pd.isna(val):
                    continue
                cat = sg_col.replace("sg_", "")
                rows.append(
                    {
                        "player_name": row["player_name"],
                        "dg_id": row.get("dg_id"),
                        "round": int(row["round_num"]),
                        "category": cat,
                        "actual_sg": float(val),
                    }
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            print(
                f"  API actuals (RAW -- not field-adjusted): "
                f"{result['player_name'].nunique()} players, {len(result)} rows"
            )
            print(
                "  WARNING: Raw SG actuals. Absolute bias reflects field-strength "
                "adjustment gap, not model error."
            )
            return result, False

    # 3. Fallback to live stats (mid-event)
    print("  Historical sources unavailable -- trying live stats API...")
    live_result = _fetch_actuals_live(max_round=4)
    if not live_result.empty:
        return live_result, False

    print("  No SG actual data available")
    return pd.DataFrame(), False


# ---------------------------------------------------------------------------
# 2d. compute_rolling_archetypes
# ---------------------------------------------------------------------------
def compute_rolling_archetypes(event_id, field_players, actuals_df=None):
    """
    Query dg_historical.db for rolling stats BEFORE the current event.
    Classify each player into an archetype based on blended rolling averages.

    Returns DataFrame with: player_name, dg_id, archetype, sg_*_rolling,
                            driving_dist_rolling, driving_acc_rolling
    """
    if not os.path.exists(DB_PATH):
        print(f"  Database not found at {DB_PATH} -- skipping archetypes")
        return _unknown_archetypes(field_players)

    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"  Database connection error: {e} -- skipping archetypes")
        return _unknown_archetypes(field_players)

    # Get the earliest round_date for this event IN THE CURRENT YEAR
    # (event_ids recur annually, so we must filter by year)
    try:
        cutoff_query = """
            SELECT MIN(round_date) FROM player_rounds
            WHERE event_id = ? AND round_date IS NOT NULL
            ORDER BY round_date DESC
        """
        # Try current year first, then most recent occurrence
        cutoff_date = None
        cutoff_row = conn.execute(
            "SELECT MIN(round_date) FROM player_rounds "
            "WHERE event_id = ? AND year = ? AND round_date IS NOT NULL",
            (int(event_id), datetime.now().year),
        ).fetchone()
        if cutoff_row and cutoff_row[0]:
            cutoff_date = cutoff_row[0]
        else:
            # Fallback: most recent year's occurrence
            cutoff_row = conn.execute(
                "SELECT MAX(round_date) FROM player_rounds "
                "WHERE event_id = ? AND round_date IS NOT NULL",
                (int(event_id),),
            ).fetchone()
            if cutoff_row and cutoff_row[0]:
                cutoff_date = cutoff_row[0]

        if not cutoff_date:
            cutoff_date = datetime.now().strftime("%Y-%m-%d")
            print(f"  No event dates in DB for event {event_id}, using today as cutoff")
        else:
            print(f"  Using cutoff date: {cutoff_date}")
    except Exception as e:
        print(f"  Error finding event date: {e}")
        cutoff_date = datetime.now().strftime("%Y-%m-%d")

    query = """
        SELECT player_name, dg_id, round_date,
               sg_ott_adj, sg_app_adj, sg_arg_adj, sg_putt_adj,
               driving_dist, driving_acc
        FROM player_rounds
        WHERE sg_ott_adj IS NOT NULL
          AND round_date < ?
        ORDER BY player_name, round_date DESC
    """

    try:
        db_df = pd.read_sql_query(query, conn, params=(cutoff_date,))
    except Exception as e:
        print(f"  Database query error: {e}")
        conn.close()
        return _unknown_archetypes(field_players)
    finally:
        conn.close()

    if db_df.empty:
        print("  No historical data found in database")
        return _unknown_archetypes(field_players)

    # Normalize DB names to lowercase + apply name_replacements
    db_df["player_name"] = (
        db_df["player_name"].astype(str).str.lower().str.strip()
    )
    db_df["player_name"] = db_df["player_name"].replace(name_replacements)

    # Also normalize dg_id for matching
    if "dg_id" in db_df.columns:
        db_df["dg_id"] = pd.to_numeric(db_df["dg_id"], errors="coerce")

    # Build dg_id lookup from actuals for fallback matching
    dg_id_map = {}
    if actuals_df is not None and "dg_id" in actuals_df.columns:
        for _, row in (
            actuals_df[["player_name", "dg_id"]]
            .drop_duplicates("player_name")
            .iterrows()
        ):
            if pd.notna(row["dg_id"]):
                dg_id_map[row["player_name"]] = int(row["dg_id"])

    records = []
    matched = 0
    for player in field_players:
        # Try name match first
        player_data = db_df[db_df["player_name"] == player]

        # Fallback: try dg_id match
        if player_data.empty and player in dg_id_map:
            pid = dg_id_map[player]
            player_data = db_df[db_df["dg_id"] == pid]

        if len(player_data) < 20:
            records.append(
                {
                    "player_name": player,
                    "dg_id": dg_id_map.get(player),
                    "archetype": "Unknown",
                    "sg_ott_rolling": np.nan,
                    "sg_app_rolling": np.nan,
                    "sg_arg_rolling": np.nan,
                    "sg_putt_rolling": np.nan,
                    "driving_dist_rolling": np.nan,
                    "driving_acc_rolling": np.nan,
                }
            )
            continue

        matched += 1
        # Take most recent 50 rounds (already sorted DESC)
        recent = player_data.head(50)

        rec = {"player_name": player, "dg_id": dg_id_map.get(player)}

        for col, out_col in [
            ("sg_ott_adj", "sg_ott_rolling"),
            ("sg_app_adj", "sg_app_rolling"),
            ("sg_arg_adj", "sg_arg_rolling"),
            ("sg_putt_adj", "sg_putt_rolling"),
            ("driving_dist", "driving_dist_rolling"),
            ("driving_acc", "driving_acc_rolling"),
        ]:
            vals = recent[col].dropna()
            if len(vals) < 20:
                rec[out_col] = np.nan
                continue

            sma_50 = vals.mean()

            # EMA with span=12
            alpha = 2.0 / (12 + 1)
            ema = vals.iloc[0]  # most recent (DESC order)
            for v in vals.iloc[1:12]:  # use up to 12
                ema = alpha * v + (1 - alpha) * ema
            # Wait -- EMA from oldest to newest is the standard.
            # Since data is DESC (newest first), reverse for EMA calc.
            vals_asc = vals.iloc[:12].iloc[::-1]
            ema = vals_asc.iloc[0]
            for v in vals_asc.iloc[1:]:
                ema = alpha * v + (1 - alpha) * ema

            rec[out_col] = 0.5 * sma_50 + 0.5 * ema

        records.append(rec)

    result = pd.DataFrame(records)
    print(f"  Archetypes: {matched}/{len(field_players)} players matched in DB")

    # Classify archetypes using percentile ranks within this field
    result = _classify_archetypes(result)
    return result


def _unknown_archetypes(field_players):
    """Return DataFrame with all players marked as Unknown archetype."""
    return pd.DataFrame(
        {
            "player_name": list(field_players),
            "dg_id": [None] * len(field_players),
            "archetype": ["Unknown"] * len(field_players),
            "sg_ott_rolling": [np.nan] * len(field_players),
            "sg_app_rolling": [np.nan] * len(field_players),
            "sg_arg_rolling": [np.nan] * len(field_players),
            "sg_putt_rolling": [np.nan] * len(field_players),
            "driving_dist_rolling": [np.nan] * len(field_players),
            "driving_acc_rolling": [np.nan] * len(field_players),
        }
    )


def _classify_archetypes(df):
    """
    Classify players into archetypes based on percentile ranks within field.
    First match wins. Uses mean-based archetypes (not variance-based).
    """
    # Compute percentile ranks (0-100) for players with data
    for col in [
        "sg_ott_rolling",
        "sg_app_rolling",
        "sg_arg_rolling",
        "sg_putt_rolling",
        "driving_dist_rolling",
        "driving_acc_rolling",
    ]:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    # Composite columns for ball-striking and short game
    df["ball_striking"] = df["sg_ott_rolling"] + df["sg_app_rolling"]
    df["short_game"] = df["sg_arg_rolling"] + df["sg_putt_rolling"]
    df["ball_striking_pct"] = df["ball_striking"].rank(pct=True) * 100
    df["short_game_pct"] = df["short_game"].rank(pct=True) * 100

    def classify(row):
        if row.get("archetype") == "Unknown":
            return "Unknown"

        dd = row.get("driving_dist_rolling_pct", np.nan)
        da = row.get("driving_acc_rolling_pct", np.nan)
        bs = row.get("ball_striking_pct", np.nan)
        sg = row.get("short_game_pct", np.nan)
        putt = row.get("sg_putt_rolling_pct", np.nan)
        ott = row.get("sg_ott_rolling_pct", np.nan)
        app = row.get("sg_app_rolling_pct", np.nan)
        arg = row.get("sg_arg_rolling_pct", np.nan)

        # Check for NaN -- can't classify without data
        if any(pd.isna(v) for v in [dd, da, bs, sg, putt]):
            return "Unknown"

        # First match wins
        if dd >= 70 and da < 50:
            return "Long Bomber"
        if da >= 70 and dd < 50:
            return "Accurate Short"
        if dd >= 60 and da >= 60:
            return "Long Accurate"
        if bs >= 70 and sg < 50:
            return "Ball Striker"
        if sg >= 70 and bs < 50:
            return "Short Game Specialist"
        if putt >= 80:
            return "Elite Putter"
        if all(30 <= v <= 70 for v in [ott, app, arg, putt] if not pd.isna(v)):
            return "All-Around"
        return "Neutral"

    df["archetype"] = df.apply(classify, axis=1)

    # Drop temporary percentile columns
    drop_cols = [c for c in df.columns if c.endswith("_pct") or c in ("ball_striking", "short_game")]
    df = df.drop(columns=drop_cols)

    # Print archetype distribution
    counts = df["archetype"].value_counts()
    print(f"  Archetype distribution:")
    for arch, cnt in counts.items():
        print(f"    {arch}: {cnt}")

    return df


# ---------------------------------------------------------------------------
# 2e. compare_predictions_vs_actuals
# ---------------------------------------------------------------------------
def compare_predictions_vs_actuals(predictions, actuals):
    """
    Inner join on (player_name, round, category).
    miss = actual_sg - predicted_sg
    Positive miss = underpredicted (player gained MORE than expected).
    """
    if predictions.empty or actuals.empty:
        return pd.DataFrame()

    merged = predictions.merge(
        actuals, on=["player_name", "round", "category"], how="inner"
    )
    merged["miss"] = merged["actual_sg"] - merged["predicted_sg"]

    pred_players = predictions["player_name"].nunique()
    act_players = actuals["player_name"].nunique()
    match_players = merged["player_name"].nunique()

    print(
        f"  Comparison: {match_players} matched players "
        f"(pred: {pred_players}, actual: {act_players})"
    )
    unmatched_pred = set(predictions["player_name"].unique()) - set(
        merged["player_name"].unique()
    )
    if unmatched_pred:
        print(f"  Unmatched from predictions: {len(unmatched_pred)}")

    return merged


# ---------------------------------------------------------------------------
# 2f. build_diagnostic_records
# ---------------------------------------------------------------------------
def build_diagnostic_records(comparison, archetypes, event_name, year, event_id):
    """
    Merge comparison with archetype rolling stats. Add metadata.
    Returns the record set written to Parquet.
    """
    if comparison.empty:
        return pd.DataFrame()

    arch_cols = [
        "player_name",
        "archetype",
        "sg_ott_rolling",
        "sg_app_rolling",
        "sg_arg_rolling",
        "sg_putt_rolling",
        "driving_dist_rolling",
        "driving_acc_rolling",
    ]
    available = [c for c in arch_cols if c in archetypes.columns]
    merged = comparison.merge(archetypes[available], on="player_name", how="left")

    merged["event_name"] = event_name
    merged["year"] = year
    merged["event_id"] = str(event_id)
    merged["run_timestamp"] = datetime.now().isoformat()

    # Fill missing archetypes
    if "archetype" not in merged.columns:
        merged["archetype"] = "Unknown"
    merged["archetype"] = merged["archetype"].fillna("Unknown")

    return merged


# ---------------------------------------------------------------------------
# 2g. append_to_diagnostic
# ---------------------------------------------------------------------------
def append_to_diagnostic(records):
    """
    Append to permanent_data/sg_diagnostic.parquet.
    Same pattern as _append_to_ledger() in sheets_storage.py.
    """
    if records.empty:
        return

    diag_dir = os.path.dirname(DIAGNOSTIC_PATH)
    os.makedirs(diag_dir, exist_ok=True)

    if os.path.exists(DIAGNOSTIC_PATH):
        try:
            existing = pd.read_parquet(DIAGNOSTIC_PATH)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if existing.empty:
        combined = records
    else:
        combined = pd.concat([existing, records], ignore_index=True)

    # Normalize dedup columns
    for col in DIAGNOSTIC_DEDUP_COLS:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.lower().str.strip()

    # Keep first occurrence (existing wins over new)
    combined = combined.drop_duplicates(subset=DIAGNOSTIC_DEDUP_COLS, keep="first")

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=diag_dir)
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, DIAGNOSTIC_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    new_count = len(combined) - len(existing)
    print(
        f"  [diagnostic] {new_count} new rows added "
        f"({len(combined)} total in diagnostic parquet)"
    )


# ---------------------------------------------------------------------------
# 2h. compute_analysis
# ---------------------------------------------------------------------------
def compute_analysis(comparison, archetypes):
    """
    Returns dict with 4 analysis tables:
    1. category_bias  -- groupby category
    2. biggest_misses -- top 15 by abs avg total miss
    3. archetype_analysis -- groupby archetype x category
    4. event_summary  -- dict with summary stats
    """
    if comparison.empty:
        return {}

    # Merge archetypes
    arch_cols = ["player_name", "archetype"]
    available = [c for c in arch_cols if c in archetypes.columns]
    merged = comparison.merge(archetypes[available], on="player_name", how="left")
    if "archetype" not in merged.columns:
        merged["archetype"] = "Unknown"
    merged["archetype"] = merged["archetype"].fillna("Unknown")

    # 1. Category Bias
    cat_bias = (
        merged.groupby("category")["miss"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cat_bias.columns = ["category", "avg_miss", "std_miss", "count"]

    # 2. Biggest Player Misses (total category, avg across rounds)
    total_rows = merged[merged["category"] == "total"]
    if not total_rows.empty:
        player_total = (
            total_rows.groupby("player_name")
            .agg(total_miss=("miss", "mean"), n_rounds=("miss", "count"))
            .reset_index()
        )
    else:
        player_total = pd.DataFrame(
            columns=["player_name", "total_miss", "n_rounds"]
        )

    # Per-category misses
    for cat in ["ott", "app", "arg", "putt"]:
        cat_rows = merged[merged["category"] == cat]
        if not cat_rows.empty:
            cat_avg = (
                cat_rows.groupby("player_name")["miss"]
                .mean()
                .reset_index()
                .rename(columns={"miss": f"{cat}_miss"})
            )
            player_total = player_total.merge(cat_avg, on="player_name", how="left")

    # Add archetype
    arch_lookup = archetypes.set_index("player_name")["archetype"].to_dict()
    player_total["archetype"] = player_total["player_name"].map(arch_lookup).fillna(
        "Unknown"
    )

    player_total["abs_total_miss"] = player_total["total_miss"].abs()
    biggest_misses = player_total.nlargest(15, "abs_total_miss").drop(
        columns=["abs_total_miss"]
    )

    # 3. Archetype Analysis -- pivot
    arch_agg = (
        merged.groupby(["archetype", "category"])["miss"]
        .agg(["mean", "count"])
        .reset_index()
    )
    arch_pivot = arch_agg.pivot_table(
        index="archetype", columns="category", values="mean"
    )
    # Add count
    arch_counts = merged.groupby("archetype")["player_name"].nunique().reset_index()
    arch_counts.columns = ["archetype", "n_players"]

    # 4. Event Summary
    n_players = merged["player_name"].nunique()
    total_possible = n_players * 4  # 4 rounds
    actual_rounds = len(merged[merged["category"] == "total"])
    sg_data_pct = actual_rounds / total_possible * 100 if total_possible > 0 else 0
    avg_abs_miss = merged[merged["category"] == "total"]["miss"].abs().mean()

    event_summary = {
        "n_players": n_players,
        "sg_data_pct": round(sg_data_pct, 1),
        "avg_abs_miss": round(avg_abs_miss, 3) if not pd.isna(avg_abs_miss) else 0,
    }

    return {
        "category_bias": cat_bias,
        "biggest_misses": biggest_misses,
        "archetype_pivot": arch_pivot,
        "archetype_counts": arch_counts,
        "event_summary": event_summary,
    }


# ---------------------------------------------------------------------------
# 2i. compute_recurring_misses
# ---------------------------------------------------------------------------
def compute_recurring_misses(diagnostic_path=None):
    """
    Read accumulated Parquet. Find players appearing in 2+ events.
    Returns DataFrame: player_name, n_events, avg_total_miss, direction, consistent_categories
    """
    path = diagnostic_path or DIAGNOSTIC_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    total_rows = df[df["category"] == "total"]
    if total_rows.empty:
        return pd.DataFrame()

    # Players with 2+ events
    player_events = total_rows.groupby("player_name")["event_id"].nunique()
    repeat_players = player_events[player_events >= 2].index.tolist()

    if not repeat_players:
        return pd.DataFrame()

    records = []
    for player in repeat_players:
        pdata = df[df["player_name"] == player]
        n_events = pdata["event_id"].nunique()

        total_miss = pdata[pdata["category"] == "total"]["miss"].mean()
        direction = "UNDER" if total_miss > 0 else "OVER"

        # Check which categories are consistently in the same direction
        consistent = []
        for cat in ["ott", "app", "arg", "putt"]:
            cat_data = pdata[pdata["category"] == cat]
            if len(cat_data) < 2:
                continue
            # Check if avg miss by event is consistently same sign
            event_avgs = cat_data.groupby("event_id")["miss"].mean()
            if len(event_avgs) >= 2:
                if (event_avgs > 0).all():
                    consistent.append(f"{cat}(+)")
                elif (event_avgs < 0).all():
                    consistent.append(f"{cat}(-)")

        records.append(
            {
                "player_name": player,
                "n_events": n_events,
                "avg_total_miss": round(total_miss, 3),
                "direction": direction,
                "consistent_categories": ", ".join(consistent) if consistent else "",
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values("avg_total_miss", key=abs, ascending=False)
    return result


# ---------------------------------------------------------------------------
# 2j. build_diagnostic_email_html
# ---------------------------------------------------------------------------
def build_diagnostic_email_html(event_name, analysis, recurring, is_adjusted=True):
    """
    HTML email following round_sim.py table styling.
    5 sections: Summary, Category Bias, Biggest Misses, Archetype, Recurring.
    """
    summary = analysis.get("event_summary", {})
    cat_bias = analysis.get("category_bias", pd.DataFrame())
    biggest = analysis.get("biggest_misses", pd.DataFrame())
    arch_pivot = analysis.get("archetype_pivot", pd.DataFrame())
    arch_counts = analysis.get("archetype_counts", pd.DataFrame())

    th_style = (
        'style="background:#343a40; color:white; padding:8px 10px; '
        'text-align:left; font-size:13px; font-family:Arial,sans-serif;"'
    )
    td_style = 'style="padding:6px 10px; font-size:13px; font-family:Arial,sans-serif; border-bottom:1px solid #dee2e6;"'

    # --- Section 1: Event Summary ---
    sg_label = "Adjusted SG" if is_adjusted else "Raw SG (not field-adjusted)"
    raw_warning = "" if is_adjusted else """
        <p style="margin:4px 0; font-family:Arial,sans-serif; font-size:12px; color:#d35400; font-weight:600;">
            Using raw SG actuals -- predictions are field-adjusted.
            Absolute bias reflects adjustment gap, not model error.
            Relative player/archetype differences are still meaningful.
        </p>"""

    summary_html = f"""
    <div style="background:#e9ecef; border-radius:8px; padding:16px; margin-bottom:20px;">
        <h2 style="margin:0 0 8px 0; color:#343a40; font-family:Arial,sans-serif;">
            SG Diagnostic: {event_name}
        </h2>
        <p style="margin:4px 0; font-family:Arial,sans-serif; font-size:14px;">
            Players compared: <b>{summary.get('n_players', 0)}</b> |
            SG data availability: <b>{summary.get('sg_data_pct', 0)}%</b> |
            Avg absolute miss: <b>{summary.get('avg_abs_miss', 0):.3f}</b> |
            Data: <b>{sg_label}</b>
        </p>{raw_warning}
        <p style="margin:2px 0; font-family:Arial,sans-serif; font-size:12px; color:#666;">
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} |
            Positive miss = underpredicted (player gained more than expected)
        </p>
    </div>
    """

    # --- Section 2: Category Bias ---
    cat_rows_html = ""
    if not cat_bias.empty:
        for _, row in cat_bias.iterrows():
            avg = row["avg_miss"]
            color = "#28a745" if avg > 0 else "#dc3545" if avg < 0 else "#333"
            interp = "Underpredicted" if avg > 0.02 else "Overpredicted" if avg < -0.02 else "Accurate"
            cat_rows_html += f"""
            <tr>
                <td {td_style}>{row['category'].upper()}</td>
                <td {td_style} style="text-align:center; color:{color}; font-weight:600;">{avg:+.3f}</td>
                <td {td_style} style="text-align:center;">{row['std_miss']:.3f}</td>
                <td {td_style} style="text-align:center;">{int(row['count'])}</td>
                <td {td_style}>{interp}</td>
            </tr>"""

    cat_html = f"""
    <h3 style="color:#343a40; font-family:Arial,sans-serif; margin:20px 0 8px 0;">
        Category Bias
    </h3>
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
        <tr>
            <th {th_style}>Category</th>
            <th {th_style}>Avg Miss</th>
            <th {th_style}>Std Dev</th>
            <th {th_style}>N</th>
            <th {th_style}>Interpretation</th>
        </tr>
        {cat_rows_html}
    </table>
    """

    # --- Section 3: Biggest Player Misses ---
    miss_rows_html = ""
    if not biggest.empty:
        for _, row in biggest.iterrows():
            def _cell(val, bold=False):
                if pd.isna(val):
                    return f'<td {td_style} style="text-align:center; color:#999;">-</td>'
                color = "#28a745" if val > 0 else "#dc3545" if val < 0 else "#333"
                weight = "font-weight:600;" if bold else ""
                return f'<td {td_style} style="text-align:center; color:{color}; {weight}">{val:+.2f}</td>'

            miss_rows_html += f"""
            <tr>
                <td {td_style}>{row['player_name'].title()}</td>
                <td {td_style}>{row.get('archetype', 'Unknown')}</td>
                {_cell(row.get('total_miss'), bold=True)}
                {_cell(row.get('ott_miss'))}
                {_cell(row.get('app_miss'))}
                {_cell(row.get('arg_miss'))}
                {_cell(row.get('putt_miss'))}
                <td {td_style} style="text-align:center;">{int(row.get('n_rounds', 0))}</td>
            </tr>"""

    miss_html = f"""
    <h3 style="color:#343a40; font-family:Arial,sans-serif; margin:20px 0 8px 0;">
        Biggest Player Misses (Top 15)
    </h3>
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
        <tr>
            <th {th_style}>Player</th>
            <th {th_style}>Archetype</th>
            <th {th_style}>Total</th>
            <th {th_style}>OTT</th>
            <th {th_style}>APP</th>
            <th {th_style}>ARG</th>
            <th {th_style}>PUTT</th>
            <th {th_style}>Rounds</th>
        </tr>
        {miss_rows_html if miss_rows_html else '<tr><td colspan="8" style="padding:10px; color:#999;">No data</td></tr>'}
    </table>
    """

    # --- Section 4: Archetype Analysis ---
    arch_rows_html = ""
    if not arch_pivot.empty:
        for archetype in arch_pivot.index:
            row = arch_pivot.loc[archetype]
            n_players = 0
            if not arch_counts.empty:
                match = arch_counts[arch_counts["archetype"] == archetype]
                if not match.empty:
                    n_players = int(match["n_players"].iloc[0])

            cells = ""
            for cat in ["ott", "app", "arg", "putt", "total"]:
                val = row.get(cat, np.nan)
                if pd.isna(val):
                    cells += f'<td {td_style} style="text-align:center; color:#999;">-</td>'
                else:
                    color = "#28a745" if val > 0 else "#dc3545" if val < 0 else "#333"
                    bold = "font-weight:700;" if abs(val) > 0.1 else ""
                    cells += f'<td {td_style} style="text-align:center; color:{color}; {bold}">{val:+.3f}</td>'

            arch_rows_html += f"""
            <tr>
                <td {td_style} style="font-weight:600;">{archetype}</td>
                <td {td_style} style="text-align:center;">{n_players}</td>
                {cells}
            </tr>"""

    arch_html = f"""
    <h3 style="color:#343a40; font-family:Arial,sans-serif; margin:20px 0 8px 0;">
        Archetype Analysis
    </h3>
    <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
        <tr>
            <th {th_style}>Archetype</th>
            <th {th_style}>N Players</th>
            <th {th_style}>OTT Miss</th>
            <th {th_style}>APP Miss</th>
            <th {th_style}>ARG Miss</th>
            <th {th_style}>PUTT Miss</th>
            <th {th_style}>Total Miss</th>
        </tr>
        {arch_rows_html if arch_rows_html else '<tr><td colspan="7" style="padding:10px; color:#999;">No data</td></tr>'}
    </table>
    """

    # --- Section 5: Recurring Misses (conditional) ---
    recur_html = ""
    if recurring is not None and not recurring.empty:
        recur_rows = ""
        for _, row in recurring.head(20).iterrows():
            val = row["avg_total_miss"]
            color = "#28a745" if val > 0 else "#dc3545"
            recur_rows += f"""
            <tr>
                <td {td_style}>{row['player_name'].title()}</td>
                <td {td_style} style="text-align:center;">{row['n_events']}</td>
                <td {td_style} style="text-align:center; color:{color}; font-weight:600;">{val:+.3f}</td>
                <td {td_style} style="text-align:center;">{row['direction']}</td>
                <td {td_style}>{row.get('consistent_categories', '')}</td>
            </tr>"""

        recur_html = f"""
        <h3 style="color:#343a40; font-family:Arial,sans-serif; margin:20px 0 8px 0;">
            Recurring Misses (Cross-Event)
        </h3>
        <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
            <tr>
                <th {th_style}>Player</th>
                <th {th_style}>Events</th>
                <th {th_style}>Avg Miss</th>
                <th {th_style}>Direction</th>
                <th {th_style}>Consistent Categories</th>
            </tr>
            {recur_rows}
        </table>
        """

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:900px; margin:0 auto; padding:20px; background:#f5f5f5;">
        <div style="background:white; border-radius:8px; padding:24px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            {summary_html}
            {cat_html}
            {miss_html}
            {arch_html}
            {recur_html}
            <p style="color:#999; font-size:11px; margin-top:24px; text-align:center;">
                Generated by Golf Sim SG Diagnostic
            </p>
        </div>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# 2k. send_diagnostic_email
# ---------------------------------------------------------------------------
def send_diagnostic_email(html, event_name):
    """Send diagnostic email via Gmail SMTP."""
    if not EMAIL_PASSWORD:
        print("  EMAIL_PASSWORD not set -- skipping email")
        return False
    if not EMAIL_TO or not EMAIL_TO[0]:
        print("  EMAIL_RECIPIENTS not set -- skipping email")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"SG Diagnostic: {event_name}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  Diagnostic email sent successfully")
        return True
    except Exception as e:
        print(f"  Warning: Email failed: {e}")
        return False


# ---------------------------------------------------------------------------
# 2l. print_console_summary
# ---------------------------------------------------------------------------
def print_console_summary(analysis, recurring, is_adjusted=True):
    """Terminal output of key findings."""
    if not analysis:
        print("\n  No analysis results to display.")
        return

    summary = analysis.get("event_summary", {})
    cat_bias = analysis.get("category_bias", pd.DataFrame())
    biggest = analysis.get("biggest_misses", pd.DataFrame())
    arch_pivot = analysis.get("archetype_pivot", pd.DataFrame())

    sg_label = "Field-Adjusted SG" if is_adjusted else "Raw SG (unadjusted)"

    print("\n" + "=" * 60)
    print("  SG DIAGNOSTIC SUMMARY")
    print(f"  Actuals source: {sg_label}")
    print("=" * 60)

    if not is_adjusted:
        print("\n  NOTE: Actuals are RAW SG (not field-adjusted).")
        print("  Predictions use adjusted SG, so systematic negative bias")
        print("  is EXPECTED and does NOT indicate model error.")

    print(f"\n  Players compared: {summary.get('n_players', 0)}")
    print(f"  SG data coverage: {summary.get('sg_data_pct', 0)}%")
    print(f"  Avg absolute miss: {summary.get('avg_abs_miss', 0):.3f}")

    # Category bias
    if not cat_bias.empty:
        print("\n  CATEGORY BIAS:")
        print("  " + "-" * 45)
        for _, row in cat_bias.iterrows():
            avg = row["avg_miss"]
            arrow = "^" if avg > 0.02 else "v" if avg < -0.02 else "="
            print(
                f"    {row['category'].upper():>5s}: {avg:+.3f}  "
                f"(std={row['std_miss']:.3f}, n={int(row['count'])}) {arrow}"
            )

    # Top 5 biggest misses
    if not biggest.empty:
        print("\n  TOP 5 PLAYER MISSES:")
        print("  " + "-" * 45)
        for _, row in biggest.head(5).iterrows():
            tm = row.get("total_miss", 0)
            print(
                f"    {row['player_name']:>25s}: {tm:+.2f} "
                f"({row.get('archetype', 'Unknown')})"
            )

    # Archetype insights
    if not arch_pivot.empty:
        print("\n  ARCHETYPE INSIGHTS:")
        print("  " + "-" * 45)
        for archetype in arch_pivot.index:
            row = arch_pivot.loc[archetype]
            total = row.get("total", np.nan)
            if pd.isna(total):
                continue
            print(f"    {archetype:>22s}: total miss {total:+.3f}")

    # Recurring misses
    if recurring is not None and not recurring.empty:
        print(f"\n  RECURRING MISSES ({len(recurring)} players across 2+ events):")
        print("  " + "-" * 45)
        for _, row in recurring.head(5).iterrows():
            print(
                f"    {row['player_name']:>25s}: {row['avg_total_miss']:+.3f} "
                f"({row['direction']}, {row['n_events']} events)"
            )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# 2m. CLI (main)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Post-event SG prediction diagnostic")
    parser.add_argument("--event-id", type=str, help="Override event ID")
    parser.add_argument("--year", type=int, help="Override year")
    parser.add_argument("--no-email", action="store_true", help="Skip email")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show accumulated cross-event report from Parquet",
    )
    args = parser.parse_args()

    # --- Report mode: cross-event analysis ---
    if args.report:
        print("\n  Cross-Event SG Diagnostic Report")
        print("  " + "=" * 40)
        if not os.path.exists(DIAGNOSTIC_PATH):
            print("  No diagnostic data found yet. Run after at least one event.")
            return

        df = pd.read_parquet(DIAGNOSTIC_PATH)
        n_events = df["event_id"].nunique()
        print(f"  Events in database: {n_events}")
        print(f"  Total rows: {len(df)}")

        # Overall category bias across all events
        total = df[df["category"] != "total"]
        cat_summary = total.groupby("category")["miss"].agg(["mean", "std", "count"])
        print("\n  Overall Category Bias (all events):")
        for cat, row in cat_summary.iterrows():
            print(f"    {cat.upper():>5s}: {row['mean']:+.3f} (n={int(row['count'])})")

        recurring = compute_recurring_misses()
        if not recurring.empty:
            print(f"\n  Recurring Misses ({len(recurring)} repeat players):")
            for _, row in recurring.head(10).iterrows():
                print(
                    f"    {row['player_name']:>25s}: {row['avg_total_miss']:+.3f} "
                    f"({row['direction']}, {row['n_events']} events) "
                    f"{row.get('consistent_categories', '')}"
                )
        return

    # --- Normal mode: run diagnostic for an event ---
    eid = args.event_id or str(event_ids[0])
    year = args.year or datetime.now().year
    event_name = tourney

    print("\n" + "=" * 60)
    print("  SG PREDICTION DIAGNOSTIC")
    print("=" * 60)
    print(f"  Event: {event_name}")
    print(f"  Event ID: {eid}")
    print(f"  Year: {year}")

    # Step 1: Load predictions
    print("\n  Loading predictions...")
    predictions = load_predictions(event_name)
    if predictions.empty:
        print("  ERROR: No predictions found. Exiting.")
        print(
            "  (avg_expected_cat_sg_{tourney}.csv must exist -- "
            "run before weekly cleanup)"
        )
        return

    # Step 2: Fetch actuals
    print("\n  Fetching actuals...")
    actuals, is_adjusted = fetch_actuals(eid, year=year)
    if actuals.empty:
        print("  No SG data available for this event. Exiting.")
        return

    if not is_adjusted:
        print("\n  ** NOTE: Using RAW (unadjusted) SG actuals. **")
        print("  ** Predictions are field-adjusted. Absolute bias reflects the **")
        print("  ** adjustment gap, NOT model error. Relative differences      **")
        print("  ** between players/archetypes are still informative.          **")

    # Step 3: Compute archetypes
    print("\n  Computing player archetypes...")
    field_players = list(
        set(predictions["player_name"].unique())
        | set(actuals["player_name"].unique())
    )
    archetypes = compute_rolling_archetypes(eid, field_players, actuals_df=actuals)

    # Step 4: Compare predictions vs actuals
    print("\n  Comparing predictions vs actuals...")
    comparison = compare_predictions_vs_actuals(predictions, actuals)
    if comparison.empty:
        print("  No matched data for comparison. Exiting.")
        return

    # Step 5: Build & store diagnostic records
    print("\n  Building diagnostic records...")
    records = build_diagnostic_records(comparison, archetypes, event_name, year, eid)
    if not records.empty:
        if not is_adjusted:
            records["sg_type"] = "raw"
        else:
            records["sg_type"] = "adjusted"
        append_to_diagnostic(records)

    # Step 6: Compute analysis
    print("\n  Computing analysis...")
    analysis = compute_analysis(comparison, archetypes)

    # Step 7: Check for recurring misses
    recurring = compute_recurring_misses()

    # Step 8: Print console summary
    print_console_summary(analysis, recurring, is_adjusted=is_adjusted)

    # Step 9: Send email
    if not args.no_email and analysis:
        print("\n  Sending diagnostic email...")
        html = build_diagnostic_email_html(event_name, analysis, recurring, is_adjusted=is_adjusted)
        send_diagnostic_email(html, event_name)
    elif args.no_email:
        print("\n  Email skipped (--no-email)")

    print("\n  Done.")


if __name__ == "__main__":
    main()

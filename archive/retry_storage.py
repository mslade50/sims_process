"""
retry_storage.py — Re-push sim results to Google Sheets without re-running the simulation.

Loads saved CSVs from the most recent new_sim.py run and calls the same
store_tournament_matchups / store_finish_positions functions.

Usage:
    python retry_storage.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

from sim_inputs import tourney, event_ids, name_replacements
from sheets_storage import (
    get_spreadsheet,
    store_tournament_matchups,
    store_finish_positions,
    load_dg_id_lookup,
)


def load_latest_matchups():
    """Load the most recent combined matchups CSV."""
    pattern = f"./{tourney}/matchups_ftsimp_{tourney}_*.csv"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"[retry] No matchup files found matching {pattern}")
        return None
    path = files[-1]
    print(f"[retry] Loading matchups from {path}")
    return pd.read_csv(path)


def load_finish_positions():
    """Load finish_test.csv and replicate the dedup/merge from new_sim.py."""
    path = "finish_test.csv"
    if not os.path.exists(path):
        print(f"[retry] {path} not found")
        return None

    df = pd.read_csv(path, index_col=0)
    print(f"[retry] Loaded {len(df)} raw finish position rows from {path}")

    if df.empty:
        return df

    # Dedup: best price per player/market (highest decimal odds)
    df = (
        df.sort_values(
            ["player_name", "market_type", "decimal_odds"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["player_name", "market_type"], keep="first")
    )

    # Merge sample + my_pred from predictions file
    pred_path = None
    for p in [f"final_predictions_{tourney}.csv", f"pre_course_fit_{tourney}.csv"]:
        if os.path.exists(p):
            pred_path = p
            break

    if pred_path:
        preds = pd.read_csv(pred_path)
        preds["player_name"] = (
            preds["player_name"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace(name_replacements)
        )
        if "sample" in preds.columns:
            sample_df = preds[["player_name", "sample"]].drop_duplicates("player_name")
            df = df.merge(sample_df, on="player_name", how="left", suffixes=("", "_pred"))
            if "sample_pred" in df.columns:
                df["sample"] = df["sample"].fillna(df["sample_pred"])
                df.drop(columns=["sample_pred"], inplace=True)

        # Look for prediction column: "my_pred" (final_predictions) or "pred" (pre_course_fit)
        pred_col = None
        if "my_pred" in preds.columns:
            pred_col = "my_pred"
        elif "pred" in preds.columns:
            pred_col = "pred"

        if pred_col:
            pred_join = preds[["player_name", pred_col]].rename(columns={pred_col: "my_pred_file"})
            pred_join = pred_join.drop_duplicates("player_name")
            df = df.merge(pred_join, on="player_name", how="left")
            if "my_pred" not in df.columns or df["my_pred"].isna().all():
                df["my_pred"] = df["my_pred_file"]
            else:
                df["my_pred"] = df["my_pred"].fillna(df["my_pred_file"])
            df.drop(columns=["my_pred_file"], inplace=True, errors="ignore")

    print(f"[retry] {len(df)} finish position rows after dedup/merge")
    return df


def main():
    print(f"[retry] Tournament: {tourney}, event_id: {event_ids[0]}")
    print()

    # Load data from disk
    matchups_df = load_latest_matchups()
    finish_df = load_finish_positions()

    if matchups_df is None and finish_df is None:
        print("[retry] Nothing to push — no saved data found.")
        sys.exit(1)

    # Authenticate once
    print("\n[retry] Connecting to Google Sheets...")
    spreadsheet = get_spreadsheet()

    dg_id_lookup = load_dg_id_lookup(tourney, name_replacements)

    # Push tournament matchups
    if matchups_df is not None and not matchups_df.empty:
        print(f"\n[retry] Pushing {len(matchups_df)} tournament matchup rows...")
        store_tournament_matchups(
            matchups_df,
            tourney,
            event_ids[0],
            dg_id_lookup=dg_id_lookup,
            spreadsheet=spreadsheet,
        )
    else:
        print("[retry] No matchup data to push.")

    # Push finish positions
    if finish_df is not None and not finish_df.empty:
        print(f"\n[retry] Pushing {len(finish_df)} finish position rows...")
        store_finish_positions(
            finish_df,
            tourney,
            event_ids[0],
            dg_id_lookup=dg_id_lookup,
            spreadsheet=spreadsheet,
        )
    else:
        print("[retry] No finish position data to push.")

    print("\n[retry] Done.")


if __name__ == "__main__":
    main()

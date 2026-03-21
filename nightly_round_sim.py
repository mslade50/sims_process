"""
nightly_round_sim.py - Self-Sufficient Backup Round Simulation Pipeline

Fully autonomous backup that auto-detects the completed round from DataGolf,
reads pre-tournament weather/scoring fallbacks from the Sheet, updates the
Sheet's primary fields, and runs the pipeline. No manual Sheet update needed.

Pipeline: live_stats_engine.py -> round_sim.py

Exit codes:
    0 = bets already stored OR just stored successfully OR no active round
    1 = pipeline failed (sim or storage error)
    2 = unexpected error

Usage:
    python nightly_round_sim.py              # Auto-detect round from DataGolf
    python nightly_round_sim.py --dry-run    # Preview checks only

Scheduled via .github/workflows/nightly-round-sim.yml:
    9:45 PM EST (01:45 UTC+1) Thursday through Sunday
"""

import os
import sys
import subprocess
import time


def _setup_env():
    """Ensure project root is on sys.path and .env is loaded."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from dotenv import load_dotenv
    load_dotenv()
    return project_root


def _get_event_id():
    """Get current event_id from Google Sheet."""
    from sheet_config import load_config
    cfg = load_config()
    return str(cfg["event_id"])


def _count_round_bets(spreadsheet, event_id, sim_round):
    """
    Count existing round matchup bets for this event + round.

    Reads the Round Matchups tab, filters to event_id and round.
    """
    from grade_bets import read_sheet_as_df

    df = read_sheet_as_df(spreadsheet, "Round Matchups")
    if df.empty:
        return 0

    # Filter to this event and round
    mask = (
        (df["event_id"].astype(str) == str(event_id)) &
        (df["round"].astype(str) == str(sim_round))
    )
    return int(mask.sum())


def _run_subprocess(cmd, label):
    """Run a subprocess. Returns True on success."""
    print(f"\n  Running {label}...")
    print(f"  Command: {' '.join(cmd)}")
    print("  " + "-" * 50)

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"  ERROR: {label} failed with exit code {result.returncode}")
        return False

    print(f"  {label} completed successfully.")
    return True


def _get_fallback_values(config, sim_round):
    """
    Extract per-round fallback values from Sheet config for the round to simulate.

    Returns (expected_score, wind_str, dew_str) where:
        expected_score: float or None (pre-tournament scoring estimate)
        wind_str: comma-separated hourly wind, or "" if unavailable
        dew_str: comma-separated hourly dew, or "" if unavailable
    """
    # Per-round expected scoring (e.g. expected_score_r2 = [70.5])
    score_key = f"expected_score_r{sim_round}"
    score_list = config.get(score_key, [])
    expected_score = score_list[0] if score_list else None

    # Per-round wind (e.g. wind_r2 = [5, 6, 7, ...])
    wind_key = f"wind_r{sim_round}"
    wind_list = config.get(wind_key, [])
    wind_str = ",".join(str(int(v)) for v in wind_list) if wind_list else ""

    # Per-round dew (e.g. dew_r2 = [36, 38, ...])
    dew_key = f"dew_r{sim_round}"
    dew_list = config.get(dew_key, [])
    dew_str = ",".join(str(int(v)) for v in dew_list) if dew_list else ""

    return expected_score, wind_str, dew_str


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Nightly backup round simulation pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Preview checks without running sims")
    args = parser.parse_args()

    project_root = _setup_env()
    python = sys.executable

    print("\n" + "=" * 60)
    print("  NIGHTLY ROUND SIM BACKUP (self-sufficient)")
    print("=" * 60)

    if args.dry_run:
        print("  MODE: DRY RUN")

    # ------------------------------------------------------------------
    # Step 1: Auto-detect completed round from DataGolf
    # ------------------------------------------------------------------
    print("\n  Auto-detecting completed round from DataGolf...")
    try:
        api_key = os.getenv("DATAGOLF_API_KEY")
        if not api_key:
            print("  ERROR: DATAGOLF_API_KEY not set")
            sys.exit(1)

        from api_utils import detect_completed_round
        detected_round = detect_completed_round(api_key)
    except Exception as e:
        print(f"  ERROR: Could not detect round from DataGolf: {e}")
        sys.exit(1)

    print(f"  Detected round: R{detected_round}")

    # Pre-event (round=0) or R4 done — nothing to sim
    if detected_round == 0:
        print("\n  Pre-event (no round data). No round sim needed.")
        sys.exit(0)

    if detected_round == 4:
        print("\n  Tournament complete (R4 done). No round sim needed.")
        sys.exit(0)

    sim_round = detected_round + 1
    print(f"  Round to simulate: R{sim_round}")

    # ------------------------------------------------------------------
    # Step 2: Check if bets already stored for this round
    # ------------------------------------------------------------------
    print("\n  Checking for existing round matchup bets...")
    try:
        event_id = _get_event_id()
    except Exception as e:
        print(f"  ERROR: Could not read event_id from sim_inputs: {e}")
        sys.exit(1)

    from sheets_storage import get_spreadsheet
    spreadsheet = get_spreadsheet()

    existing = _count_round_bets(spreadsheet, event_id, sim_round)
    print(f"  Event {event_id}, R{sim_round}: {existing} round matchup bets found")

    if existing > 0:
        print(f"\n  Already stored: {existing} R{sim_round} matchup bets exist.")
        print("  Skipping — nothing to do.")
        print("=" * 60 + "\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 3: Read Sheet config and update primary fields if stale
    # ------------------------------------------------------------------
    print("\n  Reading Sheet config...")
    try:
        from sheet_config import load_config, write_primary_fields
        config = load_config()
    except Exception as e:
        print(f"  ERROR: Could not read Sheet config: {e}")
        sys.exit(1)

    sheet_round = config["round_num"]

    if sheet_round == detected_round:
        # Sheet already up to date — user updated manually. Don't overwrite.
        print(f"  Sheet round ({sheet_round}) matches detected round. Using user config.")
    else:
        # Sheet is stale — update primary fields from per-round fallbacks
        print(f"  Sheet round ({sheet_round}) is stale (detected: {detected_round}).")
        print(f"  Updating Sheet primary fields from R{sim_round} fallbacks...")

        expected_score, wind_str, dew_str = _get_fallback_values(config, sim_round)

        if expected_score is None:
            print(f"  WARNING: No expected_score_r{sim_round} in Sheet. Using current expected_score_1.")
            expected_score = config["expected_score_1"]

        # Fall back to primary wind/dew if per-round not available
        if not wind_str:
            print(f"  WARNING: No wind_r{sim_round} in Sheet. Using current primary 'wind'.")
            wind_str = ",".join(str(int(v)) for v in config["wind"]) if config["wind"] else ""

        if not dew_str:
            print(f"  WARNING: No dew_r{sim_round} in Sheet. Using current primary 'dew'.")
            dew_str = ",".join(str(int(v)) for v in config["dew"]) if config["dew"] else ""

        print(f"  -> round: {detected_round}")
        print(f"  -> expected_score_1: {expected_score}")
        print(f"  -> wind: {wind_str[:40]}{'...' if len(wind_str) > 40 else ''}")
        print(f"  -> dew: {dew_str[:40]}{'...' if len(dew_str) > 40 else ''}")

        if not args.dry_run:
            try:
                write_primary_fields(detected_round, expected_score, wind_str, dew_str)
            except Exception as e:
                print(f"  ERROR: Could not update Sheet: {e}")
                sys.exit(1)

    print(f"\n  No R{sim_round} bets found. Running full pipeline...")

    if args.dry_run:
        print("\n  [DRY RUN] Would run: live_stats_engine.py -> round_sim.py")
        print("=" * 60 + "\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 4: Run live_stats_engine.py (creates model_predictions_r{N}.csv)
    # ------------------------------------------------------------------
    pred_file = f"model_predictions_r{sim_round}.csv"
    pred_path = os.path.join(project_root, pred_file)

    if os.path.exists(pred_path):
        print(f"\n  {pred_file} already exists, skipping live_stats_engine.py")
    else:
        success = _run_subprocess(
            [python, "live_stats_engine.py"],
            "live_stats_engine.py",
        )
        if not success:
            print("  live_stats_engine.py failed.")
            sys.exit(1)

        # Verify predictions file was created
        if not os.path.exists(pred_path):
            print(f"  ERROR: {pred_file} not created after live_stats_engine.py")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Run round_sim.py (simulates matchups, stores to Sheets)
    # ------------------------------------------------------------------
    success = _run_subprocess(
        [python, "round_sim.py", "--sim-only"],
        "round_sim.py --sim-only",
    )
    if not success:
        print("  round_sim.py failed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Verify sim cache was created
    # ------------------------------------------------------------------
    from sheet_config import load_config as _lc
    _tourney = _lc()["tourney"]
    cache_path = os.path.join(project_root, _tourney, f"sim_cache_r{sim_round}.parquet")
    if os.path.exists(cache_path):
        print(f"\n  Verified: sim cache created at {cache_path}")
    else:
        print(f"  WARNING: sim cache not found at {cache_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  SIM CACHE COMPLETE — R{sim_round}")
    print("=" * 60 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

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
    # Step 1: Detect completed round from Sheet config
    # ------------------------------------------------------------------
    print("\n  Reading round from Sheet config...")
    try:
        from sheet_config import load_config as _load_cfg
        _pre_config = _load_cfg()
        detected_round = _pre_config["round_num"]
    except Exception as e:
        print(f"  ERROR: Could not read round from Sheet config: {e}")
        sys.exit(1)

    print(f"  Sheet round: R{detected_round} complete")

    # Split-brain tripwire: the sheet (round_sim's cache dirs, bet event_ids)
    # and sim_inputs (publish target, odds scoping) must agree on the event.
    # A missed init_weekly used to leave them split: caches written under LAST
    # week's dir while publish searches the new one -> nothing publishes, the
    # board drops to consensus, and nothing said a word. Hard-fail so the
    # nightly run pages instead of simming the wrong event.
    try:
        import sim_inputs as _si
        _si_tourney = getattr(_si, "tourney", None)
        _sheet_tourney = _pre_config.get("tourney")
        if _si_tourney and _sheet_tourney and _si_tourney != _sheet_tourney:
            msg = (f"[nightly_round_sim] SPLIT-BRAIN: sheet tourney "
                   f"'{_sheet_tourney}' != sim_inputs.tourney '{_si_tourney}' - "
                   f"run init_weekly / roll sim_inputs before the nightly sim.")
            print("  " + msg)
            try:
                from maker_alerts import send_telegram
                send_telegram(msg)
            except Exception:
                pass
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"  split-brain check skipped ({e})")

    # Pre-event (round=0) — nothing to sim
    if detected_round == 0:
        print("\n  Pre-event (no round data). No round sim needed.")
        sys.exit(0)

    # R4 done — nothing to sim
    if detected_round >= 4:
        print("\n  Tournament complete (R4 done). No round sim needed.")
        sys.exit(0)

    sim_round = detected_round + 1
    print(f"  Round to simulate: R{sim_round}")

    # ------------------------------------------------------------------
    # Step 2: Use Sheet config (already loaded in Step 1)
    # ------------------------------------------------------------------
    config = _pre_config

    print(f"\n  Running sim to populate cache for R{sim_round}...")

    if args.dry_run:
        print("\n  [DRY RUN] Would run: live_stats_engine.py -> round_sim.py")
        print("=" * 60 + "\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 4: Run live_stats_engine.py (creates model_predictions_r{N}.csv)
    # ------------------------------------------------------------------
    pred_file = f"model_predictions_r{sim_round}.csv"
    pred_path = os.path.join(project_root, pred_file)
    alt_path = os.path.join(project_root, "dashboard_data", pred_file)

    if os.path.exists(pred_path):
        print(f"\n  {pred_file} already exists, skipping live_stats_engine.py")
    elif os.path.exists(alt_path):
        import shutil
        shutil.copy2(alt_path, pred_path)
        print(f"\n  {pred_file} found in dashboard_data/, copied to root. Skipping live_stats_engine.py")
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

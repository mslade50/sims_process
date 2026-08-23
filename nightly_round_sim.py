"""
nightly_round_sim.py - Backup Round Simulation Pipeline

Reads the already-current completed round from the Google Sheet and builds a
complete, internally-paired live simulation package for the next round. Round
advancement and fresh weather/scoring updates are handled by
midweek_round_automation.py; this scheduled job remains a backup once the Sheet
pointer is correct.

Pipeline: live_stats_engine.py -> round_sim.py --dry-run -> artifact validation

``round_sim.py --dry-run`` is intentional: it runs both the round and remaining-
tournament simulations while disabling email, bet storage, Telegram betting
alerts, and dashboard pushes. The workflow publishes the validated artifacts in
a separate, strict step with ``publish_sim_fairs.py --require-complete-live``.

Exit codes:
    0 = complete live package validated OR no active round
    1 = pipeline failed (sim or artifact validation error)
    2 = unexpected error

Usage:
    python nightly_round_sim.py              # Read completed round from Sheet
    python nightly_round_sim.py --dry-run    # Preview checks only

Scheduled via .github/workflows/nightly-round-sim.yml:
    9:45 PM EST (01:45 UTC+1) Thursday through Sunday
"""

import os
import sys
import subprocess
import time


_FRESHNESS_SLOP_SECONDS = 5.0


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


def _normalized_names(values, replacements=None):
    replacements = replacements or {}
    names = set()
    for value in values:
        name = str(value).strip().lower()
        if name:
            names.add(replacements.get(name, name))
    return names


def _validate_complete_live_artifacts(project_root, tourney, sim_round, run_started_at):
    """Validate that one run produced every artifact needed by a live publish.

    A full round run historically caught tournament-sim exceptions and returned
    success. Without this gate, old ``simulated_probs_live.csv`` and live tapes
    could survive on a persistent runner and be published alongside a fresh round
    cache. Require all sources to be newer than this invocation, then verify the
    live finish, made-cut, rank, and standings family shares one player/draw
    contract and that the round cache/PMF is internally complete.
    """
    import json

    import numpy as np
    import pandas as pd

    try:
        from sim_inputs import name_replacements
        replacements = name_replacements or {}
    except Exception:
        replacements = {}

    root = os.path.abspath(project_root)
    event_dir = os.path.join(root, str(tourney))
    paths = {
        "model predictions": os.path.join(
            root, f"model_predictions_r{sim_round}.csv"
        ),
        "round cache": os.path.join(event_dir, f"sim_cache_r{sim_round}.parquet"),
        "round cache metadata": os.path.join(
            event_dir, f"sim_cache_r{sim_round}_meta.json"
        ),
        "round score PMF": os.path.join(
            event_dir, f"round_score_probs_r{sim_round}.parquet"
        ),
        "live finish probabilities": os.path.join(root, "simulated_probs_live.csv"),
        "live rank probabilities": os.path.join(
            root, f"rank_probs_live_{tourney}.parquet"
        ),
        "live final-score tape": os.path.join(
            root, f"final_scores_live_{tourney}.npy"
        ),
        "live player sidecar": os.path.join(
            root, f"player_names_live_{tourney}.json"
        ),
        "live made-cut tape": os.path.join(root, f"made_cut_live_{tourney}.npy"),
        "live R2 standings tape": os.path.join(
            root, f"standings_r2_live_{tourney}.npy"
        ),
        "live R3 standings tape": os.path.join(
            root, f"standings_r3_live_{tourney}.npy"
        ),
    }

    missing = [label for label, path in paths.items() if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "complete live simulation is missing artifacts: " + ", ".join(missing)
        )

    stale = []
    freshness_floor = float(run_started_at) - _FRESHNESS_SLOP_SECONDS
    for label, path in paths.items():
        if os.path.getmtime(path) < freshness_floor:
            stale.append(label)
    if stale:
        raise RuntimeError(
            "complete live simulation did not refresh artifacts: " + ", ".join(stale)
        )

    with open(paths["round cache metadata"], encoding="utf-8") as handle:
        cache_meta = json.load(handle)
    if int(cache_meta.get("sim_round", -1)) != int(sim_round):
        raise RuntimeError(
            "round cache metadata mismatch: "
            f"expected R{sim_round}, found R{cache_meta.get('sim_round')}"
        )

    cache = pd.read_parquet(paths["round cache"])
    if cache.empty or cache.shape[1] <= 0:
        raise RuntimeError("round cache is empty")
    if not np.isfinite(cache.to_numpy(dtype=float)).all():
        raise RuntimeError("round cache contains non-finite scores")
    if int(cache_meta.get("num_players", -1)) != int(cache.shape[0]):
        raise RuntimeError("round cache player count disagrees with metadata")
    if int(cache_meta.get("num_sims", -1)) != int(cache.shape[1]):
        raise RuntimeError("round cache draw count disagrees with metadata")
    cache_players = _normalized_names(cache.index, replacements)
    if len(cache_players) != cache.shape[0]:
        raise RuntimeError("round cache contains blank or duplicate player names")

    predictions = pd.read_csv(paths["model predictions"])
    score_column = f"scores_r{sim_round}"
    if predictions.empty or not {"player_name", score_column}.issubset(
        predictions.columns
    ):
        raise RuntimeError(
            f"model predictions are empty or missing player_name/{score_column}"
        )
    prediction_players = _normalized_names(
        predictions["player_name"], replacements
    )
    if prediction_players != cache_players:
        raise RuntimeError("model prediction player set disagrees with round cache")
    prediction_scores = pd.to_numeric(predictions[score_column], errors="coerce")
    if prediction_scores.isna().any() or not np.isfinite(prediction_scores).all():
        raise RuntimeError("model predictions contain invalid round scores")

    score_pmf = pd.read_parquet(paths["round score PMF"])
    required_pmf = {"player_name", "score", "prob"}
    if score_pmf.empty or not required_pmf.issubset(score_pmf.columns):
        raise RuntimeError("round score PMF is empty or missing required columns")
    pmf_players = _normalized_names(score_pmf["player_name"], replacements)
    if pmf_players != cache_players:
        raise RuntimeError("round score PMF player set disagrees with round cache")
    pmf_prob = pd.to_numeric(score_pmf["prob"], errors="coerce")
    if pmf_prob.isna().any() or (pmf_prob < 0).any():
        raise RuntimeError("round score PMF contains invalid probabilities")
    pmf_sums = score_pmf.assign(_prob=pmf_prob).groupby("player_name")["_prob"].sum()
    if not np.allclose(pmf_sums.to_numpy(dtype=float), 1.0, atol=1e-5):
        raise RuntimeError("round score PMF probabilities do not sum to one")

    with open(paths["live player sidecar"], encoding="utf-8") as handle:
        player_names = json.load(handle)
    if not isinstance(player_names, list) or not player_names:
        raise RuntimeError("live player sidecar is empty or malformed")
    live_players = _normalized_names(player_names, replacements)
    if len(live_players) != len(player_names):
        raise RuntimeError("live player sidecar contains blank or duplicate names")

    final_scores = np.load(paths["live final-score tape"], mmap_mode="r")
    made_cut = np.load(paths["live made-cut tape"], mmap_mode="r")
    standings_r2 = np.load(paths["live R2 standings tape"], mmap_mode="r")
    standings_r3 = np.load(paths["live R3 standings tape"], mmap_mode="r")
    if final_scores.ndim != 2 or final_scores.shape[1] <= 0:
        raise RuntimeError("live final-score tape is not a non-empty 2D matrix")
    if final_scores.shape[0] != len(player_names):
        raise RuntimeError("live final-score tape disagrees with player sidecar")
    if made_cut.ndim != 2 or made_cut.shape != final_scores.shape:
        raise RuntimeError("live made-cut tape is not paired to final-score tape")
    if standings_r2.ndim != 2 or standings_r2.shape != final_scores.shape:
        raise RuntimeError("live R2 standings tape is not paired to final-score tape")
    if standings_r3.ndim != 2 or standings_r3.shape != final_scores.shape:
        raise RuntimeError("live R3 standings tape is not paired to final-score tape")
    if not np.isfinite(final_scores).all():
        raise RuntimeError("live final-score tape contains non-finite values")
    if not np.isfinite(standings_r2).all() or not np.isfinite(standings_r3).all():
        raise RuntimeError("live standings tape contains non-finite values")
    if not np.isfinite(made_cut).all() or not np.isin(made_cut, (0, 1)).all():
        raise RuntimeError("live made-cut tape must contain only binary values")

    finish = pd.read_csv(paths["live finish probabilities"])
    required_finish = {
        "player_name",
        "simulated_win_prob",
        "top_5",
        "top_10",
        "top_20",
    }
    if finish.empty or not required_finish.issubset(finish.columns):
        raise RuntimeError("live finish probabilities are empty or incomplete")
    finish_players = _normalized_names(finish["player_name"], replacements)
    if finish_players != live_players:
        raise RuntimeError("live finish probabilities disagree with player sidecar")

    rank_probs = pd.read_parquet(paths["live rank probabilities"])
    required_rank = {"player_name", "rank", "prob_u"}
    if rank_probs.empty or not required_rank.issubset(rank_probs.columns):
        raise RuntimeError("live rank probabilities are empty or incomplete")
    rank_players = _normalized_names(rank_probs["player_name"], replacements)
    if rank_players != live_players:
        raise RuntimeError("live rank probabilities disagree with player sidecar")
    rank_values = pd.to_numeric(rank_probs["prob_u"], errors="coerce")
    if rank_values.isna().any() or (rank_values < 0).any():
        raise RuntimeError("live rank probabilities contain invalid probabilities")
    rank_sums = rank_probs.assign(_prob=rank_values).groupby("player_name")["_prob"].sum()
    if not np.allclose(rank_sums.to_numpy(dtype=float), 1.0, atol=1e-5):
        raise RuntimeError("live rank probabilities do not sum to one")

    expected_mass = {
        "simulated_win_prob": 1.0,
        "top_5": min(5.0, float(len(player_names))),
        "top_10": min(10.0, float(len(player_names))),
        "top_20": min(20.0, float(len(player_names))),
    }
    for column, target in expected_mass.items():
        values = pd.to_numeric(finish[column], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise RuntimeError(f"live finish probabilities contain invalid {column}")
        if not np.isclose(float(values.sum()), target, atol=0.08):
            raise RuntimeError(
                f"live finish probability mass is invalid for {column}: "
                f"{values.sum():.4f} vs {target:.1f}"
            )

    return {
        "tourney": str(tourney),
        "sim_round": int(sim_round),
        "round_players": int(cache.shape[0]),
        "round_draws": int(cache.shape[1]),
        "tournament_players": int(final_scores.shape[0]),
        "tournament_draws": int(final_scores.shape[1]),
    }


def _write_github_outputs(summary):
    """Expose the validated round to the strict publish step in Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        should_publish = bool(summary.get("should_publish", True))
        handle.write(f"should_publish={str(should_publish).lower()}\n")
        if should_publish:
            handle.write(f"sim_round={summary['sim_round']}\n")
            handle.write(f"tourney={summary['tourney']}\n")



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
            # Do not notify from the simulation process itself. GitHub Actions
            # owns the single failure alert, preventing duplicate Telegram noise.
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"  split-brain check skipped ({e})")

    # This workflow is scheduled only after live rounds. A zero pointer here is
    # not a harmless no-op: the backup cannot produce the promised live package.
    if detected_round == 0:
        print("\n  ERROR: Sheet is still pre-event (R0); no live package can be built.")
        sys.exit(1)

    # R4 done — nothing to sim
    if detected_round >= 4:
        print("\n  Tournament complete (R4 done). No round sim needed.")
        _write_github_outputs({"should_publish": False})
        sys.exit(0)

    sim_round = detected_round + 1
    print(f"  Round to simulate: R{sim_round}")

    # ------------------------------------------------------------------
    # Step 2: Use Sheet config (already loaded in Step 1)
    # ------------------------------------------------------------------
    config = _pre_config

    print(f"\n  Running complete live sim package for R{sim_round}...")

    if args.dry_run:
        print("\n  [DRY RUN] Would run: live_stats_engine.py -> round_sim.py --dry-run")
        print("=" * 60 + "\n")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 4: Always regenerate model_predictions_r{N}.csv. Root prediction files
    # are event-agnostic and can survive on a persistent runner; existence alone
    # is never provenance.
    # ------------------------------------------------------------------
    pred_file = f"model_predictions_r{sim_round}.csv"
    pred_path = os.path.join(project_root, pred_file)
    pipeline_started_at = time.time()
    success = _run_subprocess(
        [python, "live_stats_engine.py", "--dry-run", "--no-sheet-writes"],
        "live_stats_engine.py --dry-run --no-sheet-writes",
    )
    if not success:
        print("  live_stats_engine.py failed.")
        sys.exit(1)
    if not os.path.exists(pred_path):
        print(f"  ERROR: {pred_file} not created after live_stats_engine.py")
        sys.exit(1)
    if os.path.getmtime(pred_path) < pipeline_started_at - _FRESHNESS_SLOP_SECONDS:
        print(f"  ERROR: {pred_file} was not refreshed by live_stats_engine.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Run the complete sim with every betting/notification side effect off
    # ------------------------------------------------------------------
    success = _run_subprocess(
        [python, "round_sim.py", "--dry-run"],
        "round_sim.py --dry-run (complete live markets, no alerts/storage)",
    )
    if not success:
        print("  round_sim.py failed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Fail closed unless one fresh, internally-paired package exists
    # ------------------------------------------------------------------
    _tourney = config["tourney"]
    try:
        summary = _validate_complete_live_artifacts(
            project_root, _tourney, sim_round, pipeline_started_at
        )
    except Exception as exc:
        print(f"  ERROR: complete live artifact validation failed: {exc}")
        sys.exit(1)
    _write_github_outputs(summary)
    print(
        "\n  Verified complete live package: "
        f"R{summary['sim_round']} {summary['round_players']}×{summary['round_draws']:,} "
        f"round draws; {summary['tournament_players']}×"
        f"{summary['tournament_draws']:,} tournament draws"
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  COMPLETE LIVE PACKAGE READY — R{sim_round}")
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

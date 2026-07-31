"""Reset Google Sheet round_config for a new tournament week.

First step of the weekly pipeline. Pushes the tournament-identity fields
from sim_inputs.py to the round_config tab:
    round       -> 0
    tourney     -> sim_inputs.tourney
    event_id    -> sim_inputs.event_ids[0]
    course_id   -> sim_inputs.course_id
    cut_line    -> sim_inputs.CUT_LINE   (if defined)
    course_pars -> sim_inputs.course_par (if defined)

Downstream scripts (new_sim, scoring_baseline, live_stats_engine, round_sim)
read tourney + round from the Sheet, so without this they keep using last
week's values until the Sheet is hand-edited from a phone.

Per-round arrays (wind_r1..r4, dew_r1..r4, expected_score_r1..r4) are set
later in the pipeline by humidity.py and scoring_baseline.py — not this step.

Also sweeps PAST-week per-tournament transients from the repo root (the local
mirror of the CI weekly cleanup, which only sees tracked files). These are
per-machine sim outputs nothing reads once the week ends; everything durable
they feed lives in Sheets, dashboard_data/, or permanent_data/. rank_probs
files are the SOURCE for the historical_dists archive, so a stale one is only
removed once its archived copy exists — otherwise it is kept with a warning.
"""
import glob
import os
import re

from sheet_config import reset_for_new_week

ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_DISTS = os.path.join(ROOT, "permanent_data", "historical_dists")

# Regenerated every week; safe to drop for any slug that isn't the current one.
TRANSIENT_PATTERNS = [
    "h2h_matrix_*.parquet",
    "standings_r*_live_*.npy",
    "made_cut_live_*.npy",
    "player_names_live_*.json",
    "r*_live_model_*.csv",
]
# Generic (unsuffixed) round files carry no tourney slug, so they can't be
# slug-checked — but init_weekly runs before anything this week creates them,
# so any that exist are prior-week leftovers. live_stats_engine PREFERS
# model_predictions_r1.csv over rebuilding from final_predictions when it
# exists, so a stale one silently poisons the whole live pipeline (2026-07-30).
UNCONDITIONAL_TRANSIENTS = [
    "model_predictions_r*.csv",
    "r*_live_model.csv",
    "r*_live_summary.csv",
]
# Archive-gated: source files for historical_dists — pattern -> archived name.
ARCHIVED_PATTERNS = {
    "rank_probs_live_*.parquet": "rank_probs_live.parquet",
    "rank_probs_updated_*.parquet": "rank_probs_pre.parquet",
}


def _current_tourney():
    """Regex-parse the slug from sim_inputs.py (no import: sim_inputs pulls the
    coefficient sheet at import time, which this step shouldn't depend on)."""
    try:
        with open(os.path.join(ROOT, "sim_inputs.py"), encoding="utf-8") as f:
            match = re.search(r"^tourney\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE)
        return match.group(1).lower() if match else None
    except OSError as exc:
        print(f"  [init] could not read sim_inputs.py ({exc})")
        return None


def _slug_from(fname, pattern):
    prefix, suffix = pattern.split("*")
    return fname[len(prefix):len(fname) - len(suffix)]


def sweep_stale_transients(mid_tournament=False):
    """Delete prior-week per-tournament outputs from the repo root."""
    tourney = _current_tourney()
    if not tourney:
        print("  [init] no tourney slug — skipping stale-file sweep (refuse to guess)")
        return
    removed, kept = [], []

    for pattern in TRANSIENT_PATTERNS:
        for path in glob.glob(os.path.join(ROOT, pattern)):
            fname = os.path.basename(path)
            if tourney in fname:
                continue
            try:
                os.remove(path)
                removed.append(fname)
            except OSError as exc:
                print(f"  [init] could not remove {fname}: {exc}")

    if mid_tournament:
        # Re-running init_weekly during a live event (e.g. to push a corrected
        # cut_line/course_pars) must not destroy the event's own round files —
        # they are unrecoverable and _resolve_csv would silently fall back to
        # older dashboard_data copies.
        print("  [init] mid-tournament rerun — keeping current round files "
              "(model_predictions_r*, r*_live_model*)")
    else:
        for pattern in UNCONDITIONAL_TRANSIENTS:
            for path in glob.glob(os.path.join(ROOT, pattern)):
                fname = os.path.basename(path)
                try:
                    os.remove(path)
                    removed.append(fname)
                except OSError as exc:
                    print(f"  [init] could not remove {fname}: {exc}")

    for pattern, archive_name in ARCHIVED_PATTERNS.items():
        for path in glob.glob(os.path.join(ROOT, pattern)):
            fname = os.path.basename(path)
            if tourney in fname:
                continue
            slug = _slug_from(fname, pattern)
            archived = glob.glob(os.path.join(HISTORICAL_DISTS, f"*_{slug}", archive_name))
            if not archived:
                kept.append(fname)
                print(f"  [init] KEEPING {fname} — no {archive_name} archived for "
                      f"'{slug}' under historical_dists/ (archive it, then rerun)")
                continue
            try:
                os.remove(path)
                removed.append(fname)
            except OSError as exc:
                print(f"  [init] could not remove {fname}: {exc}")

    if removed:
        print(f"  [init] swept {len(removed)} stale file(s) from past weeks: "
              f"{', '.join(sorted(removed))}")
    else:
        print("  [init] no stale per-tournament files to sweep")


def _is_mid_tournament_rerun():
    """True when the sheet (pre-reset) is mid-round on the SAME event that
    sim_inputs currently configures — i.e. this is a config touch-up, not a
    new week."""
    try:
        with open(os.path.join(ROOT, "sim_inputs.py"), encoding="utf-8") as f:
            match = re.search(r"^event_ids\s*=\s*\[\s*(\d+)", f.read(), re.MULTILINE)
        current_event = match.group(1) if match else None
        from sheet_config import get_param

        prior_round = int(float(get_param("round") or 0))
        prior_event = str(get_param("event_id") or "").strip()
        return bool(current_event) and prior_round >= 1 and prior_event == current_event
    except Exception as exc:
        print(f"  [init] could not check mid-tournament state ({exc}) — assuming new week")
        return False


if __name__ == "__main__":
    mid_tournament = _is_mid_tournament_rerun()
    reset_for_new_week()
    sweep_stale_transients(mid_tournament=mid_tournament)

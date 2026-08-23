"""Publish a compact sim_fairs.json (model fair probabilities) so the
golf_scraping odds board can grade book prices against the sim.

What it ships (per current event):
  - outrights:      winner / top_5 / top_10 / top_20 / make_cut  -> {player: prob}
                    (dead-heat-adjusted top-N — traditional sportsbooks)
  - outrights_nodh: top_5 / top_10 / top_20  -> {player: prob}
                    (no-dead-heat top-N — Kalshi / NoVig settle top-N as a binary)
  - matchups :      tournament head-to-head  -> [player_a, player_b, P(a beats b)]

Transport: this commits sim_fairs.json to THIS repo and pushes. The board
(golf_scraping) fetches it from GitHub with the SIMS_PROCESS_PAT it already has.
(Cross-account Cloudflare R2 is not available — the sim and board live in
different CF accounts — so the repo is the transport.) The board joins on the
normalized player name and computes EV = sim_prob * decimal_odds - 1. Round
matchups / 3-balls / props are round-based and not produced by the tournament
sim, so the board keeps a market-consensus fair for those.

Usage:
    python publish_sim_fairs.py              # build + write + commit + push
    python publish_sim_fairs.py --dry-run    # build + print summary, no write
    python publish_sim_fairs.py --no-push    # build + write local file only

Hook: new_sim.py calls publish_sim_fairs.publish() at the end of a run
(wrapped in try/except so a publish failure never breaks the sim).
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from r1_prediction_artifact import (
    build_r1_prediction_manifest,
    manifest_path_for,
    validate_r1_prediction_frame,
)

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
LOCAL_OUT = PROJECT_ROOT / "sim_fairs.json"
LOCAL_SAMPLES = PROJECT_ROOT / "round_samples.parquet"
ROUND_SAMPLE_N = 2000  # downsampled sims kept for board-side 3-ball/round-matchup pricing

# Per-draw tournament FINISH tape (final_scores downsample) for the board's
# correlated-Kelly portfolio optimizer — the EXACT tournament joint (winner/top-N/
# H2H), replacing the board's Gaussian-copula approximation. Committed to this repo
# and fetched by the board from GitHub, exactly like round_samples.parquet — same
# git transport, no per-machine Cloudflare creds needed.
LOCAL_TOURN_SAMPLES = PROJECT_ROOT / "tournament_samples.parquet"
# Draws in the GIT tape (fallback + client book-re-solve source). The FULL tape (all
# draws) is uploaded as a GitHub RELEASE ASSET (not git history, not R2) for build.py's
# server-side full-resolution solve — no repo bloat, no R2/cross-account, and it uses
# the GH_TOKEN this repo already has (board downloads with its SIMS_PROCESS_PAT).
TOURN_SAMPLE_N = 20000
GH_REPO = "mslade50/sims_process"
FULL_TAPE_TAG = "sim-data"
FULL_TAPE_ASSET = "tournament_samples_full.parquet"
MATCHUP_TAPE_ASSET = "matchup_scores_live.parquet"
MATCHUP_TAPE_DRAWS = 25000  # H2H prob SE ~0.3pp at 25k draws vs the maker's 5pp gate
MADE_CUT_ASSET = "tournament_made_cut_full.parquet"


def sync_r1_prediction_artifact(source=None, destination=None, payload=None):
    """Persist the complete R1 skill/weather artifact beside dashboard data.

    When the current sim payload is available, require the CSV to cover every
    active player and write an event-stamped field manifest beside it.  The CSV
    and manifest are then included in the same atomic git publish as sim_fairs.
    """
    source = Path(source or (PROJECT_ROOT / "model_predictions_r1.csv"))
    destination = Path(
        destination
        or (PROJECT_ROOT / "dashboard_data" / "model_predictions_r1.csv")
    )
    if not source.is_file():
        return None
    try:
        frame = pd.read_csv(source)
        active_field = payload.get("field", []) if payload else None
        validate_r1_prediction_frame(frame, active_players=active_field)
    except ValueError as exc:
        if payload is not None:
            raise ValueError(f"R1 prediction snapshot cannot be published: {exc}") from exc
        logger.warning(f"R1 prediction snapshot skipped: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"R1 prediction snapshot skipped: {exc}")
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if payload is not None:
        manifest = build_r1_prediction_manifest(frame, payload)
        manifest_path_for(destination).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    logger.info(
        f"Synced {len(frame)}-player R1 prediction artifact to {destination}"
    )
    return destination


# ─── config from sim_inputs ───────────────────────────────────────────────────

def _sim_inputs():
    sys.path.insert(0, str(PROJECT_ROOT))
    import sim_inputs
    return sim_inputs


def _name_replacements() -> dict:
    try:
        return getattr(_sim_inputs(), "name_replacements", {}) or {}
    except Exception:
        return {}


def _norm(name: str, repl: dict) -> str:
    """Sim canonical name: lowercase + apply name_replacements. The board applies
    its own pkey() (which also reorders "last, first" -> "first last"), so we only
    need the sim's own spelling fixes here."""
    name = str(name).strip().lower()
    return repl.get(name, name)


def _find(*candidates) -> Path | None:
    """First existing path among root- and tourney-folder candidates."""
    for c in candidates:
        p = PROJECT_ROOT / c
        if p.exists():
            return p
    return None


def _tournament_score_paths(tourney: str, *, use_live: bool = False):
    """Return one internally-paired tournament score tape and name sidecar.

    A live payload must never be paired with the pre-event ``final_scores`` tape:
    the row order differs and, more importantly, the pre-event outcomes do not
    contain completed-round scores.  Callers requesting live data therefore fail
    closed when either half of the live pair is absent instead of falling back.
    """
    if use_live:
        fs = _find(
            f"final_scores_live_{tourney}.npy",
            f"{tourney}/final_scores_live_{tourney}.npy",
        )
        pn = _find(
            f"player_names_live_{tourney}.json",
            f"{tourney}/player_names_live_{tourney}.json",
        )
        source = "final_scores_live"
    else:
        fs = _find(f"{tourney}/final_scores.npy", f"final_scores_{tourney}.npy")
        pn = _find(f"{tourney}/player_names.json", f"player_names_{tourney}.json")
        source = "final_scores"
    if fs is None or pn is None:
        return None, None, source
    return fs, pn, source


def _made_cut_path(tourney: str, *, use_live: bool = False) -> Path | None:
    if use_live:
        return _find(
            f"made_cut_live_{tourney}.npy",
            f"{tourney}/made_cut_live_{tourney}.npy",
        )
    return _find(f"{tourney}/made_cut.npy", f"made_cut_{tourney}.npy")


ROUND_CACHE_MAX_AGE_DAYS = 10


def _find_fresh(*candidates, max_age_days: int = ROUND_CACHE_MAX_AGE_DAYS) -> Path | None:
    """_find restricted to RECENTLY-written files, for round-scoped sim caches.

    Tourney dirs persist on this machine across YEARS, and recurring slugs
    (the_open, us_open, ...) mean a bare existence check happily revives LAST
    year's round_score_probs_r4/sim_cache_r4 and publishes them stamped with
    THIS year's event_id — the tourney-stamp guard compares an identical
    year-over-year string and is defeated. A round cache older than
    ~ROUND_CACHE_MAX_AGE_DAYS cannot belong to the current event; skip it."""
    import time
    cutoff = time.time() - max_age_days * 86400
    for c in candidates:
        p = PROJECT_ROOT / c
        try:
            if p.exists():
                if p.stat().st_mtime >= cutoff:
                    return p
                logger.warning(f"ignoring stale round cache {c} "
                               f"(>{max_age_days}d old — prior event/year?)")
        except OSError:
            pass
    return None


def _utc_stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_utc(ts) -> datetime | None:
    try:
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _alert(text: str) -> None:
    """Best-effort Telegram (maker_alerts env). Publish failures must be LOUD —
    a silent skipped/rejected push is how Monday's Render deploy got stranded."""
    try:
        from maker_alerts import send_telegram
        send_telegram(f"[publish_sim_fairs] {text}")
    except Exception as e:
        logger.warning(f"telegram alert failed ({e})")


def _sim_run_at(tourney: str, rnd) -> str | None:
    """When the SIM actually ran — the max mtime of the source files build_payload
    reads — NOT the publish wall-clock. `generated_at` is stamped at publish time,
    so ANY machine that re-publishes old artifacts looks 'fresh' on it, defeating
    the maker's 48h cap, prefer-fresher, and the board banner at once. Consumers
    (maker_guard.check_fairs_fresh_payload) gate on THIS stamp and fail closed
    when it's missing. None if no source file exists (caller must refuse to
    publish rather than stamp fiction)."""
    cands = [
        "simulated_probs_live.csv", "simulated_probs.csv",
        f"{tourney}/finish_equity_{tourney}.csv", f"finish_equity_{tourney}.csv",
        f"top_finish_probs_{tourney}.csv", f"{tourney}/top_finish_probs_{tourney}.csv",
        f"{tourney}/finish_equity_live_{tourney}.csv", f"finish_equity_live_{tourney}.csv",
        f"rank_probs_live_{tourney}.parquet", f"{tourney}/rank_probs_live_{tourney}.parquet",
        f"rank_probs_updated_{tourney}.parquet", f"{tourney}/rank_probs_updated_{tourney}.parquet",
        f"make_cut_probs_{tourney}.csv", f"{tourney}/make_cut_probs_{tourney}.csv",
        f"h2h_matrix_{tourney}.parquet", f"{tourney}/h2h_matrix_{tourney}.parquet",
        f"{tourney}/final_scores.npy", f"final_scores_{tourney}.npy",
        f"final_scores_live_{tourney}.npy", f"{tourney}/final_scores_live_{tourney}.npy",
    ]
    if rnd:
        cands += [
            f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet",
            f"{tourney}/round_score_probs_r{rnd}.parquet", f"round_score_probs_r{rnd}.parquet",
        ]
    mts = [(PROJECT_ROOT / c).stat().st_mtime for c in cands if (PROJECT_ROOT / c).exists()]
    return _utc_stamp(max(mts)) if mts else None


def _live_dump_is_current(tourney: str, rnd, repl: dict) -> bool:
    """Whether the generic-named simulated_probs_live.csv belongs to THIS event.

    simulated_probs_live.csv (round_sim's full-field live dump) carries no tourney
    slug, so a leftover from a PRIOR event lingers locally — the GitHub cleanup only
    prunes the runner, not your machine — and, being the highest-priority source in
    _build_outrights/_build_field, silently poisons the board's field + outright fairs
    with last week's data (e.g. Travelers' 72-player field shipping for Deere).

    Trust it only when (a) we're in a live round AND (b) its player set matches this
    week's field (pre_course_fit_{tourney}.csv). Pre-event builds (rnd falsy) always
    ignore it and fall back to this week's simulated_probs.csv.
    """
    live = PROJECT_ROOT / "simulated_probs_live.csv"
    if not live.exists():
        return False
    if not rnd or rnd < 2:
        # R1 round files are built PRE-event (Thursday's round is priced
        # Tue/Wed), so their existence must NEVER flip the publish into live
        # mode. A legitimate live outright dump only exists once the R2 files
        # do (i.e. R1 is actually underway/complete).
        logger.info(f"live dump: round={rnd or 'none'} build (pre-event; R1 files "
                    "don't mean live) -> ignoring simulated_probs_live.csv "
                    "(using simulated_probs.csv)")
        return False
    # Staleness by construction: during an event, round_sim writes the live dump
    # AFTER the pre-event new_sim wrote simulated_probs.csv. A live dump OLDER
    # than this week's full sim is therefore a leftover from a PRIOR event and
    # must never outrank it. This catches what the player-overlap check below
    # can't: consecutive events sharing most of the field (Scottish Open R4
    # leftovers passed 50% overlap vs The Open and shipped MacIntyre at 20% to
    # win The Open, 2026-07-15).
    full = _find("simulated_probs.csv")
    if full is not None and live.stat().st_mtime < full.stat().st_mtime:
        logger.warning("live dump: simulated_probs_live.csv predates this week's "
                       "simulated_probs.csv -> treating as STALE leftover from a "
                       "prior event, ignoring (using simulated_probs.csv)")
        return False
    pcf = _find(f"pre_course_fit_{tourney}.csv", f"{tourney}/pre_course_fit_{tourney}.csv")
    if pcf is None:
        return True  # no field to validate against; trust it mid-event
    try:
        cur = {_norm(n, repl) for n in pd.read_csv(pcf)["player_name"].dropna()}
        live_names = {_norm(n, repl) for n in pd.read_csv(live)["player_name"].dropna()}
    except Exception:
        return True
    if not cur or not live_names:
        return True
    overlap = len(cur & live_names) / len(live_names)
    if overlap < 0.5:
        logger.warning(f"live dump: only {overlap:.0%} of simulated_probs_live.csv players "
                       f"are in this week's field -> treating as STALE, ignoring "
                       f"(using simulated_probs.csv)")
        return False
    return True


def _outrights_run_at(tourney: str, *, use_live: bool) -> str | None:
    """Timestamp the probability source for outrights, independent of round data.

    ``sim_run_at`` is intentionally an aggregate freshness stamp because one payload
    also carries round markets.  It cannot describe carried-forward outrights: a new
    round cache would otherwise make an old outright table appear freshly simulated.
    """
    if use_live:
        source = _find("simulated_probs_live.csv")
    else:
        source = _find(
            "simulated_probs.csv",
            f"{tourney}/finish_equity_{tourney}.csv",
            f"finish_equity_{tourney}.csv",
            f"top_finish_probs_{tourney}.csv",
            f"{tourney}/top_finish_probs_{tourney}.csv",
            f"{tourney}/finish_equity_live_{tourney}.csv",
            f"finish_equity_live_{tourney}.csv",
        )
    return _utc_stamp(source.stat().st_mtime) if source is not None else None


def _matchups_provenance(tourney: str, *, use_live: bool) -> tuple[str, str | None]:
    """Return (source, sim timestamp) for tournament-long H2H probabilities."""
    if use_live:
        fs, pn, source = _tournament_score_paths(tourney, use_live=True)
        if fs is not None and pn is not None:
            return source, _utc_stamp(fs.stat().st_mtime)
    h2h = _find(
        f"h2h_matrix_{tourney}.parquet",
        f"{tourney}/h2h_matrix_{tourney}.parquet",
    )
    return "h2h_matrix", (_utc_stamp(h2h.stat().st_mtime) if h2h is not None else None)


def _payload_market_time(payload: dict, market: str) -> datetime | None:
    """Read a per-market stamp, falling back for legacy payloads."""
    return _parse_utc(
        payload.get(f"{market}_sim_run_at") or payload.get("sim_run_at")
    )


# ─── builders ─────────────────────────────────────────────────────────────────

def _build_outrights(tourney: str, cut_line: int, repl: dict, use_live: bool = True) -> tuple[dict, dict]:
    """winner/top_5/top_10/top_20 for the FULL field; make_cut derived from the
    rank-probability distribution (P(finish rank <= cut)).

    Returns (outrights, outrights_nodh). `outrights` carries the dead-heat-adjusted
    top-N probs (traditional sportsbooks reduce a top-N payout when players tie on
    the cut line). `outrights_nodh` carries the no-dead-heat top-N probs from the
    sim's `*_nodh` columns — the raw P(finish position <= N), ties counted as inside.
    Books that settle a top-N as a clean binary (Kalshi, NoVig) pay on the no-dead-heat
    outcome, so the board grades THEM against this dict. winner/make_cut are
    dead-heat-agnostic and live only in `outrights`.

    Source priority is full-field FIRST. simulated_probs_live.csv is round_sim's raw
    finish_probs dump (every simmed player x win/top-5/10/20, dead-heat AND _nodh), so
    it gives the board a fair for EVERY player. finish_equity_live_*.csv is an
    edge-filtered BETTING file (book columns; as little as 1 row late in an event), so
    it is read only LAST and can never overwrite a full-field value. (It was once the
    primary source here, which shipped the board ~1 outright player — the sparse-equity
    bug.)"""
    out = {"winner": {}, "top_5": {}, "top_10": {}, "top_20": {}, "make_cut": {}}
    out_nodh = {"top_5": {}, "top_10": {}, "top_20": {}}
    col = {"winner": "simulated_win_prob", "top_5": "top_5",
           "top_10": "top_10", "top_20": "top_20"}
    col_nodh = {"top_5": "top_5_nodh", "top_10": "top_10_nodh", "top_20": "top_20_nodh"}

    def _ingest(path, colmap, target):
        """Fill `target` from a CSV. First writer per (market, player) wins (setdefault),
        so a higher-priority source is never clobbered by a later, sparser one. A file
        without the _nodh columns simply contributes nothing to the nodh dict (the
        board falls back to the dead-heat fair per-player when a _nodh prob is absent)."""
        df = pd.read_csv(path)
        for mkt, c in colmap.items():
            if c not in df.columns:
                continue
            for _, r in df.iterrows():
                p = r[c]
                if pd.notna(p) and p > 0:
                    target[mkt].setdefault(_norm(r["player_name"], repl), round(float(p), 5))

    # Full-field probabilities first (every player), sparse betting file last. Each
    # source is ingested into both the dead-heat (out) and no-dead-heat (out_nodh) dicts.
    # When use_live is False (pre-event, or a stale live dump from a prior event), skip
    # the generic-named simulated_probs_live.csv so it can't outrank this week's
    # simulated_probs.csv. See _live_dump_is_current().
    full = (_find("simulated_probs_live.csv", "simulated_probs.csv")
            if use_live else _find("simulated_probs.csv"))
    # When the current LIVE full-field dump is the primary source, its player set
    # IS the remaining field (post-cut it holds only survivors). Remember it so
    # the pre-event layers below can backfill probs for players the live sim
    # covered sparsely, but can never RESURRECT players who are out of the
    # event — the pre-event-refill bug that ships winner mass > 1 (see
    # _check_outright_mass). Guard on a sane row count so a truncated dump
    # can't nuke the field.
    live_field = None
    if full is not None and use_live and full.name == "simulated_probs_live.csv":
        try:
            _live_names = pd.read_csv(full)["player_name"]
            if len(_live_names) >= 20:
                live_field = {_norm(n, repl) for n in _live_names}
        except Exception:
            live_field = None
    if full is not None:
        _ingest(full, col, out)
        _ingest(full, col_nodh, out_nodh)
    pre = _find(f"{tourney}/finish_equity_{tourney}.csv", f"finish_equity_{tourney}.csv")
    if pre is not None:
        _ingest(pre, col, out)
        _ingest(pre, col_nodh, out_nodh)
    tf = _find(f"top_finish_probs_{tourney}.csv", f"{tourney}/top_finish_probs_{tourney}.csv")
    if tf is not None:
        _ingest(tf, {"top_5": "top_5", "top_10": "top_10", "top_20": "top_20"}, out)
        _ingest(tf, col_nodh, out_nodh)
    live = _find(f"{tourney}/finish_equity_live_{tourney}.csv", f"finish_equity_live_{tourney}.csv")
    if live is not None:
        _ingest(live, col, out)
        _ingest(live, col_nodh, out_nodh)

    # No-dead-heat top-N fallback. Pre-event the tournament-sim CSVs above carry no
    # *_nodh columns, so out_nodh is empty and exchange books (Kalshi/NoVig) — which
    # settle a top-N as a clean binary — would be graded on dead-heat fairs. Derive the
    # no-dead-heat top-N from the rank-prob distribution (prob_ndh = raw min-rank, ties
    # counted inside = no dead heat), the same source/column rule make_cut uses below.
    # setdefault so a real *_nodh from a live CSV (round_sim) is never overwritten.
    if any(not out_nodh[m] for m in ("top_5", "top_10", "top_20")):
        rk_ndh = _find(f"rank_probs_live_{tourney}.parquet", f"{tourney}/rank_probs_live_{tourney}.parquet",
                       f"rank_probs_updated_{tourney}.parquet", f"{tourney}/rank_probs_updated_{tourney}.parquet")
        if rk_ndh is not None:
            rp_ndh = pd.read_parquet(rk_ndh)
            ndh_col = "prob_ndh" if "prob_ndh" in rp_ndh.columns else (
                "prob_u" if "prob_u" in rp_ndh.columns else None)
            if ndh_col and {"player_name", "rank"} <= set(rp_ndh.columns):
                for _n, _mkt in ((5, "top_5"), (10, "top_10"), (20, "top_20")):
                    s = rp_ndh[rp_ndh["rank"] <= _n].groupby("player_name")[ndh_col].sum()
                    for nm, p in s.items():
                        if p > 0:
                            out_nodh[_mkt].setdefault(_norm(nm, repl), round(float(min(p, 1.0)), 5))
                logger.info(f"outrights_nodh top-N derived from {rk_ndh.name} [{ndh_col}]")

    # Drop pre-event refills for players outside the live field (cut/WD).
    # setdefault stops later layers overwriting live values but not ADDING
    # eliminated players the live dump (rightly) no longer carries.
    if live_field is not None:
        dropped = 0
        for target in (out, out_nodh):
            for mkt in ("winner", "top_5", "top_10", "top_20"):
                if mkt in target:
                    before = len(target[mkt])
                    target[mkt] = {k: v for k, v in target[mkt].items() if k in live_field}
                    dropped += before - len(target[mkt])
        if dropped:
            logger.info(f"outrights: dropped {dropped} pre-event refill entries for "
                        f"players outside the live field ({len(live_field)} remaining)")

    logger.info("outrights: " + ", ".join(f"{k}={len(v)}" for k, v in out.items() if k != "make_cut"))
    logger.info("outrights_nodh: " + ", ".join(f"{k}={len(v)}" for k, v in out_nodh.items()))

    # make_cut: a LIVE outright family must use the made-cut mask from the SAME
    # remaining-tournament simulation as final_scores_live. The old priority
    # order selected make_cut_probs_{tourney}.csv first, which is pre-event and
    # could silently mix a fresh live winner/top-N family with week-start cut
    # probabilities. Never fall back across that provenance boundary.
    if use_live:
        import numpy as np

        mc_path = _made_cut_path(tourney, use_live=True)
        fs_path, pn_path, _ = _tournament_score_paths(tourney, use_live=True)
        if mc_path is None or fs_path is None or pn_path is None:
            logger.warning(
                "make_cut live pair missing — refusing pre-event make-cut fallback"
            )
        else:
            mask = np.load(mc_path, mmap_mode="r")
            fs_shape = np.load(fs_path, mmap_mode="r").shape
            names = json.loads(Path(pn_path).read_text(encoding="utf-8"))
            if mask.ndim != 2 or mask.shape != fs_shape or mask.shape[0] != len(names):
                logger.warning(
                    f"make_cut live pair mismatch: mask={mask.shape}, "
                    f"final_scores={fs_shape}, names={len(names)} — refusing fallback"
                )
            else:
                mask_values = np.asarray(mask)
                probs = mask_values.astype(float).mean(axis=1)
                if (not np.isfinite(mask_values).all()
                        or not np.isin(mask_values, (0, 1)).all()
                        or not np.isfinite(probs).all()
                        or ((probs < 0) | (probs > 1)).any()):
                    logger.warning(
                        "make_cut live mask produced invalid probabilities — refusing fallback"
                    )
                else:
                    for nm, p in zip(names, probs):
                        if p > 0:
                            out["make_cut"][_norm(nm, repl)] = round(float(p), 5)
                    logger.info(
                        f"make_cut from {mc_path.name} (paired live mask): "
                        f"{len(out['make_cut'])}"
                    )
    else:
        # Pre-event: prefer the exact cut simulation persisted by new_sim.py
        # (true cut: top-N + ties + 10-shot rule). Fall back to a rank-prob
        # estimate using raw min-rank probability when the exact file is absent.
        mc_file = _find(
            f"make_cut_probs_{tourney}.csv",
            f"{tourney}/make_cut_probs_{tourney}.csv",
        )
        if mc_file is not None:
            df = pd.read_csv(mc_file)
            for _, r in df.iterrows():
                p = r["make_cut"]
                if pd.notna(p) and p > 0:
                    out["make_cut"][_norm(r["player_name"], repl)] = round(
                        float(min(p, 1.0)), 5
                    )
            logger.info(
                f"make_cut from {mc_file.name} (exact sim cut): "
                f"{len(out['make_cut'])}"
            )
        else:
            rk = _find(f"rank_probs_live_{tourney}.parquet", f"{tourney}/rank_probs_live_{tourney}.parquet",
                       f"rank_probs_updated_{tourney}.parquet", f"{tourney}/rank_probs_updated_{tourney}.parquet")
            if rk is not None:
                rp = pd.read_parquet(rk)
                col = "prob_ndh" if "prob_ndh" in rp.columns else "prob_u"
                if {"player_name", "rank"} <= set(rp.columns) and col in rp.columns:
                    mc = rp[rp["rank"] <= cut_line].groupby("player_name")[col].sum()
                    for nm, p in mc.items():
                        if p > 0:
                            out["make_cut"][_norm(nm, repl)] = round(float(min(p, 1.0)), 5)
                    logger.info(f"make_cut from {rk.name} [{col}] (cut<={cut_line}): {len(out['make_cut'])}")

    return ({k: v for k, v in out.items() if v},
            {k: v for k, v in out_nodh.items() if v})


def _build_matchups_live(tourney: str, repl: dict) -> list:
    """Tournament H2H pairs from round_sim's LIVE tournament joint
    (final_scores_live npy + name sidecar) — ties pushed, same convention as
    new_sim's h2h_matrix (prob_a = wins / (wins + losses)). [] if the live
    files aren't on disk (pre-event / machine that never ran round_sim)."""
    import numpy as np
    fs = _find(f"final_scores_live_{tourney}.npy", f"{tourney}/final_scores_live_{tourney}.npy")
    pn = _find(f"player_names_live_{tourney}.json", f"{tourney}/player_names_live_{tourney}.json")
    if fs is None or pn is None:
        return []
    scores = np.load(fs)
    names = json.loads(Path(pn).read_text())
    if scores.ndim != 2 or scores.shape[0] != len(names):
        logger.warning(f"matchups (live): final_scores {scores.shape} vs {len(names)} "
                       f"names mismatch — falling back to h2h_matrix")
        return []
    first = {}
    for i, nm in enumerate(_norm(n, repl) for n in names):
        first.setdefault(nm, i)
    order = sorted(first)
    S = scores[[first[nm] for nm in order]]
    rows = []
    for i in range(len(order) - 1):
        rest = S[i + 1:]
        wins = (S[i] < rest).sum(axis=1)
        losses = (S[i] > rest).sum(axis=1)
        for j, (w, l) in enumerate(zip(wins, losses), start=i + 1):
            if w + l == 0:
                continue
            rows.append([order[i], order[j], round(float(w) / float(w + l), 5)])
    logger.info(f"tournament matchups (live, ties pushed): {len(rows)} pairs "
                f"from {Path(fs).name} ({S.shape[1]:,} draws)")
    return rows


def _build_matchups(tourney: str, repl: dict, *, use_live: bool = True) -> list:
    """Tournament head-to-head: [player_a, player_b, P(a beats b)] (ties pushed).

    Live-first: once round_sim has run, final_scores_live IS the current
    tournament joint, so pairwise fairs from it track the live event. The
    pre-event h2h_matrix_{tourney}.parquet (new_sim.py) is the fallback for
    machines/weeks with no live sim yet. Without the live build, a machine
    lacking the h2h matrix published matchups=[] every live run and the shrink
    guard carried origin's PRE-EVENT full-field pairs forward all weekend
    (The Open 2026: 12,090 stale pairs shipping beside fresh live outrights)."""
    rows = _build_matchups_live(tourney, repl) if use_live else []
    if rows:
        return rows
    h = _find(f"h2h_matrix_{tourney}.parquet", f"{tourney}/h2h_matrix_{tourney}.parquet")
    if h is None:
        return []
    df = pd.read_parquet(h)
    rows = []
    for _, r in df.iterrows():
        pa = r.get("prob_a")
        if pd.isna(pa):
            continue
        rows.append([_norm(r["player_a"], repl), _norm(r["player_b"], repl), round(float(pa), 5)])
    logger.info(f"tournament matchups (h2h): {len(rows)} pairs")
    return rows


def _build_field(tourney: str, repl: dict, use_live: bool = True) -> list:
    """The full simulated roster (every player, NO p>0 filter) so the board can
    drop players that aren't in our sim without relying on probabilistic markets
    (make_cut/top-N) to enumerate the field."""
    # Full-field dump first; the edge-filtered finish_equity_live (as little as 1 row)
    # must NOT define the field or the board drops all but a handful of players.
    # When use_live is False, the generic simulated_probs_live.csv is skipped so a
    # leftover from a prior event can't define this week's field (cross-event bleed).
    candidates = (["simulated_probs_live.csv"] if use_live else []) + [
        "simulated_probs.csv",
        f"{tourney}/finish_equity_{tourney}.csv", f"finish_equity_{tourney}.csv",
        f"{tourney}/finish_equity_live_{tourney}.csv", f"finish_equity_live_{tourney}.csv"]
    eq = _find(*candidates)
    if eq is None:
        return []
    df = pd.read_csv(eq)
    if "player_name" not in df.columns:
        return []
    return sorted({_norm(n, repl) for n in df["player_name"].dropna() if str(n).strip()})


def _resolve_event_name(event_id, tour, tourney) -> str:
    """Human event name embedded in sim_fairs.json so the board can scope by name
    when DataGolf is unavailable at board-build time. Resolved here via the
    DataGolf schedule (reliable in the sim's env); falls back to a humanized slug.
    event_id is unique only within a tour, so prefer the sim's tour."""
    slug = (tourney or "").replace("_", " ").title()
    if not event_id:
        return slug
    try:
        import os
        import requests
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
        except Exception:
            pass
        key = os.getenv("DATAGOLF_API_KEY")
        if not key:
            return slug
        r = requests.get("https://feeds.datagolf.com/get-schedule",
                         params={"tour": "all", "file_format": "json", "key": key}, timeout=15)
        r.raise_for_status()
        cand = [e for e in (r.json().get("schedule") or [])
                if str(e.get("event_id")) == str(event_id)]
        for e in cand:
            if str(e.get("tour", "")).lower() == str(tour).lower() and e.get("event_name"):
                return e["event_name"].strip()
        if cand and cand[0].get("event_name"):
            return cand[0]["event_name"].strip()
    except Exception as e:
        logger.warning(f"event-name resolve failed ({e}); using slug '{slug}'")
    return slug


def _latest_round(tourney: str):
    """Highest round N for which the sim has produced round-level data."""
    import glob
    import re
    import os
    import time
    cutoff = time.time() - ROUND_CACHE_MAX_AGE_DAYS * 86400
    best = None
    for pat in (f"{tourney}/round_score_probs_r*.parquet", "round_score_probs_r*.parquet",
                f"{tourney}/sim_cache_r*.parquet", "sim_cache_r*.parquet"):
        for p in glob.glob(str(PROJECT_ROOT / pat)):
            m = re.search(r"_r(\d+)\.parquet$", p)
            if not m:
                continue
            try:
                if os.path.getmtime(p) < cutoff:
                    # a recurring slug (the_open) revives LAST year's r4 cache and
                    # pins the round pointer at 4 all week — stale files don't vote
                    continue
            except OSError:
                continue
            best = max(best or 0, int(m.group(1)))
    return best


def _build_round_scores(tourney: str, rnd, repl: dict) -> dict:
    """Exact per-player round-score PMF for over/under pricing:
    {player: {score: prob}} for round `rnd`. Board computes P(under line) = CDF."""
    if not rnd:
        return {}
    f = _find_fresh(f"{tourney}/round_score_probs_r{rnd}.parquet", f"round_score_probs_r{rnd}.parquet")
    if f is None:
        return {}
    df = pd.read_parquet(f)
    if not {"player_name", "score", "prob"} <= set(df.columns):
        return {}
    out = {}
    for nm, g in df.groupby("player_name"):
        pmf = {int(s): round(float(p), 6) for s, p in zip(g["score"], g["prob"]) if p > 0}
        if pmf:
            out[_norm(nm, repl)] = pmf
    logger.info(f"round_scores (R{rnd}) PMF: {len(out)} players")
    return out


def _build_round_samples(tourney: str, rnd, repl: dict):
    """Downsampled joint per-round score matrix (players x ROUND_SAMPLE_N) from the
    sim cache, so the board can compute EXACT round-matchup + 3-ball fairs (joint
    preserved -> shared wave/weather conditions cancel). Returns a DataFrame or None."""
    if not rnd:
        return None
    f = _find_fresh(f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet")
    if f is None:
        return None
    import numpy as np
    df = pd.read_parquet(f)               # index = player, columns = sim indices
    n = df.shape[1]
    if n > ROUND_SAMPLE_N:                 # fixed-stride downsample keeps the joint
        idx = np.linspace(0, n - 1, ROUND_SAMPLE_N).round().astype(int)
        df = df.iloc[:, idx]
    df = df.astype("int16")
    df.index = [_norm(p, repl) for p in df.index]
    df.columns = [str(i) for i in range(df.shape[1])]
    logger.info(f"round_samples (R{rnd}): {df.shape[0]} players x {df.shape[1]} sims")
    return df


def _build_tournament_samples(
    tourney: str,
    event_id,
    generated_at,
    repl: dict,
    max_draws=TOURN_SAMPLE_N,
    *,
    use_live: bool = False,
):
    """Downsampled per-draw 72-hole FINISH tape (players x TOURN_SAMPLE_N) from the
    tournament sim's `final_scores` — the exact joint the board's portfolio optimizer
    uses for correlated Kelly on outrights + tournament matchups. Values are int16
    72-hole totals (lower = better; missed-cut carries a +200 penalty so MC sinks
    below the field, which is why winner/top-N/H2H are all exactly derivable and
    self-consistent with sim_fairs). Returns a pyarrow.Table stamped with
    event_id/generated_at (the board guards on these), or None if the sim cache
    (final_scores + player_names) isn't on disk."""
    import numpy as np
    import pyarrow as pa

    fs, pn, source = _tournament_score_paths(tourney, use_live=use_live)
    if fs is None or pn is None:
        kind = "live " if use_live else ""
        logger.info(f"tournament_samples: paired {kind}final_scores/player_names cache "
                    "not found — skipping (board keeps the copula)")
        return None
    scores = np.load(fs)                                  # (n_players, n_sims)
    names = json.loads(Path(pn).read_text())
    if scores.ndim != 2 or scores.shape[0] != len(names):
        logger.warning(f"tournament_samples: final_scores {scores.shape} vs {len(names)} names "
                       f"mismatch — skipping")
        return None
    n = scores.shape[1]
    if max_draws and n > max_draws:                      # fixed-stride downsample keeps the joint
        idx = np.linspace(0, n - 1, max_draws).round().astype(int)
        scores = scores[:, idx]
    scores = scores.astype("int16")
    df = pd.DataFrame(scores, index=[_norm(nm, repl) for nm in names],
                      columns=[str(i) for i in range(scores.shape[1])])
    df = df[~df.index.duplicated(keep="first")]          # first spelling wins on any dup
    tbl = pa.Table.from_pandas(df, preserve_index=True)
    meta = {**(tbl.schema.metadata or {}),
            b"event_id": str(event_id).encode(),
            b"generated_at": str(generated_at).encode(),
            b"sim_run_at": _utc_stamp(fs.stat().st_mtime).encode(),
            b"tourney": str(tourney).encode(),
            b"source": source.encode()}
    tbl = tbl.replace_schema_metadata(meta)
    logger.info(f"tournament_samples: {df.shape[0]} players x {df.shape[1]} draws "
                f"(event {event_id})")
    return tbl


def _build_made_cut_mask(
    tourney: str,
    event_id,
    repl: dict,
    max_draws=TOURN_SAMPLE_N,
    *,
    use_live: bool = False,
):
    """Made-cut mask on the SAME draw axis as the tournament finish tape — same
    players (index), same fixed-stride downsample — so the board can price
    make_cut off the joint instead of an independent copula draw (same-golfer
    win+topN+cut stacks were treated as independent = over-staked exactly where
    stacking is heaviest). None when new_sim hasn't written made_cut.npy (old
    Rust wheel, or pre-mask sim run) or when its shape disagrees with
    final_scores (never pair a mask with a tape from a different run)."""
    import numpy as np
    import pyarrow as pa

    mc = _made_cut_path(tourney, use_live=use_live)
    fs, pn, _ = _tournament_score_paths(tourney, use_live=use_live)
    if mc is None or fs is None or pn is None:
        kind = "live " if use_live else ""
        logger.info(f"made_cut mask: paired {kind}made_cut/final_scores/player_names "
                    "not all present — skipping (board keeps the copula for make_cut)")
        return None
    mask = np.load(mc)
    names = json.loads(Path(pn).read_text())
    fs_shape = np.load(fs, mmap_mode="r").shape
    if mask.ndim != 2 or mask.shape != fs_shape or mask.shape[0] != len(names):
        logger.warning(f"made_cut mask: shape {mask.shape} disagrees with final_scores "
                       f"{fs_shape} / {len(names)} names — skipping")
        return None
    n = mask.shape[1]
    if max_draws and n > max_draws:
        idx = np.linspace(0, n - 1, max_draws).round().astype(int)  # SAME stride as the tape
        mask = mask[:, idx]
    df = pd.DataFrame(mask.astype("int8"), index=[_norm(nm, repl) for nm in names],
                      columns=[str(i) for i in range(mask.shape[1])])
    df = df[~df.index.duplicated(keep="first")]
    tbl = pa.Table.from_pandas(df, preserve_index=True)
    meta = {**(tbl.schema.metadata or {}),
            b"event_id": str(event_id).encode(),
            b"sim_run_at": _utc_stamp(fs.stat().st_mtime).encode(),
            b"tourney": str(tourney).encode(),
            b"source": (b"made_cut_live" if use_live else b"made_cut")}
    tbl = tbl.replace_schema_metadata(meta)
    logger.info(f"made_cut mask: {df.shape[0]} players x {df.shape[1]} draws (event {event_id})")
    return tbl


def _upload_full_tape_release(
    tourney, event_id, generated_at, repl, *, use_live: bool = False
) -> bool:
    """Build the FULL-resolution finish tape (ALL sim draws, no downsample) and upload
    it as a GitHub RELEASE ASSET on this repo (tag `sim-data`), using GH_TOKEN. The
    board downloads it with its SIMS_PROCESS_PAT for the server-side full-resolution
    solve — no R2, no cross-account creds, no per-machine setup, no git-history bloat
    (release assets are stored separately). No-op (warns) if GH_TOKEN / requests /
    final_scores are missing."""
    import os
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("full tape: GH_TOKEN not set — skipping release upload (board uses git tape)")
        return False
    tbl = _build_tournament_samples(
        tourney,
        event_id,
        generated_at,
        repl,
        max_draws=None,
        use_live=use_live,
    )
    if tbl is None:
        return False
    try:
        import io
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        pq.write_table(tbl, buf)
        data = buf.getvalue()
        _upload_release_asset(FULL_TAPE_ASSET, data, token)
        logger.info(f"full tape: uploaded {tbl.num_rows} players x {tbl.num_columns - 1} draws "
                    f"-> release {FULL_TAPE_TAG}/{FULL_TAPE_ASSET} ({len(data) // 1_000_000}MB)")
        return True
    except Exception as e:
        logger.warning(f"full tape: release upload failed ({e})")
        return False


def _upload_release_asset(asset_name: str, data: bytes, token: str) -> None:
    """Replace one asset on this repo's `sim-data` release (created if absent).
    Delete-then-upload — asset names must be unique on a release."""
    import requests
    api = "https://api.github.com"
    H = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    r = requests.get(f"{api}/repos/{GH_REPO}/releases/tags/{FULL_TAPE_TAG}", headers=H, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{api}/repos/{GH_REPO}/releases", headers=H, timeout=30, json={
            "tag_name": FULL_TAPE_TAG, "name": "Sim data (board tapes)", "prerelease": True,
            "body": "Sim tapes for the odds-board portfolio solve and the Kalshi maker."})
    r.raise_for_status()
    rel = r.json()
    for a in rel.get("assets", []):
        if a.get("name") == asset_name:
            requests.delete(f"{api}/repos/{GH_REPO}/releases/assets/{a['id']}", headers=H, timeout=30)
    up = f"https://uploads.github.com/repos/{GH_REPO}/releases/{rel['id']}/assets?name={asset_name}"
    ur = requests.post(up, headers={**H, "Content-Type": "application/octet-stream"}, data=data, timeout=600)
    ur.raise_for_status()


def _build_live_matchup_tape(tourney: str, event_id, repl: dict, max_draws=MATCHUP_TAPE_DRAWS):
    """Per-draw LIVE score tape (players x draws) from round_sim's
    final_scores_live npy + its OWN name sidecar. The live field order differs
    from new_sim's alphabetical player_names.json — pairing across runs computes
    P(wrong player beats wrong player) — so the names ride inside the parquet as
    the index. Feeds kalshi_maker H2H pricing on machines that didn't run the
    sim (VPS). None if the live files aren't on disk."""
    import numpy as np
    import pyarrow as pa

    fs = _find(f"final_scores_live_{tourney}.npy", f"{tourney}/final_scores_live_{tourney}.npy")
    pn = _find(f"player_names_live_{tourney}.json", f"{tourney}/player_names_live_{tourney}.json")
    if fs is None or pn is None:
        logger.info("matchup tape: no live final_scores/player_names — skipping")
        return None
    scores = np.load(fs)
    names = json.loads(Path(pn).read_text())
    if scores.ndim != 2 or scores.shape[0] != len(names):
        logger.warning(f"matchup tape: final_scores {scores.shape} vs {len(names)} names "
                       f"mismatch — skipping")
        return None
    n = scores.shape[1]
    if max_draws and n > max_draws:                      # fixed-stride downsample keeps the joint
        idx = np.linspace(0, n - 1, max_draws).round().astype(int)
        scores = scores[:, idx]
    # H2H prices depend on exact ties — only compress to int16 when lossless
    if np.allclose(scores, np.round(scores)):
        scores = np.round(scores).astype("int16")
    else:
        scores = scores.astype("float32")
    df = pd.DataFrame(scores, index=[_norm(nm, repl) for nm in names],
                      columns=[str(i) for i in range(scores.shape[1])])
    df = df[~df.index.duplicated(keep="first")]
    tbl = pa.Table.from_pandas(df, preserve_index=True)
    meta = {**(tbl.schema.metadata or {}),
            b"event_id": str(event_id).encode(),
            b"sim_run_at": _utc_stamp(fs.stat().st_mtime).encode(),
            b"tourney": str(tourney).encode(),
            b"source": b"final_scores_live"}
    tbl = tbl.replace_schema_metadata(meta)
    logger.info(f"matchup tape: {df.shape[0]} players x {df.shape[1]} draws (event {event_id})")
    return tbl


def _upload_matchup_tape_release(tourney, event_id, repl) -> bool:
    """Upload the live matchup tape as a release asset so the maker can price
    H2H matchups on machines that never ran round_sim (final_scores_live is
    tens of MB and gitignored — git transport is the wrong channel)."""
    import os
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("matchup tape: GH_TOKEN not set — skipping release upload "
                       "(maker H2H needs a local round_sim run)")
        return False
    tbl = _build_live_matchup_tape(tourney, event_id, repl)
    if tbl is None:
        return False
    try:
        import io
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        pq.write_table(tbl, buf)
        data = buf.getvalue()
        _upload_release_asset(MATCHUP_TAPE_ASSET, data, token)
        logger.info(f"matchup tape: uploaded {tbl.num_rows} players x {tbl.num_columns - 1} "
                    f"draws -> release {FULL_TAPE_TAG}/{MATCHUP_TAPE_ASSET} "
                    f"({len(data) // 1_000_000}MB)")
        return True
    except Exception as e:
        logger.warning(f"matchup tape: release upload failed ({e})")
        return False


def _upload_release_tape_family(payload: dict, repl: dict, *, strict: bool = False) -> bool:
    """Upload the three full-resolution live release assets as one contract.

    GitHub Releases do not offer a multi-asset transaction, so consumers still
    validate the embedded event/sim timestamps and fall back to the atomic git
    tapes during a partial replacement. In strict nightly mode, any failed asset
    aborts before the new git fairs commit/board dispatch can advance.
    """
    import io
    import pyarrow.parquet as pq

    tourney = payload["tourney"]
    event_id = payload.get("event_id")
    use_live = payload.get("outrights_source") == "live"
    failures = []

    def _abort_strict():
        if strict and failures:
            raise RuntimeError(
                "release tape family incomplete: " + "; ".join(failures)
            )

    try:
        if not _upload_full_tape_release(
            tourney,
            event_id,
            payload.get("generated_at"),
            repl,
            use_live=use_live,
        ):
            failures.append("full tournament tape")
    except Exception as exc:
        failures.append(f"full tournament tape ({exc})")
    _abort_strict()

    try:
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        full_mask = (
            _build_made_cut_mask(
                tourney,
                event_id,
                repl,
                max_draws=None,
                use_live=use_live,
            )
            if token
            else None
        )
        if full_mask is None or not token:
            failures.append("full made-cut tape")
        else:
            buf = io.BytesIO()
            pq.write_table(full_mask, buf)
            _upload_release_asset(MADE_CUT_ASSET, buf.getvalue(), token)
            logger.info(
                f"made_cut mask: uploaded -> release {FULL_TAPE_TAG}/{MADE_CUT_ASSET}"
            )
    except Exception as exc:
        failures.append(f"full made-cut tape ({exc})")
    _abort_strict()

    try:
        if not _upload_matchup_tape_release(tourney, event_id, repl):
            failures.append("live matchup tape")
    except Exception as exc:
        failures.append(f"live matchup tape ({exc})")
    _abort_strict()

    if failures:
        message = "release tape family incomplete: " + "; ".join(failures)
        logger.warning(f"{message} (non-fatal; board uses atomic git tapes)")
        return False
    return True


def _cache_meta(tourney: str, rnd) -> dict:
    """Read the sim_cache meta sidecar (carries pred_lookup + wx_lookup)."""
    if not rnd:
        return {}
    f = _find_fresh(f"{tourney}/sim_cache_r{rnd}_meta.json", f"sim_cache_r{rnd}_meta.json")
    if f is None:
        return {}
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _sample_lookup(tourney: str, repl: dict) -> dict:
    """Per-player sample sizes from pre_sim_summary (used by the matchup filter)."""
    f = _find(f"pre_sim_summary_{tourney}.csv", f"{tourney}/pre_sim_summary_{tourney}.csv")
    if f is None:
        return {}
    try:
        df = pd.read_csv(f)
    except Exception:
        return {}
    if not {"player_name", "sample"} <= set(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        if pd.notna(r["player_name"]) and pd.notna(r["sample"]):
            out[_norm(r["player_name"], repl)] = int(r["sample"])
    return out


def _pred_lookup(tourney: str, repl: dict) -> dict:
    """Per-player skill estimate (`pred`, strokes-gained/round vs the field) from
    pre_sim_summary. Shipped in sim_fairs.json so the odds board can filter rows
    by our skill number (e.g. only players > 0)."""
    f = _find(f"pre_sim_summary_{tourney}.csv", f"{tourney}/pre_sim_summary_{tourney}.csv")
    if f is None:
        return {}
    try:
        df = pd.read_csv(f)
    except Exception:
        return {}
    if not {"player_name", "pred"} <= set(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        if pd.notna(r["player_name"]) and pd.notna(r["pred"]):
            out[_norm(r["player_name"], repl)] = round(float(r["pred"]), 3)
    return out


def _build_round_h2h(tourney: str, rnd, repl: dict):
    """All-pairs round head-to-head computed from the FULL sim-cache joint draws.

    Returns (df[player_a, player_b, p_a_lt_b, p_tie], meta) or (None, None).

    The joint is preserved (shared wave/weather conditions cancel), so a stored
    (P(a<b), P(tie)) pair is mathematically conclusive for pricing any H2H — it
    reconstructs every column round_sim.price_matchups produces, EXACTLY, with no
    draws needed at reprice time. Players are stored in canonical (lexicographic)
    order so (a, b) lookups are deterministic; the consumer orients to whichever
    side the book offers.
    """
    if not rnd:
        return None, None
    f = _find_fresh(f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet")
    if f is None:
        return None, None
    import numpy as np

    cache = pd.read_parquet(f)                       # index = player, cols = sim indices
    names = [_norm(p, repl) for p in cache.index]
    order = sorted(range(len(names)), key=lambda i: names[i])
    players = [names[i] for i in order]
    S = cache.values[order].astype(np.int16)         # (n_players, n_sims)
    n, sims = S.shape

    a_col, b_col, lt_col, tie_col = [], [], [], []
    for i in range(n):
        Sj = S[i + 1:]                               # all partners j > i
        if Sj.shape[0] == 0:
            continue
        si = S[i][None, :]
        lt = (si < Sj).sum(axis=1) / sims            # P(player_i < player_j)
        eq = (si == Sj).sum(axis=1) / sims           # P(tie)
        for k in range(Sj.shape[0]):
            a_col.append(players[i])
            b_col.append(players[i + 1 + k])
            lt_col.append(round(float(lt[k]), 5))
            tie_col.append(round(float(eq[k]), 5))

    df = pd.DataFrame({"player_a": a_col, "player_b": b_col,
                       "p_a_lt_b": lt_col, "p_tie": tie_col})
    df["p_a_lt_b"] = df["p_a_lt_b"].astype("float32")
    df["p_tie"] = df["p_tie"].astype("float32")

    cmeta = _cache_meta(tourney, rnd)
    pred = {_norm(k, repl): round(float(v), 4)
            for k, v in (cmeta.get("pred_lookup") or {}).items()}
    wx = {_norm(k, repl): round(float(v), 6)
          for k, v in (cmeta.get("wx_lookup") or {}).items()}
    meta = {
        "tourney": tourney,
        "round": rnd,
        "num_players": n,
        "num_sims": sims,
        "pred": pred,
        "sample": _sample_lookup(tourney, repl),
        "wx": wx,
    }
    logger.info(f"round_h2h (R{rnd}): {len(df)} pairs from {n} players x {sims} sims")
    return df, meta


def write_round_h2h(tourney: str, rnd, repl: dict | None = None) -> list:
    """Build + write round_h2h_r{N}.parquet (+ _meta.json) at repo root. Returns
    the repo-relative file list (empty if no sim cache exists for the round)."""
    repl = repl if repl is not None else _name_replacements()
    df, meta = _build_round_h2h(tourney, rnd, repl)
    if df is None:
        logger.info("round_h2h: no sim cache; skipping")
        return []
    meta["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cache_f = _find_fresh(f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet")
    if cache_f is not None:
        meta["sim_run_at"] = _utc_stamp(cache_f.stat().st_mtime)
    pq = f"round_h2h_r{rnd}.parquet"
    mj = f"round_h2h_r{rnd}_meta.json"
    df.to_parquet(PROJECT_ROOT / pq, index=False)
    with open(PROJECT_ROOT / mj, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"Wrote {pq} + {mj}")
    return [pq, mj]


DATAGOLF_BASE = "https://feeds.datagolf.com"


def _tee_groups(rnd, repl, tour: str = "pga"):
    """R{rnd} tee-time threesomes from DataGolf field-updates, as lists of normalized
    player names. Groups are split by (course, start_hole, teetime) so a split-tee
    start doesn't merge two groups. Only size-3 groups (3-balls) are returned —
    twosomes are round matchups (round_h2h). [] if tee times aren't posted / no key."""
    import os
    import requests
    from collections import defaultdict
    try:
        from dotenv import load_dotenv
        load_dotenv()   # CI sets the env var directly; this is for local runs
    except Exception:
        pass
    key = os.getenv("DATAGOLF_API_KEY")
    if not key:
        logger.info("round_3ball: no DATAGOLF_API_KEY; skipping")
        return []
    try:
        r = requests.get(f"{DATAGOLF_BASE}/field-updates",
                         params={"tour": tour, "file_format": "json", "key": key}, timeout=20)
        if r.status_code != 200:
            return []
        field = (r.json() or {}).get("field") or []
    except Exception as e:
        logger.warning(f"round_3ball tee-time fetch failed: {e!r}")
        return []
    groups = defaultdict(list)
    for p in field:
        tt = next((t for t in (p.get("teetimes") or []) if t.get("round_num") == rnd), None)
        if not tt or not tt.get("teetime"):
            continue
        groups[(tt.get("course_num"), tt.get("start_hole"), tt.get("teetime"))].append(
            _norm(p.get("player_name", ""), repl))
    return [sorted(g) for g in groups.values() if len(g) == 3]


def _nball_fairs(arrs):
    """P(each is the lowest score), ties split evenly, from a list of int sim arrays."""
    import numpy as np
    M = np.vstack(arrs)
    is_min = (M == M.min(axis=0)[None, :])
    return (is_min / is_min.sum(axis=0)).mean(axis=1)   # one prob per row (player)


def _build_round_3balls(tourney: str, rnd, repl: dict, tour: str = "pga"):
    """Exact 3-ball fairs for the ACTUAL R{rnd} tee-time threesomes, from the FULL
    sim cache (all draws) — only the ~groups that exist, not every triple. Returns
    (df[player_a,b,c, p_a,b,c], meta) or (None, None)."""
    if not rnd:
        return None, None
    f = _find_fresh(f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet")
    if f is None:
        return None, None
    groups = _tee_groups(rnd, repl, tour=tour)
    if not groups:
        logger.info(f"round_3ball (R{rnd}): no threesomes (2-ball round or tee times unposted)")
        return None, None
    import numpy as np
    cache = pd.read_parquet(f)
    idx = {_norm(p, repl): p for p in cache.index}
    rows, skipped = [], 0
    for g in groups:
        if not all(n in idx for n in g):
            skipped += 1
            continue
        arrs = [cache.loc[idx[n]].to_numpy().astype(np.int16) for n in g]
        pr = _nball_fairs(arrs)
        rows.append({"player_a": g[0], "player_b": g[1], "player_c": g[2],
                     "p_a": round(float(pr[0]), 5), "p_b": round(float(pr[1]), 5),
                     "p_c": round(float(pr[2]), 5)})
    if not rows:
        if groups and skipped == len(groups):
            logger.warning(f"round_3ball (R{rnd}): ALL {skipped} threesomes skipped - "
                           f"tour mismatch (field-updates tour={tour}) or "
                           f"name-normalization gap vs the sim cache?")
        return None, None
    df = pd.DataFrame(rows)
    meta = {"tourney": tourney, "round": rnd, "num_groups": len(rows),
            "num_sims": int(cache.shape[1]), "skipped": skipped}
    logger.info(f"round_3ball (R{rnd}): {len(rows)} threesomes from {cache.shape[1]} sims "
                f"({skipped} skipped — player not in sim)")
    return df, meta


def write_round_3ball(tourney: str, rnd, repl: dict | None = None,
                      tour: str | None = None) -> list:
    """Build + write round_3ball_r{N}.parquet (+ _meta.json). Returns the repo-relative
    file list (empty if no live threesomes / no sim cache for the round).
    `tour` defaults to sim_inputs.tour so non-PGA weeks fetch the right field
    (a pga-hardcoded fetch made 3-ball parquets silently absent every euro week)."""
    repl = repl if repl is not None else _name_replacements()
    if tour is None:
        tour = getattr(_sim_inputs(), "tour", "pga") or "pga"
    df, meta = _build_round_3balls(tourney, rnd, repl, tour=tour)
    if df is None:
        return []
    meta["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cache_f = _find_fresh(f"{tourney}/sim_cache_r{rnd}.parquet", f"sim_cache_r{rnd}.parquet")
    if cache_f is not None:
        meta["sim_run_at"] = _utc_stamp(cache_f.stat().st_mtime)
    pq = f"round_3ball_r{rnd}.parquet"
    mj = f"round_3ball_r{rnd}_meta.json"
    df.to_parquet(PROJECT_ROOT / pq, index=False)
    with open(PROJECT_ROOT / mj, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"Wrote {pq} + {mj}")
    return [pq, mj]


def _check_outright_mass(outrights: dict, tourney: str) -> None:
    """Refuse to publish an outright book whose probability mass is impossible.

    The pre-event finish_equity/top_finish layers can refill live-zeroed players
    with PRE-event winner probs inside an outrights_source='live' market (~20%
    of the field shipping pre-event probs, mass >1) — a 0.04 'sim' fair against
    +50000 is a massive fake edge that freezes into the closing and gets
    CLV-graded. Bands: winner must sum to ~1; top_N to ~N. make_cut/nodh are
    deliberately unbanded (post-cut they legitimately sum to the survivor
    count). Raises so nothing ships; Telegram carries the numbers."""
    bands = {"winner": (0.97, 1.05), "top_5": (5 * 0.95, 5 * 1.05),
             "top_10": (10 * 0.95, 10 * 1.05), "top_20": (20 * 0.95, 20 * 1.05)}
    problems = []
    for mkt, (lo, hi) in bands.items():
        probs = outrights.get(mkt) or {}
        if not probs:
            continue
        s = sum(float(p) for p in probs.values())
        if not (lo <= s <= hi):
            problems.append(f"{mkt}: sum={s:.3f} (band {lo:.2f}-{hi:.2f}, n={len(probs)})")
    if problems:
        msg = (f"outright probability mass INSANE for {tourney} — refusing to publish: "
               + "; ".join(problems)
               + ". Likely pre-event layers refilling live-zeroed players.")
        _alert(msg)
        raise RuntimeError(msg)


def build_payload() -> dict:
    si = _sim_inputs()
    tourney = getattr(si, "tourney", None)
    event_ids = getattr(si, "event_ids", []) or []
    event_id = str(event_ids[0]) if event_ids else None
    tour = getattr(si, "tour", "pga")
    cut_line = int(getattr(si, "CUT_LINE", getattr(si, "cutline", 65)))
    repl = _name_replacements()
    if not tourney:
        raise RuntimeError("sim_inputs.tourney is not set")

    rnd = _latest_round(tourney)
    # Guard against a stale, generic-named live dump from a PRIOR event defining this
    # week's field/outrights (cross-event bleed). Pre-event builds ignore it entirely.
    use_live = _live_dump_is_current(tourney, rnd, repl)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sim_run_at = _sim_run_at(tourney, rnd)
    if sim_run_at is None:
        # Publishing without a sim-time stamp would ship fairs consumers must
        # (and now do) reject — and with no source file there is nothing real
        # to publish anyway.
        raise RuntimeError(f"no sim source files found for '{tourney}' — cannot stamp "
                           f"sim_run_at; run the sim before publishing")
    outrights, outrights_nodh = _build_outrights(tourney, cut_line, repl, use_live=use_live)
    _check_outright_mass(outrights, tourney)
    matchup_source, matchup_run_at = _matchups_provenance(tourney, use_live=use_live)
    payload = {
        "event_id": event_id,
        "event_name": _resolve_event_name(event_id, tour, tourney),
        "tourney": tourney,
        "generated_at": now,                # publish wall-clock (display only)
        "sim_run_at": sim_run_at,           # when the SIM ran — consumers gate on this
        "round": rnd,                       # live round these round_* markets price
        "field": _build_field(tourney, repl, use_live=use_live),
        "outrights": outrights,
        # no-dead-heat top-N fairs for books that settle a top-N as a clean binary
        # (Kalshi, NoVig). The board grades those books against these instead of the
        # dead-heat `outrights` above. Older boards ignore this key (degrade to DH).
        "outrights_nodh": outrights_nodh,
        "matchups": _build_matchups(tourney, repl, use_live=use_live),
        "round_scores": _build_round_scores(tourney, rnd, repl),
        # per-player skill estimate (pred, SG/round vs field) for the board's
        # skill filter. Older boards ignore this key.
        "pred": _pred_lookup(tourney, repl),
        # provenance of the outright fairs: "live" = simulated_probs_live.csv
        # (live-conditioned, only legit once R2 exists), "pre" = the pre-event
        # full sim. The board refuses live-sourced outrights while its
        # tournament freeze isn't active (pre-R1) — a dead-man's switch against
        # exactly the 2026-07-15 stale-live-dump incident.
        "outrights_source": "live" if use_live else "pre",
        # Market-specific provenance.  A partial round publish may legitimately
        # carry these tournament-long markets forward, while the aggregate
        # sim_run_at advances with the new round cache.  Consumers and the git
        # merge guard can use these stamps without mistaking carried values for
        # output from the newest round simulation.
        "outrights_sim_run_at": _outrights_run_at(tourney, use_live=use_live),
        "matchups_source": matchup_source,
        "matchups_sim_run_at": matchup_run_at,
    }
    return payload


def _validate_complete_live_payload(payload: dict, expected_round: int | None = None) -> None:
    """Fail closed when an automated backup is not a complete live package.

    Routine/interactively-triggered publishes retain their historical partial-
    artifact carry-forward behavior. The nightly backup opts into this stricter
    contract because its purpose is to replace every live market from one run.
    """
    problems = []
    if payload.get("outrights_source") != "live":
        problems.append(
            f"outrights_source={payload.get('outrights_source')!r} (expected 'live')"
        )
    if payload.get("matchups_source") != "final_scores_live":
        problems.append(
            f"matchups_source={payload.get('matchups_source')!r} "
            "(expected 'final_scores_live')"
        )
    if expected_round is not None and int(payload.get("round") or -1) != int(expected_round):
        problems.append(
            f"round={payload.get('round')!r} (expected {int(expected_round)})"
        )
    for key in ("event_id", "tourney", "sim_run_at", "outrights_sim_run_at",
                "matchups_sim_run_at"):
        if not payload.get(key):
            problems.append(f"missing {key}")
    if not payload.get("field"):
        problems.append("empty field")

    outrights = payload.get("outrights") or {}
    for market in ("winner", "top_5", "top_10", "top_20", "make_cut"):
        if not outrights.get(market):
            problems.append(f"empty outrights.{market}")
    outrights_nodh = payload.get("outrights_nodh") or {}
    for market in ("top_5", "top_10", "top_20"):
        if not outrights_nodh.get(market):
            problems.append(f"empty outrights_nodh.{market}")
    if not payload.get("matchups"):
        problems.append("empty tournament matchups")
    if not payload.get("round_scores"):
        problems.append("empty round score PMFs")

    if problems:
        raise RuntimeError(
            "complete-live publish contract failed: " + "; ".join(problems)
        )


# ─── publish (commit to repo; board fetches via SIMS_PROCESS_PAT) ──────────────

BOARD_REPO = "mslade50/golf_scraping"


def _dispatch_board_build(sha: str | None = None) -> bool:
    """Best-effort repository_dispatch to the board repo so a fairs publish always
    triggers one board build, even when it lands outside the board's cron window.
    (The Open 2026-07-19: R4 fairs published Sat night fell in the overnight cron
    gap; every Sunday run then gate-skipped mid-play, so the board served R3-era
    fairs all day.) The dispatched run still goes through the board's own mid-play
    gate, so extra fires are harmless. Never breaks a publish."""
    import subprocess
    import time

    import requests

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # Local runs (round_sim on the laptop) have no PAT in env but do have an
        # authenticated gh CLI — borrow its token.
        try:
            r = subprocess.run(["gh", "auth", "token"], capture_output=True,
                               text=True, timeout=15)
            token = r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            token = None
    if not token:
        logger.warning("board dispatch: no GitHub token (GH_TOKEN / gh CLI) — "
                       "board picks the fairs up on its next cron build")
        _alert("board dispatch SKIPPED: no GitHub token — fresh fairs wait for "
               "the next board cron (may miss a pre-freeze window)")
        return False
    last_error = None
    for attempt in range(1, 4):
        try:
            # client_payload.sha pins the dispatched build's sims_process fetches to
            # THIS push (the GitHub contents API can serve a pre-push cached copy
            # for minutes after a push — the dispatched build then freezes the very
            # stale fairs it was fired to replace).
            body = {"event_type": "sim-fairs-published"}
            client_payload = {}
            if sha:
                client_payload["sha"] = sha
            if (os.environ.get("BOARD_SUPPRESS_SIM_CASCADE") or "").strip().lower() in (
                "1", "true", "yes"
            ):
                # Emergency/model-only refreshes must still rebuild the board from
                # this exact fairs commit, but should not launch the board's normal
                # downstream repricing simulation.  This is opt-in so routine
                # publishes retain the existing cascade behavior.
                client_payload["suppress_sim_cascade"] = True
            if client_payload:
                body["client_payload"] = client_payload
            resp = requests.post(
                f"https://api.github.com/repos/{BOARD_REPO}/dispatches",
                json=body,
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "application/vnd.github+json",
                         "X-GitHub-Api-Version": "2022-11-28"},
                timeout=15)
            if resp.status_code == 204:
                logger.info(f"board dispatch: triggered {BOARD_REPO} board build")
                return True
            last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            last_error = str(e)
        logger.warning(
            f"board dispatch attempt {attempt}/3 failed ({last_error})"
        )
        if attempt < 3:
            time.sleep(attempt)
    _alert(f"board dispatch FAILED after 3 attempts ({last_error}) — fresh "
           "fairs wait for the next board cron")
    return False


def _git_push(
    files=("sim_fairs.json",), *, require_dispatch: bool = False, _rebuilds: int = 1
) -> bool:
    """Publish the given repo-relative files to origin/main WITHOUT touching the
    local working tree, index, or branches. Builds one commit on top of origin/main
    via git plumbing and pushes it, so finishing a sim run can never rebase,
    autostash, or wedge the live repo. Automated runs can require the push and
    downstream board dispatch to succeed; interactive runs retain best-effort mode.

    Note: local main is left behind by the published commit (working-tree files
    match what was pushed); a routine `git pull` fast-forwards it."""
    import subprocess
    import tempfile
    import time

    def git(*args, env=None):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                              capture_output=True, text=True, env=env)

    fetch = None
    for attempt in range(1, 4):
        fetch = git("fetch", "origin", "main")
        if fetch.returncode == 0:
            break
        logger.warning(f"sim publish fetch attempt {attempt}/3 failed")
        if attempt < 3:
            time.sleep(attempt)
    if fetch is None or fetch.returncode != 0:
        message = "sim_fairs publish: git fetch failed after 3 attempts"
        logger.warning(message)
        if require_dispatch:
            raise RuntimeError(message)
        return False
    base = git("rev-parse", "origin/main").stdout.strip()
    if not base:
        logger.warning("sim_fairs publish: no origin/main; skipping")
        if require_dispatch:
            raise RuntimeError("sim_fairs publish: no origin/main")
        return False

    blobs = {}  # repo path -> blob, only for files that actually changed
    for fp in files:
        if not (PROJECT_ROOT / fp).exists():
            continue
        blob = git("hash-object", "-w", fp).stdout.strip()
        if blob and git("rev-parse", f"{base}:{fp}").stdout.strip() != blob:
            blobs[fp] = blob
    if not blobs:
        logger.info("sim publish: nothing changed on origin/main")
        dispatched = _dispatch_board_build(sha=base)
        if require_dispatch and not dispatched:
            raise RuntimeError("sim fairs are current but board dispatch failed")
        return True

    # ── Never regress origin's fairs (2026-07-01 audit, freshness #2). Git push
    # here is last-writer-wins on CONTENT: a machine holding older sim artifacts
    # (stale clone, unrotated laptop) builds a valid commit on top of origin/main
    # that replaces fresher fairs with older ones. Compare sim_run_at and refuse
    # the WHOLE publish (round samples from the same stale sim are equally wrong).
    if "sim_fairs.json" in blobs:
        local_pay = origin_pay = None
        try:
            local_pay = json.loads(
                (PROJECT_ROOT / "sim_fairs.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        show = git("show", f"{base}:sim_fairs.json")
        if show.returncode == 0:
            try:
                origin_pay = json.loads(show.stdout)
            except Exception:
                pass
        local_run = _parse_utc((local_pay or {}).get("sim_run_at"))
        origin_run = _parse_utc((origin_pay or {}).get("sim_run_at"))
        if origin_run is not None and (local_run is None or local_run < origin_run):
            # Cross-event escape hatch: on a 2-event week, event B's later sim
            # inflates origin's sim_run_at past anything a no-resim republish of
            # event A can produce — the un-qualified comparison then locks A out
            # deterministically. A DIFFERENT event with a FRESH local sim (<48h)
            # is a genuine concurrent-event publish, not a stale clone (a stale
            # clone's artifacts are days old). PUBLISH_ALLOW_EVENT_SWITCH=1
            # forces it manually.
            _ev_switch = (origin_pay and local_pay
                          and str(origin_pay.get("event_id")) != str(local_pay.get("event_id")))
            _force_switch = (os.environ.get("PUBLISH_ALLOW_EVENT_SWITCH") or "").strip().lower() in ("1", "true", "yes")
            _local_fresh = (local_run is not None and
                            (datetime.now(timezone.utc) - local_run).total_seconds() < 48 * 3600)
            if _ev_switch and (_local_fresh or _force_switch):
                _alert(f"EVENT SWITCH publish: replacing origin event "
                       f"{origin_pay.get('event_id')} ({origin_pay.get('tourney')}) with "
                       f"{local_pay.get('event_id')} ({local_pay.get('tourney')}) — "
                       f"{'forced' if _force_switch and not _local_fresh else 'fresh local sim'}. "
                       f"The board serving the OTHER event will drop to consensus.")
                logger.warning("sim publish: cross-event switch allowed "
                               f"(local sim_run_at {local_run})")
            else:
                msg = (f"REFUSED: origin sim_run_at {origin_run:%Y-%m-%d %H:%M} UTC is fresher "
                       f"than local {f'{local_run:%Y-%m-%d %H:%M} UTC' if local_run else '(unstamped)'} "
                       f"— this machine holds stale sim artifacts; not overwriting"
                       + (" (cross-event: set PUBLISH_ALLOW_EVENT_SWITCH=1 to force)" if _ev_switch else ""))
                logger.warning(f"sim publish {msg}")
                _alert(msg)
                if require_dispatch:
                    raise RuntimeError(msg)
                return False
        # A machine with only PARTIAL sim outputs builds a payload whose aggregate
        # sim_run_at is fresh because it just produced round scores, even though its
        # tournament-long artifacts are absent or older. Preserve the origin version
        # of those markets *with their own source/timestamp*. This lets a round-only
        # publish ship fresh round data without relabeling stale outright/H2H content
        # as output from that round run.
        # PUBLISH_ALLOW_SHRINK=1 forces a raw publish (no backfill; strips allowed).
        if (origin_pay and local_pay
                and origin_pay.get("event_id") == local_pay.get("event_id")
                and (os.environ.get("PUBLISH_ALLOW_SHRINK") or "").strip().lower()
                    not in ("1", "true", "yes")):
            carried = []  # market ids backfilled from origin (kept, not stripped)

            o_out = origin_pay.get("outrights")
            l_out = local_pay.get("outrights")
            o_nodh = origin_pay.get("outrights_nodh")
            l_nodh = local_pay.get("outrights_nodh")
            o_out = o_out if isinstance(o_out, dict) else {}
            l_out = l_out if isinstance(l_out, dict) else {}
            o_nodh = o_nodh if isinstance(o_nodh, dict) else {}
            l_nodh = l_nodh if isinstance(l_nodh, dict) else {}

            # An incomplete outright family is one partial artifact, not a safe
            # mix-and-match opportunity: winner/top-N/make-cut must share a sim
            # vintage. Copy the whole origin family when any populated submarket
            # would otherwise disappear.
            outright_incomplete = any(
                o_val and not l_out.get(sub) for sub, o_val in o_out.items()
            ) or any(
                o_val and not l_nodh.get(sub) for sub, o_val in o_nodh.items()
            )
            o_out_time = _payload_market_time(origin_pay, "outrights")
            l_out_time = _payload_market_time(local_pay, "outrights")
            outright_older = (
                o_out_time is not None
                and (l_out_time is None or l_out_time < o_out_time)
            )
            # Once the same event has live-conditioned probabilities, a partial
            # machine must not regress it to a pre-event table even if filesystem
            # mtimes were touched out of order.
            outright_source_regressed = (
                origin_pay.get("outrights_source") == "live"
                and local_pay.get("outrights_source") != "live"
            )
            if (o_out or o_nodh) and (
                outright_incomplete or outright_older or outright_source_regressed
            ):
                if o_out:
                    local_pay["outrights"] = o_out
                    carried.extend(f"outrights.{sub}" for sub, val in o_out.items() if val)
                if o_nodh:
                    local_pay["outrights_nodh"] = o_nodh
                    carried.extend(f"outrights_nodh.{sub}" for sub, val in o_nodh.items() if val)
                if origin_pay.get("field"):
                    local_pay["field"] = origin_pay["field"]
                    carried.append("field")
                local_pay["outrights_source"] = origin_pay.get(
                    "outrights_source", local_pay.get("outrights_source", "pre")
                )
                local_pay["outrights_sim_run_at"] = origin_pay.get(
                    "outrights_sim_run_at"
                ) or origin_pay.get("sim_run_at")

            o_match_time = _payload_market_time(origin_pay, "matchups")
            l_match_time = _payload_market_time(local_pay, "matchups")
            matchup_older = (
                o_match_time is not None
                and (l_match_time is None or l_match_time < o_match_time)
            )
            matchup_source_regressed = (
                origin_pay.get("matchups_source") == "final_scores_live"
                and local_pay.get("matchups_source") != "final_scores_live"
            )
            if origin_pay.get("matchups") and (
                not local_pay.get("matchups")
                or matchup_older
                or matchup_source_regressed
            ):
                local_pay["matchups"] = origin_pay["matchups"]
                local_pay["matchups_source"] = origin_pay.get(
                    "matchups_source", local_pay.get("matchups_source", "h2h_matrix")
                )
                local_pay["matchups_sim_run_at"] = origin_pay.get(
                    "matchups_sim_run_at"
                ) or origin_pay.get("sim_run_at")
                carried.append("matchups")

            if origin_pay.get("field") and not local_pay.get("field"):
                local_pay["field"] = origin_pay["field"]
                carried.append("field")
            if carried:
                carried = list(dict.fromkeys(carried))
                # Stamp provenance so a payload serving backfilled markets is
                # self-describing (debugging stale fairs starts here, not in
                # git archaeology). Consumers ignore unknown keys.
                local_pay["carried_from_origin"] = carried
                # Re-serialize the merged payload and re-hash WITHOUT touching the
                # working tree (hash-object --stdin), so the pushed commit carries
                # origin's markets we lack alongside our fresh ones.
                rehash = subprocess.run(
                    ["git", "-C", str(PROJECT_ROOT), "hash-object", "-w",
                     "--stdin", "--path", "sim_fairs.json"],
                    input=json.dumps(local_pay), capture_output=True, text=True)
                new_blob = rehash.stdout.strip()
                if rehash.returncode != 0 or not new_blob:
                    msg = ("REFUSED: merge re-hash failed; not overwriting origin "
                           f"(git: {(rehash.stderr or '').strip()[:120]})")
                    logger.warning(f"sim publish {msg}")
                    _alert(msg)
                    if require_dispatch:
                        raise RuntimeError(msg)
                    return False
                blobs["sim_fairs.json"] = new_blob
                # Mirror the merged payload to the local file so the working tree
                # matches the pushed commit (the docstring's fast-forward invariant);
                # otherwise the next `git pull` refuses to overwrite sim_fairs.json.
                try:
                    (PROJECT_ROOT / "sim_fairs.json").write_text(
                        json.dumps(local_pay), encoding="utf-8")
                except OSError as e:
                    logger.warning(f"sim publish: could not mirror merged payload "
                                   f"locally ({e}); next pull may need checkout")
                logger.info(f"sim publish: carried origin markets forward {carried} "
                            f"— fresh local markets still published")

    idx = os.path.join(
        tempfile.gettempdir(),
        f"sim_fairs_index_{os.getpid()}_{time.time_ns()}",
    )
    try:
        env = {**os.environ, "GIT_INDEX_FILE": idx}
        if git("read-tree", base, env=env).returncode != 0:
            raise RuntimeError("read-tree failed")
        for fp, blob in blobs.items():
            if git("update-index", "--add", "--cacheinfo", f"100644,{blob},{fp}",
                   env=env).returncode != 0:
                raise RuntimeError(f"update-index failed for {fp}")
        tree = git("write-tree", env=env).stdout.strip()
        ct = git("commit-tree", tree, "-p", base, "-m",
                 "sim_fairs: update model fairs + round samples")
        commit = ct.stdout.strip()
        if not commit:
            # surface git's stderr (e.g. 'Committer identity unknown' on CI
            # runners with no user.name/email) instead of a bare failure
            raise RuntimeError(
                f"commit-tree failed: {(ct.stderr or '').strip()[:160]}")
        pushed = False
        last_error = ""
        for attempt in range(1, 4):
            p = git("push", "origin", f"{commit}:main")
            if p.returncode == 0:
                pushed = True
                break
            last_error = (p.stderr or p.stdout).strip()[:160]
            logger.warning(
                f"sim publish push attempt {attempt}/3 failed: {last_error}"
            )
            fetch = git("fetch", "origin", "main")
            if fetch.returncode == 0:
                remote = git("rev-parse", "origin/main").stdout.strip()
                accepted = remote == commit or git(
                    "merge-base", "--is-ancestor", commit, "origin/main"
                ).returncode == 0
                if accepted:
                    logger.info("sim publish push was accepted despite client error")
                    pushed = True
                    break
                if remote and remote != base and _rebuilds > 0:
                    logger.info("origin/main advanced during publish; rebuilding commit")
                    return _git_push(
                        files,
                        require_dispatch=require_dispatch,
                        _rebuilds=_rebuilds - 1,
                    )
            if attempt < 3:
                time.sleep(attempt)
        if not pushed:
            message = f"sim publish push failed after 3 attempts: {last_error}"
            logger.warning(message)
            _alert(f"{message} — maker/board still serve old fairs")
            if require_dispatch:
                raise RuntimeError(message)
            return False
        logger.info(f"Pushed {', '.join(blobs)} to origin/main")
        dispatched = _dispatch_board_build(sha=commit)
        if require_dispatch and not dispatched:
            raise RuntimeError("sim fairs pushed but board dispatch failed")
        return True
    except Exception as e:
        if require_dispatch:
            raise
        logger.warning(f"sim publish failed ({e}); skipping push")
        _alert(f"publish failed ({e}) — fairs on origin NOT updated")
        return False
    finally:
        try:
            if os.path.exists(idx):
                os.remove(idx)
        except OSError:
            pass


def publish(
    push: bool = True,
    *,
    require_complete_live: bool = False,
    expected_round: int | None = None,
) -> dict:
    """Build sim_fairs.json (+ round_samples.parquet when live round data exists),
    write them, and (optionally) commit+push so the board can fetch them. Safe to
    call from new_sim.py / round_sim.py inside a try/except."""
    payload = build_payload()
    if require_complete_live:
        _validate_complete_live_payload(payload, expected_round=expected_round)
    with open(LOCAL_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info(f"Wrote {LOCAL_OUT}")
    files = ["sim_fairs.json"]
    prediction_snapshot = sync_r1_prediction_artifact(payload=payload)
    if prediction_snapshot is not None:
        files.append(prediction_snapshot.relative_to(PROJECT_ROOT).as_posix())
        prediction_manifest = manifest_path_for(prediction_snapshot)
        if prediction_manifest.is_file():
            files.append(prediction_manifest.relative_to(PROJECT_ROOT).as_posix())
    samples = _build_round_samples(payload["tourney"], payload.get("round"), _name_replacements())
    if samples is not None:
        # Stamp round/event/sim_run_at into the parquet metadata — the board's
        # round-matchup fallback guards on these (an unstamped samples file from a
        # prior round/event used to be trusted blindly).
        import pyarrow as _pa
        import pyarrow.parquet as _pq
        _tbl = _pa.Table.from_pandas(samples, preserve_index=True)
        _tbl = _tbl.replace_schema_metadata({**(_tbl.schema.metadata or {}),
                b"event_id": str(payload.get("event_id")).encode(),
                b"round": str(payload.get("round") or "").encode(),
                b"sim_run_at": str(payload.get("sim_run_at") or "").encode(),
                b"tourney": str(payload["tourney"]).encode()})
        _pq.write_table(_tbl, LOCAL_SAMPLES)
        files.append("round_samples.parquet")
        logger.info(f"Wrote {LOCAL_SAMPLES}")
    elif require_complete_live:
        raise RuntimeError("complete-live publish has no paired round sample tape")
    round_h2h_files = write_round_h2h(
        payload["tourney"], payload.get("round"), _name_replacements()
    )
    if require_complete_live and not round_h2h_files:
        raise RuntimeError("complete-live publish could not build round H2H fairs")
    files.extend(round_h2h_files)
    files.extend(write_round_3ball(payload["tourney"], payload.get("round"), _name_replacements()))

    # Tournament finish tape -> committed to this repo (git transport, exactly like
    # round_samples.parquet) so the board fetches it from GitHub. A live outright
    # payload MUST ship its paired live tape + made-cut mask; falling back to the
    # pre-event joint would make the displayed probabilities and portfolio scenarios
    # disagree. Pre-event publishing retains the historical best-effort behavior.
    live_tournament = payload.get("outrights_source") == "live"
    try:
        import pyarrow.parquet as pq
        tape = _build_tournament_samples(
            payload["tourney"],
            payload.get("event_id"),
            payload.get("generated_at"),
            _name_replacements(),
            use_live=live_tournament,
        )
        if live_tournament and tape is None:
            raise RuntimeError(
                "live outright payload has no paired final_scores_live/player_names_live tape"
            )
        if tape is not None:
            pq.write_table(tape, LOCAL_TOURN_SAMPLES)
            files.append("tournament_samples.parquet")
            logger.info(f"Wrote {LOCAL_TOURN_SAMPLES}")
        # Made-cut mask on the same draw axis (git downsample; board prices make_cut
        # off the joint). Full-res copy rides the release with the full tape below.
        mask = _build_made_cut_mask(
            payload["tourney"],
            payload.get("event_id"),
            _name_replacements(),
            use_live=live_tournament,
        )
        if live_tournament and mask is None:
            raise RuntimeError(
                "live outright payload has no made_cut_live mask paired to its finish tape"
            )
        if mask is not None:
            pq.write_table(mask, PROJECT_ROOT / "tournament_made_cut.parquet")
            files.append("tournament_made_cut.parquet")
            logger.info("Wrote tournament_made_cut.parquet")
    except Exception as e:
        if live_tournament:
            raise
        logger.warning(f"tournament_samples publish failed (non-fatal): {e}")

    # Full tournament, made-cut, and matchup release tapes are best-effort for
    # ordinary publishes. The strict nightly backup requires all three before it
    # may advance the git fairs commit, so the 100k optimizer/maker cannot lag.
    if push:
        _upload_release_tape_family(
            payload,
            _name_replacements(),
            strict=require_complete_live,
        )

    if push:
        strict_publish = require_complete_live or (
            os.environ.get("REQUIRE_SIM_FAIRS_PUBLISH") or ""
        ).strip().lower() in ("1", "true", "yes")
        _git_push(files, require_dispatch=strict_publish)
    return payload


def main():
    ap = argparse.ArgumentParser(description="Publish sim fair probabilities (commit to repo)")
    ap.add_argument("--dry-run", action="store_true", help="build + summary, no write/push")
    ap.add_argument("--no-push", action="store_true", help="write local file, skip git push")
    ap.add_argument("--round-h2h-only", action="store_true",
                    help="Build + push ONLY the round H2H artifact (for the cache-free "
                         "repricer). Skips the sim_fairs.json rebuild so a --sim-only "
                         "backup run can ship it without clobbering the board's fairs.")
    ap.add_argument(
        "--require-complete-live",
        action="store_true",
        help=("Require live outrights, tournament H2H, round score PMFs, and paired "
              "round/tournament tapes; also require push + board dispatch success."),
    )
    ap.add_argument(
        "--expected-round",
        type=int,
        help="Expected live round for --require-complete-live (fails on stale/higher caches)",
    )
    args = ap.parse_args()

    if args.round_h2h_only and args.require_complete_live:
        ap.error("--round-h2h-only cannot be combined with --require-complete-live")
    if args.no_push and args.require_complete_live:
        ap.error("--no-push cannot be combined with --require-complete-live")
    if args.expected_round is not None and not args.require_complete_live:
        ap.error("--expected-round requires --require-complete-live")

    if args.round_h2h_only:
        si = _sim_inputs()
        tourney = getattr(si, "tourney", None)
        if not tourney:
            raise RuntimeError("sim_inputs.tourney is not set")
        rnd = _latest_round(tourney)
        if rnd is None:
            # Thu-Sun this means the nightly backup found NO round caches at all
            # (sheet/sim_inputs tourney split-brain, or the sim never ran) — the
            # board silently degrades to consensus, so page instead of exiting 0.
            if datetime.now(timezone.utc).isoweekday() in (4, 5, 6, 7):
                _alert(f"round-h2h-only: no round sim cache for '{tourney}' on a "
                       f"live day — nightly backup published NOTHING; board falls "
                       f"back to consensus. Sheet vs sim_inputs tourney mismatch?")
            logger.warning(f"round-h2h-only: no round cache for {tourney} — nothing to publish")
            return
        logger.info(f"round-h2h-only: {tourney} R{rnd}")
        files = list(write_round_h2h(tourney, rnd) or [])
        # 3-balls ride the same backup publish: when this is the week's only
        # publish path, a missing round_3ball parquet silently degrades every
        # 3-ball to consensus on exactly the mornings the backup exists for.
        # Needs DATAGOLF_API_KEY in env for the tee-time threesomes.
        try:
            files += list(write_round_3ball(tourney, rnd) or [])
        except Exception as e:
            logger.warning(f"round 3-ball publish failed (non-fatal): {e}")
        if args.dry_run:
            return
        if files and not args.no_push:
            _git_push(tuple(files))
        return

    payload = build_payload()
    if args.require_complete_live:
        _validate_complete_live_payload(
            payload, expected_round=args.expected_round
        )
    o = payload["outrights"]
    ondh = payload.get("outrights_nodh") or {}
    logger.info(f"event {payload['event_id']} ({payload['tourney']}) @ {payload['generated_at']}")
    logger.info(f"  round: {payload.get('round')} | outright markets: "
                f"{{{', '.join(f'{k}:{len(v)}' for k, v in o.items())}}}")
    logger.info(f"  outrights_nodh (Kalshi/NoVig): "
                f"{{{', '.join(f'{k}:{len(v)}' for k, v in ondh.items())}}}")
    logger.info(f"  matchup pairs: {len(payload['matchups'])} | round_scores players: "
                f"{len(payload.get('round_scores') or {})} | preds: "
                f"{len(payload.get('pred') or {})}")

    if args.dry_run:
        return
    publish(
        push=not args.no_push,
        require_complete_live=args.require_complete_live,
        expected_round=args.expected_round,
    )


if __name__ == "__main__":
    main()

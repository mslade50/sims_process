"""Publish a compact sim_fairs.json (model fair probabilities) so the
golf_scraping odds board can grade book prices against the sim.

What it ships (per current event):
  - outrights: winner / top_5 / top_10 / top_20 / make_cut  -> {player: prob}
  - matchups : tournament head-to-head  -> [player_a, player_b, P(a beats b)]

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
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
LOCAL_OUT = PROJECT_ROOT / "sim_fairs.json"


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


# ─── builders ─────────────────────────────────────────────────────────────────

def _build_outrights(tourney: str, cut_line: int, repl: dict) -> dict:
    """winner/top_5/top_10/top_20 from finish_equity (live preferred), make_cut
    derived from the rank-probability distribution (P(finish rank <= cut))."""
    out = {"winner": {}, "top_5": {}, "top_10": {}, "top_20": {}, "make_cut": {}}

    # finish equity: simulated_win_prob + top_5/top_10/top_20 (live preferred)
    eq = _find(f"{tourney}/finish_equity_live_{tourney}.csv",
               f"finish_equity_live_{tourney}.csv",
               f"{tourney}/finish_equity_{tourney}.csv",
               f"finish_equity_{tourney}.csv")
    if eq is not None:
        df = pd.read_csv(eq)
        col = {"winner": "simulated_win_prob", "top_5": "top_5",
               "top_10": "top_10", "top_20": "top_20"}
        for mkt, c in col.items():
            if c in df.columns:
                for _, r in df.iterrows():
                    p = r[c]
                    if pd.notna(p) and p > 0:
                        out[mkt][_norm(r["player_name"], repl)] = round(float(p), 5)
        logger.info(f"outrights from {eq.name}: winner={len(out['winner'])}")
    else:
        # fall back to the two split files
        wp = _find("simulated_probs_live.csv", "simulated_probs.csv")
        if wp is not None:
            df = pd.read_csv(wp)
            for _, r in df.iterrows():
                p = r["simulated_win_prob"]
                if pd.notna(p) and p > 0:
                    out["winner"][_norm(r["player_name"], repl)] = round(float(p), 5)
        tf = _find(f"top_finish_probs_{tourney}.csv", f"{tourney}/top_finish_probs_{tourney}.csv")
        if tf is not None:
            df = pd.read_csv(tf)
            for mkt in ("top_5", "top_10", "top_20"):
                if mkt in df.columns:
                    for _, r in df.iterrows():
                        p = r[mkt]
                        if pd.notna(p) and p > 0:
                            out[mkt][_norm(r["player_name"], repl)] = round(float(p), 5)
        logger.info(f"outrights from split files: winner={len(out['winner'])}")

    # make_cut: prefer the exact simulated cut prob persisted by new_sim.py
    # (true cut: top-N + ties + 10-shot rule). Fall back to a rank-prob estimate
    # using prob_ndh (raw min-rank, so ties AT the cut are counted) — NOT prob_u
    # (dead-heat spread, which pushes tie mass past the cut and undercounts).
    mc_file = _find(f"make_cut_probs_{tourney}.csv", f"{tourney}/make_cut_probs_{tourney}.csv")
    if mc_file is not None:
        df = pd.read_csv(mc_file)
        for _, r in df.iterrows():
            p = r["make_cut"]
            if pd.notna(p) and p > 0:
                out["make_cut"][_norm(r["player_name"], repl)] = round(float(min(p, 1.0)), 5)
        logger.info(f"make_cut from {mc_file.name} (exact sim cut): {len(out['make_cut'])}")
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

    return {k: v for k, v in out.items() if v}


def _build_matchups(tourney: str, repl: dict) -> list:
    """Tournament head-to-head: [player_a, player_b, P(a beats b), tie%]."""
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

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "event_id": event_id,
        "event_name": _resolve_event_name(event_id, tour, tourney),
        "tourney": tourney,
        "generated_at": now,
        "outrights": _build_outrights(tourney, cut_line, repl),
        "matchups": _build_matchups(tourney, repl),
    }
    return payload


# ─── publish (commit to repo; board fetches via SIMS_PROCESS_PAT) ──────────────

def _git_push() -> None:
    """Publish sim_fairs.json to origin/main WITHOUT touching the local working
    tree, index, or branches. Builds a commit on top of origin/main via git
    plumbing and pushes it, so finishing a sim run can never rebase, autostash,
    or wedge the live repo. On any failure it logs and skips (next run retries).

    Note: local main is left behind by the published commit (working-tree
    sim_fairs.json matches what was pushed); a routine `git pull` fast-forwards it."""
    import os
    import subprocess
    import tempfile

    def git(*args, env=None):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                              capture_output=True, text=True, env=env)

    if git("fetch", "origin", "main").returncode != 0:
        logger.warning("sim_fairs publish: git fetch failed; skipping")
        return
    base = git("rev-parse", "origin/main").stdout.strip()
    if not base:
        logger.warning("sim_fairs publish: no origin/main; skipping")
        return
    blob = git("hash-object", "-w", "sim_fairs.json").stdout.strip()
    if not blob:
        logger.warning("sim_fairs publish: could not hash sim_fairs.json")
        return
    if git("rev-parse", f"{base}:sim_fairs.json").stdout.strip() == blob:
        logger.info("sim_fairs.json unchanged on origin/main — nothing to push")
        return

    idx = os.path.join(tempfile.gettempdir(), f"sim_fairs_index_{os.getpid()}")
    try:
        env = {**os.environ, "GIT_INDEX_FILE": idx}
        if (git("read-tree", base, env=env).returncode != 0 or
                git("update-index", "--add", "--cacheinfo",
                    f"100644,{blob},sim_fairs.json", env=env).returncode != 0):
            raise RuntimeError("tree assembly failed")
        tree = git("write-tree", env=env).stdout.strip()
        commit = git("commit-tree", tree, "-p", base, "-m",
                     "sim_fairs: update model fair probabilities").stdout.strip()
        if not commit:
            raise RuntimeError("commit-tree failed")
        p = git("push", "origin", f"{commit}:main")
        if p.returncode == 0:
            logger.info("Pushed sim_fairs.json to origin/main")
        else:
            logger.warning(f"sim_fairs push rejected (retries next run): "
                           f"{(p.stderr or p.stdout).strip()[:160]}")
    except Exception as e:
        logger.warning(f"sim_fairs publish failed ({e}); skipping push")
    finally:
        try:
            if os.path.exists(idx):
                os.remove(idx)
        except OSError:
            pass


def publish(push: bool = True) -> dict:
    """Build sim_fairs.json, write it, and (optionally) commit+push so the board
    can fetch it. Safe to call from new_sim.py inside a try/except."""
    payload = build_payload()
    with open(LOCAL_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info(f"Wrote {LOCAL_OUT}")
    if push:
        _git_push()
    return payload


def main():
    ap = argparse.ArgumentParser(description="Publish sim fair probabilities (commit to repo)")
    ap.add_argument("--dry-run", action="store_true", help="build + summary, no write/push")
    ap.add_argument("--no-push", action="store_true", help="write local file, skip git push")
    args = ap.parse_args()

    payload = build_payload()
    o = payload["outrights"]
    logger.info(f"event {payload['event_id']} ({payload['tourney']}) @ {payload['generated_at']}")
    logger.info(f"  outright markets: {{{', '.join(f'{k}:{len(v)}' for k, v in o.items())}}}")
    logger.info(f"  matchup pairs: {len(payload['matchups'])}")

    if args.dry_run:
        return
    with open(LOCAL_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info(f"Wrote {LOCAL_OUT}")
    if not args.no_push:
        _git_push()


if __name__ == "__main__":
    main()

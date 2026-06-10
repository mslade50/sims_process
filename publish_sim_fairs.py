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


def build_payload() -> dict:
    si = _sim_inputs()
    tourney = getattr(si, "tourney", None)
    event_ids = getattr(si, "event_ids", []) or []
    event_id = str(event_ids[0]) if event_ids else None
    cut_line = int(getattr(si, "CUT_LINE", getattr(si, "cutline", 65)))
    repl = _name_replacements()
    if not tourney:
        raise RuntimeError("sim_inputs.tourney is not set")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "event_id": event_id,
        "tourney": tourney,
        "generated_at": now,
        "outrights": _build_outrights(tourney, cut_line, repl),
        "matchups": _build_matchups(tourney, repl),
    }
    return payload


# ─── publish (commit to repo; board fetches via SIMS_PROCESS_PAT) ──────────────

def _git_push() -> None:
    """Commit sim_fairs.json to this repo and push. The golf_scraping odds board
    fetches it from GitHub with the SIMS_PROCESS_PAT it already has."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                              capture_output=True, text=True)

    git("add", "--", "sim_fairs.json")
    if git("diff", "--cached", "--quiet", "--", "sim_fairs.json").returncode == 0:
        logger.info("sim_fairs.json unchanged — nothing to push")
        return
    git("commit", "-m", "sim_fairs: update model fair probabilities", "--", "sim_fairs.json")
    git("pull", "--rebase", "--autostash")
    p = git("push")
    if p.returncode != 0:
        logger.warning(f"git push failed: {(p.stderr or p.stdout).strip()[:200]}")
    else:
        logger.info("Pushed sim_fairs.json")


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

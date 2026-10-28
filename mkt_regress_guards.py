"""Freshness / identity guards for the legacy two-pass chain (SC-05, SC-06).

Pure helpers (no pandas needed) so mkt_regress.py and new_sim.py can fail LOUDLY on stale
inputs instead of pricing from them.  No model math lives here.

Kill switches (env):
    MKT_REGRESS_ALLOW_STALE=1          skip the mtime / odds-coverage guards in mkt_regress
    MKT_REGRESS_MAX_INPUT_AGE_DAYS=N   input freshness window (default 7)
    MKT_REGRESS_MIN_ODDS_COVERAGE=F    share of field with DG win odds (default 0.8)
    NEW_SIM_ALLOW_UNVERIFIED_FINAL=1   let --final-pass run without a matching sidecar
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


class StaleInputError(RuntimeError):
    pass


def _flag(name):
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs_fresh(paths, *, now=None, max_age_days=None):
    """Raise StaleInputError naming every input older than the window."""
    if _flag("MKT_REGRESS_ALLOW_STALE"):
        print("[guard] MKT_REGRESS_ALLOW_STALE=1: input freshness NOT enforced")
        return
    if max_age_days is None:
        try:
            max_age_days = float(os.environ.get("MKT_REGRESS_MAX_INPUT_AGE_DAYS") or 7)
        except ValueError:
            max_age_days = 7.0
    now = time.time() if now is None else now
    stale = []
    for label, path in paths:
        age_days = (now - os.path.getmtime(path)) / 86400.0
        if age_days > max_age_days:
            stale.append(f"{label} {path!r} is {age_days:.1f} days old (> {max_age_days:g})")
    if stale:
        raise StaleInputError(
            "mkt_regress refuses stale inputs: " + "; ".join(stale)
            + ". Regenerate them for this event (or set MKT_REGRESS_ALLOW_STALE=1 to override)."
        )


def check_odds_coverage(n_field, n_with_odds, *, source="DataGolf outrights"):
    """The outright feed must cover the field; a different event / off week does not."""
    if _flag("MKT_REGRESS_ALLOW_STALE"):
        return
    try:
        floor = float(os.environ.get("MKT_REGRESS_MIN_ODDS_COVERAGE") or 0.8)
    except ValueError:
        floor = 0.8
    if n_field <= 0:
        raise StaleInputError("mkt_regress: empty player field")
    share = n_with_odds / n_field
    if share < floor:
        raise StaleInputError(
            f"mkt_regress: {source} priced only {n_with_odds}/{n_field} field players "
            f"({share:.0%} < {floor:.0%}) - wrong event, off week, or stale feed; refusing to "
            "regress against it (set MKT_REGRESS_ALLOW_STALE=1 to override)."
        )


def meta_path_for(final_path):
    text = str(final_path)
    return Path(text[:-4] + ".meta.json") if text.endswith(".csv") else Path(text + ".meta.json")


def write_final_meta(final_path, *, tourney, event_ids, input_paths, n_players):
    meta = {
        "tourney": tourney,
        "event_ids": [str(x) for x in (event_ids or [])],
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_players": int(n_players),
        "final_predictions_sha256": file_sha256(final_path),
        "inputs": {label: {"path": str(p), "sha256": file_sha256(p)} for label, p in input_paths},
    }
    out = meta_path_for(final_path)
    out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out


def verify_final_meta(final_path, *, tourney, event_ids, init_skill_path):
    """Raise StaleInputError unless the final predictions were produced by mkt_regress for
    THIS tourney/event from the init_sim_skill that is on disk now."""
    if _flag("NEW_SIM_ALLOW_UNVERIFIED_FINAL"):
        print("[guard] NEW_SIM_ALLOW_UNVERIFIED_FINAL=1: final predictions lineage NOT verified")
        return None
    mp = meta_path_for(final_path)
    if not mp.is_file():
        raise StaleInputError(
            f"--final-pass: no lineage sidecar {mp.name} next to {Path(final_path).name}; the "
            "file may be a previous year's or a committed copy. Re-run mkt_regress.py for this "
            "event (or set NEW_SIM_ALLOW_UNVERIFIED_FINAL=1)."
        )
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StaleInputError(f"--final-pass: corrupt sidecar {mp.name}: {exc}") from None
    problems = []
    if meta.get("tourney") != tourney:
        problems.append(f"tourney {meta.get('tourney')!r} != {tourney!r}")
    want = [str(x) for x in (event_ids or [])]
    if want and sorted(meta.get("event_ids") or []) != sorted(want):
        problems.append(f"event_ids {meta.get('event_ids')} != {want}")
    if meta.get("final_predictions_sha256") != file_sha256(final_path):
        problems.append("final predictions changed after mkt_regress wrote them")
    skill = (meta.get("inputs") or {}).get("init_sim_skill") or {}
    if os.path.exists(init_skill_path) and skill.get("sha256") != file_sha256(init_skill_path):
        problems.append("init_sim_skill differs from the one mkt_regress used")
    if problems:
        raise StaleInputError("--final-pass: final predictions do not match this run: "
                              + "; ".join(problems))
    return meta

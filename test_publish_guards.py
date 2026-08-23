"""Tests for publish_sim_fairs._git_push freshness/shrink guards (2026-07-01
audit, freshness #2). Runs against a THROWAWAY local bare repo — the real
origin is never touched. _alert is monkeypatched to record instead of send.
Run: python test_publish_guards.py"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import publish_sim_fairs as psf

_p = _f = 0


def eq(name, got, want):
    global _p, _f
    if got == want:
        _p += 1
    else:
        _f += 1
        print(f"FAIL {name}: got {got!r} want {want!r}")


def _payload(sim_run_at, top5=True, event_id=30):
    out = {"winner": {"a b": 0.1, "c d": 0.2}}
    out["top_5"] = {"a b": 0.4, "c d": 0.5} if top5 else {}
    return {"event_id": event_id, "tourney": "deere",
            "generated_at": "2026-07-02 12:00:00 UTC", "sim_run_at": sim_run_at,
            "round": 1, "outrights": out,
            "matchups": [{"p1": "a b", "p2": "c d", "p1_prob": 0.5}]}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _origin_fairs(work):
    _git(work, "fetch", "origin", "main")
    r = _git(work, "show", "origin/main:sim_fairs.json")
    return json.loads(r.stdout)


tmp = Path(tempfile.mkdtemp(prefix="pub_guard_"))
bare = tmp / "origin.git"
work = tmp / "work"
subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)],
               capture_output=True)
subprocess.run(["git", "init", "-b", "main", str(work)], capture_output=True)
_git(work, "config", "user.name", "test")
_git(work, "config", "user.email", "test@test")
_git(work, "remote", "add", "origin", str(bare))

# Seed origin with a STAMPED, full-market payload (T1).
T1 = "2026-07-01 20:00:00 UTC"
T0 = "2026-07-01 10:00:00 UTC"   # older than T1
T2 = "2026-07-02 08:00:00 UTC"   # fresher than T1
T3 = "2026-07-02 10:00:00 UTC"
T4 = "2026-07-02 11:00:00 UTC"
(work / "sim_fairs.json").write_text(json.dumps(_payload(T1)), encoding="utf-8")
_git(work, "add", "sim_fairs.json")
_git(work, "commit", "-m", "seed")
_git(work, "push", "origin", "main")

# Point the module at the sandbox repo and capture alerts.
psf.PROJECT_ROOT = work
alerts = []
psf._alert = lambda text: alerts.append(text)
psf._dispatch_board_build = lambda sha=None: True


def attempt(payload):
    (work / "sim_fairs.json").write_text(json.dumps(payload), encoding="utf-8")
    alerts.clear()
    psf._git_push(("sim_fairs.json",))
    return _origin_fairs(work)


# 1. Regress: older sim_run_at must NOT overwrite fresher origin.
got = attempt(_payload(T0))
eq("regress refused (origin unchanged)", got["sim_run_at"], T1)
eq("regress alerted", len(alerts), 1)

# 2. Unstamped local vs stamped origin: refused.
got = attempt(_payload(None))
eq("unstamped-local refused", got["sim_run_at"], T1)
eq("unstamped-local alerted", len(alerts), 1)

# 3. Shrink: fresher stamp but top_5 emptied — carry origin forward.
got = attempt(_payload(T2, top5=False))
eq("shrink backfilled (top_5 kept on origin)", len(got["outrights"]["top_5"]), 2)
eq("shrink backfill does not alert", len(alerts), 0)
eq("shrink provenance names market", "outrights.top_5" in got["carried_from_origin"], True)

# 4. Shrink on a DIFFERENT event (rotation): allowed.
got = attempt(_payload(T2, top5=False, event_id=31))
eq("rotation shrink allowed", got["event_id"], 31)
eq("rotation no alert", len(alerts), 0)

# reseed origin with the same-event full payload for the remaining cases
attempt(_payload(T2))

# 5. Fresher, full payload: pushes clean.
got = attempt(_payload("2026-07-02 09:00:00 UTC"))
eq("fresher full pushes", got["sim_run_at"], "2026-07-02 09:00:00 UTC")
eq("fresher full no alert", len(alerts), 0)

# 6. Origin unstamped (pre-guard payload): stamped local wins.
(work / "sim_fairs.json").write_text(json.dumps(_payload(None)), encoding="utf-8")
_git(work, "add", "sim_fairs.json")
_git(work, "commit", "-m", "unstamped origin")
_git(work, "push", "-f", "origin", "main")
got = attempt(_payload(T2))
eq("stamped beats unstamped origin", got["sim_run_at"], T2)

# 7. A fresh ROUND run with fully-populated but older PRE-event tournament
# artifacts must preserve origin's LIVE outright/H2H content and its own market
# timestamps. The aggregate sim_run_at may advance for the fresh round markets.
origin_live = _payload(T3)
origin_live["outrights"]["winner"] = {"a b": 0.91, "c d": 0.09}
origin_live["outrights_source"] = "live"
origin_live["outrights_sim_run_at"] = T3
origin_live["matchups"] = [["a b", "c d", 0.77]]
origin_live["matchups_source"] = "final_scores_live"
origin_live["matchups_sim_run_at"] = T3
got = attempt(origin_live)
eq("live seed pushed", got["outrights_source"], "live")

round_only = _payload(T4)
round_only["outrights"]["winner"] = {"a b": 0.11, "c d": 0.89}
round_only["outrights_source"] = "pre"
round_only["outrights_sim_run_at"] = T0
round_only["matchups"] = [["a b", "c d", 0.22]]
round_only["matchups_source"] = "h2h_matrix"
round_only["matchups_sim_run_at"] = T0
got = attempt(round_only)
eq("fresh round aggregate stamp advances", got["sim_run_at"], T4)
eq("older pre outright content rejected", got["outrights"]["winner"]["a b"], 0.91)
eq("carried outright source stays live", got["outrights_source"], "live")
eq("carried outright timestamp stays old", got["outrights_sim_run_at"], T3)
eq("older pre H2H content rejected", got["matchups"], [["a b", "c d", 0.77]])
eq("carried H2H source stays live", got["matchups_source"], "final_scores_live")
eq("carried H2H timestamp stays old", got["matchups_sim_run_at"], T3)
eq("outright carry is self-describing",
   "outrights.winner" in got["carried_from_origin"], True)
eq("H2H carry is self-describing", "matchups" in got["carried_from_origin"], True)

print(f"\n{_p} passed, {_f} failed")
shutil.rmtree(tmp, ignore_errors=True)
raise SystemExit(1 if _f else 0)

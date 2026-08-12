"""Deterministic tri-repo health checks (sims_process / sim_prep / golf_scraping).

Run by the weekly-repo-check skill. Emits PASS/WARN/FAIL/SKIP lines; exit code
is 1 if any FAIL. Modes:
  --mode monday     post-event checks (grading landed, publication, task results)
  --mode readiness  pre-event checks (stale files, fan-out, kernel, odds feed)
  --mode all        everything (default: inferred from weekday, Mon->monday)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIMS = Path(r"C:\Users\McKinley Slade\dev\sims_process")
PREP = Path(r"C:\Users\McKinley Slade\dev\sim_prep")
SCRAPE = Path(r"C:\Users\McKinley Slade\dev\golf_scraping")
ONEDRIVE = Path(r"C:\Users\McKinley Slade\OneDrive")

RESULTS: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _mtime(p: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_event() -> tuple[str, str]:
    """(tourney, event_id) from the sync manifest — cheap, no sheet fetch."""
    try:
        m = json.loads((SIMS / "dashboard_data" / ".sync_manifest.json").read_text())
        return str(m.get("event", "")), str(m.get("event_id", ""))
    except (OSError, ValueError):
        return "", ""


# ---------------------------------------------------------------- both modes

def check_git_state() -> None:
    for repo, branch in ((SIMS, "main"), (PREP, "master"), (SCRAPE, "master")):
        if not repo.is_dir():
            report("SKIP", f"git {repo.name}", "repo missing")
            continue
        run(["git", "fetch", "origin"], cwd=repo, timeout=180)
        head = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
        counts = run(["git", "rev-list", "--left-right", "--count",
                      f"{branch}...origin/{branch}"], cwd=repo).stdout.split()
        ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (0, 0)
        dirty = [l for l in run(["git", "status", "--short"], cwd=repo).stdout.splitlines()
                 if l.startswith(" M") and l.strip().endswith((".py", ".yml", ".yaml", ".rs"))]
        msgs = []
        if head != branch:
            msgs.append(f"on {head} not {branch}")
        if ahead:
            msgs.append(f"{ahead} unpushed commit(s)")
        if behind:
            msgs.append(f"{behind} behind origin")
        if dirty:
            msgs.append(f"{len(dirty)} modified pipeline file(s)")
        if msgs:
            report("WARN" if not (ahead or head != branch) else "FAIL",
                   f"git {repo.name}", "; ".join(msgs))
        else:
            report("PASS", f"git {repo.name}", f"{branch} in sync, pipeline files clean")


def check_r2_publication() -> None:
    """Writer DB data must not outrun the published R2 tip (readers/CI lag)."""
    state = PREP / "work" / ".dgdata_state.json"
    writer = PREP / "work" / "dg_historical.db"
    if not state.exists() or not writer.exists():
        report("SKIP", "dgdata publication", "state/writer db missing")
        return
    try:
        rec = json.loads(state.read_text()).get("canonical", {}).get("recorded_at", "")
        tip = datetime.fromisoformat(rec.replace("Z", "+00:00"))
    except (ValueError, OSError):
        report("FAIL", "dgdata publication", "unparseable .dgdata_state.json")
        return
    wm = _mtime(writer)
    lag = (wm - tip) if wm else timedelta(0)
    if lag > timedelta(hours=36):
        report("FAIL", "dgdata publication",
               f"writer db is {lag.days}d{lag.seconds // 3600}h newer than published tip "
               f"({rec}) - readers/CI run on stale data; run: python -m dgdata publish")
    else:
        report("PASS", "dgdata publication", f"tip {rec}")


def _export_recovered() -> bool:
    """True if weekly_shot_export.log's tail shows a clean run in the last 3 days."""
    log = PREP / "work" / "weekly_shot_export.log"
    m = _mtime(log)
    if m is None or datetime.now(timezone.utc) - m > timedelta(days=3):
        return False
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
    except OSError:
        return False
    return not any("Traceback" in l or "Error" in l or "error" in l for l in tail)


def check_scheduled_tasks() -> None:
    tasks = ["GolfRichShotCollector", "GolfPinHighShadowGenerate",
             "GolfPinHighShadowScore", "GolfShotWeeklyExport", "sim-health-check-weekly"]
    for t in tasks:
        r = run(["schtasks", "/query", "/tn", t, "/fo", "LIST", "/v"])
        if r.returncode != 0:
            report("WARN", f"schtask {t}", "task not found")
            continue
        result_line = next((l for l in r.stdout.splitlines() if "Last Result" in l), "")
        code = result_line.split(":", 1)[-1].strip() if result_line else "?"
        if code in ("0", "267011"):  # 267011 = has not yet run
            report("PASS", f"schtask {t}", f"last result {code}")
        elif t == "GolfShotWeeklyExport" and _export_recovered():
            report("WARN", f"schtask {t}", f"last scheduled result {code}, but the export "
                   "log shows a clean manual rerun since (schtasks result clears next cycle)")
        else:
            report("FAIL", f"schtask {t}", f"last result {code} (nonzero)")


def check_kernel() -> None:
    try:
        import sims_kernel  # type: ignore
        ok = sims_kernel.selftest()
        pyd = Path(sims_kernel.__file__).parent / "sims_kernel.pyd"
        pyd_m = _mtime(pyd)
        src_m = max((_mtime(p) for p in (SIMS / "rust" / "src").glob("*.rs")
                     if _mtime(p)), default=None)
        if not ok:
            report("FAIL", "sims_kernel", "selftest False")
        elif pyd_m and src_m and src_m > pyd_m:
            report("FAIL", "sims_kernel", "rust/src newer than installed pyd - rebuild "
                   "per CLAUDE.md ritual or local sims silently run old logic")
        else:
            report("PASS", "sims_kernel", f"selftest True, pyd {pyd_m:%Y-%m-%d %H:%M}")
    except Exception as e:
        report("FAIL", "sims_kernel", f"import/selftest error: {e}")


def check_wheels_release() -> None:
    """The CI kernel wheel must postdate the last rust/ change."""
    last_rust = run(["git", "log", "-1", "--format=%cI", "--", "rust/"], cwd=SIMS).stdout.strip()
    r = run(["gh", "release", "view", "wheels-latest", "--json", "assets"], cwd=SIMS, timeout=60)
    if r.returncode != 0 or not last_rust:
        report("SKIP", "CI kernel wheel", "gh release/git log unavailable")
        return
    try:
        assets = json.loads(r.stdout)["assets"]
        win = [a for a in assets if "win_amd64" in a["name"] and "0.1.0" not in a["name"]]
        newest = max(a["updatedAt"] for a in win)
        if newest < last_rust:
            report("FAIL", "CI kernel wheel",
                   f"wheels-latest asset ({newest}) predates last rust/ commit ({last_rust}) "
                   "- CI installs a kernel without the latest cascade logic")
        else:
            report("PASS", "CI kernel wheel", f"asset {newest} >= rust change {last_rust}")
    except (ValueError, KeyError) as e:
        report("WARN", "CI kernel wheel", f"could not parse release assets: {e}")


def check_odds_feed() -> None:
    tourney, event_id = _current_event()
    r = run(["gh", "api", "repos/mslade50/golf_scraping/contents/data/tournament_matchups_latest.json",
             "--jq", ".content"], cwd=SIMS, timeout=60)
    if r.returncode != 0:
        report("FAIL", "odds feed", "GitHub API fetch failed (repricing has no fresh odds source)")
        return
    import base64
    try:
        data = json.loads(base64.b64decode(r.stdout.strip()))
        stamp = data.get("last_updated", "")
        feed_event = str(data.get("event_id", ""))
        age_note = f"last_updated {stamp}, event {feed_event}"
        ts = datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        if age > timedelta(hours=30):
            report("FAIL", "odds feed", f"{age_note} - stale >30h, scraper may have stopped")
        elif event_id and feed_event and feed_event != event_id:
            report("WARN", "odds feed", f"{age_note} but manifest event is {event_id} "
                   "(early-week rollover is normal Mon/Tue)")
        else:
            report("PASS", "odds feed", age_note)
    except Exception as e:
        report("WARN", "odds feed", f"parse error: {e}")


# -------------------------------------------------------------- monday mode

def check_grading_landed() -> None:
    """Grading must actually reach origin — the silent-push-loss class (2026-08)."""
    r = run(["gh", "run", "list", "--workflow", "monday-grading.yml", "--limit", "2",
             "--json", "status,conclusion,createdAt"], cwd=SIMS, timeout=60)
    try:
        runs = json.loads(r.stdout)
    except ValueError:
        runs = []
    if not runs:
        report("SKIP", "monday grading", "no runs found")
        return
    latest = runs[0]
    if latest["status"] != "completed":
        report("WARN", "monday grading", "run still in progress - re-check later")
        return
    if latest["conclusion"] != "success":
        report("FAIL", "monday grading", f"latest run {latest['conclusion']}")
        return
    # Green run is necessary but NOT sufficient: verify the diagnostic parquet
    # on origin was actually touched within the last 3 days.
    log = run(["git", "log", "-1", "--format=%cI", "origin/main", "--",
               "dashboard_data/sg_diagnostic.parquet"], cwd=SIMS).stdout.strip()
    try:
        landed = datetime.fromisoformat(log)
        if datetime.now(timezone.utc) - landed > timedelta(days=3):
            report("FAIL", "monday grading",
                   f"workflow green but sg_diagnostic.parquet on origin last touched {log} "
                   "- output likely computed and lost on the runner (silent push failure)")
        else:
            report("PASS", "monday grading", f"green + diagnostics landed {log}")
    except ValueError:
        report("WARN", "monday grading", "could not date origin's sg_diagnostic.parquet")


def check_ci_week() -> None:
    for wf in ["nightly-round-sim.yml", "midweek-round-automation.yml", "reprice.yml"]:
        r = run(["gh", "run", "list", "--workflow", wf, "--limit", "7",
                 "--json", "conclusion"], cwd=SIMS, timeout=60)
        try:
            runs = json.loads(r.stdout)
        except ValueError:
            report("SKIP", f"CI {wf}", "unavailable")
            continue
        bad = [x for x in runs if x["conclusion"] not in ("success", "skipped", None)]
        if bad:
            report("FAIL", f"CI {wf}", f"{len(bad)}/{len(runs)} recent runs failed/cancelled")
        else:
            report("PASS", f"CI {wf}", f"{len(runs)} recent runs clean")
    r = run(["gh", "run", "list", "--workflow", "board.yml", "--limit", "7",
             "--json", "conclusion", "-R", "mslade50/golf_scraping"], timeout=60)
    try:
        runs = json.loads(r.stdout)
        bad = [x for x in runs if x["conclusion"] not in ("success", "skipped", None)]
        report("FAIL" if bad else "PASS", "CI golf_scraping board.yml",
               f"{len(bad)}/{len(runs)} failed" if bad else f"{len(runs)} recent runs clean")
    except ValueError:
        report("SKIP", "CI golf_scraping board.yml", "unavailable")


# ----------------------------------------------------------- readiness mode

def check_stale_root_files() -> None:
    tourney, _ = _current_event()
    anchor = _mtime(SIMS / f"final_predictions_{tourney}.csv") if tourney else None
    live = ["model_predictions_r1.csv", "model_predictions_r2.csv", "model_predictions_r3.csv",
            "model_predictions_r4.csv", "r1_live_model.csv", "r2_live_model.csv",
            "r3_live_model.csv", "r4_live_model.csv", "simulated_probs_live.csv"]
    if anchor is None:
        report("WARN", "stale root files", f"no final_predictions_{tourney or '?'}.csv anchor "
               "- pre-event sim not run yet?")
        return
    stale = [f for f in live if (m := _mtime(SIMS / f)) and m < anchor]
    if stale:
        report("FAIL", "stale root files",
               f"{len(stale)} live file(s) predate the {tourney} pre-event sim: "
               f"{', '.join(stale[:4])} - purge before the live run (known poisoning trap)")
    else:
        report("PASS", "stale root files", "no live-round files predate the pre-event sim")


def check_dists_fanout() -> None:
    files = ["sg_dist_player.csv", "this_week_dists_v2.csv"]
    targets = [SIMS, ONEDRIVE, ONEDRIVE / "etr-golf-sims"]
    for f in files:
        copies = {str(t): (t / f) for t in targets if (t / f).exists()}
        if str(SIMS) not in copies:
            report("FAIL", f"dists fan-out {f}", "missing from sims_process root")
            continue
        base = copies[str(SIMS)]
        base_m = _mtime(base)
        if base_m and datetime.now(timezone.utc) - base_m > timedelta(days=8):
            report("FAIL", f"dists fan-out {f}", f"sims_process copy is {base_m:%Y-%m-%d} (stale week)")
            continue
        base_h = _md5(base)
        diverged = [t for t, p in copies.items() if t != str(SIMS) and _md5(p) != base_h]
        if diverged:
            report("WARN", f"dists fan-out {f}",
                   f"copies diverge from sims_process at: {', '.join(Path(d).name or d for d in diverged)} "
                   "(a fallback consumer would use a different week)")
        else:
            report("PASS", f"dists fan-out {f}",
                   f"{len(copies)}/3 targets present, all hash-identical, {base_m:%Y-%m-%d}")


def check_coeffs_cache() -> None:
    m = _mtime(SIMS / "coeffs_cache.json")
    if m is None:
        report("FAIL", "coeffs_cache.json", "missing")
    elif datetime.now(timezone.utc) - m > timedelta(days=8):
        report("WARN", "coeffs_cache.json", f"last refreshed {m:%Y-%m-%d} - no successful "
               "sheet fetch in over a week")
    else:
        report("PASS", "coeffs_cache.json", f"fresh ({m:%Y-%m-%d})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["monday", "readiness", "all"], default=None)
    args = parser.parse_args()
    mode = args.mode or ("monday" if datetime.now().weekday() == 0 else "readiness")

    print(f"\nweekly-repo-check deterministic checks - mode: {mode}")
    print("=" * 60)
    check_git_state()
    check_r2_publication()
    check_scheduled_tasks()
    check_kernel()
    check_wheels_release()
    check_odds_feed()
    if mode in ("monday", "all"):
        check_grading_landed()
        check_ci_week()
    if mode in ("readiness", "all"):
        check_stale_root_files()
        check_dists_fanout()
        check_coeffs_cache()

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("=" * 60)
    print(f"  {len(RESULTS)} checks: {len(fails)} FAIL, {len(warns)} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

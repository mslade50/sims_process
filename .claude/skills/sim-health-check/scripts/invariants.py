"""Deterministic numeric checks for the weekly sim health audit.

Pulls the week's artifacts from origin/main (dashboard_data copies committed by
the automation) with local-file fallback, and verifies the invariants that have
historically broken. Prints one PASS/FAIL/SKIP line per check; exits nonzero if
anything FAILed so the calling skill can alert.

Run from the repo root: python .claude/skills/sim-health-check/scripts/invariants.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
FAILS: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    print(f"  [{status:4s}] {name}" + (f" - {detail}" if detail else ""))
    if status == "FAIL":
        FAILS.append(f"{name}: {detail}")


def from_main(path: str) -> pd.DataFrame | None:
    """dashboard_data copy on origin/main first (what automation committed),
    then the local file."""
    try:
        raw = subprocess.run(
            ["git", "show", f"origin/main:dashboard_data/{path}"],
            capture_output=True, cwd=REPO, timeout=30,
        )
        if raw.returncode == 0:
            return pd.read_csv(io.BytesIO(raw.stdout))
    except Exception:
        pass
    local = REPO / path
    if local.exists():
        return pd.read_csv(local)
    return None


def check_post_pre_invariant() -> None:
    specs = [
        ("r2_live_model.csv", "updated_pred", "updated_pred_r3"),
        ("r3_live_model.csv", "updated_pred_r3", "updated_pred_r4"),
    ]
    for fname, pre, post in specs:
        df = from_main(fname)
        if df is None or not {pre, post, "total_adjustment"} <= set(df.columns):
            report("SKIP", f"Post=Pre+adj ({fname})", "file or columns unavailable")
            continue
        resid = (df[post] - df[pre] - df["total_adjustment"]).abs().max()
        if resid < 1e-9:
            report("PASS", f"Post=Pre+adj ({fname})", f"max resid {resid:.1e}")
        else:
            report("FAIL", f"Post=Pre+adj ({fname})", f"max resid {resid:.3g}")


def check_r4_undo_semantics() -> None:
    """Post-b0f1058: r3_live_model must carry R3's OWN fresh adj in tot_sg_adj
    (tot_resid_adj == 0), so the R4 run undoes R3 rather than R2 twice."""
    df = from_main("r3_live_model.csv")
    if df is None:
        report("SKIP", "R4 undo semantics", "r3_live_model unavailable")
        return
    fresh_cols = [c for c in ["sg_ott_avg_adj", "sg_putt_avg_adj", "sg_app_avg_adj",
                              "sg_arg_avg_adj", "avg_great_shots_adj", "pos_6_10_adj"]
                  if c in df.columns]
    if not fresh_cols:
        report("FAIL", "R4 undo semantics", "no fresh sg_*_avg_adj columns in r3_live_model "
               "(R3 became a pure undo of R2 - check r2_live_model columns)")
        return
    if "tot_sg_adj" not in df.columns:
        report("SKIP", "R4 undo semantics", "no tot_sg_adj column")
        return
    fresh = df[fresh_cols].sum(axis=1)
    gap = (df["tot_sg_adj"].fillna(0) - fresh).abs().max()
    resid_ok = ("tot_resid_adj" not in df.columns) or (df["tot_resid_adj"].fillna(0).abs().max() < 1e-12)
    if gap < 1e-9 and resid_ok:
        report("PASS", "R4 undo semantics", "tot_sg_adj == R3 fresh adj; tot_resid_adj == 0")
    else:
        report("FAIL", "R4 undo semantics",
               f"tot_sg_adj vs fresh gap {gap:.3g}, tot_resid_adj zero: {resid_ok} "
               "(stale-undo regression - R4 would double-undo R2)")


def check_nan_skills() -> None:
    for rnd in (1, 2, 3, 4):
        fname = f"model_predictions_r{rnd}.csv"
        df = from_main(fname)
        if df is None:
            report("SKIP", f"NaN skills ({fname})")
            continue
        skill_col = "my_pred" if rnd == 1 else f"my_pred{rnd}"
        if skill_col not in df.columns:
            report("SKIP", f"NaN skills ({fname})", f"no {skill_col}")
            continue
        n_nan = int(df[skill_col].isna().sum())
        if n_nan == 0:
            report("PASS", f"NaN skills ({fname})")
        else:
            names = ", ".join(df.loc[df[skill_col].isna(), "player_name"].head(5))
            report("FAIL", f"NaN skills ({fname})",
                   f"{n_nan} players with NaN {skill_col} (merge miss?): {names}")


def check_weather_centering() -> None:
    df = from_main("model_predictions_r1.csv")
    if df is None or "dew_adj1" not in df.columns:
        report("SKIP", "weather centering", "model_predictions_r1/dew_adj1 unavailable")
        return
    dew_mean = abs(df["dew_adj1"].mean())
    dew_off = bool((df["dew_adj1"].abs() < 1e-12).all())  # tropical venue: dew slope 0
    if dew_mean < 1e-6 or dew_off:
        report("PASS", "dew mean-centered", f"|mean| {dew_mean:.2e}")
    else:
        report("FAIL", "dew mean-centered", f"|mean| {dew_mean:.4f} (should be ~0)")
    if "wind_adj1" in df.columns:
        wmean = df["wind_adj1"].mean()
        if abs(wmean) > 1e-6 or (df["wind_adj1"].abs() < 1e-12).all():
            report("PASS", "wind NOT centered", f"mean {wmean:.4f}")
        else:
            report("FAIL", "wind NOT centered", "wind_adj1 mean ~0 - was wind centered by mistake?")


def check_live_model_completeness() -> None:
    """The columns whose silent absence guts a round (audit items 8/9)."""
    df = from_main("r2_live_model.csv")
    if df is None:
        report("SKIP", "r2 sg_*_avg columns")
    else:
        missing = [c for c in ["sg_ott_avg", "sg_putt_avg", "sg_app_avg", "sg_arg_avg"]
                   if c not in df.columns]
        if missing:
            report("FAIL", "r2 sg_*_avg columns",
                   f"missing {missing} - next R3 run degrades to a pure undo of R2")
        else:
            report("PASS", "r2 sg_*_avg columns")


def check_cap_constants() -> None:
    """The fix-layer caps must agree across all implementations (CLAUDE.md)."""
    files = {
        "live_stats_engine.py": ["RESID_FIX_CAP = 6.0"],
        "round_sim.py": ["6.0", "-0.5"],
        "new_sim.py": ["6.0", "-0.5"],
        "rust/src/cascade.rs": [".min(6.0)", "-0.5"],
        "rust/src/round_cascade.rs": [".min(6.0)", "-0.5"],
        "rust/fixtures/ref_pretournament.py": ["6.0", "-0.5, 0.5"],
        "rust/fixtures/ref_remaining_rounds.py": ["6.0", "-0.5, 0.5"],
    }
    bad = []
    for rel, needles in files.items():
        p = REPO / rel
        if not p.exists():
            bad.append(f"{rel} missing")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle not in text:
                bad.append(f"{rel}: '{needle}' not found")
    if bad:
        report("FAIL", "fix-layer cap constants", "; ".join(bad))
    else:
        report("PASS", "fix-layer cap constants", "all 7 implementations carry the caps")


def check_find_pred_col() -> None:
    text = (REPO / "round_sim.py").read_text(encoding="utf-8", errors="replace")
    start = text.find("def find_pred_col")
    block = text[start:start + 700] if start >= 0 else ""
    my_i, sc_i = block.find("my_pred"), block.find("scores_r")
    if start >= 0 and 0 <= my_i < sc_i:
        report("PASS", "find_pred_col ordering", "absolute my_pred resolves first")
    else:
        report("FAIL", "find_pred_col ordering",
               "scores_r{N} (centered advantage) resolves before my_pred - bet gates loosen")


def check_kernel_sync() -> None:
    try:
        import sims_kernel
        ok = sims_kernel.selftest()
        pyd = Path(sims_kernel.__file__).with_name("sims_kernel.pyd")
        pyd_m = pyd.stat().st_mtime if pyd.exists() else 0
        last = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "origin/main", "--", "rust/src/"],
            capture_output=True, text=True, cwd=REPO, timeout=30,
        ).stdout.strip()
        stale = bool(last) and pyd_m and (pyd_m + 120 < int(last))
        ver = sims_kernel.version() if hasattr(sims_kernel, "version") else "?"
        if ok and not stale:
            report("PASS", "local Rust kernel", f"v{ver}, selftest True, newer than rust/src")
        elif not ok:
            report("FAIL", "local Rust kernel", "selftest returned False")
        else:
            report("FAIL", "local Rust kernel",
                   f"pyd predates last rust/src commit - rebuild per CLAUDE.md ritual (v{ver})")
    except ImportError:
        report("FAIL", "local Rust kernel", "sims_kernel not importable - sims run Python fallback")


def check_wheel_pins() -> None:
    prebuilt = sorted((REPO / "rust" / "prebuilt").glob("sims_kernel-*-win_amd64.whl"))
    if not prebuilt:
        report("SKIP", "workflow wheel pins", "no prebuilt wheels found")
        return
    latest = prebuilt[-1].name
    bad = []
    for wf in (REPO / ".github" / "workflows").glob("*.yml"):
        text = wf.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "sims_kernel-" in line and ".whl" in line and latest not in line:
                bad.append(wf.name)
                break
    if bad:
        report("FAIL", "workflow wheel pins", f"not pinned to {latest}: {sorted(set(bad))}")
    else:
        report("PASS", "workflow wheel pins", f"all workflows install {latest}")


def check_coeffs_cache() -> None:
    p = REPO / "coeffs_cache.json"
    if not p.exists():
        report("SKIP", "coefficients cache")
        return
    import time
    age_days = (time.time() - p.stat().st_mtime) / 86400
    d = json.loads(p.read_text())
    n = len([k for k in d if k.startswith("coefficients")])
    if age_days <= 8:
        report("PASS", "coefficients cache", f"{n} dicts, refreshed {age_days:.1f}d ago")
    else:
        report("FAIL", "coefficients cache",
               f"{age_days:.0f} days old - has the pipeline fetched the sheet this week?")
    # in-domain containment of the fitted fix-layer polys under the prod caps
    worst = 0.0
    r = np.linspace(-8, 8, 65)
    for name in ("coefficients_r1_high", "coefficients_r1_midh",
                 "coefficients_r1_midl", "coefficients_r1_low"):
        c = d.get(name, {})
        adj = r * c.get("residual", 0) + r**2 * c.get("residual2", 0)
        capped = np.maximum(np.minimum(np.where(r < 0, np.minimum(adj, 0.2), adj), 0.5), -0.5)
        worst = max(worst, float(np.abs(capped).max()))
    r2dom = np.linspace(-8, 6, 57)
    for name in ("coefficients_r2", "coefficients_r2_6_30", "coefficients_r2_30_up"):
        c = d.get(name, {})
        adj = (r2dom * c.get("residual", 0) + r2dom**2 * c.get("residual2", 0)
               + r2dom**3 * c.get("residual3", 0))
        worst = max(worst, float(np.abs(np.clip(adj, -0.5, 0.5)).max()))
    if worst <= 0.5 + 1e-9:
        report("PASS", "fix-layer containment", f"max |capped adj| {worst:.3f} <= 0.5")
    else:
        report("FAIL", "fix-layer containment", f"max |capped adj| {worst:.3f} > 0.5")


def main() -> int:
    subprocess.run(["git", "fetch", "origin", "main", "-q"], cwd=REPO, timeout=60)
    print("== sim health: numeric invariants ==")
    check_post_pre_invariant()
    check_r4_undo_semantics()
    check_nan_skills()
    check_weather_centering()
    check_live_model_completeness()
    check_cap_constants()
    check_find_pred_col()
    check_kernel_sync()
    check_wheel_pins()
    check_coeffs_cache()
    print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  * {f}")
        return 1
    print("RESULT: ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

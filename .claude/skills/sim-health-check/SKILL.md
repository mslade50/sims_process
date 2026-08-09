---
name: sim-health-check
description: Weekly Monday-night audit of the round-to-round sim pipeline — CI runs, regression coefficients, fix-layer cap consistency, kernel sync, and skill-update invariants on the week's live models. Run after Monday grading; alerts via Telegram on failures.
---

# Weekly sim health check

You are auditing last week's round-to-round sim pipeline for the golf Monte
Carlo system. The goal is to catch silent drift and invariant breaks while the
week's artifacts are still fresh — before `init_weekly.py` overwrites them.
Work from the repo root. Be skeptical: a check that can't run is a finding, not
a pass. This process found real bugs in Aug 2026 (R4 stale undo, stale 0.1.0 CI
kernel wheel, cap-less parity oracles, centered-pred bet gates) — assume the
same classes of failure can recur.

## 1. Deterministic invariants (run first)

```
python .claude/skills/sim-health-check/scripts/invariants.py
```

This fetches the week's committed artifacts from `origin/main:dashboard_data/`
and verifies: Post = Pre + total_adjustment (R2/R3, machine precision), R4 undo
semantics (`tot_sg_adj` in r3_live_model = R3's own fresh adj, not R2's), NaN
skills shipped into model_predictions files, dew centered / wind not, the
sg_*_avg columns whose silent absence guts R3, fix-layer cap constants across
all 7 implementations, `find_pred_col` ordering (absolute `my_pred` first),
local kernel selftest + freshness vs `rust/src`, workflow wheel pins vs the
latest `rust/prebuilt` wheel, and coefficient-cache freshness + cap containment.

Read every FAIL and SKIP line. A SKIP for a file that should exist this week
(e.g. no r3_live_model after a full event) is itself a red flag — investigate.

## 2. CI week in review

For each of `nightly-round-sim.yml`, `midweek-round-automation.yml`,
`monday-grading.yml`, `reprice.yml`:

```
gh run list --workflow <name> --limit 7 --json displayTitle,createdAt,conclusion
```

- Any `failure`/`cancelled` conclusions → pull the log, find the step and cause.
- For the most recent nightly run, grep the log for `sims_kernel` and `[rust]`:
  the install step must print the CURRENT kernel version (match
  `rust/prebuilt/`'s latest, e.g. `sims_kernel 0.2.0 True`), and there must be
  no `falling back to Python cascade` warnings. A wrong version here means CI
  is simulating with stale kernel logic even though everything "succeeds".

## 3. Regression model currency

- `git log -2 --format='%h %ad %s' --date=short -- coeffs_cache.json` and diff
  the cache against the previous committed version. Coefficient changes are
  fine if the user refit intentionally — flag them in the report so drift is
  never silent. New dict keys must resolve through `COL_MAPS` in
  `live_stats_engine.py` (sheet terms are free text; an unmapped term is
  silently skipped — that's audit item 9).
- Sanity-check shape: in the R2 top bucket the cubic should mean-revert leaders
  (negative adj at resid +6); the `r1_high` bucket's convex quadratic goes
  positive for large negative residuals and is contained ONLY by the
  0.2-when-residual<0 cap — if that cap rule ever changes, blow-ups get
  rewarded.

## 4. Doc and parity drift

- Compare CLAUDE.md's "Fix-layer residual caps" bullet against the code. The
  spec has gone stale before (documented a retired -0.75 band for a month).
  Also confirm comments near the caps (`live_stats_engine.py` `_totals_r1`,
  `cascade.rs`, `round_cascade.rs`) still describe what the code does.
- If anything under `rust/src/` changed this week: rebuild/reinstall the local
  kernel per the CLAUDE.md ritual, confirm CI wheel pins point at a wheel built
  AFTER the change, and run `rust/fixtures/test_cascade_parity.py` and
  `test_round_cascade_parity.py` (script-style, run with python directly; ~2
  min). The parity ORACLES (`ref_pretournament.py`, `ref_remaining_rounds.py`)
  must be updated in the same commit as any cascade change — check they were.

## 5. Live/sim parity gaps (known, structural)

Two engines compute skill updates: `live_stats_engine.py` (live truth) and the
sim cascades (round_sim/new_sim/Rust). Known divergence to re-check hasn't
widened: the live engine strips R1's residual at R2 (`r1_resid_undo`, horizon
regression 2026-08); the cascades do not. If a new adjustment feature landed in
one side this week (pin-high, gravity, etc.), verify either it landed in both
or its absence in the sims is a deliberate, documented choice.

## 6. Dev tree hygiene

- `git status` in the dev tree: uncommitted mods to pipeline code? Branch
  behind `origin/main`? (Production runs from origin/main on the runner — dev
  tree drift means local runs test different code than production runs.)
- Stale weekly files: `ls -la model_predictions_r*.csv r*_live_model.csv` —
  mtimes spanning two different event weeks poison local live runs (known
  trap). Flag any file older than the current event's start.

## 7. Report

End with a compact PASS/FAIL table by section, failures first, each failure
with file:line and the concrete consequence (what mispricing or silent
degradation it causes). If ANY check failed and this is a scheduled headless
run, send a Telegram alert:

```
python -c "from maker_alerts import send_telegram; send_telegram('sim-health-check: <N> failure(s) — <one-line summary>')"
```

Do not auto-fix anything during the scheduled run — report only. Fixes happen
interactively with the user.

---
name: weekly-repo-check
description: Twice-weekly tri-repo operational audit (sims_process, sim_prep, golf_scraping) — Monday 11am post-event mode verifies grading/publication/CI actually landed; Wednesday 5pm readiness mode verifies the machine and data pipelines are clean for R1. Report-only; Telegram on failures.
---

# Weekly repo check

You are auditing the three golf repos (sims_process, sim_prep at
`../sim_prep`, golf_scraping at `../golf_scraping`) and their contracts.
Work from the sims_process repo root. READ-ONLY: do not fix, commit, push,
delete, rerun pipelines, or dispatch workflows — report and alert only.
Fixes happen interactively with the user.

Be skeptical, and remember the defining lesson of 2026-08-12: **a green
workflow is not proof the work landed.** The monday-grading workflow was
green for two weeks while its final push silently failed and the diagnostics
were discarded on the runner. Verify *outcomes* (files on origin, fresh
timestamps, published tips), not just run conclusions.

## 1. Deterministic checks (run first)

```
python .claude/skills/weekly-repo-check/scripts/repo_checks.py
```

Mode is inferred from the weekday (Monday → post-event, otherwise →
readiness); pass `--mode all` when run manually. Read every FAIL, WARN, and
SKIP. A SKIP for something that should exist is itself a finding.

## 2. Investigate every FAIL/WARN agentically

For each deterministic failure, find the root cause before reporting:

- **CI failures** → `gh run view <id> --log-failed`; identify the step and
  cause. For "green but nothing landed" findings, grep the full log of the
  green run for `ERROR:` — swallowed errors inside successful steps are the
  known failure class (push permission, Telegram env, exit-code swallowing).
- **Publication lag** → check whether sim_prep's Monday pipeline ran
  (`work/pipeline_logs/<newest>/pipeline_run_manifest.json`, per-stage logs)
  and whether `python -m dgdata publish` was skipped.
- **Fan-out divergence** → compare mtimes/hashes across the three targets;
  the loop in sim_prep's `cat_dists_player.py` has no per-target error
  handling, so single-target misses are silent.
- **Scheduled-task failures** → the shot-data publishers "fail silently into
  work/*.log" (sim_prep runbook): read the tail of
  `sim_prep/work/weekly_shot_export.log` and `shotdata_publish.log`.
- **Odds feed issues** → check golf_scraping board.yml recent runs and the
  known R2-upload timeout pattern (15-min job cancellations).

## 3. Deeper weekly sweeps (agentic, quick)

- **Monday mode**: confirm the graded event's rows exist in
  `origin/main:dashboard_data/sg_diagnostic.parquet` (read it, group by
  event); confirm CLV/grades hit the sheet tabs if cheap to check; scan the
  week's nightly logs for `falling back to Python cascade` and for a kernel
  version that matches the newest `rust/prebuilt`/wheels-latest build.
- **Readiness mode**: confirm `dashboard_data/.sync_manifest.json` names the
  current event and lists no live-round files older than the pre-event sim;
  confirm sim_inputs/sheet config coherence for the new event (tourney slug,
  course_par vs course_pars); confirm `pin_high_r1.csv` and
  `permanent_data/player_archetypes.csv` are current-week where applicable.
- Both: anything the 2026-08 audits flagged that could regress — cap parity
  across live engine / sims / Rust / oracles if `rust/src` or the fix layers
  changed this week (then also verify the parity oracles changed in the same
  commit).

## 4. Report + alert

End with a compact PASS/FAIL table by section, failures first, each with
file:line or run URL and the concrete consequence (what mispricing, data
loss, or silent degradation it causes). Write the same summary to
`permanent_data/weekly_repo_check_last_report.md` (overwrite; this file is
the only write this skill is allowed).

If ANY check failed and this is a scheduled headless run, send Telegram:

```
python -c "from maker_alerts import send_telegram; send_telegram('weekly-repo-check (<mode>): <N> failure(s) — <one-line summary>')"
```

If everything passed on a scheduled run, stay silent (no Telegram) — the
log at `permanent_data/weekly_repo_check_last.log` is the record.

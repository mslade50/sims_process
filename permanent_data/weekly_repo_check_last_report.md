# Weekly repo check — 2026-08-17 (Monday, post-event mode, run as --mode all)

Deterministic: 21 checks — 3 FAIL, 2 WARN. After investigation: **1 real active
failure**, 1 already-remediated failure, 1 false positive.

## FAILURES (investigated)

| # | Status | Finding |
|---|--------|---------|
| 1 | **FAIL — ACTIVE, data loss** | Monday grading green but St. Jude diagnostics lost — the silent-push failure class RECURRED despite the 8/12 fix |
| 2 | REMEDIATED | GolfShotWeeklyExport schtask exit 1 on Tue 8/11 (R2 upload timeout); manual 8/12 rerun published successfully |
| 3 | FALSE POSITIVE | "odds feed stale >30h" — last write is Sunday's designed final-odds capture; Monday 9am run fired on time but has no matchups to write post-event |

### 1. Monday grading: green run, nothing landed (RECURRENCE of 2026-08-12 class)

- Run [32038336494](https://github.com/mslade50/sims_process/actions/runs/32038336494)
  (today 14:14Z, `success`) graded all 594 St. Jude bets (in-run verification:
  0 ungraded on all four tabs — **sheet grading DID land**), computed 1,365
  st_jude diagnostic rows (39,040 total), printed "Staging and pushing to
  GitHub... Done!" — and **no commit reached origin**. origin/main tip is still
  `dac463e` (8/15); `dashboard_data/sg_diagnostic.parquet` on origin is still
  the 8/12 backfill (`aa48f42`), events end at `wyndham`, no `st_jude`.
- **Root cause**: `.github/workflows/monday-grading.yml` has no git identity —
  unlike `nightly-round-sim.yml:110-116`, whose comment documents this exact
  mode: commit dies with "Committer identity unknown" and the push is silently
  skipped. In `push_dashboard_data.py` `git_push()` (~line 313) the
  `git commit` result is **never checked**; the subsequent `git push` exits 0
  ("Everything up-to-date"), so the 8/12 hardening (`_fail` + exit-code
  propagation), which only watches the push, never fires. The 8/12 fix
  hardened the wrong step.
- **Consequence**: St. Jude sg_diagnostic rows exist only on the dead runner;
  dashboard/Render shows no St. Jude diagnostics; no Telegram fired. The
  15:00Z Monday retry cron (not yet fired as of ~15:30Z) will recompute and
  lose them identically.
- **Recovery + fix (interactive, not applied — report-only)**:
  1. Rerun `sg_diagnostic.py --event-id 27 --tourney st_jude --year 2026` locally
     against the fresh R2 snapshot + `push_dashboard_data.py` (same as 8/12 backfill).
  2. Add the `GIT_AUTHOR_*`/`GIT_COMMITTER_*` env block from
     `nightly-round-sim.yml:113-116` to the grading step in `monday-grading.yml`.
  3. In `git_push()`, check the `git commit` returncode and route nonzero
     through `_fail()` (a commit that dies must not fall through to a green push).

### 2. GolfShotWeeklyExport exit 1 (Tue 8/11) — remediated 8/12

- `sim_prep/work/weekly_shot_export.log`: `botocore.exceptions.ReadTimeoutError`
  on the R2 multipart upload of `archive_inspect.sqlite3.zst` (~1.2 GB, part 22)
  during the 8/11 09:00 scheduled run.
- Manual rerun 8/12 17:43 published export `2026-08-12T17-43-07Z` (55 files);
  dgdata tip 2026-08-12T15:44Z confirms downstream consumption. Schtask
  last-result stays 1 until the next run (Tue 8/18 09:00).
- Optional hardening: bump botocore read timeout / retries for the archive upload.

### 3. Odds feed "stale >30h" — false positive (check calibration)

- `tournament_matchups_latest.json` last_updated Sun 8/16 07:34Z = the
  **designed** Sunday 3:34am ET final-odds capture. The worker's trimmed
  schedule (`board/worker/wrangler.toml`: "Sun: 3:00am ONLY", next fire Mon
  9am ET) then goes quiet; the Monday 9am run fired on time today (13:00Z,
  green) but post-event DataGolf offers no matchups, so the file isn't
  rewritten. No outage; all Sat-night/Sun fires present in the run history.
- Suggested tweak: `repo_checks.py check_odds_feed` — widen threshold to ~36h
  or special-case Monday before ~1pm ET to kill this recurring Monday FAIL.

## WARN

- sims_process local 3 behind origin (weekend sim_fairs data commits) — pull
  when convenient.
- golf_scraping local clone 61 behind origin — stale for local dev only; this
  audit read the repo via the GitHub API.

## PASS (highlights)

- Grading itself: all 594 St. Jude bets graded on the sheet tabs, verified in-run.
- dgdata publication tip 2026-08-12T15:44Z; fan-out 3/3 hash-identical (8/10);
  coeffs cache fresh (8/15).
- Kernel: local pyd 0.2.0 selftest True; CI wheel current; nightly runner used
  sims_kernel 0.2.0 (selftest True), no "falling back to Python cascade" in the
  latest nightly log.
- CI clean: nightly-round-sim, midweek-round-automation, reprice, board.yml
  (7/7 each); pin-high + sim-health-check schtasks green.

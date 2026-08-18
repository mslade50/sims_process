# Golf Tournament Monte Carlo Simulation System

Monte Carlo simulation for golf tournament prediction and DFS (DraftKings). Combines EMA-weighted strokes gained distributions, course shape adjustments, weather effects, and live skill updates to price matchups, score cards, and finish positions.

## Pipeline Execution Order

When preparing ETR, run `hole_baselines.py`; it resolves historical editions
by physical `course_id` and exact `(year, event_id)` pairs rather than assuming
the current event ID was stable across venues.

**Pre-tournament**: `init_weekly.py` → `humidity.py` → `scoring_baseline.py` → `write_base_rates.py`. The SG distribution files (`sg_dist_player.csv`, `this_week_dists_v2.csv`) are NOT built here — sim_prep's `cat_dists_player.py` is the single producer and fans them out to this repo, the OneDrive root, and etr-golf-sims (the local copy is archived).
**Pre-event sim (two-pass)**: `new_sim.py` (first pass) → `mkt_regress.py` → `new_sim.py` (second pass with regressed preds)
**Pre-event (round=0)**: `live_stats_engine.py` → `round_sim.py` (R1 matchups + score cards)
**Live (R1-R4)**: Update Google Sheet (`round_config` tab) → `live_stats_engine.py` → `round_sim.py` (round matchups + score cards)
**Reprice (automated, triggered by scraper)**: `nightly-round-sim.yml` (sim + cache) → `reprice.yml` (load cache + fresh odds → dedup + store + Telegram alert)
**Post-event (Monday, automated)**: `monday_grading.py` → `grade_bets.py` → `sg_diagnostic.py` → `push_dashboard_data.py`
**Deploy dashboard (manual)**: `push_dashboard_data.py` → triggers Render deploy. Pipeline scripts no longer auto-push.

See `WEEKLY_PROCESS.md` for exact commands and day-by-day schedule.

## Category-First Tournament Simulation

`new_sim.py` uses category-first draws: instead of drawing total SG then decomposing into categories, it draws each SG category from a course-adjusted multivariate normal and sums to total. This captures course-specific variance profiles (e.g., Bay Hill amplifies OTT variance 1.24x but barely affects PUTT at 1.01x).

**Status**: Promoted to production (March 2025). The archived total-first sim is `archive/new_sim_v1.py`.

**Key details**:
- Uses `course_cat_mults` from the Google Sheet `round_config` tab (written by `scoring_baseline.py`)
- Category means are **re-centered to sum to `my_pred`** so category-first draws only change variance structure, not base predictions
- Weather delta distributed as 0.35 OTT, 0.35 APP, 0.15 ARG, 0.15 PUTT
- Skill update shifts are distributed evenly across 4 categories (`shift / 4.0`) to preserve course covariance structure
- Per-category course multipliers and skew are computed for the exact physical course by `scoring_baseline.py`, written to `round_config`, and consumed through `sheet_config.py`
- **Week-level form latent** (`sim_inputs.WEEK_LATENT_SD`, 2026-08): one shared draw per (player, sim) added as `+w/4` to every round's category means, with idiosyncratic category stds shrunk (`sqrt(1 − σ²/round_var)`) so per-round total variance is UNCHANGED — round matchup pricing does not reprice; only the cross-round linkage changes. Fixes the missing 72-hole dependence (sim ratio was 1.019 vs empirical 1.09–1.38; target 1.15–1.20, re-scored monthly vs the closing line within [1.09, 1.30]). Lives in BOTH `new_sim.py` and `rust/src/cascade.rs` (kernel ≥0.3.0; `run_pretournament` takes `week_latent_sd` as its last arg — an old kernel TypeErrors and falls back to Python). Drawn from a separate RNG stream: `--no-week-latent` (or sd=0) is BIT-IDENTICAL to the pre-latent cascade, which is what parity fixtures must be captured with (`SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only --use-python --no-week-latent`; pin the sha256 sidecar from the ROOT `final_scores_{t}.npy`, which is saved pre-top-up).
- **72-hole totals skew top-up** (`skew_calibration.calibrate_total_skew`, target +0.23, cut-conditional): same dither→CF→stochastic-rounding machinery as the round-level top-up, applied to totals in `new_sim.py` after the cascade and BEFORE aggregation/cache save (so `--price-only`/`--reprice` inherit it; disable `--no-skew-cal`). It is SELF-CALIBRATING (adds only the residual to the absolute target), so running it last makes the joint fit with the week latent automatic — never apply it before the latent, and never ship either alone (each alone misprices favorites ~13–19% relative). A Gaussian latent DILUTES raw skew (~0.11→0.07); that is expected and absorbed by the top-up.
- **Round-score skew calibration** (`skew_calibration.py`): summing 4 category draws CLT-washes skew to ~+0.12 vs the empirical within-player +0.26 (score space; blowups drag the mean UP — a player's median round is ~0.11 better than his mean). `round_sim.py` tops up the catfirst draws post-sim (dither → CF reshape → stochastic rounding; monotone, preserves per-player mean/std and the copula). Applied before the cache save so `--price-only`/`--reprice` inherit it; disable with `--no-skew-cal`. Do NOT instead inflate `course_cat_skew` inputs — the needed OTT ≈ −1.9 puts Cornish–Fisher past its monotone fold. Mean-calibration (expected_avg) stays the anchor; the median falls out of the shape.

**Supporting analysis files** (in `archive/`, not part of weekly pipeline):
- `archive/sg_category_variance_test.py` — Tests per-category variance decomposition across 35 PGA events
- `archive/sg_category_predictive_test.py` — Player-level predictive power test (leave-one-year-out)

## Repository Layout

- **Root**: Active pipeline scripts only (19 files) + current docs
- **`archive/`**: Dead code, analysis scripts, one-time fixes, stale docs. Nothing imports from here.
- **`permanent_data/`**: Persists across weekly cleanup. Ledger, diagnostics, course data.
- **`dashboard/`**: Dash app (pages, components, data layer, assets)
- **`dashboard_data/`**: Committed copies for Render deploy

## Key Files Not Obvious From Code

- **`sim_inputs.py`**: Tracked in git (despite the name suggesting local-only). ALL tournament config: `tourney`, `course_id`, `course_par`, `event_ids`, weather arrays (`wind_1`-`wind_4`, `dewpoint_1`-`dewpoint_4`), coefficients per round/bucket, `score_adj_r{N}`, `name_replacements`. Read this first when debugging anything.
- **`sheet_config.py`**: Reads round number + weather from Google Sheet (`golf_sims` → `round_config` tab) so config can be updated from phone.
- **`api_utils.py`**: All DataGolf API calls live here. New API calls must go through this file.
- **`sheets_storage.py`**: Single auth source. `get_spreadsheet()` caches at module level. All `store_*` functions accept `spreadsheet=None`. Never duplicate auth code elsewhere.
- **`permanent_data/`**: Survives weekly GitHub Actions cleanup. Contains `bet_ledger.parquet`, `sg_diagnostic.parquet`, correlation matrices, `wind_test.csv`.
- **`dg_historical.db`**: SQLite at `~/OneDrive/dg_historical.db` (shared across repos). Used by `sg_diagnostic.py` for archetypes.

## Critical Invariants — Do Not Break

### 1. Weather centering direction
```python
# CORRECT: positive = player faces worse conditions
dew_adj = player_dew_raw - field_avg_dew    # individual - mean
# WRONG (past bug): mean - individual
```
Dewpoint is mean-centered; wind is NOT centered. Wind is absolute: `avg_wind * coefficient`.

### 2. Skill update formula
`Post = Pre + total_adjustment` must hold across ALL rounds.

**Reset-to-base form (2026-08)**: R2/R3/R4 rebuild `Post = base_pred + fresh_adj`, where `base_pred = pred + pin_high_adj + gravity_adj` (the only adjustments that persist across rounds) is written at R1 and carried through every live model. `total_adjustment = Post − Pre` is DERIVED, so the invariant holds by construction — there is no carried-column undo to go stale. If a prior live model lacks `base_pred` (pre-refactor file), the engine reconstructs it via the old undo identity (`Pre − tot_sg_adj − tot_resid_adj`). The sim cascades (round_sim/new_sim/Rust kernel) still replay the algebraically-equivalent incremental form internally.

**Prediction column flow**:
| Round | Pre | Post |
|-------|-----|------|
| R1 | `pred` | `updated_pred` |
| R2 | `updated_pred` | `updated_pred_r3` |
| R3 | `updated_pred_r3` | `updated_pred_r4` |
| R4 | `updated_pred_r4` | `updated_pred_final` |

### 3. `sg_total_adj` is RAW DATA, not an adjustment
The name is misleading. When summing R2 adjustment components, explicitly list columns — never wildcard `*_adj` or you'll catch this.

### 4. `scores_r{N}` is strokes gained, not absolute score
Convert: `expected_score = course_par - scores_r{N}`. A player with `scores_r2 = 0.83` at par 72 → expected 71.17.

### 5. Player names always lowercase
Both sides of every join/merge must go through `name_replacements`. The `cat_dists_player.py` pattern (now in sim_prep; archived copy in `archive/`) is canonical.

## Traps That Have Caused Bugs

- **Course par lives in TWO places and both must match**: the sheet's `course_pars` row (used by `round_sim.py` — its `PAR` derives from `course_pars[0]`, falling back to 72) and `sim_inputs.course_par` (used by `live_stats_engine.py` for score/SG conversion and by `scoring_baseline.py`). A par change (e.g. rocket 2026 went 72→70) with a stale value shifts every absolute score line by the difference (2026-07 bug: R1 actuals and score cards ran 2 strokes high).
- **The sims' name contract is lowercase `'last, first'`** (e.g. `'scheffler, scottie'`) — dg_historical.db's `'Last, First'` just needs lowercasing + `name_replacements` (`_normalize_db_names()` in `sg_diagnostic.py`). Do NOT flip to `'first last'`; a well-meaning flip broke archetype matching entirely (2026-07). `clv.py` is the exception — DataGolf archive odds use first-last there. Also: `sg_diagnostic` resolves the db lazily — `DG_HISTORICAL_DB` env override, then the latest R2 snapshot via `dgdata_fetch.cached_snapshot_path()` (canonical, freshest; shares sim_prep's `%LOCALAPPDATA%\etr-golf\cache`), then the OneDrive copy offline. The OneDrive copy can silently stop syncing (went stale 2026-07); R2 is the source of truth.
- **Matchup API returns string when no odds available**: Check `isinstance(match_list, list)` before iterating. The API returns a message string like `"No tournament_matchups being offered right now."` instead of a list.
- **DataGolf field-updates tee times are nested**: The API returns `teetimes` as a list of dicts (with `round_num`, `teetime`, `course_code`), NOT flat `r1_teetime` columns. `fetch_field_updates()` in `api_utils.py` handles parsing.
- **Multi-course ShotLink gaps**: Not all courses have ShotLink data. Use `.fillna(0)` for adjustment columns.
- **Wind array index 0 = 6 AM**. Wind calculation uses 5-hour window average with minute-level interpolation.
- **Wind coefficient blending** (when `wind_override == 0`): `course_wind_effect * 0.4 + baseline_wind * 0.6`. Course effect from `permanent_data/wind_test.csv`.
- **Dew coefficient is per-course**: `humidity.py` writes `dew_calculation` to the sheet via `compute_dew_factor()` (`api_utils.py`) — an EB-shrunk course slope from `permanent_data/dew_test.csv` (clamped [−0.06, 0]; tropical venues land at 0 = dew off), falling back to the sim_inputs blend (−0.0221) for uncovered courses. Regenerate the CSV yearly with `archive/dew_course_effects.py` (needs `dg_historical.db`).
- **Bayesian wind blending**: Forecast wind arrays are blended with a climatological prior (monthly hourly avg from Open-Meteo archive, 2019-2025). Climo weight = `lead_days / 12`, clamped [5%, 50%]. Round dates are Thu–Sun of current week via `get_round_dates()`. Applied in `new_sim.py` and `live_stats_engine.py`. Functions in `api_utils.py`.
- **Multi-model AI wind blend**: `humidity.py` wind values (written to the sheet's R1-R4 wind cols) are a mean of ECMWF IFS + AIFS + NOAA AIGFS via `fetch_multimodel_wind()` in `api_utils.py`, with per-hour fallback to best_match. AIFS = the model behind Windy's "AI" layer. The blend fetch MUST use the same `timezone` param as the joined forecast call (`America/New_York`) or hour keys silently misalign. Climo prior blending still applies downstream.
- **Fix-layer residual caps (2026-07)**: R1 layer — capped at 0.2 if raw residual is negative, hard cap 0.5, floored at -0.75 for residual in [-8,-6) and -0.5 elsewhere (banded floor is intentionally non-monotonic; backtest-driven). R2 layer — residual INPUT capped at +6 before the cubic terms, and `tot_resid_adj` floored at -0.5 (which must reach `total_adjustment` — summing raw components bypasses it; that was a bug). Same caps live in FOUR places and must stay in sync: `live_stats_engine.py`, `round_sim.py`, `new_sim.py`, and the Rust kernel (`rust/src/cascade.rs` + `round_cascade.rs`).
- **Tee time parsing**: Multiple formats in the wild (`%Y-%m-%d %H:%M`, `%I:%M%p`, `%m/%d/%Y %H:%M`). New formats cause `ValueError`.
- **Scraped odds loading**: `odds_loader.py`, `price_kalshi_outrights()`, and `load_score_lines()` all fetch from `mslade50/golf_scraping` GitHub API first, fall back to `permanent_data/scraped_odds/` local files. Any new scraped odds loader must follow this pattern — local-only paths break on machines without the scraping repo cloned.
- **Kalshi bid=0 lines are phantom**: Kalshi markets with `bid=0` have no real liquidity. These are filtered out in `price_kalshi_outrights()`. Without this filter, 1-cent ask prices generate fake 30%+ edges.
- **Nightly sim cache path**: `actions/cache` uses the path as part of cache identity. The save path must exactly match the restore path (e.g., `valspar/` not `./valspar/`).

## Bet Storage Architecture

- **Dual-write**: Every `store_*` call writes to Google Sheets AND `permanent_data/bet_ledger.parquet` via `_append_to_ledger()`.
- **Dedup key**: `(event_id, bet_type, round, bet_on, opponent, bookmaker)` — all lowercased/stripped. First write wins.
- **Atomic writes**: `tempfile.mkstemp()` + `os.replace()` to prevent corruption.
- **Grading**: `grade_bets.py` writes grades back to the individual source tabs (Tournament Matchups, Round Matchups, Finish Positions).
- **CLV**: `clv.py` annotates `open_odds`/`close_odds`/`tot_clv`/`clv`/`clv_book` on those same tabs + the ledger from DataGolf's historical-odds archive (called by `grade_bets.py` post-grading; re-run/backfill via `python clv.py --backfill --write`). Scope: matchups + pre-tourney finish bets at DG-covered books only — no exchanges (kalshi/novig), no Live tab. CLV convention: `implied(close) - implied(bet)` in pp, positive = beat the close.
- **Finish position sizing**: Uses kelly-stake-based units (3-43+), NOT flat 1.0. `units_wagered` derived from `kelly_stake`.
- **Auth is DRY**: `grade_bets.py` imports `get_spreadsheet` from `sheets_storage`. Never duplicate auth.

## Dashboard Rules

- **Stack**: Dash 2.14+ / dash-bootstrap-components 2.0 / dash-ag-grid / SLATE dark theme
- **Run**: `python -m dashboard.app` → `localhost:8050`. Deployed on Render via `Procfile`.
- **Data source**: Google Sheets (primary) for betting data. Local CSV/Parquet only for outrights, pricer, skill, diagnostics.
- **Caching**: 5-min in-memory cache for all Sheets reads. Google rate limit: 60 reads/min.
- **Absolute imports only**: `from dashboard.data_layer import ...` — Dash `use_pages=True` breaks relative imports.
- **dbc 2.0**: Use `color="dark"` not `dark=True` for `dbc.Table`.
- **SG diagnostic categories**: `ott`, `app`, `arg`, `putt` — NOT `sg_ott`, etc.
- **Performance data**: Read from individual tabs (Tournament MU, Round MU, Finish Pos), NEVER filtered tabs.
- **ID prefixes**: Outrights pre `outpre-`, outrights live `outlive-`, fragility `frag-`.
- **All pages must handle empty data gracefully** (show alerts, not crash).

## Rust Kernel (sims_kernel) Updates

The sims default to a compiled Rust kernel installed as
`site-packages/sims_kernel/sims_kernel.pyd`. **`git pull` does NOT update it** —
after pulling any change to `rust/src/`, the kernel must be rebuilt or the sims
silently keep running old logic while the Python fallbacks run new logic
(this exact drift shipped a stale 0.1.0 kernel once).

Update ritual (maturin is typically NOT installed; use cargo directly):
1. `cd rust && cargo test --release` (expect all green)
2. `cargo build --release --features pyo3/extension-module`
3. Back up then overwrite: copy `rust/target/release/sims_kernel.dll` over
   `<python site-packages>/sims_kernel/sims_kernel.pyd`
   (find it via `python -c "import sims_kernel; print(sims_kernel.__file__)"`)
4. Verify: `python -c "import sims_kernel; print(sims_kernel.selftest())"` → True

No Rust toolchain on this machine? The `.pyd` is an abi3 Windows-x64 binary —
copying the freshly built dll/pyd from a machine that has built it is equivalent.
Frozen parity fixtures predate code changes by definition; regenerate them after
kernel changes instead of chasing phantom diffs.

## Local Setup Quirks

- **OneDrive truncates source files.** The repo lives under OneDrive and OneDrive has emptied tracked `.py` files (notably `new_sim.py`) more than once — the file stays in place but contents go to 0 bytes. The pre-commit hook at `.githooks/pre-commit` refuses to commit a guarded file (`.py`, `.md`, `.yml`, `.yaml`, `.json`, `.toml`, `.sh`, `.sql`) that was non-empty at HEAD and is now 0 bytes. New clones must run once: `git config core.hooksPath .githooks`. Recovery for a wiped file: `git checkout HEAD -- <file>`.

## Environment

- `.env`: `DATAGOLF_API_KEY`, `EMAIL_USER`, `EMAIL_PASSWORD`, `GMAIL_APP_PASSWORD`
- `credentials.json`: Google service account (gitignored)
- GitHub secrets: `DATAGOLF_API_KEY`, `EMAIL_USER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENTS`, `GOOGLE_CREDS_JSON`
- File sync targets: `C:\Users\mckin\OneDrive\sims_process` and `C:\Users\mckin\OneDrive\etr-golf-sims`

## Code Modification Rules

- New API calls → `api_utils.py`
- New `store_*` functions → accept `spreadsheet=None`, write to Parquet ledger too
- Auth code → `sheets_storage.py` only
- Preserve CSV-based pipeline (not database-first)
- Graceful degradation for missing files (warn, don't crash)
- Follow existing print-statement debugging patterns
- Email reports → HTML tables with confidence thresholds

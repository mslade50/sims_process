# Golf Tournament Monte Carlo Simulation System

> **Purpose**: Complete context document for Claude Code. This captures architecture, data flows, key patterns, hard-won debugging lessons, and operational knowledge from the full development history of this project.

---

## Project Overview

A Monte Carlo simulation system for golf tournament prediction and DFS (DraftKings) analysis. The system generates player-specific score distributions by combining:
- Historical strokes gained (SG) data with EMA weighting
- Course-specific shape adjustments (mean, variance, tail weight)
- Real-time weather effects (wind speed, dewpoint)
- Player variance profiles
- Position-based skill updates between rounds

The system supports both **pre-tournament prediction** (R1 hole-by-hole simulation) and **live in-tournament updates** (skill adjustment + matchup/score pricing for R2-R4).

---

## File Inventory & Responsibilities

### Core Pre-Tournament Pipeline (execution order)

| Step | File | Purpose | Key Outputs |
|------|------|---------|-------------|
| 1 | `cat_dists_player.py` | Build EMA-weighted SG distributions from historical data | `sg_dist_player.csv` |
| 2 | `dists_thiswk.py` | Filter to field + apply course shape adjustments | `this_week_dists.csv`, `this_week_dists_adjusted.csv` |
| 3 | `rd_1_sd_multicourse_sim.py` | Run hole-by-hole Monte Carlo for Round 1 | `rd_1_results_sd_{tourney}.csv`, `hole_lvl_sd_1_{tourney}.csv` |

### Live Tournament Engine

| File | Purpose | Key Outputs |
|------|---------|-------------|
| `live_stats_engine.py` | Unified processor for rounds 0-4. Skill updates + next-round predictions. Replaces the old separate `live_stats.py`, `live_stats_r2.py`, `live_stats_r3.py`, `live_stats_r4.py` files. | `r{N}_live_model.csv`, `model_predictions_r{N+1}.csv` |
| `round_sim.py` | Matchup pricing, score card generation, edge calculation for R2-R4. Uses normal distribution simulation (not hole-by-hole). | `matchups_r{N}.csv`, `fair_card_r{N}.csv`, Excel workbook |
| `api_utils.py` | Shared DataGolf API functions: `fetch_live_stats()`, `fetch_field_updates()`, `calculate_average_wind()`, `compute_wind_factor()`, `clean_names()` |
| `sheet_config.py` | Google Sheets config reader. Reads round number, weather forecasts, scoring adjustments from `golf_sims` sheet (`round_config` tab) so you can update from phone. |

### Bet Storage & Analysis

| File | Purpose | Key Outputs |
|------|---------|-------------|
| `sheets_storage.py` | Google Sheets write + local Parquet ledger. Connection-pooled (`get_spreadsheet()`). All `store_*` calls auto-write to both Sheets and `permanent_data/bet_ledger.parquet`. | Sheets tabs, `bet_ledger.parquet` |
| `grade_bets.py` | Reads ungraded bets from Sheets, fetches results from DataGolf, grades with dead-heat adjustments, writes back. Also updates Parquet ledger grades via `update_ledger_grades()`. | Graded Sheets tabs, summary email, ledger updates |
| `bet_query.py` | CLI for querying local Parquet ledger. Modes: terminal summary, `--export` CSV, `--plot` Plotly dashboard. Filters: `--event`, `--type`, `--book`, `--min-edge`, `--graded`. | Terminal output, CSV, HTML dashboard |

### Configuration & Infrastructure

| File | Purpose |
|------|---------|
| `sim_inputs.py` | Tournament-specific config (NOT in repo — local only). All tournament variables, coefficients, weather arrays, name replacements. |
| `credentials.json` | Google service account key (NOT in repo — .gitignored) |
| `.github/workflows/run-sim.yml` | GitHub Actions workflow for running round_sim.py |
| `.github/workflows/test-env.yml` | Tests environment secrets are configured correctly |
| `.github/workflows/weekly-cleanup.yml` | Runs Sunday midnight UTC — deletes transient CSVs, Excel files, tournament folders from repo root |
| `requirements.txt` | Python dependencies for GitHub Actions |
| `permanent_data/` | Reference data that survives weekly cleanup (correlation matrices, calibration data, bet ledger) |

---

## Data Flow Diagram

```
                     ╔══════════════════════════════════════╗
                     ║     PRE-TOURNAMENT PIPELINE          ║
                     ╚══════════════════════════════════════╝

  field_adjusted_sg.csv (PGA rounds ≥2019, sg_*_adj columns)
          │
          ▼
  cat_dists_player.py ──► sg_dist_player.csv
          │                    │
          │              field_updates.csv
          │                    │
          ▼                    ▼
  dists_thiswk.py (Part 1: field filter)
          │
          │◄── course_shape_adjustments_{course_id}.csv
          ▼
  dists_thiswk.py (Part 2: course adjustments)
          │
          ▼
  this_week_dists_adjusted.csv
          │
          │◄── adj_hole_dist_{tourney}_{course_id}.csv
          │◄── final_predictions_{tourney}.csv
          │◄── pre_sim_summary_{tourney}.csv
          │◄── wind_test.csv (historical course wind effect)
          │◄── sim_inputs.py (weather arrays, coefficients)
          ▼
  rd_1_sd_multicourse_sim.py
          │
          ├──► rd_1_results_sd_{tourney}.csv    (aggregated player stats)
          ├──► hole_lvl_sd_1_{tourney}.csv      (per-sim, per-hole scores)
          └──► model_predictions_r1.csv


                     ╔══════════════════════════════════════╗
                     ║     LIVE TOURNAMENT FLOW             ║
                     ╚══════════════════════════════════════╝

  Google Sheet (round_config tab) ──► sheet_config.py ──► live_stats_engine.py
                                                                │
                              ┌──────────────────┬──────────────┤
                              ▼                  ▼              ▼
                    r{N}_live_model.csv   model_predictions_r{N+1}.csv
                                                │
                                                ▼
                                         round_sim.py
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                    matchups_r{N}.csv   fair_card_r{N}.csv   Excel workbook
                              │
                              ▼
                     Email report (HTML tables + attachments)


                     ╔══════════════════════════════════════╗
                     ║     BET STORAGE & GRADING            ║
                     ╚══════════════════════════════════════╝

  new_sim.py / round_sim.py
          │
          ▼
  sheets_storage.py ──► Google Sheets tabs (Tournament MU, Finish Pos, etc.)
          │
          └──► permanent_data/bet_ledger.parquet  (local Parquet write-through)
                              │
                              ▼
  grade_bets.py ──► Sheets grades + update_ledger_grades() ──► ledger updated
                              │
                              ▼
  bet_query.py ──► Terminal summary / CSV export / Plotly dashboard
```

---

## Key Algorithms & Patterns

### 1. EMA Weighting (`cat_dists_player.py`, lines 78-93)
```python
alpha = 2.0 / (span + 1.0)   # span=50 → alpha≈0.039
weights = (1-alpha)^(n-1-i)   # i=0 oldest, i=n-1 newest
```
Effective half-life ≈ 17 rounds. Minimum 20 observations per category.

### 2. Course Shape Adjustments (`dists_thiswk.py`, Part 2)
Three transforms applied per SG category:
- **`delta_mu`**: Mean shift (e.g., course rewards driving → shift sg_ott up)
- **`sigma_ratio`**: Variance scaling (e.g., course amplifies putting variance)
- **`tail_ratio`**: Kurtosis adjustment → mapped to Student-t degrees of freedom

Reference mode: `vs_tour` (default). Sanity checks flag extreme transforms.

### 3. Weather Adjustment Pattern (`rd_1_sd_multicourse_sim.py`)
```python
# Wind: absolute adjustment to score
wind_adj = avg_wind_during_round * wind_coefficient

# Dewpoint: CENTERED (individual - mean)
dew_adj = player_dew_raw - field_avg_dew  # Relative advantage/disadvantage
```
**CRITICAL**: Dewpoint is always mean-centered (`individual - mean`); wind is NOT centered.
**WRONG**: `mean - individual`. This was a past bug. The correct pattern gives positive values to players facing worse-than-average conditions.

### 4. Distribution Adjustment (`rd_1_sd_multicourse_sim.py`, lines 14-48)
For each hole, player distributions are warped by:
1. **Mean shift**: `adj_mean + wind_adj*(par/course_par) + dew_adj*(par/course_par) - skill_adj*(par/course_par) + score_adj*(par/course_par)`
2. **Variance scaling**: `player_std / (field_avg_std)` with amplification factor `player_var`
Uses `scipy.optimize.minimize` (L-BFGS-B) to find scaling factors that hit target mean while preserving distribution shape.

### 5. Wind Calculation (`calculate_average_wind`, line 51-82)
- Takes tee time string (multiple formats supported: `%Y-%m-%d %H:%M`, `%I:%M%p`, `%m/%d/%Y %H:%M`)
- Calculates 5-hour window average using minute-level interpolation from hourly wind array
- Wind array index 0 = 6 AM

### 6. Live Stats Skill Update (`live_stats_engine.py`)
**Round-specific coefficient buckets:**
- **R1**: 4 skill-based buckets (high/mid-high/mid-low/low based on pre-tournament prediction)
- **R2**: 3 position-based buckets (top 5 / 6-30 / 30+)
- **R3/R4**: 3 position-based buckets, SG-only adjustments

**Adjustment components vary by round:**
- R1: `ott_adj + putt_adj + tot_resid_adj` (residual capped at 0.5, with special cap at 0.2 if residual < 0)
- R2: `residual_adj + residual2_adj + residual3_adj + avg_ott_adj + avg_putt_adj + avg_app_adj + avg_arg_adj + delta_app_adj`
- R3/R4: SG-only; UNDO prior round's adjustments then apply fresh ones

**CRITICAL FORMULA**: `Post = Pre + total_adjustment` must hold across ALL rounds. For R3/R4 this means `total_adjustment = fresh_adj - prior_sg - prior_resid`.

**Prediction column flow:**
| Round | Pre-column | Post-column |
|-------|-----------|-------------|
| R1 | `pred` | `updated_pred` |
| R2 | `updated_pred` | `updated_pred_r3` |
| R3 | `updated_pred_r3` | `updated_pred_r4` |
| R4 | `updated_pred_r4` | `updated_pred_final` |

### 7. Multi-Course Handling
Tournaments like AT&T Pebble Beach use multiple courses. Key patterns:
- `course_id_1`, `course_id_2` in `sim_inputs.py`
- R1 simulation runs per course_id, merges at end (lines 491-550 of rd_1_sd_multicourse_sim.py)
- Live stats engine maps `expected_score_1/2/3` to courses in order of appearance in API data
- ShotLink data may be missing for some courses → handle NaN gracefully with `.fillna(0)`
- Both single and dual-course scenarios must be tested

### 8. Round Simulation Pricing (`round_sim.py`)
- Normal distribution simulation (not hole-by-hole like R1)
- **Matchup pricing**: Fetch odds from DataGolf API, simulate H2H win probabilities, calculate edges
- **Score card**: Fair UNDER prices per half-stroke line
- **Edge calculation**: Book odds vs model fair odds
- **Email reporting**: HTML tables filtered by confidence thresholds and sample size requirements

### 9. DraftKings Scoring (`rd_1_sd_multicourse_sim.py`, lines 117-161)
```
-3 (double eagle+): 16 pts    -2 (eagle): 11 pts
-1 (birdie): 5.75 pts          0 (par): 1.5 pts
+1 (bogey): -1.8 pts          +2 (double bogey+): -3.9 pts
Bonuses: Bogey-free round +5, 3+ consecutive birdies +5 (single streak only)
```

### 10. Bet Storage & Connection Pooling (`sheets_storage.py`)
- **Connection pooling**: `get_spreadsheet()` caches a single gspread Spreadsheet object at module level. All `store_*` functions accept an optional `spreadsheet=` parameter; if None they call `get_spreadsheet()`. Callers (new_sim, round_sim) get one auth per run instead of N.
- **Parquet write-through**: Every `store_*` call also writes to `permanent_data/bet_ledger.parquet` via `_append_to_ledger()`. Deduplication by `(event_id, bet_type, round, bet_on, opponent, bookmaker)` — first write wins. Atomic writes via `tempfile` + `os.replace()`.
- **Ledger grading**: `update_ledger_grades(graded_bets)` matches on the same dedup key and updates `result`, `units_wagered`, `units_won`, `actual_finish`, `p1_round_score`, `p2_round_score`, `graded_at`.
- **Querying**: `query_ledger(**filters)` returns a filtered DataFrame. Filters: `event`, `bet_type`, `book`, `min_edge`, `graded`, `year`.
- **Auth DRY**: `grade_bets.py` imports `get_spreadsheet` from `sheets_storage` instead of defining its own duplicate `_get_credentials()` / `_connect_sheets()`.

### 11. Bet Query CLI (`bet_query.py`)
Three output modes:
- **Terminal** (default): Overall W/L/P, ROI, breakdowns by bet type and edge bucket
- **CSV** (`--export`): Saves `filtered_bets_{timestamp}.csv`
- **Plotly** (`--plot`): 4-panel interactive HTML dashboard (cumulative P&L, ROI by book, edge buckets, scatter)

---

## sim_inputs.py Reference

This file is NOT in the repo (local only, .gitignored). It contains ALL tournament-specific config:

```python
# Core identifiers
tourney = "tournament_name"          # String, used in all file naming
course_id = "course_id"             # Primary course
course_id_1 = "..."                 # Multi-course: first course
course_id_2 = "..."                 # Multi-course: second course
course_name = "Course Name"
course_par = 72                     # Integer
event_ids = [...]                   # DataGolf event ID list

# Simulation config
num_sims = 10000
player_var = 0.5                    # Variance amplification factor

# Weather arrays (hourly, starting 6 AM)
wind_1 = [...]                      # R1 wind forecast (≥15 elements)
wind_2 = [...]                      # R2 wind forecast
wind_3 = [...]                      # R3 wind forecast
wind_4 = [...]                      # R4 wind forecast
dewpoint_1 = [...]                  # R1 dewpoint forecast
dewpoint_2 = [...]
dewpoint_3 = [...]
dewpoint_4 = [...]
dewpoint_wave = [...]               # Alternate dewpoint array

# Weather coefficients
baseline_wind = 0.08                # Default wind effect per MPH
wind_override = 0                   # 0 = use calculated blend
baseline_dew = ...                  # Dewpoint baseline
dew_calculation = ...               # Dewpoint adjustment coefficient

# Scoring adjustments (strokes relative to baseline)
score_adj_r1 = ...                  # Can also be imported as score_adj_r1_sd
score_adj_r2 = ...
score_adj_r3 = ...
score_adj_r4 = ...

# Skill coefficients (per-round, per-bucket)
coefficients_r1_high = [...]        # R1: high-skill players
coefficients_r1_midh = [...]        # R1: mid-high
coefficients_r1_midl = [...]        # R1: mid-low
coefficients_r1_low = [...]         # R1: low-skill
coefficients_r2 = [...]             # R2: top 5
coefficients_r2_6_30 = [...]        # R2: positions 6-30
coefficients_r2_30_up = [...]       # R2: positions 30+
coefficients_r3 = [...]             # R3/R4: top bucket
coefficients_r3_mid = [...]
coefficients_r3_high = [...]

# Name normalization
name_replacements = {
    "old_name": "new_name",
    ...
}
```

---

## Environment & Authentication

### Local Development
- Python 3.10+
- `pip install pandas numpy scipy requests python-dotenv gspread google-auth xlsxwriter statsmodels patsy scikit-learn matplotlib`
- `credentials.json` for Google Sheets (service account)
- `.env` file with: `DATAGOLF_API_KEY`, `EMAIL_USER`, `EMAIL_PASSWORD`, `GMAIL_APP_PASSWORD`
- Database: `C:/Users/mckin/OneDrive/dg_historical.db` (sqlite3, used by cat_dists_player.py as fallback; CSV is primary)

### GitHub Actions
Secrets configured at `github.com/mslade50/sims_process/settings/secrets/actions`:
- `DATAGOLF_API_KEY`
- `EMAIL_USER` / `EMAIL_PASSWORD`
- `EMAIL_RECIPIENTS`
- `GOOGLE_CREDS_JSON` (complete JSON, no line breaks)

### Google Sheets Integration
- Sheet: `golf_sims`, Tab: `round_config`
- Column A = parameter name, Column B = value, Column C = notes
- Share sheet with service account email from credentials.json
- Parameters: `round`, `expected_score_1/2/3`, `wind`, `dew`, `wind_paste`, `dew_paste`, `dew_calculation`, `wind_override`

### File Sync Targets (Windows, in dists_thiswk.py)
- `C:\Users\mckin\OneDrive\sims_process`
- `C:\Users\mckin\OneDrive\etr-golf-sims`

---

## Known Pitfalls & Hard-Won Debugging Lessons

### 1. Multi-Course ShotLink Data Gaps
**Symptom**: NaN values in SG columns for players who played a course without ShotLink coverage.
**Cause**: Not all courses in multi-course events have ShotLink data.
**Solution**: Filter by data availability, not just player status. Check for NaN in source files before simulation. Use `.fillna(0)` for adjustment columns (e.g., `delta_app_adj`).

### 2. Weather Centering Direction
**Symptom**: All players show same weather adjustment direction.
**WRONG**: `mean - individual`
**CORRECT**: `individual - mean` (positive = player faces worse conditions than average)
This was an actual bug that was caught and fixed. Never reverse this.

### 3. Distribution Identity Check
**Symptom**: "All player distributions are identical!" warning (rd_1_sd_multicourse_sim.py line 444-447)
**Cause**: Adjustments not applying correctly — variance_multiplier, skill_adj, wind_adj may not vary.
**Debug**: Print the range of each adjustment component across players.

### 4. Tee Time Parsing
Multiple formats in the wild (rd_1_sd_multicourse_sim.py lines 52-65):
- `%Y-%m-%d %H:%M` (standard)
- `%I:%M%p` (e.g., "1:55PM")
- `%m/%d/%Y %H:%M` (e.g., "1/2/2025 9:33")
If a new format appears, the code raises `ValueError` — add a new `try/except` block.

### 5. R3/R4 Cascading Skill Updates
**Symptom**: Predictions seem to double-count adjustments.
**Root cause**: R3/R4 must UNDO prior round's SG and residual adjustments before applying fresh ones.
**Formula**: `total_adjustment = fresh_adj - prior_sg - prior_resid`
This ensures `Post = Pre + total_adjustment` remains consistent.

### 6. Wind Coefficient Blending
When `wind_override == 0` (default):
```python
wind_calculation = course_wind_effect * 0.4 + baseline_wind * 0.6
```
The course-specific wind effect comes from `wind_test.csv`, filtered by `event_ids`. If no match found, defaults to 0.08.

### 7. R1 Residual Cap Logic
```python
tot_resid_adj = residual_adj + residual2_adj
# Cap at 0.2 if total > 0.2 AND raw residual is negative
# Hard cap at 0.5 regardless
```
This prevents residual adjustments from overwhelming the prediction.

### 8. Player Name Normalization
Names must be lowercase and go through `name_replacements` dict. Both sides of any join/merge must be normalized. The `dists_thiswk.py` pattern is the gold standard — apply `normalize_name()` to both dist and field DataFrames, then join on `player_key`.

### 9. R2 Adjustment Column Trap
`sg_total_adj` is RAW DATA, not an adjustment. When summing adjustment components for R2, explicitly list the columns — don't use a wildcard that might catch `sg_total_adj`.

### 10. Course Shape Adjustment File Naming
Must match exactly: `course_shape_adjustments_{course_id}.csv`. If course_id has special characters or spaces, they must match.

### 11. DataGolf Field-Updates API: Nested Tee Times
**Symptom**: R2/R3 tee times reported as unavailable; all players default to 10:00 AM, zeroing out weather differentiation.
**Cause**: The DataGolf `/field-updates` API does NOT return flat columns like `r1_teetime`, `r2_teetime`. Instead it returns a single `teetimes` column containing a nested list of dicts, each with `round_num`, `teetime`, `course_code`, `course_name`, etc.
**Why R1 wasn't affected**: R1 tee times come from `model_predictions_r1.csv` (created pre-event), not from `fetch_field_updates`. R2/R3 depend on the API for tee times.
**Fix**: `fetch_field_updates` in `api_utils.py` parses the nested `teetimes` list, filters by `round_num`, and extracts `teetime` → `r{N}_teetime` and `course_code` → `course`. Fixed Feb 2026.

### 12. Google Sheets Connection Pooling
**Symptom**: Slow store operations — each `store_*` call takes 3-5 seconds for auth.
**Cause**: Pre-Feb 2026 code called `_connect_sheets()` inside every `store_*` function, causing N separate OAuth round-trips per script run (4 in `new_sim.py`, 2 in `round_sim.py`).
**Fix**: `get_spreadsheet()` caches at module level. All `store_*` functions accept `spreadsheet=None` — callers pass one object. Fully backward compatible (None triggers `get_spreadsheet()`).

### 13. Parquet Ledger Dedup Key
**Dedup key**: `(event_id, bet_type, round, bet_on, opponent, bookmaker)` — all lowercased/stripped.
**Why this matters**: A bet stored via `store_tournament_matchups()` (raw) and then via `store_sharp_filtered()` should appear once. The dedup ensures first-write wins. If you add a new store function that writes the same bets, dedup handles it automatically.
**Atomic writes**: `_append_to_ledger()` uses `tempfile.mkstemp()` + `os.replace()` to prevent corruption on crash.

### 14. grade_bets.py Auth
`grade_bets.py` imports `get_spreadsheet` from `sheets_storage` instead of defining its own `_get_credentials()` / `_connect_sheets()`. If you see duplicate auth code, it's a regression — DRY it through `sheets_storage`.

---

## Weekly Operational Workflow

```
Sunday night:    Weekly cleanup runs (GitHub Action)
                 Deletes transient CSVs/Excel/tournament folders from repo root
                 permanent_data/ and .py files preserved
                 Grade previous week: python grade_bets.py --event-id <id>
                 Review results: python bet_query.py --summary --by-event

Monday/Tuesday:  Update sim_inputs.py for new tournament
                 Run cat_dists_player.py (if new data available)
                 Run dists_thiswk.py

Wednesday:       Run rd_1_sd_multicourse_sim.py
                 Update Google Sheet with weather forecasts

Thursday (R1):   Update Google Sheet: round=0, wind/dew for R1
                 Run live_stats_engine.py (pre-event → model_predictions_r1.csv)
                 Run new_sim.py → bets auto-saved to Sheets + Parquet ledger
                 After R1 completes: set round=1, update weather for R2
                 Run live_stats_engine.py (skill update)
                 Run round_sim.py → bets auto-saved to Sheets + Parquet ledger

Friday (R2):     Set round=2, update weather
                 Run live_stats_engine.py
                 Run round_sim.py → auto-saved

Saturday-Sunday: Same pattern for R3/R4
                 After Sunday: grade_bets.py grades all bet types + updates ledger
```

> See `WEEKLY_PROCESS.md` for the full step-by-step operational guide with exact commands.

---

## Conventions & Style

- **Player names**: Always lowercase, normalized through `name_replacements`
- **File naming**: `{description}_{tourney}_{course_id}.csv` for tournament-specific files
- **Column naming**: Underscore-separated, lowercase (e.g., `wind_adj1`, `updated_pred_r3`)
- **CSV-based pipeline**: Database path exists but isn't primary — CSV is the standard interchange format
- **Testing**: Use small `num_sims` (100-1000) before full runs (10,000)
- **Security**: All credentials in env vars or GitHub secrets — never hardcoded in committed files
- **Error handling**: Graceful degradation — missing files should warn, not crash (especially in multi-course scenarios)

---

## Debugging Checklist

When simulation results seem wrong:

1. **Verify inputs exist and have expected shape**
   ```python
   print(model_predictions.shape)
   print(model_predictions[['player_name', 'my_pred', 'std_dev']].head())
   ```

2. **Check adjustment calculations**
   ```python
   print(f"Avg skill: {model_predictions['my_pred'].mean():.3f}")
   print(f"Avg wind: {model_predictions['wind_adj1'].mean():.3f}")
   print(f"Wind range: {model_predictions['wind_adj1'].min():.3f} to {model_predictions['wind_adj1'].max():.3f}")
   ```

3. **Verify distribution variance across players**
   ```python
   # Should be False — if True, adjustments aren't working
   print(all(all_distributions[0].equals(dist) for dist in all_distributions[1:]))
   ```

4. **Check for unexpected NaNs**
   ```python
   print(model_predictions.isna().sum())
   ```

5. **Verify total_adjustment consistency**
   ```python
   # Post = Pre + total_adjustment must hold
   assert (df[post_col] - (df[pre_col] + df['total_adjustment'])).abs().max() < 1e-6
   ```

---

## Extension Points

### Adding a new weather variable
1. Add array to `sim_inputs.py` (e.g., `humidity_1`)
2. Add to `WIND_ARRAYS` or equivalent dict in `live_stats_engine.py`
3. Add calculation loop in simulation file (parallel to wind/dew pattern)
4. Add adjustment column (`humidity_adj1`)
5. Include in `player_adj_mean` calculation (rd_1_sd_multicourse_sim.py line 402-407)
6. Remember: center it (`individual - mean`) if it's a relative advantage variable like dewpoint

### Adding a new SG category
1. Add column name to `CATS` list in `cat_dists_player.py`
2. Ensure column exists in source data with `_adj` suffix
3. Course adjustments will need corresponding row in `course_shape_adjustments_{course_id}.csv`

### Adding a new round coefficient bucket
1. Add coefficients to `sim_inputs.py`
2. Import in `live_stats_engine.py`
3. Add bucket logic in the appropriate `_apply_coefficients_r{N}` function
4. Test with edge cases (bucket boundaries, players with missing data)

---

## Notes for Claude

**When reviewing this codebase:**
1. Always check `sim_inputs.py` contents first — it controls all tournament-specific behavior
2. The formula `Post = Pre + total_adjustment` must be consistent across all rounds
3. Weather centering pattern: `individual - mean`, NOT `mean - individual`
4. Multi-course tournaments require checking both course files before diagnosing issues
5. Real player data always takes precedence over manual overrides once sufficient sample sizes are reached
6. When suggesting changes, preserve existing naming conventions and the CSV-based pipeline
7. Test with small `num_sims` first, then scale up
8. The multi-course merge logic at the end of rd_1_sd_multicourse_sim.py is fragile — test both single and dual-course scenarios
9. `sg_total_adj` is raw data, not an adjustment — don't include it in adjustment sums

**When modifying code:**
- Follow the existing pattern of comprehensive print statements for debugging
- Maintain graceful degradation for missing files (especially multi-course)
- Any new API calls should go through `api_utils.py`
- Any new email reporting should follow the HTML table pattern with confidence thresholds
- Any new `store_*` functions in `sheets_storage.py` should accept `spreadsheet=None` and call `get_spreadsheet()` as fallback
- Any new bet storage should also write to the Parquet ledger (follow the `_ledger_write_*` pattern)
- Auth code belongs in `sheets_storage.py` only — never duplicate `_get_credentials()` in other files
- The Parquet ledger at `permanent_data/bet_ledger.parquet` is local-only (gitignored) — Sheets remains the authoritative store
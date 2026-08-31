# Weekly Tournament Process: Step-by-Step Guide

Complete operational playbook for running the golf simulation system from Monday grading and cleanup through the next event.

---

## Phase 0: Monday Grading & Cleanup

### 0.1 Automatic Cleanup (GitHub Action)
The `weekly-cleanup.yml` workflow runs Monday at 17:00 UTC (noon EST / 1 PM
EDT), after the grading retries. It deletes:
- All CSVs, Excel files, and tournament folders from the repo root
- `permanent_data/` and `.py` files are preserved

**Verify cleanup ran:**
```bash
git pull
ls *.csv *.xlsx  # should return nothing
```

### 0.2 Automatic Monday Grading (GitHub Action)
The `monday-grading.yml` workflow runs the full grading pipeline automatically:
- **Monday 14:00 UTC** (9 AM EST / 10 AM EDT) — primary run
- **Monday 15:00 UTC** (10 AM EST / 11 AM EDT) — retry if DataGolf data wasn't ready

**What it does** (`monday_grading.py`):
1. Detects last completed event from DataGolf API
2. Checks if grading already done (idempotent — safe for retry)
3. Runs `grade_bets.py` (grades bets, writes to Sheets + Parquet, sends email)
4. Verifies all bets graded
5. Runs `sg_diagnostic.py --no-email` (stores SG diagnostic to Parquet)
6. Runs `push_dashboard_data.py` (syncs source data and atomically publishes dashboard JSON to Cloudflare R2)

**Manual trigger** (if you don't want to wait for schedule):
```bash
# Run locally
python monday_grading.py

# Or trigger via GitHub Actions
gh workflow run monday-grading.yml
```

### 0.2b Manual Grading (edge cases only)
For re-grading, specific events, or dry-run preview:

```bash
python grade_bets.py --event-id 5 --event-name "AT&T Pebble Beach"
python grade_bets.py --event-id 5 --dry-run
python grade_bets.py --regrade --event-id 5
python grade_bets.py --all-events --regrade
```

**Note:** `grade_bets.py` no longer pushes the dashboard. Run `python push_dashboard_data.py` manually after if needed.

### 0.3 Review Season Performance
```bash
# Summary by event
python bet_query.py --summary --by-event

# Summary by bookmaker
python bet_query.py --summary --by-book

# Only sharp book bets
python bet_query.py --book pinnacle --graded --summary

# Interactive dashboard
python bet_query.py --plot

# Export to CSV for external analysis
python bet_query.py --graded --export
```

---

## Phase 1: Monday/Tuesday Prep

### 1.1 Update `sim_inputs.py`

> **⚠️ FIRST — check one-off overrides from last week.** If any of the following are
> non-None / non-zero / non-default in `sim_inputs.py`, reset them:
> ```python
> lat_override = None
> lon_override = None
> baseline_score_adj = 0.0
> tour_override = 'pga'          # only set to 'liv' (etc.) for one-off events
> historical_event_ids = []      # deliberate tour/event override or course-history disambiguation only
> ```
> **Two override modes:**
>
> 1. **Partner / no-data course** (e.g. Zurich 2026 at TPC Louisiana reusing
>    course_id=11 for variance): set `lat_override`, `lon_override`, optional
>    `baseline_score_adj`. Bypasses `course_coordinates.csv` lookup in `humidity.py`
>    + `scoring_baseline.py` and adds a flat strokes adjustment to expected scores.
>
> 2. **LIV-history course** (e.g. Cadillac 2026 at Trump National Doral — PGA event
>    at a course with only LIV history): set `lat_override` + `lon_override` (no
>    PGA `course_coordinates.csv` entry), AND `tour_override = 'liv'`,
>    `historical_event_ids = [LIV event_ids at that course]`. The `tour_override`
>    flows into all the historical SQL queries in `scoring_baseline.py` (incl. the
>    SG variance + skew computation that becomes `cat_mult_*` / `cat_skew_*` in the
>    Sheet); `historical_event_ids` lets the script pull from a different set of
>    events than the current-week `event_ids` (which still reflects this week's
>    PGA event for live API calls). LIV is 54 holes — there will be no R4 baseline,
>    set `expected_score_r4` manually in the Sheet.
>
> Leaving any of these on for a normal week will pull weather from the wrong
> location, query the wrong tour's history, or shift expected scores. One other
> valid use of `historical_event_ids` is disambiguating a venue that hosted more
> than one qualifying tournament in the same year; the baseline guard will stop
> and tell you when that explicit selection is required.

Open `sim_inputs.py` and update for the new tournament:

**Core identifiers (always update):**
```python
tourney = "genesis"                    # Used in all file naming
course_id = 20                         # DataGolf course ID
course_id_1 = 20                       # Multi-course: first
course_id_2 = 220                      # Multi-course: second (if applicable)
course_name = ""                       # For multi-course showdown sims
course_par = 71                        # Course par
event_ids = [20]                       # DataGolf event ID list
```

**Scoring adjustments (reset to 0 unless you have data):**
```python
score_adj_r1 = 0
score_adj_r2 = 0
score_adj_r3 = 0
score_adj_r4 = 0
score_adj_r1_sd = 0
score_adj_r2_sd = 0
score_adj_r3_sd = 0
score_adj_r4_sd = 0
```

**Weather coefficients (update from course research):**
```python
wind_override = 0.0                    # 0 = use computed blend
baseline_wind = 0.08                   # Default per-MPH effect
baseline_dew = -0.018                  # Dewpoint baseline
dewpoint_wave = -0.035                 # Alternate dew coefficient
dew_calculation = 0.6*baseline_dew + 0.4*dewpoint_wave
```

**Player variance (adjust per course characteristics):**
```python
player_var = 2                         # Higher = more variance in sim
```

**Course category variance multipliers (for `new_sim.py`):**
Auto-managed — `scoring_baseline.py` computes them from its SG variance analysis
(`actual_category_std / field_expected_category_std`; >1.0 = course amplifies,
<1.0 = compresses) and writes them to the `cat_mult_*` / `cat_skew_*` param rows
in the Sheet's `round_config` tab. `new_sim.py` reads `COURSE_CAT_MULTS` from the
sheet config at runtime (`sheet_config.py`). Override the sheet cells after
`scoring_baseline.py` only for a deliberate manual adjustment.

**Name replacements (add any new players with naming issues):**
```python
name_replacements = {
    'echavarria, nico': 'echavarria, nicolas',
    # ... add new entries as needed
}
```

**Overrides (for players missing data):**
```python
overrides = {}       # Dict of {player_name: skill_estimate}
manual_boosts = {}   # Dict of {player_name: boost_amount}
```

**Cut rules:**
```python
cutline = 80         # Cut line (inclusive of ties)
shot_rule = 0        # 10-shot rule: 0 = off, 10 = on
```

### 1.2 Generate Skill Predictions (external)
Run the skill model outside this repo. It produces:

- **`pre_course_fit_{tourney}.csv`** — baseline predictions (`pred`, `std_dev`, `sample`, `dg_id`). Used by `scoring_baseline.py` for field strength and by `new_sim.py` for the first-pass sim.

**`final_predictions_{tourney}.csv`** is NOT generated externally — it is produced by `mkt_regress.py` (see Phase 3.2) after the calibration pass. Pass selection is explicit: `--calibration-pass` requires `pre_course_fit_{tourney}.csv`, while `--final-pass` requires `final_predictions_{tourney}.csv`. A no-flag run is calibration-only; file existence never authorizes external delivery because tournament slugs repeat across years.

**Downstream consumers:**
- `scoring_baseline.py` — reads `pre_course_fit` for field strength adjustment
- `cat_dists_player.py` (runs in sim_prep) — uses `pre_course_fit` as the field and checks it for hot players missing the sample cut
- `new_sim.py` — reads the pass-bound prediction artifact (`pre_course_fit` for calibration; `final_predictions` only with `--final-pass`)
- `mkt_regress.py` — reads first-pass sim outputs + market odds, produces `final_predictions`
- `live_stats_engine.py` — reads `final_predictions` for round=0 baseline (live rounds only)
- `sheets_storage.py` — reads `pre_course_fit` for `dg_id` lookup

### 1.3 Distribution Files (built in sim_prep, not here)

`sg_dist_player.csv` and `this_week_dists_v2.csv` arrive from **sim_prep's**
`cat_dists_player.py` (single producer as of 2026-07-20; the local copy is in
`archive/`). It runs as part of the sim_prep weekly prep, next to the fresh
`field_adjusted_sg.csv` it needs, and fans the outputs out to this repo, the
OneDrive root, and etr-golf-sims.

**What it does:**
- Reads `field_adjusted_sg.csv` (PGA rounds since 2019)
- Computes EMA-weighted SG distributions per player per category
- Uses `pre_course_fit_{tourney}.csv` as the field (aligned with new_sim / scoring_baseline)
- Uses alpha = 2/(50+1), effective half-life ~17 rounds; minimum 20 observations per category

**Check before simming:** both files in this repo's root should be dated after
the sim_prep prep run for the current week. If they're stale, the prep didn't
run (or its fan-out failed) — rerun `cat_dists_player.py` in sim_prep.

The guarded sim_prep pipeline also (2026-07-28) refreshes `field_updates.csv`
here post-publish and runs `dists_thiswk_team.py` as its final stage, which
builds `this_week_dists_adjusted.csv` and syncs it to `~/OneDrive/etr-golf-sims`
along with `field_updates.csv`, `sg_dist_player.csv`, and `this_week_dists_v2.csv`.

### 1.3b Weekly Shot-Dispersion Input (default production model)

The sim_prep weekly pipeline must also deliver these two files to this repo's
root before any tournament or round simulation:

- `shot_dispersion_features.csv` — the current field's leakage-safe,
  pre-event shot-variance features
- `shot_dispersion_config.json` — current tournament/event identity, field
  size, category weights, and frozen hashes for both the feature file and
  `this_week_dists_v2.csv`

The weekly config is normally `enabled: true` and
`required_current_event: true`. Both CSV hashes use the tracked
`sha256_lf_normalized_v1` convention, so LF and CRLF checkouts verify
identically. A missing feature, changed distribution, hash mismatch, or stale
tournament/event identity stops `new_sim.py` and `round_sim.py`; it is never a
reason to silently fall back to last week's overlay. The same pre-event files
remain frozen through R1-R4.

`SHOT_DISPERSION_DISABLE` cannot bypass a required weekly config. An emergency
rollback must be explicit and reviewable by setting `enabled: false` in the
tracked config; restore the default enabled config in the next weekly build.

---

## Phase 2: Wednesday Pre-Tournament Simulation

### 2.1 Auto-Populate Weather & Course Codes

**Populate weather tables in Google Sheet:**
```bash
python humidity.py
```

**What this does:**
1. Looks up course coordinates from `permanent_data/course_coordinates.csv` using `course_id` in `sim_inputs.py`
2. Fetches tournament dates from DataGolf `/field-updates` API (R1 = start date, R2-R4 = +1/+2/+3 days)
3. Fetches hourly forecast from Open-Meteo (dewpoint + wind, 6 AM - 8 PM = 15 values per round)
4. Writes directly to the `round_config` tab in Google Sheets:
   - Dew columns (G, K, O, S) rows 3-17 for all 4 rounds
   - Wind columns (F, J, N, R) rows 3-17 for all four rounds
   - Rows 18-21 contain formula-generated comma-separated summaries (not overwritten)
5. Prints score adjustments, average wind speeds, rainfall, and historical variance

**Populate course codes:**
```bash
python update_sheet_courses.py
```
Fetches course codes from DataGolf and writes to the `course_codes` row in `round_config`. Then manually fill in `course_pars` and `expected_score_rN` if multi-course.

**Re-run as forecasts update** — run `humidity.py` again closer to Thursday to get fresher forecasts. It overwrites the same cells.

**Bayesian wind blending** — the sim and live_stats_engine automatically blend forecast wind arrays with a historical climatological prior (monthly hourly averages at the course location, 2019-2025). The climo weight scales with lead time: `lead_days / 12`, clamped [5%, 50%]. Running the sim Monday gives ~25% climo for R1; re-running Wednesday night gives ~5%. No action needed — blending is automatic in `api_utils.py`.

### 2.1b Compute Scoring Baselines & Expected Scores
```bash
python scoring_baseline.py
```

**What this does:**
1. Fetches historical scoring averages from `dg_historical.db` for the exact
   `course_id` (event IDs are not stable historical venue identifiers)
2. Fetches/caches historical weather from Visual Crossing API
3. Computes weather-free, tour-average-field baseline per round (de-skilled, de-winded)
4. Applies recency weighting across years
5. Adjusts baselines for THIS WEEK's field strength (from `pre_course_fit_{tourney}.csv`) and tee-time-weighted wind exposure
6. Writes `expected_score_r1` through `expected_score_r4` to the `round_config` tab;
   on single-course pre-event slates it also synchronizes `expected_score_1`
   from the new R1 estimate
7. Writes a visible breakdown table to the Sheet (cols E-J, rows 23+)
8. Saves detail CSV (`scoring_baseline_{tourney}.csv`) and Sheets tab (`Scoring Baseline`)

**Formula**: `expected_score = baseline - field_strength + avg_wind * wind_factor`

The baseline run validates that the selected course has at most one qualifying
tournament per year and stops instead of silently combining ambiguous history.
Set `historical_event_ids` only for a deliberate tour/history override; even
then, the configured course filter remains mandatory. Wind coefficients also
match `course_num` directly and never fall back to a different venue sharing
the current event ID.

**Requires**: `pre_course_fit_{tourney}.csv` (from pre-tournament pipeline) for field strength. Without it, field strength defaults to 0.

**How automation uses these**: The event-driven midweek workflow refreshes
`wind_rN`/`dew_rN`, advances the Sheet `round`, recomputes
`expected_score_rN`, synchronizes the target value to `expected_score_1`, and
runs the live engine plus round sim. The nightly workflow remains a backup for
an already-correct Sheet round; it does not advance a stale round pointer.

### 2.1c Push Tournament Config to Sheet
```bash
python init_weekly.py
```

**What this does:**
- Reads `tourney`, `event_ids[0]`, `course_id`, `cut_line`, and `course_pars` from `sim_inputs.py`
- Writes them to the `round_config` tab in the Google Sheet
- Resets `round` to `0`

**Why this runs AFTER humidity + scoring_baseline:** Those two scripts read tournament config directly from `sim_inputs.py`, so they don't need the Sheet updated first. But everything downstream (`live_stats_engine.py`, `round_sim.py`) reads from the Sheet via `sheet_config.py` — so the Sheet must reflect the new tournament before they'll work. If you skip this step, downstream scripts will silently run against last week's tournament. (The distribution builder no longer runs here — sim_prep's copy reads `tourney` from `sim_inputs.py`, so it has no Sheet dependency.)

### 2.1d Write Base Rates Reference
```bash
python write_base_rates.py
```

Writes a "Base Rates" tab to the Google Sheet comparing tour-average defaults
with this week's values, with delta columns for quick deviation spotting.

The Base Rates diagnostics use physical `course_id` history for wind, AM/PM
splits, and category variance. The script also refreshes ETR event and category
profile values. Existing ETR `skill_docks` are preserved on ordinary reruns;
use `python write_base_rates.py --reset-skill-docks` only during an intentional
new-week reset.

### 2.1e Rebuild ETR Hole Baseline (when preparing ETR)
```bash
python hole_baselines.py
```

The builder resolves each historical tournament from the physical `course_id`,
then maps those exact `(year, event_id)` pairs to IMG archive event keys. It
does not assume the current event ID was stable historically, excludes the
in-progress season, stops on ambiguous same-course tournaments within a year,
and warns when a valid course edition is absent from the shot archive. The
completed file is copied to `~/OneDrive/etr-golf-sims`.

### 2.2 Verify Google Sheet & Set Remaining Parameters
After `humidity.py`, `update_sheet_courses.py`, and `scoring_baseline.py` have run, open the `round_config` tab and verify/set:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `round` | `0` | Pre-event |
| `expected_score_1` | `71.5` | Expected scoring avg, course 1 |
| `expected_score_2` | `72.0` | Multi-course only, blank if single |
| `dew_calculation` | leave blank | Falls back to sim_inputs |
| `wind_override` | `0` | 0 = use computed blend |
| `course_codes` | `TS` or `PB,SG` | Auto-populated by `update_sheet_courses.py` |
| `course_pars` | `71` or `72,72` | Manual — match course_codes order |

Weather columns (dew/wind in the hourly tables) are auto-populated by `humidity.py`.

---

## Phase 3: Pre-Tournament Simulation (Tue–Thu)

The tournament sim is a **two-pass process**: first pass uses raw predictions, market regression adjusts them, second pass uses regressed predictions.

### 3.1 First Pass: Run Tournament Sim
```bash
python new_sim.py --calibration-pass
```

The calibration pass reads exactly `pre_course_fit_{tourney}.csv`, even if generated files from an earlier run still exist. It writes local simulation/pricing artifacts for `mkt_regress.py`, then exits before email, Sheets/ledger storage, dashboard pushes, or sim-fairs publication.

**Monday tee-time rule (America/New_York):** missing R1 or R2 tee times are
expected and do not block calibration, the final tournament sim, or email. Any
round below 98% coverage (or with more than one player missing) is neutralized
for the full field, producing zero wave differentiation and never reusing stale
times. Complete rounds still use their fresh times.

Beginning Tuesday, the final pass requires fresh DataGolf R1 and R2 tee-time
payloads with at least 98% field coverage and no more than one missing player.
An explicit early computation-only calibration may still run with
`python new_sim.py --calibration-pass --allow-missing-tee-times`; that manual
escape remains prohibited for `--final-pass`, `--price-only`, and `--reprice`.

**TOUR Championship exception:** when the pre-event config exactly matches
`tourney=tourchamp`, `event_id=60`, and East Lake `course_id=688`, R1 remains
strict but R2 is allowed to be unavailable because the field is re-paired after
Round 1. A missing or partially posted R2 payload is cleared for the entire
field and receives zero player-specific wave adjustment; a complete fresh R2
payload is used normally. This exemption does not apply after the event begins
and does not weaken any other tournament's tee-time gate. The midweek
next-round publisher still requires the real groups once each re-pair is posted.

**What this does:**
- **DG override** (before sim, first pass only): replaces `my_pred` with DataGolf's `dg_final_pred` for players with `pred < 0.5` or in `dg_override_players` list (in `sim_inputs.py`). Skipped on the explicit final pass. `live_stats_engine.py` does NOT apply threshold overrides — manual list only.
- Draws SG categories (OTT, APP, ARG, PUTT) from a course-adjusted multivariate normal, then sums to total (category-first approach)
- Uses `COURSE_CAT_MULTS` from Google Sheet to scale per-category variance
- Re-centers category means to sum to `my_pred` so only variance structure changes, not base predictions
- Uses the Rust tournament-draw kernel by default; a kernel failure stops the run instead of silently changing math. Use `--use-python` only as an explicit operator choice.
- Fetches betting odds from DataGolf matchup/outright APIs
- Runs Monte Carlo tournament simulation (matchups + finish positions)
- Produces `pre_sim_summary_{tourney}.csv`, `finish_equity_{tourney}.csv`, matchup CSVs

**The calibration pass never stores bets or publishes anything.** The Monday 3 PM EST storage time gate applies only to the explicit final pass.

**Simx edge decomposition:** The email reports a **Simx** column for matchups — this is the edge difference between the full category-first sim and a simple Normal CDF analytical model (`edge_on - edge_no_wx`). It captures the value added by skewness, correlation, and course variance multipliers. Stored as `wx_edge` in Sheets and the Parquet ledger for tracking. When weather arrays are populated, the true weather contribution can be isolated as `Wx = total_wx_edge - Simx_baseline` (where Simx_baseline is from a no-weather run).

**Weekly setup** — `COURSE_CAT_MULTS` is auto-set: `scoring_baseline.py` writes the `cat_mult_*`/`cat_skew_*` param rows to the Sheet's `round_config` tab when it runs, and `new_sim.py` reads them from the sheet config. Just make sure `scoring_baseline.py` ran for the current course before simming; hand-edit the sheet cells only to deliberately override.

`WEEK_LATENT_SD`, the current root-level `shot_dispersion_features.csv`, an
enabled `shot_dispersion_config.json` with
`required_current_event: true`, and
`permanent_data/sg_cat_corr_tour_within_player_pearson.csv` are required model
inputs. Missing values/files, stale event identity, or normalized hash
mismatches stop the run. Use `--no-week-latent` only for an intentional latent
variant; a required overlay cannot be disabled by environment variable. Older
correlation matrices and identity are never silent production fallbacks.

### 3.2 Market Regression
```bash
python mkt_regress.py
```

For an isolated comparison or shadow rerun, use
`python mkt_regress.py --local-only`. It still writes both local final-prediction
files but skips the OneDrive copy and `skill_merge.py`, so it cannot hand files
to the ETR workflow.

**What this does:**
1. Reads first-pass sim outputs (`pre_sim_summary_{tourney}.csv`, `finish_equity_{tourney}.csv`)
2. **DG override**: replaces preds with DataGolf's `dg_final_pred` for players with `pred < 0.5` or in `dg_override_players` list (in `sim_inputs.py`)
3. Applies three regression layers toward market prices:
   - **mkt_adj** (max ±0.12 SG): outright win odds disagreement (sharp book avg vs sim win%)
   - **mu_adj** (max ±0.13 SG): matchup edge disagreement (z-scored, inverted; boosted 1.5x for weak players)
   - **c_adj_regress** (up to 30% of c_adj): extra regression when course-fit drives the edge
4. Asymmetric dampening: upward adjustments dampened to 35% (UP ~40% accurate vs DOWN ~67%)
5. Outputs: `final_predictions_{tourney}.csv` (slim) and `final_predictions_{tourney}_details.csv` (full detail)
6. **Chains `skill_merge.py`** (local port of `~/OneDrive/skill_merge.py`, 2026-07-28): merges fresh
   tee times, 50/50 blends `my_pred` with DataGolf's `dg_final_pred`, writes
   `final_predictions_{tourney}_combined.csv` and copies it to `~/OneDrive/etr-golf-sims`
   (the Rust sim's `pred` input — update `sim_config.yaml` over there to point at the new week's file).
   Re-run `mkt_regress.py` (or `skill_merge.py` standalone) after tee times post to refresh it.

**Requires**: matchup odds CSV (`betcris_{tourney}.csv` or `betonline_odds_with_my_odds_tu.csv`)

### 3.3 Second Pass: Re-Run Tournament Sim
```bash
python new_sim.py --final-pass
```

This requires `final_predictions_{tourney}.csv` and is the **only normal pass allowed to deliver externally**. Edges, matchups, and finish positions are all based on market-regressed predictions.

- Requires SMTP acceptance of the main tournament report before storing any bets
- **Auto-saves to Google Sheets** (Tournament Matchups, Finish Positions tabs)
- **Auto-writes to Parquet ledger** (`permanent_data/bet_ledger.parquet`)
- Requires the sim-fairs git push and pinned odds-board build dispatch; either failure makes the run fail
- Stores all tournament matchups with edge > 3% (no pred/sample gate). Low-confidence bets are visible on the `/fragility` dashboard page. Email filters still apply (pred > 0.75, sample >= 20).
- Uses single Google auth via `get_spreadsheet()` (1 connection, not 4)

The final pass needs `EMAIL_USER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENTS`, Google credentials, and a `GH_TOKEN` that can push the fairs commit and dispatch the board build. A missing credential or rejected recipient is a failed run, not a warning.

Cached modes also require explicit final-pass authority **and a cache sealed by an earlier full final pass**. Calibration caches are rejected even when filenames/field shape happen to match. After a successful `python new_sim.py --final-pass`, use `python new_sim.py --final-pass --price-only` for strict cached pricing plus normal delivery, or `python new_sim.py --final-pass --reprice` for the existing email-only refresh that deliberately skips repeat storage and sim-fairs publication.

The cache seal binds the event/course, ordered player field, simulation count,
exact final-prediction bytes, pass type, every cached pricing artifact, and a
canonical hash of all draw inputs. That input contract includes the effective
model/category arrays, distribution and correlation bytes, weather/dew, course
multipliers/skew, cut rules, week latent, simulation flags, skill-update
coefficients, shot-dispersion inputs, and relevant Python/Rust source hashes.
Cached commands must repeat non-default simulation flags used for the full run.
The prediction file is hashed before and after reading and rechecked with the
complete contract immediately before delivery. Seals expire after seven days
and at the next UTC ISO week, so a repeated tournament slug cannot reuse a prior
edition's tape.

**Verify bet storage:**
```python
import pandas as pd
df = pd.read_parquet("permanent_data/bet_ledger.parquet")
print(f"Ledger: {len(df)} rows")
print(df[df['event_name'] == 'att'][['bet_type', 'bet_on', 'bookmaker', 'edge']].head())
# Check no duplicates
print(f"Duplicates: {df.duplicated(subset=['event_id','bet_type','round','bet_on','opponent','bookmaker']).sum()}")
```

---

## Phase 4: Thursday (Round 1)

### 4.1 Pre-Round: Generate Model Predictions
Before R1 tee times, with `round=0` in the Google Sheet:

```bash
python live_stats_engine.py
```

**What this does (round=0):**
- Reads config from Google Sheet (`round_config` tab)
- Fetches field updates from DataGolf API
- Generates `model_predictions_r1.csv` with pre-tournament skill estimates (reads `final_predictions_{tourney}.csv`)
- No skill adjustments applied (pre-event baseline)

> **IMPORTANT: `round=0` is used for ALL R1 operations** — both `live_stats_engine.py` (creates `model_predictions_r1.csv`) and `round_sim.py` (prices R1 matchups + score cards). The `expected_score_1` field in the sheet is the R1 expected scoring average. Do NOT change round to 1 until R1 is complete and you're ready to run the R2 pipeline.

### IMG shot archive diagnostics (optional)

The cleaned IMG archive can be joined into `live_stats_engine.py` after a round
by setting the current IMG event ID in the process environment:

```powershell
$env:IMG_SHOT_EVENT_ID = "<current IMG event id>"
python live_stats_engine.py
```

`IMG_SHOT_READ_TOKEN` and `IMG_SHOT_API_URL` are stored in the local user
environment. The event ID is deliberately not persisted: set it for the current
event so a prior week's archive can never be merged by accident. The integration
adds `img_*` diagnostic and completeness fields only; DataGolf remains the
source for strokes-gained inputs.

For direct use from `round_sim.py`, a notebook, or another analysis process:

```python
from api_utils import fetch_img_player_rounds, fetch_img_player_shots

player_rounds = fetch_img_player_rounds(1400, 4)
round_shots = fetch_img_player_shots(1400, 4)
```

For a remote runner such as GitHub Actions, configure `IMG_SHOT_READ_TOKEN`,
`IMG_SHOT_API_URL`, and the current `IMG_SHOT_EVENT_ID` in that environment.

### 4.2 Post-Round 1: Update Sheet & Run Skill Update

After R1 scores are final:

**Update Google Sheet:**

| Parameter | Value |
|-----------|-------|
| `round` | `1` |
| `wind` | R2 hourly wind forecast |
| `dew` | R2 hourly dewpoint forecast |
| `expected_score_1` | Actual R1 scoring avg |

```bash
python live_stats_engine.py
```

**What this does (round=1):**
- Fetches R1 live stats from DataGolf
- Applies R1 skill adjustments (4 buckets: high/mid-high/mid-low/low)
- Components: `ott_adj + putt_adj + tot_resid_adj` (residual capped at 0.5)
- Outputs:
  - `r1_live_model.csv` (R1 skill-adjusted model)
  - `model_predictions_r2.csv` (predictions for R2)

For rounds 1-3, the same run also records a context-aware scoring forecast in
the dedicated `Scoring Shadow` Sheet tab. It combines the target-round active
cohort's completed scores, exact-course historical round transitions, realized
weather, a robust structural anchor, and—when all 18 target-round holes are
available—a guarded daily-setup yardage adjustment. The setup adjustment uses
the rich shot archive's round-specific hole geometry, rejects incomplete or
implausible layouts, and records its raw, historically centered, and capped
values separately. Missing setup data is zero-impact and warning-only.

The entire forecast remains diagnostic: it runs after the authoritative
expected-score write and is not read by simulations or odds pricing. Any
shadow calculation or storage failure therefore cannot change the published
round expectation.

### 4.2b R1 Round Sim (with round=0 still in the sheet)
```bash
python round_sim.py
```
**Note:** `round=0` in the sheet means "R1 hasn't happened yet" — `round_sim.py` reads this and prices R1 matchups using `model_predictions_r1.csv` and `expected_score_1` as the R1 expected scoring average. Make sure `expected_score_1` is set to the R1 expected score (e.g., 72.6), not the generic baseline.

### 4.3 Post-Round 1: Run R2 Matchup Pricing
```bash
python round_sim.py
```

**What this does:**
- Reads `model_predictions_r2.csv`
- Fetches R2 matchup odds from DataGolf API
- Simulates round matchups using normal distributions
- Generates:
  - `matchups_r2.csv` (all matchup edges)
  - `fair_card_r2.csv` (score card with fair UNDER prices)
  - Excel workbook
- Sends email with filtered edges
- **Auto-saves to Google Sheets** (Round Matchups tab)
- **Auto-writes to Parquet ledger**

### 4.3a Automated Midweek Transition

`.github/workflows/midweek-round-automation.yml` listens for the odds board's
`sentinel-scrape-complete` dispatch. It advances and runs a round only when:

- `round_matchups_latest.json` is fresh, stamped for R2-R4, and has at least
  five valid event-scoped matchups from both BetCris and BetOnline
- the target round is exactly the next safe transition from the Sheet
- DataGolf has at least ten completed-round stat rows and ten posted
  next-round tee times
- the target round is single-course (multi-course rounds remain manual)
- all remaining rounds have complete 6 AM-8 PM Open-Meteo dewpoint and
  ECMWF IFS/AIFS + NOAA AIGFS wind arrays

The workflow uses `course_lat_lon` from the Sheet and asks Open-Meteo to
resolve the course-local IANA timezone automatically. It then writes the
remaining weather grids, updates the primary
`wind`/`dew` fields, increments `round`, runs
`live_stats_engine.py --automation`, and runs the full `round_sim.py`.

Durable `automation_*` rows in `round_config` make duplicate scraper dispatches
no-ops and let a failed run resume. The transition, nightly backup, manual sim,
and runner smoke workflows share one GitHub Actions concurrency group so two
full simulations cannot mutate round state simultaneously. Lightweight
repricing remains in its own queue. The readiness gate runs on GitHub-hosted
infrastructure; the live engine and round simulation run on the Windows
self-hosted runner labeled `golf-sim`. See
`.github/self-hosted-runner/README.md` for setup and availability behavior.

Manual preflight:

```bash
python midweek_round_automation.py --dry-run
```
- Uses single Google auth (1 connection, not 2)

**Backup & reprice cache:** `round_sim.py` automatically triggers the
`nightly-round-sim.yml` GitHub Actions workflow after every successful local
sim. This runs the sim on the Windows self-hosted runner and saves the cache to Actions
cache, so the overnight `reprice.yml` workflow can load it and re-price with
fresh odds for CLV checking. The nightly workflow also runs on schedule at
02:45 UTC (9:45 PM EST / 10:45 PM EDT on Thu/Fri/Sat) as a fallback. It reads the
already-current completed round from the Sheet, runs
`live_stats_engine.py --dry-run --no-sheet-writes`, then runs the full
`round_sim.py --dry-run`. The live-stats stage always regenerates the target
round prediction file; an existing root/dashboard copy is never trusted by the
backup. This deliberately creates both the round cache and a
fresh remaining-tournament simulation without sending email, storing bets,
writing Sheet config, or firing routine Telegram alerts. Before publishing, it
requires fresh and aligned round PMFs, rank probabilities, final-score,
made-cut, and R2/R3 standings tapes. The strict publisher atomically ships live
outrights, tournament/round matchup fairs, round score PMFs, and paired portfolio
tapes. Immediately before both transports it re-hashes the exact files bound by
`tournament_live_<tourney>_health.json`; a changed finish CSV, final-score tape,
name sidecar, or made-cut mask aborts. Full-resolution release tapes are uploaded
under immutable, hash-versioned names, and `sim_release_manifest.json` binds those
objects and every git artifact to the same approved simulation. The git commit is
the activation pointer, so a partial release upload is invisible to consumers.
Strict R2-R4 publishing also requires a conclusive DataGolf tee-group fetch (three
bounded attempts) with tee times for 100% of the active simulation field (cut and
withdrawn players are already excluded from that cache): it publishes either current 3-ball fairs or a current
event/round `no_groups_offered` contract and empty parquet, never a prior-round
file by omission. It then requests a
pinned model-only board refresh that preserves frozen book odds and suppresses
the bet-alert cascade. A stale R0 Sheet pointer,
incomplete artifact family, failed push, or failed board dispatch fails the job
and sends the workflow's single failure alert. The event-driven midweek workflow
must advance the Sheet round first.

**Bet-side health gate:** every round tape now carries a content-addressed
health manifest. Before any betting email, Sheets write, or Telegram bet alert,
the pipeline rechecks the active event/round, exact scoring-average target,
empirical field centering (within 0.12 strokes), player coverage (minimum 20 and
exact model/tape field alignment), simulation count and probability mass,
shot-dispersion provenance, and an 18-hour freshness limit. The lightweight
repricer additionally verifies byte-for-byte hashes for its H2H parquet and
metadata; cached live outright files are bound the same way. A failed check
exits nonzero and cannot be downgraded to a warning. Sheet-driven runs receive
automatic approval after all checks pass. CLI runs that can email/store require
`--health-approved-by <operator>`; this names the approver but never bypasses a
failed invariant. The manual `run-sim.yml` workflow supplies the GitHub actor.

**Decimal score-estimate refreshes:** a cached `--score-est` move uses
`uniform_rounding_bin_v1`, the same approximation as the odds-board control.
Each integer cache bucket is treated as uniform inside its one-stroke rounding
bin, so a 0.3-stroke move deterministically transfers 30% of its mass to the
adjacent settlement score. Whole-stroke moves retain exact legacy parity, while
uniformly shifted joint draws preserve matchup and 3-ball outcomes. During a
live event, a non-zero score-estimate refresh is not allowed to reuse its parent
outright tape: it requires current centered predictions and known-round inputs,
rebuilds the remaining-tournament/finish tape, binds both the score PMF and live
tape to the exact derived health manifest, and only then promotes the shifted
round cache. Missing inputs, a stale parent tape, or any partial artifact family
fails closed with a full-simulation requirement.

**Odds-screen R2 publication:** `push_odds_screen.py` writes each refresh beneath
`odds_data/generations/<generation>/` with byte hashes. The reprice workflow
uploads every immutable object first and updates root `odds_data/meta.json` last.
The odds-screen Worker keeps the legacy fixed market URLs but resolves them through
that pointer and verifies size/SHA-256 before returning data. Missing or corrupt
activated objects return 503; readers never fall back to an overwritten fixed key.

---

## Phase 5: Friday (Round 2)

### 5.1 Update Sheet & Run Skill Update

**Update Google Sheet:**

| Parameter | Value |
|-----------|-------|
| `round` | `2` |
| `wind` / `wind_r3` | R3 hourly wind forecast |
| `dew` / `dew_r3` | R3 hourly dewpoint forecast |
| `expected_score_1` | Actual R2 scoring avg |

```bash
python live_stats_engine.py
```

**What this does (round=2):**
- Fetches R2 live stats
- Applies R2 skill adjustments (3 buckets: top 5 / 6-30 / 30+)
- Components: `residual_adj + residual2_adj + residual3_adj + avg_ott_adj + avg_putt_adj + avg_app_adj + avg_arg_adj + delta_app_adj`
- Outputs: `r2_live_model.csv`, `model_predictions_r3.csv`

### 5.2 Run R3 Matchup Pricing
```bash
python round_sim.py
```
Same flow as Phase 4.3 but for R3.

---

## Phase 6: Saturday (Round 3)

### 6.1 Update Sheet & Run Skill Update

**Update Google Sheet:**

| Parameter | Value |
|-----------|-------|
| `round` | `3` |
| `wind` / `wind_r4` | R4 hourly wind forecast |
| `dew` / `dew_r4` | R4 hourly dewpoint forecast |
| `expected_score_1` | Actual R3 scoring avg |

```bash
python live_stats_engine.py
```

**What this does (round=3):**
- Fetches R3 live stats
- **CRITICAL**: UNDOES R2 SG + residual adjustments, then applies fresh R3 SG-only adjustments
- Formula: `total_adjustment = fresh_adj - prior_sg - prior_resid`
- Outputs: `r3_live_model.csv`, `model_predictions_r4.csv`

### 6.2 Run R4 Matchup Pricing
```bash
python round_sim.py
```

---

## Phase 7: Sunday (Round 4)

### 7.1 Optional: Run R4 Skill Update
Only needed if you want final-round predictions for analysis.

**Update Google Sheet:**

| Parameter | Value |
|-----------|-------|
| `round` | `4` |

```bash
python live_stats_engine.py
```

### 7.2 Post-Tournament: Grading is Automated
Grading runs automatically Monday morning via `monday-grading.yml` (see Phase 0.2). No manual action needed.

To grade immediately (Sunday night):
```bash
python monday_grading.py
```

### 7.3 Review Results
```bash
# Quick terminal summary
python bet_query.py --event genesis --graded

# Full season view
python bet_query.py --summary --by-event

# Interactive dashboard
python bet_query.py --plot

# Export for spreadsheet analysis
python bet_query.py --graded --export

# Web dashboard (includes fragility + performance round filter)
python -m dashboard.app   # → localhost:8050
# /fragility — review low-confidence bets
# /performance — filter by round (R1-R4) for round matchup analysis
```

---

## Phase 8: Post-Event SG Diagnostic

### 8.1 When to Run
Runs automatically as part of the Monday grading pipeline (`monday_grading.py`). Manual run only needed if you want the email report or need to override the event ID.

**Note:** Requires `avg_expected_cat_sg_{tourney}.csv` which is deleted by Sunday cleanup. The Monday pipeline runs `--no-email` and gracefully skips if the file is missing (GitHub Actions). Works fully when run locally before cleanup.

### 8.2 Run the Diagnostic
```bash
# Full run: fetch actuals, compare to predictions, store in Parquet, send email
python sg_diagnostic.py

# Override event ID (if sim_inputs.py already updated for next week)
python sg_diagnostic.py --event-id 5

# Local only (skip email)
python sg_diagnostic.py --no-email
```

**What this does:**
1. Reads `avg_expected_cat_sg_{tourney}.csv` (per-category SG predictions from the Monte Carlo sim)
2. Fetches actual round-level SG from DataGolf `historical-raw-data/rounds` API
3. Queries `dg_historical.db` (canonical location: `~/OneDrive/dg_historical.db`) for rolling player stats to classify archetypes (Long Bomber, Accurate Short, Ball Striker, etc.)
4. Computes prediction miss (actual - predicted) by category and archetype
5. Stores results in `permanent_data/sg_diagnostic.parquet` (persists across weeks)
6. Sends diagnostic email with category bias, biggest misses, and archetype analysis

**Note:** Only works for ShotLink-equipped events (~32/year). Non-ShotLink events will exit gracefully with a message.

### 8.3 Accumulated Cross-Event Report
After 2+ events, view trends across tournaments:
```bash
python sg_diagnostic.py --report
```

Shows overall category bias, players who are consistently mispredicted, and directional patterns.

---

## Quick Reference: Google Sheet Parameters

All parameters go in the `round_config` tab of the `golf_sims` Google Sheet (Column A = name, Column B = value).

| Parameter | Type | Description |
|-----------|------|-------------|
| `round` | int (0-4) | 0 = pre-event, 1-4 = round just completed |
| `expected_score_1` | float | Scoring avg for primary course |
| `expected_score_2` | float | Multi-course: 2nd course scoring avg |
| `expected_score_3` | float | Multi-course: 3rd course scoring avg |
| `wind` | comma-sep | Hourly wind array, 15 elements (6 AM - 8 PM) |
| `dew` | comma-sep | Hourly dewpoint array, 15 elements |
| `wind_r1` through `wind_r4` | comma-sep | Round-specific wind forecast (set by `humidity.py`, blank = use `wind`) |
| `dew_r1` through `dew_r4` | comma-sep | Round-specific dew forecast (set by `humidity.py`, blank = use `dew`) |
| `dew_calculation` | float | Dew effect factor (blank = use sim_inputs) |
| `wind_override` | float | 0 = use computed blend (blank = use sim_inputs) |
| `course_codes` | comma-sep | Course codes from API (e.g., "PB,SG") |
| `course_pars` | comma-sep | Par values matching course_codes order |
| `course_lat_lon` | `lat,lon` | Primary course coordinates used for weather and timezone |
| `course_timezone` | IANA name | Auto-resolved by Open-Meteo from `course_lat_lon` |
| `expected_score_r1` | float | Pre-tourney R1 scoring estimate (backup fallback) |
| `expected_score_r2` | comma-sep | R2 expected scoring (multi-course: comma-sep, backup fallback) |
| `expected_score_r3` | comma-sep | R3 expected scoring (backup fallback) |
| `expected_score_r4` | comma-sep | R4 expected scoring (backup fallback) |

---

## Quick Reference: Key Commands

```bash
# Pre-tournament pipeline
python humidity.py                               # Step 1: Auto-populate weather to Sheet (reads sim_inputs)
python update_sheet_courses.py                   # Step 1: Auto-populate course codes
python scoring_baseline.py                       # Step 2: Scoring baselines + expected scores (reads sim_inputs)
python init_weekly.py                            # Step 3: Push tourney/event_id/course_id/round=0 to Sheet
# Step 4: SG distributions + required shot_dispersion files arrive from sim_prep
python write_base_rates.py                       # Step 5: Base rates reference tab
python hole_baselines.py                         # ETR only: exact-course hole profile

# Pre-tournament sim (two-pass)
python new_sim.py --calibration-pass             # First pass (pre_course_fit; local artifacts only)
python mkt_regress.py                            # Market regression -> final_predictions
# Shadow comparison only: mkt_regress.py --local-only (no OneDrive/ETR handoff)
python new_sim.py --final-pass                   # Second pass (strict email/storage/publish)

# Live rounds
python live_stats_engine.py                      # Skill update (reads round from Sheet)
python round_sim.py                              # Round matchup pricing

# Bet grading
python grade_bets.py                             # Auto-detect last event
python grade_bets.py --event-id 5                # Specific event
python grade_bets.py --event-id 5 --dry-run      # Preview

# Bet analysis
python bet_query.py                              # All bets this year
python bet_query.py --event farmers              # Filter by event
python bet_query.py --type round_matchup         # Filter by bet type
python bet_query.py --book pinnacle              # Filter by book
python bet_query.py --min-edge 5                 # Edge >= 5%
python bet_query.py --graded                     # Only graded
python bet_query.py --summary --by-event         # Grouped by event
python bet_query.py --summary --by-book          # Grouped by book
python bet_query.py --export                     # Save to CSV
python bet_query.py --plot                       # Plotly dashboard
python bet_query.py --all-years                  # Include prior years

# Dashboard deploy (only when you need to push manually — Monday pipeline handles this)
python push_dashboard_data.py                    # Copy + commit + publish JSON to Cloudflare
python push_dashboard_data.py --dry-run          # Preview only

# Monday grading pipeline (automated via GitHub Actions, or run manually)
python monday_grading.py                         # Full pipeline: grade + diagnostic + deploy
python monday_grading.py --dry-run               # Preview checks only

# Nightly round sim backup (automated at 02:45 UTC Thu-Sat nights, or run manually)
python nightly_round_sim.py                      # Check + run if no bets stored yet
python nightly_round_sim.py --dry-run            # Preview checks only

# Dashboard (local)
python -m dashboard.app                          # localhost:8050

# SG diagnostic
python sg_diagnostic.py                          # Full diagnostic + email
python sg_diagnostic.py --event-id 5             # Specific event
python sg_diagnostic.py --no-email               # Local only
python sg_diagnostic.py --report                 # Cross-event trends
```

---

## Troubleshooting Checklist

### Simulation results look wrong
1. Verify `sim_inputs.py` has correct `event_ids`, `course_id`, `course_par`
2. Check weather arrays have 15 elements each
3. Run with `num_sims=1000` first to validate
4. Check `model_predictions_r1.csv` has expected players and reasonable `my_pred` values
5. Verify the `cat_mult_*` rows in the Sheet's `round_config` tab reflect this course — if stale, re-run `scoring_baseline.py` (it auto-writes them)

### Bets not saving to Sheets
1. Check `is_valid_run_time()` — storage only works after Monday 3 PM EST
2. Verify `credentials.json` exists and is valid
3. Check Google Sheet is shared with service account email
4. Look for `[storage]` print messages in output

### Bets not in Parquet ledger
1. Check `permanent_data/` directory exists
2. Verify with: `python -c "import pandas as pd; print(pd.read_parquet('permanent_data/bet_ledger.parquet').shape)"`
3. Look for `[ledger]` print messages in output

### Grading finds no bets
1. Verify event_id matches what was stored (check Sheets manually)
2. Check player name normalization — both sides must be lowercase
3. Verify DataGolf API returns results for that event_id

### Live stats engine fails
1. Verify Google Sheet `round_config` tab has correct `round` value
2. Check wind/dew arrays are properly comma-separated (no spaces after commas)
3. Verify DataGolf API key is valid: `echo $DATAGOLF_API_KEY`

### round_sim.py no matchup odds
1. DataGolf matchup odds API may not be available yet for upcoming round
2. Check that `model_predictions_r{N}.csv` exists and has data
3. Verify round number matches what you expect

---

## File Dependencies Map

```
sim_inputs.py ──────────────────────┐
                                    ▼
cat_dists_player.py (in sim_prep) ──► sg_dist_player.csv + this_week_dists_v2.csv
                                                              │
                                              pre_course_fit_{tourney}.csv
                                                              │
humidity.py ──► round_config (weather grid + formulas)        │
                              │                               │
                              ▼                               ▼
                    scoring_baseline.py ──► expected_score_r1-r4 (Sheet)
                              │
Google Sheet ──► sheet_config.py ──► live_stats_engine.py (round=0) ──► model_predictions_r1.csv
                                            │                                     │
                                            │                                     ▼
                                            │                              new_sim.py ──► Sheets + Ledger
                                            │
                                    live_stats_engine.py (round=1-3)
                                            │
                                            ├──► r{N}_live_model.csv
                                            └──► model_predictions_r{N+1}.csv
                                                         │
                                                         ▼
                                                  round_sim.py ──► Sheets + Ledger
                                                                        │
                                                                        ▼
                                                               grade_bets.py ──► Sheets + Ledger
                                                                        │
                                                                        ▼
                                                               bet_query.py ──► Terminal / CSV / HTML
```

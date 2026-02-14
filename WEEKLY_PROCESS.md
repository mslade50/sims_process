# Weekly Tournament Process: Step-by-Step Guide

Complete operational playbook for running the golf simulation system from Sunday cleanup through post-tournament grading.

---

## Phase 0: Sunday Night Cleanup & Prior Week Grading

### 0.1 Automatic Cleanup (GitHub Action)
The `weekly-cleanup.yml` workflow runs Sunday at midnight UTC. It deletes:
- All CSVs, Excel files, and tournament folders from the repo root
- `permanent_data/` and `.py` files are preserved

**Verify cleanup ran:**
```bash
git pull
ls *.csv *.xlsx  # should return nothing
```

### 0.2 Grade Previous Week's Bets
After the Sunday tournament finishes, grade all bets from the week.

**Auto-detect last event:**
```bash
python grade_bets.py
```

**Specific event (if auto-detect fails):**
```bash
python grade_bets.py --event-id 5 --event-name "AT&T Pebble Beach"
```

**What this does:**
1. Connects to Google Sheets (single auth via `get_spreadsheet()`)
2. Reads ungraded bets from Tournament Matchups, Round Matchups, Finish Positions tabs
3. Deduplicates bets (keeps first occurrence per key)
4. Fetches tournament results from DataGolf `historical-raw-data/rounds` API
5. Grades each bet (win/loss/push with dead-heat adjustments for finish positions)
6. Writes results back to source Sheets tabs
7. Writes detailed results to Sharp/Retail/Other results tabs
8. **Updates the Parquet ledger** (`permanent_data/bet_ledger.parquet`) with grades
9. Calculates performance metrics and writes to Bet Results Summary tab
10. Sends two email reports: full results + filtered (pred > 0.75)

**Preview without writing:**
```bash
python grade_bets.py --event-id 5 --dry-run
```

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

### 1.2 Run Distribution Builder (if new SG data available)
```bash
python cat_dists_player.py
```

**What this does:**
- Reads `field_adjusted_sg.csv` (PGA rounds since 2019)
- Computes EMA-weighted SG distributions per player per category
- Outputs `sg_dist_player.csv`
- Uses alpha = 2/(50+1), effective half-life ~17 rounds
- Minimum 20 observations per category

**When to skip:** If no new historical data has been added since last run.

### 1.3 Run Distribution Adjustment
```bash
python dists_thiswk.py
```

**What this does:**
1. Fetches current field from DataGolf `/field-updates` API
2. Filters `sg_dist_player.csv` to players in this week's field
3. Applies course shape adjustments from `course_shape_adjustments_{course_id}.csv`:
   - `delta_mu`: Mean shift per SG category
   - `sigma_ratio`: Variance scaling
   - `tail_ratio`: Kurtosis adjustment (Student-t df)
4. Outputs:
   - `this_week_dists.csv` (pre-adjustment)
   - `this_week_dists_adjusted.csv` (post-adjustment)
5. Syncs to OneDrive targets

**Required input files:**
- `sg_dist_player.csv` (from Step 1.2)
- `course_shape_adjustments_{course_id}.csv` (in repo or permanent_data)

**Verify:**
```python
import pandas as pd
df = pd.read_csv("this_week_dists_adjusted.csv")
print(f"Players: {df['player_name'].nunique()}")
print(df[['player_name', 'sg_total_mean', 'sg_total_std']].head(10))
```

---

## Phase 2: Wednesday Pre-Tournament Simulation

### 2.1 Update Weather Forecasts in `sim_inputs.py`
Get hourly wind and dewpoint forecasts for R1 through R4. Arrays start at 6 AM, need 15 elements (6 AM - 8 PM).

```python
# Example: hourly wind speeds (MPH)
wind_1 = [5, 5, 6, 8, 10, 12, 14, 15, 14, 12, 10, 8, 7, 6, 5]
wind_2 = [3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5]
wind_3 = [...]
wind_4 = [...]

# Example: hourly dewpoint (F)
dewpoint_1 = [42, 42, 43, 44, 45, 46, 47, 48, 48, 47, 46, 45, 44, 43, 42]
dewpoint_2 = [...]
dewpoint_3 = [...]
dewpoint_4 = [...]
```

**Sources:** Weather.com, WeatherUnderground, or DarkSky hourly forecast.

### 2.2 Run R1 Hole-by-Hole Simulation
```bash
python rd_1_sd_multicourse_sim.py
```

**What this does:**
- Reads `this_week_dists_adjusted.csv` and course hole distributions
- For each simulation (num_sims iterations):
  - Adjusts player distributions per hole (mean shift + variance scaling)
  - Applies wind, dewpoint, scoring adjustments
  - Simulates all 18 holes
  - Computes DraftKings fantasy points
- Outputs:
  - `rd_1_results_sd_{tourney}.csv` (aggregated player stats)
  - `hole_lvl_sd_1_{tourney}.csv` (per-sim, per-hole scores)
  - `model_predictions_r1.csv`

**Test first with small num_sims:**
```python
# In sim_inputs.py temporarily:
num_sims = 1000  # Then bump to 50000 for production
```

**Verify:**
```python
df = pd.read_csv(f"rd_1_results_sd_{tourney}.csv")
print(f"Players: {len(df)}")
print(df[['player_name', 'mean_score', 'dk_mean']].head(10))
# Check distributions aren't identical
print(f"Score range: {df['mean_score'].min():.2f} to {df['mean_score'].max():.2f}")
```

### 2.3 Update Google Sheet Weather
Open the `golf_sims` Google Sheet, `round_config` tab:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `round` | `0` | Pre-event |
| `expected_score_1` | `71.5` | Expected scoring avg, course 1 |
| `expected_score_2` | `72.0` | Multi-course only, blank if single |
| `wind` | `5,5,6,8,10,...` | R1 wind, comma-separated |
| `dew` | `42,42,43,44,...` | R1 dewpoint, comma-separated |
| `dew_calculation` | leave blank | Falls back to sim_inputs |
| `wind_override` | `0` | 0 = use computed blend |
| `course_codes` | `TS` or `PB,SG` | Auto-populated or manual |
| `course_pars` | `71` or `72,72` | Matching course_codes order |

---

## Phase 3: Thursday (Round 1)

### 3.1 Pre-Round: Generate Model Predictions
Before R1 tee times, with `round=0` in the Google Sheet:

```bash
python live_stats_engine.py
```

**What this does (round=0):**
- Reads config from Google Sheet (`round_config` tab)
- Fetches field updates from DataGolf API
- Generates `model_predictions_r1.csv` with pre-tournament skill estimates
- No skill adjustments applied (pre-event baseline)

**Verify:**
```python
df = pd.read_csv("model_predictions_r1.csv")
print(f"Players: {len(df)}")
print(df[['player_name', 'my_pred', 'std_dev']].head(10))
```

### 3.2 Pre-Round: Run Tournament Sim
```bash
python new_sim.py
```

**What this does:**
- Fetches betting odds from DataGolf matchup/outright APIs
- Runs Monte Carlo tournament simulation (matchups + finish positions)
- Calculates edges vs book odds
- Sends email report with filtered bets
- **Auto-saves to Google Sheets** (Tournament Matchups, Finish Positions, Sharp Filtered, All Filtered tabs)
- **Auto-writes to Parquet ledger** (`permanent_data/bet_ledger.parquet`)
- Uses single Google auth via `get_spreadsheet()` (1 connection, not 4)

**Storage only runs after Monday 3 PM EST** (time gate in `is_valid_run_time()`).

**Verify bet storage:**
```python
import pandas as pd
df = pd.read_parquet("permanent_data/bet_ledger.parquet")
print(f"Ledger: {len(df)} rows")
print(df[df['event_name'] == 'att'][['bet_type', 'bet_on', 'bookmaker', 'edge']].head())
# Check no duplicates
print(f"Duplicates: {df.duplicated(subset=['event_id','bet_type','round','bet_on','opponent','bookmaker']).sum()}")
```

### 3.3 Post-Round 1: Update Sheet & Run Skill Update

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

### 3.4 Post-Round 1: Run R2 Matchup Pricing
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
- **Auto-saves to Google Sheets** (Round Matchups, Sharp Filtered tabs)
- **Auto-writes to Parquet ledger**
- Uses single Google auth (1 connection, not 2)

---

## Phase 4: Friday (Round 2)

### 4.1 Update Sheet & Run Skill Update

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

### 4.2 Run R3 Matchup Pricing
```bash
python round_sim.py
```
Same flow as Phase 3.4 but for R3.

---

## Phase 5: Saturday (Round 3)

### 5.1 Update Sheet & Run Skill Update

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

### 5.2 Run R4 Matchup Pricing
```bash
python round_sim.py
```

---

## Phase 6: Sunday (Round 4)

### 6.1 Optional: Run R4 Skill Update
Only needed if you want final-round predictions for analysis.

**Update Google Sheet:**

| Parameter | Value |
|-----------|-------|
| `round` | `4` |

```bash
python live_stats_engine.py
```

### 6.2 Post-Tournament: Grade All Bets
Once the tournament is final:

```bash
python grade_bets.py --event-id <id>
```

Or wait for Sunday night and run with auto-detect:
```bash
python grade_bets.py
```

### 6.3 Review Results
```bash
# Quick terminal summary
python bet_query.py --event genesis --graded

# Full season view
python bet_query.py --summary --by-event

# Interactive dashboard
python bet_query.py --plot

# Export for spreadsheet analysis
python bet_query.py --graded --export
```

---

## Phase 7: Post-Event SG Diagnostic

### 7.1 When to Run
After the tournament completes (Sunday evening or Monday), but **BEFORE the weekly cleanup** (which deletes `avg_expected_cat_sg_{tourney}.csv`).

### 7.2 Run the Diagnostic
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
3. Queries `dg_historical.db` for rolling player stats to classify archetypes (Long Bomber, Accurate Short, Ball Striker, etc.)
4. Computes prediction miss (actual - predicted) by category and archetype
5. Stores results in `permanent_data/sg_diagnostic.parquet` (persists across weeks)
6. Sends diagnostic email with category bias, biggest misses, and archetype analysis

**Note:** Only works for ShotLink-equipped events (~32/year). Non-ShotLink events will exit gracefully with a message.

### 7.3 Accumulated Cross-Event Report
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
| `wind_r2` through `wind_r4` | comma-sep | Round-specific wind (blank = use `wind`) |
| `dew_r2` through `dew_r4` | comma-sep | Round-specific dew (blank = use `dew`) |
| `dew_calculation` | float | Dew effect factor (blank = use sim_inputs) |
| `wind_override` | float | 0 = use computed blend (blank = use sim_inputs) |
| `course_codes` | comma-sep | Course codes from API (e.g., "PB,SG") |
| `course_pars` | comma-sep | Par values matching course_codes order |
| `expected_score_r2` | comma-sep | R2 expected scoring (multi-course) |
| `expected_score_r3` | comma-sep | R3 expected scoring (multi-course) |
| `expected_score_r4` | comma-sep | R4 expected scoring (multi-course) |

---

## Quick Reference: Key Commands

```bash
# Pre-tournament pipeline
python cat_dists_player.py                       # Step 1: SG distributions
python dists_thiswk.py                           # Step 2: Field filter + course adjust
python rd_1_sd_multicourse_sim.py                # Step 3: Hole-by-hole R1 sim

# Live rounds
python live_stats_engine.py                      # Skill update (reads round from Sheet)
python new_sim.py                                # Tournament matchups + finish positions
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
5. Verify course shape adjustment file exists: `course_shape_adjustments_{course_id}.csv`

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
cat_dists_player.py ──► sg_dist_player.csv
                              │
                              ▼
dists_thiswk.py ──► this_week_dists_adjusted.csv
                              │
                              ├──► rd_1_sd_multicourse_sim.py ──► model_predictions_r1.csv
                              │                                            │
                              │                                            ▼
                              │                                     new_sim.py ──► Sheets + Ledger
                              │
Google Sheet ──► sheet_config.py ──► live_stats_engine.py
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

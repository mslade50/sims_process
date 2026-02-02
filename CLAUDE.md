# Golf Tournament Monte Carlo Simulation System

> **Purpose**: This document provides context for AI assistants (Claude) to quickly understand the codebase structure, data flows, and key patterns.

---

## Project Overview

A Monte Carlo simulation system for golf tournament prediction and DFS (DraftKings) analysis. The system generates player-specific score distributions by combining:
- Historical strokes gained (SG) data with EMA weighting
- Course-specific shape adjustments
- Real-time weather effects (wind, dewpoint)
- Player variance profiles

---

## File Inventory & Responsibilities

### Core Pipeline (execution order)

| Step | File | Purpose | Key Outputs |
|------|------|---------|-------------|
| 1 | `cat_dists_player.py` | Build EMA-weighted SG distributions from historical data | `sg_dist_player.csv` |
| 2 | `dists_thiswk.py` | Filter to field + apply course shape adjustments | `this_week_dists.csv`, `this_week_dists_adjusted.csv` |
| 3 | `rd_1_sd_multicourse_sim.py` | Run hole-by-hole Monte Carlo for Round 1 | `rd_1_results_sd_{tourney}.csv`, `hole_lvl_sd_1_{tourney}.csv` |

### External Config (NOT in repo - local)
- **`sim_inputs.py`** - Tournament-specific configuration. Contains:
  - `tourney` (str): Tournament identifier
  - `course_id`, `course_id_1`, `course_id_2`: Course identifiers
  - `course_name`: Course name string
  - `event_ids`: List of DataGolf event IDs
  - `num_sims`: Simulation count (typically 10,000)
  - `wind_1`, `wind_2`: Hourly wind arrays (24-element lists)
  - `dewpoint_1`, `dewpoint_wave`: Dewpoint arrays
  - `baseline_wind`, `baseline_dew`: Baseline weather coefficients
  - `wind_override`: Override value (0 = use calculated)
  - `dew_calculation`: Dewpoint adjustment coefficient
  - `player_var`: Variance adjustment factor
  - `score_adj_r1_sd`: Round 1 scoring adjustment
  - `name_replacements`: Dict for normalizing player names

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HISTORICAL DATA LAYER                            │
├─────────────────────────────────────────────────────────────────────────┤
│  field_adjusted_sg.csv                                                  │
│  (PGA rounds ≥2019, sg_*_adj columns)                                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  cat_dists_player.py                                                    │
│  ─────────────────────                                                  │
│  • EMA weighting (span=50) by round_date                                │
│  • Per-player, per-category: sg_ott_adj, sg_app_adj, sg_arg_adj,        │
│    sg_putt_adj                                                          │
│  • Min 20 observations per category                                     │
│  • Outputs: mean, std, skew, excess_kurtosis, quantiles, histogram JSON │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼ sg_dist_player.csv
                               │
┌──────────────────────────────┼──────────────────────────────────────────┐
│  dists_thiswk.py             │                                          │
│  ─────────────────           │                                          │
│  PART 1: Field filter        │◄── field_updates.csv                     │
│  • Normalize names           │                                          │
│  • Keep only this week's     │                                          │
│    field                     │                                          │
│  • Report missing players    │                                          │
│                              ▼                                          │
│  PART 2: Course adjustments ◄── course_shape_adjustments_{course_id}.csv│
│  • delta_mu (mean shift)     │                                          │
│  • sigma_ratio (variance)    │                                          │
│  • tail_ratio (kurtosis)     │                                          │
│  • Sanity checks on extremes │                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼ this_week_dists_adjusted.csv
                               │
┌──────────────────────────────┼──────────────────────────────────────────┐
│  rd_1_sd_multicourse_sim.py  │                                          │
│  ────────────────────────────┤                                          │
│                              │◄── adj_hole_dist_{tourney}_{course_id}.csv
│  1. Load hole distributions  │◄── final_predictions_{tourney}.csv       │
│  2. Calculate weather adj    │◄── pre_sim_summary_{tourney}.csv         │
│     • Wind per tee time      │◄── wind_test.csv (historical course wind)│
│     • Dewpoint centered      │                                          │
│  3. Build player-specific    │                                          │
│     distributions            │                                          │
│     • Skill adjustment       │                                          │
│     • Weather adjustment     │                                          │
│     • Variance adjustment    │                                          │
│  4. Parallel simulation      │                                          │
│     (multiprocessing)        │                                          │
│  5. Aggregate results        │                                          │
│  6. Merge multi-course       │                                          │
│     outputs                  │                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
    rd_1_results_sd_{tourney}.csv    hole_lvl_sd_1_{tourney}.csv
    (aggregated player stats)        (per-sim, per-hole scores)
```

---

## Key Algorithms & Patterns

### 1. EMA Weighting (cat_dists_player.py, lines 78-93)
```python
# Most recent observations weighted highest
alpha = 2.0 / (span + 1.0)   # span=50 → alpha≈0.039
weights = (1-alpha)^(n-1-i)  # i=0 oldest, i=n-1 newest
```
Effective half-life ≈ 17 rounds.

### 2. Weather Adjustment Pattern (rd_1_sd_multicourse_sim.py)
```python
# Wind: absolute adjustment to score
wind_adj = avg_wind_during_round * wind_coefficient

# Dewpoint: CENTERED (individual - mean)
dew_adj = player_dew_raw - field_avg_dew  # Relative advantage/disadvantage
```
**Critical**: Dewpoint is always mean-centered; wind is not.

### 3. Distribution Adjustment (rd_1_sd_multicourse_sim.py, lines 14-48)
For each hole, player distributions are warped by:
1. **Mean shift**: skill + weather + scoring adjustment
2. **Variance scaling**: based on player std_dev vs field average
Uses `scipy.optimize.minimize` to preserve distribution shape while hitting target mean.

### 4. Multi-Course Handling (rd_1_sd_multicourse_sim.py, lines 491-550)
Tournaments like AT&T Pebble Beach use multiple courses. The script:
- Runs simulation per course_id
- Merges results at the end
- Handles cases where only one course file exists

### 5. DraftKings Scoring (rd_1_sd_multicourse_sim.py, lines 117-161)
```python
scoring = {
    -3: 16,    # Double eagle+
    -2: 11,    # Eagle
    -1: 5.75,  # Birdie
     0: 1.5,   # Par
     1: -1.8,  # Bogey
     2: -3.9   # Double bogey+
}
# Bonuses: +5 for birdie streak (3+), +5 for bogey-free round
```

---

## Column Conventions

### Player Predictions (`final_predictions_{tourney}.csv`, `model_predictions_r1.csv`)
| Column | Description |
|--------|-------------|
| `player_name` | Lowercase normalized name |
| `my_pred` | Predicted strokes vs field (negative = better) |
| `std_dev` | Player's round-to-round volatility |
| `r1_teetime` | Round 1 tee time (multiple formats supported) |
| `wind_adj1` | Wind effect in strokes |
| `dew_adj1` | Centered dewpoint effect |
| `scores_r1` | Expected Round 1 score |

### Hole Distributions (`adj_hole_dist_{tourney}_{course_id}.csv`)
| Column | Description |
|--------|-------------|
| `Round` | Round number (1-4) |
| `Hole` | Hole number (1-18) |
| `Par` | Hole par value |
| `mean`, `adj_mean` | Average score on hole |
| `1`, `2`, `3`, ... | Count of each score |

### Simulation Outputs
| Column | Description |
|--------|-------------|
| `DraftKings_Points` | Mean DK points across sims |
| `Total_Score` | Mean strokes |
| `frl_%` | First round leader probability |
| `wind_benefit` | Relative wind advantage vs field |

---

## Common Issues & Solutions

### 1. NaN Cascades in Multi-Course Tournaments
**Symptom**: Some players have NaN predictions after merging courses.
**Cause**: Incomplete ShotLink coverage on one course.
**Solution**: Filter by data availability, not just player status. Check for NaN in source files before simulation.

### 2. Weather Centering Errors
**Symptom**: All players show same weather adjustment direction.
**Wrong**: `mean - individual`
**Correct**: `individual - mean` (positive = player faces worse conditions than average)

### 3. Distribution Identity Check Fails
**Symptom**: "All player distributions are identical!" warning (line 444-447)
**Cause**: Adjustments not applying correctly.
**Debug**: Check that `variance_multiplier`, `skill_adj`, `wind_adj` vary across players.

### 4. Tee Time Parsing
The system handles multiple formats (lines 52-65):
- `%Y-%m-%d %H:%M` (standard)
- `%I:%M%p` (e.g., "1:55PM")
- `%m/%d/%Y %H:%M` (e.g., "1/2/2025 9:33")

---

## Environment & Dependencies

```python
# Core
pandas >= 1.5
numpy >= 1.24
scipy >= 1.10

# Database (cat_dists_player.py only - currently using CSV instead)
sqlite3  # Optional, DB_PATH defined but CSV preferred

# Parallelization
multiprocessing  # Built-in, uses mp.cpu_count()
```

### File Paths (Windows-specific in current code)
- Database: `C:/Users/mckin/OneDrive/dg_historical.db`
- Sync targets: `C:\Users\mckin\OneDrive\sims_process`, `C:\Users\mckin\OneDrive\etr-golf-sims`

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
   # Should be False
   print(all(all_distributions[0].equals(dist) for dist in all_distributions[1:]))
   ```

4. **Check for unexpected NaNs**
   ```python
   print(model_predictions.isna().sum())
   ```

---

## Extension Points

### Adding a new weather variable
1. Add array to `sim_inputs.py` (e.g., `humidity_1`)
2. Add calculation loop in simulation file (parallel to wind/dew)
3. Add adjustment column (`humidity_adj1`)
4. Include in `player_adj_mean` calculation (line 402-407)

### Adding a new SG category
1. Add column name to `CATS` list in `cat_dists_player.py`
2. Ensure column exists in source data with `_adj` suffix
3. Course adjustments will need corresponding row in `course_shape_adjustments_{course_id}.csv`

---

## Notes for Claude

**When reviewing this codebase:**
1. Always check `sim_inputs.py` contents first—it controls all tournament-specific behavior
2. The formula `Post = Pre + total_adjustment` must be consistent across all rounds
3. Weather centering pattern: `individual - mean`, not `mean - individual`
4. Multi-course tournaments require checking both course files before diagnosing issues
5. Line numbers referenced above are approximate and may shift with edits

**When suggesting changes:**
- Preserve existing naming conventions (lowercase player names, underscore-separated columns)
- Maintain the CSV-based pipeline (database path exists but isn't primary)
- Test with small `num_sims` (100-1000) before full runs
- Multi-course logic at end of simulation file is fragile—test both single and dual-course scenarios

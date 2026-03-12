# Migration Guide: Unified Live Stats Engine

## Files Overview

| New File | Replaces | Purpose |
|----------|----------|---------|
| `api_utils.py` | Duplicated fetch functions in all 4 files | Shared DataGolf API + wind/dew calculation |
| `live_stats_engine.py` | `live_stats.py`, `live_stats_r2.py`, `live_stats_r3.py`, `live_stats_r4.py`, prediction logic from `rd_1_sd_multicourse_sim.py` | Single parameterized processor for all rounds |

## Files NOT Changed

- `sim_inputs.py` — No changes. All coefficients, arrays, and config imported as-is.
- `new_sim.py` — Stays independent. Tournament-level sim is already unified.
- `rd_1_sd_multicourse_sim.py` — Can be archived. Its prediction logic is now in the engine.

---

## Usage

### Default Mode: Google Sheets (no terminal input needed)

Update the `round_config` tab in your `golf_sims` Google Sheet, then just run:
```bash
python live_stats_engine.py
```
The script reads everything it needs from the sheet: round number, step, wind/dew arrays, scoring adjustments.

#### Sheet Layout (`round_config` tab)
| Row | Parameter | Example Value | Notes |
|-----|-----------|---------------|-------|
| 2 | round | 1 | Round that just completed (0=pre-event, 1=predict R2, etc.) |
| 3 | expected_score_1 | 0.15 | Scoring adj, 1st course in API data (or only course) |
| 4 | expected_score_2 | | 2nd course — leave blank for single-course weeks |
| 5 | expected_score_3 | 0 | 3rd course — leave blank for single-course weeks |
| 6 | wind | 5,5,5,5,5,5,5,5,5,5,5,5,5,5,5 | Hourly wind forecast for NEXT round (6AM onward) |
| 7 | dew | 36,36,38,38,38,38,38,40,40,42,44,44,44,44,44 | Hourly dewpoint forecast for NEXT round |
| 8 | wind_paste | | Optional alternate wind source |
| 9 | dew_paste | | Optional alternate dew source |

**Multi-course note:** The engine maps `expected_score_1` to the first `course_x` value it finds in the API data, `expected_score_2` to the second, etc. After running, it prints which course ID mapped to which number so you can verify and swap if needed.

#### Typical Tournament Week Flow

**Pre-event (Wednesday):**
Set `round=0`, enter R1 wind/dew forecast → Run script

**After R1 (Thursday evening):**
Set `round=1`, update wind/dew to R2 forecast → Run script
(R2 tee times already exist, so skill + predictions created in one go)

**After R2 (Friday evening):**
Set `round=2`, update wind/dew to R3 forecast → Run script
- Skill update runs immediately
- If R3 tee times are available → predictions created automatically
- If not yet → script tells you, run again later when they populate

**After R3 (Saturday evening):**
Same as R2: set `round=3`, update forecasts → Run script

**After R4 (Sunday — optional):**
Set `round=4` → Run for record-keeping only

### Fallback Mode: CLI Args
If the sheet is unavailable or you prefer terminal control:
```bash
python live_stats_engine.py --cli --round 2
```

---

## Validation Checklist

Before retiring old files, validate each round's output matches:

### R1 Validation
1. Run old: `live_stats.py` → `r1_live_model_OLD.csv`
2. Run new: `python live_stats_engine.py --round 1 --step skill` → `r1_live_model.csv`
3. Compare key columns:
   ```python
   old = pd.read_csv('r1_live_model_OLD.csv')
   new = pd.read_csv('r1_live_model.csv')
   compare_cols = ['player_name', 'updated_pred', 'total_adjustment',
                   'residual', 'weather_signal', 'tot_resid_adj']
   merged = old[compare_cols].merge(new[compare_cols], on='player_name', suffixes=('_old', '_new'))
   for col in compare_cols[1:]:
       diff = (merged[f'{col}_old'] - merged[f'{col}_new']).abs()
       print(f"{col}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
   ```

### R2 Validation
1. Run old pipeline through R2
2. Compare `updated_pred_r3` between old `r2_live_model.csv` and new
3. Check `total_adjustment` is consistent

### R3/R4 Validation
1. Compare `updated_pred_r{N+1}` between old and new
2. Verify coefficient adjustments match

### Weather Predictions Validation
1. Compare `model_predictions_r2.csv` from new engine vs old rd_1_sd_multicourse_sim output
2. Key columns: `scores_r2`, `wind_adj2`, `dew_adj2`

---

## Known Assumptions to Verify

### 1. R3/R4 Adjustment Column Naming
**In old code**: `live_stats_r3.py` creates columns like `sg_ott_avg_adj_r3`
**In new code**: The unified engine uses the same `{key}_adj_r{N}` suffix pattern
**How to verify**: Check that `total_adjustment` sums match

### 2. Multi-Course Residuals for R2/R3
**New behavior**: If `course_x` has multiple values in R2/R3, the engine computes
course-specific residuals and splines (same as R1 does)
**Old behavior**: R2/R3 files did NOT do course-specific processing
**Impact**: Should be more accurate on multi-course weeks; identical on single-course weeks

---

## Architecture Diagram

```
    ┌─────────────────────────────┐
    │  Google Sheet: golf_sims    │  ← Update from phone/laptop
    │  Tab: round_config          │
    │                             │
    │  round = 2                  │
    │  step = skill               │
    │  wind = 5,5,5,5,...         │
    │  dew = 36,36,38,...         │
    │  expected_score_1 = 0.15    │
    └──────────────┬──────────────┘
                   │
                   ▼
         python live_stats_engine.py   ← Just hit run
                   │
          ┌────────┴────────┐
          │  sheet_config   │  Reads sheet, returns config dict
          └────────┬────────┘
                   │
```
                    PRE-EVENT
                        │
    final_predictions + pre_sim_summary
                        │
                  ┌─────▼──────┐
                  │ --pre-event │  Creates model_predictions_r1.csv
                  └─────┬──────┘
                        │
                   R1 PLAYS OUT
                        │
              ┌─────────▼──────────┐
              │   --round 1        │  Fetches R1 live stats from API
              │   (full pipeline)  │  Computes residuals + spline
              │                    │  Applies R1 coefficients (skill buckets)
              │                    │  Leaderboard gravity
              │                    │  Creates R2 weather predictions
              └────┬──────────┬────┘
                   │          │
        r1_live_model.csv  model_predictions_r2.csv
                   │          │
              [round_sim.py for R2 matchup + score pricing]
                              │
                        R2 PLAYS OUT
                              │
              ┌───────────────▼───────────────┐
              │  --round 2 --step skill       │  Immediate skill update
              │  (inspect outputs)            │  
              └───────────────┬───────────────┘
                              │
                    r2_live_model.csv
                              │
              ┌───────────────▼───────────────┐
              │  --round 2 --step weather     │  After tee times available
              └───────────────┬───────────────┘
                              │
                   model_predictions_r3.csv
                              │
              [round_sim.py for R3 matchup + score pricing]
                              │
                        R3 PLAYS OUT
                              │
                    (same pattern as R2)
                              │
                        R4 PLAYS OUT
                              │
              ┌───────────────▼───────────────┐
              │  --round 4                    │  Record-keeping only
              └───────────────────────────────┘
```

---

## Next Steps (Phase 2)

Once live_stats_engine.py is validated:

1. **Merge round_mu_sim + round_scores** → `round_sim.py`
   - Multi-course aware (per-course expected scoring avg)
   - Score pricing output: CSV with fair UNDER prices per half-stroke line
   - Matchup pricing output: Fair odds against book lines

2. **Main orchestrator** → `main.py`
   - Runs: live_stats_engine → round_sim sequentially
   - Sets all variables in one place

3. **Automation** (Phase 3)
   - Scheduled runs triggered by round completion
   - HTML email with edge tables
   - Telegram bot for bet alerts
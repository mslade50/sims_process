# Round-by-Round Simulation Mechanics

How skill updates work at each stage, where data comes from, and what you need to touch.

---

## Prediction Chain

Each round updates a prediction column that feeds the next:

| After round | Input prediction | Output prediction |
|-------------|-----------------|-------------------|
| Pre-event | `final_predictions` / `pre_course_fit` | `model_predictions_r1.csv` |
| R1 | `pred` | `updated_pred` |
| R2 | `updated_pred` | `updated_pred_r3` |
| R3 | `updated_pred_r3` | `updated_pred_r4` |
| R4 | `updated_pred_r4` | `updated_pred_final` |

Invariant: `Post = Pre + total_adjustment` for every round.

**Reset-to-base form (2026-08)**: R2/R3/R4 rebuild the prediction from
`base_pred = pred + pin_high_adj + gravity_adj` (the only adjustments that
persist across rounds) plus this round's fresh adjustments; `total_adjustment`
is *derived* as `Post − Pre`, so the invariant holds by construction. There is
no carried-column undo to go stale. `base_pred` is written to every live model
and carried forward; if a prior file lacks it (pre-refactor), the engine
reconstructs it via the old undo identity.

---

## Pre-Event (round=0)

**Data sources**: `final_predictions_{tourney}.csv` (or fallback `pre_course_fit_{tourney}.csv`), tee times from DataGolf `/field-updates`, wind/dew arrays from Sheet.

**What it does**: No skill adjustment. Merges predictions with R1 weather exposure (per-player wind/dew based on tee time) and outputs `model_predictions_r1.csv`.

**You update**: Set `round=0` in Sheet. Weather arrays should already be populated by `humidity.py`.

**Then run**: `new_sim.py` (tournament matchups + finish positions) and `round_sim.py` (R1 round matchups).

> `round=0` is used for ALL R1 operations. Don't change to 1 until R1 is finished.

---

## R1 Skill Update (round=1)

**Data sources**: R1 actual SG from DataGolf live stats API, ShotLink category data (OTT, putt) when available.

**Bucketing**: 4 skill-based buckets by pre-tournament `pred` value (>1, 0.5-1, -0.5-0.5, <-0.5). Each bucket has its own coefficient set.

**Adjustment components**:
- `ott_adj` — OTT strokes gained (ShotLink only)
- `putt_adj` — Putting SG (ShotLink only)
- `tot_resid_adj` — Residual (actual SG minus predicted, weather-adjusted via B-spline)

**Residual caps**: If raw residual is negative but adjustment > 0.2 → capped at 0.2. Hard cap at 0.5 always.

**Weather spline**: B-spline fit to residuals vs tee time extracts wave/weather signal so the residual reflects true skill delta, not just "played in hard conditions."

**Formula**: `total_adjustment = ott_adj + putt_adj + tot_resid_adj`

**You update**: `round=1`, `expected_score_1` = actual R1 scoring avg, R2 wind/dew arrays, `realized_wind_r1`.

---

## R2 Skill Update (round=2)

**Data sources**: R1+R2 actual SG from DataGolf, full ShotLink category averages across rounds played.

**Bucketing**: Switches to 3 position-based buckets by leaderboard position (top 5, 6-30, 30+).

**Adjustment components** (8 total — more than R1):
- `residual_adj`, `residual2_adj`, `residual3_adj` — Linear, squared, cubic residual terms
- `avg_ott_adj`, `avg_putt_adj`, `avg_app_adj`, `avg_arg_adj` — Category SG averages across rounds
- `delta_app_adj` — Change in approach SG from R1→R2 (multi-course signal)

**Residual floor**: Clipped at -0.5 (no hard ceiling like R1).

**Formula**: `updated_pred_r3 = base_pred + tot_resid_adj + tot_sg_adj` (fresh components only — R1's adjustments drop out via the base reset; `total_adjustment` = Post − Pre)

**You update**: `round=2`, `expected_score_1` = actual R2 scoring avg, R3 wind/dew arrays, `realized_wind_r2`.

---

## R3 Skill Update (round=3) — The Critical One

**Data sources**: R1-R3 actual SG from DataGolf, cumulative SG category averages.

**Bucketing**: 3 position-based buckets (top 5, 6-20, 20+) — tighter mid-range than R2.

**The base reset**: R3's fresh adjustments REPLACE R2's — the prediction is rebuilt from `base_pred` rather than undoing R2's components term-by-term (`updated_pred_r3` has R2's adjustments baked in; applying R3 on top would double-count).

```
fresh_adj = sg_ott_avg_adj + sg_putt_avg_adj + sg_app_avg_adj + sg_arg_avg_adj
updated_pred_r4 = base_pred + fresh_adj
total_adjustment = updated_pred_r4 - updated_pred_r3   (derived)
```

**Key difference**: SG-only. No residual terms. Coefficients are applied to cumulative SG category averages, not single-round values.

**You update**: `round=3`, `expected_score_1` = actual R3 scoring avg, R4 wind/dew arrays, `realized_wind_r3`.

---

## R4 Skill Update (round=4) — Optional

Same base-reset logic as R3. Reuses R3 coefficients. Outputs `updated_pred_final`. No next-round predictions. Mainly for backtesting and diagnostics.

**You update**: `round=4`, `realized_wind_r4`.

---

## Actuals Writeback

Runs automatically after each skill update. Writes a row per round to the Sheet (`round_config`, rows 11-14):

| What | Source |
|------|--------|
| Forecast dew | Sheet dew array avg BEFORE Open-Meteo refresh |
| Realized dew | Sheet dew array avg AFTER refresh (now actuals) |
| Realized wind | `realized_wind_r{N}` — **you enter this manually** |
| Wind impact | `realized_wind * wind_coefficient` |
| Dew impact | `(realized_dew - dewpoint_base) * dew_calc` |
| Expected score | `base_score + field_adj + wind_impact + dew_impact` |
| Actual score | Field scoring avg from DataGolf live stats API |
| Delta | `actual - expected` |

Consistent positive delta = course playing harder than you predicted. Adjust coefficients or baseline.

---

## What You Touch Each Round

| Input | Where | When |
|-------|-------|------|
| `round` | Sheet `round_config` | After each round completes |
| `expected_score_1` | Sheet `round_config` | After each round — set to actual scoring avg |
| Wind/dew arrays | Sheet `round_config` | Update for upcoming round's forecast (or re-run `humidity.py`) |
| `realized_wind_r{N}` | Sheet `round_config` | After each round — actual avg wind for actuals writeback |

Everything else is automatic: skill updates, weather spline, actuals comparison, next-round predictions.

---

## Where Things Go Wrong

| Issue | Cause |
|-------|-------|
| No tee time data for spline | DataGolf hasn't published next-round times yet. Re-run later; skill update is already saved. |
| R3/R4 predictions look wild | The base reset needs `base_pred` from the prior round's live model. If it's missing, the engine falls back to reconstructing it from `tot_sg_adj`/`tot_resid_adj`; if those are also missing/NaN, the base degrades to the Pre prediction (fresh adj stacks on top). |
| Actuals row skipped | You didn't enter `realized_wind_r{N}` in the Sheet. |
| Prediction chain breaks | Prior round's `live_stats_engine.py` didn't complete. Re-run it first. |
| Non-ShotLink event | OTT/putt adjustments auto-skipped in R1; R2+ uses averaged SG which may also be empty. Engine handles gracefully. |

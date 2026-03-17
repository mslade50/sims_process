# Category-First Tournament Simulation: Technical Reference

Comprehensive technical documentation of how `new_sim.py` works — from raw SG data through category distribution building, course-adjusted draws, skill updates, scoring, and pricing. Written to serve as a blueprint for refactoring hole-by-hole sims to use the same architecture.

---

## Overview: What "Category-First" Means

The sim draws each SG category (OTT, APP, ARG, PUTT) independently from a course-adjusted distribution, then sums to total SG. This is the opposite of the archived v1 approach, which drew total SG first, then decomposed into categories.

**Why this matters**: Different courses amplify variance in different categories. Bay Hill amplifies OTT variance 1.24x but barely affects PUTT (1.01x). Category-first draws capture this naturally — players with extreme OTT skill show wider outcome distributions at OTT-sensitive courses, producing fatter tails in finish positions and more realistic edge distributions.

**Pipeline**: `cat_dists_player.py` → `new_sim.py` → finish positions + matchup pricing

---

## Stage 1: Building Per-Player Category Distributions

**Script**: `cat_dists_player.py`
**Input**: `field_adjusted_sg.csv` (PGA rounds since 2019)
**Output**: `sg_dist_player.csv` (one row per player × category), `this_week_dists_v2.csv` (filtered to this week's field)

### 1.1 Raw Data

Source is `field_adjusted_sg.csv` with per-round SG values in four categories:
- `sg_ott_adj` — Off the Tee
- `sg_app_adj` — Approach
- `sg_arg_adj` — Around the Green
- `sg_putt_adj` — Putting

Filtered to `tour == "pga"`, `year >= 2019`, non-null `sg_app_adj`.

### 1.2 EMA Weighting

Each player's rounds are sorted chronologically and weighted with exponential moving average:

```
alpha = 2 / (EMA_SPAN + 1)    # EMA_SPAN = 50 → alpha = 0.0392
decay = 1 - alpha              # 0.9608
weight[i] = decay^(n-1-i)      # most recent round has weight 1.0, oldest has decay^(n-1)
weights = weights / sum(weights)  # normalize to sum to 1
```

Effective half-life is ~17 rounds, so recent form dominates but career patterns still contribute.

### 1.3 Per-Category Statistics

For each player × category pair (minimum 20 observations):

| Statistic | Formula |
|-----------|---------|
| **Weighted mean** | `μ = Σ(w × x)` |
| **Weighted std** | `σ = √(Σ(w × (x - μ)²))` |
| **Weighted skew** | `m₃ / σ³` (standardized 3rd central moment) |
| **Weighted excess kurtosis** | `m₄ / σ⁴ - 3` |
| **Effective sample size** | `n_eff = 1 / Σ(w²)` |

Tail clipping applied before computation: values outside [-8.0, 8.0] strokes are clipped.

### 1.4 Output: `sg_dist_player.csv`

One row per player × category. Key columns:
- `player_name`, `category_clean` (e.g., `sg_ott`, `sg_app`)
- `mean`, `std`, `skew`, `excess_kurtosis`
- `n` (raw observation count), `n_eff` (effective sample size)
- `q01` through `q99` (EMA-weighted percentiles)
- `hist_counts_json` (81-bin histogram, 0.25-stroke resolution from -10 to +10)

### 1.5 Output: `this_week_dists_v2.csv`

Same structure as `sg_dist_player.csv` but filtered to players in this week's tournament field (from `field_updates.csv` or DataGolf API). This is the file the sim reads.

Key columns consumed by the sim: `player_name`, `category_clean`, `mean`, `std`, `skew`, `n_eff`.

**Important**: These are RAW (un-scaled) distributions. Course variance scaling is NOT applied here — the sim applies it via `COURSE_CAT_MULTS`. This avoids double-counting since `dists_thiswk.py` produces a separate `this_week_dists_adjusted.csv` with course shape adjustments for v1-style use, but the category-first sim does not use that file.

---

## Stage 2: Sim Configuration & Inputs

**Script**: `new_sim.py`

### 2.1 Weekly Config (from Google Sheet via `sheet_config.py`)

| Parameter | Source | Example |
|-----------|--------|---------|
| `tourney` | Sheet | `"bayhill"` |
| `SIMULATIONS` | Sheet | `10000` |
| `STD_DEV` | Sheet | `1.92` |
| `CUT_LINE` | Sheet | `70` |
| `USE_10_SHOT_RULE` | Sheet | `True` |
| `WIND_FACTOR_SIM` | Sheet | `0.08` |
| `COURSE_CAT_MULTS` | Sheet | `{'sg_ott': 1.24, 'sg_app': 1.09, ...}` |
| `COURSE_CAT_SKEW` | Sheet | `{'sg_ott': 0.0, ...}` |
| `wind_1`, `wind_2` | Sheet | 15-element hourly arrays (6 AM - 8 PM) |
| `dewpoint_1`, `dewpoint_2` | Sheet | 15-element hourly arrays |

### 2.2 Stable Model Params (from `sim_inputs.py`)

Skill update coefficients (per-round, per-bucket) and `name_replacements` dict. These rarely change.

### 2.3 Prediction File

Prefers `final_predictions_{tourney}.csv`, falls back to `pre_course_fit_{tourney}.csv`.

Key columns: `my_pred` (player's expected SG per round), `std_dev`, `sample` (observation count for confidence filtering), `dg_final_pred` (DataGolf decomposition fallback).

**Low-confidence replacement**: If `|my_pred| < 0.5` and DataGolf decomposition is available, the DG value replaces the model prediction.

### 2.4 Correlation Matrix

Loaded from `permanent_data/` in preference order:
1. `sg_cat_corr_tour_within_player_pearson.csv`
2. `sg_cat_corr_tour_spearman.csv`
3. `sg_cat_corr_tour_pearson.csv`
4. Identity matrix (fallback)

Shape: (4, 4) — correlations between OTT, APP, ARG, PUTT.
Decomposed via Cholesky: `L_corr = cholesky(R)`, shape (4, 4).

The within-player category correlations are weak (~0.03), so this matrix is close to identity but still preserves whatever structure exists.

---

## Stage 3: Building Player Parameters

For each player, the sim constructs two vectors:
- `mu`: shape (4,) — re-centered category means
- `std_course`: shape (4,) — course-adjusted category standard deviations

### 3.1 Load Raw Distributions

From `this_week_dists_v2.csv`, pivot into matrices:
- `mu_w`: (n_players, 4) — raw category means
- `std_w`: (n_players, 4) — raw category stds
- `skew_w`: (n_players, 4) — raw category skewness
- `neff_w`: (n_players, 4) — effective sample sizes

Global fallbacks computed for players with missing categories:
- `global_mu`: mean of `mean` per category (field average)
- `global_std`: median of `std` per category

### 3.2 Re-center Category Means to `my_pred`

The raw category means from `sg_dist_player.csv` reflect a player's historical tour-average performance. But the prediction for *this week* (`my_pred`) may differ due to form, course fit, etc. The re-centering ensures the sim's base prediction matches the model prediction.

```python
cat_sum = mu[0] + mu[1] + mu[2] + mu[3]   # sum of raw category means
target = my_pred                            # this week's model prediction
shift = (target - cat_sum) / 4.0            # distribute gap equally
mu = mu + shift                             # now sum(mu) == my_pred
```

**Why equal split**: Distributing the shift equally across all 4 categories preserves the relative category proportions from the player's historical data. Proportional re-centering was considered but not adopted — the equal split is simpler and the category correlation matrix is weak enough that proportional allocation wouldn't meaningfully change outcomes.

**Example**: Player has raw category means [0.25, 0.15, 0.10, -0.05] (sum = 0.45). This week's `my_pred` = 0.50. Shift = (0.50 - 0.45) / 4 = 0.0125. Adjusted means: [0.2625, 0.1625, 0.1125, -0.0375] (sum = 0.50).

### 3.3 Apply Course Variance Multipliers

```python
course_mults = [COURSE_CAT_MULTS['sg_ott'], COURSE_CAT_MULTS['sg_app'],
                COURSE_CAT_MULTS['sg_arg'], COURSE_CAT_MULTS['sg_putt']]
std_course = std * course_mults   # element-wise multiply
```

**What `COURSE_CAT_MULTS` represent**: The ratio of actual category standard deviation at this course to the tour-average expected standard deviation. Computed by `scoring_baseline.py` from historical event data.

- `> 1.0` = course amplifies variance in this category (e.g., OTT at Bay Hill = 1.24)
- `< 1.0` = course compresses variance
- `= 1.0` = no course effect

This means a player with raw OTT std of 0.85 at a course with OTT multiplier 1.24 will have effective OTT std of 1.054 — wider draws, fatter tails, more upside/downside from driving.

---

## Stage 4: The Draw Pipeline (Per Round)

For each of 4 rounds, the sim draws per-player category SG values. The R1 draw is the simplest; R2-R4 add skill update shifts.

### 4.1 Weather Delta (R1 and R2 only)

Weather effects are computed per player based on tee time:

```python
wind_adj = WIND_FACTOR_SIM * avg_wind_during_round(tee_time, wind_array)
dew_adj = dew_calculation * avg_dew_during_round(tee_time, dewpoint_array)
```

Both are mean-centered across the field so the average player has zero weather impact. The per-player delta captures AM/PM wave advantages.

```python
weather_delta = wind_adj + dew_adj  # scalar per player, mean-centered
```

Distributed across categories:
```python
WEATHER_CAT_SPLIT = [0.35, 0.35, 0.15, 0.15]  # OTT, APP, ARG, PUTT
cat_mu = mu + weather_delta * WEATHER_CAT_SPLIT
```

R3 and R4 have no weather delta (tee times are based on leaderboard position, not waves).

#### Bayesian Wind Blending

Wind forecast arrays are blended with a climatological prior before use. The prior is the historical average hourly wind speed (6 AM–8 PM) for the tournament month at the course coordinates, averaged across 2019–2025 via Open-Meteo archive data.

The blend weight is computed dynamically from the actual lead time (hours until the round starts), not hard-coded per round number. This means running the sim Monday vs Wednesday night produces different blends:

```python
climo_weight = clamp(lead_days / 12.0, 0.05, 0.50)
```

| Lead time | Climo weight | Example |
|-----------|-------------|---------|
| 12 hours | 5% | Re-run night before R1 |
| 2 days | 17% | Wednesday run for R1 Thursday |
| 3 days | 25% | Tuesday run for R1 Thursday |
| 5 days | 42% | Monday run for R3 Saturday |
| 6+ days | 50% (cap) | Early-week R4 forecast |

```python
blended_wind[i] = (1 - w_climo) * forecast[i] + w_climo * climo[i]
```

Round dates are computed as Thursday–Sunday of the current week via `get_round_dates()` (PGA events are always Thu–Sun, rounds start ~7 AM).

Applied in both `new_sim.py` (tournament sim) and `live_stats_engine.py` (round-level predictions). Functions live in `api_utils.py`: `fetch_historical_hourly_wind()`, `blend_wind_with_climo()`, `climo_weight_for_lead()`, `get_round_dates()`.

### 4.2 Skew-Normal Parameters

**Tour-wide baseline skewness** (stable across courses):
```python
BASELINE_CAT_SKEW = {'sg_ott': -0.93, 'sg_app': -0.21, 'sg_arg': -0.18, 'sg_putt': -0.05}
```

OTT is strongly left-skewed (bad drives hurt more than good drives help). PUTT is nearly symmetric.

**Course skew adjustments** (`COURSE_CAT_SKEW`): Additive adjustment per course, typically small.

**Per-player effective skew** — blended from course baseline and player-specific skew:
```python
confidence = min(n_eff / 100.0, 1.0)           # ramp 0→1 as n_eff reaches 100
blend_weight = 0.5 * confidence                  # max 50% player-specific
effective_skew = (1 - blend_weight) * course_skew + blend_weight * player_skew
```

Players with large sample sizes (n_eff ≥ 100) get 50% weight on their personal skew. Low-sample players default to the course/tour baseline.

### 4.3 The Draw: Correlated Skew-Normal

For each player `i` in each round:

```python
# Step 1: Draw 4 independent standard normals
Z = RNG.standard_normal(size=(SIMULATIONS, 4))       # shape (SIMS, 4)

# Step 2: Induce correlation via Cholesky
corr_z = Z @ L_corr.T                                 # shape (SIMS, 4)

# Step 3: Apply skewness via Cornish-Fisher expansion
for j in range(4):
    gamma = effective_skew[i, j]                       # e.g., -0.93 for OTT
    if abs(gamma) >= 0.01:
        gamma_adj = gamma * calibration_multiplier(gamma)
        corr_z[:, j] = corr_z[:, j] + (gamma_adj / 6) * (corr_z[:, j]**2 - 1)
        corr_z[:, j] /= sqrt(1 + gamma_adj**2 / 18)   # variance correction

# Step 4: Scale to player's distribution
draws = cat_mu + corr_z * std_course                   # shape (SIMS, 4)

# Step 5: Clip extreme values
draws = clip(draws, -8.0, 8.0)

# Step 6: Sum to total SG
sg_total = draws.sum(axis=1)                           # shape (SIMS,)
```

**Cornish-Fisher calibration**: The first-order CF expansion saturates at high |γ|. A polynomial correction compensates:
```python
def calibration_multiplier(gamma):
    ag = abs(gamma)
    if ag < 0.2: return 1.0
    return 1.0 + 0.0234 * ag**2 + 0.0125 * ag**3
```

### 4.4 Integer Scoring

```python
strokes = round(PAR - sg_total)    # PAR = 72
# A player with sg_total = 0.83 → 72 - 0.83 = 71.17 → rounds to 71
```

---

## Stage 5: Skill Updates Between Rounds

After each round's actual results are "observed" (simulated), the sim adjusts each player's expected skill for the next round. This models how R1 performance updates our R2 prediction, etc.

### 5.1 R1→R2 Update

**Residual**: `resid = sg_r1_actual - my_pred_base` (how much the R1 draw deviated from prediction)

**Bucket selection** based on `my_pred_base`:
| Bucket | Condition |
|--------|-----------|
| High | `my_pred > 1.0` |
| Mid-high | `0.5 < my_pred ≤ 1.0` |
| Mid-low | `-0.5 < my_pred ≤ 0.5` |
| Low | `my_pred ≤ -0.5` |

**Adjustment components**:
```python
ott_adj = ott_r1 * coeff['ott']
putt_adj = putt_r1 * coeff['putt']
resid_adj = resid * coeff['residual'] + resid² * coeff['residual2']
```

**Residual cap**: If residual is negative (underperformed) and adjustment would be > 0.2, cap at 0.2. Hard cap at 0.5 regardless.

**Updated prediction**: `updated_skill_r2 = my_pred_base + ott_adj + putt_adj + resid_adj`

### 5.2 R2→R3 Update

**Position-based buckets** (based on 36-hole rank after cut):
| Bucket | Condition |
|--------|-----------|
| Top | Position < 6 |
| Middle | Position 6-30 |
| Bottom | Position > 30 |

**More adjustment terms than R1**: residual, residual², residual³, plus per-category SG adjustments (OTT, APP, ARG, PUTT, delta_APP).

**Critical**: Undoes the R1→R2 adjustment first, then applies fresh R2→R3 adjustment:
```python
total_adj_r2 = (resid_adj_r2 + sg_adj_r2) - prior_r1_adjustment
updated_skill_r3 = updated_skill_r2 + total_adj_r2
```

### 5.3 R3→R4 Update

**Average-SG coefficients** (no residual terms). Uses weighted average of category SG across first 3 rounds:
```python
avg_cat = 0.66 * (0.5 * (cats_r1 + cats_r2)) + 0.34 * cats_r3
```

Position buckets: < 6, 6-20, > 20.

Also undoes the R2→R3 adjustment before applying R3→R4.

### 5.4 How Skill Shifts Enter Category Draws

For R2-R4, the skill update produces a per-simulation shift (each simulation had different R1 draws, so each has a different R2 prediction). This shift is distributed equally across 4 categories:

```python
base_total_mu = mu.sum() + weather_delta    # what the static mean would be
skill_shift = updated_skill - base_total_mu  # per-sim delta, shape (SIMS,)
cat_mu_shifted = cat_mu + skill_shift[:, None] / 4.0  # shape (SIMS, 4)
```

Equal distribution preserves the course covariance structure encoded in `std_course` and `L_corr`.

---

## Stage 6: Cut, Final Scores, Ranking

### 6.1 Cut Logic

After 36 holes (R1 + R2):
```python
cut_score = sorted_scores[CUT_LINE - 1]   # e.g., 70th-best score
made_cut = (score <= cut_score)
if USE_10_SHOT_RULE:
    made_cut |= (score <= leader_score + 10)
```

Players who miss the cut get a 200-stroke penalty on R3+R4 (effectively infinite — they can't finish in the money).

### 6.2 Final Scores

```python
final_scores = strokes_r1 + strokes_r2 + strokes_r3 + strokes_r4
# shape: (n_players, SIMULATIONS)
# missed-cut players have ~472+ (200 penalty on R3+R4)
```

### 6.3 Win Probabilities

For each simulation, find minimum final score. If tied, randomly select one winner:
```python
min_score = final_scores[:, sim].min()
tied = players where score == min_score
winner = random_choice(tied)
```

Win probability = fraction of simulations won.

### 6.4 Top-N Probabilities (Dead-Heat Adjusted)

For top-5/10/20 finishes, ties are handled via dead-heat factor:
```python
def dead_heat_factor(position, tie_count, threshold):
    overlap = min(position + tie_count - 1, threshold) - max(position, 1) + 1
    return max(0, overlap) / tie_count
```

Example: 3-way tie for 4th (positions 4, 5, 6) in top-5 → each gets 2/3 credit.

### 6.5 Full Rank Probabilities

Ties are expanded: a 3-way tie at position 4 becomes three entries (positions 4, 5, 6) each with weight 1/3. Aggregated across simulations and normalized:

```python
rank_probs[player, rank] = sum(weights) / SIMULATIONS
```

Stored as `rank_probs_updated_{tourney}.parquet`.

---

## Stage 7: Market Pricing

### 7.1 Outright / Top-N Markets

Sim probabilities compared to market odds from DataGolf API:
```python
edge = (sim_prob * (decimal_odds - 1)) - (1 - sim_prob)
```

Markets priced: Win, Top-5, Top-10, Top-20.

### 7.2 Tournament Matchups

For each bookmaker matchup (Player A vs Player B):
- Count simulations where A finishes ahead of B (ties handled per book rules — push or loss)
- Compare to implied probability from book odds
- Filter: edge > 3% for storage, pred > 0.75 + sample ≥ 20 for email

### 7.3 Finish Position Sizing

Uses Kelly criterion for stake sizing:
```python
kelly_stake = (edge * decimal_odds - 1) / (decimal_odds - 1)
```
Units wagered derived from `kelly_stake`, not flat sizing.

---

## Key Design Decisions and Why

### Why category-first instead of total-first?

Total-first draws a single SG number and decomposes it into categories using fixed ratios. This means every course has the same variance profile across categories. Category-first lets Bay Hill amplify OTT variance (1.24x) while barely touching PUTT (1.01x), producing wider tails for players whose skill is concentrated in OTT — a more realistic model.

### Why re-center to `my_pred`?

Without re-centering, the sum of historical category means diverges from the weekly prediction by up to 1.2 strokes for some players. This creates spurious edges — the sim would think a player is better/worse than the model predicts. Re-centering ensures category-first only changes *variance structure*, not *base predictions*.

### Why equal-split re-centering (`shift / 4.0`)?

Proportional re-centering (distributing shift in proportion to category means) was considered. Equal split was chosen because:
1. Simpler and more predictable
2. The within-player category correlation is weak (~0.03), so proportional allocation wouldn't meaningfully change covariance structure
3. The shift is typically small (mean ~0.02 strokes per category)

### Why Cornish-Fisher instead of drawing from a skew-normal distribution directly?

CF is a perturbation of standard normal draws — it preserves the Cholesky correlation structure. Drawing from a proper skew-normal would require a multivariate skew-normal, which is harder to parameterize and slower to sample. CF is fast and accurate for |γ| < 1.5, which covers all golf SG categories.

### Why `WEATHER_CAT_SPLIT = [0.35, 0.35, 0.15, 0.15]`?

Wind and dewpoint primarily affect ball flight (OTT and APP), with smaller effects on short game (ARG) and putting (PUTT). The 35/35/15/15 split is based on domain knowledge of how weather conditions interact with shot types.

### Why different skew confidence thresholds?

OTT skewness is very stable across players (most players are left-skewed in OTT). PUTT skewness is nearly symmetric and has low signal-to-noise. The confidence ramp (0→100 n_eff) prevents noisy small-sample skew estimates from dominating.

---

## File Reference

| File | Produced By | Consumed By | Persistence |
|------|-------------|-------------|-------------|
| `field_adjusted_sg.csv` | External | `cat_dists_player.py` | Permanent (historical data) |
| `sg_dist_player.csv` | `cat_dists_player.py` | `dists_thiswk.py`, dashboard | Regenerated weekly |
| `this_week_dists_v2.csv` | `cat_dists_player.py` | `new_sim.py` | Regenerated weekly |
| `this_week_dists_adjusted.csv` | `dists_thiswk.py` | Not used by category-first sim | Regenerated weekly |
| `final_predictions_{tourney}.csv` | External skill model | `new_sim.py`, `live_stats_engine.py` | Auto-pushed weekly |
| `pre_course_fit_{tourney}.csv` | External skill model | `new_sim.py` (fallback), `scoring_baseline.py` | Auto-pushed weekly |
| `sg_cat_corr_tour_*.csv` | One-time analysis | `new_sim.py` | `permanent_data/` |
| `rank_probs_updated_{tourney}.parquet` | `new_sim.py` | Dashboard | Cleaned weekly |
| `finish_equity_{tourney}.csv` | `new_sim.py` | Email, dashboard | Cleaned weekly |

---

## Array Shape Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `mu` | `(4,)` | Re-centered category means for one player |
| `std_course` | `(4,)` | Course-adjusted category stds for one player |
| `course_mults` | `(4,)` | `[OTT_mult, APP_mult, ARG_mult, PUTT_mult]` |
| `L_corr` | `(4, 4)` | Lower-triangular Cholesky of correlation matrix |
| `effective_skew` | `(n_players, 4)` | Blended skewness per player per category |
| `Z` | `(SIMS, 4)` | Independent standard normal draws |
| `corr_z` | `(SIMS, 4)` | Cholesky-correlated draws |
| `draws` | `(SIMS, 4)` | Scaled category SG draws |
| `cats_rN` | `(n_players, SIMS, 4)` | Per-category draws for round N |
| `sg_rN` | `(n_players, SIMS)` | Total SG for round N |
| `strokes_rN` | `(n_players, SIMS)` | Integer scores for round N |
| `final_scores` | `(n_players, SIMS)` | 72-hole totals |
| `made_cut_mask` | `(n_players, SIMS)` | Boolean cut mask |
| `weather_delta_rN` | `(n_players,)` | Mean-centered weather effect per player |

---

## Constants Reference

| Constant | Value | Purpose |
|----------|-------|---------|
| `PAR` | 72 | Course par (constant — doesn't affect relative rankings) |
| `EMA_SPAN` | 50 | Distribution recency weighting (half-life ~17 rounds) |
| `MIN_OBS` | 20 | Minimum rounds per player-category for inclusion |
| `CLIP_CAT` | (-8.0, 8.0) | Category draw clipping bounds |
| `WEATHER_CAT_SPLIT` | [0.35, 0.35, 0.15, 0.15] | Weather delta distribution [OTT, APP, ARG, PUTT] |
| `BASELINE_CAT_SKEW` | OTT=-0.93, APP=-0.21, ARG=-0.18, PUTT=-0.05 | Tour-wide skewness defaults |
| `SKEW_BLEND_MAX` | 0.5 | Maximum weight on player-specific skew |
| `SKEW_CONFIDENCE_N` | 100 | n_eff at which player reaches full skew confidence |
| `RNG seed` | 456 | Reproducibility seed |
| `CORR_PREFS` | within-player Pearson > Spearman > Pearson | Correlation matrix preference |

# Shot-Level Feature Study — 2026-08-19

## Executive decision

Two feature families merit an R2-native shadow build, but neither should change
production prices yet:

1. **Per-category shot dispersion (50-round span)** as an input to simulated
   variance. It improved out-of-sample Gaussian log score in all four SG
   categories and against the production total-SG standard deviation. The
   pricing-scale gain was small, so it needs a complete correlated-simulation
   test before replacing existing standard deviations.
2. **Approach cross-track directional RMS** as a small approach-skill
   calibration. Its forward MSE improvement repeated in the 2026 confirmation
   sample, although the event-level 2026 evidence is only modest.

Do not build the proposed approach distance-by-lie course-fit feature. The
exact proposal overfit, and strongly shrunk variations did not improve the
production R1 model. Retire the generic putting-speed and great-shot-tail
families based on the tests below.

## Data and leakage controls

The tests used two complementary sources:

- The August 18 R2 production export: 6,485,758 valid, completed-hole PGA
  shot-SG rows from 2020–2026. This was used for the shot-dispersion and
  approach-distance-by-lie studies.
- The existing shot-bridge libraries: 82,210 player-round feature rows through
  July 19, 2026, merged to production predictions/residuals. These supported
  the broader feature screens.

All primary tests used only information from completed prior events. Events
with the same start date were treated as one batch where relevant. Model tests
were expanding-year walk-forwards; the bridge screens used 2022–2025 as
discovery and 2026 as a separate confirmation check. Loss differences were
clustered by event.

The 2026 sample is confirmation-like, not pristine: related work had already
looked at part of 2020–2026. Production promotion should therefore require a
prospective shadow period.

## Results

### 1. Per-category shot dispersion — shadow candidate

The tested feature pooled within-round per-shot SG variance with an EMA and
multiplied it by expected category shot count. The baseline was prior
round-level category variance. A 50-round span was best among 16, 32, 50, and
100.

Out-of-sample category results covered 61,736 player-events and 182,628 rounds.
Negative NLL change is better.

| Category | NLL change vs round variance | Event-cluster 95% CI |
|---|---:|---:|
| OTT | -0.002664 | [-0.003651, -0.001735] |
| APP | -0.000735 | [-0.001256, -0.000211] |
| ARG | -0.005620 | [-0.009005, -0.002632] |
| PUTT | -0.001872 | [-0.002374, -0.001383] |
| **Pooled** | **-0.002723** | **[-0.003614, -0.001877]** |

The sign was favorable in every test year. The 2025 confidence interval barely
crossed zero; the pooled interval did not.

Against the production `std_dev_50` on 16,255 player-events / 47,154 rounds:

- Gaussian NLL improved by 0.000641, with event-cluster 95% CI
  [0.000239, 0.001034].
- Spearman correlation with realized squared error rose from 0.0297 to 0.0413.
- The gain was favorable in 2023, 2024, 2025, and 2026, but only 2023 was
  individually significant.

The shot estimate carried most of the incremental information. Second moments
and positive/negative tail rates added nothing. Contrary to the audit's
convergence claim, 16 rounds was clearly weaker than 32–50, and some category
rankings continued improving toward 100 rounds.

**Decision:** generate `shot_indep_var50` plus effective shot count for each
category, shrink low-history cases, and evaluate it in shadow mode with course
multipliers and category correlations active. Do not directly replace the
production standard deviations from this isolated test.

### 2. Approach directional consistency — shadow candidate

`app_dir_rms_cross_trail16` was the most repeatable mean-skill feature in the
new category-level screen:

| Sample | Approach-SG MSE improvement | Event-level t |
|---|---:|---:|
| 2022–2025 walk-forward | 0.285% | 3.22 |
| 2026 confirmation | 0.328% | 1.61 |

The baseline was an event-safe 50-round category-SG history; the current event
was excluded wholesale. This agrees with the earlier chronological feature
work, which also rated the cross-track feature as a provisional go.

**Decision:** rebuild it statelessly from R2 and shadow it as an approach-mean
calibration. The effect is too small, and the 2026 event evidence too modest,
for an immediate production merge.

### 3. Approach distance × lie × course mix — reject

The exact proposal used eight distance cells, two lie cells, a trailing
40-round player profile, and the course's historical shot mix. It made both
targets worse:

| Target | Baseline RMSE | Feature RMSE |
|---|---:|---:|
| Future event APP SG | 1.109503 | 1.109700 |
| Production R1 residual | 2.703846 | 2.704120 |

A full-history version with very strong hierarchical shrinkage (`k=400`
shots) found tiny shot-level specialization: MSE improved by 0.0000816 per
shot, roughly 0.06% of shot variance. Crossing it with a leakage-safe course
mix improved APP-event RMSE by only 0.000280 overall, with 90.8% bootstrap
probability of improvement, while production R1 RMSE worsened by 0.001118 and
worsened in every test year.

Even an explicitly leaky oracle using the current event's realized mix had
negligible value. This is not a case where a better course-mix forecast would
rescue the feature.

The export's `course_id` must not be used as a physical-course key: it changes
with round tee/pin geometry and commonly produces four IDs in one ordinary
single-course event. Physical courses were reconstructed here from 18-hole tee
coordinate geometry. Prior-course mix then beat tour-average mix in 92.8% of
events, confirming that the course reconstruction itself worked.

**Decision:** archive the feature. If it is ever revisited, first establish a
canonical physical-course identifier and use full-history, heavily shrunk
profiles on a genuinely untouched future sample.

### 4. Putting process features — keep only lag control as low priority

- `lag_putt_leave_control_trail36` had small, consistent gains: approximately
  0.024% discovery / 0.060% confirmation on putting-SG MSE, and 0.034% /
  0.088% on squared production-residual MSE. The confirmation event t-stat was
  2.43 for putting SG and above 3 for the squared-residual screen.
- `speed_ideal_pct_trail24` improved discovery MSE by 0.126% but worsened 2026
  by 0.008%. `speed_short_pct_trail24` also flipped negative in confirmation.
- The existing `putt_make_luck_sum_trail24` retained a small forward signal,
  consistent with the prior chronological study; it is better supported than
  the generic speed family.

**Decision:** lag control can remain in a later shadow queue. Retire the
generic speed-control pair.

### 5. Great-shot capability and blowup proxies — no promotion

Great-shot rates were tested against future within-event top-decile outcomes,
not just the mean:

- The p90-style rate group improved Brier score by only 0.008% in discovery
  and 0.061% in 2026; the confirmation event t-stat was 1.26.
- The p95 group was flat/worse.
- Great-shot rates made the negative-tail forecast worse in both periods.

This reinforces the earlier null next-round mean tests. Retire the great-shot
feature unless a materially different definition becomes available.

Existing penalty/recovery/tail proxies improved the 2026 negative-tail Brier
score by about 0.062%, but with an event t-stat of only 0.73 and inconsistent
discovery evidence. That does not reject the audit's exact double-bogey
residual/skew idea, which was not present in the library, but it gives no reason
to prioritize the required build.

### 6. Other own variations

- A stable driving-distance baseline improved forward OTT MSE by 0.241% in
  discovery and 0.444% in 2026. This is a useful baseline-calibration finding,
  not a new course-fit feature, and much of it may already be represented in
  production inputs.
- Core approach proximity improved APP MSE by 0.076% / 0.164%, but 2026
  event-level evidence was weak (`t=0.91`).
- Short-putt make skill and the aggregate ARG leave/conversion feature had
  favorable point estimates but weak confirmation evidence.

## Ranked next actions

1. Build only the R2-native 50-round category shot-dispersion feature first.
   Keep its output out of pricing while collecting full-sim CRPS, matchup log
   loss, and calibration diagnostics.
2. Rebuild approach cross-track RMS from the same stateless R2 job and shadow
   its mean adjustment separately, so variance and mean effects are not
   confounded.
3. Require a prospective shadow sample before promotion. The latest four weeks
   were outside the exact crosswalk used by these tests.
4. Fix or create a canonical physical-course identifier before attempting any
   other shot-profile × course interaction.
5. Do not spend build time on distance×lie course fit, generic putting speed,
   great-shot tails, shot second moments, or shot tail-rate variance variants.

## Reproducible local studies

- `archive/shot_feature_predictive_study.py` — total production-residual mean,
  absolute-error, and squared-error screens.
- `archive/shot_process_category_study.py` — event-safe category-SG tests.
- `archive/shot_tail_feature_study.py` — top/bottom-decile tail tests.

The dispersion study's raw outputs are in the session artifact directory
`C:\Users\McKinley Slade\AppData\Local\Temp\codex_dispersion_20260819_8f3c2a`.
No production model or data file was changed.

# Extended approach distance x lie course-fit study

## Decision

Do **not** promote the approach distance-by-lie course-fit feature. The
expanded study confirms that player specialization is real at the individual
shot level, but it is too small to survive the course-mix aggregation and it
does not improve the production R1 residual target.

This conclusion is stronger than the first study because the extension tests
coarser cells, much heavier shrinkage, longer histories, smooth and low-rank
profiles, a distinctive-course subgroup, two realized-mix oracle ceilings,
and a leakage-safe personalized mix. None of the usable variants improves R1
reliably.

## Data and leakage controls

- Local R2 export through August 18, 2026.
- The export contains 1,803,826 valid, complete PGA approach shots from
  2020-2026. The distance/lie definition retains 1,788,533 shots after the
  50-yard and supported-lie filters; joining to events with a production date
  and target leaves 1,734,542 shots in the modeling panel. This reconciles the
  earlier 1.80M headline with the smaller extended-study sample.
- 30,348 player-events / 242 events; 27,708 player-events have at least 100
  prior approach shots.
- Eight 25-yard distance cells from 50 yards through 225+ yards, crossed with
  clean (`tee/fairway`) versus bad (`rough/sand/recovery`) lies. The coarser
  specification merges adjacent distance cells into four distance bands.
- Profiles are frozen before each event. Events sharing a start date are
  snapshotted as one batch before any same-date result enters history.
- Model coefficients are fit on years strictly before each 2023-2026 test
  fold. Targets and features are field-demeaned inside each event.
- Physical courses are matched from 18-hole tee-coordinate geometry with a
  strict 25-coordinate-unit RMSE threshold. The current event's realized mix
  is used only in explicitly labeled oracle tests.
- Positive delta RMSE below means improvement.

Reproducibility snapshot: the local R2 export manifest was generated at
`2026-08-18T13:18:16.482726+00:00` and has SHA-256
`d4f6e55526b98d82b3cc326a10012c9d7c60012394f8ed1103408bd1e172f0c4`.
The production target file is
`C:\Users\McKinley Slade\dev\sim_prep\work\sg_preds_resids.csv`, SHA-256
`fe300c1a03015f422d73e2042d39f508a16bf956b69d728a0c46d36f8def6190`.

The physical-course proxy is not the weak link. On 128 testable events from
2022 onward, the prior physical-course mix beat the prior tour mix in 93.75%
of events. Median L1 error was 0.1317 for the physical prior versus 0.2369 for
the tour prior.

## Full-history hierarchical grid

The 8x2 profile retains the tiny APP-event signal seen in the first study, but
the evidence is weak and year-unstable. Every shrinkage setting worsens R1,
and the R1 loss grows slightly as shrinkage increases. The 4x2 profile reduces
the R1 damage but does not reverse it.

| Profile | k | APP delta RMSE | R1 delta RMSE | R1 P(better), event bootstrap |
|---|---:|---:|---:|---:|
| 8x2 | 100 | +0.000043 | -0.001165 | 0.007 |
| 8x2 | 200 | +0.000049 | -0.001198 | 0.006 |
| 8x2 | 400 | +0.000068 | -0.001229 | 0.008 |
| 8x2 | 800 | +0.000092 | -0.001267 | 0.007 |
| 4x2 | 100 | -0.000119 | -0.000413 | 0.098 |
| 4x2 | 200 | -0.000119 | -0.000447 | 0.096 |
| 4x2 | 400 | -0.000115 | -0.000477 | 0.098 |
| 4x2 | 800 | -0.000111 | -0.000505 | 0.087 |

For the pre-specified 8x2/full-history/k=400 profile, APP-event RMSE is
1.105050 to 1.104982, an improvement of 0.000068 (0.0123% MSE). Its
event-cluster 95% interval for delta MSE is [-0.000700, +0.000899], with only
62.6% bootstrap probability of improvement. The annual APP delta RMSE values
are -0.000021, -0.000593, +0.000253, and +0.000828 for 2023-2026.

The same feature changes production R1 RMSE from 2.716256 to 2.717485, a
0.001229 worsening (negative 0.0905% MSE). Its event-cluster delta-MSE interval
is [-0.016615, -0.001662], t=-2.34, and it worsens all four test years:

| Test year | R1 delta RMSE | Fitted feature coefficient |
|---|---:|---:|
| 2023 | -0.002078 | -6.080 |
| 2024 | -0.001251 | -3.504 |
| 2025 | -0.001142 | -2.216 |
| 2026 | -0.000044 | -0.973 |

The negative coefficient and its movement toward zero are inconsistent with a
stable positive approach adjustment.

## Longer histories and parsimonious profiles

- The only realistic R1 variant with a positive pooled delta is 4x2 trailing
  40 rounds: +0.000044 RMSE. Its bootstrap probability of improvement is only
  40.9%, and its annual deltas alternate signs (-0.000214, +0.000378,
  -0.000301, +0.000529). This is noise, not a candidate.
- 4x2 trailing 80 and 160 rounds worsen R1 by 0.000094 and 0.000417.
- The smooth 8x2 ridge worsens APP by 0.000142 and R1 by 0.000589.
- The fixed four-factor low-rank profile is the least harmful full-history R1
  model, but still worsens APP by 0.000097 and R1 by 0.000218. Its R1
  event-cluster probability of improvement is 8.6%.

## Shot-level specialization exists, but is too small

Direct next-event shot prediction confirms a small, persistent player-by-cell
interaction. The best variants improve per-shot MSE in every test year, but
only by roughly five to seven hundredths of one percent.

| Profile | MSE improvement / shot | Relative MSE improvement |
|---|---:|---:|
| 4x2, trailing 160, k=400 | 0.0001059 | 0.0719% |
| 4x2, full history, k=400 | 0.0000922 | 0.0626% |
| Smooth 8x2, full history | 0.0000898 | 0.0610% |
| 8x2, full history, k=400 | 0.0000816 | 0.0554% |
| Low-rank, full history | 0.0000695 | 0.0472% |

For full-history 8x2/k=400, the per-shot gains are 0.0000562, 0.0000959,
0.0000873, and 0.0000851 in 2023-2026. This is genuine profile persistence,
but a course has to alter a player's mix enough for only a fraction of this
tiny effect to reach round-level SG.

## Oracle ceilings and personalization

The oracle results clarify why the path fails.

| Mix input | Best R1 delta RMSE | Relative R1 MSE gain | Interpretation |
|---|---:|---:|---|
| Prior physical-course mix | +0.000044 | negligible | No reliable usable variant |
| Realized event-average mix | +0.000487 | 0.0359% | Tiny leaky ceiling |
| Realized player-specific mix | +0.002615 | 0.1930% | Larger, but outcome-contaminated |
| Prior course + prior player mix | -0.000094 | -0.0069% | Leakage-safe blend fails |

The event-average oracle's best result is the smooth profile. Its annual R1
deltas are +0.000083, +0.000726, +0.000350, and +0.000839, but its
event-cluster delta-MSE interval still includes zero [-0.000189, +0.007579].

The player-specific realized-mix oracle is not a deployable forecast. A
player's realized approach distribution and shot count are partly produced by
current-event drives, misses, recoveries, hole completion, cut survival, and
performance itself. It therefore exposes a leaky composite signal that can
contain OTT, selection, and current-form information; it does **not** prove
that approach-mix personalization is the missing input. Consistent with that
caution, it improves the total-SG R1 residual far more than the APP-event
target. The best APP result from the same player-specific oracle is only
+0.000057 RMSE with 68.8% bootstrap probability.

A leakage-safe personalized forecast combined prior physical-course mix with
the player's prior distance/lie propensity. It did not transfer the oracle
gain: the best R1 variant worsened RMSE by 0.000094, and every full-history
personalized variant worsened R1.

## Distinctive courses do not rescue the feature

The subgroup was defined before looking at outcomes as events whose strictly
prior physical-course mix differs from the prior tour mix by L1 >= 0.15. The
OOS evaluation covers 90 events / 9,862 R1 player-events.

| Profile | APP delta RMSE | R1 delta RMSE | R1 P(better) |
|---|---:|---:|---:|
| 8x2, k=400 | +0.000060 | -0.001251 | 0.008 |
| 4x2, k=400 | -0.000147 | -0.000527 | 0.095 |
| Smooth 8x2 | -0.000177 | -0.000531 | 0.035 |
| Low-rank | -0.000124 | -0.000222 | 0.092 |

The global null is therefore not hiding a useful effect at courses with the
most unusual forecastable distance/lie mix.

## BMW / Bellerive

The 2020-2026 R2 export contains no Bellerive event, so there is no historical
Bellerive approach distance/lie mix from which to build the strict physical-
course feature. Under the leakage-safe production rule, Bellerive falls back
to the tour mix; `course_mix - tour_mix` is zero and every BMW course-fit delta
is therefore zero/unavailable.

Using an event-name proxy or a generic course mix would create player-profile
deltas, but they would not be Bellerive course-fit deltas. This path should not
be used to explain or alter this week's BMW prices. The separate cross-track
approach-mean and shot-dispersion shadow features do not have this coverage
problem and remain the relevant BMW experiment.

## Recommendation

Close the distance-by-lie course-fit path for production. Keep the durable
harness so a truly new season or a future Bellerive-specific data source can be
tested without reusing the discovery sample. If revisited, the minimum bar
should be a pre-registered, frozen 4x2 or smooth profile that improves
production R1 residual in a new season and does so on both ordinary and
distinctive courses. Shot-level MSE alone is not enough.

Artifacts:

- `archive/approach_lie_course_fit_extended.py`
- `permanent_data/approach_lie_course_fit_extended_results.json`

# Conditional variance audit

2026-09-09. Step 1 of the [variance research brief](VARIANCE_RESEARCH_BRIEF_2026.md).

## Decision

The aggregate round spread is close to calibrated. Shot-level dispersion improves **which players receive more or less variance**, with a small improvement in overall round-distribution scoring. This audit does not support a blanket variance increase or a new player/course penalty. Continue to player-specific course exposure, testing whether it adds information beyond the existing shot overlay.

## Scope and method

- 34 PGA events, January 15 through August 13, 2026; 435 players and 38 physical courses represented across the panel.
- Primary: 3,939 observed R1 player-event cases. Secondary: 12,725 observed player-rounds using the same frozen pre-event skill, without live updates.
- Two paired arms: reconstructed category distributions from round histories; the same distributions with the existing shot-dispersion blend. Both receive exactly the same dated skill means, course multipliers, skew inputs and category correlation matrix.
- 20,000 paired simulations per player/course/arm. Histories stop before the event. Multi-course events use the player's actual physical course for outcome alignment, with that course's strictly prior shape.
- Remove the common realized course/round scoring level from outcomes and from simulated errors symmetrically. This is a conditional distribution audit, not a forecast of the absolute scoring level. Individual simulated shapes are centered on the fixed skill target, removing numerical mean drift. Profiling the field mean reduces residual variance, so simulated errors receive the same operation.
- Primary scale diagnostic: sum of squared forecast errors divided by sum of predicted conditional variances. One is the target; above one suggests insufficient spread or mean error, below one suggests excessive spread. Average directional error and a diagnostic ratio after subtracting subgroup mean bias are shown separately.
- Buckets use forecast/history information, never outcomes. Intervals resample whole events. They are exploratory, unadjusted for the many subgroup comparisons and previous research; recurring players can also connect events. No candidate was fit or selected for production.

## Main result: relative player variance improves

| Pre-event shot-overlay change | R1 cases | Round-history ratio | Shot-overlay ratio | Shot-overlay 95% event interval |
|---|---:|---:|---:|---:|
| SD reduced by more than 5% | 668 | 0.772 | 0.910 | 0.822–1.000 |
| SD increased by more than 5% | 886 | 1.213 | 1.030 | 0.918–1.148 |
| SD change within 5% | 2,385 | 1.016 | 1.020 | 0.964–1.077 |
| **All R1 cases** | **3,939** | **1.004** | **1.003** | **0.957–1.051** |

The first group had substantially less realized error than its original variance forecast; the second had more. The shot estimate moves both toward one. Their small mean biases (−0.092 and −0.016 strokes) explain essentially none of this pattern.

The direction repeats in both chronological portions of the panel. For SD reductions, ratios improve 0.763→0.903 in the earlier 16 events and 0.780→0.918 in the later 18. For SD increases, they improve 1.296→1.098 and 1.136→0.966. Removing any single event also preserves the direction. These are stability checks on an already-studied season, not new holdouts.

Overall central-90% interval coverage moves from 89.64% to 89.74%. Round CRPS, a score of the complete predictive distribution where lower is better, improves 1.552372→1.551219: approximately **0.074%**, with paired delta −0.001153 and event interval [−0.002300, −0.000124]. Most of the measured CRPS gain comes from the SD-reduction group. Improving the variance ratio alone does not guarantee a material improvement in betting probabilities.

The secondary frozen all-round panel agrees on aggregate scale: ratio 0.978 with shot dispersion and 90.37% central-90% coverage. Its later-round rows are subject to cut selection and do not test the live model.

## What remains after the overlay

**Forecast volatility.** Using the baseline's fixed SD buckets, shot-overlay ratios are 0.977 below 2.5 strokes, 0.993 at 2.5–2.8, 0.999 at 2.8–3.1, and 1.093 above 3.1. All event intervals include one. The highest bucket's interval is 0.924–1.297 across 326 R1 cases. It merits monitoring, but not a variance adjustment chosen from this season.

**Recent evidence.** Players with fewer than 10 historical rounds in the preceding 180 days have a ratio of 1.107, interval 0.952–1.279, and 87.90% interval coverage across 595 R1 cases. Players with 20+ recent rounds have a ratio of 0.972 and 90.34% coverage. This is a plausible lead for forecast reliability or persistent uncertainty. Total category-history length alone is less distinguishing: ratios are 1.046 for fewer than 50 observations, 0.994 for 50–149, and 0.995 for 150+. Sparse-history players without dated skill estimates are not covered by this reconstruction.

**Course.** Courses with broad category multipliers have ratio 1.041 across eight events, interval 0.954–1.152. The ordinary multiplier group has ratio 1.000. Only one event occupies the low-multiplier group, preventing a useful cross-event inference. One-to-two-year usable course histories produce ratio 0.907 across five events, interval 0.821–1.011; zero usable history means the reconstruction used its fallback, not that the course has never hosted golf.

Each individual physical course contributes only one event in this panel. Its apparent variance miss cannot distinguish a persistent course effect from that week's realization. Player/course listings are therefore descriptive; they do not establish player-by-course interaction parameters.

**Selected sharp-book exposure.** Linking the prior 643-signal evidence identifies 67 unique backed-player R1 cases: 31 only in the >=5% Kelly theoretical-EV group, 34 only below 5%, and two appearing in both groups. Ratios are 0.983 [0.551, 1.458] and 0.607 [0.334, 0.976] for the first two groups. These small samples do not show generally understated variance in the strongest-edge group. They describe one backed player's reconstructed distribution, not the matchup score difference, opponent dependence, or the actual issued probability. Later-round membership is available in the evidence but retains frozen pre-event means.

## Limits that matter

This reruns the existing research machinery against the September 1 database and September 8 shot export. The observations used for features predate the predicted event, but their processing vintage and dated skill table are retrospective. This is not an exact replay of each historical production model or the July repair counterfactual, and it does not include BMW or later events.

The reconstructed kernel comparison omits the final round-score skew top-up and the full tournament/live cascade. The production skew top-up targets a largely unchanged variance, but exact final production tail calibration is outside this audit. The observed good/bad tail imbalance in these raw arms is **not** evidence that production needs an additional skew change. No new matchup or finish probability backtest was run here.

Common course/round conditions are removed; individual conditions, skill errors, incomplete shot histories and observation selection can still contribute to residual spread. A variance ratio is not a causal decomposition. Bootstrap intervals with few events are suppressed, and the many overlapping player observations are not independent bets.

## Next action

Step 2 should ask whether **the difference between a player's expected shot exposure on this course and their usual exposure predicts residual variance after the shot overlay**. Freeze the present baseline and skill means. Test a small, shrunk exposure adjustment using only prior-course shot counts and opportunity mixes, then evaluate complete round distributions and paired matchup probabilities on common cases. Do not estimate separate unpooled variances for every player/course combination. Carry the weak recent-evidence lead into step 3 rather than adding a blanket low-history multiplier now.

## Evidence and verification

Artifacts: `permanent_data/research/conditional_variance_20260909/` contains the executable audit, source manifest, paired case panel, subgroup tables, chronological and leave-one-event-out checks, descriptive player results, and verification output.

Checks passed: unique paired cases; identical means and observed errors across arms; dated skill cutoffs; course/round centering; positive variances; consistent coverage/tail accounting; nonnegative CRPS. Synthetic calibration is approximately one, halving SD correctly quadruples the error/variance ratio, and a deliberately biased group is distinguished from extra variance. Simulated versus analytic centered-variance mean absolute difference is 0.0090 strokes squared. The source audit found no duplicate physical shots or shot keys in any 2020–2026 PGA export. Production code, data, prices, accounts and shared services were not modified.

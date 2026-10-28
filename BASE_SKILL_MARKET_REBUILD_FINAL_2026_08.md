# Base-skill rebuild: market-first final research report

Date: 2026-08-23  
Status: research only; no production model, input, consumer, or weekly-pipeline file changed

## Decision

Do not promote the rebuilt base skill model.

The clean discovery search did not produce a challenger that passed the frozen
proper-score, calibration, money/CLV, concentration, and search-correction gates.
The best diagnostic was a translated cross-tour local-level Kalman model using
adjusted total strokes gained only. It looked modestly better than the strict
point-in-time current-architecture proxy when the two discovery markets were pooled,
but it was worse in R1 and its apparent improvement was not robust to the familywise
correction over the 57 candidates admitted by the no-market numerical gate from the
92-model catalog.

That diagnostic then failed the sealed 2023-2026 test. In the exact last-three-year
slice it was worse than the point-in-time legacy proxy on both 72-hole and R1 matchup
log loss, lost more money in R1, was materially overconfident, and was worse in every
calendar-year proper-score split. The evidence supports retaining the current base
skill architecture while continuing research on calibration and market-aware overlays.

This is a negative result, not a failed analysis. It prevents an attractive discovery
artifact from being promoted after seeing the later market outcomes.

## What was tested

The study rebuilt every player-event skill snapshot strictly from prior rounds and
built a 92-candidate catalog frozen before the complete evaluation. Fifty-seven
candidates passed the no-market numerical gate into market scoring:

- adjusted-total multi-horizon ridge models;
- total plus orthogonalized category histories, with category-specific shrinkage;
- clean IMG stroke-accounting variants and adjusted-total-plus-IMG variants;
- Broadie-motivated long-game and stronger short-game-shrinkage variants;
- W0 uncleaned history, W1 weather-cleaned features, W1 weather-cleaned targets, and
  robust weather variants;
- ridge, elastic-net, robust IRLS, dynamic empirical-Bayes, and local-level Kalman
  estimators;
- annual, quarterly, and strict event-batch refits;
- cross-tour translation, explicit low-history fallback, source coverage, staleness,
  age, and effective-sample-size controls;
- R1-R2 primary targets and a separately labeled post-cut nuisance sensitivity.

Adjusted category SG was not double-counted with total SG: category inputs were
residualized against the category shares of adjusted total. IMG inputs were kept
separate because DataGolf and IMG may share ShotLink observations and therefore are
not guaranteed to be independent measurements.

The original IMG accounting transform had used sealed-era evidence in its lie mapping
and putting endpoint. It was excluded from clean selection. A semantically frozen,
point-in-time accounting rebuild was used for clean arms; the full-archive IMG signal
was retained only as a retrospective ceiling.

## Frozen chronology and benchmark

- Warm-up: rounds before 2021-01-01.
- Discovery and all model/hyperparameter selection: 2021-01-01 through 2022-12-31.
- Sealed evaluation: 2023-01-01 through 2026-08-22.
- Direct three-year comparison: 2023-08-22 through 2026-08-22, inclusive.
- Events sharing a start date were one chronological batch.
- Closing prices were outcomes/evaluation data and never model features.
- Whole events, including zero-bet events, were resampled for inference.

Exact historical production cannot be replayed because contemporaneous weekly
configs, overrides, DataGolf blends, and weather forecasts were not archived. The
primary benchmark is therefore a strict-prior reconstruction of the legacy neutral
core inserted into the same current-code downstream bridge as the challenger. That
isolates the base-skill question, but the shared course/history bridge contains
retrospective current coefficients and is not a completely point-in-time real-money
replay. Current-coefficient legacy and retrospective `pred_step2` controls are
reported separately and are not mislabeled as historical production.

The canonical build contains 55,952 player-event snapshots, 174,785 PGA rounds, and
302 events. Every challenger emitted the identical 38,409 requested prediction keys.
The clean IMG join retained 83,811 one-to-one player-round records.

## Discovery result

No candidate passed all frozen gates. Twenty-six of 57 selectable candidates improved
pooled discovery matchup log loss, but none improved both 72-hole and R1 markets.

The diagnostic leader was
`complete__dynamic_kalman__xtour__w0__event_batch__q0.0005`: adjusted total SG,
strict-prior cross-tour translation, event-batch updating, process variance 0.0005 per
day, and a prior precision equivalent to 20 rounds. The catalog's inherited 80-round
dynamic-horizon field is not consumed by the Kalman implementation.

Its event-equal discovery log-loss deltas versus the strict point-in-time control were:

| Market | Challenger minus control |
|---|---:|
| 72-hole matchup | -0.001862 |
| R1 matchup | +0.000809 |
| Pooled | -0.000526 |

Negative is better. The apparent pooled gain did not survive the familywise search
over the 57 candidates that passed the no-market numerical admission gate into market
scoring from the 92-model catalog:
the null-centered whole-event bootstrap p-value was 0.291, White-style reality-check
p-value was 0.578, and the simultaneous 95% upper bound was +0.001717. Consecutive
2/3/4-event block variants also failed to certify it.

Discovery close best-of-three flat-stake P&L was -190.78 units (-3.46%) for the
diagnostic and -234.39 units (-4.63%) for the control. That 43.61-unit difference was
not enough to override the worse R1 proper score, failed calibration gate, or search
uncertainty. The sequential scorecard stopped before the diagnostic reached its CLV
gate, so this is not evidence that it failed that later gate.

## What the candidate-family search learned

The simplest adjusted-total state was the strongest family. Mean pooled discovery
log-loss deltas versus control and the best member of each family were:

| Source family | Family mean | Best member | Members improving |
|---|---:|---:|---:|
| Adjusted total | -0.000120 | -0.000526 | 14/18 |
| Total + clean IMG | -0.000056 | -0.000509 | 3/6 |
| Post-cut nuisance | -0.000101 | -0.000295 | 2/3 |
| Long-game IMG | +0.000141 | -0.000234 | 3/5 |
| Total + categories + IMG | -0.000047 | -0.000196 | 4/6 |
| Total + orthogonal categories | +0.000411 | +0.000275 | 0/19 |

The category profiles did not add reliable market information once total skill was
already present. The clean IMG accounting stream also did not produce a stable R1
gain. This does not say category skill is unreal; it says these historical summaries
did not improve the quoted two-player probabilities after shrinkage and point-in-time
controls.

Weather-cleaned regression arms did not earn promotion or, as unmatched groups, beat
the control. Average pooled deltas versus control were +0.000016 for W0, +0.000104 for
W1 neutral-target, +0.000140 for W1 feature-only, and roughly +0.00043 to +0.00049 for
robust W1 variants. Those group averages have different model-family composition and
are not a causal W1-versus-W0 comparison. Among exactly matched admitted regression
siblings, W1 modestly improved pooled score for 23 of 24 pairs: mean W1-minus-W0 was
-0.000103 for feature cleaning and -0.000139 for neutral-target cleaning. Robust W1
also modestly improved its exact W0 siblings, but every weather-arm group average
remained worse than the control and no weather arm passed the gates. A
like-for-like dynamic Kalman W1 sibling was not implemented in discovery, so this
result cannot establish that weather cleaning would never help that state-space
family.

## Legitimate discovery extensions

A pre-unseal extension blended the Kalman state with a rich adjusted-total + category
+ IMG + weather-neutral ridge model. It improved each clean parent on pooled proper
score but still failed the control comparison, R1 calibration, CLV, and familywise
search gates. Its close best-of-three P&L was -155.73 units (-2.89%) versus -234.39
units (-4.63%) for the control, but mean CLV was lower by 0.232 percentage points.

A retrospective bridge empirical-Bayes arm looked much stronger in pooled results,
but it embedded full-archive/sealed-period information and worsened R1 versus the
rebuilt bridge. It is deliberately excluded from any promotion claim.

## Sealed pre-tournament matchup result

### Full sealed period: 2023-01-01 through 2026-08-22

Best of BetOnline, Betcris, and Pinnacle close; flat one-unit risk; 3% model-EV gate:

| Market | Model | Bets | P&L | ROI | Max drawdown | Proper-score delta vs control |
|---|---|---:|---:|---:|---:|---:|
| 72-hole | Kalman diagnostic | 4,363 | -263.81 | -6.05% | 273.68 | +0.000775 |
| 72-hole | Point-in-time control | 4,017 | -259.57 | -6.46% | 264.31 | — |
| R1 | Kalman diagnostic | 2,836 | -158.01 | -5.57% | 195.59 | +0.002438 |
| R1 | Point-in-time control | 1,760 | -41.93 | -2.38% | 59.73 | — |

Positive proper-score delta is worse. The paired P&L delta was -4.23 units for 72
holes (95% event-bootstrap interval -110.00 to +104.03) and -116.07 units for R1
(-240.48 to +9.86). The diagnostic did not beat the benchmark.

### Exact last three years: 2023-08-22 through 2026-08-22

| Market | Model | Bets | P&L | ROI | Max drawdown | Log loss | Calibration slope |
|---|---|---:|---:|---:|---:|---:|---:|
| 72-hole | Kalman diagnostic | 3,324 | -240.47 | -7.23% | 249.13 | 0.692902 | 0.727 |
| 72-hole | Point-in-time control | 3,084 | -222.58 | -7.22% | 230.15 | 0.691925 | 1.302 |
| R1 | Kalman diagnostic | 2,189 | -112.52 | -5.14% | 153.14 | 0.694043 | 0.349 |
| R1 | Point-in-time control | 1,366 | -42.69 | -3.12% | 48.34 | 0.691830 | 0.716 |

The three-year proper-score deltas were +0.000976 for 72 holes and +0.002213 for R1.
The R1 interval excluded zero on the harmful side. The Kalman probabilities were
especially overconfident, as shown by calibration slopes far below one.

The diagnostic was worse than the control on proper score in every available calendar
year—2023, 2024, 2025, and partial 2026—in both markets. Moving-block max-stat adjusted
p-values for the primary frozen comparisons were all 1.0.

### Price, edge, and skill diagnostics

The failure was not confined to one book. At best-of-three close, the diagnostic lost
the most selected-book money at Betcris in 72-hole matchups (-108.71 units, -8.31%)
and at Pinnacle in R1 (-60.40 units, -7.27%). Fixed-book results were also negative at
all three books in both markets.

Higher estimated edge did not translate into better returns. At close, the diagnostic
lost -3.85%, -8.54%, and -8.98% in the 3-5%, 5-10%, and over-10% 72-hole edge buckets.
R1 returned -2.48%, -7.81%, and -4.45%. This is the opposite of the monotonicity a
well-calibrated betting model should show.

Absolute projected-skill buckets were also noisy rather than monotonic. There was no
credible evidence that the rebuild worked only for elite players, weak players, or a
particular favorite/underdog structure.

Open-price sensitivity was less bad for 72-hole matchups but worse for R1. Archived
openers are asynchronous and the exact model-at-open timestamp is unavailable, so
these are not guaranteed executable portfolios.

### Market blending

A 75% same-book no-vig opening-price blend was selected only on discovery data. The original sealed
artifact compared challenger-plus-opener with an unblended control, so a post-unseal,
research-only supplement applied that identical frozen opener weight to both bases on
the same 31,221 quote rows.

The fair comparison still favored the control. Exact-three-year proper log loss was
0.689223 for challenger-plus-opener versus 0.689028 for control-plus-opener in 72-hole
matchups (delta +0.000195; whole-event 95% interval -0.000068 to +0.000459), and
0.691146 versus 0.690717 in R1 (delta +0.000430; -0.000106 to +0.000976). At close,
best-of-three, flat 1-unit risk, and the 3% EV gate, the 72-hole arms lost -101.57
units on 1,141 bets (-8.90%) and -78.86 units on 1,123 bets (-7.02%), respectively.
R1 returned +6.92 units on only 109 bets versus +2.31 units on 99 bets; the paired
delta interval (-14.80 to +24.11 units) was uninformative.

The apparent benefit in the asymmetric artifact was therefore largely market-
information attribution, not evidence that the rebuilt base added value. Opening
quotes are also asynchronous and the model-at-open timestamp is unavailable. This is
a separate market-augmentation sensitivity, not a promotable base-skill result.

## R1-R4 cascade matchup result

The round-market extension rebuilt every R1-R4 boundary and tested the frozen base in
four downstream forms: no live update, the current retrospective live update, a
discovery-selected live-update shrinkage, and a strict point-in-time continuous
partial-pooling cascade. A W2 arm allowed only weather from already-completed rounds;
realized weather for the round being priced was excluded. All 3,431,702 offered-edge
rows and 1,822,888 selected rows were settled with zero ungraded bets.

Exact-three-year, close best-of-three, production risk-to-win sizing, and a 3% EV
gate produced:

| Cascade arm | Bets | Stake | P&L | ROI | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Current-code retrospective cascade | 5,099 | 5,600.66 | +52.46 | +0.94% | 64.66 |
| Kalman base + current live update | 9,216 | 10,009.47 | -161.89 | -1.62% | 165.74 |
| Kalman base, no live update | 9,162 | 9,819.09 | -181.01 | -1.84% | 197.72 |
| Kalman + strict-PT flexible cascade, W0 | 9,830 | 10,654.19 | -251.45 | -2.36% | 260.26 |
| Kalman + strict-PT flexible cascade, W2 | 9,871 | 10,712.89 | -223.67 | -2.09% | 238.42 |

The paired Kalman-plus-current-update P&L delta versus the current retrospective
cascade was -214.35 units, with a 10,000-draw whole-event 95% interval of -395.91 to
-28.40 units. The W0 and W2 flexible-cascade deltas versus that benchmark were
-303.91 (-494.59 to -112.99) and -276.13 (-464.09 to -85.11), respectively.

The current retrospective cascade lost R1 (-36.07 units) but won R2 (+21.77), R3
(+18.10), and R4 (+48.66). The Kalman-plus-current-update arm lost R1 (-84.19), R2
(-62.51), and R4 (-28.38), with only R3 positive (+13.19). Its nominal edges were
also inverted: the exact-three-year 5-10% and over-10% buckets lost -101.50 and
-60.99 units, while the current cascade's over-10% bucket gained +38.48 units.

Fixed-book pooled close log loss was worse for Kalman at Betcris (0.694155 versus
0.691324), BetOnline (0.694841 versus 0.691764), and Pinnacle (0.691487 versus
0.688614). The W2 flexible arm remained worse at all three. Thus neither removing the
live update, refitting a more flexible partial-pooling cascade, nor adding completed-
round weather rescued the rebuilt base.

An independent audit found that the original best-of-three proper-score postprocessor
deduplicated quotes before grouping by model. That defect omitted 336 of 390 required
best-of-three summary rows and invalidated the surviving non-C0 best-of-three proper
scores. It did **not** affect fixed-book proper scores, raw probabilities, best-price
bet selection, settlement, P&L, performance tables, or the event bootstrap. The money
results above were independently reconstructed. A separate corrected proper-score
supplement preserves the original artifacts rather than relabeling or rewriting them.
The corrected exact-three-year pooled best-of-three close log loss is 0.693390 for C1
versus 0.690394 for C0.

The positive benchmark is a retrospective current-code cascade, not a bit-for-bit
archive of historical production. It is therefore evidence about relative model
architecture under a common bridge, not a claim that these were executable historical
production tickets.

## Finish-market sensitivity

The first 5,000-draw screen exposed a book-rule inconsistency: it priced every
top-5/10/20 ticket with expected dead-heat shares even though Pinnacle pays a
qualifying tie at full odds. That artifact remains immutable and is not treated as
final. A fresh corrected screen used the native no-dead-heat qualification
probability for Pinnacle and retained expected dead-heat shares for BetOnline and
Betcris. Win and matchup probabilities were unchanged.

The corrected 128-event screen used paired event seeds and common menus. At close,
best of the three books, production-style sizing, and the reconciled primary book
rules, it produced:

| Market group | Model | Bets | P&L | ROI |
|---|---|---:|---:|---:|
| All | Retrospective baseline | 9,844 | -165.91 | -3.43% |
| All | Kalman challenger | 13,810 | -749.33 | -8.76% |
| Matchups | Retrospective baseline | 3,617 | -143.67 | -3.65% |
| Matchups | Kalman challenger | 5,465 | -245.03 | -4.06% |
| Outrights | Retrospective baseline | 6,227 | -22.24 | -2.47% |
| Outrights | Kalman challenger | 8,345 | -504.30 | -20.06% |

The all-market challenger-minus-baseline delta was -583.42 units, with a 10,000-draw
whole-event 95% interval of -1,027.65 to -113.95 and only 0.79% of draws above zero.
The outright-only delta was -482.06 (-908.30 to -23.07). The matchup-only delta was
-101.36, but its interval (-271.35 to +72.38) crossed zero. Flat one-unit sizing was
even more adverse: -2,541.50 units (-4,467.01 to -730.77).

All 48 price-point, line-policy, sizing, and settlement-scenario combinations had a
negative aggregate challenger delta. Requiring selected bets to clear their gate by
at least five estimated 5,000-draw Monte Carlo standard errors still left the
production-style delta at -472.83 units. The corrected artifact passed 300/300 output
hashes, all 326 upstream hashes, 128/128 event checkpoints, exact book-rule selection,
zero ungraded bets, and independent reproduction of all 741,762 selected-row
calculations and 54 headline bootstrap cells.

A 50,000-draw rerun would make a decisively rejected model more precise; it has no
reasonable decision value and was not run. This is a staged-compute futility decision,
not a claim that the contract contained a formal numeric early-stop boundary.

During the corrected finish run, live process monitoring observed roughly 14 GB
working set and 18.4 GB private memory while assembling the full panel; that telemetry
was not persisted in the artifact manifest. Before making this a routine backtest,
stream or aggregate the event checkpoints instead of holding the full
ticket/probability panel in memory. The staged 5,000-draw screen is computationally
feasible, but the current full-panel implementation is unnecessarily memory-heavy.

Finish results are a retrospective current-coefficient sensitivity. The simulation
coefficients and correlation inputs are not reconstructed point-in-time, so finish
markets cannot select or promote the finalist.

## Weather normalization and bad-week attribution

The terminal weather audit covered 172 events, 687 event-rounds, and 66,083
player-rounds. Wind-plus-dew and tee-time coverage was 98.22%. Physical-course identity
was preserved through 35 multi-course event-rounds before event aggregation. Historical
realized weather was permitted only for training history and already-completed rounds;
realized weather for the target/upcoming round was post-hoc explanatory data and was
never credited as a tradable feature. A historical forecast archive does not exist.

The frozen sealed base-market menu contains no scored W1 sibling, so that specific
sealed W1-versus-W0 comparison is correctly reported unavailable rather than rebuilt
after seeing outcomes. The exact same-family post-unseal Kalman W1 diagnostic is
reported separately below. In the cascade, two W2/W0 pairs had exact 42,177-key proper-
score menus. For best-of-three, comparisons with C0 and money attribution used the
37,419-key three-arm common menu; 4,758 W2/W0 quote keys absent from C0 were excluded.
Fixed-book money attribution likewise used its own three-arm common menu: 14,452 keys
at BetOnline, 16,679 at Betcris, and 13,464 at Pinnacle.
Common quote menus do not imply identical tickets: each arm applies its own 3% EV
selection, so W2-minus-W0 and W2-minus-C0 P&L are portfolio deltas rather than
same-ticket treatment effects.

Full-sealed close, best-of-three, production risk-to-win P&L was:

| Base inside strict-PT partial-pooling cascade | W0 | W2 | W2 minus W0 | Common C0 |
|---|---:|---:|---:|---:|
| Kalman diagnostic | -201.24 | -186.64 | +14.60 | +135.48 |
| Point-in-time legacy proxy | -108.62 | -103.30 | +5.32 | +135.48 |

The Kalman W2 improvement was nominal and unstable by round. Whole-event 10,000-draw
intervals were:

| Round | W2 minus W0 P&L | 95% interval | W2 minus W0 log loss | 95% interval |
|---|---:|---:|---:|---:|
| R1 | -4.12 | -13.34 to +4.77 | +0.0000214 | +0.0000136 to +0.0000293 |
| R2 | +6.08 | -10.92 to +23.45 | -0.0000424 | -0.0000977 to +0.0000113 |
| R3 | +13.91 | -6.87 to +34.74 | -0.00000984 | -0.0000672 to +0.0000482 |
| R4 | -1.27 | -18.98 to +16.26 | -0.000000413 | -0.0000729 to +0.0000667 |

The P&L columns use R1-R4 event counts of 172/141/141/138; bookless proper-score
columns use 172/169/169/165, respectively.

Negative proper-score deltas are better. R1 has no completed current-event round to
clean; its tiny but statistically detectable worsening comes from the historical
training-weather path. R2-R4 point estimates were slightly better in proper score,
but every interval crossed zero. In the exact-three-year cascade slice reported above,
W2 improved nominal P&L by 27.78 units versus W0 but still lost 223.67 units while C0
gained 52.46. Weather normalization did not rescue either base architecture.

Realized target-round weather was then used only to ask whether it described the bad
weeks. The label `extreme` is broad: it is the maximum of six weather-component
percentiles, so it covers 42.2% of primary rows rather than a top-decile set. It
appeared in 30 of 62 round-level C0 bad-week rows (48.4%), only modest enrichment.
Spearman correlations between C0 P&L and severity were +0.053, +0.043, +0.017, and
-0.106 for R1-R4. After withdrawal, newcomer, and coverage proxies, weather added only
1.31%, 0.25%, 1.12%, and 3.14% of in-sample descriptive R-squared by round; across all
slices the median was 1.29 percentage points and the maximum 5.08 percentage points. These event-equal
models were not cross-validated and have no causal or standard-error interpretation.

The three largest C0 round losses illustrate the mixed causes: the 2024 Open R2 lost
13.79 units in `extreme` weather with no flagged withdrawal/newcomer/coverage
Kalman-W2 challenger ticket;
the 2025 John Deere R1 lost 12.78 in `high` weather and included seven
withdrawal-affected and three low-coverage challenger tickets; the 2026 Memorial R4
lost 10.74 in `elevated` weather with none of those challenger flags. Across all 62
bad-week rows, withdrawal-affected, newcomer, and low-coverage Kalman-W2 challenger
tickets appeared in 9, 11, and 21 rows, respectively, with overlap. W2 recovered only
+2.48 units versus W0 inside those C0 bad weeks. These flags do not establish an injury
reason, and neither weather nor the available data-hole proxies explain most of the
bad-week dispersion.

One coverage caveat is explicit: 717 physical-course field-mean context rows without a
matched player observation are concentrated in the 2023 Fortinet R1-R2 and 2024
Procore R1-R4, with only one or two actual observations per affected round. They were
not treated as player observations. The audit otherwise reproduced all settlement,
weather-centering algebra, pair menus, association aggregates, and 10,000-draw whole-
event plus 2/3/4-event moving-block intervals.

Operationally, the final weather audit took about 19 minutes. Live monitoring observed
roughly 9.0 GB working set and 9.9 GB private memory at peak; that telemetry was not
persisted in the manifest. This is feasible for research, but its full-ledger joins and
Python event-attribution loops should be streamed or vectorized before routine use.

## Post-unseal uncertainty and exact dynamic-weather diagnostic

A post-unseal implementation review identified a real omission in the selected Kalman
implementation: it retained posterior means but discarded player-specific posterior
variance and did not age uncertainty from the last observation to the target date. A
post-unseal diagnostic corrected both issues and built the exact same-family W1
weather-normalized Kalman sibling that the original catalog lacked. Its mapper
parameters were fitted only on 2020-2022 and then frozen. Because this work followed
the sealed result, it can diagnose the failure but cannot reselect or promote a model.

The correction did not rescue the challenger. Under the like-for-like globally refit
mapper, in the exact three-year slice W1 versus W0 changed event-equal log loss by
-0.000006 in 72-hole matchups (95% whole-
event interval -0.000053 to +0.000041) and -0.000041 in R1 (-0.000141 to +0.000057).
The pair-specific uncertainty coefficient selected exactly zero for 72-hole markets;
in R1 it produced a microscopic discovery fit gain and then worsened sealed log loss
by +0.000011. The combined W1-plus-uncertainty model still trailed the strict point-in-
time control by +0.000938 in 72-hole and +0.001978 in R1 proper score.

Close best-of-three production-style P&L was -237.34 units in 72-hole and -142.88 in
R1 for W1 plus uncertainty, versus -223.28 and -50.47 for the strict control. W1 moved
estimated player skill by only 0.0109 SG/round on average. It used exact historical
PGA round weather for 166,112 prior observations, with finite wind and dewpoint for
98.7%; unmatched observations remained W0. The implementation loaded the complete
weather panel before applying its strict prior-date filter, but an independent replay
verified that zero target/upcoming rows entered any state or mapper. A non-mutating
erratum corrects the original overstrong `weather_read=false` metadata and a separate
tie-refund count label; probabilities, settlement, P&L, and bootstrap are unchanged.
The conclusion is narrow: this simple independence-approximation implementation of
posterior variance, plus normalization of prior-round SG for observed weather, did not
rescue the model. Pair variance used `v1 + v2` and omitted covariance induced by shared
field adjustment and tour translation, so this does not exhaust a fully hierarchical,
covariance-aware state-space model.

## Broadie and Peabody published-principles review

This review applies published work and public interviews. It is not a claim that Mark
Broadie or Rufus Peabody personally reviewed, endorsed, or communicated about the
model.

The Broadie lens was implemented by preserving SG additivity, separating total skill
from orthogonal category shape, testing long-game-specific persistence, shrinking
short-game/putting more strongly, and auditing the stroke-level accounting chain.
Those are sound modeling choices, but the category and IMG signals did not earn
incremental market value in this sample.

Broadie sources: [Assessing Golfer Performance on the PGA TOUR](https://business.columbia.edu/sites/default/files-efs/pubfiles/4996/assessing_golfer_performance.full.pdf)
and [A simulation model to analyze the impact of distance and direction on golf
scores](https://business.columbia.edu/faculty/research/simulation-model-analyze-impact-distance-and-direction-golf-scores).

The Peabody lens was implemented by treating field strength, weather, recency, limited
prior-withdrawal/staleness status proxies, sharp de-vigged prices, simulation, line
shopping, calibration, CLV, uncertainty, and concentration as first-class parts of
the test; timestamped injury/news status was unavailable. The biggest lesson is
that a model's nominal 10% edge is not evidence of edge when calibration is weak and
the market may contain player news the statistical history does not.

Peabody sources: [RotoWire interview on modeling, context, weather, and market
information](https://www.rotowire.com/football/article/beating-the-book-qa-with-rufus-peabody-40172)
and [Golf Digest profile of his golf betting and simulation
process](https://www.golfdigest.com/story/golf-betting-high-stakes).

## Limitations and pitfalls

- This is not a bit-for-bit historical production replay.
- The early incomplete DG-only plumbing screen exposed sealed-period RMSE before the
  complete 92-model build and 57-candidate market scoring. Candidate definitions and
  gates were frozen and
  mechanically enforced afterward, but the study is not perfectly analyst-blind.
- Bookmaker matchup menus are selected samples of players, not random pairs.
- Closing best-of-three prices may not have been simultaneously available at useful
  limits.
- IMG and DataGolf can inherit correlated errors from the same ShotLink observations.
- The IMG column labeled `field_adjusted` is centered within the same hole rather than
  adjusted for a common field-strength scale; the label overstates its comparability
  with DataGolf adjusted SG.
- The selected Kalman implementation carries the posterior mean forward but does not
  propagate player-specific posterior uncertainty or age that uncertainty from the
  last observation to the target date. A single market sigma then converts all skill
  differences to probabilities. The post-unseal diagnostic tested this plausible
  source of overconfidence and did not improve the result.
- Historical injury/news state is incomplete. Closing prices may correctly react to
  information absent from both models.
- Realized future-round weather is explanatory only; no three-year forecast archive
  exists.
- Finish simulations use current coefficients/correlations and are retrospective
  sensitivities.
- Ninety-two candidates are broad coverage of the plausible transparent model class,
  not literally every imaginable model. The familywise correction is therefore
  essential.

## Recommendation

1. Keep production unchanged.
2. Do not promote the Kalman diagnostic, category-rich ridge, clean IMG arms, weather
   cleaning, or opener blend as a new base skill.
3. Focus the next research cycle on probability calibration and decision abstention,
   especially R1 and nominal edges above 5%, rather than adding more raw features.
   If the state-space family is revisited, prospectively lock and validate the existing
   post-unseal W1/uncertainty sibling; do not tune it again on 2023-2026.
4. Archive contemporaneous weekly production inputs, open timestamps, forecasts,
   overrides, and status/news flags. That is the prerequisite for a true historical
   production replay and a tradable weather test.
5. Keep the staged-compute funnel used here: cheap point-in-time proper-score and
   market gates first, cascade only for surviving finalists, paired 5,000-draw finish
   screens next, and 50,000 draws only when Monte Carlo uncertainty could change the
   decision. Stream event checkpoints instead of assembling full ticket panels.
6. If research continues, run a fresh future shadow sample with one or two locked
   models. Do not reuse the 2023-2026 sealed window to choose another winner.

## Principal reproducible artifacts

- Frozen contract: `BASE_SKILL_MARKET_REBUILD_CONTRACT_2026_08.md`
- Published-principles review: `BASE_SKILL_PUBLISHED_PRINCIPLES_REVIEW_2026_08.md`
- Complete model build: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_completion_v3_provenance`
- Discovery selection: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_complete_discovery_v5_provenance`
- Search correction: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_complete_discovery_v5_search_correction_v2`
- Discovery extension: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_complete_discovery_extension_v1_exploratory`
- Frozen diagnostic handoff: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_frozen_finalist_v5_diagnostic_provenance`
- Full sealed matchups: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_complete_sealed_v5_diagnostic_full_20230101_20260822`
- Exact three-year slice: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_complete_sealed_exact_3y_v1_provenance`
- Terminalized R1-R4 cascade: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_round_cascade_v2_terminalized_20260823`
- Corrected cascade best-three scores: `permanent_data/research/base_skill_rebuild_20200101_20260822/model_eval_round_cascade_v2_best3_proper_correction_20260823`
- Dynamic uncertainty/W1 diagnostic: `permanent_data/research/base_skill_rebuild_20200101_20260822/dynamic_kalman_uncertainty_weather_postunseal_full_v1_20260823`
- Dynamic diagnostic erratum: `permanent_data/research/base_skill_rebuild_20200101_20260822/dynamic_kalman_uncertainty_weather_postunseal_full_v1_20260823_erratum_v1`
- Corrected finish-market screen: `permanent_data/research/base_skill_rebuild_20200101_20260822/finish_market_exact3y_book_rule_corrected_screen_5k_v1_20260823` (manifest SHA-256 `4a359b1bfcfbfaafd7f46ce5bd4671c793377fbb6bf05bc9b9a556d246fb8232`)
- Weather and bad-week audit: `permanent_data/research/base_skill_rebuild_20200101_20260822/weather_bad_week_audit_terminalized_10k_v9_final_20260823` (manifest SHA-256 `bd6c380c724aff927f8c90635983bc16d8672d05026a7414af6dea64ee7f4cf6`)
- Symmetric opener-control audit: `permanent_data/research/base_skill_rebuild_20200101_20260822/POST_UNSEAL_SYMMETRIC_OPENER_CONTROL_AUDIT_20260823.md` (SHA-256 `1143edd5f40161266059b57e27a44e5affc45ce1e4ea0e3a92f741c3ec3c31d0`)

Each terminal artifact directory contains its own hash-bound manifest; the standalone
opener-control audit records bound source checksums and QA in the report.

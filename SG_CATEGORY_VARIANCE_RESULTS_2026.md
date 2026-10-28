# SG-category variance forecasts: scale, players and courses

Completed 2026-09-10, in the requested order. **The clearest improvement to test is stronger shrinkage of player-specific APP and ARG variance differences.** Putting personalization also deserves a conservative benchmark. Preserve OTT personalization and the shot-level overlay while testing this. The evidence does not support a blanket category-variance increase or removing course adjustments. No production setting changed.

## 1. Category scale

The common sample is **152 editions, 36 physical venues, 750 players and 18,201 R1 player-editions**, spanning 2020–2026. Every admitted player has all four observed SG categories, yielding 72,804 category observations. This uses the same eligible course editions and frozen pre-event fields as the broad exposure study; it is not every tournament in those years.

The ratio below is observed mean squared forecast error divided by predicted conditional variance. One is the target. Values below one lean toward excessive predicted spread; values above one lean toward insufficient spread or unresolved mean error. Ratios refer to **variance, not standard deviation**.

| Category | All-years ratio | Simultaneous 95% venue interval | 2026-only ratio | Interpretation |
|---|---:|---:|---:|---|
| OTT | 0.937 | 0.859–1.014 | 0.921 | Leans wide, particularly in recent years |
| APP | 1.000 | 0.964–1.038 | 1.003 | Aggregate scale is close to target |
| ARG | 0.989 | 0.929–1.048 | 0.941 | Aggregate scale is close; recent years lean wide |
| PUTT | 1.045 | 0.995–1.102 | 1.004 | Slightly narrow across the full sample; near target in 2026 |

The 2026 slice has 23 editions / 2,505 player cases. The all-years intervals account for testing four categories together; all include one. This is not sufficient evidence for an across-the-board scale adjustment. The narrow putting tendency is not a finding that 2026 putting variance was generally understated.

We remove the common realized event/category scoring level symmetrically from outcomes and forecasts. That makes overall signed error zero by construction, so meaningful mean-bias checks use groups defined by **pre-event predicted category skill**. Across those quintiles, low-rated players tend to outperform their category means and high-rated players tend to underperform. For example, APP errors range from +0.154 SG in the lowest mean quintile to −0.099 in the highest; PUTT ranges from +0.198 to −0.211. These are conditional group averages, not new mean adjustments.

Removing those retrospective group biases changes the all-years variance ratios only modestly: OTT 0.935, APP 0.998, ARG 0.979, PUTT 1.039. That calculation is descriptive and does not remove all possible individual mean errors. It does show that the displayed group biases do not explain most of the dispersion pattern.

Normal-reference central-90% coverage is 92.12% OTT, 89.94% APP, 89.63% ARG and 89.39% PUTT. These standardized-error intervals are diagnostic; they are not the actual skewed category-distribution quantiles.

## 2. Differences between players

The forecasts separate players more strongly than their observed errors do in APP, ARG and PUTT. The table compares the highest and lowest **pre-event variance quintiles within each field**, not player skill quintiles.

| Category | Forecast high/low variance ratio | Observed high/low squared-error ratio |
|---|---:|---:|
| OTT | 1.87× | 1.60× |
| APP | 1.31× | 1.06× |
| ARG | 1.55× | 1.12× |
| PUTT | 1.23× | 1.02× |

OTT retains a clear ordering: forecast low-volatility players actually have substantially less error dispersion. APP's observed ordering is weak and not monotonic; ARG separates less than predicted; the all-years putting groups have nearly the same observed dispersion. These ratios are descriptive, with uncertainty assessed through the paired score comparisons below.

![Predicted and observed category dispersion across player variance quintiles](permanent_data/research/category_variance_20260910/player_category_variance.png)

Two specific misses are actionable research leads:

- **Low-variance ARG:** error variance is **23.8% above prediction**, with simultaneous interval +10.9% to +38.3% across the eight extreme-quintile checks. Mean error is only +0.057 SG; removing that group bias leaves essentially the same ratio, 1.234. The same directional problem appears in 2026: +21.4%.
- **High-variance APP:** error variance is **10.2% below prediction**, with simultaneous interval approximately −19.5% to −1.4%. It is also about 10.6% below prediction in 2026. Low-variance APP goes the other way: +11.6% overall and +9.3% in 2026, although its adjusted interval narrowly includes calibration.

High-variance OTT also has excess forecast spread: ratio 0.887, interval 0.780–0.986. That suggests possible moderation at the extreme, while retaining the useful overall player ordering. Low-variance PUTT is narrow in the full sample, ratio 1.157, but its adjusted interval includes one and the corresponding 2026 ratio is 0.995. Putting should therefore receive a conservative comparison, not an asserted universal correction.

As a direct forecast test, compare the personalized baseline with one common category variance for all players in each field. That common variance is the average of the full pre-event modeled field; means and course-level context are unchanged. Gaussian quasi log loss evaluates the mean/variance forecasts on subsequent outcomes. Lower is better.

| Personalized versus common field variance | Log-loss change | Simultaneous 95% venue interval |
|---|---:|---:|
| OTT | **−0.00818** | **−0.01416 to −0.00308** |
| APP | +0.00117 | −0.00035 to +0.00268 |
| ARG | +0.00315 | −0.00069 to +0.00738 |
| PUTT | +0.00127 | −0.00034 to +0.00331 |

Only OTT demonstrates a reliable benefit from the current player differentiation under this primary venue-based test. The other three point toward stronger pooling, but their aggregate score intervals include zero. Removing any one venue preserves the favorable OTT sign and the adverse APP/ARG/PUTT signs. In the 2024–2026 half, OTT remains favorable and all three other categories remain adverse; individual years vary, including slightly favorable PUTT personalization in 2026.

This supports a **chronological shrinkage experiment**, not immediately making APP, ARG and PUTT identical for every player. Some real player differences may exist even when the current estimates are too noisy or too extreme.

## 3. Course adjustments and shot-level inputs

The shot overlay and course multiplier are separate layers. We rebuilt a paired four-arm comparison: round-history or shot-overlay player variances, each with or without the existing course SD multiplier. Category means and category skew remain fixed throughout. Removing the multiplier does not remove all historical course influence from a player's underlying round history.

| Category | Shot overlay versus round-history variance | Course multiplier versus no multiplier, with overlay |
|---|---:|---:|
| OTT | **−0.01145** [−0.01925, −0.00420] | −0.00643 [−0.01712, +0.00356] |
| APP | **−0.01156** [−0.01688, −0.00487] | −0.00358 [−0.00893, +0.00008] |
| ARG | **−0.01229** [−0.01750, −0.00611] | −0.00520 [−0.01914, +0.00372] |
| PUTT | **−0.01424** [−0.02084, −0.00922] | +0.00123 [−0.00141, +0.00546] |

Values are paired changes in Gaussian quasi log loss, with simultaneous venue intervals across the eight primary component comparisons. Negative is favorable.

**Keep the shot overlay.** It improves all four categories in every annual panel and both chronological halves. It improves at 27/36 venues for OTT, 32/36 for APP, 32/36 for ARG and 35/36 for PUTT; every leave-one-venue-out aggregate remains favorable. This is compatible with the player-differentiation finding: the overlay improves on round-history estimates without proving that all remaining player differences are useful.

**Retain course adjustments while testing the player layer.** Their point estimates help OTT, APP and ARG, and slightly hurt PUTT, but all primary adjusted venue intervals include zero. The APP interval only narrowly crosses zero. Course effects remain uncertain at individual venues, and the putting direction is less stable across chronological halves. There is no basis here for a blanket course-multiplier removal or a single-course retune. Supplemental factorial comparisons and repeated-venue sensitivity are archived.

## Recommended next experiment

Test a small, prespecified grid of **category-specific player-variance shrinkage toward each event's category average**, fitted only on completed earlier events. Keep the category means, course layer and field-average category variance fixed. Preserve the shot overlay as the starting point.

Prioritize **ARG**, then **APP**: their low/high variance assignment misses repeat in 2026, and the ARG low-variance issue has the clearest adjusted subgroup evidence. Include PUTT with stronger pooling as a candidate while respecting its year-to-year instability. Keep OTT personalization as the baseline; allow a modest shrinkage comparison without assuming it should disappear.

The first success criterion is improved chronological category-variance scoring and calibration at the low/high ends. Any eventual promotion also needs a check that the category changes improve or preserve round-matchup probability scores. Do not use this diagnostic to claim a cause for the season drawdown or automatically alter the ≥5% Kelly group.

## Fidelity, limits and evidence

- All **152** rebuilt baseline mean, skew and shot-plus-course SD arrays reproduce the sealed prior inputs within 1e−12. Prior events must have completed before each target starts; **37** poisoned target/future-history checks cover every tested layout.
- Category variances are integrated from the production Cornish–Fisher transformation with clipping at ±8 SG. Thirty independent integration/direct-draw checks pass. Maximum baseline clipping-induced mean shift is 0.0046 SG; the maximum individual marginal-variance change from clipping is 2.49%. Comparisons hold the category forecast means fixed despite these small emitted-mean shifts. Raw observed outcomes remain untrimmed.
- Forecast variances receive the same field-centering operation as errors, including the finite-field correction. **144** dense-matrix/SQL checks, independent aggregation and Gaussian score checks, synthetic scale/bias tests, and **470** source-hash checks pass. No source data, production code, configuration, Sheet or betting rule was changed.
- **Source accounting exception:** a final strict check of raw category sums against total SG failed. There are 156 discrepancies larger than 0.01 SG, mostly common event offsets; twelve player-specific discrepancies remain after subtracting event median offsets, across nine editions. The largest raw gap is 0.726 SG. Their cause is unresolved, and no gap was redistributed to a category. A post-primary sensitivity excludes all nine target editions, retaining 143 editions / 17,044 cases: the main conclusions hold, including low-variance ARG ratio 1.252 [1.123–1.394], high-variance APP ratio 0.898 [0.803–0.988], useful OTT personalization, and favorable shot-overlay scores in all categories. This sensitivity removes affected evaluation editions; it does not rebuild all prior histories without them. [Discrepancies and accounting receipt](permanent_data/research/category_variance_20260910/source_accounting_receipt.json) · [Exclusion results](permanent_data/research/category_variance_20260910/source_exclusion_sensitivity.csv).
- Main results weight editions equally within equally weighted physical venues; PGA National's two layout IDs remain separate for input construction and are pooled only for inference. Ten thousand venue bootstrap draws supply the reported intervals. Category scale, player comparison and primary component comparison have separate multiplicity families; subgroup results are exploratory. Event and recurring-player cluster sensitivities are retained.
- This is a fixed-method, prior-observation reconstruction using the retained data vintages. The shot overlay weights are OTT 0.90, APP 0.90, ARG 0.80 and PUTT 0.95. It does not assert that every historical issued model used the overlay. R1 avoids cut selection; later live rounds and historical issued betting prices are outside this study. Common event/category conditions are removed, but individual conditions and unresolved player mean errors can still contribute to the measured error dispersion.
- Gaussian quasi log loss is used to judge mean/variance forecasts. It is not a claim that the actual category distribution is Gaussian, nor a full tail, covariance or finish-probability validation. Previously studied years are not new holdouts. No corrective parameter was fit or selected in this audit.

Research archive: [contract and ordered tests](permanent_data/research/category_variance_20260910/contract.json), [category scale](permanent_data/research/category_variance_20260910/scale_summary.csv), [player and component comparisons](permanent_data/research/category_variance_20260910/component_contrasts.csv), [quintile diagnostics](permanent_data/research/category_variance_20260910/subgroups.csv), [2026 quintiles](permanent_data/research/category_variance_20260910/variance_quintiles_2026.csv), [verification receipt](permanent_data/research/category_variance_20260910/verification.json). Reproduce with `prepare.py`, `evaluate.py`, `summarize.py`, `verify_results.py`, `detail_tables.py` and `chart.py` in that folder using the existing Python environment.

[Living variance brief](VARIANCE_RESEARCH_BRIEF_2026.md) · [Research worklist](BETTING_RESEARCH_WORKLIST.md)

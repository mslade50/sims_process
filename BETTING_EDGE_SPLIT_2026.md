# 2026 round-matchup results by Kelly theoretical edge

Updated 2026-09-09. Supporting analysis for [the living research worklist](BETTING_RESEARCH_WORKLIST.md).

## Main finding

The ≥5% group generated the season's profit and most of its early favorable run. The <5% group was a persistent drag overall, particularly in R3. However, the seven events after Scotland reversed that relationship: the stronger-edge group produced the late drawdown and the lower-edge group offset part of it. This supports testing edge thresholds and calibration by round and player profile; it does not establish that a 5% cutoff prevents drawdowns.

## Scope and edge definition

This partitions the same verified snapshot as the first review: 29 logged events from WM Phoenix through the TOUR Championship, 643 best-first-run sharp-book records. It retains the original choices before splitting, so the groups are mutually exclusive and reconcile to +24.07u. There is no new selection of a different bookmaker after applying the cutoff. A second view includes all 731 eligible sharp-book records and reconciles to +19.50u.

Sharp books are Pinnacle, BetOnline, and Betcris. First record per event/round/side/opponent/book wins; among the earliest run per matchup side, the highest recorded edge is retained. All selected bets void on ties. Risk is 1u on plus prices and absolute American odds/100 on favorites, matching existing grading. ROI includes pushed stakes in its denominator. Results are logged signals, not independently reconciled account fills. Known invalid-model rows remain excluded.

Kelly theoretical edge follows the system convention: `100 × (model probability × decimal odds − 1)`. For these void-on-tie matchups, probability is conditional on a resolved matchup. This is neither the Kelly stake fraction nor the simple difference between model and book probabilities. A true ex-ante EV including refunds also depends on push probability, which is not recovered for this report.

The primary split uses the recorded one-decimal edge, including exactly 5.0% in ≥5%. Five Masters R2 records from April 9 match a legacy probability-edge formula, confirmed by the April 10 `ece1caa` code correction. Only these records are converted with `stored probability edge × decimal odds`; independently reconstructed fair-price EV verifies the diagnosis. Ben Griffin's +1u record moves from <5% to ≥5%. This changes subgroup attribution only; no source data or total P&L changes.

The <5% group is the existing logged selection, approximately 3.1–4.9% after normalization, not the complete board of all sub-5% opportunities. Existing filters favor `pred_on > 1` for smaller edges, so the groups have different player mixes.

## Results

| Edge / period | Records | Wins–losses–pushes | Risk | Profit | ROI | CLV, pp |
|---|---:|---|---:|---:|---:|---:|
| ≥5%, through Memorial | 264 | 137–94–33 | 295.42u | +43.51u | +14.73% | +0.82 |
| ≥5%, afterward | 163 | 71–75–17 | 190.00u | −9.58u | −5.04% | +0.91 |
| ≥5%, season | 427 | 208–169–50 | 485.42u | +33.93u | +6.99% | +0.85 |
| <5%, through Memorial | 114 | 46–50–18 | 131.86u | −7.76u | −5.88% | +0.23 |
| <5%, afterward | 102 | 44–39–19 | 124.77u | −2.10u | −1.68% | +0.45 |
| <5%, season | 216 | 90–89–37 | 256.63u | −9.86u | −3.84% | +0.34 |

Across all sharp records, ≥5% returned +38.75u on 481 records (+7.09% ROI); <5% returned −19.25u on 250 (−6.45%). The distinction is not created by best-price deduplication.

The ≥5% group's early resolved win rate was 59.3%, versus 55.4% implied by its recorded fair prices. Later it was 48.6%, versus 55.7% implied. The stronger-edge early results exceeded the model's own expectations and then underperformed them. In contrast, <5% win rate improved from 47.9% to 53.0%, although neither period was profitable.

After the Scottish Open specifically, ≥5% lost **20.69u on 85 records (−20.10% ROI)**, while <5% gained **7.34u on 62 (+9.32%)**. A 5% gate would have improved the full-season result descriptively, but worsened the post-Scottish result. These are two different questions.

CLV did not deteriorate within ≥5%. Its same-book-only CLV went +0.76→+0.92 pp. Recorded CLV is raw implied-probability movement, not a vig-free expected return. Positive CLV and losing realized outcomes can coexist.

## Putting, approach, and the early winning run

Residuals use each player's adjusted SG less a pre-event rolling category baseline, then take backed player minus opponent. They describe favorable/unfavorable outcomes relative to prior skill, not pure luck. All 643 matchup pairs have actual category coverage; both scores were independently checked against the historical DB. Baselines use only pre-event rounds and the existing 50/100-round convention, not a frozen live bet-time category forecast.

| Edge | Category | Early surprise | Later surprise | Early/later residual SD |
|---|---|---:|---:|---:|
| ≥5% | OTT | +0.09 | −0.02 | 1.34 / 1.46 |
| ≥5% | APP | +0.19 | −0.15 | 2.33 / 2.30 |
| ≥5% | ARG | +0.09 | −0.03 | 1.50 / 1.32 |
| ≥5% | PUTT | +0.16 | −0.08 | 2.30 / 2.24 |
| <5% | OTT | −0.08 | +0.30 | 1.37 / 1.41 |
| <5% | APP | +0.12 | +0.20 | 2.21 / 2.23 |
| <5% | ARG | +0.09 | +0.14 | 1.55 / 1.41 |
| <5% | PUTT | +0.02 | −0.10 | 2.34 / 2.58 |

The APP deterioration belongs to ≥5%; it is absent in the smaller-edge group. Putting deteriorated in both, more in mean terms for ≥5%. The stronger-edge group's APP and PUTT residual spreads actually narrowed slightly. The low-edge PUTT spread increased descriptively, but the smaller slice does not establish a change in the underlying variance process.

For ≥5%, the 95% whole-event bootstrap interval for the APP mean change is approximately −0.86 to +0.13 SG; for PUTT, −0.73 to +0.18 SG. The direction is informative, but neither establishes a persistent change. After Scotland, ≥5% APP surprise was −0.61 SG and PUTT −0.08; <5% APP was +0.48 and PUTT +0.11. Approach is particularly relevant to the final seven-event drawdown.

## R3: the previous aggregate conclusion needs refinement

| Round | ≥5% early | ≥5% later | <5% early | <5% later |
|---|---:|---:|---:|---:|
| R1 | +4.56u | −3.99u | +2.74u | −1.21u |
| R2 | +8.15u | −0.67u | −2.32u | +2.58u |
| R3 | +29.00u | −0.48u | −4.29u | −8.24u |
| R4 | +1.80u | −4.44u | −3.89u | +4.77u |

Strong-edge R3 supplied two-thirds of that group's early +43.51u and then went approximately flat. Its resolved win rate fell from 69.3% to 53.5%. APP surprise fell +0.43→−0.35 SG; PUTT +0.34→+0.06. This is mainly disappearance of exceptional early performance, not the entirety of the later R3 loss.

Sub-5% R3 supplied **−12.53u across 61 records**, worse than the entire sub-5% portfolio, because other rounds offset it. It generated **−8.24u of the aggregate later R3 loss of −8.72u**. Its later 27 records contain only 21 resolved outcomes, so the apparent 38.1% win rate is imprecise despite being economically concerning.

Within ≥5%, later actual losses instead spread across R1 (−3.99u), R2 (−0.67u), and R4 (−4.44u), with R3 nearly flat. A blanket R3 shutdown would discard the season's largest profitable round/edge slice and miss other late losses.

## Archetypes

Labels are rebuilt consistently from pre-event histories using the current classifier, as in the first review. Counts are small in several cells and these are exploratory comparisons, not independently validated rules.

| Backed archetype | ≥5% records / profit | <5% records / profit |
|---|---:|---:|
| Balanced APP | 45 / −13.27u | 37 / −7.53u |
| OTT_APP | 20 / −7.16u | 11 / +0.93u |
| Balanced Putter | 42 / +2.14u | 18 / −5.58u |
| Elite Putter | 19 / +3.91u | 14 / −6.24u |

Balanced APP remains a concern at either edge level. Its ≥5% ROI was −27.1%; APP surprise was −0.73 SG and PUTT −1.15. Raising the threshold does not rescue that profile. The ≥5% group lost 9.04u early and 4.23u later, reinforcing that it is a persistent issue rather than a newly emerging one.

Conversely, the putting-oriented groups' seasonal losses were concentrated in their smaller-edge bets. Their ≥5% records were profitable. Do not generalize the aggregate results into a blanket penalty on strong putters. The Elite Putter <5% group's CLV was also negative (−0.30 pp), but only 14 records support that observation.

## Selection mix and uncertainty

The ≥5% group averaged `pred_on = 1.21`, while <5% averaged 1.52. Favorite shares were 54.6% and 69.0%. This is partly structural: current entry rules admit less highly rated players only at larger edges. Almost every <5% record has `pred_on > 1`.

Within ≥5%, players with `pred_on <= 1` made **+35.24u on 92 early records**, then lost **3.29u on 49 later records**. Their APP surprise moved +0.43→−0.60 SG and PUTT +0.69→+0.17. This slice explains much of the early favorable run. It does not justify favoring lower-rated players mechanically.

Restricting both edge groups to `pred_on > 1` narrows the contrast: ≥5% makes only +1.98u for the season (+8.27 early, −6.29 later); <5% loses 10.91u (−8.81 early, −2.10 later). This is a coarse sensitivity, not a fully matched causal comparison. A threshold study must account for round, skill, price, and repeated player exposures.

Twenty thousand paired whole-event bootstrap resamples, including events with zero bets in a group, give season ROI intervals of approximately **−2.2% to +15.5% for ≥5%**, **−15.4% to +7.6% for <5%**, and **−5.1 to +25.9 percentage points for their difference**. The observed gap is material, but does not establish an enduring advantage at the chosen cutoff. Resampling events does not remove all cross-event player dependence or the effects of inspecting many slices.

## Boundary and metric sensitivities

- Directly using the Sheet's recorded edge without the legacy conversion gives ≥5% **426 records, +32.93u**; <5% **217 records, −8.86u**.
- Reconstructing all EVs from integer-rounded fair odds and recorded book prices gives ≥5% **423 records, +32.49u**; <5% **220 records, −8.42u**. Ten records change bands versus the primary normalized split because reconstruction adds fair-price rounding error near 5%.
- Exactly 5.0% is included in the upper group (six primary records). The source does not preserve enough precision to recover every unrounded threshold decision. These sensitivities preserve the substantive conclusions.
- No fresh optimal cutoff was searched. Five percent is the user's requested split. Further round/archetype/profile findings remain exploratory and should generate tests on other or future data.

## Implications for the worklist

Add a chronological threshold study, with marginal-edge R3 as a named hypothesis. Refine the R3 implementation audit to distinguish a lost early advantage from persistent marginal-edge losses. Keep Balanced APP calibration high priority even above 5%. Test continuous player profiles and model reliability alongside edge size; do not attribute the whole gap to the threshold. Preserve unrounded, versioned edge inputs going forward.

No production threshold, coefficients, stakes, or Sheet records were changed.

## Reproduction and evidence

The analysis runs from `C:/Users/McKinley Slade/.codex/visualizations/2026/09/09/01a083fd-5e9a-7061-b2be-a4885dcc8097/`:

- `analyze_edge_split.py` reads the already-verified `best_evidence.csv` and `sharp_evidence.csv` without fetching or modifying live records.
- `best_edge_evidence.csv` and `sharp_edge_evidence.csv` preserve the original edge, normalized edge, legacy flag, reconstructed edge, and all band assignments.
- `best_edge_*.csv` / `sharp_edge_*.csv` contain season, phase, round, archetype, book, player, event, category, and sensitivity results; `edge_split_audit.json` contains the bootstrap intervals and boundary audit.
- Group counts, stakes, P&L, and constituent round/phase totals reconcile to the first analysis. No unexplained edge discrepancies greater than 0.5 pp remain after identifying the legacy records. The original 643/731 score-match and join-cardinality validations remain applicable.

Source code checked: `round_sim.py` edge construction and entry filters, and Git commit `ece1caa` for the legacy formula correction. Supporting historical actuals use the September 1 read-only database snapshot from the original review.

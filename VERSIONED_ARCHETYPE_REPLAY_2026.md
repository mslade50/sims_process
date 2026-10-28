# Dated archetype replay and finish-tail check

2026-09-09. Follow-up to [the full archetype screen](ARCHETYPE_PROBABILITY_FRAGILITY_2026.md).

**The replay strengthens the case for investigating selection conditional on model-market disagreement. It does not support a blanket APP/putting penalty or a rollback of the midseason repairs. Newer live-update accounting reduces some old forecast advantages, but most of the performance shortfall remains. Finish probabilities are sensitive to distribution assumptions even with fixed input means; that sensitivity alone does not identify which forecast is correct.**

## What is now tied to dated evidence

| Evidence | Coverage | What it establishes |
|---|---:|---|
| Balanced APP / Balanced Putter signals | 142 | Same sharp-book, first-run sample as prior review |
| Live signals within that sample | 125 | Excludes 17 R1 signals, which have no completed-round live update |
| Historical live totals reproduced | **114/125** | Both logged mean predictions match an archive; historical totals reproduce saved predictions to <1e-8 SG |
| Archived pre-event finish menus matched to recorded probabilities | **23 events** | All prices in a matched run agree within 0.000051; at least three distinct backed players; event/roster/time checks |
| Full archived player-market forecasts | 8,358 | 8,238 have an outcome match; 433 were selected at the matched run |
| Dated pre-course / pre-sim component bundles | 10 events | Fifteen selected Balanced APP/Putter player-events have additive pre-course calculations that reconcile to floating-point precision |
| Frozen distribution experiments | Memorial and BMW | Six scenarios, two fixed seeds, 100,000 simulations per seed/scenario; input skill/coefficients/weather held fixed |

Ten live signals have no matching archive, and one Bay Hill R2 signal involving Straka fails the historical totals check by 0.07 SG; it is excluded rather than forced to match. Source manifests retain commit IDs, times, file hashes, source-column choices and coverage failures. A normal archive-write lag of up to 15 minutes is allowed, while underlying live data must precede the bet. This is strong evidence for matching outputs and final arithmetic, not a complete proof of the original upstream data/flags/kernel on every run.

## 1. Replaying newer accounting on the old live components

The historical totals function is isolated from its archived code and run on archived numeric components. The counterfactual runs the current totals rules on the same components: archived features and coefficient contributions stay fixed, weather stays fixed, and permanent base adjustments are preserved. This tests caps/reset/undo accounting. **It does not refit July's repaired training data or compute counterfactual betting P&L.**

Positive prediction gap means our backed player is expected to outperform the opponent. Error is realized score advantage minus the expected advantage, in strokes per round; negative means underperformance.

| Replayed group | Signals | Average change in expected advantage | Original mean error | Error with current totals |
|---|---:|---:|---:|---:|
| Balanced APP ≥5% | 37 | **−0.089 SG** | −1.503 | −1.414 |
| Balanced APP <5% | 29 | −0.057 SG | −0.195 | −0.138 |
| Balanced Putter ≥5% | 32 | **−0.133 SG** | −0.521 | −0.388 |
| Balanced Putter <5% | 16 | −0.072 SG | −1.361 | −1.289 |

The changes are directionally helpful on average but insufficient to explain most of the miss. For ≥5% APP the correction removes about 6% of the average shortfall; for ≥5% Balanced Putter about 25%. Squared-error changes are favorable point estimates in all four groups, but every event-bootstrap interval includes zero. This is not evidence that the resulting prices would produce a profitable strategy.

Some individual corrections matter. Cantlay versus Young at Rocket R3 loses 0.403 SG of its projected advantage under the current accounting. Hovland versus Harman at RBC R4 loses 0.566 SG. Fowler versus Kitayama at RBC R3 loses 0.404 SG. Other corrections worsen the realized error: changing the forecast does not reliably identify the winning side in hindsight. Named-player evidence includes seven Cantlay, sixteen Fowler, one Hojgaard, one Knapp and eight Rose replays; sparse player counts limit individual conclusions.

Most change is in older forecasts. In the matched post-July20 sample, Balanced APP's average adjustment is −0.040 SG across ten signals; Balanced Putter has three with zero change. The later fixes therefore already address much of this accounting mechanism in the late sample. Neither this result nor the earlier code audit supports undoing the repairs.

## 2. Where the category shortfall enters

The existing archive-first category reconstruction now decomposes APP and putting across both archetypes. It separates the raw category gap, alignment with pre-event total skill, and subsequent live skill changes. Older pre-March17 round allocations remain diagnostic, since that engine drew total SG directly. Reconstructed history is clearly flagged; it is not represented as an issued category forecast.

For all 45 ≥5% Balanced APP signals, the APP pair error was **−0.699 SG against raw category profiles**, **−0.714 against the pre-event target**, and **−0.767 against the live target**. Putting pair error was −1.122, −1.137 and −1.191 respectively. Thus most of this descriptive shortfall exists before the small recentering and average live additions. Both sides of the matchup are included; prior work identified opponent outperformance as a material part of the miss.

Balanced Putter's <5% putting deficit similarly exists against the raw profile: −1.228 SG, versus −1.235 after live allocation. A live-update rollback alone would not remove that pattern. At ≥5%, its paired putting deficit is much smaller, around −0.31 SG on the 41 signals with both anchors.

The ten recovered pre-event component bundles do not reveal a broad excessive course boost in the fifteen selected APP/Putter player-events with exact additive accounting. Examples:

- Hojgaard at Memorial: course-fit contribution −0.049 SG and course-history −0.023, reducing his base projection.
- Rose at RBC Canada: +0.006 course fit and −0.009 history, effectively neutral together.
- Fowler at Memorial: +0.033 course fit and negligible history.
- Knapp at the Open: the pre-course prediction was 0.721 SG and the subsequent rounded pre-sim prediction 0.60, a reduction.

These are recovered component checks, not a complete causal test of every upstream skill term. Some non-target cold-start/override rows do not reconcile to the ordinary additive formula; they remain flagged. The stored `comp_c_adj` already includes the 0.5 course multiplier, verified against historical source; it must not be halved again during analysis.

## 3. Full finish menus sharpen the selection question

Select the earliest eligible probability-matched menu per event using inputs only. Keep picks from that exact run; do not label a player selected using later revised forecasts. This produces 433 contemporaneous selected player-market forecasts, rather than the broader 783-signal season sample. All selected outcomes match; 120 non-selected menu rows lack an outcome and are excluded from calibration, not counted as losses.

| Forecast payout-share range | Selected forecasts: expected → actual | Remaining menu: expected → actual |
|---|---:|---:|
| <5% | 3.31% → 3.69% (140) | 1.97% → 1.67% (3,483) |
| 5–10% | **7.22% → 5.17% (137)** | 7.22% → 7.90% (1,642) |
| 10–20% | **13.67% → 8.15% (101)** | 14.36% → 14.65% (1,618) |
| 20–40% | 27.43% → 33.82% (49) | 27.13% → 28.29% (901) |

These are dead-heat-adjusted payout shares, not ordinary binary hit rates. They suggest investigating the selected 5–20% probability range, rather than applying a general longshot or high-probability haircut. The under-5% and 20–40% selected slices were favorable in this matched-menu sample.

Matching comparisons within the same event, finish market and probability bin retains the concern, but uncertainty remains: selected-minus-unselected error is −2.62pp in the 5–10% bin (95% event-bootstrap range −7.55 to +2.98pp), and −6.32pp in the 10–20% bin (−12.08 to +0.57pp). Both include zero; repeated players and prior exploratory searches further limit inference.

The full top-5 menu was close to expectation in its two middle bins: 6.99% expected versus 6.94% actual at 5–10%, and 13.48% versus 13.51% at 10–20%. **An aggregate full-menu average near expectation is not evidence of calibration by itself:** probabilities sum to five/ten/twenty positions by construction. The probability-bin and within-event/market comparisons are the informative checks. The archived simulated fields also occasionally differ from the final field; unmatched outcomes/absent entrants are retained as limitations.

The exact-run subset changes archetype results too: selected Balanced Putter forecasts are approximately on expectation here (28 forecasts), despite losses in the broader season sample. Balanced APP remains below expectation (51 forecasts), with substantial uncertainty. This is another reason to avoid a static archetype penalty and instead examine time, price and selection.

## 4. Fixed-input distribution experiment

The six scenarios are: no week dependence/no total-skew correction; dependence alone; skew correction alone; both; and both with category standard deviations multiplied by 0.9 or 1.1. Every player and top-5/10/20 probability is retained. No parameter is selected using results. Cut handling is unchanged within each fixture, and skew correction applies only to made-cut simulations at Memorial. Changes in conditional mean among cut-makers can occur because selection changes; fixed input means do not imply identical conditional realized moments.

The combined dependence/skew change moves most probabilities modestly, but some materially:

| Frozen scenario example | Baseline | Combined dependence/skew |
|---|---:|---:|
| Scheffler Memorial top-20 | 84.63% | **80.90%** |
| Scheffler Memorial top-10 | 66.86% | **63.94%** |
| Scheffler BMW top-20 | 85.00% | **82.14%** |
| Scheffler BMW top-10 | 65.35% | **62.74%** |

Across all players and three markets, average absolute moves are 0.24pp at Memorial and 0.29pp at BMW; maxima are 3.73pp and 2.86pp. Relative to the combined scenario, changing category SDs by 10% moves some probabilities by up to about **4–5pp**. This demonstrates how uncertain distribution shape can matter economically even if the scalar skill forecast is credible. It does not establish that every high-probability forecast is wrong or that the new shape is better calibrated.

BMW's average simulated 72-hole SD moves from 5.73 to 6.18 strokes with the combined change; total-score skew moves from approximately +0.07 to +0.23. The no-cut overall mean remains effectively unchanged. The median absolute difference between the two seeds is 0.092pp, with a maximum of 0.785pp across all reported cells. Tiny scenario differences should not be overinterpreted.

The BMW Python reference reproduced the committed 100,000-draw production checksum exactly. Rust versus that reference passed the existing five-standard-error gates: top-5 worst z=2.88, top-10=2.03, top-20=2.69 and rank-1–20=2.65. The experiment uses the currently installed binary, whose hash is recorded; package metadata says 0.2.0 despite the supported week-latent argument, so metadata alone is not used as proof of source lineage. Memorial uses its frozen input fixture on the same current kernel. These are controlled mechanism comparisons, not complete original-versus-new production price replays or a two-event calibration validation.

## Decision and remaining boundary

Keep the existing corrections. Prioritize **conditional selection calibration within the latest effective model version**, with market disagreement, probability cushion and player exposure recorded together. Evaluate APP and putting on both matchup sides; separately test finish distribution shape while holding means fixed. Do not choose a new probability penalty, archetype ban or stake rule using these results.

The next causal boundary is the July training-data repair and coefficient refit. This work recovered outputs and ten input-decomposition bundles, but did not recreate the full pre-repair database plus all original training/runtime inputs. The inspected local cache contains August12 and September1 databases, both after the repair; the documented July backup was not found in the inspected local backup locations. This is not an exhaustive remote R2/D1 inventory. Reconstructing from corrected history cannot by itself establish what the old training run saw.

Updated [research brief](BETTING_RESEARCH_WORKLIST.md). Evidence, scripts, exact coverage, named-player cases, source hashes, probability menus, all six scenarios and verification results: [versioned replay folder](permanent_data/research/versioned_replay_20260909/). No production files, inputs, grades, betting rules, accounts or external services were changed.

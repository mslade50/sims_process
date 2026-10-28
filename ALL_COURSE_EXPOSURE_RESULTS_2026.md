# Course exposure variance: expanded validation and chronological blending

Research date: 2026-09-10. Completed and independently verified: **152 editions, 36 physical venues, 18,201 R1 player appearances, 750 players**. The 1,058,563 non-tied matchup pairs overlap heavily; they are not independent trials.

## Decision

**Keep the current variance baseline.** The small favorable findings from the original six venues do not generalize to the additional 30 venues under this implementation. Chronological blending limits the effect of full replacement, but does not establish an accuracy gain. This is evidence about the tested nine-cell APP exposure model and its variance conversion, not proof that distance, lie, hazards or player/course interactions cannot help.

The most useful next step is the brief's persistent-uncertainty test: determine how much forecast error carries from one round into the next, separating mean bias, player uncertainty and shared event effects. Keep a bounded shot-chain/hazard-tail test queued. Further tuning of this same broad exposure replacement has a lower priority after this result.

## Accuracy results

Equal weight per physical venue, then equal weight per edition within venue. **Negative relative score changes are better.**

| Comparison | All venues: CRPS | All venues: matchup Brier | Added venues: CRPS | Added venues: matchup Brier |
|---|---:|---:|---:|---:|
| Distance/lie beyond course counts | +0.00031% | +0.00162% | +0.00097% | +0.00196% |
| Less-shrunk player-relative exposure vs baseline | +0.00105% | −0.00060% | +0.00133% | −0.00063% |
| Full combined replacement vs baseline | +0.00561% | +0.01198% | +0.00749% | +0.02028% |
| Chronologically learned blend vs baseline | +0.00046% | +0.00058% | +0.00123% | +0.00189% |

The original six-venue combined Brier improvement of **0.03023%** reproduces exactly; the additional 118 editions instead worsen Brier **0.02028%** on average. This is why the broader test was necessary. These are all very small effects, including the adverse effects.

All four primary simultaneous venue-cluster intervals include zero. For combined replacement, the relative 95% intervals are **−0.01679% to +0.02801% CRPS** and **−0.01810% to +0.04206% Brier**. For the learned blend, they are **−0.00384% to +0.00477% CRPS** and **−0.00515% to +0.00631% Brier**. The evidence does not establish a deployable improvement.

![Primary effects and simultaneous intervals](C:/Users/McKinley%20Slade/dev/sims_process/permanent_data/research/all_course_exposure_20260910/primary_effects.png)

## What the blend and robustness checks say

The learned exposure weight averages **23.65%** across trained editions; five initial editions use the declared 50/50 fallback. During 2026 the weight ranges from **20.17% to 36.35%**, ending at **34.07%**. Thus the past-only fitting retains most of the existing course variance estimate. That fitting preference is not itself evidence of better subsequent predictions: the evaluated blend is effectively unchanged overall.

Results depend on which venues receive weight. Full replacement improves Brier at 22 of 36 venues and the blend at 24, but larger adverse effects elsewhere offset those gains. Full replacement remains slightly adverse on both headline metrics after removing any single venue. Weighting by matchup count instead of equally by venue makes its average Brier change favorable, demonstrating why the weighting rule matters.

The predeclared sensitivity restricted to **28 venues with at least three test editions** improves combined Brier by **0.00946%**, while worsening CRPS **0.00209%**. The blend gives −0.00340% Brier and +0.00048% CRPS there. On the 27 venues represented in both time periods, combined Brier is favorable in both 2020–2023 and 2024–2026, but CRPS changes sign; incremental distance/lie changes from slightly favorable to slightly adverse. These are useful stability diagnostics, not a rule to trade venues selected using their eventual number of test editions. Event-resampling intervals conditional on this fixed venue panel are slightly adverse for combined Brier; venue and repeated-player uncertainty remain wider.

## Exposure prediction and stronger probabilities

Among **17,318** cases with observed complete APP exposure, course-specific mix prediction reduces distance/lie mix error **10.01%** versus tour averages, helping 32 of 36 venues. Less-shrunk personalization adds **1.14%** improvement over the common course mix, helping 35 venues. Personalized shot-count MSE improves **4.07%**, helping 31 venues. Neither course-factor clipping limit is hit.

We are therefore predicting the type and number of approach shots better. The missing step is a reliable incremental conversion of that information into score-distribution accuracy beyond a baseline that already uses shot dispersion and historical course variance. The test does not identify whether redundancy, coarse cells, shot dependence or other omitted factors are responsible; those are candidate explanations rather than measured causes.

In fixed baseline-confidence buckets, averaged equally by venue and edition:

| Baseline bucket | Overlapping pairs | Mean baseline win probability | Realized win rate |
|---|---:|---:|---:|
| At least 65% | 212,323 | 70.39% | 71.12% |
| At least 70% | 96,339 | 74.49% | 76.11% |

This panel does **not** show blanket overconfidence in stronger round probabilities. Full replacement slightly worsens Brier in both buckets overall. The blend has tiny favorable overall changes, but its 70%+ change is slightly adverse on the added venues. These exploratory, correlated, unpriced-board results do not establish a high-confidence betting improvement, answer the ≥5%/<5% Kelly selection question, or explain the season's drawdown.

## Scope and interpretation

This extends the [six-course study](SIX_COURSE_EXPOSURE_RESULTS_2026.md) with the same six frozen variants, plus one chronologically fitted blend. The coverage inventory contains 152 eligible editions from 2020–2026, representing 37 layout IDs at 36 physical venues. The original six venues and the additional 30 are reported separately. This is a retrospective, fixed-method R1 comparison, not an archived replay of issued probabilities or a causal explanation of the season's downswing.

Player mean inputs, other category variances, correlations and skew inputs are shared across variants. The retained round kernel, clipping and final skew treatment are used for scoring. The two headline metrics are conditional round-distribution CRPS and non-tied, push-conditioned matchup Brier; lower is better. Relative changes in these scores are neither ROI changes nor percentage-point changes in win probability.

The matchup panel contains all player pairs, including pairings that a sportsbook may never offer. It does not reproduce selected sharp-book bets, bet-time tee-time matching, quoted prices, Kelly edges or finish markets. Shared realized scoring conditions are removed through symmetric field centering for the conditional round-score audit; within-field differences such as weather by tee time remain. Absolute score-line pricing is outside this test.

## Comparisons fixed before scoring

- **Baseline:** the verified shot-overlay method with prior physical-course category multipliers.
- **Course counts:** replace the APP course variance factor with the prior course/tour APP shot-count ratio.
- **Course exposure:** add the nine-cell distance/lie mixture to that count-based replacement.
- **Player-relative exposure:** conservative and less-shrunk allocation across players, preserving field-average APP variance.
- **Combined replacement:** course exposure plus the less-shrunk player allocation.
- **Learned blend:** interpolate APP variances between baseline and course exposure with a scalar weight learned only from completed earlier editions. The weight is bounded from zero to one. Training uses equal venue weight and equal edition weight within venue, a Gaussian variance-loss approximation, fixed baseline means, category covariance and the effect of field centering. With fewer than five editions, three venues or 500 player cases, the declared fallback is 50/50. Evaluation uses the full simulation engine, rather than this fitting approximation.

The four primary contrasts are incremental distance/lie versus counts, less-shrunk player allocation versus baseline, combined replacement versus baseline, and learned blend versus baseline. Simultaneous course-cluster intervals account for these four comparisons separately within each metric. Event and repeated-player resampling, balanced early/later course panels, and leave-course/year-out checks are sensitivities. The blend was proposed after seeing the six-course results; chronological fitting does not turn that earlier exploration into a fresh holdout.

## Course identity and coverage

The inventory lists 308 editions: 152 initially eligible, 68 without 100 complete prior course APP player-rounds, 58 without a validated processed source event, and 30 with multiple physical courses. Modeling also requires at least 20 players and 80% skill coverage. Sparse individual shot histories use explicit fallback estimates rather than silent exclusion.

PGA National IDs 734 and 928 represent the same venue across a par change. All 18 holes match geometrically (coordinate RMSE 9.04; maximum displacement 50.43 in the archive's coordinate units), but total par changed from 70 to 71. Separate layout histories are retained for forecasting; both are grouped as one venue for training weights, headline averages and uncertainty. The verified Workday/Muirfield and Renaissance aliases from the earlier study remain in place.

## Renaissance 2022 reconstruction finding

The current shot adapter adds three missing-leading-stroke repairs to the 2023 reference export. Running the archived processing vintage in memory reproduces all 31,348 reference shots, their validity and complete-hole flags, and all 452 player-round category summaries exactly. This version adjustment is isolated to research; the shared adapter and source database were not edited.

The 2022 gap is in the original measurements. All 33,319 retained stroke records have blank distance and distance-remaining fields. None of the 25,003 nonterminal records has usable endpoint coordinates. The 463 retained compressed payloads contain the same omissions; this is not merely a loss during normalized-table export. Stroke sequences and lie labels cannot recover distance-dependent SG faithfully. The reconstruction fails its minimum putting-observation gate and 2022 remains excluded, leaving 2023 without sufficient validated prior course APP history. Renaissance therefore remains 2024–2026 in this test.

## Fidelity and evidence

The normalized-history extension preserves all 425,045 previously validated rows exactly and adds the completed-event history required through the August 27 cutoff. All 204 original input sets—34 editions times six variants—also reproduce exactly. Modeled coverage is 18,201 of 18,696 roster appearances (97.35%); 511 admitted appearances have no prior complete APP rounds and 1,259 have fewer than 12, with explicit fallback treatment. Normalization training uses only matured historical windows. External field-strength and processed shot-SG vintages remain fixed; this does not reconstruct the exact information version available to a bettor at every historical date.

The reproducible research folder is `permanent_data/research/all_course_exposure_20260910/`. It contains the frozen contract, complete eligibility inventory, course-identity audit, normalized histories, per-edition input arrays, chronological blend training ledger, retained probability distributions, source hashes, code snapshots, precision records and independent verification receipts. Each scored edition uses two fixed seeds with 50,000 draws per player; declared precision failures require the full paired edition to be rerun at 200,000 draws per seed.

All 152 editions passed the two-seed precision checks at 50,000 draws; no refinement was required. Target/future mutations, non-vacuous prior-data controls, actual-field production-skill parity, all-layout Python/Rust checks, retained final-PMF reproduction, independent blend optimization/weighting checks, exact training-cutoff membership, discrete probability and CRPS oracles, grouped aggregation and source/input hashes pass. The chart was visually inspected. Primary evidence: [verification](permanent_data/research/all_course_exposure_20260910/final_verification.json), [all edition effects](permanent_data/research/all_course_exposure_20260910/paired_deltas.csv), [uncertainty](permanent_data/research/all_course_exposure_20260910/uncertainty.csv), [balanced panels](permanent_data/research/all_course_exposure_20260910/balanced_panel_sensitivity.csv), and [blend weights](permanent_data/research/all_course_exposure_20260910/blend_weights.csv).

Production inputs and betting rules are unchanged. The [living variance brief](VARIANCE_RESEARCH_BRIEF_2026.md) records the decision and next step.

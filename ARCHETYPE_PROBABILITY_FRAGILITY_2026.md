# Archetypes, probability fragility and model changes

2026-09-09. Exploratory review of actual logged 2026 signals. No model, betting policy, grading, account or external system changed.

**The season does not show that our highest round-matchup probabilities are generally the most fragile. The more defensible concerns are specific selection errors, thin probability cushions in finish markets, correlated exposure, and a season containing substantially different data/model versions.**

## Probability confidence is different from theoretical edge

Primary samples: 643 sharp-book round-matchup signals across 29 events, and 783 pre-event YES top-5/10/20 signals across 28 events. Books: Pinnacle, BetOnline and Betcris. Earliest quote per book first, then best edge in the first available run per matchup side or player/finish market. Logged signals are not independently reconciled account fills. Finish markets exclude outrights, NO contracts, exchanges and live bets. Do not generalize these tables to those products.

Round fair probabilities are conditional on a decisive result; 87 pushes are excluded from probability scoring but remain in the ROI risk denominator. Finish probability means expected payout share under dead-heat settlement, not a binary qualification rate.

| Round model probability | Signals / decisive outcomes | Mean model probability on decisive outcomes | Realized win rate | ROI on all stakes |
|---|---:|---:|---:|---:|
| <50% | 107 / 90 | 46.6% | 46.7% | +5.9% |
| 50–55% | 212 / 177 | 52.6% | 51.4% | +3.9% |
| 55–60% | 199 / 175 | 57.3% | 55.4% | +3.3% |
| 60–65% | 92 / 82 | 61.8% | 54.9% | −4.1% |
| ≥65% | 33 / 32 | 69.7% | 71.9% | +11.0% |

The highest-probability group is small and 19 of its 33 signals are Studs. Its 95% event-bootstrap calibration-error interval spans roughly −10 to +19 percentage points: this is no guarantee of accuracy. It does, however, contradict a general claim that higher probability produced progressively worse calibration. The 60–65% dip includes −5.00u on ten Balanced APP signals and −7.31u on seven OTT_APP signals; those exploratory slices are not entry rules.

Larger round edges also did not fail uniformly. The 308 signals at 5–10% EV returned +6.1%; 117 at 10–20% returned +9.4%. Only two exceeded 20%. The 216 <5% signals returned −3.8%. After controlling descriptively for round and broad July phase, a ten-point increase in probability has approximately zero estimated relationship to calibration error, with a wide interval. Adding model-market disagreement does not establish a reliable deterioration either.

The vig-free reference uses both sides of the same offered matchup quote. Round model log loss was 0.6834 versus 0.6889 for that market reference; lower is better. The paired event-clustered difference interval is −0.0171 to +0.0060. This supports neither replacing the model with the market nor declaring a durable model advantage. The 27 signals with ≥10pp model-market disagreement returned essentially flat (+0.07u), versus +29.78u on 274 signals at 6–10pp; small extreme bins do not establish a monotonic selection-error rule.

## Why finish edges can look large yet be sensitive

The probability cushion is `model probability − 1/decimal odds`. Theoretical EV is `model probability × decimal odds − 1`. A 12% probability at decimal 10 produces 20% EV, but moving it to 10% eliminates the entire edge. A 60% probability at decimal 2 also produces 20% EV, with a ten-percentage-point cushion.

| Fixed sensitivity scenario | Fraction losing positive EV after subtracting 2pp |
|---|---:|
| Round, ≥5% theoretical EV | 0.2% (1/427) |
| Round, <5% | 51.4% (111/216) |
| Finish, ≥5% | 80.1% (490/612) |
| Finish, <5% | 99.4% (170/171) |

This is a sensitivity calculation, not an estimated forecast-error bound or a profitable filter. Indeed, the finish bets that survive this stress had worse realized ROI than those that fail it. A universal two-point deduction would be an unjustified model change.

Of the 783 finish signals, 506 had predicted payout share below 10%. They averaged 5.18% expected versus 3.43% realized. The 183 at 10–20% averaged 13.66% versus 8.32%. The sixteen at ≥40% averaged 49.12% versus 54.17%; their slightly negative stake-weighted return shows why average calibration and portfolio P&L are different measures. The observed finish concern is concentrated in lower-probability tail outcomes, not established overconfidence in the largest probabilities.

![Probability calibration and sensitivity](permanent_data/research/archetype_fragility_20260909/probability_fragility.png)

## Complete archetype screen

Each cell is **signal count / recorded profit**. Round and finish units use different sizing rules, so profit alone does not compare forecast quality. Detailed probabilities, ROI, unique observations, exposure and intervals are in the CSVs.

| Archetype | Round ≥5% | Round <5% | Finish ≥5% | Finish <5% |
|---|---:|---:|---:|---:|
| APP_ARG | 10 / -1.94u | 7 / +0.45u | 1 / -0.35u | 5 / -0.10u |
| APP_PUTT | 3 / +1.11u | 5 / +0.82u | 13 / -3.72u | 3 / -0.03u |
| ARG_PUTT | 23 / +1.90u | 15 / +2.15u | 22 / -3.09u | 4 / -0.11u |
| Balanced | 26 / +2.48u | 8 / +1.89u | 83 / -5.34u | 25 / -0.86u |
| Balanced APP | 45 / -13.27u | 37 / -7.53u | 52 / -4.27u | 23 / +0.63u |
| Balanced ARG | 17 / +6.64u | 9 / +3.50u | 21 / -1.80u | 8 / -0.18u |
| Balanced OTT | 18 / +2.66u | 8 / +3.03u | 10 / -0.30u | — |
| Balanced Putter | 42 / +2.14u | 18 / -5.58u | 41 / -6.07u | 14 / +0.41u |
| Ball Striker | 18 / +3.22u | 7 / +0.95u | 28 / -0.36u | 8 / -0.21u |
| Elite Putter | 19 / +3.91u | 14 / -6.24u | 26 / -2.53u | 5 / -0.18u |
| Long Accurate | 14 / -1.12u | 9 / -1.66u | 23 / -2.34u | 5 / -0.16u |
| Long Wild | 40 / +9.58u | 13 / +0.07u | 79 / +4.26u | 14 / +0.21u |
| Low Skill | 10 / +3.43u | — | 43 / +3.08u | 12 / -0.24u |
| OTT_APP | 20 / -7.16u | 11 / +0.93u | 8 / +2.11u | 4 / -0.04u |
| OTT_PUTT | 22 / +4.12u | 13 / -1.06u | 9 / -3.19u | 3 / -0.47u |
| Short Accurate | 32 / +7.17u | 4 / -1.08u | 81 / +16.69u | 13 / -0.43u |
| Short Game Specialist | 18 / +2.67u | 1 / +0.00u | 57 / -2.10u | 16 / -0.23u |
| Stud | 50 / +6.39u | 37 / -0.50u | 15 / +1.70u | 9 / +0.48u |

**Balanced APP remains the first round replay priority.** Across 82 signals it returned −20.80u, with 36.4% wins versus 53.9% expected on decisive bets. Losses existed before the July repair (65 signals, −18.59u) and continued afterward (12, −3.02u), so this pattern did not originate with that repair. Late evidence is too sparse to say whether the repair changed it.

The pair matters: across those signals, our backed players' historical APP error was −0.183 SG while opponents' was +0.225; putting errors were −0.357 versus +0.499. These ticket-weighted comparisons show substantial opponent outperformance as well as own-player misses. They do not justify attributing the entire loss to overrating the backed player's APP. On unique backed player-rounds, APP error was −0.180 and putting −0.258. The previous finish audit's −0.268 APP error remains valid, but finish probabilities and actual category skill require separate diagnoses.

**Balanced Putter is the second replay priority.** It lost 3.44u on 60 round signals and 5.65u on 55 finish signals. Unique selected player-round putting error was −0.487 SG, versus −0.019 for unselected Balanced Putters in the same broad event/round opportunities. On first-two-round finish player-events it was −0.347 versus +0.003. Those are exploratory population comparisons, not matched causal estimates. Fowler contributed −6.85u in round bets, more than the group's net loss; the finish losses were spread across MacIntyre, Fowler, English, Knapp and Burns. Within edge bands the pattern differs: ≥5% round bets were profitable; <5% were weak.

**Other losses often point somewhere else.** Elite Putters lost money despite favorable own putting errors (+0.245 SG in rounds, +0.164 in finish first-two-round checks). OTT_APP round losses occurred despite favorable own APP error, and that archetype made money on its small finish sample. Balanced and several dual-specialist finish groups lost heavily, but rare top-N hits, repeated-player exposure and tiny samples prevent reliable rankings. No specialist-wide APP or putting penalty follows from these results.

Recorded recent-history counts also do not isolate the weakness: round samples of 20–29 returned +9.3% ROI, 30–49 +3.0%, and ≥50 −0.6%, with overlapping uncertainty. Thirty-four rows lack that count. Reconstructed increases of ≥0.5 SG in the backed-minus-opponent live skill gap returned +8.8% over 105 signals; more positive live uplift is not monotonically associated with worse results. The live-gap analysis has 619/643 coverage and inherits the prior reconstruction's vintage limitations.

## The midseason fixes are a material confounder

The adjacent sim_prep repository documents a **July 11 data repair**, followed by consolidation, historical rebuilds, refits and pipeline hardening. This is separate from the June simulator transition. The July 11 verification report explicitly says data repairs preceded some required production refits; commit dates alone cannot date effective deployment for every bet.

| Documented period | Relevant change | Why it could affect these findings |
|---|---|---|
| March 9 / March 17 | Category-first tournament / round defaults | The early season did not use one unchanged distribution engine. |
| June 8–15 | Course-specific variance filtering, tail treatment, Rust defaults, live skill/cut fixes, rotating-course history blending | Can change spread, tails, field membership or adjustment behavior. Earlier audit did not establish that the June live fix caused R3 losses. |
| July 11–14 | Duplicate skill records; cross-tour grouping errors; player identity; post-cut field strength; synthetic-row and future-skill leakage; missing-category fallback; training/prediction course-file mismatch | Could change learned means and which apparent edges are selected. The missing-category fallback previously used −1.5 for many training rows; the post-cut correction changes historical skill levels. |
| July 13–21 | Per-event survivor uplift, historical skill rebuild, recentering/refits and guarded weekly input publication | Changes the data/model combination. Profile-shrinkage code was added then explicitly disabled for production July 21; its existence does not prove use. |
| July 25–August 15 | Residual caps, stale-adjustment fixes, reset-to-base updates, current kernel pins, prediction-column gates and feedback changes | Can alter some live means or eligibility; requires bet-specific runtime provenance. |
| August 18 | Shared player-week form plus total-score skew correction; market regression retune | Changes joint tournament tails even with approximately unchanged one-round variance. Particularly relevant to finish probabilities. |

For a broad sensitivity, label July 11–19 as a transition and July 20 onward as post-transition. **These are availability-era proxies, not verified treatment labels.**

| Market | Before July 11 | July 11–19 | July 20 onward |
|---|---:|---:|---:|
| Round signals / events | 491 / 22 | 58 / 2 | 94 / 6 |
| Round profit / ROI | +34.30u / +6.2% | +0.78u / +1.0% | −11.01u / −9.8% |
| Finish signals / events | 654 / 22 | 46 / 1 | 83 / 5 |
| Finish profit / ROI | +7.565u / +7.8% | −7.089u / −100.0% | −9.581u / −75.8% |

The finish opportunity mix changed substantially: 73/83 later signals were top-5, none top-20, versus 392/654 top-5 and 88 top-20 before repair. The later finish sample risked only 12.64u. All five events lost money. This is worth investigating, but five events plus multiple overlapping changes cannot identify which fix helped or hurt. After August 18 specifically, there are only 13 finish signals from BMW; that cannot assess the latest distribution model.

Simple common-mix reweighting by archetype or probability bucket leaves a late calibration shortfall in both markets, so those measured mix changes do not mechanically explain all of it. This reweighting has small cells and omits unobserved selection and deployment differences; it is not a causal repair-effect estimate. Whole-event bootstrap round post-transition ROI spans roughly −32% to +4%. All season-wide interpretations must preserve version uncertainty.

The August 18 release record reports that adding player-week dependence corrected understated multi-round variance while keeping marginal round variance approximately unchanged, and required a companion skew correction. That is a concrete explanation for why credible one-round means do not guarantee credible top-N probabilities. It is documented mechanism evidence, not a replay proving the cause of these season losses.

## Recommended next work

1. **Create a bet-time version map before fitting a correction.** Recover dated pre_sim summaries, category distributions, coefficient/data snapshot versions, live files, flags and kernel versions around the July repair and August changes. Prioritize the five late finish events and the largest repeated player exposures. A current corrected DB reconstruction is a useful benchmark, not proof of what pre-repair runs consumed.
2. **Replay Balanced APP and Balanced Putter selections, including opponents and non-selections.** Decompose raw category means, total-SG recentering, course terms, market regression and live adjustments. Start with Cantlay/Rose/Hojgaard and Fowler/Knapp; compare issued outputs with the repaired pipeline on identical archived inputs and quotes where possible. Do not infer a bad category from a losing ticket alone.
3. **Validate finish tail calibration separately from round accuracy.** Hold means fixed while testing player-week dependence, category spread/skew and cut/dead-heat treatment. Inspect complete probability menus, not just selected longshots. Distinguish model misspecification, simulation uncertainty and outcome randomness.
4. **Report probability cushion and correlated player exposure alongside EV.** This makes sensitivity visible; it is not approval for an automatic stress filter or stake change. Any proposed adjustment must improve chronologically held-out probability scores and economic results across both edge bands, with event/player dependence preserved.

## Scope, evidence and verification

Archetypes use strict pre-event historical windows and the original round-study reference populations. The database is a corrected September 1 snapshot; past statistical revisions and repairs can affect retrospective profiles. Category residuals here compare adjusted SG with historical profiles, not exact issued category forecasts. Selected-versus-unselected round observations are unique player-rounds; finish observations are unique player-events over the first two rounds, avoiding weekend survivor bias. Opponent and exposure tables retain the appropriate ticket-level relationships.

All eighteen observed archetypes were screened with fixed bins. Intervals are exploratory and do not correct the whole research history of multiple comparisons; recurring players can also link events. Sparse near-zero-hit finish cells should not be assigned reassuring residual-Wald intervals, so those are suppressed. No archetype was selected for production on this evidence. The 2pp stress is not an uncertainty interval. Outcome variance remains plausible for the round swing; the data support a targeted investigation, not a proven general failure of high probabilities.

Inputs preserve the prior audits' source selection, legacy-edge handling and invalid-row exclusions. All finish settlements were previously independently reconciled; primary probabilities, stake formulas, quote orientation, outcome accounting, uniqueness and archetype coverage were checked again. The field comparison covers all 551 unique backed round player-rounds and 538/541 unique finish player-events with the required first two rounds. Current source hashes were unchanged; `verification.json` records the checks. Scripts only read existing data and write this research folder/report. Sources: [July verification](../sim_prep/docs/FIX_VERIFICATION_REPORT.md), [July repair inventory](../sim_prep/docs/AUDIT_AND_CLOUD_MIGRATION_HANDOFF.md), [July recalibration notes](../sim_prep/notes/RECALIBRATION_PASS2.md), [August handoff](FOLLOW_UPS.md), and inspected Git commits recorded in `model_timeline.csv`.

Reproduce with `analyze.py`, `detail.py`, then `write_report.py` in [the evidence folder](permanent_data/research/archetype_fragility_20260909/). Main outputs: `summary.csv`, `cross_summaries.csv`, `field_selection.csv`, `opponent_archetypes.csv`, `conditional_regressions.csv`, `detail.json`, source hashes and full row-level evidence. Exploratory regressions are descriptive controls, not fitted production corrections.

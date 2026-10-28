# Four-round SG-category distribution results

Completed 2026-09-10. Research only; production inputs, update rules and publishing were unchanged.

**The strongest actionable finding is excessive separation between low- and high-variance ARG players. The concern about overly generous extreme APP upside appears in the no-cut sample, but it does not repeat clearly in the cut-event sample or establish a general failure of four-round regression.** A fixed, partial variance shrink improves cumulative ARG forecasts modestly. APP's improvement is smaller and less certain.

This evaluates the distribution of each player's **sum of OTT, APP, ARG and PUTT across four rounds**, with the current in-event updating process retained. It does not substitute a next-round prediction test for the full-event question.

## What was tested

The main sample contains **16 no-cut editions at seven physical courses, 932 player-event outcomes and 177 distinct players**. A separate cut-conditional sample contains **four editions at four additional courses, 386 starters and 249 four-round completers**. All selected editions have complete modeled actual-starter fields and complete category measurements. Editions with withdrawals/disqualifications were excluded. Older Tour Championships with starting strokes were excluded.

No-cut courses: Kapalua, TPC Southwind, Harbour Town, Quail Hollow, TPC River Highlands, East Lake and Albany. Cut events: Charles Schwab 2023, Genesis 2026, Players 2026 and Memorial 2026. The complete [eligibility matrix](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/event_category_shapes_20260910/eligibility.csv>) records exclusions from the original 152-edition foundation. Coverage therefore limits this study, especially its application to ordinary full-field cut events.

Three comparisons were frozen before cumulative outcomes were scored:

1. Current reconstructed category-variance baseline.
2. Move each player's APP variance halfway toward that event's average APP variance.
3. Apply the same halfway move to APP and ARG.

The 50% choice was a fixed diagnostic candidate, not a fitted optimum. Initial player/category means, course multipliers, skew, correlations, current update coefficients and weekly latent SD of 0.65 remain fixed. The change preserves each event's average **input** category variance. Existing nonlinear updates, clipping and the latent's variance adjustment can still produce small changes in cumulative means and in other categories; those effects are included in the scores.

Each event/arm used two independent seeds of 50,000 tournament paths. Metrics are averaged across those two estimates. Primary aggregation gives each course equal weight, each edition equal weight within course, and each player equal weight within edition. Alternative player-case weighting and event/player uncertainty are retained.

## Full-event scale and both tails

The ratio below is average squared forecast error divided by average simulated variance. One is the reference. It includes mean-estimation error, so it is not automatically a pure variance estimate. Positive SG is good; the two tail columns show how often outcomes exceeded the model's good-week or bad-week 10% boundary.

| Category | Four-round error/variance ratio | Descriptive 95% course interval | Good-week tail, expected 10% | Bad-week tail, expected 10% |
|---|---:|---:|---:|---:|
| OTT | 0.921 | 0.800–1.050 | 8.3% | 8.8% |
| APP | 0.927 | 0.826–1.025 | 9.3% | 10.2% |
| ARG | 1.049 | 0.980–1.112 | 11.7% | 10.4% |
| PUTT | 1.140 | 1.020–1.239 | 12.2% | 9.9% |

These are no-cut results. They do not show a large, uniform APP right-tail excess at the 10% boundary. PUTT has more cumulative error than forecast variance, but this requires a like-for-like round comparison before blaming the update process.

![Observed four-round category tails](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/event_category_shapes_20260910/cumulative_tails.png>)

## APP: extreme upside and an ordinary good week behave differently

Among the **highest pre-event APP-variance quintile**, there are 180 no-cut player-event outcomes, involving 79 distinct players:

| Diagnostic | Forecast reference | Observed |
|---|---:|---:|
| Good-week 10% boundary | 10% | 11.3% |
| Extreme good-week 5% boundary | 5% | **1.1%** |
| Bad-week 10% boundary | 10% | 8.6% |
| Extreme bad-week 5% boundary | 5% | 2.8% |
| Error/variance ratio | 1.000 | **0.842** |

The model gives these players too much extreme upside in this sample, while ordinary strong weeks occur readily. Both extreme tails occur less often than forecast. That is more specific than saying the model lacks regression after every good APP round. It suggests inspecting the outer distribution separately from its central spread.

The group's average actual-minus-forecast APP total is **+0.26 SG**. Thus its far-upside problem is not simply an overly optimistic average APP forecast. A clearly labeled, retrospective removal of this group mean bias leaves the error/variance ratio almost unchanged, at 0.836; this is explanatory analysis, not a learned correction.

There are important counterexamples:

- Only approximately **two to three of the 180 outcomes** exceed the simulated 95th-percentile boundary, depending on seed. The 1.1% figure is course-balanced; the unweighted frequency is about 1.4%. Rare-tail inference is fragile, and resampling cannot invent extreme outcomes absent from a small sample.
- The high-variance APP ratio is 0.773 before 2026 but **1.020 in 2026**, with only 49 current-year cases. The apparent excess spread is substantially an earlier-sample finding.
- Among **cut-event completers**, high-variance APP has a ratio of **1.005** and exceeds the good-week 5% boundary **8.3%** of the time, across 49 cases. That conditional sample does not reproduce the no-cut extreme-upside shortage.

**Decision:** the outer APP tail deserves a targeted check; a blanket APP variance reduction or stronger regression is not established. Variance quintiles here are also distinct from the earlier “balanced APP” skill archetype.

## ARG: the clearest improvement is in player variance allocation

| Pre-event ARG group | No-cut error/variance ratio | After APP/ARG half-shrink | Cut-conditional baseline ratio |
|---|---:|---:|---:|
| Lowest-variance quintile | **1.424** | **1.304** | **1.583** |
| Highest-variance quintile | **0.898** | **0.975** | **0.888** |

The baseline treats the low-variance ARG group as too predictable and the high-variance group as somewhat too variable. Bringing those estimates closer together helps without changing average input ARG variance. The low group's mean error is +0.26 SG; removing that retrospectively leaves a ratio of 1.407, so mean bias explains little of the squared-error shortfall.

The magnitude is not constant over time: the low-variance no-cut group's ratio is only 1.092 in 2026. However, the partial-shrink distribution-score improvement remains favorable in 2026 and in both chronological halves. This supports testing restrained player-variance shrinkage, not imposing a permanent 42% variance uplift on this group.

## How much did the candidate improve forecasts?

CRPS scores the whole predictive distribution in SG units; lower is better. The primary family adjustment covers eight category-by-challenger contrasts. The following table compares combined APP/ARG shrinkage with baseline.

| Category | No-cut CRPS change | Relative change | Simultaneous 95% course interval for change | Cut-conditional relative change |
|---|---:|---:|---:|---:|
| OTT | +0.000010 | +0.001% | −0.000016 to +0.000035 | +0.020% |
| APP | −0.001024 | **−0.059%** | −0.002143 to +0.000173 | −0.041% |
| ARG | −0.001288 | **−0.102%** | **−0.002368 to −0.000628** | −0.153% |
| PUTT | −0.000015 | −0.001% | −0.000060 to +0.000032 | −0.003% |

The APP-only arm gives essentially the same APP benefit (−0.058%) and essentially no ARG benefit. The ARG gain appears when ARG variance shrinkage is added.

ARG's no-cut improvement remains negative under event and recurring-player cluster intervals, both seeds, both chronological halves, the 2026 panel, and every leave-one-course-out comparison. APP's point estimate also improves in both halves and every leave-one-course-out comparison, but its adjusted course interval and its recurring-player interval include zero. These are small distribution-score gains, not measured betting-profit gains.

Player-case weighting gives the same direction, with approximately 0.041% APP and 0.085% ARG improvements. No-cut seed-to-seed changes in the paired APP/ARG score gains are below 0.000006 SG, much smaller than the gains themselves. The cut sample is less precise; its results are corroborative, not an independent broad validation.

## Does this implicate momentum or regression across the four rounds?

**This test includes the existing update process, but does not isolate it as the cause.** R2/R3/R4 depend on simulated earlier performance and position; fresh adjustments replace earlier adjustments, and total-skill shifts are divided across categories. Those rules were held unchanged in every arm.

As a descriptive composition check, R1 was evaluated on the **same 932 no-cut player-event cases**, with the same initial means and current latent setting:

| Category | R1 error/variance ratio, identical sample | Four-round ratio |
|---|---:|---:|
| OTT | 0.751 | 0.921 |
| APP | 0.870 | 0.927 |
| ARG | 0.889 | 1.049 |
| PUTT | **1.124** | **1.140** |

Putting's shortfall is already present at R1 in this sample. Comparing the four-round ratio with the earlier 152-event R1 audit would confound time aggregation with a change in events and player mix. The other categories change as rounds accumulate, but R1 alone cannot separate later-round marginal errors, cross-round dependence and updating. None of those components was tuned here.

The high-variance APP group's ratio is already 0.801 at R1, versus 0.842 cumulatively. Its excess spread therefore does not require a four-round compounding explanation. ARG's low-variance group moves from 1.037 at R1 to 1.424 cumulatively, making that group the more useful category-specific follow-up.

## Fidelity, selection and limits

- **Engine:** an isolated copy of the Rust kernel adds cumulative category outputs after the original random draws and original output calculations. All seven original outputs match the installed kernel exactly in eight BMW/Memorial fixture combinations, covering cut/no-cut, baseline/shrink and active/inactive latent. Passing both through the existing final total-score adjustment also matches exactly. Production code and the installed package were not edited.
- **Category accounting:** raw cumulative draws recover the existing sum of four round-category means to within 1.1e−14. Their sum reconciles with pre-reshape integer tournament scores within the expected four-round rounding bound of two strokes. No player-specific source category/total discrepancies above 0.01 SG remain after removing common round offsets in these 20 editions. The earlier nine-edition discrepancy sensitivity therefore does not remove any selected edition here.
- **Cut selection:** every simulated event begins with the full modeled actual-starter field. Documented top-65 or top-50/ten-shot rules reproduce the observed weekend roster exactly. Each actual completer is scored against that player's own simulated made-cut distribution. No hypothetical missed-cut weekend is scored. At least 14,349 conditional draws are available per scored player/seed. Make-cut probabilities and Brier scores for all 386 starters are saved separately; they are practically unchanged by the candidate. [Rule sources](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/event_category_shapes_20260910/rule_sources.md>).
- **Target:** category SG is centered within each round's active field, symmetrically for observations and simulations. Weekend simulated centering uses simulated survivors. This is a field-relative category diagnostic with zero forecast-weather deltas, not an exact historical weather or finish-price replay.
- **Boundary:** category draws precede the simulator's final total-score skew reshaping. That adjustment has no category allocation. The category study therefore does not validate the final finish-position distribution, prices, or the selected ≥5% Kelly betting group.
- **Chronology:** frozen player/course inputs retain prior-event cutoffs; all selected editions have cutoff receipts, and the earlier adversarial future-history test covered a representative edition of every selected course. Initial category SDs match the frozen shot-plus-course baseline exactly. Current cached update coefficients are frozen for the whole historical comparison. This cannot attribute the season's downswing to a particular midseason model version; external historical SG vintages are not completely point-in-time.
- **Uncertainty:** seven primary courses and four secondary courses are few independent settings. All historical outcomes have been examined before. Course intervals are descriptive for scale/tails; score comparisons also have an eight-contrast adjustment. Rare-event bootstrap intervals, especially zero-event cells, should not be treated as proof that an unseen tail cannot occur. WD/DQ exclusions and elite-field coverage also limit generalization.
- **Verification:** independent CRPS and aggregation identities pass; physical-course identity is verified in all four rounds; original Rust sources, source DB and frozen inputs match their opening hashes. The same-sample R1 integration agrees between 24- and 48-node quadrature within 3.2e−15. [Completion receipt](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/event_category_shapes_20260910/completion_manifest.json>).

## Recommended next work

1. **Prioritize ARG player-variance shrinkage for chronological validation.** Test a restrained rule learned only from earlier events, retaining the current means/course layer/updating. Score all four rounds together and keep low/high groups separate. The fixed 50% candidate is promising, not production-approved or optimal.
2. **Keep APP central spread and outer-tail shape separate.** Before adding regression, test whether an APP-specific shape change can improve the 90th–95th-percentile region while preserving mean and variance. Require repetition across no-cut and cut-conditional targets and newer data; the present split does not justify a blanket fix.
3. **Keep a PUTT scale follow-up, using matched samples.** Its cumulative shortfall largely exists at R1 here. Do not describe that as evidence for a new momentum adjustment or a reduction in PUTT variance.

The broader latent/persistence tuning branch remains parked. The category-specific differences above are the useful next leads. No production adjustment follows automatically from this report.

Reproducible design, code, summaries, player cases and sensitivity tables: [research archive](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/event_category_shapes_20260910/contract.json>).

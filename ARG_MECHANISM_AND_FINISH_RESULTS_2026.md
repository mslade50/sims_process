# ARG-only test: four-round error mechanism and final finish forecasts

Completed 2026-09-10. **ARG-only variance shrinkage improves the category distribution slightly, but it does not clear the final-finish accuracy test. Keep production unchanged.** The larger diagnostic lead is repeated ARG forecast errors across rounds, with a separate R4 error contribution that is partly concentrated in one small event.

The analysis retains the requested good-golfer/high-sample definition: baseline total SG >0.5 and either >40 PGA rounds in the trailing 12 calendar months or >=20 in six months, using only completed earlier events. The same 16 no-cut editions and four cut editions are used. Category scoring covers 648 no-cut outcomes and 134 cut-event completers. Finish scoring includes **all 827 qualifying starters**, including the **45 qualifying missed cuts**.

## What creates the low-variance ARG group's cumulative shortfall?

For each round, the error is actual field-relative ARG minus the simulated pre-event marginal mean for that round. The marginal forecasts come from the full four-round cascade with current updating enabled; they are not live predictions conditioned on the actual earlier rounds.

The identity is:

`squared four-round error = sum of squared round errors + twice the sum of cross-round error products`

The same decomposition applies to simulated variance, using round variances and covariances. This directly measures the components rather than comparing R1 with four times R1 variance.

For **149 qualifying no-cut player-event cases in the lowest pre-event ARG-variance quintile**:

| Component, SG squared | Model | Observed forecast error | Gap |
|---|---:|---:|---:|
| Sum of individual-round components | 3.799 | 4.532 | **+0.733** |
| Cross-round component | 0.253 | 1.391 | **+1.138** |
| Four-round total | **4.052** | **5.924** | **+1.871** |

Approximately **61% of the point-estimate gap comes from cross-round error products and 39% from individual-round errors**. This is an accounting split, not a causal attribution or a precisely estimated percentage.

The model's cross-round component adds about **6.7%** to its sum of round variances. The observed cross-products add about **30.7%** to the sum of squared round errors. Removing the subgroup's weighted mean error in each round retrospectively leaves gaps of **0.683** for the individual-round component and **1.167** for the cross-round component. A common subgroup mean bias therefore does not explain the pattern away.

![ARG error decomposition](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/arg_mechanism_20260910/arg_decomposition.png>)

These cross-products are **not pure momentum**. They can also reflect persistent errors in a player's baseline skill, course fit, or an unobserved event-specific state. Nor does every adjacent-round pair behave the same way: observed R1 errors co-move with R2/R3/R4, while the R3–R4 product is slightly negative. A blanket increase in an autoregressive momentum coefficient is not identified by this result.

The distinction is player-group-specific: across **all 648 qualifying no-cut cases**, the cross-round gap is only **0.035 SG²**, versus 1.138 in the lowest-variance group. The aggregate evidence does not support adding the same amount of ARG dependence to everyone.

Uncertainty remains material. Descriptive 95% course-bootstrap intervals are **+0.136 to +1.664** for the individual-round gap and **+0.034 to +2.474** for the cross-round gap. The event-cluster interval for the cross-round gap narrowly includes zero; the recurring-player interval for the individual-round gap also includes zero. Both point gaps remain positive after excluding each course individually.

The current-year panel strengthens the case for separating the mechanisms: in 2026, the individual-round gap is **−0.259**, while the cross-round gap is **+1.031**. Widening every round is not a clean explanation of that panel's cumulative error.

## The individual-round issue is concentrated in R4

| Round | Squared error / predicted variance |
|---|---:|
| R1 | 1.003 |
| R2 | 0.834 |
| R3 | 0.929 |
| R4 | **2.008** |

This is not evidence that all four rounds need wider ARG distributions. The R4 result is also sensitive to Albany/Hero 2025: that event contains only **four** players in this subgroup, with an R4 ratio of 4.36, and receives one course's weight in the course-balanced estimate.

Excluding Albany reduces the aggregate R4 ratio from **2.01 to 1.28**. The overall individual-round and cross-round gaps remain positive, at **0.297 and 0.554** respectively, but much smaller. In 2026 alone, R4's ratio is 1.48 across 41 cases, with a course interval including one. This is a useful diagnostic concentration, not a validated universal Sunday variance multiplier.

Among the 36 qualifying low-variance ARG cut-event completers, the sum of individual-round errors almost exactly matches forecast variance (**4.153 versus 4.158**). The cross-round contribution is **1.456 versus 0.103**. That is consistent with a dependence/forecast-error lead, but conditional selection and only four events limit inference; its player-cluster interval includes zero. No hypothetical missed-cut weekend was included.

## What ARG-only shrinkage changes

The challenger moves each player's ARG input variance halfway toward the full starter field's average ARG variance. APP, OTT and PUTT input variances are untouched. Initial means, course inputs, skew, correlation, current cached updating coefficients and weekly latent SD 0.65 remain fixed. The 50% amount was fixed before scoring; no parameter was fitted to these outcomes.

For the low-variance no-cut ARG group:

| Quantity | Baseline | ARG-only shrink |
|---|---:|---:|
| Predicted sum of round variances | 3.799 | **4.176** |
| Predicted cross-round component | 0.253 | **0.247** |
| Cumulative error/variance ratio | 1.462 | **1.339** |

It improves cumulative calibration by adding round-level variance, while barely changing the cross-round component. It therefore helps one part of the error structure and leaves the larger measured cross-round gap essentially intact. In the cut-event low group, it increases the sum of round variances from 4.158 to 4.569, despite the baseline individual-round sum already matching the observed error sum.

Across all qualifying players, ARG cumulative CRPS improves **0.125% in no-cut events** and **0.041% conditional on making the cut**. The no-cut absolute gain is −0.001567 SG, with the conservative eight-contrast adjusted course interval **−0.003288 to −0.000555**. It persists in both seeds, both chronological halves, 2026 and all course exclusions. The cut gain is uncertain and one event worsens. Other category-score changes are very small.

This confirms that the previous category-score gain did not require changing APP variance. It does not establish that round-variance shrinkage fixes the main full-event mechanism.

## Does it improve final finish probabilities?

**No reliable improvement was measured.** Both arms were passed through the existing final total-score skew adjustment and production aggregation routines before grading.

The primary finish score is the average top-5/top-10/top-20 **including-ties Brier score**, averaged over available thresholds, players, editions and courses. Thresholds at or above field size are excluded, so Hero's certain top-20 outcome is not scored. These probabilities are graded against actual finish outcomes for all qualifying starters, including missed cuts. Dead-heat payout-share forecasts are evaluated separately against realized payout shares.

| Metric, ARG-only versus baseline | No-cut relative change | Cut-event relative change |
|---|---:|---:|
| ARG category CRPS | **−0.125%** | −0.041% |
| Final top-finish Brier | **+0.008%** | **+0.023%** |
| Final dead-heat payout-share squared error | +0.007% | +0.028% |

Lower is better. Final-finish point estimates worsen slightly, but the changes are very small and uncertainty includes zero:

- No-cut Brier change: **+0.0000140**, course interval **−0.000028 to +0.000063**.
- Cut-event Brier change: **+0.0000334**, course interval **−0.000020 to +0.000107**.
- Event and recurring-player intervals also include zero. Both seeds give a slightly worse composite Brier point estimate. The 2026 panels move slightly in the favorable direction, with uncertainty including zero, so there is no stable finish gain across time.

The higher-probability finish panels do not rescue the candidate. For baseline probabilities >=0.50, score changes remain uncertain. The >=0.75 panel worsens at the point estimate, but has only 48 no-cut player-event-market cells and three cut-event cells; it is exploratory, not a new betting filter. Win forecasts were checked separately and do not provide a convincing improvement either.

The finish test crosses the previously untested final-score adjustment boundary, but remains a **fixed-method historical reconstruction**: current coefficients and zero forecast-weather deltas are applied to frozen historical inputs. It is not an exact replay of historical issued prices, selected Kelly edges, market regression or realized betting returns. No production promotion should be based on the small category gain alone.

## Fidelity and archived proof

- The isolated research kernel adds ARG round readouts after the original draws and outputs. All nine outputs from the preceding research kernel match exactly on BMW/Memorial baseline and ARG-only fixtures; current production source files remain unchanged.
- ARG round sums match the saved cumulative category draws. Every player's simulated covariance matrix sums to its cumulative variance. The observed and predicted decomposition identities reconcile with separately stored cumulative metrics within 1e−10.
- Every baseline player/category forecast in all 20 historical events reproduces the preceding saved forecast, including means, variances, CRPS and extreme quantiles, within 1e−12.
- Both arms use two 50,000-path seeds. Primary finish probabilities pool the seed estimates before scoring; seed-specific results are retained. No-cut paired ARG score gains are −0.001566 and −0.001568; paired finish Brier changes are approximately +0.000014 in both seeds.
- Independent ranking/dead-heat calculations match the production aggregator, including deliberate ties at payout boundaries and small fields. Actual top-finish indicators derived from scores match official finish labels in every selected event. Actual winners are uniquely identified for the supplemental win check.
- All 827 qualified starters appear in finish evaluation. All 45 qualifying missed cuts receive zero realized top-finish/payout outcomes. Category analysis remains conditional on actual completion in cut events.
- Source DB, frozen distributions, membership, cached coefficients, prior kernel sources and production source hashes match their opening values. No model, source data or publishing change was made.

## Decision

**Do not promote the ARG-only shrink rule.** It earns a small category-score gain but has not improved final finish forecasts, and it mainly increases individual-round variance while the larger remaining lead is cross-round forecast-error dependence.

The next useful question, if continuing this branch, is narrowly category-specific: **which pre-event ARG prediction errors persist across the event, and which R4 errors are reproducible outside small influential events?** Distinguish baseline-skill/course-fit error from event-specific persistence before choosing a correction. This result does not justify a generic latent change, a blanket R4 multiplier or a larger shrinkage percentage.

[Frozen design](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/arg_mechanism_20260910/contract.json>) · [Mechanism tables](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/arg_mechanism_20260910/mechanism.csv>) · [Finish comparison](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/arg_mechanism_20260910/finish_contrasts.csv>) · [Verification receipt](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/arg_mechanism_20260910/completion_manifest.json>)

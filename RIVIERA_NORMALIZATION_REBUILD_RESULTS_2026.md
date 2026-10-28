# Riviera normalization rebuild and forecast accuracy

2026-09-09. **The controlled normalization rebuild, dependent player/course regeneration, fidelity checks and R1 accuracy comparison are complete. There is no material accuracy gain in this pilot.** The rebuilt normalization produces a very small deterioration, predominantly through mean predictions; its variance effect is negligible.

## What was rebuilt

Historical source events were processed chronologically across PGA, European Tour and Korn Ferry data. Each source-event skill estimate uses only earlier completed events. Cut-survivor normalization is fitted from training events beginning in 2021 only after their entire ±540-day proxy-ability window has closed. Skill-gap regressors are generated from the rebuilt earlier histories, avoiding dependence on the retrospectively adjusted skill table.

Each historical event retains the correction fitted before that source event. The model available at a Riviera cutoff is also saved; it is not reapplied retrospectively to every earlier event. This conservative online reconstruction differs from the current database's global backfill policy and is the policy evaluated here.

The three forecast cutoffs admit 74, 172 and 363 calibration training events respectively. The registered early-history fallback is 0.25 when there are fewer than 30 eligible events, an unsupported tour, or a deficient regression design. The model retains the fitted slope without target-result selection and clamps uplift to [0, 0.6]. These chronology/maturity policies are a specified reconstruction, not a claim to reproduce original historical deployment decisions.

For Riviera 2023, 2024 and 2026, the rebuild regenerates player skills, category distributions, sparse-history priors, course SD multipliers/skew, shot-overlay scaling and final simulation inputs. The corrected shot-event calendar is reused. Physical course 500 determines course history; 2025 Genesis at Torrey Pines is excluded.

Provider full-field strength, current shot-SG processing, skill-weight arrays, correlation, distribution-policy constants and the final skew target are held fixed across comparisons. The provider archive has no publication timestamps establishing historical availability. Consequently this is a **controlled retrospective R1 comparison with chronological fitted normalization**, not certification of a fully as-issued or entirely historically selected model. It does not close the separate finish/tournament fidelity gate.

## Comparison fixed before outcome scoring

The common sample contains 268 player-edition observations: 126 in 2023, 70 in 2024 and 72 in 2026. Three 2023 starters lack a usable common skill estimate and remain excluded. Six arms separate the original reconstruction, calendar correction, variance changes with all category means fixed, mean changes with variance/skew fixed, the full rebuild, and a current-recipe control on original histories.

The control reproduces stored skills exactly for all 268 admitted cases. The normalization comparison therefore is not being driven by a skill-computation version discrepancy.

Every arm uses 200,000 simulations for each of two independent seeds, with matched raw random streams between arms and the production round-skew postprocessing. There are 36 scored runs. Conditional score evaluation removes common field scoring conditions symmetrically; matchup evaluation uses the unpriced, non-tied R1 board, totaling 11,531 overlapping pairs. Those pairs are not independent evidence or actual wagers. No finish probabilities, odds-based selections or Kelly edge groups were evaluated.

## Accuracy results

CRPS measures the entire predictive score distribution; Brier score measures matchup probability error. Lower is better. Positive changes below mean deterioration. Aggregate CRPS uses player weights; aggregate Brier uses pair weights.

| Controlled change | Relative CRPS change | Relative matchup Brier change |
|---|---:|---:|
| Calendar correction alone | +0.0328% | +0.0258% |
| Rebuilt normalization, including dependent means and variance | +0.0137% | +0.0187% |
| Normalization: variance/skew changes with means fixed | −0.00028% | −0.00016% |
| Normalization: mean changes with variance/skew fixed | +0.0140% | +0.0188% |
| Calendar correction plus full rebuild versus original | +0.0465% | +0.0446% |

The normalization-only changes by edition are:

| Edition | CRPS before | CRPS rebuilt | Relative CRPS change | Relative Brier change |
|---|---:|---:|---:|---:|
| 2023 | 1.607540 | 1.607858 | +0.0198% | +0.0184% |
| 2024 | 1.428102 | 1.428219 | +0.0082% | +0.0201% |
| 2026 | 1.488621 | 1.488732 | +0.0075% | +0.0186% |

The full normalization rebuild slightly worsens CRPS, matchup Brier and log loss in all three editions. The equal-edition CRPS change is +0.000182; dropping any one edition leaves a positive change between +0.000114 and +0.000217. These are tiny observed effects, not evidence of a practically meaningful forecasting disadvantage. The variance-only point estimates are especially small and should not be promoted as an improvement.

Central-90% predictive coverage is unchanged at **89.55% overall**. Error/variance ratios barely move: approximately 1.04, 0.85 and 0.89 across the three editions. These limited samples do not justify tuning course multipliers to those realized ratios.

Calendar correction has mixed score effects: slightly worse CRPS in 2023 and slightly better in 2024/2026. Calendar correctness stands on the metadata evidence independently of these performance results.

## Why the effect is small

The 95th-percentile absolute change in field-relative skill is only 0.0141, 0.0089 and 0.0086 strokes. The corresponding changes in the sum of category variance inputs are 0.0184%, 0.0093% and 0.0125%. All course SD multipliers remain identical at their stored two-decimal precision.

The revised historical normalization therefore produces little change in the forecasts actually being tested. The small deterioration tracks the mean-only arm almost exactly. This provides no evidence that this particular normalization layer explains the season's high-confidence fragility or downswing; other model/data changes were not isolated here.

## Fidelity and statistical limits

- The current production Python skill oracle matches all 1,506 available numeric cells for the 126 modeled 2023 players exactly, including missing-value behavior. Earlier minimum-history checks also pass. The bundled legacy skill wheel was not used to calculate forecasts.
- Each calibration fit matches the production WLS fitting function exactly; an independent normal-equation calculation agrees too. All training-window and source-event completion guards pass.
- Repeating the full earliest normalization rebuild after corrupting its target and all later events leaves the historical inputs, skill estimates and fitted model exactly unchanged.
- Regenerated inputs pass category-mean accounting and mean/variance isolation checks. Each edition's round kernel agrees with the Python reference within the registered statistical gates, with exact category means and repeatable fixed-seed draws.
- Independent CRPS integration, matchup probability enumeration, PMF/CDF recounts, final skew moment guards and output reconciliation pass. All registered two-seed metric precision gates pass. Numerical repeatability does not establish population significance with only three editions.

The contract, addendum, normalized history snapshots, fitted models, frozen input arrays, PMFs, per-player results, source snapshots and verification receipts are retained in `permanent_data/research/riviera_normalization_rebuild_20260909/`. Original data and production settings are unchanged.

## Research implication

Keep the repaired chronology and the verified rebuild as the foundation for further controlled tests. This pilot offers little reason to spend additional performance-tuning effort on the cut-uplift normalization itself. The next substantive question remains whether player-specific course exposure or persistent weekly uncertainty explains spread and tail errors across more editions. Those mechanisms need their own frozen comparisons; this pilot neither validates them nor resolves historical production-vintage limitations.

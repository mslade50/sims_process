# Base Skill Market Rebuild — Frozen Research Contract

Date frozen: 2026-08-22  
Status: research only; no production consumer or weekly-pipeline file may be changed

## Question

Can a point-in-time neutral golfer-skill model built from adjusted total strokes gained,
orthogonalized adjusted category strokes gained, and independent IMG Arena shot-level
accounting data outperform the current-code legacy core in real betting markets? Can
weather-cleaning prior observations improve the base model and the R2-R4 cascade?

The test is explicitly market-first. Round/event RMSE and R-squared are diagnostics,
not promotion criteria.

## Claims this study may and may not make

The benchmark is a walk-forward reconstruction of the current code's legacy core. It
is not a bit-for-bit replay of historical production: historical weekly configs,
manual overrides, the contemporaneous DataGolf skill blend, and most as-of weather
forecasts were not archived.

The materialized IMG `predictive_skill_pga_validated_v2` transform used evidence from
2020-2024 and 2026. It is retrospective-only. The clean primary shot arm uses raw IMG
accounting strokes gained, with transformations and shrinkage learned strictly inside
the training window.

Closing prices are outcomes/evaluation data. They may not be model inputs. A separate
opening-price augmentation may be evaluated against same-book closing prices.

## Chronology and sealed samples

- Warm-up history: all valid rounds before 2021-01-01.
- Discovery/market selection: 2021-01-01 through 2022-12-31.
- Sealed money test: 2023-01-01 through 2026-08-22.
- The old rolling-three-year window, 2023-08-22 through 2026-08-22, is reported as a
  directly comparable slice of the sealed test.
- Events with the same start date are one batch. Every feature requires
  `source_round_date < event_start`; fitted coefficients and nuisance parameters must
  come only from earlier event batches.
- Candidate families, hyperparameters, transforms, gates, and market scoring rules are
  frozen using discovery data before inspecting sealed P&L.
- Any post-unseal model change is labeled exploratory and does not replace the sealed
  result.

This split permits published-principles criticism and revision on discovery data, then
preserves a genuine multi-year market test.

## Canonical observations

Player-event snapshots are keyed by:

`tour, year, event_id, event_start, dg_id`

Player-round inputs retain the physical course, round number, exact round date, raw
and adjusted outcomes, source coverage flags, and the timestamp boundary used for the
snapshot.

Primary training label: the player's mean field-adjusted SG in R1-R2 of the next
event. This limits made-cut survivor bias and avoids the current post-cut uplift model,
which was fitted retrospectively and depends on legacy skill.

Secondary labels:

- full-event mean with a separately walk-forward survivor-field correction;
- raw SG with an event-field latent intercept;
- next-round SG for the live cascade;
- category outcomes for multitask/shrinkage diagnostics.

## Weather treatment

Adjusted SG is field-strength adjusted, not weather adjusted. For positive-is-better
SG, historical observations are neutralized as:

`SG_neutral = SG_field + beta_w * (wind - course_round_mean_wind) + beta_d * (dew - course_round_mean_dew)`

The nuisance coefficients are fitted within each chronological training fold, mainly
from R1-R2. Existing `+0.155 SG/mph` wind and `-0.026 SG/degree F` dew estimates are
priors/checks, not full-sample constants silently backfilled into the headline test.

For categories, use partially pooled category-specific effects constrained to sum to
the total correction. Do not use the simulator's 35/35/15/15 weather allocation.
Center within physical course for multi-course events. Outcome-selected historical
weather exclusions may not be imported into the clean arm.

Weather arms:

- W0: field-adjusted observations, no weather cleaning.
- W1: rebuild every lag/EMA/category/shot aggregate from weather-neutralized history.
- W2: before R2-R4 pricing, neutralize each completed event round with realized
  weather, which is then legitimately known.
- W3: upcoming-round realized weather, reported only as an oracle/explanatory ceiling.

No three-year target-round forecast P&L claim is allowed because historical forecast
snapshots do not exist. The small set of preserved 2026 production forecasts may be
reported separately.

## Base-skill candidates

All neutral-base candidates exclude course fit, course history, majors, links, and
market data.

1. Current-code legacy reconstruction.
2. Adjusted-total multi-horizon state: half-lives/windows roughly 10, 20, 40, 80, and
   160 rounds plus 60/180/365/730 calendar-day summaries.
3. Total plus orthogonal category profiles, where
   `category_residual_k = category_adj_k - field_share_k * total_adj`.
4. Raw IMG accounting SG histories.
5. Adjusted total plus raw IMG accounting histories.
6. Total plus categories plus raw IMG accounting histories.
7. The same families with W1 weather cleaning.
8. Retrospective-only ablations using materialized IMG validated-v2 signals.

Every family includes effective sample size, source coverage/missingness, days since
last event, age/age-squared where available, and explicit low-data/tour-transition
handling.

Transparent regularized linear models are the headline candidates: ridge, elastic net,
and robust/Huber variants with empirical-Bayes low-data shrinkage. A hierarchical
dynamic/Kalman state model is a prespecified challenger. Flexible trees/boosting may be
screened only as a labeled sensitivity and must satisfy the same sealed protocol.

Adjusted total and adjusted categories are not independent measurements: field
strength is allocated across categories in fixed shares that sum back to total.
Therefore only orthogonalized category deviations may accompany total SG.

## Preserving the rest of the tournament model

The first isolation arm replaces only the established-PGA `pred_step2` neutral core.
All frozen downstream terms are retained:

`pred = new_base + comp_relative_dsle + comp_c_exp + comp_c_adj + comp_c_hist + comp_major + comp_links`

The second arm refits downstream course/history coefficients, chronologically, against
the new base residuals. It is separate because those coefficients were originally fit
against legacy residuals.

Own-model-only comparisons are primary. Any DataGolf blending is applied identically
after both arms and reported separately. `mkt_regress.py` is disabled for closing-price
tests.

## Cascade candidates and round markets

The direct round-matchup benchmark reconstructs the deterministic
`compute_residuals -> apply_adjustments -> compute_totals` flow and then uses the same
single-round simulation. Only completed target-event rounds enter the next-round state.

- C0: current-code cascade benchmark.
- C1: no live update.
- C2: stage-specific scalar shrinkage of C0.
- C3: strict-prior continuous/partially pooled state update using base skill, position,
  completed-round residuals, category evidence, and measurement reliability.
- C4: C3 with W2 completed-round weather normalization.

R1 uses the frozen pre-event base. R2-R4 use that same pre-event base plus only the
information available after earlier rounds. Current hard buckets and late-round OTT
effects are allowed to shrink to zero if discovery markets support it.

## Odds and execution contract

Books: BetOnline, Betcris, and Pinnacle only.

Markets:

- 72-hole two-way matchups;
- R1, R2, R3, and R4 two-way matchups, separately and pooled;
- win/top-5/top-10/top-20 as secondary tournament-simulation confirmation;
- three-balls only as a separately labeled sensitivity.

Closing panel: latest quote per book, then best price across the three books for an
identical logical wager. Fixed-book results are also reported. Best-of-three close is
not proof of simultaneous fillability, limits, or liquidity.

Opening panel: earliest quote per book, then best price. The selected opening book is
paired to its own close for CLV. Asynchronous openers are a sensitivity, not guaranteed
execution.

At most one directional model wager is permitted per canonical matchup. A true
cross-book both-sides arbitrage is reported separately, not credited as model alpha.
Pushes and dead heats use the audited book rules. Pricing and settlement must use
the same rule: Pinnacle top-5/top-10/top-20 uses the native no-dead-heat/full-tie
qualification probability (`top_nodh`), while BetOnline and Betcris use the native
expected standard dead-heat paid share (`top_dh`). Win keeps split-first-place
dead-heat expectation and matchup win/push probabilities are unchanged. Raw-archive
and universal-standard settlement scenarios are outcome-only sensitivities and do
not redefine the primary executable-book pricing rule.

Prespecified portfolio gates/sizing:

- matchups: 3% model EV; flat 1u risk and production risk-to-win;
- finish markets: 5% model EV; flat 1u and capped quarter-Kelly;
- 0-3%, 3-5%, 5-10%, and over-10% EV buckets are always shown, including the
  no-bet counterfactual below the gate.

## Selection and scorecard

Candidate selection on discovery data is lexicographic:

1. improve event-equal out-of-sample matchup log loss versus the legacy model;
2. calibration intercept/slope and edge-bucket monotonicity must not deteriorate;
3. positive paired CLV and ROI guardrails at fixed rules;
4. lower worst-event/player concentration and acceptable drawdown;
5. complexity/reliability tie-break in favor of the simpler model.

Sealed reporting leads with actual P&L, ROI, stake, bet count, maximum drawdown, and
paired delta versus legacy. It also includes conditional-on-no-push log loss, Brier,
calibration, de-vigged sharp-close comparison, CLV, turnover, and overlap/new/dropped
ticket attribution.

Inference resamples whole canonical events, including zero-bet events, with at least
10,000 paired bootstrap draws. A 2-4 consecutive-event block sensitivity and a
max-statistic/reality-check correction address week clustering and exhaustive candidate
search. Results are broken out by year, book, round, skill bucket, skill gap, weather
severity, edge bucket, withdrawal status, coverage, and player/event concentration.

## Compute ladder

- Analytic/proper-score screen for all candidate families.
- 2,000-5,000 common-random-number simulations for broad screening.
- 10,000 simulations for the Pareto frontier.
- 50,000 multi-seed paired simulations only for frozen legacy plus one to three
  finalists.

Immutable event inputs, seeds, market menus, and checkpoints are hashed. A candidate
cannot gain or lose eligibility because its field/market menu differs from the legacy
arm.

## Published-principles critique before unsealing

The design is reviewed against Mark Broadie's published strokes-gained framework:
correct shot-value attribution, category-specific persistence, and separation of long-
game signal from short-game noise. It is also reviewed against Rufus Peabody's publicly
described betting principles: sharp de-vigged prices, simulation, closing-line value,
calibration, quantified uncertainty, and skepticism of fragile narratives.

This is a critique using their published work, not a claim that either person personally
reviewed or endorsed the model.

## Promotion rule

This study cannot change production. A research candidate earns a later shadow-test
recommendation only if the sealed market result is directionally better across proper
scores and money outcomes, uncertainty is not dominated by one or two events, round
effects are coherent, leakage audits pass, and the advantage survives fixed-book,
common-opportunity, and reasonable execution sensitivities.

## Pre-scoring integrity amendment — 2026-08-22

This amendment was added before any sealed model score or P&L was accepted. It narrows
the admissible primary features; it does not use betting results to change a candidate.

- The current IMG accounting generation is not a clean primary input. Its v4
  surface-code-to-lie map was selected on 2026-07-28 using the full archived PGA shot
  sample, including the sealed test era. APP and ARG shot values can also depend on an
  event-fitted putting endpoint. Existing IMG accounting total/category columns are
  therefore retrospective diagnostics until they are recomputed with a pre-2021 or
  purely semantic frozen lie map and a pre-2021 frozen putting curve. The materialized
  IMG validated-v2 signal remains retrospective-only for the separate reason already
  stated above.
- The clean first-stage screen is consequently DG-only. A clean stroke-level rebuild
  must be completed before any finalist may claim to combine adjusted total,
  categories, and independent stroke-level SG.
- The canonical database has substantial field-adjusted total-SG history outside the
  PGA (Euro, KFT, and LIV). A PGA-only screen is incomplete for rookies and tour
  transitions. The revision set must test strict-prior cross-tour total histories with
  tour/source reliability indicators; PGA category and clean IMG features may be
  layered on when available.
- DB `sg_*_adj` does not correct the made-cut survivor-strength shift in R3-R4. A clean
  revision must either use prior R1-R2 history only or estimate a post-cut nuisance
  correction strictly inside each training fold.
- The primary R1-R2 target requires two observed rounds. One-round WD/DQ targets are a
  separately labeled robustness/injury-state sensitivity or receive an explicit
  exposure weight; they are not silently treated as equally reliable two-round means.
- Offered matchup players who did not start the event must still receive a strict-prior
  snapshot where possible. They may not disappear through an inner join or a global
  all-candidate common-menu restriction. Candidate-versus-baseline common menus and
  missing/fallback reasons are audited explicitly.

## Pre-scoring candidate-completeness amendment — 2026-08-22

This amendment was also frozen before accepting or using a sealed score. The first
DG-only ridge/Huber run is a plumbing screen, not the finalist search.

- The clean revision set must include: regularized total-only and total-plus-category
  models; a robust-loss model; an elastic-net sparsity sensitivity; and a transparent
  empirical-Bayes/dynamic-state candidate whose recency and low-sample shrinkage are
  chosen only on discovery markets.
- Calendar-year refits are a conservative audit arm. At least one quarterly or
  event-batch strict-prior refit must be tested so a model is not rejected merely
  because its coefficients were artificially stale. Events sharing a start date stay
  in one batch.
- Cross-tour adjusted-total histories must distinguish PGA, Euro/DPWT, KFT, and LIV
  observations and learn any tour translation from earlier cross-tour transitions
  only. A zero-history field/tour prior and an explicit fallback reason are required
  for every offered golfer.
- The primary clean shot candidate uses a surface map and putting curve frozen from
  pre-discovery data, then aggregates only shots from completed prior rounds. A
  full-archive IMG arm may be shown only as a retrospective ceiling and cannot win
  finalist status.
- Candidate-family and weather-arm selection uses discovery event-equal matchup log
  loss under the lexicographic scorecard. Sealed P&L may evaluate that frozen choice
  but may not choose among families.
- An opening-price blend is a separate market-augmentation sensitivity. Its blend
  weight is selected on discovery opening boards, its skill component remains
  close-free, and it is evaluated against same-book closes as well as raw realized
  settlement. It is never relabeled as the neutral base-skill model.
- W1 must distinguish feature-only cleaning from full neutral-target training. In the
  latter, the post-event R1-R2 training label is rebuilt as adjusted SG plus the
  already-frozen weather correction for those completed target rounds. Realized
  target weather may form a historical label after completion, but it may never enter
  that event's prediction features. Candidate evaluation remains against actual
  market outcomes.

## Protocol-deviation log — 2026-08-23

During an integrity check of the newly frozen clean IMG table, the primary analyst ran
one coarse ridge diagnostic that displayed 2023-2026 R1-R2 RMSE for DG-only features
and DG-plus-clean-IMG features. It was not a market backtest, did not use closing prices,
and did not produce or select a finalist. Nevertheless, the sealed outcome period was
viewed, so it is not described as perfectly analyst-blind from this point forward.

The candidate families, clean-IMG primary/sensitivity definitions, weather arms,
discovery years, scorecard, and compute ladder above were all frozen before that check
and may not be altered in response to it. Family/hyperparameter selection remains a
mechanical 2021-2022 discovery-market operation. The 2023-2026 results are reported as
locked out-of-sample evaluation under the frozen rules, with this deviation disclosed;
they are not used to revise the candidate set or choose among finalists.

After the DG-only plumbing run had also displayed its sealed-period scores, a
contract-completeness audit found that the implementation catalog did not yet contain
an IMG-primary arm, did not apply W1 to every clean source family, and had no
covariate-only robust-W1 arm despite those tests being required above. Those missing
arms were added before any complete-candidate score was computed. Ridge alpha 1 was
also added because the already-designated 2021-2022 discovery screen selected it for
the DG total families; the sealed DG-only result was not used for that choice. The
resulting clean candidate catalog was frozen before complete-candidate scoring. The
final catalog contains 92 candidates and has SHA-256
`3356629482d178f71029ce66f84c15660212e2fa1457ca9c519c8377dd770f1f`.
This final hash supersedes the provisional pre-benchmark catalog hashes. The
intervening changes (a) made the prespecified elastic-net sensitivity's deterministic
SGD solver and convergence settings explicit after the coordinate-descent benchmark
proved computationally infeasible and (b) implemented the already-required Broadie
category-shrinkage sensitivity with exact 2x and 4x ARG/PUTT ridge penalties. A first
scaling draft was corrected from `1 / multiplier` to `1 / sqrt(multiplier)` before any
candidate fit was scored, so the named penalty multipliers are mathematically exact.
No complete-candidate outcome or market score had been run.
No later sealed result may change that catalog. This timing means the complete study is
locked out-of-sample under a pre-scoring catalog, but not perfectly analyst-blind to the
earlier, incomplete DG-only sealed screen.

Before the complete catalog was scored, an independent implementation audit rejected
the first full-build attempt (`model_completion_v1`). The process had been launched
before later downstream-interface edits to its source file and did not preserve the
executed source bytes, command line, runtime fingerprint, or start-of-run input hashes.
That attempt was intentionally stopped; it produced no candidate prediction panel or
completion manifest and is not an admissible result. The replacement build must
preserve its exact launch source, hash every actual input before loading, verify source
and inputs are unchanged at exit, record the runtime/CLI, and require all 92 frozen
challengers to emit the identical requested player-event key set.

The same pre-score audit found two implementation errors in feature materialization.
First, clean-IMG history counted all prior IMG rounds but divided by prior PGA R1-R2
rounds, allowing nominal coverage above 100% and overstating the IMG-primary blend for
some players. The replacement uses all prior PGA rounds since the IMG start as the
matching denominator and fails if coverage leaves [0, 1]. Second, W1 candidates still
received raw PGA cross-tour SG histories in addition to their weather-neutralized PGA
histories. The replacement excludes raw PGA performance signals from W1/W1-robust
models and makes the IMG-primary fallback weather-arm aware. Raw Euro/KFT/LIV history
remains an explicitly labeled exception because comparable historical weather is not
available for those tours. These fixes implement the already-frozen W1 and coverage
definitions; they do not add a candidate or use an outcome/market score. The frozen
92-candidate catalog and its SHA-256 therefore remain unchanged.

The audit also found that clean-IMG rows used canonical odds event keys while the DG
round panel used local research-panel event keys. Joining weather on the incompatible
strings made every clean-IMG weather exposure missing and silently collapsed clean-IMG
W1 to W0. The replacement joins on exact year, event ID, player, round date/number,
and physical course number, asserts uniqueness and coverage, and refuses unexplained
missing weather. This identity correction was likewise made before any complete
candidate score was computed.

### Pre-score numerical solver amendment — 2026-08-23

The provenance-verifiable v2 build reached candidate 63 of 92 and failed on its first
elastic-net arm, `complete__elastic_sgd__total__w0__quarterly__a0.001_l10.25`,
before it wrote a final prediction panel, completion manifest, market score, or sealed
evaluation. Its exact executed builder SHA-256 was
`2be3785b0469d30bf3ff35f80fbdc8bd82f5205c235ada61284421f69ecbfaa3`, and its
failed output directory is
`permanent_data/research/base_skill_rebuild_20200101_20260822/model_completion_v2_provenance`.
A separate numerical-only
chronological preflight then found that the frozen averaged-SGD implementation of the
four elastic-net sensitivities was not admissible: eta 0.01 produced weighted MSE as
large as `3.27e15`, coefficients as large as `3.29e8`, and exhausted its 3,000-epoch
budget; eta 0.003 could report convergence despite weighted MSE `5.2e19` and
coefficients `9.46e8`. No market price, betting result, discovery ranking, or sealed
outcome score was inspected to make this amendment.

The same four prespecified elastic-net sensitivities retain their sources, alpha
values (`0.001`, `0.01`), L1 ratio (`0.25`), quarterly schedule, and selection status,
but now use deterministic cyclic `sklearn.linear_model.ElasticNet` coordinate descent
with `precompute=True`, `tol=1e-3`, and `max_iter=50,000`. It fails closed on any
`ConvergenceWarning`, budget exhaustion, non-finite result, training weighted MSE over
1,000, or absolute fitted coefficient/intercept over 100. In the numerical preflight,
all 108 chronological fits converged: maximum 33,201 iterations, weighted MSE
4.2494–4.4374, maximum absolute coefficient 1.3901, and approximately 27 seconds of
solver time (154.55 seconds including preprocessing).

This is a numerical-stability correction, not a response to predictive or market
performance. Candidate count remains 92, but the four family/name identifiers change
from `elastic_sgd` to `elastic_cd` and the catalog metadata records the amendment. The
amended catalog SHA-256 is
`dfcc7b25521d44f4b7c42a036b66de1914b28ccb8d805ebc3dc7fc4c3e88ef64`, which
supersedes `3356629482d178f71029ce66f84c15660212e2fa1457ca9c519c8377dd770f1f`
before any complete-candidate market scoring.

The remaining four deterministic Huber-IRLS and four empirical-Bayes/Kalman arms were
then run through a separate no-market numerical preflight using the provenance-audited
v2 augmented feature panel. All eight candidates emitted the exact 38,409 requested
player-event keys, with no duplicates, missing/extra keys, non-finite predictions, or
chronology violations; every Huber fold converged. Runtime was 1,392.847 seconds and
peak RSS was 2.83347 GiB. The artifact directory is
`permanent_data/research/base_skill_rebuild_20200101_20260822/model_completion_v3_nonridge_numerical_preflight`;
its manifest SHA-256 is
`5f2b2847fc704de397eda6c36ad5b39abc016d01a091406460dedd8ec06d395a`, its
prediction-panel SHA-256 is
`cf2d09b6ea36e620375927cc2c7b9aed40568a37b9a9c5ff1aecc078274ee97f`, and its
fit-log SHA-256 is
`28b273a4da558a99f2871e2292c710447c0915badbc22e9638a1e43abf334e4e`.

That preflight exposed a second numerical issue: converged low-alpha linear fits could
extrapolate far outside their strict-prior fitted support when sparse, correlated
recency/count/ESS features changed regimes. Across all 27 chronological folds, raw
total-feature Ridge alpha 1 reached [-21.04, 7.17] SG/round and Huber alpha 1 reached
[-22.29, 11.84]; the total-plus-category-plus-IMG versions reached [-35.72, 25.77]
and [-33.57, 24.08], respectively. The coefficients themselves were finite and
moderate; the failure was test-fold leverage, not solver divergence. Stronger alpha,
ordinary standardized-vector radial projection, and marginal train-quantile caps did
not reliably remove the tails.

Two strict-prior, no-market support diagnostics were persisted. The ordinary support
diagnostic source SHA-256 is
`95b6f21fe71bf0a6d26dc09f2d91d9bbf9c7b88676bb4ae61500253054de2901`, ran for
401.500 seconds, and produced manifest SHA-256
`2965a29453e4be1031cd9e0b1e1f3cc13156dd0752aa55a19f67d8d0bcda4ed2`.
The shrinkage-Mahalanobis diagnostic source SHA-256 is
`760160560e8f2891af988c91360f96922c7e27c13ce888da74bba5574b3a1d0a`, ran for
118.420 seconds, and produced manifest SHA-256
`ccf132d163e43900f7bf8769b6d268de35c93f19270e2c50d313798bd5411007`.
Both used Python 3.10.11 on Windows 10 and the same augmented-panel SHA-256
`47d4d7580599ba0452ef53a4893802ee69f5690677c3179c9dde5ef0adf44a96`.
Whitened q99.5 Mahalanobis projection bounded the tested tails below 8 SG/round, but
altered 7.7% of total rows and 17.6% of total-plus-category-plus-IMG rows; the per-PC
variant altered 78%-95%. It was rejected as too invasive.

The final pre-score amendment instead gives every regression candidate and refit fold
the same transparent extrapolation envelope. It computes the event-equal-weighted
median and weighted MAD of that fold's fitted training predictions, converts MAD to
robust sigma with 1.4826, and clips the raw test regression component to median plus or
minus six robust sigmas. It uses no test outcome, price, market score, or sealed result.
The raw component, guarded component, center, MAD/scale, bounds, clip flag/distance,
and fold diagnostics are retained, but the raw component is explicitly selection
ineligible. For IMG-primary candidates, the regression component is guarded before it
is blended with the explicit cross-tour fallback. Validation fails closed unless every
regression row has internally exact bounds/clip accounting, constant fold parameters,
and the correct guard-then-fallback blend. Dynamic candidates are unchanged.

Before the all-regression numerical preflight produced any summary result, the guard
was also given a fixed admission rule: it must clip no more than 1.0% of a candidate's
requested prediction rows overall and no more than 5.0% in any single refit fold. A
candidate exceeding either threshold is guard-dependent rather than merely protected
from rare extrapolation and may not remain selection eligible without a transparent
catalog amendment and a fresh numerical preflight. These thresholds cannot be revised
using discovery or sealed market performance.

During the numerical review, an ad hoc history query inadvertently printed 2023 R1-R2
realized rows for two already-identified dynamic extreme examples (DG IDs 2878 and
18631). No odds or aggregate scores were viewed, the dynamic candidates were explicitly
left unchanged, and mechanical discovery/sealed selection is unaffected. This further
limits analyst blindness and is disclosed rather than treated as model evidence.

The regression-envelope metadata is bound into every regression `CandidateSpec` and
the catalog-level runtime protocol. Candidate names/count remain fixed at 92. The final
pre-score catalog SHA-256 is
`d881c88e588be0a3ac969230c82175016e85e064a1862d0838c3e4c96c885cb4`, which
supersedes both the coordinate-descent-only hash
`dfcc7b25521d44f4b7c42a036b66de1914b28ccb8d805ebc3dc7fc4c3e88ef64` and the
original solver catalog hash
`3356629482d178f71029ce66f84c15660212e2fa1457ca9c519c8377dd770f1f` before any
complete-candidate market score is accepted.

Before the all-regression envelope preflight printed any candidate-level result, the
numerical admission rule was fixed as follows: the envelope is a rare extrapolation
guard, not a substantive model. Each candidate must clip at most 1.0% of its requested
rows overall and at most 5.0% inside every refit fold. A candidate that exceeds either
limit is guard-dependent and cannot remain selection-eligible without an explicit
catalog amendment and a fresh no-market preflight. The audit also reports every
candidate's maximum guarded-component and final-prediction range. These gates are
evaluated without prices or market outcomes.

### Pre-market evaluator interface amendment — 2026-08-23

Before the complete prediction panel was scored against any market, two independent
fail-closed interface audits corrected the discovery loader without changing a model,
candidate, gate, price rule, or scoring rule.

First, the full builder panel is now validated against the frozen 92-name catalog and
all three declared controls both before and after the physical 2020-2022 discovery
filter. This prevents a challenger that exists only in sealed rows from silently
disappearing before discovery selection. The post-filter name equality check runs
before menu validation or catalog/screen filtering.

Second, exact event/player-menu equality applies to the selection universe: all 92
frozen challengers plus the strict point-in-time current-architecture control. The two
selection-ineligible retrospective controls are not globally complete because their
historical source values were not archived for every player-event. They must be strict
subsets of the point-in-time reference menu, their missing counts and coverage ratios
are reported, and no missing row may be filled or used to shrink the challenger menu.
On the persisted no-market nonridge artifact, the exact selection menu contained
16,741 discovery player-event keys; the current-coefficient legacy-skill diagnostic
covered 16,056 (95.9082%) and the exact retrospective `pred_step2` diagnostic covered
15,900 (94.9764%), with zero imputation.

The independently audited discovery implementation is
`archive/evaluate_complete_base_skill_market_candidates.py`, SHA-256
`4520cf3866926c53a340687c6fd2456b80a28a03c6ce855d0f676c1fc771ecb5`.
Its 411-byte wrapper remains SHA-256
`92d298661c5f189703c960fcab54e2c7cda321e8d8bbc8351f3d615467374f47`;
the sealed evaluator remains SHA-256
`cbb0db44bf64e22b5f4bfd6f878e2b3eaa33fdfc25ca5a4bd3b4a8d430c3d8ae`.
At that audit stage, the loader also required the completed run's live-source
before/after hashes and preserved executed-source before/after hashes to equal builder
SHA-256 `e15f0e83b740a05ac3ee445b20f512d721a533054c2aaa6f6d14e231477e3dfe`;
all source/input unchanged flags must be true and `production_files_changed` must be
false. The combined discovery/sealed synthetic suite passed 44 of 44 tests after
these changes. The later numerical-admission amendment below supersedes that exact
builder/catalog binding before any discovery or sealed market score was run.

### Pre-market numerical-admission result — 2026-08-23

The completed all-regression, no-market envelope preflight ran all 88 regression
candidates across 2,336 strict-prior fits. It emitted exactly 38,409 player-event keys
per challenger, with no duplicates, missing/extra keys, chronology violations, or
non-finite predictions. Runtime was 3,221.859 seconds and peak RSS was 5.66173 GiB.
All 108 elastic coordinate-descent fits converged with zero warnings; maximum
iterations were 33,201, maximum dual gap was 0.0054333, training weighted MSE ranged
from 4.24943 to 4.43745, and the maximum absolute coefficient was 1.39009. Guarded
and final predictions remained in [-5.31541, 5.46414] SG/round.

The predeclared numerical-admission rule nevertheless failed, so the full model build
and every market evaluator remained blocked. Nine candidates exceeded the 1.0%
overall clip cap, 31 exceeded the 5.0% within-fold cap, and six appeared in both sets.
The strict OR-rule therefore disqualifies exactly 34 candidates. Three IMG-primary
alpha-100 arms are the easy-to-miss overall-only failures: W0 at 422/38,409
(1.09870%), W1 feature-only at 423/38,409 (1.10130%), and W1 neutral-target at
424/38,409 (1.10391%); their worst individual folds were below 3.03% but that does
not excuse the overall breach.

The thresholds will not be relaxed and the prediction envelope, alphas, features, and
prediction values will not be changed in response. The smallest admissible amendment
is catalog-only: retain all 34 candidates and both raw/guarded predictions as
diagnostics, but set `selection_eligible=false` and `sensitivity_only=true`. Together
with the already-ineligible literal-full IMG sensitivity, the 92-name catalog will
therefore contain 57 selectable candidates (53 regressions plus four dynamic models)
and 35 diagnostic-only candidates. The amended builder must fail closed unless those
exact 34 remain the only numerical-gate failures and every still-selectable regression
passes both caps. Its next fresh, raw-input, no-market 92-candidate build serves as the
required fresh numerical preflight; no discovery price or outcome may be read unless
that build completes and is independently audited.

The persisted preflight audit is
`permanent_data/research/base_skill_rebuild_20200101_20260822/model_completion_v3_regression_envelope_preflight/numerical_preflight_audit.json`,
SHA-256 `37edc6ab44975ae50eddf0faa1503ed01e820be35b529e087e1ff4e1a8dc7fac`.
Its candidate summary SHA-256 is
`5e532080f0614a4b892c5783e8ada918f11bbc99eb5a3b6269c6c836dba7b171` and its
fold summary SHA-256 is
`c1b074b12ec43d8591a99f95540bb98baaa369ce3e5a6e22953a0e3be4cefbbb`.
The underlying run manifest, fit log, prediction panel, preserved builder source, and
subset frozen catalog hashes are respectively
`1f94a02d5b926a83d825ea13ad569cde42086216d32cf91ee06360f51714c18e`,
`627c54a04f51a31c6bd4645dbaf5260f72fe5ee19a2d9bdb50a55ab614c52183`,
`b944e5fc76d72d40544a9e960fa39cbe67632679357e330459282bc7fb85ef87`,
`e15f0e83b740a05ac3ee445b20f512d721a533054c2aaa6f6d14e231477e3dfe`, and
`52547bbbd2793324b287f2d97aa785bc9ef6cbf92fe98a9817a7140d2dd9d4b5`.
No market data was read and no candidate was selected in producing this amendment.

The mechanically amended builder is
`archive/build_complete_base_skill_candidates.py`, SHA-256
`69d442b7c91a6191861842290fa69ecee437d35830510bcf0fd3edd9d28cc019`
(213,897 bytes). Its full 92-spec frozen catalog SHA-256 is
`d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23`.
The amended discovery loader now requires those exact hashes and treats
`selection_ineligibility_reason` as part of the frozen candidate specification. Its
SHA-256 is
`fc196e951f65c661ce1e3bf135486d126860ab98a28a97e3167f8726e7ecc6a2`
(134,380 bytes); the wrapper and sealed evaluator remain unchanged. The focused
builder suite passed 33 of 33 tests and the combined discovery/sealed suite passed 44
of 44 tests. The next full raw-input run still must reproduce and persist the exact
numerical-admission result before the discovery loader is permitted to score it.

### Pre-discovery implementation clarification — 2026-08-23

This clarification was recorded after the full no-market build completed but before
any discovery or sealed market score was produced. The first attempted discovery run
failed during input validation, before it loaded or scored a matchup, because its
warm-up guard incorrectly required an event on exactly 2020-01-01. No candidate rank,
proper score, bet, P&L, or sealed result was emitted or viewed.

The implemented W1 estimand is more conservative than the earlier wording above. The
weather nuisance coefficients are fit once using only pre-2021 completed rounds and
are then frozen for every 2021-2026 feature and historical target correction. They do
not expand or refit inside later chronological folds. Therefore this study tests a
`fixed-pre-2021 W1` arm, not a fold-expanding W1 arm. This prevents discovery or sealed
outcomes from altering the weather correction, but it may be less adaptive to later
weather/scoring regimes. Of 83,811 exactly joined clean-IMG player-rounds, 2,014
(2.40%) lack a usable weather delta; those observations receive a zero weather
correction rather than being deleted, so the IMG W1 arm is explicitly partial for
those rows.

The implemented pre-event downstream sensitivity also has narrower scope than the
earlier second-arm wording. It preserves the exact audited aggregate downstream term
and tests frozen scalar multipliers of 0, 0.5, and 1.0. It does not yet chronologically
refit the individual course-shape, course-experience, course-history, major, links, and
relative-distance coefficients against each new base core. Results from the scalar
arm may support the neutral-core isolation claim, but they may not be described as the
promised full downstream-coefficient refit. That unimplemented arm remains an explicit
limitation unless completed before the sealed sample is opened.

The completed raw build is frozen to manifest SHA-256
`53a6446d683d6a4709acd24cdf7843bf2db7393c38e7f09eb3cc803fa416e8e0`,
prediction-panel SHA-256
`c5ad98f85ce19a750922735c067a14cde0bd1b2d002e35df1103a8392c5aaae2`,
resolved candidate-catalog SHA-256
`1ab117a5f4aec78c2885f0db0d7ce6e60541177f7b288c61b10227e96c0feaae`,
and frozen-spec SHA-256
`d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23`.
An independent post-run audit reproduced all 92 challenger menus, all numerical gates,
and all chronology/feature invariants without reading market results.

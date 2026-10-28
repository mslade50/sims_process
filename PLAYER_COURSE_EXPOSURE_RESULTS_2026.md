# Player/course exposure variance pilot

Completed 2026-09-09. **Decision: this distance-and-lie adjustment does not produce a meaningful forecast improvement.** Course history predicts the approach situations players face, but personalizing that exposure adds little to the variance already captured by player shot dispersion and course multipliers. Keep the candidate in research. The next priority is uncertainty that persists across rounds.

**Scope correction, 2026-09-10:** the decision above applies only to this heavily shrunk Riviera candidate. It does not establish that distance and lie are unhelpful generally. This pilot held field-average APP variance fixed, so it did not test whether exposure improves the overall course variance estimate. The recommendation to move away from exposure research was premature; the user-directed next step is a frozen multi-edition study at Bay Hill, Muirfield Village, Renaissance Club, TPC Twin Cities, Sedgefield and Harbour Town, with separate course-level, player-level and shrinkage comparisons.

That expanded comparison is now complete: [six-course / 34-edition results](SIX_COURSE_EXPOSURE_RESULTS_2026.md). It finds a small consistent course-level distance/lie CRPS indication, little incremental matchup benefit from those terms, and mixed distribution calibration under full course-multiplier replacement. It supersedes this pilot's recommendation to move on from exposure research.

## What was tested

The comparison uses the completed, chronologically rebuilt Riviera inputs for 2023, 2024 and 2026: **268 R1 player-edition cases and 11,531 overlapping non-tied matchups**. These are three independent event editions, not 11,531 independent observations. The 2025 Genesis edition at Torrey Pines is correctly excluded from Riviera history.

Every candidate holds category means, skill, correlation, skew and non-APP variances fixed. It also preserves the field's mean APP variance, so it measures whether exposure improves the allocation of uncertainty between players beyond the common course multiplier. Existing course multipliers remain in place.

Prior shots are grouped into four distance bands (50–99, 100–149, 150–199 and 200+ yards), each split between clean tee/fairway lies and rough/sand/recovery lies. Other APP shots remain in a residual cell. Raw shot SG, including hole-outs, is used for variance. The mixture calculation includes both within-cell variation and differences between cell means.

Player profiles use up to 80 earlier complete rounds with EMA span 50. Sparse mixes and cell moments shrink toward earlier tour data. A player's predicted course mix combines the course's historical mix with the player's historical exposure tendencies. The primary candidate uses 200-shot cell shrinkage, with course-only mix and stronger 800-shot shrinkage as frozen comparisons. The exposure variance ratio is itself shrunk 50% in log space. These choices were fixed before scoring; none were tuned to the three results.

Only completed events ending before each forecast cutoff enter features. Prior physical Riviera editions contribute 1,136, 1,529 and 1,768 player-rounds to the respective course profiles. The corrected calendar bridge is used throughout.

## Exposure is predictable; player differentiation is small

The following errors compare frozen predictions with actual complete R1 shot exposure. Lower is better. Mix error is the sum of absolute differences between predicted and observed cell shares.

| Edition | Player rounds with exposure | Tour-average mix error | Prior Riviera mix error | Personalized Riviera mix error |
|---|---:|---:|---:|---:|
| 2023 | 125 | 0.49255 | 0.38774 | 0.38474 |
| 2024 | 70 | 0.51081 | 0.39894 | 0.39790 |
| 2026 | 72 | 0.46455 | 0.44402 | 0.43786 |

Giving editions equal weight, course history reduces mix error **16.2%** versus tour averages. Personalizing that course mix reduces error another **0.83%**. Both comparisons improve in all three editions, although this is a small, previously studied sample. Shot-mix cross-entropy agrees on direction.

Personalized expected shot counts are less convincing: their equal-edition mean squared error is **3.95% worse** than simply using the historical course-average count, with improvement only in 2024. This is a simple multiplicative count model, not a test of every possible count predictor.

A count-only course/tour multiplier is an exact null control here. The baseline already has player-specific expected shot counts; multiplying all players by the same course factor cancels when field APP variance is preserved. A useful new count feature would need to predict player/course interactions beyond that common factor.

The historical Riviera profile is distinctive: before the 2026 event, 48.4% of APP shots came from 150–199 yards, versus 35.8% tour-wide. That is useful course information, but much of it is shared across the field. After shrinkage and field normalization, the primary candidate's 95th-percentile absolute APP SD change is only **0.21–0.27%**, and its corresponding total round-score SD change is **0.067–0.077%** across the editions.

## Forecast accuracy

Each non-null arm uses 200,000 simulations per player with each of two fixed seeds, followed by the production round-skew calibration. Conditional CRPS measures the full round-score distribution after removing shared field scoring conditions symmetrically. Brier scores measure the unpriced, non-tied matchup board; lower is better for both.

| Edition | Baseline CRPS | Personalized exposure CRPS | Baseline Brier | Personalized exposure Brier |
|---|---:|---:|---:|---:|
| 2023 | 1.6078576 | 1.6078161 | 0.239116 | 0.239113 |
| 2024 | 1.4282190 | 1.4282316 | 0.241878 | 0.241883 |
| 2026 | 1.4887319 | 1.4887577 | 0.269427 | 0.269428 |

The primary candidate improves both metrics in 2023 and worsens them in 2024 and 2026. Equal-edition CRPS changes by **−0.00000104** (roughly −0.00007%); Brier changes by **+0.00000122** (roughly +0.00049%). These are negligible effects. The aggregate CRPS difference is smaller than observed paired-seed variation, so its sign should not be presented as an accuracy gain.

Player/pair weighting gives a slightly favorable aggregate because 2023 has the largest field. Leaving out an edition changes the sign. Course-only mix and stronger shrinkage give the same practical conclusion. Central-90% coverage remains **240/268, or 89.55%**. The error/variance ratios remain approximately 1.039, 0.849 and 0.891.

For matchups whose baseline favorite probability is at least 65%, Brier again improves in 2023 and worsens in 2024/2026; the equal-edition change is just +0.00000262. Across the full matchup board, the 95th-percentile absolute probability change is only **0.006–0.009 percentage points**. This adjustment therefore does very little to soften or improve the highest-confidence probabilities.

## Fidelity and interpretation

- The contract and all candidate inputs were frozen before scoring. Corrupting target/future shot moments and course flags leaves every feature exactly unchanged.
- Physical course identity and completed-event cutoffs pass. Complete-round signal exports contain 50 duplicate membership rows from different model versions in a future 2026 event; completion flags agree, and membership is used once. No signal values are used in this study.
- Baseline simulated probability distributions reproduce the preceding normalization study exactly for all six edition/seed combinations. The count-only control is also exactly identical.
- Candidate Rust draws pass independent Python-reference checks for all editions; maximum standardized CDF discrepancy is below 3.97 against the frozen limit of 5. Deterministic means and fixed-seed repeats pass.
- Independent probability, CRPS and mixture-variance calculations reconcile. Mean/variance isolation, final-skew moment guards and the registered Monte Carlo precision checks pass. Precision sufficient to rule out a material effect does not establish the sign of a near-zero effect.
- Source and input hashes remain unchanged. Code snapshots, contract, player inputs, exposure predictions, PMFs, metrics and receipts are retained in the evidence folder.

One 2023 player, Adrian Meronk, lacks the prior source-ID bridge and receives the explicit tour-prior fallback. His score remains in the 268-player comparison; his shot exposure is unavailable, leaving 267 exposure cases.

This is a **controlled fixed-method R1 pilot**, not a reconstruction of historically issued probabilities or a fresh holdout. The inherited external field-strength, shot-SG and final-skew calibration vintages are shared between arms. Three editions at one course cannot establish whether the same result holds everywhere. There are no archived-odds Kelly splits here, no finish-position scoring, and no causal attribution to the season drawdown.

## What to do next

Prioritize **persistent uncertainty across rounds**: estimate how much pre-event forecast error repeats for the same player through the week, after accounting for updated skill, common course/round conditions and selection into later rounds. Then distinguish ordinary round noise from uncertainty about skill and week-specific form, and test their separate effects on round matchups and finishes without double counting existing variance.

The exposure branch is not exhausted. Better player-specific count predictions, exact hole/hazard exposure, and sequences linking drives, approaches and recovery shots remain untested here. This pilot gives little reason to spend the next cycle tuning these distance/lie variance multipliers. A later expansion should test a stronger mechanism on several courses with frozen settings and more independent editions.

Evidence: [research folder](permanent_data/research/player_course_exposure_20260909/), [frozen contract](permanent_data/research/player_course_exposure_20260909/contract.json), [final verification](permanent_data/research/player_course_exposure_20260909/final_verification.json), [independent reconciliation](permanent_data/research/player_course_exposure_20260909/independent_verification.json). No production inputs or betting rules changed.

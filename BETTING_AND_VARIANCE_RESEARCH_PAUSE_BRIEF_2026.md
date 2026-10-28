# Betting and SG variance research — stopping-point brief

Updated September 10, 2026. **Status: paused at the user's request.** This consolidates the completed season review and subsequent distribution research. It supersedes older next-step recommendations in the living worklists; the linked reports retain the detailed methods and evidence.

**Main conclusion:** We have not established a model defect as the cause of the round-matchup downswing, nor validated a production correction from this research. Randomness remains a plausible leading explanation for the betting pattern. We did identify weaknesses in how player/category variance is differentiated, but improving an intermediate category score has not yet translated into reliably better final finish probabilities.

## 1. Season results and the 5% edge split

The review covered 643 first-run round-matchup signals across 29 events, selecting the best quote per side at Pinnacle, BetOnline or Betcris. These are logged signals, not a reconciliation of actual fills. “Edge” means Kelly theoretical EV, not the fraction of bankroll wagered. Early means through Memorial; later means afterward.

| Theoretical edge | Signals | Early profit | Later profit | Season profit / ROI |
|---|---:|---:|---:|---:|
| ≥5% | 427 | +43.51u | −9.58u | +33.93u / +6.99% |
| <5% | 216 | −7.76u | −2.10u | −9.86u / −3.84% |

The late downswing was concentrated in stronger-edge bets. Their approach surprise moved from +0.19 to −0.15 SG and putting from +0.16 to −0.08, without either residual distribution widening. Closing-line value remained positive, +0.82 to +0.91 percentage points. Early R3 results were exceptionally favorable; their disappearance does not by itself identify a broken model. Player-selection differences also confound the edge comparison.

**Decision:** Neither the early winning streak nor the later losses establish durable skill or a model failure. No new 5% entry gate, sizing rule or blanket variance increase was validated. [Edge-split report](<C:/Users/McKinley Slade/dev/sims_process/BETTING_EDGE_SPLIT_2026.md>).

## 2. June changes, APP errors and weather

We audited adjustment accounting and reconstructed archived pre-event/live inputs. The June R3 rollback hypothesis weakened: the change labeled an R3 category blend actually affects R4, and direct prior-adjustment removal affected only two of 14 matched post-change R3 bets. The original Python/Rust parity checks passed.

Expanded reconstruction recovered pre/live APP splits for 181 of 196 R3 bets, including all 144 under the category-first round model. Among the 98 with ≥5% edges, APP forecast error changed from +0.494 to −0.464 SG while mean live uplift barely changed, +0.080 to +0.079. This points more toward baseline errors, selection and realized outcomes than an increase in live APP optimism. Weather sensitivity was negligible, consistent with nearby matchup tee times.

**Decision:** No rollback or weather investigation is justified by these results. Historical runtime gaps prevent complete causal attribution. Some exceptional early profit also predates the March 17 category-first round-engine change. [June audit](<C:/Users/McKinley Slade/dev/sims_process/R3_JUNE_TRANSITION_AUDIT_2026.md>), [initial APP audit](<C:/Users/McKinley Slade/dev/sims_process/APP_FORECAST_AUDIT_2026.md>), [expanded reconstruction](<C:/Users/McKinley Slade/dev/sims_process/APP_EXPANDED_RECONSTRUCTION_2026.md>).

## 3. Archetypes, finish bets and model versions

Balanced APP underperformed in selected sharp-book finish bets too: 75 signals returned −3.64u versus model-expected +3.44u. Selected player-events missed their historical APP baseline by about 0.27 SG per round; Balanced APP outside that selection was almost neutral. Across all books, the group's finish results were approximately breakeven. This is a selection/calibration lead, not evidence that the entire archetype needs a penalty.

The broader screen covered 18 archetypes. Balanced Putter warranted attention; Elite Putter losses did not show the same mechanism. There was no blanket “highest probabilities are worst” result: round forecasts ≥65% won 71.9% of 32 decisive outcomes versus 69.7% predicted, with substantial uncertainty.

Dated replay reproduced 114 of 125 selected historical live totals. Updated accounting explained only 0.089 SG of a 1.503 SG mean miss in the ≥5% APP group. A matched finish-menu audit found selected 10–20% payout-share forecasts realized 8.15% against 13.67% expected, but clustered uncertainty included zero. Controlled shape changes moved some finish probabilities by 3–5 percentage points: evidence of sensitivity, not proof of mispricing.

**Decision:** July data repairs/refits and August distribution changes matter when interpreting pooled results. Missing pre-repair training lineage prevents a complete causal replay. No archetype penalty or rollback was validated. [Balanced APP finishes](<C:/Users/McKinley Slade/dev/sims_process/BALANCED_APP_FINISH_AUDIT_2026.md>), [archetype screen](<C:/Users/McKinley Slade/dev/sims_process/ARCHETYPE_PROBABILITY_FRAGILITY_2026.md>), [versioned replay](<C:/Users/McKinley Slade/dev/sims_process/VERSIONED_ARCHETYPE_REPLAY_2026.md>).

## 4. Historical reconstruction and course exposure

We established replay gates for physical course identity, information cutoffs, source hashes, engine parity and simulation precision. The preflight inventoried 200 course-edition rows across 82 course IDs. Calendar/event identity corrections and discovery of future-informed cut-uplift normalization required rebuilding past-only research inputs and their dependent player/course inputs.

The rebuilt Riviera comparison covered three editions and 268 R1 cases. It improved historical validity but did not improve forecast accuracy: CRPS, a distribution-accuracy score, worsened about 0.014% and matchup Brier about 0.019%. These tiny changes were mainly mean-related.

Distance/lie exposure testing then expanded from Riviera to the six requested courses—Bay Hill, Memorial, Renaissance, 3M, Wyndham and Harbour Town—and ultimately **152 editions, 36 venues and 18,201 R1 cases**. Predicting exposure more accurately did not yield a reliable forecast gain. The full replacement slightly worsened scores; a past-only blend was effectively unchanged. Results did not consistently generalize beyond the original courses.

**Decision:** Retain the existing exposure baseline. This conclusion comes from the expanded sample, not a single event. Exact hazard exposure and within-hole shot sequences remain untested possibilities; the simple distance/lie layer is not a demonstrated improvement. Renaissance 2022 lacks original measurements needed for faithful reconstruction.

[Replay protocol](<C:/Users/McKinley Slade/dev/sims_process/COURSE_REPLAY_ROBUSTNESS_PROTOCOL_2026.md>), [preflight](<C:/Users/McKinley Slade/dev/sims_process/COURSE_REPLAY_PREFLIGHT_2026.md>), [lineage findings](<C:/Users/McKinley Slade/dev/sims_process/RIVIERA_CALIBRATION_LINEAGE_2026.md>), [rebuild results](<C:/Users/McKinley Slade/dev/sims_process/RIVIERA_NORMALIZATION_REBUILD_RESULTS_2026.md>), [Riviera exposure](<C:/Users/McKinley Slade/dev/sims_process/PLAYER_COURSE_EXPOSURE_RESULTS_2026.md>), [six-course results](<C:/Users/McKinley Slade/dev/sims_process/SIX_COURSE_EXPOSURE_RESULTS_2026.md>), [all-course results](<C:/Users/McKinley Slade/dev/sims_process/ALL_COURSE_EXPOSURE_RESULTS_2026.md>).

## 5. Shot data, persistence and the category audit

The initial 34-event conditional-variance audit found the shot-level overlay improved distribution scores modestly and corrected some overly narrow/wide player estimates; matchup accuracy was essentially unchanged. The subsequent category audit covered 72,804 category observations across the 152 editions. Aggregate squared-error/predicted-variance ratios were close to one: **OTT 0.937, APP 1.000, ARG 0.989, PUTT 1.045**.

OTT player-specific variance had a reliable predictive benefit. APP/ARG/PUTT differentiation looked too strong in some groups, particularly low predicted ARG variance and high predicted APP variance. The shot overlay helped all four categories across annual panels. Incremental course-multiplier gains remained uncertain. Excluding nine editions containing unresolved category/total-SG discrepancies preserved the main findings.

A separate 143-edition persistence diagnostic found repeated forecast errors, but did not establish a reliable incremental scalar correction beyond existing updating and latent uncertainty. Selecting only four-round completers radically changed apparent dependence, demonstrating the danger of cut-selection bias.

**Decision:** Preserve shot-level information and OTT personalization. Generic latent/persistence tuning is parked at the user's direction. [Conditional audit](<C:/Users/McKinley Slade/dev/sims_process/CONDITIONAL_VARIANCE_AUDIT_2026.md>), [persistence measurement](<C:/Users/McKinley Slade/dev/sims_process/PERSISTENT_UNCERTAINTY_RESULTS_2026.md>), [ordered category audit](<C:/Users/McKinley Slade/dev/sims_process/SG_CATEGORY_VARIANCE_RESULTS_2026.md>).

## 6. Four-round shapes and the good-golfer subset

We tested cumulative category distributions with current in-event updating retained, using 16 no-cut editions and four cut editions. Controlled comparisons moved APP variance, or APP and ARG variance together, halfway toward the field average. Initial means and other distribution settings were held fixed; the amount was not fitted to outcomes.

The follow-up restricted evaluation to **pre-event baseline total SG >0.5 and either >40 PGA rounds in the trailing 12 calendar months or ≥20 in six months**, all completed earlier. Strictly >20 in six months produced identical membership. Full simulation fields remained intact. The subset contained 648 no-cut outcomes and 179 cut-event starters, of whom 134 completed four rounds.

The no-cut findings survived this filter: low/high ARG variance groups had error/variance ratios of 1.462/0.818; high-variance APP was 0.681. High-variance APP golfers reached their modeled extreme good-week 5% boundary only about 0.3% of the time. **However, the cut-event sample contradicted a universal APP correction:** its high-variance APP ratio was 1.068 and extreme-good-week exceedance 10.9%; APP shrinkage worsened category CRPS 0.138%.

PUTT's qualified four-round ratio of 1.114 was already 1.123 in R1 on the same sample, so this does not identify an accumulation/momentum problem.

**Decision:** The patterns are not simply poor golfers or sparse history. They also do not justify universal stronger regression or a combined APP/ARG adjustment. [Four-round test](<C:/Users/McKinley Slade/dev/sims_process/FOUR_ROUND_CATEGORY_SHAPES_RESULTS_2026.md>), [qualified subset](<C:/Users/McKinley Slade/dev/sims_process/GOOD_GOLFERS_CATEGORY_SHAPES_RESULTS_2026.md>).

## 7. Latest completed test: ARG mechanism and final finishes

ARG-only shrinkage was isolated and tested through the final score adjustment and production finish aggregation. All 827 qualifying starters were scored, including 45 missed cuts.

For 149 low-variance ARG no-cut cases, modeled cumulative variance was 4.052 SG² against squared forecast error of 5.924. About 39% of the point-estimate gap came from individual-round errors and 61% from cross-round error products. Those products can reflect persistent baseline/course errors as well as event state; they are not proof of momentum. The R4 ratio fell from 2.01 to 1.28 when the four-player Albany subgroup was excluded.

| ARG-only challenger versus baseline; lower is better | No-cut | Cut events |
|---|---:|---:|
| ARG category CRPS | −0.125% | −0.041% |
| Final top-5/10/20 Brier | +0.008% | +0.023% |
| Dead-heat payout-share squared error | +0.007% | +0.028% |

The no-cut category gain was supported, but **final finish forecasts did not reliably improve**. Both seeds gave slightly worse finish point estimates; uncertainty included zero. Kernel parity, decomposition identities, archived baseline reproduction, ties and missed-cut scoring passed the documented checks.

**Decision: do not promote ARG shrinkage.** A category-score gain alone is insufficient. [Mechanism, finish results and archived proof](<C:/Users/McKinley Slade/dev/sims_process/ARG_MECHANISM_AND_FINISH_RESULTS_2026.md>).

## Stopping point and eventual restart

Research tools, rebuilt historical inputs and detailed reports are preserved. This research has not promoted a production setting, betting rule or rollback. Historical reconstructions generally test a fixed method on past inputs; they are not automatically exact replays of issued prices or historically available upstream data. Error/variance ratios include mean errors, and the latest full-event evidence is concentrated in relatively few courses. No-cut and cut-conditional findings must remain separate.

**Priority correction:** Another narrow ARG investigation has not been established as the best use of time. The next available diagnostic should not automatically become the next project. Earlier ARG-first recommendations are superseded; generic persistence tuning also remains parked.

If this work is resumed, the proposed starting point is to compare the value of a broader **full-event distribution and final-finish calibration review across all four SG categories**, using the qualified-golfer subset and separating model versions where archives permit. It should distinguish central spread, both tails and cross-category dependence, then require controlled changes to improve final forecast scores on held-out events. That review is **proposed, not completed, and not yet proven to be the highest-return research project**. No further analysis is underway as part of this pause.

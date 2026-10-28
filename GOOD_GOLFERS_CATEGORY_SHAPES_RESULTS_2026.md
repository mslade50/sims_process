# Four-round category shapes: good golfers with substantial recent samples

Completed 2026-09-10. **The no-cut findings persist after filtering to better golfers with substantial recent histories. ARG's variance-allocation problem and high-variance APP's excessive spread are descriptively stronger. However, APP shrinkage worsens forecasts in the qualifying cut-event sample, so this does not support a combined APP/ARG adjustment across all events.**

## Definition and fidelity

A player qualifies before an event when:

`baseline total SG > 0.5 AND (trailing 12-month rounds > 40 OR trailing 6-month rounds >= 20)`

Baseline SG is the frozen pre-event skill used by the tournament simulator, not the player's realized event performance. Month windows are calendar months measured backward from event start. Only rounds from events completed before that start count. Primary counts use PGA TOUR history feeding the category model. Counting complete category-data rounds or other available tours produces the same directional conclusions. Interpreting the six-month threshold as strictly greater than 20 changes no qualifying players in this sample.

The filter retains:

| Cohort | Qualifying starters | Four-round outcomes scored | Distinct players scored | Editions / courses |
|---|---:|---:|---:|---:|
| No cut | 648 of 932 | 648 | 120 | 16 / 7 |
| Cut events | 179 of 386 | 134 | 71 | 4 / 4 |

The original full-field simulated distributions were reused exactly. The filter affects evaluation only: it does not remove opponents, alter the cut, recenter SG on the selected players, or change the reference mean for variance shrinkage. Actual cut-event completers are still evaluated against their own simulated made-cut distributions. The original two 50,000-draw seeds and current updating rules remain unchanged.

Primary variance groups retain the original pre-event full-field quintiles, making the comparison directly interpretable. A sensitivity also reranks variance among qualified **starters**, before observing who makes the cut.

## Do the original effects hold?

The table compares the original population with the qualified subset in **no-cut events**. Ratios are average squared forecast error divided by average predicted variance. One is the reference; values include mean-estimation error and are not automatically pure variance estimates. Estimates give equal weight to courses, then editions, then players within editions.

| Diagnostic | Original population | Qualified subset | Qualified cases |
|---|---:|---:|---:|
| Lowest-variance ARG quintile | 1.424 | **1.462** | 149 |
| Highest-variance ARG quintile | 0.898 | **0.818** | 102 |
| Highest-variance APP quintile | 0.842 | **0.681** | 99 |
| PUTT, all players | 1.140 | **1.114** | 648 |

**ARG:** the model still treats the low-variance group as too predictable and the high-variance group as too volatile. Descriptive 95% course intervals are 1.126–1.843 and 0.692–0.930, respectively. Reranking only within the qualified population gives similar ratios, **1.454 and 0.878**. This is not simply an artifact of comparing good golfers with weak or inexperienced players.

The fixed APP/ARG half-shrink candidate moves the original low/high ARG groups to **1.339 and 0.878**, improving calibration but leaving a residual problem in the low group. The low group's average actual-minus-forecast ARG is +0.15 SG over four rounds; the high group's is +0.04. Those group means should not be mistaken for the entire variance shortfall.

**APP:** qualified players in the highest-variance quintile have substantially less cumulative error than predicted spread. The ratio is **0.701 before 2026 and 0.619 in 2026**, although the current-year panel has only 27 cases. Reranking among qualified starters still gives a low ratio, **0.785**. The earlier finding no longer disappears in the small 2026 no-cut subset after this quality/sample filter.

## APP upside: the extreme tail still differs from an ordinary good week

For the 99 qualifying no-cut player-event cases in the original highest APP-variance quintile:

| Tail | Expected frequency | Observed, course-balanced |
|---|---:|---:|
| Good-week 10% boundary | 10% | **9.7%** |
| Extreme good-week 5% boundary | 5% | **0.3%** |
| Bad-week 10% boundary | 10% | 5.6% |
| Extreme bad-week 5% boundary | 5% | 2.9% |

Only **zero or one of the 99 outcomes** crosses the model's 95th-percentile threshold, depending on simulation seed. The course-balanced figure averages the two seed estimates; it does not mean a fraction of a golfer had an outcome. These cases involve only **42 distinct players**, and rare-event resampling has limited information.

This is consistent with overly generous extreme APP upside in the no-cut sample despite reasonable ordinary-good-week frequency. Average actual-minus-forecast APP is **+0.29 SG**, so the problem in this group is not simply an overly optimistic average forecast. Both extreme tails occur less often than modeled. This supports separating tail shape and player variance allocation from any blanket increase in regression.

## The cut-event counterexample matters

Among qualifying cut-event completers, the highest-variance APP group has a ratio of **1.068** and reaches its modeled good-week 5% threshold **10.9%** of the time. That is only **23 cases involving 16 players**; reranking within qualified starters gives a ratio of 1.013 and 5% exceedance of 8.8% across 29 cases. Neither version reproduces the no-cut extreme-upside shortage.

The lowest-variance APP group in cut events is already too broad: ratio **0.598**, versus 1.072 in no-cut events. Pulling its variance upward toward the field average makes its cut-conditional ratio worse, to **0.562**. This helps explain why the same shrinkage rule does not transfer cleanly.

ARG's low-variance cut group still has more error than predicted variance: **1.316** across 36 cases. Its high-variance group has ratio **1.185**, but only 11 cases across three events/courses; the original two-sided low/high pattern is not established there. Reranking within qualifying players also produces unstable high-group estimates. Do not interpret these thin conditional groups as reliable player or course rankings.

## Does the candidate actually improve full-event forecasts?

CRPS evaluates the whole four-round predictive distribution; lower is better. Results below compare the frozen combined APP/ARG half-shrink candidate with baseline.

| Category | Qualified no-cut CRPS change | Qualified cut-conditional CRPS change |
|---|---:|---:|
| OTT | +0.001% | −0.003% |
| APP | **−0.063%** | **+0.138% — worse** |
| ARG | **−0.125%** | **−0.045% — uncertain** |
| PUTT | −0.002% | −0.001% |

- **No-cut ARG remains the strongest result.** The absolute CRPS change is −0.001574 SG, with an eight-contrast adjusted course interval of **−0.003315 to −0.000552**. Event and recurring-player intervals also remain negative. Improvement appears in both seeds, both chronological halves, the 2026 panel, and every leave-one-course-out comparison. Case-weighted improvement is 0.099%.
- **No-cut APP remains a small, uncertain aggregate gain.** Its adjusted interval includes zero, as do event and recurring-player intervals. The high-variance group's point estimate improves more, but that does not establish a general APP correction.
- **Cut-event APP worsens in all four events**, under both seeds and every one-event exclusion. The adjusted course interval is positive, but the recurring-player interval includes zero; four events are still few independent settings. APP-only shrinkage also worsens APP CRPS by 0.136%, showing that this problem already exists before ARG shrinkage is added.
- **Cut-event ARG is no longer a clear replication.** Its score improvement is small and uncertain, and one event worsens. It does not retain the original broad sample's strength.

These are distribution-score changes, not observed betting returns. No shrinkage amount was fitted to this subset.

## Other categories and the four-round question

| Category | Qualified no-cut error/variance ratio | Qualified cut-conditional ratio |
|---|---:|---:|
| OTT | 0.964 | 0.958 |
| APP | 0.921 | 0.947 |
| ARG | 1.058 | 1.282 |
| PUTT | 1.114 | 1.335 |

PUTT's qualified no-cut R1 ratio is **1.123**, versus 1.114 across four rounds. Its four-round average error is **−0.24 SG** in no-cut events and −0.36 in cut events, so mean calibration should remain visible alongside spread. This still does not establish an error in how updating accumulates uncertainty.

High-variance APP's ratio is already **0.660 at R1**, versus 0.681 cumulatively. Its excess spread does not require a four-round momentum explanation. Conversely, low-variance ARG goes from **1.002 at R1 to 1.462 cumulatively** in the identical selected cases. That makes full-event ARG behavior a useful specific follow-up, but this comparison alone cannot identify later-round marginal errors, cross-round dependence or the update rule as the cause.

## Robustness and decision

- Requiring complete OTT/APP/ARG/PUTT measurements for counted history rounds retains 647 no-cut starters and 178 cut starters. No-cut ARG improvement remains 0.138%; APP still worsens in the scored cut subset.
- Counting other available tours, with duplicated player/date/round records removed, retains 649 no-cut and 183 cut starters. No-cut ARG improves 0.128%; cut-conditional APP worsens 0.117%. These sensitivity counts are not substituted for the PGA histories used by the model.
- Strictly greater than 20 versus at least 20 six-month rounds gives identical membership. Independent direct counts reproduce every player's trailing-window counts, and frozen baseline skills match the input arrays exactly.
- Reaggregating the unfiltered population reproduces the parent study's metrics within 1e−12. Independent explicit aggregation reproduces selected score changes. Source/history/forecast hashes remain unchanged. Forecasts, cut masks, fields and production inputs were not modified.
- The parent study's limited course coverage, exclusion of withdrawal/disqualification editions, current-method historical reconstruction, field-relative weather-neutral target and raw-category/final-score boundary still apply. This does not establish final finish-price accuracy or attribute a change to a historical model version. These are previously examined historical events, not a fresh holdout.

**Recommendation:** keep ARG as the first category to validate, now explicitly with an **ARG-only** candidate and this pre-event quality/sample subgroup retained. The present experiment adds ARG shrinkage on top of APP shrinkage; it has not yet established the isolated ARG-only production effect. Do not package APP shrinkage with it. For APP, investigate the no-cut outer tail separately while requiring the change to survive cut-conditional evaluation. High samples do not eliminate the no-cut findings, but they sharpen the evidence against a universal combined correction.

[Membership and counts](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/good_golfers_20260910/membership.csv>) · [Frozen design](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/good_golfers_20260910/contract.json>) · [Verification receipt](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/good_golfers_20260910/completion_manifest.json>) · [Parent full-event report](<C:/Users/McKinley Slade/dev/sims_process/FOUR_ROUND_CATEGORY_SHAPES_RESULTS_2026.md>)

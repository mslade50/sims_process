# Historical course replay: fidelity and robustness protocol

2026-09-09. Status: initial preflight completed; the larger replay has not passed all gates or started. [Preflight results and unresolved gates](COURSE_REPLAY_PREFLIGHT_2026.md). Companion to the [variance research brief](VARIANCE_RESEARCH_BRIEF_2026.md).

## Question and claim boundary

Primary: does the specified variance model, supplied with information from before each event, reliably predict player-relative round distributions across repeated editions of the same physical course? Secondary: does an exposure-based change improve it?

This is a reconstruction of a specified modeling method. It is not automatically a replay of originally issued probabilities, actual bets, or realized account profits. Historical production replay needs separately bound run inputs, code, runtime and outputs. Previously inspected seasons remain retrospective evidence; a new random split or new simulation seed cannot make them untouched.

## Required gates, in order

1. **Inventory before outcomes.** Build a course-edition eligibility matrix recording physical course/venue changes, event identity, tee-time/course routing, input cutoff, field provenance, raw/category/shot coverage, skill source, fitted-parameter provenance and missing inputs. Select the pilot from coverage and event structure, without ranking its historical results. Exclude unsupported cases explicitly; never substitute another venue or silently use current profiles.

2. **Historical information boundary.** Cap source observations before the whole event. Freeze inputs for every event before consuming any result from that event or another event sharing the cutoff. Audit every learned layer, including normalization, shot-SG processing, calibration weights, shrinkage and market-derived priors. A historical row date does not prove that its fitted value was known then. Refit using earlier information where necessary; distinguish a genuinely chronological refit from an exploratory replay using retrospectively selected fixed parameters. No event's realized future shot counts, lies, hazards or scoring may enter its forecast. Changing/deleting post-cutoff data in an isolated test fixture must leave that event's inputs and output unchanged.

3. **Production-engine fidelity.** Pin code, compiled kernel, dependency versions, configuration, seeds and all input hashes. Reproduce a fully bound current production fixture using its identical inputs before applying historical reconstruction. Compare intermediate means, category scales, correlations and skew, then final round and tournament outputs. Include the actual final skew treatment, player-week dependence, cut rules, ties and live cascade when those markets are evaluated. Deterministic identical-input paths must match; statistically equivalent paths need declared Monte Carlo tolerances fixed before results. Passing a kernel fixture alone does not validate the historical input builder.

4. **Separate distribution and operational tests.** R1 is primary. Keep the same skill means, fields and other assumptions in baseline and challenger, changing only the registered variance/exposure component. Preserve a diagnostic that removes common course/round scoring conditions symmetrically, but label it conditional and do not use it as an executable price replay. Absolute-score/market evaluation requires supported pre-event conditions or an explicit common reconstruction assumption. R2-R4 with frozen pre-event skill cannot stand in for live forecasts. Finish evaluation needs full pre-event fields and a joint tournament simulation; never use only eventual cut-makers or the realized cut size as a forecast input. Actual-starter reconstruction and documented withdrawals must be distinguished from what was known at pricing time.

5. **Numerical precision for the intended claim.** Use paired draws to compare arms and independent seeds plus increasing draw counts to measure numerical error. A fixed seed establishes reproducibility only. Register practical tolerances before the pilot's outputs are inspected; separately assess moments/distribution scores, individual probabilities, and near-threshold selection/EV stability. Evaluate precision by market and rare-tail probability, not just a pooled percentile. Stop at a declared compute cap if precision fails; report the unresolved comparison. Do not infer a change's profitability from numerically unstable selected sets or average separately sized small simulation runs.

6. **Statistical robustness.** Register the baseline, a small challenger set, metrics, course groups, testing periods and decision rule before evaluating candidates. Primary evidence is paired proper distribution/probability scores, spread/interval calibration and separate mean bias. Use temporal validation with all fitting confined to earlier observations. Assess uncertainty and sensitivity at event/edition, recurring-course and repeated-player levels; thousands of overlapping pairs are not independent events. Show leave-one-edition/course-out checks and stability across periods. Shrink sparse course and player-course interactions. Account for the candidate search; mark previously explored years accordingly. Reserve prospective evidence before promotion.

7. **Economic and independent verification.** Only after the preceding gates, evaluate matchup and finish markets separately with timestamped quotes, declared timezone, both sides, book-specific ties/dead heats/withdrawals, and the existing >=5% versus <5% Kelly theoretical-EV definitions. Do not substitute another study's EV gate. Quotes are not fills. Independently recompute selected cases and reconcile field totals, probability/payout accounting, means, variances and settlement. A self-consistent result from one code path is insufficient. Save immutable manifests and all exclusions so another run can reproduce it.

## Pilot and stop rules

First deliverable is the eligibility matrix and a small fidelity/precision pilot covering differing field/cut structures and physical-course routing. No broad course-performance ranking or feature optimization precedes it.

Proceed to the larger multi-edition study only for the claims that pass. Unresolved future information, identity, mean/shape ownership or joint-simulation fidelity blocks the affected analysis. Insufficient power produces an inconclusive result; it is not grounds to tune on the same outcomes. A valid round-variance diagnostic may proceed while an unsupported historical profitability claim remains excluded. No production changes follow automatically from research.

## Existing evidence informing the gates

- The recent [conditional audit](CONDITIONAL_VARIANCE_AUDIT_2026.md) intentionally reconstructs distributions and omits the final skew top-up and full cascade. Its aggregate spread result does not certify final production tails.
- The September 7 production archive catalog identified stale/misbound event artifacts and incomplete historical runtime inputs. A directory name or nearby timestamp is insufficient lineage.
- The September 7 convergence study found that 50,000 draws were insufficient for precise betting comparisons, and no market/model passed every registered upper-range condition even at two million draws. This does not invalidate simpler moment diagnostics; it requires claim-specific precision checks.
- Existing corrected prepared inputs and source/roster verification are reusable evidence, not automatic proof that all current production layers or historical data transformations are reproduced.

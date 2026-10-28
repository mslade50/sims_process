# Distance, lie and course variance: six-course results

Completed 2026-09-10. **The broader test gives a small, consistent positive indication for course-level distance/lie information. It does not establish a useful incremental matchup improvement from distance/lie, or support replacing every existing course multiplier.** The earlier Riviera pilot was insufficient grounds to move away from this research.

Across 34 editions, adding distance and lie to a count-only course model improves average round-distribution scores at all six courses and in both simulation seeds. The size is small: **0.0029% relative CRPS improvement**. The combined course/player exposure proposal improves average matchup Brier scores **0.0302%**, with favorable course averages at all six courses, but nearly all of that movement is already present in the count-only course replacement. Uncertainty across editions and the correction for the three primary comparisons still include no improvement.

## Coverage and question

The comparison includes **4,131 modeled R1 player-edition cases, 548 distinct players and 239,939 overlapping non-tied matchup pairs**. Those pairs are not independent observations. Primary aggregation gives each physical course equal weight and each edition within a course equal weight.

| Requested course / event | Editions scored | Years | Modeled R1 player cases |
|---|---:|---|---:|
| Bay Hill / Arnold Palmer Invitational | 6 | 2021–2026 | 563 |
| Muirfield Village / Memorial | 7 | 2020–2026 | 694 |
| Renaissance Club / Scottish Open | 3 | 2024–2026 | 460 |
| TPC Twin Cities / 3M Open | 6 | 2021–2026 | 877 |
| Sedgefield / Wyndham | 6 | 2021–2026 | 912 |
| Harbour Town / RBC Heritage | 6 | 2021–2026 | 625 |

All 41 course-edition rows in the 2020–2026 inventory are retained, including exclusions. Seven lack the required validated prior course exposure or target SG coverage. The shot-SG sample starts in 2020, leaving no earlier course exposure for most 2020 tests. Memorial 2020 has the preceding Workday Charity Open at the same physical venue as history; Workday itself has no earlier course exposure and is not scored.

Renaissance 2022 has **33,319 raw shot rows but no processed shot-SG or player-round records in the validated export**. Thus 2023 lacks an admissible prior course exposure profile in this study. Those raw shots are a reconstruction opportunity, not evidence that the data cannot be recovered. They were not silently substituted into the frozen comparison.

Course identity checks resolve two important cases. Workday's DG course ID 893 maps to Muirfield's 23 after all-18-hole geometry/par matching. Renaissance's 2026 ID 945 maps to 977: the same physical holes were reordered. After matching, coordinate RMSE is 15.18 versus roughly 1,915 with naive hole-number alignment. The aggregate exposure model can use that verified venue history; a hole-sequence model would also need the explicit routing map.

The modeled sample covers 4,131 of 4,226 observed R1 roster cases (97.75%). Missing minimum skill histories are recorded. Eighty-nine admitted cases lack prior complete APP shot history and use the explicit tour-prior fallback; 259 have fewer than 12 such rounds. Actual exposure validation is available for 3,949 modeled cases. This is an actual-starter reconstruction, not an archived bet-time roster.

## Frozen comparisons

All means, other SG-category variances, correlation and skew parameters stay fixed. The same normalized historical skill recipe, shot-dispersion baseline and final round-skew treatment are used throughout.

- **Course count:** replace the APP course variance multiplier with a prior course/tour expected-shot-count ratio. This is a model replacement, so its result cannot by itself identify shot counts as the cause of improvement.
- **Course exposure:** add the prior distance/lie mixture variance to that count model. Comparing it directly with course count isolates the incremental distance/lie component at course level.
- **Player exposure:** adjust how APP variance differs between players while preserving field-average APP variance. Test both the conservative Riviera settings and a predeclared less-shrunk version.
- **Combined exposure:** combine the course-exposure replacement with the less-shrunk player allocation.

The nine APP cells cover four distance bands crossed with clean/bad lies plus a residual cell. Raw valid SG includes hole-outs; mixture variance includes both within-cell variance and differences between cell means. Historical events must finish before the target starts. Player profiles use up to 80 previous complete rounds, EMA span 50. Less-shrunk settings use 25-shot cell shrinkage and 200-shot player-mix shrinkage, versus 200 and 1,000 in the conservative arm, and remove the half-log damping. Neither player-factor clipping bound is reached, so clipping does not explain the small player-level effect.

## Results

Negative changes below mean better scores. These are **relative changes in accuracy metrics**, not percentage-point changes in win probability, and not ROI.

| Primary comparison | CRPS change | Brier change | Courses with better CRPS / Brier |
|---|---:|---:|---:|
| Distance/lie added to course counts | −0.00287% | −0.00013% | 6/6 · 2/6 |
| Less-shrunk player exposure vs baseline | −0.00033% | −0.00044% | 3/6 · 4/6 |
| Combined exposure vs baseline | −0.00348% | −0.03023% | 4/6 · 6/6 |

The distance/lie CRPS direction repeats at every course in each independent seed and improves 22 of 34 individual editions. The average incremental matchup effect is essentially zero. The largest course-average incremental CRPS gain is at Renaissance (0.0137%); the other five are smaller.

Count-only replacement already improves Brier 0.02970%, compared with 0.03023% for the combined proposal. **Most of the apparent matchup improvement therefore comes from the course-level replacement, not the incremental distance/lie or player-personalization terms.** Less shrinkage increases player input movement but does not reveal a large forecast-accuracy effect.

Here is the combined proposal versus the baseline reconstruction by course:

| Course | CRPS change | Brier change |
|---|---:|---:|
| Bay Hill | −0.0082% | −0.0223% |
| Memorial | +0.1018% | −0.1155% |
| Renaissance Club | −0.0636% | −0.0021% |
| 3M / TPC Twin Cities | −0.0571% | −0.0186% |
| Wyndham / Sedgefield | +0.0433% | −0.0069% |
| Harbour Town | −0.0514% | −0.0162% |

The disagreement between scores matters. Memorial's overall spread calibration worsens: its average squared-error/predicted-variance ratio moves **1.021 → 1.137**, while central-90% coverage falls **89.16% → 87.72%**. Sedgefield moves **0.981 → 0.934**, making its forecasts wider relative to errors. Bay Hill and Harbour Town move closer to ratio 1. A blanket course-multiplier replacement is therefore not supported by this study, despite the favorable average matchup direction.

## Uncertainty and stability

![Primary effects and simultaneous uncertainty](<C:/Users/McKinley Slade/dev/sims_process/permanent_data/research/six_course_exposure_20260910/primary_effects.png>)

The uncertainty analysis resamples whole recurring courses and, separately, editions within courses. A repeated-player cluster sensitivity is also retained for CRPS. There are 10,000 replicates per analysis. Approximate 95% simultaneous course-cluster intervals account for the three primary contrasts separately within each metric:

- Incremental distance/lie CRPS change: **−0.0069% to +0.0012%**.
- Less-shrunk player-exposure CRPS change: **−0.0024% to +0.0017%**.
- Combined Brier change: **−0.0667% to +0.0062%**.

All include zero after this adjustment. Unadjusted distance/lie CRPS intervals are favorable under course, event and player resampling; that is suggestive evidence, not a basis to ignore the multiple-comparison result. The combined Brier interval also includes zero when resampling editions. Six selected courses give limited precision about broader course generalization.

Combined Brier stays favorable after leaving out any one course or year. The CRPS conclusion changes sign when some courses or years are omitted and when players are weighted equally instead of courses. Holding course composition fixed at the five courses available in both periods, combined CRPS improves **0.0102% in 2020–2023** but worsens **0.0201% in 2024–2026**; Brier improves **0.0363%** and **0.0423%**, respectively. Incremental distance/lie CRPS remains favorable in both periods for those same five courses.

Excluding Memorial 2020 or Renaissance 2026 does not reverse the combined Brier direction, but can materially affect the tiny CRPS average. These are target-edition exclusions, not a claim that Workday history has been removed from all subsequent Memorial forecasts.

## Does the exposure mechanism itself work?

It predicts exposure modestly better across the broader panel:

- Prior course mix reduces distance/lie mix error **6.04%** versus tour-average mix, averaging equally over courses and editions. It helps at five courses; Renaissance's historical course mix is slightly worse than the tour mix on this measure.
- Personalizing the course mix with less shrinkage reduces error another **1.00%**, with improvement at all six courses. Conservative shrinkage gives roughly half that benefit.
- Personalized expected APP counts reduce count MSE **4.07%** versus the course-average count, with improvement at all six courses. This reverses the unfavorable count finding from the small Riviera pilot.

These exposure-prediction gains do not automatically translate into materially better score distributions. Course replacements move some total round SDs by roughly 2–7%, whereas the 95th-percentile absolute change from less-shrunk player allocation is only about 0.28–0.59% across courses. The current cells and independent-shot variance formula can capture only part of course difficulty, shot dependence and tails.

## Verification and limits

All 34 editions were sealed before outcome scoring. Observation cutoffs pass future-data corruption tests; synthetic future skill rows make the boundary test non-vacuous. Pre-cutoff negative controls change the forecasts as expected. Normalization extends the verified online history by 53 source events, with every training proxy window fully closed before use; earlier 2023/2024 snapshots remain exact.

The production Python skill oracle matches all 145 modeled Wyndham 2026 players exactly across 1,722 available numeric cells. Independent Python/Rust checks pass for an actual candidate field at every course, and the retained Riviera final probability distribution reproduces exactly. Probability, CRPS, factor-replacement, aggregation and bootstrap controls reconcile independently. All primary edition comparisons pass the registered numerical gates at 50,000 draws per seed for two seeds; maximum paired-seed gaps are 0.000419 CRPS and 0.0000538 Brier. These checks support the reported accuracy comparisons, not fine betting-threshold stability. Source/input hashes remain unchanged.

This is conditional R1 research using a fixed current method, not historically issued probabilities or fresh out-of-sample evidence. Shared external field-strength, shot-SG and skew-calibration vintages remain inherited limitations. Full finish dependence, live updates, weather-specific pricing, actual quoted/selected matchups, and ≥5% versus <5% Kelly theoretical-EV subsets are not evaluated here. These results do not attribute the season drawdown to variance.

## Recommended next work

**Continue the exposure investigation.** Extend this unchanged comparison to the remaining supported physical courses, and reconstruct/validate the missing Renaissance 2022 SG input separately. Keep the six-course study fixed as the initial evidence.

The promising modeling direction is **course-level calibration that blends exposure information with the existing round-based course estimate**. Memorial and Sedgefield show why completely replacing that estimate can damage distribution calibration. Any blending weight should be learned only from earlier events and tested on later editions, with both distribution and matchup scores retained. Keep player-specific adjustments exploratory until they produce a larger, repeatable benefit. Hazard exposure, within-hole shot chains and persistent weekly uncertainty remain separate questions.

Evidence: [frozen contract](permanent_data/research/six_course_exposure_20260910/contract.json), [all editions and exclusions](permanent_data/research/six_course_exposure_20260910/course_edition_inventory.csv), [edition-level results](permanent_data/research/six_course_exposure_20260910/paired_deltas.csv), [clustered uncertainty](permanent_data/research/six_course_exposure_20260910/uncertainty.csv), [final independent verification](permanent_data/research/six_course_exposure_20260910/final_verification.json). All artifacts are local; no production inputs or betting rules changed.

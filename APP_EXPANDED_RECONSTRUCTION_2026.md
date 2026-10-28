# Expanded APP reconstruction

2026-09-09 · Follow-up to the [initial APP audit](APP_FORECAST_AUDIT_2026.md)

**The expanded analysis supports the initial finding: the ≥5% R3 APP deterioration is already present against the pre-event target; average net APP uplift from live updates did not increase. Weather recovery was not a meaningful barrier.** The earlier 83-bet strict archive was a validation subset, not the limit of the analysis we could do.

There is also an important model-version distinction: the round matchup simulator introduced category-first draws on **March 17**. Before that, it drew total SG directly. An APP allocation from those earlier total predictions is a diagnostic benchmark, not an independently issued APP forecast. The live skill process could use APP information even when the final round draw was total-only.

## What was reconstructed

Used the existing 643 verified sharp-book signals, the cached September 1 R2 historical database snapshot, local Git archives, and recorded prediction/weather differences. No fresh remote R2/D1 retrieval was required. The weekly R2 archive itself remains uninspected; this report does not claim an exhaustive remote inventory.

1. **Pre-event category profiles:** use an eligible dated archived profile where available; otherwise reconstruct the historical EMA50 method from PGA observations since 2019, clipped to ±8 SG, stopping strictly before the event starts. Archived profiles must precede the event, be no more than 14 days old, and cover at least 60% of the event roster. This allows retained profiles such as the one actually used at Travelers. These eligibility checks identify plausible historical inputs; they do not alone prove every profile version was used in production.
2. **Live total-SG advantage:** take the recorded `pred_on − pred_against` rather than rebuild every upstream adjustment. Subtract the recorded weather difference when available; use zero with an explicit flag otherwise.
3. **APP allocation:** the category-first formula gives `APP gap = raw APP gap + (neutral total-SG gap − raw category-sum gap) / 4`. Thus a recorded total target and the category profile are sufficient for this category-location audit. No simulation draws or historical course variance settings are needed.
4. **Pre-event anchor:** recover the original `pred` carried in event-identified R1 live-model files. Compare it with the recorded live target using the same category profile. The difference is the net change since that anchor, including live adjustments and any subsequent overrides; it is not solely the explicit APP coefficient. R1 itself has no completed-round live update to remove.

The method reconstructs location targets before draw clipping. It does not recreate exact fair probabilities, alternative bet selections, or counterfactual P&L. No target-round outcomes select inputs or fit a correction.

### Coverage and validation

| Evidence | Coverage |
|---|---:|
| Full-season diagnostic APP estimates | 643 / 643 bets |
| Both category profiles drawn from eligible pre-event archives | 488 bets |
| At least one historical EMA reconstruction | 155 bets |
| Low-history reconstructions within those 155, separately flagged | 8 bets |
| Pre-event / live split | 619 / 643 bets |
| R3 diagnostic APP estimates | 196 / 196 bets |
| R3 pre-event / live split | 181 / 196 bets |
| Category-first-era R3 pre-event / live split | **144 / 144 bets** |

The eight sparse-history cases are excluded in a sensitivity; none belongs to the category-first-era ≥5% R3 comparison. Eight Genesis R3/R4 pre-event anchors come from a later archive preserving earlier R1 data; they are marked as reconstructed, with a sensitivity excluding them. The remaining 24 missing pre-event anchors are WM Phoenix and AT&T R2–R4 cases. Their total-target-normalized APP diagnostic is still available, but its pre/live split is not asserted.

Against the **83 strict forecasts** previously recovered, the expanded method's live APP target differs by **0.0027 SG on average**, with a maximum of **0.0127 SG**, consistent with rounded logged total and weather inputs. Pre-event target differences are at most 0.0018 SG. Rebuilding the category shape entirely from the DB, rather than using the archives, has a mean absolute paired-shape discrepancy of 0.0154 SG on that benchmark. This is evidence of useful reconstruction accuracy, not proof that every unverified historical row is exact. Historical adjustments in the present database may contain later revisions; the cutoff prevents use of later rounds, not every possible later data correction.

## Main comparison: the same category-first round model family

Commit `53d05f6`, March 17 at 12:52 ET, introduced `simulate_round_scores_catfirst` and made it the default, with an explicit `--legacy` option. Era labels follow the repository default; original command-line flags are not recovered for every run. Compare March 17 through Memorial with afterward, holding this architecture distinction fixed.

Error is actual backed-minus-opponent adjusted APP SG less its reconstructed expectation. Positive is favorable to our side.

| ≥5% R3 | March 17–Memorial | After Memorial |
|---|---:|---:|
| Bets | **50** | **48** |
| Logged profit | +9.77u | −0.48u |
| APP error against pre-event target | **+0.574 SG** | **−0.385 SG** |
| Net live uplift allocated to APP | **+0.080 SG** | **+0.079 SG** |
| APP error against live target | **+0.494 SG** | **−0.464 SG** |

**The pre-event error moved −0.959 SG; live-target error moved −0.958 SG. The live uplift changed only −0.001 SG.** This is strong descriptive evidence against an increase in average live APP uplift as the explanation for this swing. It does not prove the baseline developed a persistent bias: the 95% whole-event bootstrap interval for the live-error change is **−1.85 to +0.18 SG**. Average uplift can also remain stable while its player-level calibration changes.

All 98 bets in this main comparison have recorded weather differences and timely archived pre-event scalar anchors. Ninety-four also use eligible archived category profiles. Restricting to those profiles gives early/live APP error +0.332 versus later −0.464; uplift is +0.084 versus +0.079. The conclusion does not depend on the four EMA-profile reconstructions.

The <5% group remains different. In the same model era its APP error **improved from −0.920 to −0.199 SG**, while profit deteriorated from −0.50u to −8.24u. APP differential deterioration cannot explain those later losses. Removing or shrinking APP terms alone is not an adequate explanation or solution.

## The entire early run, including the previous round engine

Of the full early ≥5% R3 profit of +29.00u, **+19.23u came from 37 bets before March 17**, and +9.77u from 50 afterward. This is an additional reason to avoid treating the whole early run as one unchanged model. It does not establish that the engine change caused a return difference; a causal comparison would require the same inputs, prices, and opportunity set.

Using one consistent category-allocation diagnostic across all 135 ≥5% R3 bets, early APP error is **+0.347 SG** and later **−0.464 SG**. The expanded pre/live split covers 76 of the 87 early bets and **+21.83u of their +29.00u**, compared with +1.65u covered by the original strict subset. On that common early sample, pre-event error is +0.422 and live uplift +0.078; later values are −0.385 and +0.079. These earlier-engine allocations are explicitly diagnostic rather than recovered category outputs.

## Weather sensitivity

The user's proposed simplification is supported by the data:

- All 643 pairs have matching historical physical-course IDs.
- Median tee-time gap is **10 minutes**; the 90th percentile is 33 minutes.
- Weather differences are logged for 520 bets; **93.3% are rounded to zero**.
- Flagged 21 bets overall, including 13 R3 bets, for a tee-time gap over 60 minutes or an absolute logged weather difference over 0.10 SG. Some pairings do have larger gaps; they are listed individually in `weather_exceptions.csv`.
- Setting all recorded weather differences to zero changes average ≥5% R3 APP error by only **0.0013 SG** across the season.
- Excluding weather exceptions in the category-first ≥5% R3 comparison leaves **47 early and 47 later bets**. APP error still moves **+0.452 → −0.476 SG**; live uplift is **+0.081 → +0.079**.

Tee-time proximity does not prove identical experienced weather, particularly with interruptions, but the reported conclusion does not require that strong assumption.

## Balanced APP and implications

Across all 45 ≥5% Balanced APP bets, reconstructed APP error is **−0.767 SG**, versus **−0.714 before the net live change**. The live allocation adds only +0.054 SG of expected advantage. Most of the deficit is already present before that change. The corresponding full-season <5% error is −0.047 SG, roughly neutral.

Restricting to the category-first era gives the same central concern: **36 ≥5% bets, −9.13u, and −0.737 SG APP error**, with an exploratory event-bootstrap interval of −1.37 to −0.16. The 33 <5% bets have −0.199 SG APP error, with an interval spanning zero. The prior finding of substantial putting underperformance in Balanced APP remains relevant; the label alone does not identify which category caused a loss.

The broad ≥5% R3 deterioration survives excluding any single player or event, dropping sparse-history cases, using only archived profiles, rebuilding every profile from the historical DB, and excluding weather exceptions. The DB-only diagnostic is +0.347 early versus −0.471 later, nearly identical to the archive-first reconstruction.

**Recommended next work remains conditional pre-event APP calibration**, using prior reliability, recent-versus-longer-term form, and model-market disagreement, validated chronologically. Keep a separate player-level test of net live adjustments; their stable average does not demonstrate optimal calibration. Keep March 17 and June changes separate in future implementation studies. This audit supports neither a blanket APP penalty nor a rollback, and it does not establish that all deterioration was luck.

## Reproduction and checks

Research files: [app_expanded_20260909](permanent_data/research/app_expanded_20260909/).

- `build_inputs.py`: dated profile and pre-anchor recovery, historical EMA reconstruction, logged-total APP allocation, tee-time/weather flags, and source manifest.
- `analyze.py`: frozen-target validation, complete and common-sample comparisons, architecture-era split, event bootstrap, and exclusion sensitivities.
- `analysis_evidence.csv`: all 643 original signals, source flags, estimates, and era labels. `frozen_validation.csv` retains every benchmark discrepancy. `summary.csv`, `event_bootstrap.csv`, and `weather_exceptions.csv` provide reviewable results.
- `verify.py`: sample/P&L reconciliation, pre-event cutoffs, algebra and rounding checks, original source hashes, and architecture-source checks.

All work is confined to research outputs and documentation. Production code, Sheets, cloud storage, model coefficients, and stakes were not changed.

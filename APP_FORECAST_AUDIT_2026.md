# APP forecast-error audit

2026-09-09 · Research worklist R3 · Sharp-book round matchups

Follow-up: the [expanded reconstruction](APP_EXPANDED_RECONSTRUCTION_2026.md) extends beyond this strict saved-output subset using historical inputs and logged total/weather differences, and separates the pre/post-March 17 round engines. The coverage limits below describe this initial audit, not all recoverable data.

**The strongest new lead is conditional forecast calibration, especially ≥5% Balanced APP exposure. The recovered evidence does not show a midseason increase in APP uplift from live updates.** In ≥5% R3, the early APP advantage largely disappeared because backed players went from exceeding their pre-event rolling priors to falling short. The actual frozen-forecast subset shows a similar deterioration already against the reconstructed pre-event target, before adding live updates.

These are retrospective diagnostics, not a validated production change. No model coefficients, filters, stakes, or live records were changed.

## 1. Whose APP performance changed?

Full-season coverage: the same 643 best-first-run sharp-book signals, with the existing ≥5% / <5% Kelly theoretical EV split. Early means through Memorial; later starts after Memorial. Error here means actual adjusted APP SG minus a pre-event rolling prior. Positive means better than that prior. Pair error is backed-player error minus opponent error.

| Scope / band | Bets early → later | Backed-player APP error | Opponent APP error | Pair APP error |
|---|---:|---:|---:|---:|
| All rounds, ≥5% | 264 → 163 | +0.027 → −0.095 | −0.165 → +0.059 | +0.193 → −0.154 |
| R3, ≥5% | 87 → 48 | +0.187 → −0.412 | −0.243 → −0.064 | +0.431 → −0.348 |
| R3, <5% | 34 → 27 | −0.236 → −0.597 | +0.227 → −0.431 | −0.463 → −0.167 |

Across all ≥5% rounds, approximately **65%** of the APP deterioration came from opponents improving relative to their priors. **Within R3, approximately 77% came from the backed players deteriorating.** The cause of the aggregate change should not be applied indiscriminately to each round.

The early ≥5% R3 run benefited from both sides: our players exceeded APP priors by 0.19 SG while opponents missed theirs by 0.24. Later our players missed by 0.41; opponents were still slightly below their priors. The +0.43 pair advantage became −0.35. This coincided with profit falling from +29.00u to −0.48u, but SG errors are not an additive dollar-profit attribution.

Sub-5% R3 tells a different story. Its pair APP error **improved** by 0.30 SG even as profit went from −4.29u to −8.24u. A worsening APP differential therefore does not explain that subgroup's worsening P&L. Other categories, prices, and the joint distribution of round scores matter.

The ≥5% R3 APP change is −0.778 SG; its 95% whole-event bootstrap interval is **−1.59 to +0.25**. This does not establish a lasting change in forecast quality or isolate luck. The direction survives removing any one event (change remains −1.02 to −0.57) or any one player appearing on either side (−1.10 to −0.59). It also survives counting unique player-round exposures and retaining all sharp-book quotes. Repeated exposures matter, but no single player or event creates the direction.

## 2. Did the live updates become too optimistic?

Recovered **91 of 196** R3 bets with matching archived prediction targets for both players and matching category-sum targets. Archive selection uses event identity, input matches, and timestamps, never outcomes. A commit may follow the recorded run by up to 15 minutes to allow for the archive write; the underlying R1/R2 data must already exist at the run.

For **83 bets across 12 events**, the four raw player-category means also reproduce the saved category targets to within `1e-8`, and the exact category-profile file was archived before the event began. This is the strict decomposition sample: 59 ≥5% and 24 <5%. Eight other matched bets are excluded from that decomposition because of missing raw profiles or a small reconstruction discrepancy.

For the strict sample, reconstruct APP as:

```text
Pre-event target = raw APP mean + (stored pre-event total skill − sum of raw category means) / 4
Live APP target  = pre-event target + (R3 total skill − pre-event total skill) / 4
Weather target  = live APP target + 0.35 × archived weather SG
```

The “live APP shift” is APP's allocated share of the **entire total-skill update**. It includes other category and residual information; it is not synonymous with the explicit APP update terms. These are the simulator's category location targets before draw clipping, not independently fitted APP-only forecasts. We remove the weather allocation when comparing with adjusted historical SG, and retain the weather-inclusive sensitivity in the data.

| Strict ≥5% R3 sample | Early | Later |
|---|---:|---:|
| Bets / events | 27 / 6 | 32 / 6 |
| Pair APP error against reconstructed pre-event target | +0.251 SG | −0.513 SG |
| Pair APP uplift allocated from live updates | +0.078 SG | +0.075 SG |
| Pair APP error against live skill target | +0.173 SG | −0.588 SG |

**The early/later error deterioration is already present before live updates.** The allocated APP uplift changed by only −0.003 SG, while pre-event-target error changed by −0.764 SG. There is no increase in average live APP optimism in this subset.

The explicit `avg_app_adj + delta_app_adj` terms, measured as their contribution to the backed player's total-skill advantage, averaged +0.016 SG early and −0.005 later. Their contribution also did not increase. This rules out that particular average-shift explanation within the recovered sample, not all possible update-calibration issues.

Updates did slightly worsen selected-bet APP error: ≥5% pair APP RMSE went from **2.278 to 2.304 SG**. The mean squared-error increase was +0.118 SG², with an exploratory event-bootstrap interval of +0.008 to +0.207. However, selection was based on the live model, and APP receives part of a total-skill update that may be justified by other categories. This is not sufficient evidence to remove that update or change the 25% allocation.

**Coverage is not representative of the whole early winning run.** The strict ≥5% early subset accounts for only **+1.65u of the full +29.00u**. The reconstruction supports a mechanism check on those bets; it cannot establish the bet-time forecast quality of all the highly profitable early bets. The earlier 50/100-round prior analysis has full coverage but is a different baseline.

## 3. A field-wide check prevents overreading selected bets

Used the earliest strictly matched archived version per event and expanded to **805 player-event R3 observations across 12 events** with valid actuals and complete, reproducible profiles. These are the recovered active fields, including players we did not bet. Errors are centered within each recovered event field so common SG-reference differences do not dominate the comparison. Events receive equal weight.

| Fixed comparison | Mean event MSE change | 95% event-bootstrap interval |
|---|---:|---:|
| Live APP target minus pre-event APP target | +0.0115 SG² | −0.0039 to +0.0300 |
| Live total-skill target minus pre-event total-skill target | +0.0514 SG² | −0.0543 to +0.1813 |
| Remove explicit APP terms from live total skill, versus retain them | +0.0100 SG² | −0.0143 to +0.0293 |

Positive is worse. None of these overall intervals establishes an improvement from a proposed change. Removing explicit APP terms made total-skill prediction worse in 9 of the 12 events; its average early-period effect was also worse. This supports **testing calibration carefully, not deleting APP terms**. These arithmetic target comparisons do not rerun selection, pricing, or the full live pipeline.

## 4. Balanced APP is a stronger, more specific concern

Archetypes use the same consistent pre-event classifier as the season review. This section covers all rounds; it should not be conflated with R3 alone.

| Backed Balanced APP | ≥5% | <5% |
|---|---:|---:|
| Bets / events | 45 / 18 | 37 / 19 |
| Profit | −13.27u | −7.53u |
| Backed-player APP error | −0.506 SG | +0.209 SG |
| Opponent APP error | +0.224 SG | +0.227 SG |
| Pair APP error | **−0.729 SG** | −0.018 SG |
| Pair PUTT error | **−1.153 SG** | −0.495 SG |

The ≥5% group's own APP underperformance accounts for about 69% of its APP deficit. Opponents also exceeded their APP priors. Putting was an even larger deficit: backed players missed their putting priors by 0.64 SG, while opponents exceeded theirs by 0.51.

The ≥5% APP deficit was present both early (−0.783) and later (−0.622); it is not a newly emerging June problem. Its seasonal 95% event-bootstrap interval is **−1.23 to −0.25 SG**, and the PUTT interval is **−1.81 to −0.33**. Both remain exploratory because the profile was selected after inspecting results, and event resampling does not remove all cross-event player dependence.

Removing any one backed or opposing player still leaves the ≥5% Balanced APP group with negative pair APP error (−0.92 to −0.59 SG) and negative profit (−15.31u to −9.22u). Removing any single event also preserves both signs. This is a broader concern than one bad golfer or week, although only 10 distinct backed players form the 45-bet group.

Specific examples show why an archetype ban would be crude:

- **Hovland:** five ≥5% Balanced APP bets lost 4.40u, with own APP error −1.32 and PUTT −1.21 SG. His six <5% bets won 2.00u, with APP +0.33 and PUTT +0.61. These small, differently selected samples are not evidence that his lower-edge bets are inherently better.
- **Straka:** ten ≥5% bets lost 4.05u even though his own APP slightly exceeded its prior (+0.10). His putting error was −0.90. Reducing his APP prediction because of these losses would misdiagnose the observed category performance.

Only 12 of the ≥5% Balanced APP bets were in R3, with just two later. Strict live decomposition covers only six of them, one later. It cannot establish a profile-specific change in live-update behavior.

## Recommended next test

**Prioritize pre-event APP calibration conditional on model-market disagreement and forecast reliability.** Use continuous APP strength, prior sample size, recent-versus-longer-term APP divergence, round, opponent strength, and pre-bet disagreement with a vig-free sharp-market reference. Evaluate the backed and opposing players separately. A market price implies total matchup strength, so do not pretend it provides an APP-only forecast.

The candidate should be a modest, chronologically estimated correction or shrinkage toward a reliable prior. Choose and fit it using earlier events, then test untouched events or another season. Compare category error, total-SG error, matchup probability calibration, CLV, opportunity count, and economic results. Preserve the distinction between changing an APP prior, changing its contribution to total skill, and merely reallocating a fixed total across category labels.

Keep the live-update calibration test open as a secondary question: the selected-bet APP results show some optimism, but the field-wide evidence does not justify removing APP terms. Investigate putting alongside APP for Balanced APP bets; do not infer the source of a loss from the player's archetype.

No blanket archetype exclusion, APP variance cut, 5% gate, or coefficient change is supported for deployment by this audit alone. Complete bet-time snapshot capture remains high priority because current archive coverage excludes much of the early profit.

## Evidence and verification

Files are under [the research directory](permanent_data/research/app_forecast_audit_20260909/):

- `analyze_errors.py`: verified 643/731-record inputs, side/category identities, full-season attribution, event resampling, unique exposures, and leave-one-out comparisons. `summary.csv`, `event_bootstrap.csv`, `players.csv`, and `events.csv` contain the results.
- `recover_live.py`: local Git recovery with event/time/input checks. `snapshot_manifest.json` records original file hashes; `frozen_coverage.csv` includes unsuccessful recoveries.
- `assess_frozen.py`: confirms pre-event raw-profile availability and decomposes matched targets. `frozen_r3_verified.csv` preserves both the strict flag and reconstruction discrepancies; `frozen_summary.csv` contains the before/after comparisons.
- `check_field.py`: fixed APP/total target comparisons across recovered fields. `field_comparisons.csv` and `field_player_evidence.parquet` preserve results and inputs.

Actuals come from the September 1 read-only historical DB snapshot already used in the season review. The old diagnostic tournament-category export was not treated as a frozen live forecast. No historical outcomes were used to choose archive versions or edge bands. No alternative thresholds or optimal correction coefficients were searched.

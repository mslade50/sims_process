# Persistent uncertainty across rounds

Completed 2026-09-10. **Round errors do persist, but this study does not support adding another blanket weekly-uncertainty component.** The observed size is close to the existing 0.65-SG weekly setting considered alone. The simulator's live-update cascade creates additional persistence, so the next priority is calibrating those two mechanisms jointly before making uncertainty more player- or course-specific. Production is unchanged.

## What the historical results show

The main sample contains **17,712 player-editions, 748 players, 143 editions and 34 physical venues**, from July 2020 through August 2026. It begins with the previous study's 152 eligible editions and frozen pre-event skill estimates. Of 18,201 modeled players, 18,033 have both first-round and second-round scores; all have readable tee times. Small joint tee-time groups and the minimum 30-player edition requirement reduce the final sample. Nine small-field editions drop out, leaving Albany and East Lake unrepresented in the primary panel. These exclusions are recorded individually.

Each error is actual performance minus the frozen pre-event skill forecast. Shared scoring conditions are removed by centering within the same joint R1/R2 AM/PM groups, with a noon boundary and at least ten players per group. This is a **conditional diagnostic**, not a forecast using realized weather. Field-only centering on the identical players provides a sensitivity check. It does not suggest weather caused the selected round-matchup downswing.

Covariance measures whether the same player's good or bad miss carries from one round to the next; units are squared strokes per round. Positive values indicate persistence.

| R1–R2 residual covariance | Estimate | 95% venue-cluster interval |
|---|---:|---:|
| Joint wave controls, frozen skill means | 0.435 | 0.292 to 0.601 |
| Joint wave controls, additional past-only bias correction | 0.480 | 0.333 to 0.656 |
| Field controls, frozen skill means | 0.402 | 0.258 to 0.570 |
| Field controls, additional past-only bias correction | 0.449 | 0.303 to 0.627 |

Results weight editions equally within equally weighted physical venues. Event-cluster and player-cluster intervals also remain positive. Bias-adjusted estimates remain positive when any one venue is omitted, ranging from 0.420 to 0.501. Annual estimates are positive in every year, although 2020 contains just four editions. The size varies: adjusted covariance is 0.239 in 2025 and 0.607 in 2026.

Under a simple additive model with independent round noise plus one shared player-week draw, the square root of covariance is an equivalent shared standard deviation: **0.66 SG before the bias correction, or 0.69 SG after it**. The former's interval is approximately 0.54–0.78. The adjusted residual correlation is about **6.3%**. Ordinary round noise still dominates.

That equivalent SD is not a uniquely identified weekly-form parameter. Player/course mean errors, unresolved skill uncertainty and other within-event effects can produce similar covariance. The current-method historical means also retain the external-data vintage limitations of the earlier rebuild; this is not an exact replay of every issued model version.

## Does acknowledging persistence improve a forecast score?

Yes, modestly, relative to an independent-round diagnostic. After twenty completed prior sample editions, **123 editions / 15,059 player-editions** remain for chronological scoring. Every learned value uses events completed before the target starts.

The comparison holds means and marginal variances fixed and evaluates a two-round Gaussian joint log loss. It compares zero covariance, fixed covariance of 0.65², and total covariance estimated from earlier residuals. R2 reuses the frozen R1 marginal variance. Group-centering adjustments are applied to both marginal variance and covariance.

| Diagnostic comparison | Change in log loss, nats per player-edition | 95% venue interval |
|---|---:|---:|
| Fixed 0.65-equivalent persistence versus independence | −0.002109 | −0.003592 to −0.000991 |
| Past-only learned total persistence versus independence | −0.002048 | −0.003630 to −0.000813 |
| Learned versus fixed persistence | +0.000061 | −0.000094 to +0.000221 |

Lower is better. The fixed setting improves this diagnostic loss by approximately **0.044%**. Both persistence arms improve the aggregate result when any one venue is removed and in each adequately populated annual scoring panel, 2022–2026. The sole scored 2021 edition is adverse. Learning the scalar does not reliably improve upon the fixed setting.

These are conditional composite scores, **not actual finish probabilities, issued live-R2 forecasts, betting returns or a comparison against the entire production cascade**. In particular, the fixed 0.65 arm represents total covariance in this diagnostic. It does not add 0.65 on top of a simulated live update.

## Why simply adding more would be premature

The current tournament code already contains both a shared weekly draw and sequential skill updates. I extracted its exact first-two-round calculation, without executing production I/O, and ran frozen BMW and Memorial inputs. Each arm uses two 100,000-draw main-RNG seeds; production's separate fixed latent stream is reused, so those are not 200,000 independent weekly shocks. The coefficients remain those in the archived fixtures.

| R1–R2 simulated covariance | BMW fixture | Memorial fixture |
|---|---:|---:|
| Weekly draw off; R1 update off | −0.005 | −0.002 |
| 0.65 weekly draw only | 0.418 | 0.420 |
| R1 update only | 0.409 | 0.378 |
| Both mechanisms enabled | **0.833** | **0.801** |

The near-zero first row is Monte Carlo noise. The weekly draw preserves approximately the same individual-round variance while increasing dependence across rounds. The update also creates dependence, and their effects are nearly additive in these fixtures.

**This identifies a plausible overlap to test, not a proven production overstatement.** Two fixture fields, their archived coefficients, and first-two-round moments cannot settle four-round calibration across courses. The historical sample and these fixture fields are also different populations. Turning off the latent now would be as premature as adding another one.

## Other useful findings

**The simple recurring-player bias correction failed.** It uses only the previous 730 days of completed sample editions, fits a skill-calibration slope, and shrinks each player's prior average error with twenty pseudo-observations. It increases two-round mean squared error by **1.11%**, rather than explaining away persistence. Do not adopt this correction. A separately labeled post-primary check removes it entirely: the fixed persistence arm still improves joint log loss by −0.001790, with venue interval −0.003160 to −0.000718. The persistence finding therefore does not depend on this unsuccessful correction.

**Cut selection can reverse the apparent answer.** Restricting the same study to four-round completers leaves 9,843 cases. Their R1–R2 covariance is **−1.096**, versus +0.480 in the primary panel. Conditioning on surviving the first two rounds can favor a good round compensating for a bad one. Later-round covariances in that selected panel range from +0.088 to +0.301, but they cannot identify the pre-event four-round dependence without modeling the cut. Do not fit weekly uncertainty from eventual cut-makers alone.

**Longer-lasting errors are not established.** Adjacent observed player-editions within sixty days give 15,006 comparisons. The raw cross-event error product is −0.068, with interval −0.143 to +0.010. This does not support blindly carrying the same residual shock between tournaments. It also does not rule out underlying skill uncertainty: pre-event forecasts already update, intervening tournaments can be absent from this course sample, and the unsuccessful player correction makes the adjusted measure more negative.

**Player specificity remains a hypothesis.** Prespecified skill bands all show positive persistence. Their adjusted descriptive estimates run from 0.470 for skill ≤−0.5 to 0.617 for skill >1. Band populations and venue weights differ, so this is not an isolated player-skill effect or evidence for a new archetype multiplier. This study does not establish that balanced APP players or the selected ≥5% Kelly bets need larger uncertainty.

## Decision and next bounded test

Retain the present settings while testing **weekly latent plus the live-update cascade as one joint system**. Before introducing player/course-specific persistence:

1. Complete a frozen full-tournament comparison across supported historical fields, with explicit coefficient-version provenance, original cut rules, withdrawals and all final score transformations. The newly verified pre-cut calculation does not close the remaining final-finish fidelity gate.
2. Hold pre-event means and initial marginal spreads fixed. Compare a small prespecified set of joint dependence settings; fit any choice using completed prior events only. Track the cascade's R2–R4 mean and variance effects as well as the nominal latent SD.
3. Score make-cut and top-5/10/20 probabilities on the full pre-event population, with event/venue/player uncertainty and separate results by year. Add selected ≥5% and <5% Kelly groups only where exact archived prices and model versions are recoverable.
4. Introduce player history, course or player/course dependence only if it improves that chronological joint baseline. Avoid estimating bespoke effects from one or two player/course observations.

This result supports accounting for persistence. It does not attribute the season drawdown to missing persistence, prove excessive finish variance, or justify a blanket confidence haircut.

## Verification and reproducibility

All completed checks pass: independent NumPy covariance and venue aggregation, frozen skill-array equality, direct SQL checks at all 34 venues, **286 target/future mutation checks**, and **452 source-hash checks** across the study and mechanism manifests. Exact latent-off first-two-round arrays match the independent reference in both fixtures. Against the Rust kernel, category means and make-cut probabilities pass the prespecified five-Monte-Carlo-SE gate in all eight fixture/arm combinations; worst statistics are 3.45 and 3.15 respectively. The BMW make-cut check is trivial because its fixture has no cut. The simulator variance identity also reconciles exactly.

These checks cover the stated pre-cut boundary. They do not verify active-latent final-score skew calibration or historical issued-price fidelity. Bootstrap intervals above are pointwise, with 10,000 draws; already examined historical years are not fresh holdouts.

The research archive is [persistent_uncertainty_20260910](permanent_data/research/persistent_uncertainty_20260910/contract.json). Run `study.py`, `mechanisms.py`, `verify_results.py`, `report_tables.py`, and `raw_mean_sensitivity.py` there with the existing Python environment to reproduce the analysis. The frozen contract distinguishes the primary tests from the later raw-mean robustness check. Source manifests, per-player errors, chronological fits, coverage, annual tables, mechanism moments and verification receipts are retained. No source data, production model, Sheet or betting rule was changed.

[Living variance brief](VARIANCE_RESEARCH_BRIEF_2026.md) · [Parent research worklist](BETTING_RESEARCH_WORKLIST.md)

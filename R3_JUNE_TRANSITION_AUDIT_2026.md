# June R3 transition audit

2026-09-09 · Worklist R1 · Read-only production investigation

**The initial audit does not support reversing the June R3 correction or the original Rust migration.** The correction removes duplicated R1 information. Its direct skill effect is zero for 12 of the 14 post-change R3 matchups whose archived inputs match the betting log, including every Travelers matchup. The original June Python and Rust draw implementations passed a controlled statistical comparison. This narrows the suspects; it does not establish that every June change was harmless or explain the whole season.

## What changed

The June 9 commit `e61f07a` routed single-round category draws to Rust. The June 10 commit `d57a1aa` changed live updates and weather, among other things.

**The category-average blend labeled “R3” updates the prediction for R4.** The engine's round number refers to the round just completed. Restoring 66% prior average / 34% current-round categories therefore does not directly change the R3 matchup forecast. R3 predictions come from `_totals_r2`, after R2 has finished.

The direct R3 skill change was:

```text
Before: R3 skill = R2 skill + fresh two-round category terms + residual terms
After:  R3 skill = the same expression − R1 OTT/PUTT adjustment
```

The R1 category adjustment was already embedded in R2 skill. The new two-round category estimate should replace it, rather than add another estimate of the same signal. A positive R1 adjustment is removed; a negative one is added back. This is not a blanket reduction in player skill or putting variance. R1 OTT/PUTT coefficients were zero for the high and mid-high skill groups, so many players had nothing to remove.

The dew term also changed sign. Holding archived weather factors fixed, the expected-SG change is `−2 × dew_adj`. We isolate that sign change below. We cannot reconstruct the separate effect of changed Sheet override handling without the historical Sheet settings.

## Which archives support a betting comparison

Recovered R1/R2 component files reproduce the active archived R3 skill and weather-adjusted targets to within `1e-10` at all five events. “Matched bets” additionally requires **both** logged player prediction targets to match the archive within the log's two-decimal precision.

| Event | Selected R3 bets | Both logged targets match | Historical score distributions | Use |
|---|---:|---:|---|---|
| Charles Schwab | 8 | 8 | Available | Betting comparison, before change |
| Memorial | 9 | 0 | Not recovered | Component accounting only |
| Canadian Open | 4 | 0 | Not recovered | Component accounting only |
| U.S. Open | 7 | 7 | Available | Betting comparison, after change |
| Travelers | 7 | 7 | Available | Betting comparison, after change |

The Memorial and Canadian Open files reflect later reruns. They are not exact bet-time inputs and are excluded from the probability comparison. Some older common dashboard files contained the wrong event entirely; those were rejected. The [archive manifest](permanent_data/research/r3_june_transition_20260909/manifest.json) records source commits, timestamps, event checks, and file hashes.

Here `pred_on`/`pred_against` are the logged **weather-adjusted expected SG targets** (`scores_r3`), not pure underlying player skill. `my_pred3` is the separate skill target. Player-mix comparisons using `pred_on` should preserve that distinction.

## Before/after effects, retaining the original edge bands

Bands below use the historical normalized Kelly theoretical EV: ≥5% versus <5%. They are not reassigned based on hindsight or the counterfactual estimates.

| Event / band | Matched bets | Direct skill-gap change nonzero | Mean absolute estimated win-probability change |
|---|---:|---:|---:|
| Schwab ≥5% | 6 | 4 | 0.871 percentage points |
| Schwab <5% | 2 | 0 | 0.173 pp |
| U.S. Open ≥5% | 6 | 2 | 0.569 pp |
| U.S. Open <5% | 1 | 0 | 0.033 pp |
| Travelers ≥5% | 3 | 0 | 0.012 pp |
| Travelers <5% | 4 | 0 | 0.010 pp |

The skill-gap changes are exact component replays on frozen inputs. **Probability changes are approximate sensitivities**, including the isolated dew sign change. Historical score distributions are available, but the complete runtime course multipliers, skew settings, and Sheet overrides are not. We therefore shift the archived integer score-margin distribution by the exact forecast-gap change and interpolate between neighboring score bins. This assumes a location shift and does not recreate category clipping or changes in distribution shape.

At the archived target, probabilities reconstructed from the two players' score distributions differ from logged fair probabilities by at most **0.175 pp** across the 22 matchups. That supports using the archived distributions as the sensitivity baseline; it does not make the counterfactual exact. Probabilities below exclude pushes, consistent with fair matchup pricing.

| Matchup | Band | Exact skill-gap change | Estimated before → after win probability | Actual result |
|---|---|---:|---:|---|
| Schwab: Smotherman over Tom Kim | ≥5% | −0.2346 SG | 51.93% → 49.29% | Loss |
| Schwab: Brennan over Meissner | ≥5% | −0.1308 SG | 48.48% → 46.85% | Loss |
| Schwab: Ewart over Garnett | ≥5% | +0.0663 SG | 58.21% → 58.96% | Win |
| U.S. Open: Brennan over Fox | ≥5% | −0.2545 SG | 54.75% → 52.24% | Win |
| U.S. Open: Woodland over Hisatsune | ≥5% | +0.0715 SG | 58.30% → 59.00% | Push |

“Before” and “after” mean the two accounting/dew rules applied to the same archived inputs and prices. Schwab actually ran before the change; the U.S. Open and Travelers ran afterward. Results are shown for context, not used to estimate probabilities or select the rule.

Across the matched sample, all six nonzero skill-gap changes were in ≥5%. **None of the seven <5% matchups had a direct skill change.** At Travelers the largest estimated total probability movement was only 0.038 pp. Consequently, this correction cannot explain those marginal-edge losses through the direct R1-undo mechanism. The sample is too small and incomplete to generalize that conclusion to every later event.

The [full matchup table](permanent_data/research/r3_june_transition_20260909/matchup_probability_sensitivity.csv) includes both probability estimates, isolated skill/dew sensitivities, original edges, and conditional EV at the same recorded price. No hypothetical strategy P&L is claimed: the archives do not recover the entire eligible universe or alternative attainable fills.

## Original Python versus original Rust

Built the Rust source from `e61f07a` offline into an isolated research directory. Tested the actual single-round entrypoints immediately before and after that commit, using the archived June Memorial kernel fixture. Both received identical category means, course-adjusted standard deviations, skew, correlation, skill, and weather arrays. A common scoring baseline of 72 was specified for this controlled test; this is not a Memorial bet-time replay.

- 72 players × 200,000 simulations per implementation; 36 predefined disjoint matchup pairs.
- Category means: **exact agreement**.
- Largest player mean-score difference: **0.024 strokes**, at most **2.61 combined Monte Carlo standard errors**.
- Largest tested matchup probability difference: **0.278 pp**, at most **1.79 combined standard errors**.
- Deterministic checks for missing player scores, missing weather, course-average fallback, half-even rounding, and category/score clipping: **passed**.

The RNG implementations differ, so equal seeds do not imply identical draws. These results are consistent with sampling noise and provide no evidence that the initial Rust port systematically changed probabilities. They do not test every later optimization or the multi-round tournament cascade. See [parity results](permanent_data/research/r3_june_transition_20260909/kernel_parity_summary.json).

## Decision and next work

**Keep the correction; do not roll back Rust on this evidence.** The timing of the downswing remains an observation, not proof that the June transition caused it. This audit weakens a specific mechanical explanation; it does not prove the downswing was entirely luck.

Next, prioritize the **APP forecast-error investigation within ≥5% R3 and Balanced APP exposures**. Separate pre-event category expectations from in-week changes, then compare early and later errors on the backed player and opponent. The season analysis already found the larger APP/PUTT deterioration inside ≥5%; this test can distinguish early favorable outcomes fading from persistent forecast bias. Treat missing historical expected-category inputs as missing rather than substituting later ratings.

Promote immutable bet-time snapshot capture from medium to high priority. Exact future replay needs effective runtime config, active field, player category parameters, live components, forecast/weather arrays, code/kernel versions, simulation settings, raw probability and push probability, unrounded edge, and both market prices. Merely archiving a prediction CSV is insufficient. Snapshot implementation and production model changes remain proposed.

Still unresolved: historical Sheet override effects; full field-filter effects; exact Memorial/Canada bet-time probabilities; later Rust changes; and calibration over the rest of the season. Hold the current archived field and weather factors fixed when interpreting this report.

## Reproducibility

Research directory: `permanent_data/research/r3_june_transition_20260909/`.

1. `prepare.py` recovers locally available Git blobs without checkout or external writes.
2. `replay_components.py` executes only recovered pure totals functions; checks accounting, persisted R1 components, active skill targets, and expected-SG targets. Outputs player and matchup component tables, including the secondary archives.
3. `probability_sensitivity.py` computes the explicitly approximate archived-distribution comparison and checks mass, bounds, zero-shift identity, and direction of movement.
4. Build the isolated kernel with `cargo build --release --offline --lib --manifest-path permanent_data/research/r3_june_transition_20260909/legacy_rust/Cargo.toml`, then run `check_legacy_kernel.py` for the original-entrypoint comparison.

No production files, installed simulation module, Sheet settings, bets, or live services were changed.

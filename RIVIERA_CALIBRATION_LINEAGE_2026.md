# Riviera historical calibration: lineage and reconstruction results

2026-09-09. The next bounded step is complete: 14 input/processing layers cataloged for each of the three Riviera editions, source dates reconciled, and isolated course-calibration controls run. **Proceed to a past-only normalization refit before the broader chronological study.** No target-event performance or betting returns were evaluated.

Follow-up completed: [Normalization rebuild, dependent inputs and R1 accuracy results](RIVIERA_NORMALIZATION_REBUILD_RESULTS_2026.md). The controlled comparison finds no material accuracy gain. The lineage findings below are preserved as the earlier audit record.

## A concrete calendar correction

The shot archive's year is sometimes a season year; DG's round table uses calendar year. Matching those numbers directly misdates some fall editions and drops others. Across the 253 source events represented in the inspected player-round/shot-model exports, the reviewed bridge resolves all 253: 25 editions move to the preceding calendar year and three require a changed event ID.

The three ID changes are fall 2023 Sanderson Farms, RSM and Hero. DG labels these editions with a `#2` suffix. They belong to fall 2023, not fall 2022. Consequently they cannot enter the February 2023 Riviera forecast. Event names generate candidate matches only; acceptance requires matching observed player-round membership. The predeclared floor was 90% source membership and a 10-percentage-point lead over an alternate edition. All accepted matches exceed those gates, and no two source editions resolve to the same DG edition. Both sides use the repository's name replacements; no fuzzy player matching is used.

All 1,005 previously inspected shot-model/round metadata records now have a resolved event calendar. This addresses the identity/calendar mechanism behind the preflight's missing-date exclusions; it does not certify raw-shot completeness or independently archived pricing timestamps. DG ingestion derives round dates from event completion. The new research path additionally requires the whole source event to end before the forecast cutoff. No overlapping event was found at the three pilot cutoffs; a synthetic overlap is rejected.

The correction has measurable input effects. Holding shot SG values and the feature method fixed, the following shows the 95th percentile of absolute relative change in **shot-derived category variance**, before the final overlay, course multipliers and simulation:

| Riviera edition | Players compared per category | OTT | APP | ARG | Putting |
|---|---:|---:|---:|---:|---:|
| 2023 | 125 | 17.55% | 8.94% | 19.18% | 6.84% |
| 2024 | 70 | 9.64% | 4.19% | 10.70% | 4.86% |
| 2026 | 72 | 0.057% | 0.031% | 0.042% | 0.023% |

For 2023, APP variance changes by more than 5% for 32 of 125 compared players; putting does so for 26. These are input changes, not demonstrated improvements in accuracy. Older misdated history has much less weight by 2026. The tested calendar correction therefore supplies little evidence for an explanation of this season's downswing, while materially affecting an older-edition study. The repaired bridge is confined to research outputs; production loaders were not edited.

## The normalization issue is confirmed

This is more than a missing timestamp. Every post-cut PGA row in the three pilot fields' prior histories is associated with the `2026-07-13` uplift model: 14,101 rows for the 2023 field, 9,640 for 2024 and 10,714 for 2026. These overlapping history counts must not be added as independent observations.

The model file records training years 2021–2026. Its fitting script constructs a survivor-ability target from observations within **540 days before and after** each training event, excluding that event. Excluding the event itself does not make that fit chronological for an earlier prediction cutoff.

I verified the stored transformations numerically: adjusted category SG equals raw category SG plus its allocation of stored field skill, and the post-cut versus pre-cut field-skill difference matches the recorded uplift to floating-point precision. Thus the retrospective calibration is present in the histories consumed by the player/category and course calculations, rather than merely sitting in an unused file.

This does not mean the July data fixes were wrong. Deterministic corrections to identities, parsing and arithmetic can be applied when reconstructing a specified method. A parameter estimated using later observations requires an earlier-data refit for a chronological claim. We have not measured how much the uplift refit changes variance or probability accuracy.

## What can already be rebuilt reliably

The course builder reproduced exactly on isolated, prior-only history copies for all three editions. Extreme synthetic future category values changed neither its multipliers nor skew; perturbing earlier course observations changed them, confirming the test was sensitive.

| Forecast cutoff | Riviera history used | OTT SD multiplier | APP | ARG | Putting |
|---|---|---:|---:|---:|---:|
| 2023-02-16 | 2019–2022 | 0.91 | 0.96 | 1.19 | 1.02 |
| 2024-02-15 | 2019–2023 | 0.92 | 0.97 | 1.20 | 1.02 |
| 2026-02-19 | 2019–2024 | 0.91 | 0.95 | 1.21 | 1.01 |

These are reproducibility outputs using today's normalized historical values, not certified chronological parameters or evidence of forecast success. Riviera has no 2025 edition because that Genesis used Torrey Pines. Separate positive-definite, prior-only category-correlation references were also generated; they are not claimed to reproduce the production CSV's unbound estimation recipe.

The audit resolves prior-event source IDs for 126/129, 70/70 and 72/72 field members. Dated base-skill estimates exist for 268 of 271 field-edition rows; three missing 2023 estimates remain explicit. Skill dates alone do not establish the training history of the producer's external weight arrays or its normalized SG inputs.

## Next work, in dependency order

1. **Rebuild normalization at each cutoff.** Bind the field-strength source, then refit the cut-uplift model with every training observation before the cutoff. To retain its existing full ±540-day target definition, admit only training events whose entire window ends before the cutoff. A shortened-window target would be a separately declared method. Rebuild the skill inputs used by that regression without relying on the same retrospectively normalized estimates.
2. **Rebuild dependent player/course parameters.** Use the verified course cutoff path and repaired event bridge. Bind skill weights, category correlation and empirical surface/lie calibration to their fitting sources. Preserve current choices as a clearly labeled fixed-method retrospective comparator; do not optimize them on the Riviera target outcomes. Completed earlier events may legitimately supply their own putting calibration for later forecasts.
3. **Resolve remaining field coverage and finish fidelity before widening claims.** Keep the same means and player sets across variance arms. Full tournament/finish claims still need active weekly dependence and final-output production fidelity.

The shot-variance loader consumes `sg_raw`; its separate per-hole field-adjusted shot SG is not used in this branch. Likewise the frozen APP directional mean coefficient is not part of this tested variance builder. The lineage catalog marks these exclusions so they are not mistaken for active causes.

Evidence is in `permanent_data/research/riviera_calibration_lineage_20260909/`: the 42-row calibration catalog, event bridge and competing matches, player-level input changes, normalization identities, isolated databases, parameter hashes and verification receipts. Source hashes agree before and after the audit. All scoped checks passed; strict chronological readiness remains false. No production or shared data was changed.

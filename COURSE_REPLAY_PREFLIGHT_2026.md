# Historical course replay: preflight results

2026-09-09. The coverage matrix and bounded fidelity/precision pilot are complete. **Continue with calibration-provenance work; do not yet expand to a broad, strictly chronological course-performance study.** No historical outcomes were scored or ranked in this preflight.

Follow-up completed: [Riviera calibration lineage and reconstruction](RIVIERA_CALIBRATION_LINEAGE_2026.md). It resolves the inspected shot-event calendar, verifies course cutoff refits and identifies the historical normalization layer that needs a past-only refit. The results below remain the original preflight record.

## Coverage

The local metadata inventory covers 177 PGA events from January 2023 through August 2026, producing 200 event/physical-course rows across 82 course IDs. Of those courses, **42 have at least three editions**. Prior observations extend to 2019 for round data and 2020 for shot data.

Eighty course-edition rows across 26 courses meet the deliberately limited input screen: single-course event, at least three prior DG course editions, and at least 90% of observed R1 players having 12 prior category-shot rounds in each category, with at least two valid shots per category-round for variance estimation. This is a data-coverage screen, not certification of model chronology or production fidelity. Counts are conservative exact-name/source-ID joins; the pilot's reviewed aliases have not been silently propagated across the inventory.

125 existing prepared event bundles passed their stored checksums. They reconstruct actual starter rosters and contain model inputs, but use retrospective parameters and reconstructed conditions. The other 52 inventory events need preparation. Observed R1 membership and recorded tee times are not proof of the field or tee times known at an original pricing timestamp.

The coverage-only deterministic selection chose **Riviera in 2023, 2024 and 2026** as the repeat-course input pilot and **the 2023 American Express** as the routing check. The 2025 Genesis is correctly assigned to Torrey Pines South, not folded into Riviera. The 2023 Riviera edition has no existing verified prepared bundle, and only 126 of its 129 observed R1 players have a dated skill row; missing historical priors need explicit treatment.

Files: [course-edition matrix](permanent_data/research/course_replay_preflight_20260909/course_editions.csv), [course summary](permanent_data/research/course_replay_preflight_20260909/courses.csv), [selected pilot](permanent_data/research/course_replay_preflight_20260909/historical_pilot.csv).

## Fidelity and cutoff checks

| Check | Result | Claim supported |
|---|---|---|
| Rust primitives versus frozen Python oracle | Pass: rounding, ranks, skew transform and category sums | Low-level arithmetic/routing agreement |
| Single-round Rust versus Python reference | Pass: deterministic category means exact; score distributions within registered 5-SE gates | Raw single-round implementation parity |
| BMW raw tournament cascade | Pass: Python reproduces stored production checksum; Rust distribution gates pass | Raw, latent-off fixture fidelity |
| Memorial against current reference/kernel | Initial failure: checksum mismatch and four-versus-five coefficient interface | Old fixture cannot be mixed with current code |
| Memorial against its original June 9 reference | Pass: exact stored production checksum recovered | Original fixture is sound when bound to its original version |
| Historical input cutoff mutation | Pass on 12 Riviera 2024 players; extreme post-cutoff category/shot rows leave mean, both SD arrays and skew exactly unchanged | Tested pure input paths respect observation cutoffs |
| Cutoff negative controls | Pass: moving the same corruption before the cutoff changes the corresponding SD arrays | Mutation test is sensitive, rather than vacuously passing |
| American Express routing | Pass: all 537 player-round/layout routes resolve; ten round/layout groups map unambiguously to three physical courses | One event's metadata bridge, not global geometry stability or forecast-time tee routing |

The routing test first failed because raw-shot display names and category-SG canonical names use different formats. The corrected join uses provider player IDs, same-event canonical names, then DG identities. Three event-specific aliases are documented: Seung-Yul Noh, Paul Haley II, and Augusto Nunez (whose DG surname encoding is corrupted). No fuzzy matching is used. Earlier failed artifacts and source versions are retained.

The cutoff test does **not** certify the fitting history of supplied calibration constants, course multipliers, correlation matrix or base skills. Those require separate provenance/refit checks.

## Precision pilot, including final round skew treatment

Before inspecting new outputs, the contract specified two existing fixtures, two seeds, 20k/80k/200k draws, practical error tolerances and a 200k cap. The current round skew correction was applied to each entire draw panel. Independent histogram/CDF recounts and fixed-seed repeatability passed.

Both fixtures fail the proposed tight per-player moment criteria at 20k. All six required comparisons pass at the upper range: between seeds at 200k and 80k→200k for each seed and fixture.

Across required comparisons, the largest P95 mean difference is 0.0253 strokes, P95 relative variance difference 1.44%, P95 score-CDF difference 0.160 percentage points, and maximum score-CDF difference 0.609 points. At 200k, the round top-up's largest P95 mean change is 0.00252 strokes and relative SD change 0.667%, within the declared guards.

This supports the tested **round-distribution** precision at the upper range. It does not establish precision for rare finish tails, Kelly/EV threshold decisions or profits. Nor does failure of a tighter per-player 20k criterion by itself invalidate the prior audit's pooled 34-event diagnostics.

Both frozen tournament fixtures have the week latent disabled and their stored production hashes precede the final total-skew correction. Current full tournament/live behavior with active weekly dependence and final postprocessing still lacks a bound final-output oracle in this pilot. The new round precision checks do not close that separate fidelity gate.

## Calibration provenance: the main unresolved gate

All 1,005 exported shot model/round records inspected resolve to local model metadata, and all identify the correct source event. Their putting curves use rounds from that source event. For an earlier event that was fully completed before the next prediction cutoff, this within-event calibration is not automatically future leakage. Four records allow their own target round into fitting; this also needs to be interpreted as historical feature processing, not a prospective forecast of that source round.

However, all 1,005 records carry the **`empirical_2026_07` surface/lie mapping**. The source code says the mapping was chosen by checking implied strokes-gained values on archived shots in July 2026. Its historical training cutoff is not established for a 2023 prediction. Processing old shots now is acceptable for a retrospective fixed-method diagnostic; it is insufficient evidence for a strictly chronological validation.

The shot-overlay weights/window, category correlation/skew calibration and dated base-skill estimates need the same distinction: current selected parameters versus values fitted only on earlier data. A row's historical date cannot establish the latter.

The inventory also excludes 11,402 aggregated player/round/category shot-stat rows whose event dates could not be mapped through the inspected DG course-bearing rows. These are **not 11,402 individual shots**. The calendar/ID gaps need an explicit source resolution before affected histories can be certified. Unresolved physical geometry changes and historical roster/conditions provenance remain visible in the matrix.

## Next bounded work

1. Build a calibration-lineage table for the selected Riviera editions: source observations, source-event completion, fitting cutoff and parameter hash for every learned layer. Resolve the excluded calendar keys relevant to those histories.
2. Define a chronological calibration/refit path where provenance is insufficient. Preserve today's fixed model as an explicitly retrospective comparison; do not relabel it as historically available.
3. Bind an active-latent cut-event fixture through final production postprocessing and independent payout accounting before making tournament/finish claims.
4. Only then expand eligible course editions. The matrix's zero `strict_walkforward_ready` rows mean the full gate has not been certified, not that the underlying course histories are unusable.

All scoped receipt and eligibility reconciliation checks passed. End-of-pilot dependency fingerprints are recorded, but do not prove that every dependency had a pre-run snapshot.

All work was local research. Production code, model settings, odds, accounts and shared services were unchanged by this work. See the [registered protocol](COURSE_REPLAY_ROBUSTNESS_PROTOCOL_2026.md) and `permanent_data/research/course_replay_preflight_20260909/` for contract, source hashes, logs, controls and precision receipts.

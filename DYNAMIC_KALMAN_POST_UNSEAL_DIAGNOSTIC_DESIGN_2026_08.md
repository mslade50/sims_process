# Dynamic Kalman uncertainty and weather diagnostic

Status: research-only, post-unseal exploratory design. This diagnostic cannot
promote a model or revise the frozen discovery/sealed conclusions.

## Questions

1. Does carrying the selected translated cross-tour Kalman posterior variance
   to the target date improve matchup probability calibration relative to the
   existing single global `sigma` mapper?
2. Does an otherwise identical dynamic state built from historically
   weather-neutralized PGA observations improve on its W0 sibling?

## Frozen state logic

- Preserve `q = 0.0005` SG-squared per day, a 20-round prior, R1-R2
  observations, translated cross-tour histories, event-batch cutoffs, and the
  selected W0 posterior mean.
- Recompute the scalar Kalman covariance after every observation and add
  `q * days_since_last_observation` between the final observation and the
  target event. Target-date aging changes variance only, never the W0 mean.
- Verify the replayed W0 mean against the pinned 38,409-row frozen prediction
  artifact before scoring any extension.

## Pair-specific uncertainty mapper

The predeclared heteroskedastic mapper is

`effective_sigma = sqrt(base_sigma^2 + alpha * (v_player_1 + v_player_2))`.

`base_sigma`, `alpha >= 0`, and the tie threshold are fitted with event-equal
multinomial log loss on 2020-2022 only and frozen before any 2023-2026 score is
computed. A nested `alpha = 0` refit on the identical 2020-2022 menu separates
the posterior-variance contribution from a mere calibration-window change.
The original 2021-2022 global mapper remains a separate exact-reproduction
arm.

This diagnostic treats the two scalar player filters as independent, so pair
variance is `v1 + v2`. It does not estimate covariance induced by common field
adjustments or tour translations; that limitation must remain explicit.

## Dynamic W1 sibling

- Only completed rounds strictly before the target event can enter state.
- PGA observations with an exact player/year/event/round/date/course join and
  finite centered wind and dew covariates are replaced by
  `W0 + beta_wind * delta_wind + beta_dew * delta_dew`.
- Positive wind delta means the golfer faced worse-than-field wind; the plus
  sign removes the expected weather penalty from observed SG.
- Non-PGA histories and unmatched/incomplete PGA weather rows remain W0.
- Weather coefficients are the existing expanding coefficients whose fit
  maximum must be strictly before the target. No `weather_excluded`, target
  round weather, future weather, forecast, closing price, or 2023+ outcome is
  read during fitting.
- W1 tour translation and PGA observation variance use the same target-as-of
  weather arm; all other mechanics remain identical.

## Evaluation

- Exact normalized 72-hole and R1 matchup menus at BetOnline, Betcris, and
  Pinnacle.
- Calibration: canonical resolved pairs, event-equal log loss and Brier score.
- Money: open and close, each fixed book and best-of-three, 3% EV gate, flat
  1u and production risk-to-win sizing, existing settlement rules.
- Samples: 2020-2022 calibration/in-sample diagnostic, sealed
  2023-01-01 through 2026-08-22, and exact last-three-year
  2023-08-22 through 2026-08-22.
- Comparisons use identical quote menus and 10,000 whole-event bootstrap draws.
- The current strict-prior control and original W0/global Kalman arm are
  included so the extension cannot hide the original negative benchmark.

## Fail-closed checks

- Pinned hashes/byte sizes for upstream predictions, snapshots, quotes, round
  panel, coefficients, manifests, and all-tour database.
- Fresh output directory; no overwrite.
- Unique player-event predictions and exact quote-menu equality across arms.
- Strict source-round and coefficient chronology.
- Exact W0 posterior-mean replay and exact original logical-probability replay.
- Book-invariant logical probabilities, valid American odds, complementary
  outcomes, no ungraded selected bets, and exact reproduction of original W0
  and strict-control aggregate portfolios.
- Final manifest fingerprints every output and explicitly states
  `research_only=true`, `post_unseal_exploratory=true`, and
  `production_files_changed=false`.

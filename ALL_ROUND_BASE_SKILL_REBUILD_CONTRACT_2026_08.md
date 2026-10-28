# All-Round Base-Skill Rebuild Contract

Status: frozen before 2024-2026 outcome evaluation. Research only; no production
files or consumers may be changed by this experiment.

## Question

Can a pre-event golfer-skill model trained on every available R1-R4 round in
2023 improve on the current production/control architecture when it is frozen
and applied to 2024-2026 betting markets?

## Time boundary

- Pre-2023 observations may be used only to construct strictly prior player
  histories and frozen exogenous weather transforms.
- Calendar 2023 is the sole coefficient-training, feature/hyperparameter
  selection, probability-calibration, course-fit-lambda, live-update-lambda,
  and betting-rule-selection period.
- Calendar 2024 through the final cached 2026 event is the locked evaluation
  period.
- Candidate predictions used for 2023 selection must be chronological and
  out-of-event: no target event or later 2023 event may train the prediction
  used for that event. Events sharing a start date are one batch.
- After selection is written and hashed, only the chosen specification is fit
  on all eligible 2023 rows. No coefficient, imputer, scaler, probability map,
  or downstream lambda may be refit from 2024-2026 outcomes.
- Strictly prior player histories may naturally update during 2024-2026, as
  they would have operationally.

The 2024-2026 archive was viewed by an earlier, incorrectly targeted study, so
this is a newly locked rerun rather than a claim of never-before-seen evidence.
A production change would still require prospective shadow validation.

## Target arms

All arms use DataGolf field-adjusted total strokes gained and require at least
two observed rounds for an event-level target. One-round WD/DQ cases are kept
only in coverage diagnostics.

1. **T0 RAW-OBS4**: player-event mean of every observed R1-R4 adjusted-SG
   round. Player-event precision weight begins as observed-round count divided
   by four, then is normalized so every training event contributes equal total
   mass.
2. **T1 SURVCOMP-OBS4**: the same mean after subtracting an explicit
   survivor-composition delta from each event-round. The delta is the
   difference between the mean strict-prior player anchor among golfers
   observed in that round and the corresponding R1 field mean. The anchor is
   an EWM40 of strictly prior R1-R2 adjusted SG, shrunk toward zero by a
   20-round-equivalent prior. This arm does not use the legacy skill model.
3. **T2 MAR-ROUND4**: one row per observed R1-R4 round with player-event
   predictors repeated and event-round nuisance intercepts removed during
   fitting. Every observed row is retained and event-rounds receive equal total
   weight. This is the primary
   missing-at-random likelihood arm; T0 and T1 are transparent sensitivity
   arms.

The retired `post_mean - pre_mean` weekend correction is prohibited. It mostly
measured regression to the mean after making the cut and added roughly 1.2 SG
to weekend observations.

## Weather arms

- **W0** uses unneutralized all-round histories and the corresponding adjusted
  SG target.
- **W1** uses the existing strictly prior weather-neutral all-round histories.
  Target rounds are neutralized with the frozen pre-2021 wind/dew coefficient
  table before T0/T1/T2 construction. Positive wind delta remains player wind
  minus field wind; the neutralization coefficient is added to observed SG.
- W1 is a second-pass ablation. Realized target-event weather is used only to
  remove weather noise from a training label, never as a pre-event feature.

## Predictor boundary

Challengers may use only pre-event, point-in-time columns. They may not use
legacy skill or contemporaneous betting odds as features.

Frozen feature families:

- `T`: all-round DataGolf total-SG EWM/rolling/all-history levels and
  reliability counts.
- `T_APP`: `T` plus all-round APP levels and reliability.
- `T_BALL`: `T` plus all-round OTT and APP levels and reliability.
- `T_FULL`: `T` plus all-round OTT/APP/ARG/PUTT levels and reliability.
- `T_IMG`: `T` plus clean point-in-time IMG total-SG histories and IMG
  coverage/reliability context.
- `T_BALL_IMG`: `T_BALL` plus clean IMG total/OTT/APP histories and context.
- `T_FULL_IMG`: `T_FULL` plus clean IMG total/all-category histories and
  context.

Common context is limited to strictly prior history volume/recency, IMG
coverage, missing-recency flags, and prior-event withdrawal. Cached player age
is excluded because the 2023-only design audit found impossible negative ages.
Missing values are median-imputed inside each 2023 training batch and
continuous columns are standardized inside the batch.

Frozen selection-eligible menu:

- T2/W0 ridge: all seven feature families at alpha 100, 1,000, and 3,000
  (21 candidates).
- T2/W1 feature-only weather pass: `T`, `T_BALL`, and `T_FULL` at ridge alpha
  1,000 (three candidates). The all-round W1 target is rebuilt here; the cached
  R1-R2 neutral label is prohibited.
- T2/W0 robust sensitivity: Huber IRLS, epsilon 1.35 and ridge penalty 1,000,
  on `T`, `T_BALL`, and `T_FULL` (three candidates).
- T1/W0 survivor-corrected sensitivity: ridge alpha 1,000 on `T`, `T_BALL`,
  `T_FULL`, and `T_FULL_IMG` (four candidates).
- T1/W1 sensitivity: ridge alpha 1,000 on `T`, `T_BALL`, and `T_FULL` (three
  candidates).

T0/W0 ridge-alpha-1,000 versions of `T` and `T_FULL` are nonselectable raw-label
sanity checks. Constrained histogram-boosting and elastic-net arms may be
reported on 2023 as nonselectable diagnostics but are not allowed to expand the
locked 2024-2026 menu. Any arm that cannot satisfy its declared columns or
chronology is rejected, not silently altered.

## 2023 model selection

- January-March 2023 is coefficient warm-up. Beginning in April, each distinct
  event-start batch is predicted from earlier 2023 completed events only.
  April-June predictions fit probability maps; July-December predictions select
  the architecture. Q-school on 2023-12-14 has no adjusted-SG target and is
  excluded from target scoring.
- Final refitting on all 2023 occurs only after selection.
- Preserved downstream/course-fit multiplier menu: 0, 0.5, 1.0.
- A single common R2-R4 live cascade multiplier menu: 0, 0.5, 1.0. R1 has no
  completed-round update. Using one common multiplier avoids three additional
  small-sample searches.
- The current control remains the exact cached pre-event/cascade prediction.
- Each candidate and market receives a symmetric probability scale fitted from
  April-June chronological OOF pairs, then frozen for July-December selection.
  The final map is refit from all April-December chronological OOF pairs and
  frozen for 2024-2026.
- Candidate selection gives equal weight to five markets: 72h, R1, R2, R3,
  and R4. Within each market, 70% of rank weight comes from flat-stake betting
  return at the frozen 3% theoretical-edge gate and 30% from proper-score rank.
  The final winner maximizes the mean of those five market ranks. Log loss is
  the primary proper score and Brier is reported.
- If a candidate has worse aggregate 2023 log loss than control and worse
  aggregate 2023 P&L than control, it is ineligible regardless of rank.

## Betting evaluation

- Books: BetOnline, Betcris, and Pinnacle only.
- One canonical side per logical matchup; if books overlap, take the best
  available selected-side price across the three.
- Closing price is primary. Opening price repeats the frozen analysis as a
  sensitivity and never participates in selection.
- Primary gate: model EV at least 3%. Report 0-3%, 3-5%, 5-10%, and greater
  than 10% theoretical-edge buckets without changing the gate after seeing
  evaluation outcomes.
- Report flat one-unit risk, risk-to-win-one-unit, and fractional Kelly. Flat
  risk is the primary model comparison.
- Report by market, year, selected book, skill bucket, event/week, and golfer.
- Paired candidate-control comparisons use exact common quote intersections.
- Confidence intervals resample whole events with 10,000 deterministic draws.

## Promotion rule

This experiment can reject a challenger. It cannot directly change production.
A promotion recommendation requires positive aggregate paired 2024-2026 P&L,
no material proper-score regression, no single market responsible for the
entire gain, chronology/integrity checks passing, and a subsequent prospective
shadow period.

# Normalized donor chain: 50k evaluation contract

Status: archive-only research contract. It does not authorize production use,
deployment, pricing changes, or a nonzero shot-quality coefficient.

Implementation:
`archive/evaluate_normalized_chain_50k.py`.

## Question this test can answer

Given the same frozen pre-event production skill mean, does beta-zero
player-state spatial donor normalization produce a better *round-score shape*
than the raw strict-prior donor chain?

The comparison is:

1. `raw`: frozen strict-prior native donor chain;
2. `normalized`: donor player/setup spatial effects removed in native space and
   the target player's strict-prior spatial profile reapplied; and
3. shot-quality beta fixed at exactly zero in both arms.

This is not a test of an additional player-skill signal. Production skill is
the sole mean owner. A course-fit or shot-quality mean residual would require a
different, separately validated experiment.

## Three distinct gates

### 1. Engineering schema gate

A 5,000-particle artifact can validate:

- input shape and dtype;
- file/manifest hashes;
- counter-RNG and chunk contracts;
- exact raw/normalized particle-axis alignment;
- terminal/fail-closed behavior; and
- runtime and memory.

It cannot emit CRPS, log-score, PIT, tail, variance, H2H, “better,” “worse,” or
production-readiness claims.

An artifact whose manifest says
`production_mean_anchor_status=not_calibrated_mechanics_only` remains
mechanics-only even if it contains 50,000 or 100,000 particles.

### 2. Particle-precision gate

Predictive scoring requires at least 50,000 particles for every
event/player/round and both arms. The first 50,000 paths of a 100,000-path run
must be bit-identical to the standalone 50,000-path run.

Passing this gate controls Monte Carlo resolution. It does not make one player
or one event representative.

### 3. Heldout-panel decision gate

The default broad-panel floor is five heldout events and 100 primary
player-round cases. Results below that floor may be reported as an anchored,
high-particle pilot, but the evaluator sets `decision_claim_permitted=false`.

## Generator-time mean contract

The evaluator refuses to fix a distribution after it is loaded.

Predictive scoring requires the producer to attest:

- `production_mean_anchor_status` is
  `generator_time_shared_production_skill_anchor_v1`;
- owner: `production_skill`;
- policy: `shared_production_skill_mean_pre_persistence_v1`;
- stage: `generator_before_persistence`;
- raw and normalized have the same target;
- no arm-specific post-hoc location adjustment occurred; and
- the evaluator applies no location adjustment.

At event/player/round grain:

`production_mean_strokes = production_field_mean_strokes - production_skill_sg`

The case table must record both generator expected means, and each must equal
that shared production target. Persisted samples are then scored unchanged.
Sample mean errors and the paired normalized-minus-raw mean are audited using
their Monte Carlo standard errors. A failed audit blocks a decision claim; it
does not trigger recentering.

No latent mixture, support translation, exponential location tilt, variance
matching, or temperature transformation is permitted inside the evaluator.

## Particle bundle

Each versioned player bundle contains:

- `manifest.json`;
- `round_totals_int16.npz`;
- `arm_names = [raw, normalized]`;
- `round_totals`, shape `[2, 4, N]`, in physical integer strokes;
- `round_numbers = [1, 2, 3, 4]`;
- one `event_key` and one `player_id`; and
- explicit `particle_ids` or `particle_ids_by_arm` equal to `0..N-1`.

Both arms must have exactly the same explicit particle axis. An implicit axis
is accepted for engineering inspection but not predictive scoring.

The manifest must attest:

- research-only, production untouched;
- exactly `raw` and `normalized` arms;
- `player_state_residualized` mode;
- beta exactly zero;
- one rigid alignment;
- hierarchy/base mass multiplied by a mean-one model weight;
- arm and chunk excluded from the counter key;
- exact 50k/100k prefix behavior;
- the closed shared tail policy, including counter-prefix coverage through
  transition index 17;
- no survivor filtering;
- unsupported states fail closed;
- identical source-code hashes at run start/end; and
- the NPZ byte hash.

## Physical-score tail policy

`archive/hole_sim_tail_policy.py` is the single policy owner for the full
chain, streaming kernel, 50k runner, and evaluator. Its hard rules are:

- at most 18 continuation transitions, indexed `0..17`;
- no physical hole-score cap, score clipping, or terminal-by-cap;
- `HIGH_SCORE_DIAGNOSTIC_STROKES=15` is reporting-only;
- a path still active after transition 17 becomes
  `transition_loop_unresolved` and fails the whole run;
- survivor filtering and rerolls are forbidden;
- the legacy status `max_hole_strokes_unresolved` remains readable for frozen
  diagnostics but cannot be emitted by the current policy; and
- int16 overflow or a round total outside the frozen `40..140` lattice fails
  closed before persistence.

The 5k gate also performs the bounded raw Travelers R3 H13 regressions for
particle IDs `27481`, `35295`, and `48797`. Their exact scores are `14`, `13`,
and `14`, respectively, both batched and one particle at a time.

### Legacy Markov simulator exemption and call graph

`archive/hole_sim_markov.py::MarkovHoleModel.simulate_hole` retains its
historical `max_score=12` cap for that file's standalone legacy research CLI.
It is explicitly outside the normalized-chain admission graph and is not an
exception to the shared tail policy. Static call-site audit shows that only
`hole_sim_markov.py` itself calls that method.

The admitted paths reuse Markov donor data and lookup machinery, but never its
standalone hole loop:

```text
legacy standalone only
  hole_sim_markov.run_evaluation
    -> MarkovHoleModel.simulate_hole(max_score=12)

scalar normalized-chain parity/current precompute
  hole_sim_normalized_atom_precompute._run_terminal_parity
    -> hole_sim_full_chain._continue_case_arm
      -> PopulationTransitionSampler.sample
        -> MarkovHoleModel._pool

streaming 5k/50k
  StreamingShotChain.simulate_hole
    -> BatchedPopulationTransitionSampler.sample_many
      -> preindexed MarkovHoleModel._pool / TransitionPool data
```

The current scalar and streaming hole loops therefore take their continuation
guard and score semantics only from `hole_sim_tail_policy.py`. The
`min_exact=12` argument used when constructing `MarkovHoleModel` is a minimum
donor-support threshold, not a transition or hole-score cap.

The frozen 128-particle precompute reference remains immutable v1 evidence and
is verified against its historical producer SHA separately. It does not attest
the new tail policy and is usable only for bounded legacy parity. New v2
precompute/5k manifests bind the current precompute SHA, the historical
reference-producer SHA and role, and the exact shared tail policy.

Changing this policy invalidates the prior artifact chain. Admission requires,
in order: a newly generated 128/5k artifact carrying the exact policy, an
independent review of that artifact, a new v4 approval lock binding the 5k
manifest and all source/dependency hashes, a passing 50k dry-run admission,
and only then a newly authorized 50k execution. Prior failed-run diagnostics
and historical artifacts remain immutable evidence, not admissible inputs.

The required transform order is:

1. donor-native strict-prior normalization;
2. exactly one rigid neutral-endpoint alignment;
3. beta-zero target-profile reapplication; and
4. physical lie/surface reclassification after the generated start exists.

## Pre-event case index

The case index contains one row for every frozen roster player and every round,
including simulated R3/R4 rows for players who later miss a cut. Required
fields are exposed by `contract_schema()` in the evaluator.

Key rules:

- roster, skill, and every simulation input cutoff are strictly before the
  event start;
- names are canonical lower-case;
- the case identifier is `event_key|source_player_id`;
- every case contains R1-R4 exactly once;
- generator expected means for both arms equal the production anchor; and
- the case roster exactly equals the bundle roster.

Observed truth is a separate, independently hashed post-event table. It is not
read from the simulator output.

## No survivor-selected primary panel

For a predeclared no-cut event, R1-R4 are primary.

For a cut-after-R2 event, R1-R2 are primary for the entire frozen pre-event
roster. R3/R4 may be shown only as
`survivor_conditional_secondary`; they never enter the primary verdict.

Missing primary truth is an error. Missing R3/R4 truth for a cut-event player
is expected and is not filled or silently filtered.

## Frozen common lattice

Every round distribution is represented on the same physical integer-stroke
lattice:

`40, 41, ..., 140`

The bounds and unit spacing are frozen before predictions and truth. A sample
or outcome outside the lattice fails closed. The evaluator never:

- infers support from either arm;
- includes the actual outcome merely to expand support;
- translates relative SG and physical strokes onto different phases;
- uses a 128-particle latent grid; or
- scores only the intersection of observed arm supports.

A Jeffreys pseudo-count of 0.5 per fixed cell creates one valid predictive PMF
for log score and CRPS. At 50,000 paths the total pseudo-mass is approximately
0.1%.

Primary one-round losses are:

- CRPS on the fixed unit-spaced lattice; and
- negative log probability of the realized integer score.

Lower is better. Deltas are always
`normalized minus raw`, paired on the identical heldout case roster. Both
case-weighted and event-weighted summaries are emitted.

## Calibration, tails, and variance

Per arm/case the evaluator records:

- lower, upper, midpoint, and deterministic randomized PIT;
- 50%, 80%, and 90% interval coverage and width;
- 5th, 10th, 25th, 50th, 75th, 90th, and 95th percentiles;
- probability of scoring at least 3 or 5 strokes better/worse than the frozen
  production mean;
- predictive mean, variance, and standard deviation;
- predictive MSE around the production anchor; and
- realized squared error around that same anchor.

The randomized PIT uniform is hash-derived from the event, player, round, and
a frozen evaluator salt. It is identical across arms.

## H2H is secondary

The matchup roster must be frozen before the event independently of either
arm. No “market-like” pairs may be selected after examining candidate
probabilities or outcomes.

The default H2H calculation uses exact independent marginal convolution on the
fixed stroke lattice. This avoids pretending that equal integer particle IDs
from separately keyed player simulations encode a cross-player copula.

The evaluator reports win/tie/loss probabilities, three-way Brier and log
scores, and market-like two-way no-tie Brier/log scores for non-push outcomes.
All H2H deltas use the exact same pair roster and are explicitly secondary.

A joint-particle H2H method would require a separate, explicit event-level
copula/shared-shock attestation; it must not be inferred from particle IDs.

## Interpretation rules

- `<5k`: incomplete engineering scale.
- `5k to <50k`: schema/runtime/MC mechanics only.
- `>=50k` but unanchored: mechanics only.
- `>=50k`, anchored, narrow heldout panel: predictive pilot only.
- `>=50k`, anchored, broad heldout panel, all coherence gates green:
  decision-grade heldout comparison, still research-only until reviewed.

Particle count alone never promotes the normalized chain to production.

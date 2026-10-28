# Research production-category mean anchor for the shot chain

Status: implemented and unit-tested as an archive-only research boundary. It
is not imported by `new_sim.py`, the streaming chain, the full-chain prototype,
or any weekly pipeline.

## Decision

The future physical chain should own conditional shot shape and path topology,
while the production model remains the sole owner of each player's expected
OTT, APP, ARG and PUTT SG for a round. The correct bridge is a joint
transition-probability calibration. It is not a translation of completed
round scores.

The implementation is
`archive/hole_sim_transition_mean_anchor.py`. Its frozen contract is
`joint_recursive_production_category_mean_anchor_v1`.

The transferred quality priors come from the completed strictly-prior direct
reward-atom evaluation:

| Category | Conservative beta (natural parameter per SG/round) | Initial policy |
|---|---:|---|
| OTT | 0.90 | primary |
| APP | 0.20 | primary |
| ARG | 0.30 | primary |
| PUTT | 0.00 | transferred quality signal disabled |

The full evaluation covered 243 events and used one fixed green-value curve.
The complete-event postprocessor retained 179 primary event panels. Its pooled
proper-score deltas favored the selected tilts for OTT, APP and ARG. These
betas are transfer priors for conditional atom quality, not authorization to
change the production mean and not by themselves a production-enablement gate.
PUTT's positive secondary result is deliberately not transferred yet.

Evidence artifacts:

- `permanent_data/shot_quality_tilt_eval_20260820_v2/QUALITY_TILT_EVALUATION.md`
- `permanent_data/shot_quality_tilt_complete_events_20260821_v1/COMPLETE_EVENT_QUALITY_TILT.md`
- fixed reward-curve SHA-256:
  `aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93`

## Probability contract

For atom `i` in category `c`, let:

- `b_i` be its declared hierarchy/spatial base mass;
- `r_i` be its SG reward under one immutable reference value function;
- `a_i` be the strictly-prior donor ability being removed;
- `a*_c` be the target player's strictly-prior category ability;
- `beta_c` be the transferred coefficient above; and
- `lambda_c` be the fitted local production-mean correction.

First remove donor quality as a whole-atom exponential tilt:

```text
q_i = b_i exp(-beta_c a_i r_i)
```

Within each donor event/category/state group, `q_i` is rescaled so its total
equals that group's original `b_i` total. The implementation uses a
base-mass-weighted normalizer, so donor-event mixture mass is preserved even
when hierarchy weights are not uniform.

Then reapply target quality and the production correction:

```text
p_i(lambda) proportional to
    q_i exp((beta_c a*_c + lambda_c) r_i)
```

Every field of an atom moves together: endpoint, lie, hazard, penalty, cost,
terminal flag and reward. SG is never subtracted from an unchanged endpoint,
and a hole-out or penalty cannot remain in a separate untilted side channel.
The log-weight cap and fail-closed clipping policy are hashed inputs. If even
one donor or target atom would be clipped, `whole_atom_probabilities` raises;
that support cannot produce a calibrated or production-admissible result.

For a player-round, solve all four parameters together:

```text
E_CRN[category_SG_round(lambda)] = production_category_SG_round
```

The expectation is recursive. For example, changing APP mass changes how
often the chain enters ARG and PUTT, so APP lambda generally changes three
category means. Four independent scalar calibrations are therefore invalid.

## Mean ownership and double-counting guard

The exact production four-vector is the only final mean target. The
`beta * target_ability` term provides conditional transition shape; lambda is
solved after that term and after donor de-tilting. The calibrated chain output
already owns the requested mean. The following operations are forbidden:

- adding the production SG vector to the calibrated chain a second time;
- adding the raw-versus-normalized chain mean delta to production skill;
- translating integer round scores after simulation;
- separately recentering total SG after the four category equations pass; or
- fitting each category while holding its endogenous opportunity count fixed.

`MeanAnchorProblem` optionally accepts the production total only as an audit
assertion. It fails if that total differs from the sum of the four category
targets.

PUTT beta being zero does not force PUTT lambda to zero. It disables the
unvalidated transferred player-quality shape. A PUTT lambda may still be
needed to make the physical chain reproduce the already-authoritative
production PUTT mean.

## Common-random solver

`solve_production_category_mean_anchor` requires an expectation callback and
an immutable `CommonRandomContext`. Every callback evaluation must return the
same:

- counter/uniform lattice hash and simulation count;
- transition-support hash;
- fixed-reference reward hash; and
- structural category-support bounds;
- quality-weight config and parameter hashes; and
- zero donor and target clipped-log counts.

The solver calls the callback twice at every unique parameter point, including
every nonzero finite-difference, line-search, Broyden and fallback point. Any
bit-level discrepancy in the returned numeric contract fails closed; there is
no configurable determinism tolerance. It then tries, in order:

1. damped bounded Newton with a common-random finite-difference Jacobian;
2. projected Broyden updates; and
3. bounded nonlinear least squares.

Componentwise mean accuracy alone is not admission. The final point must also
have a full-rank four-by-four Jacobian and a condition number no larger than
the explicit `maximum_jacobian_condition_number`. These gates are evaluated
even when lambda zero already matches all four means. A rank-deficient or
ill-conditioned point is nonconverged and not production-admissible.

The solver reports rank, condition number, category opportunity means, active
lambda bounds, cross-category bound coupling, structural marginal bounds and
unresolved residuals. An outward local Newton direction at an active bound is
reported as `locally_bound_limited_unresolved` with
`local_solver_point_bound_limited` support evidence. That attribution is
useful, but it is only a statement about the final local linearization: a
non-monotone response can still have another root inside the declared lambda
bounds. It must never be called global infeasibility. A target outside a
category's structural min/max is a valid one-category infeasibility
certificate. A target inside all four marginal ranges is only a necessary
diagnostic; it is not proof that the joint recursive four-vector is
attainable. Joint feasibility still requires the coupled solve and all
admission gates. No output recentering hides a failure.

For the real chain, the preferred callback is Rao-Blackwellized: use the
fixed particle/state lattice but integrate each current transition support by
its probability rather than relying only on hard inverse-CDF selections. That
keeps the expectation smooth enough for Newton while still honoring
endogenous state visitation. A hard-draw 50k/100k callback is deterministic
under common random numbers, but its stepwise objective should be treated as a
fallback/verification callback rather than the primary derivative engine.

## Actual integration map

No existing file was changed for this research implementation. A later
integration should use these exact seams.

### 1. Freeze the production target before simulation

`new_sim.py` currently emits per-player realized simulation category means in
the `avg_expected_cat_sg_<event>.csv` block near its "Per-player expected SG
summaries" section. That is useful as an audit bridge, but it contains finite
100k Monte Carlo noise.

The production integration should instead export the deterministic
post-recenter, post-weather, post-live-update category mean vector before
random draws, keyed by:

```text
event_key, player_id, round_no,
target_ott_sg, target_app_sg, target_arg_sg, target_putt_sg,
target_total_sg, production_input_sha256
```

The category sum must equal the production total exactly. The anchor consumes
that frozen vector; it does not estimate a new marginal skill model.

### 2. Retain quality fields in every complete support atom

`archive/hole_sim_normalized_atom_precompute.py` currently establishes the
right mass seam in `AtomSupportProvider`: final spatial support mass is
`base_hierarchy_mass * model_weight`, normalized once. Its `SupportArrays`
does not yet retain quality fields.

For nonzero quality weighting, each TEE/APP support needs aligned numeric
arrays for:

```text
fixed_reference_reward, quality_reward_available,
donor_ability_sg, donor_quality_group_id,
base_hierarchy_mass, spatial_model_weight,
reward_reference_sha256, quality_input_sha256
```

The quality base mass is the already-spatially-reapplied mass. Donor quality
de-tilting is then applied once with `whole_atom_probabilities`.

Rows without a valid fixed-reference reward may remain geometry donors only
with zero quality exposure and separately reported mass; they cannot silently
pretend that observed SG was zero. APP hole-outs, hazards and linked penalties
must become explicit complete atoms before beta or lambda is nonzero. The
current beta-zero-only separate `holeout_probability` representation is not a
valid quality-tilt support.

### 3. Extend the numeric streaming cache without changing coordinates

`archive/hole_sim_normalized_chain_streaming.py` stores only atom IDs, points,
probability, CDF and holeout mass in `NumericSupport`. A research quality cache
should add the arrays above and rebuild probability/CDF for the candidate
lambda. Coordinates and surface classification do not change.

- TEE can usually evaluate reward when its fixed start and physical endpoint
  are materialized.
- APP must first rotate its deferred miss vector from the generated start,
  classify the physical endpoint/lie/hazard, and evaluate the same reference
  reward. A nonzero APP beta/lambda cannot be baked into the current
  state-agnostic miss CDF.
- The state grouping and quality CDF build must remain vectorized outside the
  per-particle Python loop.

### 4. Extend ARG/PUTT population supports coherently

`archive/hole_sim_markov.py::TransitionPool` already carries SG and a validity
flag, but it drops donor event/player identity. The exact batched sampler in
`archive/hole_sim_batched_population_transition.py::sample_many` returns the
transition but not its donor index or fixed reward.

A research quality version needs donor event/player/group identity,
strictly-prior donor ability, fixed-reference reward, selected donor index and
quality eligibility. Its nearest-64 distance mass remains the base mass;
quality de-tilt/reapplication/lambda multiplies that mass before inverse-CDF
selection. The existing exact hierarchy and donor ordering cannot change.

### 5. Accumulate category reward inside recursion

`StreamingHoleResult` currently reports strokes and terminal/state diagnostics
but no category reward. Add per-particle four-column category SG sums and four
opportunity counts. Update them at the exact transition that selected the
complete atom, including coherent penalty/hole-out behavior.

The expectation callback runs all 18 holes for one frozen player-round and
returns:

```text
mean_category_sg[4], mean_opportunities[4],
structural_support_lower_sg[4], structural_support_upper_sg[4],
crn_sha256, support_sha256, reward_reference_sha256,
quality_weight_config_sha256, quality_parameter_sha256,
clipped_donor_logs, clipped_target_logs
```

Those are the inputs to the joint solver. Completed scores are used only for
the telescoping audit, not to manufacture category residuals.

### 6. Cache the calibrated player-round parameters

The cache identity must include at least:

```text
event, physical course, player, round,
production target SHA, donor/profile/support SHA,
fixed reward SHA, beta-registry SHA,
geometry/classifier SHA, CRN contract and particle count,
quality log-weight limit and clipping-policy SHA,
solver contract/problem SHA, bounds, rank tolerance, conditioning threshold and mean tolerance
```

Persist lambda, effective slope, achieved/target vector, residual, Jacobian,
condition number, opportunities, clipping counts, support diagnostics and
solver status. Only a componentwise-matched, full-rank, well-conditioned,
zero-clipping, hash-consistent player-round can enter a decision simulation.

## Week and round latent effects are deferred

The implemented contract is explicit:

```text
player_week_and_round_latents_deferred_outside_transition_anchor
```

No shared player-week latent or extra round latent is applied inside the
current transition calibration. The target is the unconditional frozen
production player-round mean.

If a future shot simulator moves those latents inside the chain, it must solve
conditional targets, for example `mu_category(z_week, z_round)`, or calibrate a
validated lambda mapping over those nodes. It may not calibrate the
unconditional mean and then add a latent SG shift after absorption. Until that
conditional design is tested, the existing outer marginal/dependence adapter
remains the latent owner.

## Synthetic absorbing-chain proof

`test_hole_sim_transition_mean_anchor.py` contains an 18-hole absorbing model
with frozen exogenous course variants. APP outcomes change ARG and PUTT
opportunities; ARG outcomes further change PUTT opportunities. The callback
analytically integrates transition supports conditional on one common-random
lattice.

Using a hidden lambda of `(0.23, -0.34, 0.29, 0.17)`, the generated production
target and recovered result were:

| Category | Target SG | Achieved SG | Absolute residual |
|---|---:|---:|---:|
| OTT | 3.5951093920 | 3.5951093920 | 0.0 |
| APP | -0.9925049889 | -0.9925049889 | 1.2e-15 |
| ARG | 0.4231174703 | 0.4231174703 | 3.9e-14 |
| PUTT | 1.4991152212 | 1.4991152212 | 4.7e-15 |

Bounded Newton recovered the hidden vector to less than `3e-14`, with a
full-rank four-by-four Jacobian. Its off-diagonal terms were material; a
diagonal one-pass calibration failed the test. The suite also proves donor
group-mass preservation, every-point deterministic-CRN rejection,
componentwise-feasible cache preference, rank/conditioning admission at zero
residual, fail-closed clipping with hashed provenance, cross-category bound
attribution, explicit marginal-support diagnostics, production-total
consistency and latent deferral.

This is an engineering and identification gate, not a claim that real player
rounds have been calibrated or simulated. No broad run was launched.

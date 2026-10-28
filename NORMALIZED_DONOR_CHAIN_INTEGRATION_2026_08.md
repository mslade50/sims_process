# Normalized TEE/APP donor integration contract

Status: the archive-only precompute/parity harness is implemented and its
bounded Travelers gate passed. Production and the existing full chain remain
untouched. Shot-quality weighting remains disabled (`beta = 0`); the
normalized output is diagnostic, not promotion evidence.

## Decision

The normalized-donor experiment is feasible, but the existing static player
profile CSVs cannot be used to neutralize historical donors. The safe path is:

1. build a player/state snapshot strictly before every donor event;
2. neutralize that donor in its own dated setup frame;
3. rigid-align the neutral residual exactly once into the target event's latest
   verified-prior frame;
4. reapply the target player's dated profile once per player/setup/state;
5. classify the moved endpoint on target prior geometry; and
6. pass the resulting endpoint, lie, remaining distance, holeout and hazard
   state into the existing recursive full-hole mechanics.

The primary advancing comparison should retain the empirically stronger target
APP transport and treat target TEE transport as a sensitivity. The initial
normalized arms should compare event-centering with full player/state
residualization, because the existing endpoint study favored event-centering
more consistently while the complete spatial-chain study favored player APP
and did not favor player TEE.

## Files and ownership

Shared contracts that this design consumes, rather than redefines:

- `archive/hole_sim_field_normalization.py`: vectorized inverse/reapplication
  transforms, strict profile validation and atom-weight accounting.
- `archive/hole_sim_asof_spatial_profiles.py`: expanding, completed-event-only
  TEE/APP spatial profiles.
- `archive/hole_sim_state_foundations.py`: full-sequence causal state and
  penalty linkage.
- `archive/hole_sim_hazard_resolution.py`: endpoint-linked causal hazard
  resolution.
- `archive/hole_sim_full_chain.py`: recursive scoring and long round-particle
  contract.
- `archive/hole_sim_calibration_adapter.py`: production mean anchor, marginal
  variance control and shared player-week latent.

New research-only support:

- `archive/hole_sim_normalized_atom_precompute.py`: donor-native
  normalization, exactly-once alignment, contiguous target supports and the
  raw 128-particle equivalence harness.
- `archive/test_hole_sim_normalized_atom_precompute.py`: focused CDF,
  hierarchy-mass, target-weight and deferred-APP contract tests.
- `archive/hole_sim_normalized_chain_hook.py`: exact key builders, versioned
  beta gate, content-addressed cache key and vectorized stateless RNG.
- `archive/audit_hole_sim_normalization_inputs.py`: supplemental donor-coverage
  audit. The authoritative bridge remains
  `archive/audit_shot_field_normalization_inputs.py`.
- `permanent_data/hole_sim_normalized_donor_design/`: supplemental audit
  outputs. It is not a competing input authority.
- `permanent_data/hole_sim_normalized_atom_precompute_travelers_smoke/`:
  frozen Travelers 2026 parity artifacts and beta-zero diagnostic caches.

## What the current chain does, and where the hook belongs

The current research chain prepares a verified latest-prior 4 x 18 forecast,
aligns historical course donors into it, and then uses:

- `SpatialTeeSampler.sample` for par-4/5 tee shots and re-tees;
- `SpatialAppSampler.sample` for par-3 starts and every later state routed to
  APP;
- exact-physical-course population ARG/PUTT transitions;
- endpoint-linked hazard resolution; and
- integer stroke aggregation into player/round/particle paths.

The normalized hook replaces the support preparation inside the TEE and APP
samplers, not their state return contract and not the downstream recursion.
Each sampler must continue to return endpoint, lie, remaining distance,
holeout/status and penalty hooks for the same player/round/hole/particle axes.

One important boundary follows from the as-of contract: the already-aligned
`course_drives` and `course_app` returned by the present
`prepare_event_foundations` are too late in the pipeline to be neutralized.
The new preparation path may reuse its verified event scope, forecast,
foundations, rigid-transform helpers and target surfaces, but it must intercept
native donor atoms before `_align_drives_to_forecast` and
`_align_app_course_to_forecast`.

## Input audit

### Static spatial profiles

| Profile | Rows | Players | Training represented | Provenance issue |
|---|---:|---:|---|---|
| `player_tee_profiles.csv` | 1,246 | 1,246 source IDs | aggregate 2020-2025 | no cutoff, version or source hash |
| `player_app_profiles_structured.csv` | 19,290 | 1,261 source IDs | aggregate 2020-2025 | no cutoff, version or source hash |

SHA-256:

- TEE: `38fbe5eaa43555d0da830bee1d0bee4a7248dd6c60bc1f164cf0fadffdec7e73`
- APP: `09dda6beddddd934c27078946dfa68edc707d2fb823334f414f500bfc4a404ee`

These files are valid legacy 2026 target-transform references, but they leak
future spatial observations if used as profiles for a 2020-2025 donor. They
are therefore comparison inputs only, not authoritative inputs to the new
normalized chain. `hole_sim_asof_spatial_profiles.py` replaces them for both
donor and target snapshots.

The authoritative built snapshot is
`permanent_data/hole_sim_asof_spatial_profiles_v2/`, contract
`strict_prior_spatial_profiles_v1`: 5,800 TEE rows, 69,600 APP rows, 21 event
editions and zero cutoff violations. Its serialized and logical hashes match
the manifest. The Travelers integration below consumes this real bundle, not
the earlier schema-only smoke fixture.

Local ShotLink begins in 2020. A 2020 donor cannot have a locally fitted
strictly-prior player spatial history. The as-of builder correctly emits an
identity profile (`distance_ratio/scales = 1`, biases `= 0`) with explicit
availability/status flags; it must not drop those atoms or borrow future rows.

### Identity and pre-event skill

ShotLink `source_player_id` is not DataGolf `dg_id`; a direct numeric join has
negligible coverage. The authoritative path is:

`source_player_id -> yearly pga_<year>_player_rounds canonical name -> dg_id`

Ambiguous names fail closed. The authoritative bridge contract is
`strict_prior_skill_global_identity_alias_v2` and currently reports zero
cutoff violations. It permits explicit season-wide aliases only as stable
identity metadata; those aliases are never a predictor. Player skill remains
the latest stored row with `skill_date < event_start`.

The local database
`C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db` contains daily
`model_results2` history from 2018-12-30 through 2026-08-17, including
`sg_ott`, `sg_app`, `ag_arg`, `ag_putt`, `rel_dd`, `skill_est` and `std_dev`.
The consumer rule is the stored snapshot itself:

`model_results2.Date < event_start`

An equal-date row is not authoritative, even though its producer internally
uses rounds with `round_date < Date`.

Across the 18 Sony/Travelers/Wyndham donor editions from 2020-2025, the
supplemental shot-weighted audit found:

| Donor proxy | Shots | Unique identity/skill | Category SG | `rel_dd` |
|---|---:|---:|---:|---:|
| TEE | 107,975 | 96.37% | 92.31% | 92.24% |
| raw APP category | 143,650 | 96.29% | 92.20% | 92.13% |

The APP count is only a raw `sg_category == app` coverage proxy. Full-sequence
spatial/coherence filtering produces a different final atom count.

## Exact state and setup taxonomy

The normalized chain must use the architecture-owned condition columns:

`sg_category, state_key, setup_key`

TEE states:

- `sg_category = ott`
- `state_key = tee|par=4` or `tee|par=5`

APP states:

- `sg_category = app`
- start lies: `tee`, `fairway`, `rough`, `sand`, `recovery`, `other`
- four distance bands: `<=100`, `100-150`, `150-215`, `>215`
- `state_key = app|lie={lie}|band={band}`

Setup:

`{event_key}|course={course_id}|r={round_no}|h={hole_no}`

Here `course_id` is the foundation's audited physical-course geometry
fingerprint. Raw ShotLink layout IDs can vary across round/setup rows at one
venue and must not split the target setup cache.

The APP profile band and the transition kernel's 15-yard distance bucket are
different axes and both are required. The four-band state selects the dated
player modifier. The existing 15-yard bucket remains part of the direct APP
donor hierarchy/cache. Do not replace one with the other.

## Table contracts

### 1. Native donor atoms

One stable `atom_id` per physical causal shot, sorted by `shot_key`. Required
logical fields are:

- identity/provenance: donor `event_key`, `event_start`, `course_id`, season,
  player ID, `shot_key`, round, hole, par;
- condition: `sg_category`, `state_key`, donor `setup_key`;
- spatial state: native-frame `long_yd`, `lat_yd` in yards, start lie/distance,
  original endpoint lie/hazard, holeout flag;
- reward/accounting: fixed-reference `sg_raw`, causal penalty linkage and
  resolution keys; and
- setup provenance: native geometry/reference cutoff and frame version.

TEE coordinates are native local yards along/cross the donor setup's tee-pin
axis. APP residual coordinates are yards along/cross the generated
start-to-reference-pin axis. Absolute ShotLink geometry remains in feet at the
foundation/rigid-transform boundary; unit conversion must be explicit.

### 2. As-of profile snapshots

The builder emits the core profile fields, including:

`event_key, source_player_id, sg_category, state_key, profile_cutoff,
distance_ratio, long_bias_yd, lat_bias_yd, long_scale, lat_scale,
long_lat_corr, ability_sg, exposure_weight, normalization_source,
snapshot_kind, profile_sha256`

Every donor snapshot uses only events with `event_end < donor event_start`.
Same-date and overlapping events are excluded. Missing spatial history becomes
an explicit identity profile, not a missing row.

### 3. Setup materialization

`materialize_normalization_profiles` receives pre-outcome `atom_conditions`:

`event_key, sg_category, state_key, setup_key, reference_long_yd,
reference_lat_yd, reference_cutoff`

For donor normalization, both the profile and the setup/reference cutoff must
strictly predate that donor's event start. Earliest events therefore need dated
Tour/state prior centers. Never assign a 2020 donor a 2025 center and never
relabel a donor as the 2026 target to satisfy a merge.

### 4. Neutral aligned atom cache

The durable handoff preserves both donor and target setup identities:

- stable `atom_id`, `shot_key`, `donor_event_key`, `donor_setup_key`;
- target-frame `setup_key`, category/state and the APP 15-yard kernel bucket;
- `neutral_resid_long_yd`, `neutral_resid_lat_yd`;
- base hierarchy weight, `neutral_weight`, holeout mass/type;
- donor/profile/reference cutoffs and profile availability;
- `alignment_count = 1`, source/target frame versions and rigid-transform
  diagnostics; and
- normalization/config/profile/atom hashes.

After alignment, the donor setup key is retained only as provenance and the
condition setup key is changed to the matching target latest-prior setup.

### 5. Target atom cache

The target cache is keyed by:

`target event, target player, category, state, target setup, normalization
mode, donor atom hash, donor profile hash, target profile hash, normalization
config hash, coefficient-registry hash, frame version, classifier version`

TEE supports contain target `model_long_yd`, `model_lat_yd`, a deterministic
physical-classification audit, model weight/CDF, availability flags and the
target-reapplication provenance hash. The deterministic label is not sampled;
the runtime sampler still evaluates the complete surface-probability vector
and shared lie uniform at the moved endpoint.

APP supports contain only target-reapplied pin-relative miss vectors, base
mass/model weight/CDF and provenance. A generated APP start/direction varies
by particle, so absolute endpoint, lie, hazard and remaining distance are
computed only after rotating the selected vector from that generated start.
Par threes enter APP directly with start lie `tee`; par fours/fives enter TEE.
A nonzero APP quality tilt cannot be baked into this state-agnostic CDF because
its fixed-reference reward also depends on the generated start/end state.

## Transform order: exactly one safe order

1. Resolve the physical course and event calendar without held-out predictor
   outcomes.
2. Read a native donor causal atom and its own event/setup frame.
3. Join the donor player's latest unique identity/skill snapshot with stored
   date strictly before donor event start.
4. Fit/load the expanding spatial profile using completed events only.
5. Materialize the donor event's dated state/setup reference center.
6. Run `normalize_donor_points` in the native donor frame. For transformed
   modes, old endpoint lie/hazard labels are invalidated.
7. Apply the edition-to-target-latest-prior rigid transform exactly once to
   the neutral point/residual. Retain donor frame/setup as provenance and set
   `alignment_count = 1`.
8. Retag the condition to the target setup and join the target player's latest
   profile strictly before target event start.
9. Run target reapplication once per player/setup/state support, not inside the
   particle loop.
10. Reclassify the moved endpoint on the target latest-prior geometry and
    recompute remaining distance geometrically to its forecast pin.
11. Build the same TEE/APP support hierarchy using the same stable atom IDs and
    base mixture weights.
12. Sample and recurse through APP, endpoint-linked hazards, ARG and PUTT until
    terminal, then sum integer hole strokes to rounds. The shared computational
    guard is transitions `0..17`; there is no physical score cap or clipping.
    A still-active path fails the whole run as `transition_loop_unresolved`.
13. Apply the frozen production mean/variance adapter and shared player-week
    latent once.

For APP, pin-relative residual alignment is often the identity numerically,
but its provenance/order assertion remains mandatory.

Forbidden orders include:

- normalizing already target-aligned course rows;
- using latest-prior target centers to normalize older donors;
- applying a donor-to-target transform twice;
- copying a raw lie or penalty label to a moved endpoint; and
- applying a target transform separately for every particle.

## Lies, hazards, holeouts and penalties

Spatial transport changes where the ball finished, so the original endpoint
label is not an invariant:

- reclassify lie and hazard from the moved endpoint;
- recompute APP/ARG/PUTT routing from the new lie plus remaining distance;
- do not carry a donor's original `penalty_count` or stroke cost to a moved
  non-hazard endpoint;
- when a moved endpoint is a supported hazard, query the endpoint-linked
  causal resolver and resume from its linked resolution coordinates/lie;
- charge the causal physical shot once and then add the resolver's
  span-derived `penalty_stroke_cost`; and
- keep unsupported hazard endpoints fail-closed.

APP holeouts are a separate terminal atom/mass. At beta zero their mass and
shared holeout uniform must remain identical across raw and normalized arms.
Spatial transforms apply only to continuous non-holeout endpoints. A future
quality tilt must include terminal atoms coherently or remain disabled; it may
not change only the continuous support.

## Common-random-number contract

At beta zero, normalization changes coordinates but not donor identity or the
base mixture weight for a fixed support query. Therefore the same donor-index
uniform must select the same `atom_id` when raw and normalized arms reach the
same state/setup support. Once an upstream moved endpoint changes lie or
distance bucket, later APP supports may legitimately diverge; common random
numbers still couple those new conditional queries but cannot force the same
atom across different supports. This is stronger than merely using the same
seed without pretending the recursive paths remain identical.

Rules:

- sort every support by stable `atom_id/shot_key` before CDF construction;
- key uniforms by seed, event, player, round, hole, transition index, semantic
  stream, particle ID and optional draw index;
- exclude arm, normalization mode, worker and chunk from the key;
- use separate semantic streams for donor index, APP holeout, lie/surface,
  hazard, ARG, PUTT and any alias second draw;
- generate streams counter-wise even when a branch is inactive, so branch
  divergence cannot shift later randomness; and
- require particles `0..49,999` in a 100k run to match the 50k run bit for
  bit.

`hole_sim_normalized_chain_hook.counter_uniforms` implements the proposed
vectorized `blake2b + SplitMix64` counter contract. The current chain's
per-particle CRC32 string loop is deterministic but is not suitable for a
100k inner loop.

With nonzero beta, CDF weights legitimately differ, so equal uniforms no
longer guarantee equal selected atoms. They still provide the correct
monotone inverse-CDF coupling. This must be labeled a weaker CRN arm.

## Mean-skill and variance ownership

The integration must assign each effect one owner.

### Spatial style

- TEE `distance_ratio` owns driving-distance transport. Do not also shift TEE
  coordinates with `rel_dd`.
- APP long/lateral bias and scales own APP spatial style. Do not convert
  `sg_app` to an additional yardage shift.
- Player scales may alter topology, skew and which hazards/lies are reached;
  they are not a second marginal round-variance multiplier.

### Transition quality

- Default beta is exactly zero.
- Category skill can later enter only through a coherent whole-atom reward
  tilt from a versioned coefficient table.
- It may not be used as a coordinate shift and the core's historical
  placeholder `1.0` may never be used.
- Before enabling beta, quality ability should be residualized or audited
  against the spatial style variables, especially TEE distance ratio, so the
  same distance/style signal is not represented twice.

### Final score mean

The frozen production `target_sg` is the sole final round-mean owner. Raw
physical-chain means are removed by the discrete adapter, the integer support
is translated, and the residual exponential tilt targets
`round_par - target_sg`. Do not add the normalized chain's mean delta to
`target_sg`.

### Marginal variance

For the decision comparison, every spatial arm receives the exact matching
full-strength `category_shot` marginal round variance. Power tempering changes
mass on fixed integer support to hit that variance after the week mixture.
This isolates spatial topology/tails and avoids multiplying player spatial
scales by the production dispersion overlay a second time.

### Week dependence

The shared player-week latent is added once by the final adapter (currently SD
0.65 in the frozen research fixture). Do not add a separate week latent inside
shot transitions. The adapter preserves upstream round path ranks while
assigning its conditional integer PMFs, but it does not itself create a
course-wide latent shared across different players.

## Quality-beta gate

The available direct shot-quality calibration is still an exploratory,
disabled three-course artifact. Its current registry explicitly requires a
runtime default of zero, and the coherent direct-quality full-chain evaluator
has not completed the wider decision run. Numeric natural-parameter estimates
from earlier calibration snapshots are therefore not integration claims and
are intentionally omitted here.

This precompute/parity harness uses beta zero throughout. Enabling quality
later requires a completed direct-quality evaluator plus a versioned registry
with:

`coefficient_version, calibration_contract, sg_category, state_key,
beta_natural_per_sg_round, training_shots, training_cutoff, source_sha256,
enabled, shrinkage_status`

and a manifest with contract
`normalized_chain_quality_coefficients_v1`, matching table hash, coefficient
version, calibration contract and explicit research enablement. Resolution is
exact state first, explicit category wildcard second, otherwise zero. Missing
rows never fall back to `1.0`.

The existing `strict_prior_transition_quality_v1` calibration/disabled
registry is provenance for that future evaluation, not authorization to pass
nonzero coefficients into this hook. APP quality additionally requires reward
evaluation after the generated start-relative vector is rotated to a physical
endpoint, or a separately audited generated-state bucket approximation.

## 50k/100k execution architecture

The large run must move dataframe work out of the particle loop.
The 128 harness still groups active APP particles with a Python loop; this is
an accepted parity implementation only. Before 50k, encode normalized lie and
15-yard bucket arrays, group with vectorized sort/`np.unique`, and call only
prebuilt support/CDF and surface arrays. No per-particle pandas, KNN build or
profile materialization belongs in the timed kernel.

### Precompute once

1. authoritative event/calendar/name bridge and strictly-prior skill snapshots;
2. expanding donor and target spatial profiles;
3. native neutral donor atom tables, partitioned by donor event/category;
4. exactly-once alignment into each target latest-prior setup;
5. target reapplication once per player/setup/state;
6. stable support arrays and CDF/alias tables; and
7. vector transition pools, target surfaces and endpoint-linked hazard search
   structures.

TEE support cache:

`round, hole, par, donor mode, atom/profile/config hashes`

APP support cache:

`round, hole, start lie, four-band profile state, 15-yard kernel bucket,
donor mode, atom/profile/config hashes`

At beta zero, CDFs can be shared across target players because target spatial
reapplication changes coordinates, not probabilities. Target coordinate/label
arrays remain player/setup specific.

Each support is a contiguous array bundle: stable atom IDs, base weights,
optional neutral/model weights, CDF, neutral residual coordinates, target
coordinates, holeout type/mass, label/status code and provenance/cache digest.

### Particle loop

- use `numpy` arrays with compact integer state codes;
- group active particles by finite state key and batch sample each group;
- call surface and hazard nearest-neighbor queries in vector batches;
- pre-index ARG/PUTT pools and batch their log-distance queries by state group;
- never call pandas merge/groupby, `build_distribution`, or a Python KNN loop
  per particle; and
- process disjoint player or particle chunks with counter-keyed RNG.

The current direct APP hierarchy already reduces remaining distance to a
15-yard bucket. ARG/PUTT still use exact within-pool log-distance weighting;
the high-volume implementation should batch those numeric queries rather than
silently replace them with a coarser model. Any distance lattice would be a
separately named approximation arm and must reproduce the 128-particle exact
chain within a prespecified tolerance.

### Storage and streaming

Persist aligned raw/calibrated round strokes as `int16` memmaps shaped
`[arm, player, round, particle]`. At 156 players:

- 50k particles: about 62.4 MB per arm;
- 100k particles: about 124.8 MB per arm; and
- four 100k arms: about 499 MB before diagnostics.

Do not persist every state of every 18-hole path at 100k. Keep online counts
for PMFs, states, hazards, penalties, terminal failures and transition depth,
plus a deterministic audit sample (for example 512 complete paths per arm).
Full field tournament aggregation needs aligned round particles; round-only
scores can use streaming integer PMF counts.

All arithmetic stays wide until final validation. Int16 overflow or a physical
round total outside the predeclared `40..140` lattice fails closed; neither is
allowed to wrap or clip during persistence. Hole scores of 15+ are diagnostics
only and remain exact if the path holes within the 18-transition guard.

Run 50k first. A 100k run uses the same particle IDs and appends
`50,000..99,999`, so the first half need not be recomputed if the checkpoint
manifest/cache hashes match. Parallel workers own disjoint player or particle
slices and never seed their own RNG.

The cache/run manifest records source hashes, profile/config/coefficient hashes,
frame/alignment/classifier versions, RNG version, seed, particle interval,
chunk size, and the SHA-256 of the 50k prefix.

## Research arms and run sequence

### 128-particle equivalence gate

Run the current chain and the new cached/vectorized hook with normalization
mode `raw`, beta zero and identical atom order. Require exact selected atom IDs
and exact terminal hole/round strokes. This separates implementation drift from
the normalization intervention.

The frozen Travelers 2026 gate now passes for one frozen pre-event player,
R1's 18 holes and 128 particles (2,304 hole-particles):

- zero legacy-versus-cached initializer mismatches across 21 physical,
  status and accounting fields;
- zero fresh-versus-cached TEE/APP atom-ID mismatches;
- zero terminal mismatches for strokes, status, hazard/penalty accounting,
  transition counts and final state;
- 66 hazard rows in both paths and identical ARG/PUTT backoff counts; and
- contiguous raw and normalized support/CDF artifacts.

The raw alignment audit covers 33,091 TEE atoms and exactly 42,489 admitted
APP spatial atoms. Of 42,606 native APP rows, 115 are nonspatial, holeout or
nonfinite and two additional finite spatial rows are outside the legacy
post-alignment plausibility boundary. The normalization universe is defined
from the legacy accepted ID set; APP holeout probability is audited separately.
The normalized initializer remains diagnostic only and was not terminally
promoted by this equivalence result.

Then run the bounded factorial diagnostic:

1. `raw:population` -- current physical population control;
2. `raw:player_app` -- current best spatial control;
3. `event_centered:population`;
4. `event_centered:player_app`;
5. `player_state_residualized:population`;
6. `player_state_residualized:player_app`;
7. target TEE transport only as a named sensitivity, not a primary advancing
   arm.

All use beta zero and the same full-strength category-shot marginal variance.

### 50k decision run

Advance at most these four arms after the equivalence gate:

- `raw:player_app` control;
- `event_centered:player_app` candidate;
- `player_state_residualized:player_app` candidate; and
- `raw:population` or `event_centered:population` attribution control,
  selected from the 128-particle diagnostic before looking at 50k outcomes.

### 100k confirmation

Advance only the best normalized candidate plus `raw:player_app` and the frozen
`category_shot` reference. A quality-beta sensitivity is a later, separately
versioned experiment; it must not share the primary beta-zero decision label.

## Required gates

Provenance:

- zero `skill_date >= event_start` rows;
- zero `profile/reference cutoff >= donor event_start` rows;
- target profile cutoff strictly before target event start;
- exactly one rigid alignment per atom;
- target geometry absent from predictor setup except documented course-identity
  audit; and
- every artifact hash reconciled to the authoritative bridge/profile manifest.

Transformation/coherence:

- inverse then forward with the same player/setup returns coordinates within
  numerical tolerance;
- identity fallback leaves an atom unchanged;
- beta-zero neutral/model weights equal the base weights after normalization;
- raw and normalized beta-zero arms select identical atom IDs under equal U
  for the same state/setup support; recursive support divergence is reported;
- moved endpoint labels are reclassified, never copied;
- geometric remaining distance matches target pin coordinates;
- hazard stroke-cost telescoping and terminal stroke accounting remain exact;
  and
- missing profiles preserve support and emit status flags.

RNG/performance:

- 50k output is the exact prefix of 100k;
- output is invariant to 1, 17 and production chunk counts;
- no pandas/KNN call occurs in the timed particle loop;
- raw cached mode exactly reproduces the 128-particle existing chain; and
- per-arm terminal/fail-closed rates are reported, never silently imputed.

Final distribution:

- target round mean error within the existing adapter tolerance;
- spatial marginal variance equals the matching full-strength category-shot
  variance;
- no unintended extra week latent;
- empirical integer support/tails remain auditable; and
- CRPS, log score, PIT/tail coverage and frozen H2H metrics use common observed
  cases and equal event weighting.

## Current caveats

- The strict-prior profile builder, generic normalizer and archive-only
  native-donor-before-alignment harness are now real. The harness is not wired
  into `hole_sim_full_chain.py` or production, and the normalized arm has only
  bounded diagnostic output.
- The available quality calibration covers only three recurring courses and
  the direct-quality decision run is pending; runtime beta remains zero.
- The existing endpoint normalization evaluator is an independent
  approximation, not proof that the shared core reproduces it.
- `hole_sim_batched_population_transition.py` now provides an exact tested
  ARG/PUTT batch API. The cached APP sampler's lie/bucket grouping and some
  hazard operations still contain per-particle work; no 50k/100k run should
  begin until those paths are vectorized and preserve the passed 128-particle
  contract.
- Static profile files remain useful for legacy comparisons but are excluded
  from the normalized walk-forward contract.
- `hole_sim_markov.py::MarkovHoleModel.simulate_hole(max_score=12)` is a
  standalone legacy research loop only. The normalized scalar path imports
  Markov transition construction/pools and calls `MarkovHoleModel._pool`
  through `PopulationTransitionSampler`; the streaming path calls the same
  pool data through `BatchedPopulationTransitionSampler`. Neither calls the
  legacy `simulate_hole` method. Its `min_exact=12` constructor value is a
  donor-support threshold, not a score cap. See the 50k evaluation contract
  for the explicit call graph and immutable-v1/current-v2 provenance split.

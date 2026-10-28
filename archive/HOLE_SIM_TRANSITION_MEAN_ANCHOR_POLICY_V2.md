# Policy-v2 transition-weighting seam

Status: isolated research-only analytical weighting. No chain expectation, mean solve, simulation, streaming, full-chain, or production integration is enabled.

## Purpose

`hole_sim_transition_mean_anchor_policy_v2.py` is a narrow authority boundary between adapter-v4 policy evidence and the unchanged `whole_atom_probabilities` primitive in the frozen mean anchor. It explicitly supports only:

```text
frozen_strict_prior_donor_ability_policy_v2
  -> f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7
```

The declaration is immutable but is not an authority capability. Importing it cannot create a validated weighting session and does not change the frozen anchor or adapter-v4 readiness declaration.

## Trust boundary

Admission accepts candidate atom rows and a strict authority manifest, but it accepts no caller-supplied expected digest. The manifest must equal one closed synthetic approval identity pinned inside the reviewed source:

- approval status: `synthetic_fixture_only_real_chain_hold`
- approval-lock SHA-256: `626bedaa62c23604bc51e3fb0cc04e40b349f8f8697745ff55450816bebe52ab`
- approved authority-manifest SHA-256: `e6177dc0ed5f7545af06d4f61a23dd8458abdcbb98f2cda1f52e981f0330562c`
- approved inventory SHA-256: `33bf857bcfe7e3dd3bd8f5b6237d41a8bf4f629c1dabce7ac44d6135320f26f8`
- approved ordered-root SHA-256: `b0bba4a8150b8363491692f8636cb6dbb8b0bab9d94559c0ff8b55643b233d12`

A caller cannot replace these values through the API. A mutated graph can be semantically valid, fully resealed, and accompanied by a freshly hashed manifest, but it still fails because its inventory and manifest are not named by the closed lock. All real-chain manifests likewise remain fail-closed until a future reviewed source or separate approval-lock change explicitly names one.

The authority manifest is strict: missing and extra fields fail. It binds:

- research-only and production-disabled status;
- the sole frozen policy contract-to-identity mapping;
- frozen mean-anchor contract, descriptor, and source SHA-256;
- adapter-v4 contract, descriptor, and source SHA-256;
- the directly executed v3 physical-adapter contract, descriptor, and source SHA-256;
- frozen policy contract, identity, descriptor, and source SHA-256;
- adapter-v4 support-inventory contract and source-pinned inventory SHA-256;
- ordered root-support contract, root list, round-hole count, and root digest;
- reward-reference and donor-source digests; and
- target-event identity and cutoff.

The ordered root digest includes the full inventory digest, target identity/cutoff, root order, and expected hole count. After adapter-v4 validation, the seam requires every root to exist and every supplied support to be reachable from those roots.

## Runtime binding

Admission and every weighting call recheck these source files by content:

| Source | SHA-256 |
| --- | --- |
| frozen mean anchor | `55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28` |
| adapter-v4 | `e6ab1aa587149af2143ed122ae1307994c9d3277694e153f444f07f30daa711c` |
| v3 physical adapter | `84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff` |
| frozen donor policy v2 | `3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339` |

The combined runtime-binding digest is `be1aceea9d1f8e94dae04b3bb8382e7f2e7c2c80d9813cf48a8b3b814b027b56`.

The seam imports only the frozen parameter/result types and the weighting primitive under private names. It exposes no mean-anchor, v3, adapter-v4, or policy module namespace, no solver, and no public direct-array weighting function. Session constructors cannot be used directly, and module-global capability tokens were removed.

## Donor admission and weighting

Candidate rows pass through `adapter-v4.validate_policy_v2_atom_frame` using only values bound by the trusted authority. The seam then enforces:

- observed rows remain available;
- imputed rows remain unavailable;
- original and current availability remain identical;
- imputed applied ability is exact positive floating-point zero;
- nullable original skill dates remain unchanged through adapter-v4; and
- each donor quality normalization group is wholly observed or wholly imputed.

The group-homogeneity rule is intentionally conservative. It avoids introducing unapproved normalization semantics for a donor group containing both observed abilities and policy-imputed zeros.

A successfully admitted `PolicyV2WeightingSession` accepts only an authorized support key and frozen `TransitionQualityParameters`. It extracts the validated atom arrays itself and passes the applied numeric abilities—including exact zero on unavailable rows—into the unchanged `anchor.whole_atom_probabilities`. Availability remains evidence and is never changed to true or used to erase an imputed row.

The frozen primitive continues to preserve each donor quality group's normalized hierarchy mass. With transfer beta and target slope equal to zero, neutral and final probability mass equal normalized base mass. Nonzero donor/target quality terms change within-support probabilities while preserving donor-group mass during de-tilting.

## Adversarial coverage

`test_hole_sim_transition_mean_anchor_policy_v2.py` covers:

- direct construction of an unvalidated session;
- absence of a direct array-weighting or solve/simulation API;
- attempts to pass the policy identity, inventory digest, or root digest through the removed adjacent-authority argument;
- an extra caller-authored authority digest embedded in the manifest;
- wrong or expanded policy mapping;
- wrong anchor, adapter-v4, v3 physical-adapter, and policy source hashes;
- simulated mutation of the directly executed v3 source;
- wrong inventory with a self-consistent new root digest;
- wrong root digest and a wrong but self-consistent root list;
- a semantically valid, fully resealed graph mutation checked against the closed authority;
- that same mutated and resealed graph accompanied by its own freshly computed authority manifest;
- imputed availability and date alteration;
- an otherwise valid v4 frame that mixes observed and imputed rows inside one donor quality group;
- exact positive zero reaching the frozen numerical primitive;
- donor-group mass preservation;
- beta-zero identity and nonzero quality weighting; and
- defensive atom copies plus the explicit remaining integration HOLD.

## Bounded verification fixture

The fixture is synthetic contract construction only; it does not execute a chain or simulation.

| Measure | Value |
| --- | ---: |
| atom rows | 512 |
| supports | 128 |
| ordered roots | 18 |
| observed rows | 476 |
| imputed/unavailable rows | 36 |
| null skill-date rows | 18 |
| weighting ready | true |
| integration ready | false |

Fixture authority identities:

- inventory SHA-256: `33bf857bcfe7e3dd3bd8f5b6237d41a8bf4f629c1dabce7ac44d6135320f26f8`
- ordered-root SHA-256: `b0bba4a8150b8363491692f8636cb6dbb8b0bab9d94559c0ff8b55643b233d12`
- authority-manifest SHA-256: `e6177dc0ed5f7545af06d4f61a23dd8458abdcbb98f2cda1f52e981f0330562c`

Focused verification: `19 passed`. Combined regression verification across this seam, adapter-v4, the frozen mean anchor, and the frozen donor policy: `109 passed`.

## Remaining integration HOLD

The seam proves only trusted analytical support weighting. Integration remains blocked by:

- `no_policy_v2_physical_chain_expectation_callback`; and
- `no_policy_v2_mean_solve_or_simulation_integration`.

A future change must separately adapt the physical chain expectation callback to this capability, preserve ordered-root authority throughout that callback, and then undergo solve/simulation validation. This module cannot be used to publish or change production pricing.

Source hashes at final verification:

| Source | SHA-256 |
| --- | --- |
| policy-v2 weighting seam | `4095f66f30b646cb03ea7584c954d1d2b321ac04030e898e9adda835bc216bbb` |
| focused tests | `a0c1cd1f33e1ab3e0e293f29453a54f9e49336c29888b3768a6b577c323f4de2` |

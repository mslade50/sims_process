# Transition-mean chain donor admission v4

Status: research-only contract validation. No simulation, streaming, full-chain, or production integration is enabled.

## Why this is a separate module

`hole_sim_transition_mean_chain_adapter.py` remains the sealed v3 implementation. Its donor contract requires every donor ability to be observed and strictly prior. Editing that source would invalidate existing sealed smoke artifacts, so policy-v2 admission lives only in `hole_sim_transition_mean_chain_adapter_v4.py`.

V4 accepts exactly two donor states:

1. `observed_strict_prior`: availability is true, the original skill date is present and strictly before the donor event, and the applied value is the frozen policy-v2 clipped value.
2. `imputed_frozen_neutral_zero`: availability remains false, the applied value is exactly zero, and the actual skill date remains either its strictly-prior timestamp or null.

All other states fail closed.

## Bound evidence

Every atom binds all of the following through the v4 donor-source, donor-snapshot, and support-quality hashes:

- policy contract: `frozen_strict_prior_donor_ability_policy_v2`
- policy identity SHA-256: `f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7`
- frozen policy source: `archive/hole_sim_donor_ability_policy.py`
- frozen policy source SHA-256: `3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339`
- the policy row's `donor_ability_row_sha256`
- the original availability flag and nullable original skill date
- the raw strict-prior value, applied value, value kind, clipping flag, donor identity, category source column, donor/target event identities, and cutoffs
- the externally checked SHA-256 of the complete canonical atom/support inventory

The validator recomputes the frozen policy output from its input fields. A caller-chosen valid-looking hash is insufficient.

## Preserved v3 physical semantics

V4 changes only donor-ability/date admission. Before any v4 seal can be issued, it re-applies every ability-independent v3 atom and support invariant, including:

- complete one-causal-shot and linked-penalty accounting;
- canonical physical-atom and reward-reference hashes;
- positive hierarchy mass and spatial weights, with exact quality-mass multiplication;
- coherent penalty/hazard resolution;
- exact terminal/holed, terminal-state, and next-support rules;
- one category, start state, and start reference value per support;
- reference-value consistency, existing next supports, matching edge states, and an acyclic fully unrolled support graph; and
- the complete APP representation contract, including explicit terminal holeout atoms and equality between declared and actual terminal base mass.

These checks run inside `seal_policy_v2_contracts`, so recomputing accessible hashes cannot bless physically invalid rows.

## External inventory authority

Every sealed row carries the same `support_inventory_sha256`, computed over the complete canonical atom inventory without self-reference or per-support quality hashes. Validation requires a separate `expected_support_inventory_sha256` from the caller, recomputes the inventory digest, and checks both the caller's value and the value repeated on every row. A semantically valid addition, removal, or modification therefore fails against the previously authorized digest even after all local seals are refreshed.

The sealing function computes evidence; it does not create external authority. The expected digest must come from a trusted manifest or other independently controlled handoff. This research-only module has no simulation runner and does not define a root-support list; a future runner must bind its authorized roots separately rather than infer them here.

## Anchor readiness

Observed-only v4 frames are compatible with the existing strict-prior numerical donor term. A frame containing any policy-v2 imputed row remains not anchor-ready because the current anchor API does not explicitly declare this policy.

Future support must be explicit and identity-bound through:

```python
EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT = {
    "frozen_strict_prior_donor_ability_policy_v2":
        "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7",
}
```

Validation and readiness reporting never change `donor_ability_available`. The v4 module exports no simulation runner.

## Adversarial coverage

`test_hole_sim_transition_mean_chain_adapter_v4.py` covers:

- valid observed-only and observed/imputed contracts;
- exact neutral-zero behavior with dated and undated missing abilities;
- forged policy identity, contract, and source;
- zero values without a valid policy row attestation;
- erased and invented dates, including a fully recomputed otherwise-valid contract checked against the original expected donor-source hash;
- observed/imputed mixing for one donor identity;
- attempts to silently turn imputed availability true; and
- negative hierarchy mass;
- resolved hazard status on a non-penalty atom;
- terminal atoms with live next support and nonterminal atoms with empty next support;
- mixed categories inside one support;
- APP terminal mass that differs from its declared probability;
- forged APP holeout representation;
- missing support edges, end-state/next-support mismatch, and graph cycles;
- a fully valid, fully resealed inventory change checked against the original external digest; and
- deterministic fail-closed readiness.

Authoritative persistence is Parquet only. CSV loading is disabled for this hash-bound contract.

## Verification snapshot

The bounded 512-atom contract fixture produced these readiness states:

| Fixture | Observed atom rows | Imputed atom rows | Unavailable rows | Null skill-date rows | Anchor ready |
| --- | ---: | ---: | ---: | ---: | --- |
| observed-only | 512 | 0 | 0 | 0 | true |
| observed plus policy-v2 imputation | 476 | 36 | 36 | 18 | false |

The mixed fixture's exact blocker is `anchor_api_lacks_explicit_frozen_policy_v2_neutral_zero_support`. Availability preservation is true.

Verification completed without a simulation run:

- v4 adversarial contract tests: 25 passed;
- untouched v3 adapter plus frozen policy tests: 78 passed;
- total focused tests: 103 passed.

Source hashes at verification:

| Source | SHA-256 |
| --- | --- |
| untouched v3 adapter | `84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff` |
| v4 admission layer | `e6ab1aa587149af2143ed122ae1307994c9d3277694e153f444f07f30daa711c` |
| frozen policy v2 source | `3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339` |
| untouched mean anchor | `55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28` |
| v4 adversarial tests | `1b8c5f4c507a539f25f6a00b9608e3f76f97783aa31ecd39493763b0188f9eb8` |

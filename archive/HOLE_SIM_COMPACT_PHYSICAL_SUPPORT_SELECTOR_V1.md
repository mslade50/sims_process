# Compact Physical Support Weighting/Selector V1

## Status

This is an archive-only, synthetic-fixture-only research seam. It does not
authorize a real support, implement a physical-chain expectation callback,
solve a mean anchor, run a simulator, or change production.

- synthetic weighting ready: **yes**, for the one closed fixture only
- synthetic hierarchical selection ready: **yes**, for the one closed fixture only
- real-chain ready: **no**
- expectation-callback integration ready: **no**
- mean-solve or simulation integration ready: **no**

The implementation is
`archive/hole_sim_compact_physical_support_selector_v1.py`; focused tests are
`test_hole_sim_compact_physical_support_selector_v1.py`.

## Contract boundary

One `CompactPhysicalSupport` carries aligned, ordered arrays for one complete
conditional support:

- base atom mass and fixed-reference reward;
- every input and output field required by frozen donor-ability policy v2;
- the original donor normalization-group ID;
- one category;
- contiguous proposal, surface, and hazard hierarchy orders plus their frozen
  parent CDFs;
- complete-atom, terminal, positive stroke-cost, and next-state semantics.

Admission validates the exact policy-v2 row attestations. Honest observed rows
and exact neutral-imputed rows may coexist inside the same original donor
group. The group is never rewritten or split. An imputed row remains
`donor_ability_available=false`, and the applied value reaching the private
weighting primitive is bit-sign-checked exact positive `+0.0`.

The session privately calls frozen
`whole_atom_probabilities`. It exposes neither the anchor module nor a raw
weighting, solver, or simulator callable.

## Source-pinned synthetic authority

Admission is closed to the exact fixture named by the module's private
approval lock. Supplying a candidate support hash, recomputing the CDF hash,
and resealing an adjacent manifest does not create authority. A forged private
session carrying candidate-derived hashes is also rejected because its full
authority value must equal the closed source literal.

Closed identities:

| Item | SHA-256 |
|---|---|
| synthetic support | `f0729fa1eae20bc5bc4f0a34381b26ac26aef5d11268ea4063e7aeca5a11a8cc` |
| frozen hierarchy CDF | `4a91fbc2efb727f14927ff5dae02b0bcb1d27b14ae1dcaced43bd4d92554ba28` |
| authority manifest | `4c36f9e8b0f6c8102f2e7c3442486992bfeafc6112a5d15f1ca832ece85885b5` |
| runtime binding | `880844ccd0438d85315509bd1ebc4ef538b20ef1ac8e2cc4aacd3d688507e58e` |
| reward reference | `a70ebc4aa3ec0dab67eb4d69722e48ec3f1fd6ec390643b53f21a63d6c45219b` |

Direct source bindings:

| Source | SHA-256 |
|---|---|
| frozen mean-anchor / whole-atom primitive | `55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28` |
| frozen donor-ability policy v2 | `3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339` |
| parent TEE/APP complete-support selector semantics | `70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83` |

The closed caps are 16 atoms, four proposals, four surfaces per proposal,
four hazards per surface, and absolute log weight 30. Any source, cap, or
authority change fails closed.

## Selection semantics

For nonzero beta or lambda, the seam starts only from the final flattened
whole-atom probabilities returned by the frozen primitive. It sums those
probabilities into proposal marginals, surface conditionals, and hazard
conditionals. Their product reconstructs every flattened atom probability
within two floating-point ulps/tolerance, and all CDFs are formed in retained
support order.

Selection preserves the parent's three semantic uniforms:

- proposal: right-CDF (`searchsorted(..., side="right")`);
- surface: the parent's strict `u > cdf` comparison, so an exact boundary stays
  in the lower surface;
- hazard: right-CDF.

Every call independently selects once through the hierarchy and once by
scanning the flattened semantic intervals. Divergence fails closed; the
selected atom IDs must be identical.

When every beta and lambda is exact zero, selection takes the explicit frozen
CDF fast path. It does not reconstruct the parent CDF from the flattened base
mass. The fixture intentionally makes its first reconstructed proposal mass
one `nextafter` step above the frozen `0.4` boundary, so tests prove that exact
`0.4` advances under the frozen right-CDF while `nextafter(0.4, 0)` does not.
Surface and hazard exact-boundary/nextafter behavior is tested separately.

## Adversarial coverage

Focused tests cover:

- honest observed and clipped observed policy rows;
- neutral-imputed availability and positive-zero preservation;
- mixed observed/imputed states inside unchanged original donor groups;
- attempted donor-group splitting;
- forged policy clipping, availability, and negative zero;
- incomplete atoms, zero cost, terminal live next state, and nonterminal empty
  next state;
- atom/proposal caps and log-cap mismatch or actual log clipping;
- malformed frozen CDFs;
- mutated support plus freshly recomputed self-authority;
- forged private session authority;
- wrong source hash and wrong authority scope;
- absence of public anchor, raw weighting, solver, and simulator routes;
- nonzero flat-factorization and flat-versus-hierarchical pathwise equality;
- exact beta/lambda-zero proposal, surface, and hazard boundary/nextafter rules.

Focused verification: **22 passed**.

## Integration HOLD

Three blockers remain explicit in every readiness result:

1. no externally reviewed real-support authority;
2. no compact-support-to-physical-chain expectation callback integration;
3. no mean-solve or simulation integration.

A future review must mint a separately governed real-support authority and
verify the actual callback's full graph, state/reward ownership, uniform
lattice, and accumulation semantics. This synthetic lock cannot be widened by
a caller or by supplying new candidate digests.

## File hashes

| File | SHA-256 |
|---|---|
| compact selector module | `c5740a9db35c33df736d4c4b8e879fe031ba16ba99983ce26881c14a3c1dfe18` |
| focused tests | `2590c9164ae348b5f2ae63e613f98702f06759dd7b1dcf2f08e8d50b0bd4b1cd` |
| this document | reported at handoff after sealing (not self-referential) |

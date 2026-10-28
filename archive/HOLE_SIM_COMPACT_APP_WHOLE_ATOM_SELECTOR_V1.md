# Compact APP Whole-Atom Selector V1

Status: `synthetic_app_fixture_only_integration_hold`.

This research seam proves APP whole-atom reweighting and retained-uniform
selection without admitting real support or exposing an expectation callback,
solver, simulator, or raw weighting primitive. It does not edit or call the
complete TEE/APP seam, streaming chain, full chain, anchor solver, or production
pipeline.

## Closed input

`admit_synthetic_app_support()` accepts exactly one 11-atom synthetic fixture:

- two ordered terminal holeout donors;
- nine spatial atoms in three numeric proposals;
- proposal/surface/hazard leaves that remain complete atoms;
- one leading, zero-mass surface sentinel;
- two original donor groups, each mixing policy-v2 observed and neutral-imputed
  rows; and
- exact positive `+0.0` for every unavailable imputed ability.

Admission revalidates the policy-v2 row attestations, semantic hierarchy,
whole-atom factorization, caps, support digest, frozen-CDF digest, reward
reference, runtime binding, and the hard-coded authority manifest. A caller
cannot replace any digest with a digest derived from its candidate.

The authority pins these sources byte-for-byte:

- `hole_sim_transition_mean_anchor.py` —
  `55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28`
- `hole_sim_donor_ability_policy.py` —
  `3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339`
- `hole_sim_tee_app_complete_atom_support.py` —
  `70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83`
- superseding reviewed TEE selector
  `hole_sim_compact_physical_support_selector_v1.py` —
  `2a3e53ab337d07184d34c939a9722ebe5f2be5ef47efb07d849ca49d5586580d`

## Weighting and selection

The opaque session privately calls the frozen `whole_atom_probabilities`
primitive once per weight operation with category `app` and the original,
unsplit donor groups. Endpoint, terminal status, lie, surface, hazard, cost,
reward, and donor ability always move as one atom.

For a nonzero tilt, let `p` be terminal mass divided by total final whole-atom
mass. The four retained uniforms are interpreted as follows:

1. `holeout_u < p` selects the terminal branch. Its residual
   `holeout_u / p` selects the ordered donor with a right-CDF; equality advances.
2. `holeout_u >= p` selects the spatial branch. `proposal_u` uses the
   conditional numeric-proposal right-CDF.
3. `surface_u` uses the source seam's strict surface rule: equality remains in
   the lower surface and only `u > upper` advances.
4. `hazard_u` uses a right-CDF; equality advances.

Every tilted hierarchy is rebuilt from final flat whole-atom probabilities,
and its interval widths must reconstruct every final atom. A separate flattened
semantic scan must choose the same atom as hierarchical traversal.

At exact `beta=lambda=0`, selection deliberately replays the frozen scalar
holeout probability, terminal residual CDF, numeric spatial proposal CDF,
strict surface CDF, and hazard right-CDF instead of rebuilding them. The
leading zero-mass surface sentinel is reachable only at `surface_u == 0` on
this frozen path; tilted selection skips it.

Positive probability that cannot be represented by a positive-width binary64
CDF interval fails closed. Weight-log clipping, changed caps, malformed CDFs,
changed policy state, negative zero imputation, source drift, container cycles,
subclassed authority/session/parameter/uniform objects, and non-plain authority
strings also fail closed.

## Verification and hold

Focused verification is in
`test_hole_sim_compact_app_whole_atom_selector_v1.py`. It uses synthetic data
only and covers multiple terminal donors, exact/equality/`nextafter`/zero
boundaries, zero-mass sentinels, tilted factorization, policy-v2 mixed groups,
caps and clipping, CDF collapse, mutation/resealing attacks, and absence of a
public primitive/solver/simulator route.

This artifact is not authority for a real APP support, is not integrated into
the complete seam or any chain, and must remain on explicit integration HOLD
pending independent review and a separately approved real source authority.

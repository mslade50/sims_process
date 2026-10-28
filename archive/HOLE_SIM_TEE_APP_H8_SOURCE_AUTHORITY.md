# Lowry R1 H8 source-authority candidate contract

`hole_sim_tee_app_h8_source_authority.py` is phase one of a two-phase,
non-self-authorizing workflow. It creates evidence for review. It does not
create an authority, edit the complete-atom seam, enable production, or run a
simulation.

## Fixed scope

The builder has one variable: a fresh output directory. Everything that can
change the result is fixed in source:

- query: Lowry (`33204`), `pga:R2026034`, 2026-06-25, raw APP, R1 H8,
  par 3, tee start, exact forecast coordinates and 204.54171527805522 yards;
- physical course: `geometry:16790842a3043da1`;
- frozen complete seam SHA-256:
  `70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83`;
- proposal parent output-index SHA-256:
  `3e646b49fb6cf7732b72577b6496ab4aa208b038f4cae73dc2185f9b6ac0dade`;
- 50k numeric parent output-index SHA-256:
  `b50f50c9a420726d50c30acf3b04023cc2b6ec580125f8f989fef09729de2b70`;
- proposal measure:
  `8a646ebc165f86eb9859ea1208fd11bfd1b8efba45128b8a63ea1c7c9a1495f6`;
- physical numeric payload:
  `44bde6f9a6943c5bb256b549e8de6bed9fa5c23ab9e91e78efb4f8ae2372a0d7`;
- caps: 8,192 batch rows, 8,192 hazard donors per surface, 8,192 absolute
  provider ceiling, and 16,384 complete atoms per proposal;
- hazard provider preflight: exactly 1,046 rows, split 682 OTT and 364 APP.

There are no CLI flags for sources, parent artifacts, query fields, hashes,
provider counts, policy, reference model, or caps.

## Why preparation is still required

The existing 904-row H8 proposal and the persisted raw numeric support are
reused; they do not need regeneration. They do not contain the exact H8 APP
surface constructor rows or the scalar hazard-resolver provider/parameters.
Consequently, one fresh `prepare_streaming_event` is unavoidable when building
the real candidate.

The prepared byte and logical hashes must exactly match those sealed in the
H8 proposal parent. Immediately after preparation, the builder checks the
1,046/682/364 provider shape against the absolute 8,192 ceiling before its own
resolver hash, copy, or complete distribution. It then snapshots the minimal
surface constructor rows and the complete ordered resolver provider.

## Candidate contents

The closed candidate contains:

- the original H8 proposal bytes and frozen-policy-v2 proposal;
- the original raw numeric NPZ and a physical-atom-ID NPZ (`<U64` IDs);
- exact H8 surface constructor rows;
- the complete hazard provider and resolver parameters;
- the exact query and copied fixed-reference model;
- copied parent manifests and checksum indexes;
- ordered proposal, surface, and hazard inventory projections in shards of at
  most 8,192 rows;
- independent ordered inventory hashes checked against the frozen seam audit;
- source, dependency, prepared-input, and parent-closure start/end maps;
- `authority_candidate.json`, `manifest.json`, report, and a non-circular
  checksum index covering every file.

Expected real expansion observations are 904 proposal records, 903 surface
records, 30,702 hazard records, 35,218 complete atoms, one terminal atom, and
1,989 complete branches with frozen-policy imputed ability.

Every candidate is forced to contain:

```json
{
  "authority_id": null,
  "authority_ready": false,
  "production_enabled": false,
  "admission_status": "candidate_external_review_required",
  "review_lock_sha256": null,
  "candidate_digest_is_not_authority": true
}
```

The builder has no API that writes to or admits the frozen seam.

## Phase two: independent review and lock

A reviewer independent of the builder must validate the complete candidate
closure and then create a separate immutable lock in new code/artifacts. The
reviewer must hard-code, rather than accept from the candidate or CLI:

- candidate checksum-index SHA-256 and candidate ID;
- builder/seam/dependency hashes;
- both parent closure hashes;
- exact query context and complete query key;
- proposal, numeric, ability-policy, surface, hazard-resolver, and reference
  identities;
- provider counts and all four caps;
- each ordered inventory record count/hash and the global source-inventory
  hash;
- readiness/admission decision and reviewer identity/process evidence.

Only after that review may a separate bounded expansion runner be implemented.
That runner must consume the hard-pinned review lock, reconstruct inputs only
from the closed candidate snapshots, recheck the 8,192 provider ceiling before
resolver construction/distribution, and reject all candidate-, environment-,
or CLI-supplied digests. The phase-one builder must never be imported as an
authority source.

## Real candidate command (do not run before source review)

```powershell
python archive/hole_sim_tee_app_h8_source_authority.py --output-dir permanent_data/hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_v1
```

After that command completes, hold. Review the candidate before creating any
lock or runner.

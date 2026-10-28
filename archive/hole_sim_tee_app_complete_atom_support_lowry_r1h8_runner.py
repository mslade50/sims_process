"""Fixed, bounded materializer for the reviewed Lowry R1 H8 APP support.

This runner has no path, digest, query, or cap inputs.  It consumes the one
reviewed source snapshot, delegates physical expansion to the frozen complete
atom seam, validates that inner artifact, and closes it inside a runner-owned
outer artifact.  It is research evidence only: no tournament simulation,
mean-anchor solve, or production scoring is exposed here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import audit_hole_sim_tee_app_h8_source_authority as auditor  # noqa: E402
import hole_sim_tee_app_complete_atom_support as complete  # noqa: E402
import hole_sim_tee_app_h8_source_authority as builder  # noqa: E402


CONTRACT = "fixed_lowry_r1h8_complete_atom_materialization_v1"
PREFLIGHT_CONTRACT = "fixed_lowry_r1h8_complete_atom_materialization_preflight_v1"
OUTPUT_INDEX_CONTRACT = "fixed_lowry_r1h8_outer_output_sha256_v1"
OUTPUT_INDEX = "OUTPUT_SHA256.json"
OUTPUT = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1"
)
INNER_NAME = "complete_atom_support"

EXPANSION_BATCH_ROWS = 8_192
MAX_ROWS_PER_PART = 8_192
MAX_HAZARD_DONORS_PER_SURFACE = 8_192
MAX_COMPLETE_ATOMS_PER_PROPOSAL = 16_384

EXPECTED_SEAM_SHA256 = (
    "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
)
EXPECTED_SEAM_TEST_SHA256 = (
    "eb648d94447d8a60497dcf95b34dcbbe1a64c21cd5458bb27ad87c1673898397"
)
EXPECTED_AUTHORITY_LOCK_SHA256 = (
    "edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c"
)
EXPECTED_AUTHORITY_RECORD_SHA256 = (
    "efaf969daea1ac9a7680b0fd8d534ffa700430523aa7cb13cec6fabaed9c347c"
)
EXPECTED_AUTHORITY_ID = "lowry_r1h8_raw_app_exact_query_v1"
EXPECTED_AUTHORITY_SCOPE = "lowry_r1h8_raw_app_exact_query_only"

EXPECTED_BUILDER_SHA256 = (
    "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
)
EXPECTED_AUDITOR_SHA256 = (
    "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06"
)
EXPECTED_AUDITOR_TEST_SHA256 = (
    "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c"
)

CANDIDATE = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_v1"
)
AUDIT = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_audit_v1"
)
EXPECTED_CANDIDATE_ID = (
    "341ce3818fcc5301179c8858fcba02a94778384e2e554e81fd762d2f9626b7ff"
)
EXPECTED_CANDIDATE_INDEX_SHA256 = (
    "28bc60ca52ec42ee1abba4eaa26d5d10a6cc50f6df1fdb4b0e1631594c977aa5"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "d0be4068daf67c05243eeed4eb2963168ec8555ba2dc5842d72ade434e09bfc2"
)
EXPECTED_AUTHORITY_CANDIDATE_SHA256 = (
    "f046624aa48601a37b7ad6e8b2ab94f8558e857fca6220e286b9611dc010d5d4"
)
EXPECTED_AUDIT_ID = (
    "16c79fb1fd622db1ee7f4b8cb638ec0b72b746d0c2370541c23957d12d23b6e1"
)
EXPECTED_AUDIT_INDEX_SHA256 = (
    "cdfb7b50a958ddd397aaf5297767d0f1d5e11ca80b946eb0a13475a04f8f3431"
)
EXPECTED_AUDIT_JSON_SHA256 = (
    "57231a27cbb7df00fe4af95d957efccdcbdad67d5e202340a3ff2f4cd0c43402"
)
EXPECTED_AUDIT_REPORT_SHA256 = (
    "405428f00644bb53984dbc89ec2394d6f54441865d55f971324329b5de2f5368"
)

EXPECTED_QUERY_KEY = (
    "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "37a91ecae53d7945132a9eca7871490250b57f978ec95df4f06e477550c9f92c"
)
EXPECTED_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
EXPECTED_COUNTS = {
    "proposal_atoms": 904,
    "surface_source_inventory_records": 903,
    "hazard_source_inventory_records": 30_702,
    "complete_atoms": 35_218,
    "terminal_atoms": 1,
    "imputed_ability_atoms": 1_989,
}

EXPECTED_COMPLETE_DEPENDENCIES = {
    "ability_policy": "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339",
    "app": "6acceff91e9130e3584d02d4d113daecc5e84831dfe3fff6f7190d40f5450e1f",
    "batched_population": "0da3a934b671c3ed74a24c8c754a8404e3a329eb4f54f3b429152e45cd7fca23",
    "chain_adapter": "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    "fixed_reference": "39df1d57a6623b49f90f3ae82cb9867cb77a3f7cff740754794cdef324ba8b8b",
    "full_chain": "10d8971b45843bc74d4dbd7b5c53d94cdb3f35e0e8621bcba357acce77a20c89",
    "hazard": "8a3d8e019c39156299ebde6e7e8a344ba2edf5b55dc8602981ab75e4466163e5",
    "proposal_ledger": "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97",
    "spatial": "dc85b38b5601ba5f6afbe2ce4d900e109243f60cb6a9dd7a03426b436443ea97",
    "streaming": "0d8c2be5027c5a5325d78eea6205691091455104624f6a9acb3006de5993f807",
    "tee": "006f1de02d98be47dfd621a5e46d14672466ba6ee673cf7e0f56bc94daf56b43",
}


class FixedH8RunnerError(ValueError):
    """Raised when the fixed materialization chain is not exactly admitted."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FixedH8RunnerError(f"Expected JSON object: {path}")
    return value


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise FixedH8RunnerError(f"{label} changed")


def _source_state() -> dict[str, object]:
    paths = {
        "runner_source": Path(__file__),
        "complete_seam": Path(complete.__file__),
        "complete_seam_test": REPO / "test_hole_sim_tee_app_complete_atom_support.py",
        "candidate_builder": Path(builder.__file__),
        "detached_auditor": Path(auditor.__file__),
        "detached_auditor_test": REPO / "test_audit_hole_sim_tee_app_h8_source_authority.py",
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    expected = {
        "complete_seam": EXPECTED_SEAM_SHA256,
        "complete_seam_test": EXPECTED_SEAM_TEST_SHA256,
        "candidate_builder": EXPECTED_BUILDER_SHA256,
        "detached_auditor": EXPECTED_AUDITOR_SHA256,
        "detached_auditor_test": EXPECTED_AUDITOR_TEST_SHA256,
    }
    for name, digest in expected.items():
        if observed[name] != digest:
            raise FixedH8RunnerError(
                f"Frozen source changed: {name}={observed[name]} != {digest}"
            )
    dependencies = complete._dependency_sha256()
    _require_equal(dependencies, EXPECTED_COMPLETE_DEPENDENCIES, "complete dependency map")
    observed["complete_dependencies"] = dependencies
    return observed


def _artifact_state() -> dict[str, object]:
    paths = {
        "candidate_output_index": CANDIDATE / OUTPUT_INDEX,
        "candidate_manifest": CANDIDATE / "manifest.json",
        "authority_candidate": CANDIDATE / "authority_candidate.json",
        "audit_output_index": AUDIT / OUTPUT_INDEX,
        "audit_json": AUDIT / "audit.json",
        "audit_report": AUDIT / "REPORT.md",
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    expected = {
        "candidate_output_index": EXPECTED_CANDIDATE_INDEX_SHA256,
        "candidate_manifest": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "authority_candidate": EXPECTED_AUTHORITY_CANDIDATE_SHA256,
        "audit_output_index": EXPECTED_AUDIT_INDEX_SHA256,
        "audit_json": EXPECTED_AUDIT_JSON_SHA256,
        "audit_report": EXPECTED_AUDIT_REPORT_SHA256,
    }
    _require_equal(observed, expected, "candidate/audit artifact pins")
    # Rehash the complete candidate and detached-audit trees at both admission
    # and finalization.  Merely rechecking their unchanged root index bytes
    # would miss a payload modified in place after the initial readback.
    candidate_entries = builder._validate_closed_output_index(CANDIDATE)
    audit_closure = auditor._audit_closed_tree(AUDIT)
    observed["candidate_closed_files"] = int(len(candidate_entries))
    observed["candidate_closed_index_payload_sha256"] = _stable_sha(
        candidate_entries
    )
    observed["audit_closure"] = dict(audit_closure)
    return observed


def _authority_record(candidate: Mapping[str, object]) -> Mapping[str, object]:
    records = complete.source_inventory_authority_records()
    matches = [
        record
        for record in records
        if str(record.get("authority_id", "")) == EXPECTED_AUTHORITY_ID
    ]
    if len(matches) != 1:
        raise FixedH8RunnerError("Fixed H8 authority record is absent or duplicated")
    record = matches[0]
    if _stable_sha(_plain(record)) != EXPECTED_AUTHORITY_RECORD_SHA256:
        raise FixedH8RunnerError("Fixed H8 authority record changed")
    if complete._SOURCE_PINNED_AUTHORITY_LOCK_SHA256 != (
        EXPECTED_AUTHORITY_LOCK_SHA256
    ):
        raise FixedH8RunnerError("Fixed source authority lock changed")
    lock = _stable_sha(
        {
            "contract": complete.SOURCE_INVENTORY_AUTHORITY_CONTRACT,
            "records": tuple(_plain(item) for item in records),
        }
    )
    if lock != EXPECTED_AUTHORITY_LOCK_SHA256:
        raise FixedH8RunnerError("Fixed source authority lock does not recompute")
    expansion_audit = candidate.get("expansion_audit")
    if not isinstance(expansion_audit, Mapping):
        raise FixedH8RunnerError("Candidate expansion audit is absent")
    projection = builder._source_inventory_record(expansion_audit)
    admitted = complete._source_inventory_authority([projection])
    if admitted is not record:
        raise FixedH8RunnerError("Reviewed H8 projection is not the admitted record")
    return record


def _assert_candidate(candidate: Mapping[str, object]) -> None:
    if candidate.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise FixedH8RunnerError("Candidate identity changed")
    if candidate.get("source_inventory_sha256") != EXPECTED_SOURCE_INVENTORY_SHA256:
        raise FixedH8RunnerError("Candidate source inventory changed")
    audit = candidate.get("expansion_audit")
    if not isinstance(audit, Mapping):
        raise FixedH8RunnerError("Candidate expansion audit is absent")
    for field, expected in EXPECTED_COUNTS.items():
        if type(audit.get(field)) is not int or audit.get(field) != expected:
            raise FixedH8RunnerError(f"Candidate count changed: {field}")
    if audit.get("query_key") != EXPECTED_QUERY_KEY:
        raise FixedH8RunnerError("Candidate exact query changed")
    if audit.get("support_key") != EXPECTED_SUPPORT_KEY:
        raise FixedH8RunnerError("Candidate support key changed")
    quality_mass = audit.get("quality_mass")
    if type(quality_mass) is not float or not math.isclose(
        quality_mass, 1.0, abs_tol=3e-13
    ):
        raise FixedH8RunnerError("Candidate quality mass changed")
    if type(audit.get("terminal_mass")) is not float or audit.get(
        "terminal_mass"
    ) != 0.002:
        raise FixedH8RunnerError("Candidate terminal mass changed")
    if audit.get("anchor_ready") is not False:
        raise FixedH8RunnerError("Candidate unexpectedly became anchor-ready")


def _admit_fixed_inputs() -> dict[str, object]:
    source_start = _source_state()
    artifact_start = _artifact_state()
    candidate = builder.validate_candidate_artifact(
        CANDIDATE, verify_current_sources=False
    )
    audit_result = auditor.validate_audit_artifact(deep=False)
    audit = audit_result["audit"]
    if not isinstance(audit, Mapping) or audit.get("audit_id") != EXPECTED_AUDIT_ID:
        raise FixedH8RunnerError("Detached audit identity changed")
    _assert_candidate(candidate)
    authority = _authority_record(candidate)
    source_end = _source_state()
    artifact_end = _artifact_state()
    _require_equal(source_end, source_start, "source start/end closure")
    _require_equal(artifact_end, artifact_start, "artifact start/end closure")
    return {
        "candidate": candidate,
        "audit": dict(audit),
        "authority": authority,
        "source_sha256_start": source_start,
        "source_sha256_end": source_end,
        "artifact_sha256_start": artifact_start,
        "artifact_sha256_end": artifact_end,
    }


def _load_request(candidate: Mapping[str, object]) -> complete.CompleteAtomRequest:
    proposal = pd.read_parquet(CANDIDATE / "proposal_policy_v2.parquet")
    complete.apply_frozen_policy_to_proposal(proposal)
    numeric = builder._load_candidate_numeric(
        CANDIDATE / "numeric_support_physical.npz", proposal
    )
    query_payload = candidate.get("query")
    if not isinstance(query_payload, Mapping):
        raise FixedH8RunnerError("Candidate query payload is absent")
    query = builder._query_from_candidate(query_payload)
    surface = builder._surface_from_snapshot(
        pd.read_parquet(CANDIDATE / "surface_source_h8.parquet")
    )
    provider_path = CANDIDATE / "hazard_provider.parquet"
    if builder._parquet_metadata_rows(provider_path) != builder.EXPECTED_PROVIDER_TOTAL:
        raise FixedH8RunnerError("Hazard provider preflight count changed")
    provider_snapshot = pd.read_parquet(provider_path)
    if not np.array_equal(
        provider_snapshot["hazard_provider_row_order"].to_numpy(np.int64),
        np.arange(builder.EXPECTED_PROVIDER_TOTAL, dtype=np.int64),
    ):
        raise FixedH8RunnerError("Hazard provider row order changed")
    provider = provider_snapshot.drop(columns="hazard_provider_row_order")
    builder._preflight_hazard_provider(SimpleNamespace(donors=provider))
    params = builder._read_json(CANDIDATE / "hazard_resolver_params.json")
    constructor_fields = (
        "setup_prior",
        "hole_prior",
        "spatial_bandwidth_yd",
        "max_nearest_distance_yd",
        "same_hole_fallback_max_distance_yd",
        "minimum_category_support",
    )
    resolver = builder.hazard.EndpointLinkedHazardResolver(
        provider, **{field: params[field] for field in constructor_fields}
    )
    builder._preflight_hazard_provider(resolver)
    reference = builder.fixed_reference.ReferenceValueModel.load(
        CANDIDATE / "reference_model"
    )
    return complete.CompleteAtomRequest(
        proposal_atoms=proposal,
        numeric_support=numeric,
        query=query,
        surface_map=surface,
        hazard_resolver=resolver,
        reference_model=reference,
    )


def preflight() -> dict[str, object]:
    admission = _admit_fixed_inputs()
    return {
        "contract": PREFLIGHT_CONTRACT,
        "ready": True,
        "output_exists": OUTPUT.exists(),
        "materialization_launched": False,
        "simulation_launched": False,
        "mean_solve_launched": False,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "audit_id": EXPECTED_AUDIT_ID,
        "authority_id": EXPECTED_AUTHORITY_ID,
        "authority_lock_sha256": EXPECTED_AUTHORITY_LOCK_SHA256,
        "query_key": EXPECTED_QUERY_KEY,
        "expected_complete_atoms": EXPECTED_COUNTS["complete_atoms"],
        "source_sha256_start": admission["source_sha256_start"],
        "source_sha256_end": admission["source_sha256_end"],
        "artifact_sha256_start": admission["artifact_sha256_start"],
        "artifact_sha256_end": admission["artifact_sha256_end"],
    }


def _safe_relative(value: object) -> str:
    relative = str(value)
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or relative == OUTPUT_INDEX
    ):
        raise FixedH8RunnerError(f"Unsafe outer artifact path: {relative!r}")
    return relative


def _tree(root: Path, *, exclude_root_index: bool) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    directories: list[str] = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise FixedH8RunnerError(f"Outer artifact contains a symlink: {relative}")
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            if exclude_root_index and relative == OUTPUT_INDEX:
                continue
            files[relative] = path
        else:
            raise FixedH8RunnerError(f"Outer artifact has a non-regular node: {relative}")
    return files, directories


def _write_outer_index(root: Path) -> None:
    files, directories = _tree(root, exclude_root_index=True)
    payload = {
        "contract": OUTPUT_INDEX_CONTRACT,
        "directories": directories,
        "files": {
            relative: {
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for relative, path in sorted(files.items())
        },
    }
    _write_json(root / OUTPUT_INDEX, payload)


def _validate_outer_index(root: Path) -> dict[str, object]:
    index = _read_json(root / OUTPUT_INDEX)
    if index.get("contract") != OUTPUT_INDEX_CONTRACT:
        raise FixedH8RunnerError("Outer output-index contract changed")
    declared_files = index.get("files")
    declared_directories = index.get("directories")
    if not isinstance(declared_files, Mapping) or not isinstance(
        declared_directories, list
    ):
        raise FixedH8RunnerError("Outer output index is malformed")
    files, directories = _tree(root, exclude_root_index=True)
    normalized = {_safe_relative(key): value for key, value in declared_files.items()}
    if set(normalized) != set(files) or list(declared_directories) != directories:
        raise FixedH8RunnerError("Outer artifact tree closure changed")
    for relative, path in files.items():
        metadata = normalized[relative]
        if not isinstance(metadata, Mapping):
            raise FixedH8RunnerError("Outer output-index entry is malformed")
        if (
            type(metadata.get("bytes")) is not int
            or metadata.get("bytes") != path.stat().st_size
        ):
            raise FixedH8RunnerError(f"Outer byte count changed: {relative}")
        if (
            type(metadata.get("sha256")) is not str
            or metadata.get("sha256") != _sha256_file(path)
        ):
            raise FixedH8RunnerError(f"Outer byte hash changed: {relative}")
    nested = f"{INNER_NAME}/{OUTPUT_INDEX}"
    if nested not in normalized:
        raise FixedH8RunnerError("Outer index does not bind the inner output index")
    return index


def _assert_inner(
    manifest: Mapping[str, object], validation: Mapping[str, object]
) -> None:
    if manifest.get("contract") != complete.PERSISTENCE_CONTRACT:
        raise FixedH8RunnerError("Inner persistence contract changed")
    expected_caps = {
        "expansion_batch_rows": EXPANSION_BATCH_ROWS,
        "max_rows_per_part": MAX_ROWS_PER_PART,
        "max_hazard_donors_per_surface": MAX_HAZARD_DONORS_PER_SURFACE,
        "absolute_max_hazard_provider_donors": (
            complete.ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
        ),
        "max_complete_atoms_per_proposal": MAX_COMPLETE_ATOMS_PER_PROPOSAL,
    }
    for field, expected in expected_caps.items():
        if type(manifest.get(field)) is not int or manifest.get(field) != expected:
            raise FixedH8RunnerError(f"Inner cap changed: {field}")
    if type(manifest.get("requests")) is not int or manifest.get("requests") != 1:
        raise FixedH8RunnerError("Inner request count changed")
    if (
        type(manifest.get("complete_atoms")) is not int
        or manifest.get("complete_atoms") != EXPECTED_COUNTS["complete_atoms"]
    ):
        raise FixedH8RunnerError("Inner complete-atom count changed")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != 5:
        raise FixedH8RunnerError("Inner part count changed")
    expected_part_rows = (8_192, 8_192, 8_192, 8_192, 2_450)
    for index, (part, expected_rows) in enumerate(zip(parts, expected_part_rows)):
        if not isinstance(part, Mapping):
            raise FixedH8RunnerError("Inner part metadata changed")
        if part.get("path") != f"parts/part-{index:06d}.parquet":
            raise FixedH8RunnerError("Inner part path changed")
        if type(part.get("rows")) is not int or part.get("rows") != expected_rows:
            raise FixedH8RunnerError("Inner part row count changed")
    if type(manifest.get("peak_buffered_rows")) is not int or manifest.get(
        "peak_buffered_rows"
    ) != MAX_ROWS_PER_PART:
        raise FixedH8RunnerError("Inner peak buffer count changed")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or len(inventory) != 1:
        raise FixedH8RunnerError("Inner query inventory changed")
    _assert_candidate({"candidate_id": EXPECTED_CANDIDATE_ID,
                       "source_inventory_sha256": manifest.get("source_inventory_sha256"),
                       "expansion_audit": inventory[0]})
    if manifest.get("source_inventory_authority_id") != EXPECTED_AUTHORITY_ID:
        raise FixedH8RunnerError("Inner authority ID changed")
    if manifest.get("source_inventory_authority_scope") != EXPECTED_AUTHORITY_SCOPE:
        raise FixedH8RunnerError("Inner authority scope changed")
    if manifest.get("source_inventory_authority_ready") is not True:
        raise FixedH8RunnerError("Inner authority readiness changed")
    if manifest.get("production_enabled") is not False:
        raise FixedH8RunnerError("Inner artifact unexpectedly enables production")
    expected_validation_counts = {
        "queries": 1,
        "parts": 5,
        "complete_atoms": EXPECTED_COUNTS["complete_atoms"],
    }
    for field, expected in expected_validation_counts.items():
        if type(validation.get(field)) is not int or validation.get(field) != expected:
            raise FixedH8RunnerError(f"Inner validation count changed: {field}")


def _outer_report() -> str:
    return (
        "# Lowry R1 H8 complete APP atom support\n\n"
        "Status: **bounded mechanics/support evidence only**.\n\n"
        f"- Exact query: `{EXPECTED_QUERY_KEY}`\n"
        f"- Complete atoms: {EXPECTED_COUNTS['complete_atoms']}\n"
        f"- Terminal atoms: {EXPECTED_COUNTS['terminal_atoms']}\n"
        f"- Policy-v2 neutral-imputed branches: {EXPECTED_COUNTS['imputed_ability_atoms']}\n"
        "- Anchor ready: false\n"
        "- Production enabled: false\n"
        "- No tournament simulation or mean solve is performed by this runner.\n"
    )


def materialize() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"Fixed H8 output already exists: {OUTPUT}")
    admission = _admit_fixed_inputs()
    candidate = admission["candidate"]
    assert isinstance(candidate, Mapping)
    request = _load_request(candidate)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    inner = OUTPUT / INNER_NAME
    inner_manifest = complete.persist_complete_atom_requests(
        [request],
        inner,
        expansion_batch_rows=EXPANSION_BATCH_ROWS,
        max_rows_per_part=MAX_ROWS_PER_PART,
        max_hazard_donors_per_surface=MAX_HAZARD_DONORS_PER_SURFACE,
        max_complete_atoms_per_proposal=MAX_COMPLETE_ATOMS_PER_PROPOSAL,
    )
    inner_validation = complete.validate_persisted_complete_atom_artifact(inner)
    _assert_inner(inner_manifest, inner_validation)
    source_end = _source_state()
    artifact_end = _artifact_state()
    _require_equal(source_end, admission["source_sha256_start"], "final source closure")
    _require_equal(artifact_end, admission["artifact_sha256_start"], "final input closure")
    manifest = {
        "contract": CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "anchor_ready": False,
        "simulation_launched": False,
        "mean_solve_launched": False,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "audit_id": EXPECTED_AUDIT_ID,
        "authority_id": EXPECTED_AUTHORITY_ID,
        "authority_record_sha256": EXPECTED_AUTHORITY_RECORD_SHA256,
        "authority_lock_sha256": EXPECTED_AUTHORITY_LOCK_SHA256,
        "query_key": EXPECTED_QUERY_KEY,
        "support_key": EXPECTED_SUPPORT_KEY,
        "complete_atoms": EXPECTED_COUNTS["complete_atoms"],
        "terminal_atoms": EXPECTED_COUNTS["terminal_atoms"],
        "imputed_ability_atoms": EXPECTED_COUNTS["imputed_ability_atoms"],
        "inner_relative_path": INNER_NAME,
        "inner_output_index_sha256": _sha256_file(inner / OUTPUT_INDEX),
        "inner_validation": dict(inner_validation),
        "source_sha256_start": admission["source_sha256_start"],
        "source_sha256_end": source_end,
        "artifact_sha256_start": admission["artifact_sha256_start"],
        "artifact_sha256_end": artifact_end,
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    (OUTPUT / "REPORT.md").write_text(_outer_report(), encoding="utf-8")
    _write_outer_index(OUTPUT)
    return validate_materialized()


def validate_materialized() -> dict[str, object]:
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise FixedH8RunnerError("Fixed H8 output is absent or invalid")
    _validate_outer_index(OUTPUT)
    manifest = _read_json(OUTPUT / "manifest.json")
    if manifest.get("contract") != CONTRACT:
        raise FixedH8RunnerError("Outer materialization contract changed")
    for field in ("production_enabled", "anchor_ready", "simulation_launched", "mean_solve_launched"):
        if manifest.get(field) is not False:
            raise FixedH8RunnerError(f"Outer readiness changed: {field}")
    expected_identity = {
        "research_only": True,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "audit_id": EXPECTED_AUDIT_ID,
        "authority_id": EXPECTED_AUTHORITY_ID,
        "authority_record_sha256": EXPECTED_AUTHORITY_RECORD_SHA256,
        "authority_lock_sha256": EXPECTED_AUTHORITY_LOCK_SHA256,
        "query_key": EXPECTED_QUERY_KEY,
        "support_key": EXPECTED_SUPPORT_KEY,
        "complete_atoms": EXPECTED_COUNTS["complete_atoms"],
        "terminal_atoms": EXPECTED_COUNTS["terminal_atoms"],
        "imputed_ability_atoms": EXPECTED_COUNTS["imputed_ability_atoms"],
    }
    for field, expected in expected_identity.items():
        observed = manifest.get(field)
        if type(expected) is bool:
            exact = observed is expected
        elif type(expected) is int:
            exact = type(observed) is int and observed == expected
        else:
            exact = type(observed) is type(expected) and observed == expected
        if not exact:
            raise FixedH8RunnerError(f"Outer identity changed: {field}")
    if (OUTPUT / "REPORT.md").read_text(encoding="utf-8") != _outer_report():
        raise FixedH8RunnerError("Outer report changed")
    admission = _admit_fixed_inputs()
    for field in (
        "source_sha256_start",
        "source_sha256_end",
        "artifact_sha256_start",
        "artifact_sha256_end",
    ):
        if _stable_sha(manifest.get(field)) != _stable_sha(admission[field]):
            raise FixedH8RunnerError(f"Outer provenance changed: {field}")
    inner = OUTPUT / INNER_NAME
    if manifest.get("inner_relative_path") != INNER_NAME:
        raise FixedH8RunnerError("Inner relative path changed")
    if manifest.get("inner_output_index_sha256") != _sha256_file(inner / OUTPUT_INDEX):
        raise FixedH8RunnerError("Inner output index identity changed")
    inner_manifest = _read_json(inner / "manifest.json")
    inner_validation = complete.validate_persisted_complete_atom_artifact(inner)
    _assert_inner(inner_manifest, inner_validation)
    if _stable_sha(manifest.get("inner_validation")) != _stable_sha(
        dict(inner_validation)
    ):
        raise FixedH8RunnerError("Outer persisted inner validation changed")
    expected_root = {"manifest.json", "REPORT.md"}
    parts = inner_manifest["parts"]
    assert isinstance(parts, list)
    expected_inner = {
        f"{INNER_NAME}/manifest.json",
        f"{INNER_NAME}/{OUTPUT_INDEX}",
        *{
            f"{INNER_NAME}/{str(part['path'])}"
            for part in parts
            if isinstance(part, Mapping)
        },
    }
    files, _ = _tree(OUTPUT, exclude_root_index=True)
    if set(files) != expected_root | expected_inner:
        raise FixedH8RunnerError("Outer artifact contains undeclared files")
    return {
        "contract": "validated_fixed_lowry_r1h8_complete_atom_materialization_v1",
        "valid": True,
        "complete_atoms": EXPECTED_COUNTS["complete_atoms"],
        "terminal_atoms": EXPECTED_COUNTS["terminal_atoms"],
        "imputed_ability_atoms": EXPECTED_COUNTS["imputed_ability_atoms"],
        "anchor_ready": False,
        "production_enabled": False,
        "simulation_launched": False,
        "mean_solve_launched": False,
        "output_index_sha256": _sha256_file(OUTPUT / OUTPUT_INDEX),
        "inner_output_index_sha256": _sha256_file(inner / OUTPUT_INDEX),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or materialize the one reviewed Lowry R1 H8 raw APP "
            "complete-atom support. No paths, digests, queries, or caps are configurable."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--validate-only",
        action="store_true",
        help="Read-only admission; validate the fixed output too when it exists",
    )
    group.add_argument(
        "--materialize",
        action="store_true",
        help="Create the one fixed bounded research artifact",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        result = validate_materialized() if OUTPUT.exists() else preflight()
    else:
        result = materialize()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

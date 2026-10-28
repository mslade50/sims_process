"""Independent, non-authorizing audit of the fixed Lowry R1 H8 candidate.

This module is intentionally detached from the candidate builder.  It does
not import the builder, prepare the historical event, expand complete atoms,
simulate shots, or mutate the complete-atom seam.  It reads the already
closed candidate and its frozen parents, independently reconstructs the
ordered source inventories and their source bridges, and writes one fixed
review-evidence artifact.

The emitted digest is evidence, never authority.  Admission still requires a
separate, manual source-pinned record in the complete-atom seam.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
DEV = REPO.parent
AUDITOR_SOURCE = Path(__file__).resolve()
AUDITOR_TEST = REPO / "test_audit_hole_sim_tee_app_h8_source_authority.py"
CANDIDATE = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_v1"
)
OUTPUT = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_audit_v1"
)
H8_PARENT = (
    REPO
    / "permanent_data"
    / "hole_sim_streaming_transition_ledger_lowry_r1h8_raw_app_holeout_5_v1"
)
CHAIN_PARENT = (
    REPO / "permanent_data" / "hole_sim_normalized_chain_50k_travelers_tail_v1"
)
REFERENCE_PARENT = REPO / "permanent_data" / "hole_sim_fixed_reference_travelers_v2"

CONTRACT = "independent_lowry_r1h8_source_authority_candidate_audit_v1"
MANIFEST_CONTRACT = "independent_lowry_r1h8_candidate_audit_manifest_v1"
OUTPUT_INDEX = "OUTPUT_SHA256.json"
STATUS = "review_evidence_only_not_authority"

EXPECTED_BUILDER_SHA256 = (
    "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
)
EXPECTED_SEAM_SHA256 = (
    "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
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
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "37a91ecae53d7945132a9eca7871490250b57f978ec95df4f06e477550c9f92c"
)
EXPECTED_QUERY_KEY = (
    "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
)
EXPECTED_QUERY_CONTEXT_SHA256 = (
    "3737e0d82354591d3395204b35fd434299f43e6ced66d817fa1e8032ef276fbd"
)
EXPECTED_PROPOSAL_INVENTORY_SHA256 = (
    "1f3d370647885ac3e9917fc1ebac81a7594b6a11c592b92e86ab65cda362d7a8"
)
EXPECTED_SURFACE_INVENTORY_SHA256 = (
    "2fda69c1b3a666f23d9041023a75e68efe36ed0f68816fde4fa90fd7cbc9ae38"
)
EXPECTED_HAZARD_INVENTORY_SHA256 = (
    "55b324ee433f0e2841e896f0862a5e829c4d40a37358e7fb14fd4d4f93bbb465"
)
EXPECTED_PROPOSAL_MEASURE_SHA256 = (
    "8a646ebc165f86eb9859ea1208fd11bfd1b8efba45128b8a63ea1c7c9a1495f6"
)
EXPECTED_PROPOSAL_ABILITY_SHA256 = (
    "ae861bbdf849764e3d0d7c9ac73e9f9f790e22c88b6ebb36d694912bcdb46cb9"
)
EXPECTED_NUMERIC_PAYLOAD_SHA256 = (
    "44bde6f9a6943c5bb256b549e8de6bed9fa5c23ab9e91e78efb4f8ae2372a0d7"
)
EXPECTED_SOURCE_CACHE_SHA256 = (
    "5adde84d8fc7b5ac89c5a20a08e321c393d32acb452b3dac5390afc1e44b70d2"
)
EXPECTED_ABILITY_POLICY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)
EXPECTED_SURFACE_MODEL_SHA256 = (
    "b5f8a57a4c328e1b30506f899bed3f8ff68705f46ab52b357b89c4c3928c6ae1"
)
EXPECTED_HAZARD_RESOLVER_SHA256 = (
    "f7a674bb282afea5281657feb0f58c88ad4fad444f41c758be3192bde58ebce6"
)
EXPECTED_REFERENCE_MODEL_SHA256 = (
    "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
)
EXPECTED_PUTTING_SHA256 = (
    "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
)
EXPECTED_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"

EXPECTED_QUERY = {
    "arm": "raw",
    "round_no": 1,
    "hole_no": 8,
    "par": 3,
    "target_event_key": "pga:R2026034",
    "target_event_start": "2026-06-25T00:00:00+00:00",
    "target_player_id": "33204",
    "start_lie": "tee",
    "start_distance_yd": 204.54171527805522,
    "start_long_yd": 0.0,
    "start_lat_yd": 0.0,
    "forecast_tee_x": 9094.13,
    "forecast_tee_y": 7540.82,
    "forecast_pin_x": 9211.99,
    "forecast_pin_y": 8143.02,
    "forecast_pin_long_yd": 204.54171527805522,
    "physical_course_id": "geometry:16790842a3043da1",
}

EXPECTED_COUNTS = {
    "proposal_atoms": 904,
    "surface_records": 903,
    "hazard_records": 30_702,
    "complete_atoms": 35_218,
    "terminal_atoms": 1,
    "imputed_proposals": 51,
    "imputed_complete_atoms": 1_989,
    "observed_proposals": 853,
    "provider_total": 1_046,
    "provider_app": 364,
    "provider_ott": 682,
}

EXPECTED_REFERENCE_FILES = {
    "REPORT.md": "00b5b9e4033f97db8a3b6fe08291a6850a3b58ceb9539874d90922fc29ef85d1",
    "manifest.json": "ea37a25af0301e0613078a1d4d837024904e89645cfd0d97967fa1d402d35db4",
    "reference_value_knots.parquet": (
        "55c0c3d2559d018fe50fc7eb1f6c00337bb6c57787af9d9beb6cb5365b462347"
    ),
}

SOURCE_PATHS = {
    "auditor_source": AUDITOR_SOURCE,
    "auditor_test": AUDITOR_TEST,
    "candidate_builder": REPO / "archive" / "hole_sim_tee_app_h8_source_authority.py",
    "complete_seam": REPO / "archive" / "hole_sim_tee_app_complete_atom_support.py",
    "proposal_ledger": REPO / "archive" / "hole_sim_streaming_transition_ledger.py",
    "streaming": REPO / "archive" / "hole_sim_normalized_chain_streaming.py",
    "full_chain": REPO / "archive" / "hole_sim_full_chain.py",
    "batched_population": REPO / "archive" / "hole_sim_batched_population_transition.py",
    "hazard": REPO / "archive" / "hole_sim_hazard_resolution.py",
    "spatial": REPO / "archive" / "hole_sim_tee_app_integrated_v2.py",
    "tee": REPO / "archive" / "hole_sim_tee_transition.py",
    "app": REPO / "archive" / "hole_sim_app_transition.py",
    "fixed_reference": REPO / "archive" / "hole_sim_fixed_reference_value.py",
    "ability_policy": REPO / "archive" / "hole_sim_donor_ability_policy.py",
    "chain_adapter": REPO / "archive" / "hole_sim_transition_mean_chain_adapter.py",
    "precompute": REPO / "archive" / "hole_sim_normalized_atom_precompute.py",
    "asof": REPO / "archive" / "hole_sim_asof_spatial_profiles.py",
    "normalizer": REPO / "archive" / "hole_sim_field_normalization.py",
    "hook": REPO / "archive" / "hole_sim_normalized_chain_hook.py",
    "markov": REPO / "archive" / "hole_sim_markov.py",
    "state_foundations": REPO / "archive" / "hole_sim_state_foundations.py",
    "tail_policy": REPO / "archive" / "hole_sim_tail_policy.py",
    "production_target": REPO / "archive" / "hole_sim_production_anchor_inputs.py",
    "traced_population": REPO / "archive" / "hole_sim_traced_population_transition.py",
}

EXPECTED_PINNED_SOURCE_SHA256 = {
    "candidate_builder": EXPECTED_BUILDER_SHA256,
    "complete_seam": EXPECTED_SEAM_SHA256,
    "proposal_ledger": "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97",
    "streaming": "0d8c2be5027c5a5325d78eea6205691091455104624f6a9acb3006de5993f807",
    "full_chain": "10d8971b45843bc74d4dbd7b5c53d94cdb3f35e0e8621bcba357acce77a20c89",
    "batched_population": "0da3a934b671c3ed74a24c8c754a8404e3a329eb4f54f3b429152e45cd7fca23",
    "hazard": "8a3d8e019c39156299ebde6e7e8a344ba2edf5b55dc8602981ab75e4466163e5",
    "spatial": "dc85b38b5601ba5f6afbe2ce4d900e109243f60cb6a9dd7a03426b436443ea97",
    "tee": "006f1de02d98be47dfd621a5e46d14672466ba6ee673cf7e0f56bc94daf56b43",
    "app": "6acceff91e9130e3584d02d4d113daecc5e84831dfe3fff6f7190d40f5450e1f",
    "fixed_reference": "39df1d57a6623b49f90f3ae82cb9867cb77a3f7cff740754794cdef324ba8b8b",
    "ability_policy": "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339",
    "chain_adapter": "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    "precompute": "3bd580b135f343cf2dbb0f9347b8678dd9c6eace66218826852f411e02d6e773",
    "asof": "9463aa4b0c2fd67adec11ccd109126d0bbf144088dca8095ad5965216482df3c",
    "normalizer": "57beb668622f2635dd47bc60b05de323ee05cae63247f0afdafb71e6e4b75e17",
    "hook": "311523564b178227ade614ffa598ed9787142b1a23136da9ea71f5c8b37e03a7",
    "markov": "6740b3c2ce6119084bbb5371a3ea84ab159f5eeeb1db746188fe25d5ab67a4cf",
    "state_foundations": "3f499f4a4746f896f3cf8ae39465cc6be4ab5cd5d7688151e99886bd4a381eaf",
    "tail_policy": "1f01e4719febe527cf064c3e5e0ff7281958c1d8adcd7027352d83dd9d95f4de",
    "production_target": "d9823f03c613b1985a73aa9d32fa0d68153f670924c16d297870ac19ebe5dc89",
    "traced_population": "3967ab682b28b3848481450ffc405c18b111e4ab9806384d51a4f8f168114a42",
}

PROPOSAL_COLUMNS = (
    "query_key", "proposal_support_key", "proposal_row_order",
    "proposal_atom_id", "category", "proposal_support_measure_sha256",
    "numeric_payload_sha256", "source_cache_sha256",
    "proposal_base_hierarchy_mass", "proposal_spatial_model_weight",
    "proposal_probability", "proposal_branch",
    "proposal_branch_conditional_probability", "proposal_cdf_lower",
    "proposal_cdf_upper", "proposal_branch_cdf_lower",
    "proposal_branch_cdf_upper", "frozen_spatial_cdf_lower",
    "frozen_spatial_cdf_upper", "frozen_terminal_residual_cdf_lower",
    "frozen_terminal_residual_cdf_upper", "terminal", "holed",
    "holeout_representation", "declared_app_base_holeout_probability",
    "implicit_holeout_probability", "proposal_app_holeout_equivalence_sha256",
    "donor_event_key", "donor_player_id", "donor_player_name",
    "donor_round_no", "donor_hole_no", "donor_stroke_no",
    "donor_event_start", "donor_skill_date", "bridge_row_id",
    "target_event_key", "target_event_start", "donor_ability_source_column",
    "donor_ability_available", "donor_ability_raw_strict_prior",
    "donor_ability_value_kind", "donor_ability_sg",
    "donor_ability_was_clipped", "donor_ability_policy_sha256",
    "donor_ability_row_sha256", "donor_ability_snapshot_kind",
    "proposal_policy_binding_sha256",
)
SURFACE_COLUMNS = (
    "query_key", "proposal_row_order", "proposal_atom_id",
    "proposal_endpoint_long_yd", "proposal_endpoint_lat_yd",
    "surface_model_sha256", "surface_class_count",
    "surface_full_probability_json", "surface_full_cdf_upper_json",
    "surface_selection_boundary",
)
HAZARD_COLUMNS = (
    "query_key", "proposal_row_order", "proposal_atom_id",
    "surface_class_index", "surface_class_label", "hazard_resolver_sha256",
    "hazard_donor_order", "hazard_donor_key", "hazard_probability",
    "hazard_cdf_lower", "hazard_cdf_upper", "hazard_donor_event_key",
    "hazard_donor_season", "hazard_donor_round_no", "hazard_donor_hole_no",
    "hazard_donor_category", "hazard_donor_causal_endpoint_x",
    "hazard_donor_causal_endpoint_y", "hazard_penalty_count",
    "hazard_penalty_stroke_cost", "hazard_resolution_shot_key",
    "hazard_resolution_event_type", "hazard_resolution_x",
    "hazard_resolution_y", "hazard_resolution_lie",
    "hazard_donor_physical_course_id", "hazard_foundation_version",
    "hazard_coordinate_reference_year",
    "hazard_coordinate_reference_event_key",
    "hazard_coordinate_reference_mode", "hazard_data_role",
    "hazard_support_source", "hazard_fallback_used", "hazard_support_setup",
    "hazard_support_hole", "hazard_support_category",
    "hazard_component_masses",
    "hazard_nearest_category_endpoint_distance_yd",
    "hazard_nearest_same_hole_endpoint_distance_yd",
    "hazard_support_radius_limit_yd", "hazard_causal_endpoint_x",
    "hazard_causal_endpoint_y", "penalty_strokes", "end_lie",
    "end_distance_yd", "end_long_yd", "end_lat_yd",
)

INVENTORY_SPECS = {
    "proposal": (
        "ordered_tee_app_proposal_source_inventory_v1",
        PROPOSAL_COLUMNS,
        EXPECTED_COUNTS["proposal_atoms"],
    ),
    "surface": (
        "ordered_tee_app_surface_source_inventory_v1",
        SURFACE_COLUMNS,
        EXPECTED_COUNTS["surface_records"],
    ),
    "hazard": (
        "ordered_tee_app_hazard_source_inventory_v1",
        HAZARD_COLUMNS,
        EXPECTED_COUNTS["hazard_records"],
    ),
}

END_LIES = (
    "green", "fairway", "rough", "sand", "recovery", "tee", "hole",
    "penalty", "other",
)

PROPOSAL_MEASURE_COLUMNS = (
    "arm", "support_key", "proposal_row_order", "target_event_key",
    "target_event_start", "target_player_id", "target_round_no",
    "target_hole_no", "target_par", "target_start_lie",
    "target_distance_bucket", "atom_id", "category", "atom_kind",
    "terminal", "holed", "donor_event_key", "donor_player_id",
    "source_shot_key", "base_hierarchy_mass", "spatial_model_weight",
    "quality_base_mass", "proposal_probability", "proposal_cdf_lower",
    "proposal_cdf_upper", "holeout_representation",
    "declared_app_base_holeout_probability", "implicit_holeout_probability",
    "app_holeout_equivalence_sha256", "proposal_branch", "branch_total_mass",
    "branch_conditional_probability", "branch_cdf_lower", "branch_cdf_upper",
    "source_cache_sha256", "numeric_payload_sha256",
)
PROPOSAL_ABILITY_COLUMNS = (
    "arm", "support_key", "proposal_row_order", "atom_id",
    "proposal_support_measure_sha256", "donor_event_key", "donor_player_id",
    "donor_event_start", "donor_skill_date", "donor_ability_sg",
    "donor_ability_available", "donor_ability_snapshot_kind",
)
POLICY_BINDING_COLUMNS = (
    "proposal_support_key", "proposal_row_order", "proposal_atom_id",
    "proposal_support_measure_sha256", "bridge_row_id", "category",
    "donor_player_id", "donor_event_key", "donor_event_start",
    "target_event_key", "target_event_start", "donor_skill_date",
    "donor_ability_source_column", "donor_ability_available",
    "donor_ability_raw_strict_prior", "donor_ability_value_kind",
    "donor_ability_sg", "donor_ability_was_clipped",
    "donor_ability_policy_sha256", "donor_ability_row_sha256",
    "donor_ability_snapshot_kind",
)
HAZARD_IDENTITY_COLUMNS = (
    "donor_key", "event_key", "season", "round_no", "hole_no", "category",
    "causal_endpoint_x", "causal_endpoint_y", "penalty_count",
    "penalty_stroke_cost", "resolution_shot_key", "resolution_event_type",
    "resolution_x", "resolution_y", "resolution_lie", "physical_course_id",
    "foundation_version", "coordinate_reference_year",
    "coordinate_reference_event_key", "coordinate_reference_mode", "data_role",
)


class AuditError(ValueError):
    """Raised when the independent audit cannot reproduce the candidate."""


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


def _canonical_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return {"type": "null"}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return {"type": "null"}
        parsed = (
            parsed.tz_localize("UTC")
            if parsed.tzinfo is None
            else parsed.tz_convert("UTC")
        )
        return {"type": "timestamp", "value": parsed.isoformat()}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return {"type": "null"}
        if not math.isfinite(number):
            raise AuditError("Identity hashes reject non-finite floats")
        if number == 0.0:
            number = 0.0
        return {"type": "float", "value": number.hex()}
    return {
        "type": "str",
        "value": unicodedata.normalize("NFKC", str(value)).strip(),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return None
        parsed = (
            parsed.tz_localize("UTC")
            if parsed.tzinfo is None
            else parsed.tz_convert("UTC")
        )
        return parsed.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise AuditError("JSON output rejects non-finite floats")
        return number
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def _require_exact(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise AuditError(f"{label} changed: expected={expected!r}, observed={observed!r}")


def _require_maps_equal(
    start: Mapping[str, object], end: Mapping[str, object], label: str
) -> None:
    if dict(start) != dict(end):
        changed = {
            key: {"start": start.get(key), "end": end.get(key)}
            for key in sorted(set(start) | set(end))
            if start.get(key) != end.get(key)
        }
        raise AuditError(f"{label} changed start-to-end: {changed}")


def _safe_relative(value: object) -> str:
    relative = str(value)
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.name == OUTPUT_INDEX
    ):
        raise AuditError(f"Unsafe or reserved indexed path: {relative!r}")
    return relative


def _tree_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise AuditError(f"Closed tree contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _read_output_index(root: Path) -> dict[str, Mapping[str, object]]:
    payload = _read_json(Path(root) / OUTPUT_INDEX)
    if set(payload) == {"contract", "files"}:
        if payload.get("contract") != "non_circular_output_sha256_v1":
            raise AuditError("Wrapped output-index contract changed")
        raw = payload.get("files")
        if not isinstance(raw, Mapping):
            raise AuditError("Wrapped output index has no files map")
        payload = dict(raw)
    result: dict[str, Mapping[str, object]] = {}
    for name, metadata in payload.items():
        relative = _safe_relative(name)
        if not isinstance(metadata, Mapping) or set(metadata) != {"bytes", "sha256"}:
            raise AuditError(f"Invalid checksum metadata: {relative}")
        result[relative] = metadata
    return result


def _audit_closed_tree(
    root: Path,
    *,
    expected_index_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    root = Path(root)
    index_sha = _sha256_file(root / OUTPUT_INDEX)
    if expected_index_sha256 is not None:
        _require_exact(index_sha, expected_index_sha256, f"{root.name} output index")
    manifest_sha = _sha256_file(root / "manifest.json")
    if expected_manifest_sha256 is not None:
        _require_exact(manifest_sha, expected_manifest_sha256, f"{root.name} manifest")
    entries = _read_output_index(root)
    actual = _tree_files(root)
    expected = set(entries) | {OUTPUT_INDEX}
    if actual != expected:
        raise AuditError(
            f"{root.name} tree is not closed: "
            f"missing={sorted(expected - actual)}, added={sorted(actual - expected)}"
        )
    total = 0
    for relative, metadata in entries.items():
        path = root / PurePosixPath(relative)
        size = int(path.stat().st_size)
        digest = _sha256_file(path)
        if size != int(metadata["bytes"]) or digest != str(metadata["sha256"]):
            raise AuditError(f"{root.name} checksum mismatch: {relative}")
        total += size
    return {
        "root": root.relative_to(REPO).as_posix(),
        "indexed_files": len(entries),
        "files_including_index": len(actual),
        "indexed_bytes": total,
        "output_index_sha256": index_sha,
        "manifest_sha256": manifest_sha,
        "closed_exact": True,
        "symlinks": 0,
    }


def _write_output_index(root: Path) -> Path:
    root = Path(root)
    path = root / OUTPUT_INDEX
    if path.exists():
        raise FileExistsError(path)
    entries: dict[str, object] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise AuditError(f"Audit output contains a symlink: {item}")
        if not item.is_file():
            continue
        relative = _safe_relative(item.relative_to(root).as_posix())
        entries[relative] = {
            "bytes": int(item.stat().st_size),
            "sha256": _sha256_file(item),
        }
    _write_json(path, entries)
    return path


class _IndependentOrderedHasher:
    def __init__(self, contract: str, columns: Sequence[str]):
        self.contract = str(contract)
        self.columns = tuple(map(str, columns))
        self.records = 0
        self.chain = _stable_sha({"contract": self.contract, "genesis": True})

    def add(self, values: Mapping[str, object]) -> None:
        if any(column not in values for column in self.columns):
            raise AuditError("Ordered inventory row misses a required column")
        record = {
            column: _canonical_scalar(values[column]) for column in self.columns
        }
        self.chain = _stable_sha(
            {
                "contract": self.contract,
                "prior_sha256": self.chain,
                "record_order": self.records,
                "record": record,
            }
        )
        self.records += 1

    def hexdigest(self) -> str:
        return _stable_sha(
            {
                "contract": self.contract,
                "records": self.records,
                "chain_sha256": self.chain,
            }
        )


def _ordered_frame_hash(
    frame: pd.DataFrame, contract: str, columns: Sequence[str]
) -> str:
    expected_columns = ("inventory_record_order", *tuple(columns))
    if tuple(frame.columns) != expected_columns:
        raise AuditError("Inventory schema or column order changed")
    order = frame["inventory_record_order"].to_numpy(np.int64)
    if not np.array_equal(order, np.arange(len(frame), dtype=np.int64)):
        raise AuditError("Inventory record order is not contiguous")
    hasher = _IndependentOrderedHasher(contract, columns)
    for row in frame.loc[:, list(columns)].to_dict(orient="records"):
        hasher.add(row)
    return hasher.hexdigest()


def _read_inventory(
    name: str, candidate: Mapping[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_inventories = candidate.get("inventories")
    if not isinstance(raw_inventories, Mapping):
        raise AuditError("Candidate inventory metadata is absent")
    metadata = raw_inventories.get(name)
    if not isinstance(metadata, Mapping):
        raise AuditError(f"Candidate {name} inventory metadata is absent")
    contract, columns, expected_records = INVENTORY_SPECS[name]
    _require_exact(metadata.get("contract"), contract, f"{name} contract")
    _require_exact(metadata.get("columns"), list(columns), f"{name} columns")
    _require_exact(int(metadata.get("records", -1)), expected_records, f"{name} rows")
    _require_exact(int(metadata.get("shard_rows_cap", -1)), 8_192, f"{name} shard cap")
    raw_shards = metadata.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise AuditError(f"{name} has no inventory shards")
    parts: list[pd.DataFrame] = []
    next_order = 0
    shard_facts = []
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, Mapping):
            raise AuditError(f"{name} shard metadata is invalid")
        relative = _safe_relative(raw_shard.get("path"))
        path = CANDIDATE / PurePosixPath(relative)
        size = int(path.stat().st_size)
        digest = _sha256_file(path)
        _require_exact(size, int(raw_shard.get("bytes", -1)), f"{name} shard bytes")
        _require_exact(digest, str(raw_shard.get("sha256")), f"{name} shard hash")
        frame = pd.read_parquet(path)
        if len(frame) > 8_192:
            raise AuditError(f"{name} shard exceeds the fixed cap")
        if tuple(frame.columns) != ("inventory_record_order", *columns):
            raise AuditError(f"{name} shard schema changed")
        expected_order = np.arange(next_order, next_order + len(frame), dtype=np.int64)
        if not np.array_equal(
            frame["inventory_record_order"].to_numpy(np.int64), expected_order
        ):
            raise AuditError(f"{name} shard order changed")
        _require_exact(
            int(raw_shard.get("first_inventory_record_order", -1)),
            next_order,
            f"{name} shard first order",
        )
        _require_exact(
            int(raw_shard.get("last_inventory_record_order", -1)),
            next_order + len(frame) - 1,
            f"{name} shard last order",
        )
        _require_exact(int(raw_shard.get("records", -1)), len(frame), f"{name} shard rows")
        shard_facts.append(
            {
                "path": relative,
                "records": len(frame),
                "first": next_order,
                "last": next_order + len(frame) - 1,
                "bytes": size,
                "sha256": digest,
            }
        )
        next_order += len(frame)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    _require_exact(len(combined), expected_records, f"{name} combined rows")
    digest = _ordered_frame_hash(combined, contract, columns)
    _require_exact(digest, str(metadata.get("inventory_sha256")), f"{name} digest")
    return combined, {
        "contract": contract,
        "records": len(combined),
        "sha256": digest,
        "shards": shard_facts,
    }


def _candidate_id(candidate: Mapping[str, object]) -> str:
    body = dict(candidate)
    body.pop("candidate_id", None)
    return _stable_sha(
        {
            "contract": "lowry_r1h8_tee_app_source_authority_candidate_v1",
            "role": "review_only_non_authorizing_candidate",
            "payload": body,
        }
    )


def _assert_non_authorizing(candidate: Mapping[str, object]) -> None:
    required = {
        "authority_id": None,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": "candidate_external_review_required",
        "review_lock_sha256": None,
        "candidate_digest_is_not_authority": True,
    }
    for field, expected in required.items():
        if candidate.get(field) != expected:
            raise AuditError(f"Candidate is not fail-closed: {field}")


def _query_identities(query: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, str]:
    _require_exact(dict(query), EXPECTED_QUERY, "exact Lowry R1 H8 query")
    canonical = {
        key: _canonical_scalar(
            pd.Timestamp(value) if key == "target_event_start" else value
        )
        for key, value in query.items()
    }
    context = _stable_sha(
        {"contract": "exact_tee_app_generated_start_state_v2", "query": canonical}
    )
    query_key = _stable_sha(
        {
            "contract": "exact_tee_app_generated_start_state_v2",
            "query": canonical,
            "proposal_support_measure_sha256": candidate[
                "proposal_support_measure_sha256"
            ],
            "numeric_payload_sha256": candidate["numeric_payload_sha256"],
            "surface_model_sha256": candidate["surface_model_sha256"],
            "hazard_resolver_sha256": candidate["hazard_resolver_sha256"],
            "reference_model_sha256": candidate["reference_model_sha256"],
            "ability_policy_sha256": candidate["ability_policy_sha256"],
        }
    )
    return {"query_context_sha256": context, "query_key": query_key}


def _frame_digest(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    def canonical_json(value: object) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    digest = sha256(canonical_json(list(columns)).encode("utf-8"))
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        digest.update(canonical_json(list(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bridge_id(support_key: object, order: object, atom_id: object) -> str:
    return _stable_sha(
        {
            "contract": "tee_app_proposal_ability_bridge_row_v1",
            "support_key": str(support_key),
            "proposal_row_order": int(order),
            "atom_id": str(atom_id),
        }
    )


def _policy_stable_sha(value: object) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _policy_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise AuditError("Policy row contains an invalid timestamp")
    return pd.Timestamp(parsed)


def _policy_timestamp_text(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _independent_policy_expected(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy(deep=True).reset_index(drop=True)
    kinds: list[str] = []
    applied: list[float] = []
    clipped: list[bool] = []
    bridge_ids: list[str] = []
    source_columns: list[str] = []
    raw_values: list[float | None] = []
    row_hashes: list[str] = []
    snapshot_kinds: list[str] = []
    for _, row in raw.iterrows():
        bridge = _bridge_id(row["support_key"], row["proposal_row_order"], row["atom_id"])
        available = row["donor_ability_available"]
        if not isinstance(available, (bool, np.bool_)):
            raise AuditError("Policy availability must be explicit boolean")
        available = bool(available)
        donor_event = _policy_timestamp(row["donor_event_start"])
        target_event = _policy_timestamp(row["target_event_start"])
        skill_date = _policy_timestamp(row["donor_skill_date"])
        if donor_event is None or target_event is None or not donor_event < target_event:
            raise AuditError("Policy event cutoff changed")
        raw_value = row["donor_ability_sg"]
        missing = raw_value is None or raw_value is pd.NA or pd.isna(raw_value)
        if available:
            if missing or skill_date is None or not skill_date < donor_event:
                raise AuditError("Observed policy row is not strict prior")
            raw_number = float(raw_value)
            value = float(np.clip(raw_number, -2.0, 2.0))
            kind = "observed_strict_prior"
            was_clipped = value != raw_number
            snapshot = "strict_prior_asof_content_verified_v2"
        else:
            if not missing or (skill_date is not None and not skill_date < donor_event):
                raise AuditError("Missing policy row is not honest")
            raw_number = None
            value = 0.0
            kind = "imputed_frozen_neutral_zero"
            was_clipped = False
            snapshot = "strict_prior_missing_imputed_frozen_neutral_zero_v2"
        source_column = "sg_app"
        payload = {
            "bridge_row_id": bridge,
            "category": "app",
            "donor_player_id": str(row["donor_player_id"]),
            "donor_event_key": str(row["donor_event_key"]),
            "donor_event_start": _policy_timestamp_text(donor_event),
            "target_event_key": str(row["target_event_key"]),
            "target_event_start": _policy_timestamp_text(target_event),
            "donor_skill_date": _policy_timestamp_text(skill_date),
            "donor_ability_source_column": source_column,
            "donor_ability_available": available,
            "donor_ability_raw_strict_prior": raw_number,
            "donor_ability_value_kind": kind,
            "donor_ability_sg": value,
            "donor_ability_was_clipped": was_clipped,
            "donor_ability_policy_sha256": EXPECTED_ABILITY_POLICY_SHA256,
        }
        bridge_ids.append(bridge)
        source_columns.append(source_column)
        raw_values.append(raw_number)
        kinds.append(kind)
        applied.append(value)
        clipped.append(was_clipped)
        row_hashes.append(_policy_stable_sha(payload))
        snapshot_kinds.append(snapshot)
    result["bridge_row_id"] = bridge_ids
    result["donor_ability_source_column"] = source_columns
    result["donor_ability_raw_strict_prior"] = raw_values
    result["donor_ability_value_kind"] = kinds
    result["donor_ability_sg"] = np.asarray(applied, dtype=float)
    result["donor_ability_was_clipped"] = np.asarray(clipped, dtype=bool)
    result["donor_ability_policy_sha256"] = EXPECTED_ABILITY_POLICY_SHA256
    result["donor_ability_row_sha256"] = row_hashes
    result["donor_ability_snapshot_kind"] = snapshot_kinds
    return result


def _array_sha(*arrays: np.ndarray) -> str:
    digest = sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        if array.dtype.kind in {"O", "U", "S"}:
            for item in array.reshape(-1):
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _support_payload(
    key: str,
    atom_ids: np.ndarray,
    points: np.ndarray,
    probability: np.ndarray,
    cdf: np.ndarray,
    holeout_probability: float,
) -> str:
    digest = sha256(str(key).encode("utf-8"))
    digest.update(
        _array_sha(
            atom_ids,
            points,
            probability,
            cdf,
            np.asarray([holeout_probability], dtype=float),
        ).encode("ascii")
    )
    return digest.hexdigest()


def _surface_reconstruction(
    snapshot: pd.DataFrame,
    endpoints: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, int]]:
    expected_columns = (
        "surface_row_order", "hole_no", "spatial_valid", "is_holeout",
        "common_end_long_yd", "common_end_lat_yd", "end_lie_class",
    )
    if tuple(snapshot.columns) != expected_columns:
        raise AuditError("Surface snapshot schema changed")
    if not np.array_equal(
        snapshot["surface_row_order"].to_numpy(np.int64),
        np.arange(len(snapshot), dtype=np.int64),
    ):
        raise AuditError("Surface snapshot order changed")
    if not snapshot["hole_no"].eq(8).all():
        raise AuditError("Surface snapshot contains another hole")
    regular = snapshot.loc[
        snapshot["spatial_valid"].astype(bool)
        & ~snapshot["is_holeout"].astype(bool)
    ].copy()
    points = regular[["common_end_long_yd", "common_end_lat_yd"]].to_numpy(float)
    codes = pd.Categorical(regular["end_lie_class"], categories=END_LIES).codes.astype(int)
    if (codes < 0).any():
        raise AuditError("Surface snapshot has an unknown lie")
    counts = np.bincount(codes, minlength=len(END_LIES)).astype(float)
    counts[END_LIES.index("hole")] = 0.0
    prior = (counts + 0.5) / (counts.sum() + 0.5 * len(END_LIES))
    prior[END_LIES.index("hole")] = 0.0
    prior /= prior.sum()
    array_digest = sha256()
    array_digest.update(np.ascontiguousarray(points).tobytes())
    array_digest.update(np.ascontiguousarray(codes).tobytes())
    array_digest.update(np.ascontiguousarray(prior).tobytes())
    model_sha = _stable_sha(
        {
            "contract": "empirical_endpoint_surface_model_v2",
            "category": "app",
            "hole_no": 8,
            "class_labels": list(END_LIES),
            "prior_strength": float(10.0).hex(),
            "array_sha256": array_digest.hexdigest(),
            "surface_type": "hole_sim_app_transition.AbsoluteSurfaceMap",
            "surface_implementation_sha256": _sha256_file(SOURCE_PATHS["app"]),
            "surface_support_mask_sha256": _sha256_file(SOURCE_PATHS["spatial"]),
        }
    )
    tree = cKDTree(points)
    distance, index = tree.query(endpoints, k=min(120, len(points)))
    local_scale = np.maximum(np.median(distance, axis=1), 3.0)
    weight = np.exp(-0.5 * (distance / local_scale[:, None]) ** 2)
    probabilities = np.tile(prior * 10.0, (len(endpoints), 1))
    for code in range(len(END_LIES)):
        probabilities[:, code] += np.sum(weight * (codes[index] == code), axis=1)
    probabilities[:, END_LIES.index("hole")] = 0.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[:, counts == 0.0] = 0.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return model_sha, probabilities, {
        label: int(value) for label, value in zip(END_LIES, counts)
    }


def _hazard_resolver_sha(provider: pd.DataFrame, params: Mapping[str, object]) -> tuple[str, str]:
    donors = provider.sort_values("donor_key", kind="mergesort").reset_index(drop=True)
    records = [
        {column: _canonical_scalar(row[column]) for column in HAZARD_IDENTITY_COLUMNS}
        for row in donors.loc[:, list(HAZARD_IDENTITY_COLUMNS)].to_dict(orient="records")
    ]
    donor_panel_sha = _stable_sha(
        {"contract": "hazard_donor_panel_v1", "records": records}
    )
    resolver_sha = _stable_sha(
        {
            "contract": "exact_endpoint_linked_hazard_distribution_v1",
            "resolver_version": params["resolver_version"],
            "foundation_version": params["foundation_version"],
            "donor_panel_sha256": donor_panel_sha,
            "physical_course_id": params["physical_course_id"],
            "coordinate_reference_year": int(params["coordinate_reference_year"]),
            "coordinate_reference_event_key": params[
                "coordinate_reference_event_key"
            ],
            "setup_prior": float(params["setup_prior"]).hex(),
            "hole_prior": float(params["hole_prior"]).hex(),
            "spatial_bandwidth_yd": float(params["spatial_bandwidth_yd"]).hex(),
            "max_nearest_distance_yd": float(
                params["max_nearest_distance_yd"]
            ).hex(),
            "same_hole_fallback_max_distance_yd": float(
                params["same_hole_fallback_max_distance_yd"]
            ).hex(),
            "minimum_category_support": int(params["minimum_category_support"]),
        }
    )
    return resolver_sha, donor_panel_sha


def _hazard_replay(
    provider: pd.DataFrame,
    surface: pd.DataFrame,
    hazard_inventory: pd.DataFrame,
    query: Mapping[str, object],
) -> dict[str, object]:
    donors = provider.sort_values("donor_key", kind="mergesort").reset_index(drop=True)
    app = donors.loc[donors["category"].eq("app")].copy()
    app_xy = app[["causal_endpoint_x", "causal_endpoint_y"]].to_numpy(float)
    vector = np.asarray(
        [
            float(query["forecast_pin_x"]) - float(query["forecast_tee_x"]),
            float(query["forecast_pin_y"]) - float(query["forecast_tee_y"]),
        ]
    )
    unit = vector / np.linalg.norm(vector)
    perpendicular = np.asarray([-unit[1], unit[0]])
    surface_by_order = surface.set_index("proposal_row_order")
    key_errors = 0
    support_errors = 0
    component_errors = 0
    source_errors = 0
    max_weight_residual = 0.0
    max_nearest_residual = 0.0
    max_same_hole_residual = 0.0
    for proposal_order, observed in hazard_inventory.groupby(
        "proposal_row_order", sort=False
    ):
        endpoint = surface_by_order.loc[int(proposal_order)]
        absolute = np.asarray(
            [float(query["forecast_tee_x"]), float(query["forecast_tee_y"])]
        ) + 3.0 * (
            float(endpoint["proposal_endpoint_long_yd"]) * unit
            + float(endpoint["proposal_endpoint_lat_yd"]) * perpendicular
        )
        distances = np.hypot(
            app_xy[:, 0] - absolute[0], app_xy[:, 1] - absolute[1]
        ) / 3.0
        nearest = float(distances.min())
        same_hole = app["hole_no"].eq(8).to_numpy()
        nearest_same_hole = float(distances[same_hole].min())
        within = distances <= 120.0
        candidates = app.loc[within].copy()
        candidates["_distance"] = distances[within]
        setup = candidates["round_no"].eq(1) & candidates["hole_no"].eq(8)
        hole = candidates["hole_no"].eq(8)
        pools = (
            ("setup", setup, 12.0),
            ("hole", hole, 40.0),
            ("category", pd.Series(True, index=candidates.index), math.inf),
        )
        mass = pd.Series(0.0, index=candidates.index)
        remaining = 1.0
        supports: dict[str, int] = {}
        components: dict[str, float] = {}
        for name, mask, prior in pools:
            pool = candidates.loc[mask]
            supports[name] = len(pool)
            if pool.empty:
                components[name] = 0.0
                continue
            values = pool["_distance"].to_numpy(float)
            bandwidth = max(
                24.0,
                float(np.quantile(values, 0.40)) if len(values) > 2 else 0.0,
            )
            kernel = np.exp(-0.5 * (values / bandwidth) ** 2)
            kernel = np.maximum(kernel, 1e-12)
            kernel /= kernel.sum()
            fraction = 1.0 if not np.isfinite(prior) else len(pool) / (len(pool) + prior)
            component = remaining * fraction
            mass.loc[pool.index] += component * kernel
            remaining -= component
            components[name] = float(component)
        positive = mass[mass > 0.0]
        selected = candidates.loc[positive.index].copy()
        weights = positive.to_numpy(float)
        weights /= weights.sum()
        order = np.argsort(selected["donor_key"].astype(str).to_numpy())
        keys = selected.iloc[order]["donor_key"].astype(str).tolist()
        weights = weights[order]
        if keys != observed["hazard_donor_key"].astype(str).tolist():
            key_errors += 1
        max_weight_residual = max(
            max_weight_residual,
            float(np.max(np.abs(weights - observed["hazard_probability"].to_numpy(float)))),
        )
        max_nearest_residual = max(
            max_nearest_residual,
            abs(nearest - float(observed["hazard_nearest_category_endpoint_distance_yd"].iloc[0])),
        )
        max_same_hole_residual = max(
            max_same_hole_residual,
            abs(nearest_same_hole - float(observed["hazard_nearest_same_hole_endpoint_distance_yd"].iloc[0])),
        )
        expected_support = {
            "setup": int(observed["hazard_support_setup"].iloc[0]),
            "hole": int(observed["hazard_support_hole"].iloc[0]),
            "category": int(observed["hazard_support_category"].iloc[0]),
        }
        support_errors += int(supports != expected_support)
        component_text = ",".join(
            f"{name}={float(value):.17g}" for name, value in components.items()
        )
        component_errors += int(
            component_text != str(observed["hazard_component_masses"].iloc[0])
        )
        source_errors += int(
            not (
                observed["hazard_support_source"].eq(
                    "same_category_primary_radius"
                ).all()
                and (~observed["hazard_fallback_used"].astype(bool)).all()
                and observed["hazard_support_radius_limit_yd"].eq(120.0).all()
            )
        )
    facts = {
        "groups": int(hazard_inventory["proposal_row_order"].nunique()),
        "donor_key_errors": key_errors,
        "support_errors": support_errors,
        "component_mass_errors": component_errors,
        "source_errors": source_errors,
        "max_weight_residual": max_weight_residual,
        "max_nearest_category_residual_yd": max_nearest_residual,
        "max_nearest_same_hole_residual_yd": max_same_hole_residual,
    }
    if any(facts[name] for name in (
        "donor_key_errors", "support_errors", "component_mass_errors", "source_errors"
    )) or max_weight_residual > 3e-14:
        raise AuditError(f"Independent hazard replay disagrees: {facts}")
    return facts


def _frame_logical_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(map(str, frame.columns))
    work = frame.reindex(columns=columns).copy()
    for name in columns:
        if work[name].dtype == object:
            work[name] = work[name].map(
                lambda value: (
                    json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (list, tuple, dict, set, np.ndarray))
                    else str(value)
                )
            )
    hashes = pd.util.hash_pandas_object(
        work, index=False, categorize=True
    ).to_numpy(np.uint64)
    hashes.sort()
    digest = sha256(
        json.dumps(
            [(name, str(work[name].dtype)) for name in columns],
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _prepared_input_paths() -> dict[str, Path]:
    export = DEV / "golf_scraping" / "output" / "export"
    profiles = REPO / "permanent_data" / "hole_sim_asof_spatial_profiles_v2"
    reference = (
        REPO
        / "permanent_data"
        / "hole_sim_normalized_atom_precompute_travelers_smoke"
    )
    paths = {
        "streaming_source": SOURCE_PATHS["streaming"],
        "precompute_source": SOURCE_PATHS["precompute"],
        "full_chain_source": SOURCE_PATHS["full_chain"],
        "batched_population_source": SOURCE_PATHS["batched_population"],
        "hazard_source": SOURCE_PATHS["hazard"],
        "spatial_v2_source": SOURCE_PATHS["spatial"],
        "tee_source": SOURCE_PATHS["tee"],
        "app_source": SOURCE_PATHS["app"],
        "asof_source": SOURCE_PATHS["asof"],
        "normalizer_source": SOURCE_PATHS["normalizer"],
        "hook_source": SOURCE_PATHS["hook"],
        "markov_source": SOURCE_PATHS["markov"],
        "state_foundations_source": SOURCE_PATHS["state_foundations"],
        "tail_policy_source": SOURCE_PATHS["tail_policy"],
        "normalization_profile_manifest": profiles / "manifest.json",
        "normalization_profile_tee": profiles / "tee_player_spatial_profiles_asof.parquet",
        "normalization_profile_app": profiles / "app_player_spatial_profiles_asof.parquet",
        "history_db": DEV / "sim_prep" / "work" / "dg_historical.db",
        "baseline_cases": (
            REPO
            / "permanent_data"
            / "spatial_chain_common_baselines"
            / "round_case_pmfs.parquet"
        ),
        "legacy_tee_profiles": (
            REPO / "permanent_data" / "hole_sim_prototype" / "player_tee_profiles.csv"
        ),
        "legacy_app_profiles": (
            REPO
            / "permanent_data"
            / "hole_sim_app_transition_strict"
            / "player_app_profiles_structured.csv"
        ),
        "export_events": export / "events.parquet",
        "reference_manifest": reference / "manifest.json",
        "reference_raw_support": reference / "raw_support_arrays.npz",
        "reference_normalized_support": reference / "normalized_support_arrays.npz",
        "reference_neutral_tee": reference / "neutral_tee_atoms.parquet",
        "reference_neutral_app": reference / "neutral_app_atoms.parquet",
        "reference_legacy_terminal": reference / "legacy_terminal_scores.parquet",
        "reference_selected_atom_ids": reference / "selected_atom_ids.parquet",
    }
    for year in range(2020, 2027):
        for stem in ("shots", "shot_sg", "hole_geometry"):
            paths[f"export_pga_{year}_{stem}"] = export / f"pga_{year}_{stem}.parquet"
    return paths


def _hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise AuditError(f"Audited inputs are missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def _source_state() -> dict[str, str]:
    state = _hash_paths(SOURCE_PATHS)
    for name, expected in EXPECTED_PINNED_SOURCE_SHA256.items():
        _require_exact(state[name], expected, f"pinned source {name}")
    return state


def _artifact_input_state() -> dict[str, str]:
    paths = {
        "candidate_output_index": CANDIDATE / OUTPUT_INDEX,
        "candidate_manifest": CANDIDATE / "manifest.json",
        "candidate_payload": CANDIDATE / "authority_candidate.json",
        "h8_parent_output_index": H8_PARENT / OUTPUT_INDEX,
        "h8_parent_manifest": H8_PARENT / "manifest.json",
        "chain_parent_output_index": CHAIN_PARENT / OUTPUT_INDEX,
        "chain_parent_manifest": CHAIN_PARENT / "manifest.json",
        "reference_manifest": REFERENCE_PARENT / "manifest.json",
        "reference_knots": REFERENCE_PARENT / "reference_value_knots.parquet",
    }
    return _hash_paths(paths)


def _proposal_inventory_source_bridge(
    proposal_inventory: pd.DataFrame, policy: pd.DataFrame
) -> None:
    mapping = {
        "proposal_support_key": "support_key",
        "proposal_row_order": "proposal_row_order",
        "proposal_atom_id": "atom_id",
        "category": "category",
        "proposal_support_measure_sha256": "proposal_support_measure_sha256",
        "numeric_payload_sha256": "numeric_payload_sha256",
        "source_cache_sha256": "source_cache_sha256",
        "proposal_base_hierarchy_mass": "base_hierarchy_mass",
        "proposal_spatial_model_weight": "spatial_model_weight",
        "proposal_probability": "proposal_probability",
        "proposal_branch": "proposal_branch",
        "proposal_branch_conditional_probability": "branch_conditional_probability",
        "proposal_cdf_lower": "proposal_cdf_lower",
        "proposal_cdf_upper": "proposal_cdf_upper",
        "proposal_branch_cdf_lower": "branch_cdf_lower",
        "proposal_branch_cdf_upper": "branch_cdf_upper",
        "terminal": "terminal",
        "holed": "holed",
        "declared_app_base_holeout_probability": "declared_app_base_holeout_probability",
        "implicit_holeout_probability": "implicit_holeout_probability",
        "proposal_app_holeout_equivalence_sha256": "app_holeout_equivalence_sha256",
        "donor_event_key": "donor_event_key",
        "donor_player_id": "donor_player_id",
        "donor_player_name": "donor_player_name",
        "donor_round_no": "donor_round_no",
        "donor_hole_no": "donor_hole_no",
        "donor_stroke_no": "donor_stroke_no",
        "donor_event_start": "donor_event_start",
        "donor_skill_date": "donor_skill_date",
        "bridge_row_id": "bridge_row_id",
        "target_event_key": "target_event_key",
        "target_event_start": "target_event_start",
        "donor_ability_source_column": "donor_ability_source_column",
        "donor_ability_available": "donor_ability_available",
        "donor_ability_raw_strict_prior": "donor_ability_raw_strict_prior",
        "donor_ability_value_kind": "donor_ability_value_kind",
        "donor_ability_sg": "donor_ability_sg",
        "donor_ability_was_clipped": "donor_ability_was_clipped",
        "donor_ability_policy_sha256": "donor_ability_policy_sha256",
        "donor_ability_row_sha256": "donor_ability_row_sha256",
        "donor_ability_snapshot_kind": "donor_ability_snapshot_kind",
    }
    for target, source in mapping.items():
        try:
            pd.testing.assert_series_equal(
                proposal_inventory[target].reset_index(drop=True),
                policy[source].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise AuditError(f"Proposal inventory bridge changed: {target}") from exc
    expected_bindings = [
        _stable_sha(
            {
                "contract": "source_bound_tee_app_proposal_policy_bridge_v1",
                "fields": {
                    column: _canonical_scalar(row[column])
                    for column in POLICY_BINDING_COLUMNS
                },
            }
        )
        for row in proposal_inventory.to_dict(orient="records")
    ]
    if not np.array_equal(
        proposal_inventory["proposal_policy_binding_sha256"].astype(str).to_numpy(),
        np.asarray(expected_bindings, dtype=object),
    ):
        raise AuditError("Proposal policy bindings changed")


def _audit_hazard_source_bridge(
    provider: pd.DataFrame,
    hazard_inventory: pd.DataFrame,
    surface_inventory: pd.DataFrame,
    query: Mapping[str, object],
) -> dict[str, object]:
    indexed = provider.set_index("donor_key", drop=False)
    if not set(hazard_inventory["hazard_donor_key"]).issubset(set(indexed.index)):
        raise AuditError("Hazard inventory contains a donor outside the provider")
    matched = indexed.loc[hazard_inventory["hazard_donor_key"]].reset_index(drop=True)
    mapping = {
        "hazard_donor_event_key": "event_key",
        "hazard_donor_season": "season",
        "hazard_donor_round_no": "round_no",
        "hazard_donor_hole_no": "hole_no",
        "hazard_donor_category": "category",
        "hazard_donor_causal_endpoint_x": "causal_endpoint_x",
        "hazard_donor_causal_endpoint_y": "causal_endpoint_y",
        "hazard_penalty_count": "penalty_count",
        "hazard_penalty_stroke_cost": "penalty_stroke_cost",
        "hazard_resolution_shot_key": "resolution_shot_key",
        "hazard_resolution_event_type": "resolution_event_type",
        "hazard_resolution_x": "resolution_x",
        "hazard_resolution_y": "resolution_y",
        "hazard_resolution_lie": "resolution_lie",
        "hazard_donor_physical_course_id": "physical_course_id",
        "hazard_foundation_version": "foundation_version",
        "hazard_coordinate_reference_year": "coordinate_reference_year",
        "hazard_coordinate_reference_event_key": "coordinate_reference_event_key",
        "hazard_coordinate_reference_mode": "coordinate_reference_mode",
        "hazard_data_role": "data_role",
    }
    for target, source in mapping.items():
        try:
            pd.testing.assert_series_equal(
                hazard_inventory[target].reset_index(drop=True),
                matched[source],
                check_names=False,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise AuditError(f"Hazard provider bridge changed: {target}") from exc
    group_errors = 0
    group_sizes: list[int] = []
    for _, group in hazard_inventory.groupby(
        ["proposal_row_order", "surface_class_index"], sort=False
    ):
        probability = group["hazard_probability"].to_numpy(float)
        upper = np.cumsum(probability)
        lower = np.concatenate(([0.0], upper[:-1]))
        group_sizes.append(len(group))
        group_errors += int(
            not (
                np.array_equal(
                    group["hazard_donor_order"].to_numpy(np.int64),
                    np.arange(len(group), dtype=np.int64),
                )
                and group["hazard_donor_key"].astype(str).tolist()
                == sorted(group["hazard_donor_key"].astype(str).tolist())
                and np.array_equal(group["hazard_cdf_lower"].to_numpy(float), lower)
                and np.array_equal(group["hazard_cdf_upper"].to_numpy(float), upper)
            )
        )
    if group_errors:
        raise AuditError(f"Hazard ordered CDF groups changed: {group_errors}")
    vector = np.asarray(
        [
            float(query["forecast_pin_x"]) - float(query["forecast_tee_x"]),
            float(query["forecast_pin_y"]) - float(query["forecast_tee_y"]),
        ]
    )
    unit = vector / np.linalg.norm(vector)
    perpendicular = np.asarray([-unit[1], unit[0]])
    surface_by_order = surface_inventory.set_index("proposal_row_order")
    endpoints = surface_by_order.loc[hazard_inventory["proposal_row_order"]][
        ["proposal_endpoint_long_yd", "proposal_endpoint_lat_yd"]
    ].to_numpy(float)
    absolute = np.asarray(
        [float(query["forecast_tee_x"]), float(query["forecast_tee_y"])]
    ) + 3.0 * (
        endpoints[:, 0, None] * unit + endpoints[:, 1, None] * perpendicular
    )
    causal_residual = float(
        max(
            np.max(np.abs(absolute[:, 0] - hazard_inventory["hazard_causal_endpoint_x"])),
            np.max(np.abs(absolute[:, 1] - hazard_inventory["hazard_causal_endpoint_y"])),
        )
    )
    delta = hazard_inventory[["hazard_resolution_x", "hazard_resolution_y"]].to_numpy(float) - np.asarray(
        [float(query["forecast_tee_x"]), float(query["forecast_tee_y"])]
    )
    local = np.column_stack([delta @ unit / 3.0, delta @ perpendicular / 3.0])
    local_residual = float(
        max(
            np.max(np.abs(local[:, 0] - hazard_inventory["end_long_yd"])),
            np.max(np.abs(local[:, 1] - hazard_inventory["end_lat_yd"])),
        )
    )
    pin_distance = np.hypot(
        hazard_inventory["hazard_resolution_x"] - float(query["forecast_pin_x"]),
        hazard_inventory["hazard_resolution_y"] - float(query["forecast_pin_y"]),
    ) / 3.0
    distance_residual = float(
        np.max(np.abs(pin_distance - hazard_inventory["end_distance_yd"]))
    )
    if causal_residual > 5e-11 or local_residual != 0.0 or distance_residual != 0.0:
        raise AuditError("Hazard absolute/local geometry bridge changed")
    return {
        "groups": len(group_sizes),
        "group_min_donors": min(group_sizes),
        "group_max_donors": max(group_sizes),
        "ordered_cdf_group_errors": group_errors,
        "used_provider_donors": int(hazard_inventory["hazard_donor_key"].nunique()),
        "causal_absolute_max_residual_feet": causal_residual,
        "resolution_local_max_residual_yd": local_residual,
        "resolution_distance_max_residual_yd": distance_residual,
    }


def _derive_expansion_facts(
    policy: pd.DataFrame,
    surface_inventory: pd.DataFrame,
    hazard_inventory: pd.DataFrame,
) -> dict[str, object]:
    surface = surface_inventory.set_index("proposal_row_order")
    hazard_groups = {
        int(order): group["hazard_probability"].to_numpy(float)
        for order, group in hazard_inventory.groupby("proposal_row_order", sort=False)
    }
    running_mass = 0.0
    max_residual = 0.0
    complete_atoms = 0
    imputed_atoms = 0
    for row in policy.itertuples(index=False):
        order = int(row.proposal_row_order)
        if bool(row.terminal):
            branches = [1.0]
        else:
            probabilities = json.loads(
                str(surface.loc[order, "surface_full_probability_json"])
            )
            branches = []
            for class_index, surface_mass in enumerate(probabilities):
                boundary_only = class_index == 0 and surface_mass == 0.0
                if surface_mass == 0.0 and not boundary_only:
                    continue
                if class_index == 7:
                    branches.extend(
                        float(surface_mass) * float(value)
                        for value in hazard_groups[order]
                    )
                else:
                    branches.append(float(surface_mass))
        observed = 0.0
        for conditional in branches:
            mass = float(row.proposal_probability) * conditional
            observed += mass
            running_mass += mass
        complete_atoms += len(branches)
        if row.donor_ability_value_kind == "imputed_frozen_neutral_zero":
            imputed_atoms += len(branches)
        max_residual = max(
            max_residual, abs(observed - float(row.proposal_probability))
        )
    return {
        "proposal_atoms": len(policy),
        "surface_records": len(surface_inventory),
        "hazard_records": len(hazard_inventory),
        "complete_atoms": complete_atoms,
        "terminal_atoms": int(policy["terminal"].astype(bool).sum()),
        "imputed_complete_atoms": imputed_atoms,
        "quality_mass": running_mass,
        "terminal_mass": float(
            policy.loc[policy["terminal"].astype(bool), "proposal_probability"].sum()
        ),
        "max_proposal_mass_residual": max_residual,
        "nonhazard_nonterminal_atoms": (
            complete_atoms
            - len(hazard_inventory)
            - int(policy["terminal"].astype(bool).sum())
        ),
    }


def _mutation_checks(
    candidate: Mapping[str, object],
    query: Mapping[str, object],
    proposal_inventory: pd.DataFrame,
    hazard_inventory: pd.DataFrame,
    surface_snapshot: pd.DataFrame,
    source_state: Mapping[str, str],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for field, value in (
        ("authority_ready", True),
        ("production_enabled", True),
        ("authority_id", "candidate-derived"),
        ("review_lock_sha256", "d" * 64),
        ("candidate_digest_is_not_authority", False),
        ("admission_status", "admitted"),
    ):
        mutated = dict(candidate)
        mutated[field] = value
        try:
            _assert_non_authorizing(mutated)
        except AuditError:
            checks[f"status_{field}_rejected"] = True
        else:
            checks[f"status_{field}_rejected"] = False
    checks["candidate_id_mutation_detected"] = (
        _candidate_id(candidate) != "0" * 64
    )
    query_mutation = dict(query)
    query_mutation["hole_no"] = 9
    mutated_canonical = {
        key: _canonical_scalar(
            pd.Timestamp(value) if key == "target_event_start" else value
        )
        for key, value in query_mutation.items()
    }
    mutated_context = _stable_sha(
        {
            "contract": "exact_tee_app_generated_start_state_v2",
            "query": mutated_canonical,
        }
    )
    checks["query_mutation_changes_identity"] = (
        mutated_context != EXPECTED_QUERY_CONTEXT_SHA256
    )
    original_hash = _ordered_frame_hash(
        proposal_inventory,
        INVENTORY_SPECS["proposal"][0],
        INVENTORY_SPECS["proposal"][1],
    )
    mutated_inventory = proposal_inventory.copy()
    mutated_inventory.loc[0, "donor_player_id"] = "mutated"
    checks["proposal_inventory_mutation_changes_hash"] = (
        _ordered_frame_hash(
            mutated_inventory,
            INVENTORY_SPECS["proposal"][0],
            INVENTORY_SPECS["proposal"][1],
        )
        != original_hash
    )
    reordered = proposal_inventory.iloc[[1, 0, *range(2, len(proposal_inventory))]].reset_index(drop=True)
    try:
        _ordered_frame_hash(
            reordered,
            INVENTORY_SPECS["proposal"][0],
            INVENTORY_SPECS["proposal"][1],
        )
    except AuditError:
        checks["inventory_reorder_rejected"] = True
    else:
        checks["inventory_reorder_rejected"] = False
    hazard_mutation = hazard_inventory.copy()
    hazard_mutation.loc[0, "hazard_donor_key"] = "0" * 64
    checks["hazard_inventory_mutation_changes_hash"] = (
        _ordered_frame_hash(
            hazard_mutation,
            INVENTORY_SPECS["hazard"][0],
            INVENTORY_SPECS["hazard"][1],
        )
        != EXPECTED_HAZARD_INVENTORY_SHA256
    )
    altered_surface = surface_snapshot.copy()
    altered_surface.loc[0, "common_end_long_yd"] += 1e-6
    dummy_endpoints = np.zeros((1, 2), dtype=float)
    mutated_surface_sha, _, _ = _surface_reconstruction(
        altered_surface, dummy_endpoints
    )
    checks["surface_mutation_changes_model_hash"] = (
        mutated_surface_sha != EXPECTED_SURFACE_MODEL_SHA256
    )
    changed_source = dict(source_state)
    changed_source["complete_seam"] = "0" * 64
    checks["source_drift_detected"] = changed_source != dict(source_state)
    candidate_entries = _read_output_index(CANDIDATE)
    actual_files = _tree_files(CANDIDATE)
    checks["added_file_would_break_closure"] = (
        actual_files | {"unindexed.txt"}
        != set(candidate_entries) | {OUTPUT_INDEX}
    )
    report = CANDIDATE / "REPORT.md"
    checks["byte_mutation_would_break_checksum"] = (
        sha256(report.read_bytes() + b"mutation").hexdigest()
        != str(candidate_entries["REPORT.md"]["sha256"])
    )
    if not all(checks.values()):
        raise AuditError(f"An in-memory mutation check did not fail closed: {checks}")
    return checks


def collect_independent_audit() -> dict[str, object]:
    """Reconstruct the fixed candidate without preparation or expansion."""

    source_start = _source_state()
    prepared_paths = _prepared_input_paths()
    prepared_start = _hash_paths(prepared_paths)
    artifact_start = _artifact_input_state()

    candidate_closure = _audit_closed_tree(
        CANDIDATE,
        expected_index_sha256=EXPECTED_CANDIDATE_INDEX_SHA256,
        expected_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    h8_closure = _audit_closed_tree(
        H8_PARENT,
        expected_index_sha256=(
            "3e646b49fb6cf7732b72577b6496ab4aa208b038f4cae73dc2185f9b6ac0dade"
        ),
        expected_manifest_sha256=(
            "0012e317da92d2cba0c77dd842f525b36906b1747c120a83b83c4aa9e5e08212"
        ),
    )
    chain_closure = _audit_closed_tree(
        CHAIN_PARENT,
        expected_index_sha256=(
            "b50f50c9a420726d50c30acf3b04023cc2b6ec580125f8f989fef09729de2b70"
        ),
        expected_manifest_sha256=(
            "a8f688bbef83abb833a8ad4414b7fb4c274cd38e854a1277ba36a3eef07c9759"
        ),
    )
    reference_files = _tree_files(REFERENCE_PARENT)
    _require_exact(reference_files, set(EXPECTED_REFERENCE_FILES), "reference closure")
    for relative, expected in EXPECTED_REFERENCE_FILES.items():
        _require_exact(
            _sha256_file(REFERENCE_PARENT / relative), expected, f"reference {relative}"
        )

    candidate = _read_json(CANDIDATE / "authority_candidate.json")
    manifest = _read_json(CANDIDATE / "manifest.json")
    query_file = _read_json(CANDIDATE / "query.json")
    _assert_non_authorizing(candidate)
    _require_exact(candidate.get("candidate_id"), EXPECTED_CANDIDATE_ID, "candidate ID")
    _require_exact(_candidate_id(candidate), EXPECTED_CANDIDATE_ID, "recomputed candidate ID")
    _require_exact(manifest.get("candidate_id"), EXPECTED_CANDIDATE_ID, "manifest candidate ID")
    if "authority_lock.json" in _read_output_index(CANDIDATE):
        raise AuditError("Candidate unexpectedly contains an authority lock")
    query = candidate.get("query")
    if not isinstance(query, Mapping):
        raise AuditError("Candidate query is absent")
    identities = _query_identities(query, candidate)
    _require_exact(
        identities["query_context_sha256"],
        EXPECTED_QUERY_CONTEXT_SHA256,
        "query context",
    )
    _require_exact(identities["query_key"], EXPECTED_QUERY_KEY, "query key")
    _require_exact(query_file.get("query"), dict(query), "query snapshot")

    course_axis = math.hypot(
        float(query["forecast_pin_x"]) - float(query["forecast_tee_x"]),
        float(query["forecast_pin_y"]) - float(query["forecast_tee_y"]),
    ) / 3.0
    _require_exact(course_axis, float(query["forecast_pin_long_yd"]), "course axis")
    bucket = max(1, int(np.rint(float(query["start_distance_yd"]) / 15.0)))
    support_key = (
        f"app|{query['arm']}|R{query['round_no']}|H{query['hole_no']}|"
        f"{query['start_lie']}|B{bucket}|{query['target_player_id']}"
    )
    _require_exact(support_key, EXPECTED_SUPPORT_KEY, "support key")

    proposal_inventory, proposal_inventory_facts = _read_inventory("proposal", candidate)
    surface_inventory, surface_inventory_facts = _read_inventory("surface", candidate)
    hazard_inventory, hazard_inventory_facts = _read_inventory("hazard", candidate)
    _require_exact(
        proposal_inventory_facts["sha256"],
        EXPECTED_PROPOSAL_INVENTORY_SHA256,
        "proposal inventory identity",
    )
    _require_exact(
        surface_inventory_facts["sha256"],
        EXPECTED_SURFACE_INVENTORY_SHA256,
        "surface inventory identity",
    )
    _require_exact(
        hazard_inventory_facts["sha256"],
        EXPECTED_HAZARD_INVENTORY_SHA256,
        "hazard inventory identity",
    )
    inventory_record = {
        "query_key": EXPECTED_QUERY_KEY,
        "proposal_source_inventory_records": len(proposal_inventory),
        "proposal_source_inventory_sha256": proposal_inventory_facts["sha256"],
        "surface_source_inventory_records": len(surface_inventory),
        "surface_source_inventory_sha256": surface_inventory_facts["sha256"],
        "hazard_source_inventory_records": len(hazard_inventory),
        "hazard_source_inventory_sha256": hazard_inventory_facts["sha256"],
    }
    global_inventory_sha = _stable_sha(
        {
            "contract": "externally_anchored_tee_app_source_inventory_v1",
            "queries": [
                {
                    field: _canonical_scalar(value)
                    for field, value in inventory_record.items()
                }
            ],
        }
    )
    _require_exact(
        global_inventory_sha,
        EXPECTED_SOURCE_INVENTORY_SHA256,
        "global source inventory",
    )
    _require_exact(candidate.get("source_inventory"), [inventory_record], "candidate inventory record")

    raw_proposal = pd.read_parquet(
        CANDIDATE / "parents" / "h8_proposal_support_atoms.parquet"
    )
    policy = pd.read_parquet(CANDIDATE / "proposal_policy_v2.parquet")
    _require_exact(len(raw_proposal), EXPECTED_COUNTS["proposal_atoms"], "raw proposal rows")
    _require_exact(
        _frame_digest(raw_proposal, PROPOSAL_MEASURE_COLUMNS),
        EXPECTED_PROPOSAL_MEASURE_SHA256,
        "proposal measure",
    )
    _require_exact(
        _frame_digest(raw_proposal, PROPOSAL_ABILITY_COLUMNS),
        EXPECTED_PROPOSAL_ABILITY_SHA256,
        "proposal ability identity",
    )
    expected_policy = _independent_policy_expected(raw_proposal)
    try:
        pd.testing.assert_frame_equal(
            policy,
            expected_policy,
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as exc:
        raise AuditError("Independent policy replay changed") from exc
    _proposal_inventory_source_bridge(proposal_inventory, policy)
    observed_policy = int(policy["donor_ability_available"].astype(bool).sum())
    imputed_policy = int((~policy["donor_ability_available"].astype(bool)).sum())
    clipped_policy = int(policy["donor_ability_was_clipped"].astype(bool).sum())
    _require_exact(observed_policy, EXPECTED_COUNTS["observed_proposals"], "observed policy rows")
    _require_exact(imputed_policy, EXPECTED_COUNTS["imputed_proposals"], "imputed policy rows")
    _require_exact(clipped_policy, 1, "clipped policy rows")

    with np.load(
        CANDIDATE / "parents" / "numeric_support_parent_raw.npz",
        allow_pickle=False,
    ) as archive:
        parent_numeric = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(CANDIDATE / "numeric_support_physical.npz", allow_pickle=False) as archive:
        numeric = {name: np.asarray(archive[name]) for name in archive.files}
    for name in (
        "points_yd", "probability", "cdf", "holeout_probability",
        "source_cache_sha256",
    ):
        if not np.array_equal(numeric[name], parent_numeric[name]):
            raise AuditError(f"Physical numeric snapshot changed parent array: {name}")
    if not np.array_equal(parent_numeric["atom_ids"], np.arange(903)):
        raise AuditError("Parent numeric local IDs changed")
    spatial_policy = policy.loc[~policy["terminal"].astype(bool)].sort_values(
        "proposal_row_order", kind="mergesort"
    )
    if numeric["atom_ids"].dtype != np.dtype("<U64") or not np.array_equal(
        numeric["atom_ids"].astype(str), spatial_policy["atom_id"].astype(str).to_numpy()
    ):
        raise AuditError("Physical numeric atom identities changed")
    numeric_payload = _support_payload(
        str(numeric["support_key"].astype(str)[0]),
        numeric["atom_ids"],
        numeric["points_yd"],
        numeric["probability"],
        numeric["cdf"],
        float(numeric["holeout_probability"][0]),
    )
    _require_exact(numeric_payload, EXPECTED_NUMERIC_PAYLOAD_SHA256, "numeric payload")
    expected_numeric_probability = spatial_policy["proposal_probability"].to_numpy(float)
    expected_numeric_probability /= expected_numeric_probability.sum()
    if not np.allclose(
        numeric["probability"], expected_numeric_probability, atol=2e-15, rtol=2e-15
    ):
        raise AuditError("Numeric/proposal probability bridge changed")
    if not np.array_equal(
        proposal_inventory.loc[
            ~proposal_inventory["terminal"].astype(bool),
            "frozen_spatial_cdf_upper",
        ].to_numpy(float),
        numeric["cdf"],
    ):
        raise AuditError("Frozen spatial CDF bridge changed")

    surface_snapshot = pd.read_parquet(CANDIDATE / "surface_source_h8.parquet")
    endpoints = np.column_stack(
        [
            float(query["forecast_pin_long_yd"]) + numeric["points_yd"][:, 0],
            numeric["points_yd"][:, 1],
        ]
    )
    if not np.array_equal(
        endpoints,
        surface_inventory[
            ["proposal_endpoint_long_yd", "proposal_endpoint_lat_yd"]
        ].to_numpy(float),
    ):
        raise AuditError("Numeric/surface endpoint bridge changed")
    surface_sha, surface_probability, surface_counts = _surface_reconstruction(
        surface_snapshot, endpoints
    )
    _require_exact(surface_sha, EXPECTED_SURFACE_MODEL_SHA256, "surface model")
    stored_probability = np.asarray(
        [json.loads(value) for value in surface_inventory["surface_full_probability_json"]],
        dtype=float,
    )
    stored_cdf = np.asarray(
        [json.loads(value) for value in surface_inventory["surface_full_cdf_upper_json"]],
        dtype=float,
    )
    surface_probability_residual = float(
        np.max(np.abs(surface_probability - stored_probability))
    )
    surface_cdf_residual = float(
        np.max(np.abs(np.cumsum(stored_probability, axis=1) - stored_cdf))
    )
    if surface_probability_residual != 0.0 or surface_cdf_residual != 0.0:
        raise AuditError("Surface probability replay changed")

    provider_snapshot = pd.read_parquet(CANDIDATE / "hazard_provider.parquet")
    if not np.array_equal(
        provider_snapshot["hazard_provider_row_order"].to_numpy(np.int64),
        np.arange(EXPECTED_COUNTS["provider_total"], dtype=np.int64),
    ):
        raise AuditError("Hazard provider order changed")
    provider = provider_snapshot.drop(columns="hazard_provider_row_order")
    categories = {
        str(key): int(value)
        for key, value in provider["category"].value_counts().sort_index().items()
    }
    _require_exact(
        categories,
        {"app": EXPECTED_COUNTS["provider_app"], "ott": EXPECTED_COUNTS["provider_ott"]},
        "hazard provider categories",
    )
    if provider["donor_key"].astype(str).duplicated().any():
        raise AuditError("Hazard provider donor keys are duplicated")
    params = _read_json(CANDIDATE / "hazard_resolver_params.json")
    resolver_sha, donor_panel_sha = _hazard_resolver_sha(provider, params)
    _require_exact(resolver_sha, EXPECTED_HAZARD_RESOLVER_SHA256, "hazard resolver")
    provider_logical_sha = _frame_logical_sha256(provider)
    prepared_logical = candidate.get("prepared_logical_sha256")
    if not isinstance(prepared_logical, Mapping):
        raise AuditError("Candidate prepared logical map is absent")
    _require_exact(
        provider_logical_sha,
        prepared_logical.get("hazard_donors"),
        "hazard provider logical identity",
    )
    hazard_source_bridge = _audit_hazard_source_bridge(
        provider, hazard_inventory, surface_inventory, query
    )
    hazard_replay = _hazard_replay(provider, surface_inventory, hazard_inventory, query)

    reference_manifest = _read_json(CANDIDATE / "reference_model" / "manifest.json")
    _require_exact(
        reference_manifest.get("reference_model_sha256"),
        EXPECTED_REFERENCE_MODEL_SHA256,
        "reference model identity",
    )
    putting = reference_manifest.get("putting_curve")
    if not isinstance(putting, Mapping):
        raise AuditError("Reference putting identity is absent")
    _require_exact(putting.get("sha256"), EXPECTED_PUTTING_SHA256, "putting identity")
    reference_knots = pd.read_parquet(
        CANDIDATE / "reference_model" / "reference_value_knots.parquet"
    )
    _require_exact(len(reference_knots), 189, "reference knot rows")

    expansion = _derive_expansion_facts(policy, surface_inventory, hazard_inventory)
    expected_expansion = {
        "proposal_atoms": 904,
        "surface_records": 903,
        "hazard_records": 30_702,
        "complete_atoms": 35_218,
        "terminal_atoms": 1,
        "imputed_complete_atoms": 1_989,
        "quality_mass": 1.000000000000006,
        "terminal_mass": 0.002,
        "max_proposal_mass_residual": 2.168404344971009e-18,
        "nonhazard_nonterminal_atoms": 4_515,
    }
    _require_exact(expansion, expected_expansion, "derived expansion audit")
    candidate_audit = candidate.get("expansion_audit")
    if not isinstance(candidate_audit, Mapping):
        raise AuditError("Candidate expansion audit is absent")
    candidate_expansion_projection = {
        "proposal_atoms": int(candidate_audit["proposal_atoms"]),
        "surface_records": int(candidate_audit["surface_source_inventory_records"]),
        "hazard_records": int(candidate_audit["hazard_atoms"]),
        "complete_atoms": int(candidate_audit["complete_atoms"]),
        "terminal_atoms": int(candidate_audit["terminal_atoms"]),
        "imputed_complete_atoms": int(candidate_audit["imputed_ability_atoms"]),
        "quality_mass": float(candidate_audit["quality_mass"]),
        "terminal_mass": float(candidate_audit["terminal_mass"]),
        "max_proposal_mass_residual": float(
            candidate_audit["max_proposal_mass_residual"]
        ),
        "nonhazard_nonterminal_atoms": 4_515,
    }
    _require_exact(candidate_expansion_projection, expansion, "candidate expansion projection")

    h8_manifest = _read_json(CANDIDATE / "parents" / "h8_manifest.json")
    h8_inputs = h8_manifest.get("inputs")
    if not isinstance(h8_inputs, Mapping):
        raise AuditError("Copied H8 manifest lacks inputs")
    _require_exact(
        candidate.get("prepared_input_sha256_start"),
        candidate.get("prepared_input_sha256_end"),
        "candidate prepared start/end",
    )
    _require_exact(
        candidate.get("prepared_input_sha256_start"), prepared_start,
        "candidate/current prepared inputs",
    )
    _require_exact(
        h8_inputs.get("prepared_input_sha256"), prepared_start,
        "H8/current prepared inputs",
    )
    _require_exact(
        candidate.get("prepared_logical_sha256"),
        h8_inputs.get("prepared_logical_sha256"),
        "candidate/H8 prepared logical inputs",
    )

    seam_text = SOURCE_PATHS["complete_seam"].read_text(encoding="utf-8")
    current_seam_contains_candidate = (
        EXPECTED_SOURCE_INVENTORY_SHA256 in seam_text or EXPECTED_QUERY_KEY in seam_text
    )
    if current_seam_contains_candidate:
        raise AuditError("Candidate was already minted into the reviewed seam")
    mutation_checks = _mutation_checks(
        candidate,
        query,
        proposal_inventory,
        hazard_inventory,
        surface_snapshot,
        source_start,
    )

    prepared_end = _hash_paths(prepared_paths)
    artifact_end = _artifact_input_state()
    source_end = _source_state()
    _require_maps_equal(source_start, source_end, "auditor source closure")
    _require_maps_equal(prepared_start, prepared_end, "prepared input closure")
    _require_maps_equal(artifact_start, artifact_end, "artifact input closure")

    return {
        "contract": CONTRACT,
        "authority_id": None,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": STATUS,
        "review_lock_sha256": None,
        "audit_digest_is_not_authority": True,
        "scope": "lowry_r1h8_raw_app_exact_query_candidate_review_only",
        "candidate_path": CANDIDATE.relative_to(REPO).as_posix(),
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_output_index_sha256": EXPECTED_CANDIDATE_INDEX_SHA256,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "reviewed_builder_sha256": EXPECTED_BUILDER_SHA256,
        "reviewed_pre_admission_seam_sha256": EXPECTED_SEAM_SHA256,
        "query": dict(query),
        "query_context_sha256": identities["query_context_sha256"],
        "query_key": identities["query_key"],
        "support_key": support_key,
        "course_axis_yd": course_axis,
        "course_axis_residual_yd": course_axis - float(query["forecast_pin_long_yd"]),
        "source_inventory_sha256": global_inventory_sha,
        "source_inventory_record": inventory_record,
        "inventory_facts": {
            "proposal": proposal_inventory_facts,
            "surface": surface_inventory_facts,
            "hazard": hazard_inventory_facts,
        },
        "proposal_facts": {
            "measure_sha256": EXPECTED_PROPOSAL_MEASURE_SHA256,
            "ability_sha256": EXPECTED_PROPOSAL_ABILITY_SHA256,
            "rows": len(policy),
            "observed_policy_rows": observed_policy,
            "imputed_policy_rows": imputed_policy,
            "clipped_policy_rows": clipped_policy,
            "policy_sha256": EXPECTED_ABILITY_POLICY_SHA256,
            "policy_replay_exact": True,
            "inventory_source_bridge_exact": True,
        },
        "numeric_facts": {
            "records": len(numeric["atom_ids"]),
            "atom_id_dtype": str(numeric["atom_ids"].dtype),
            "payload_sha256": numeric_payload,
            "source_cache_sha256": str(numeric["source_cache_sha256"].astype(str)[0]),
            "holeout_probability": float(numeric["holeout_probability"][0]),
            "parent_arrays_exact": True,
            "proposal_probability_bridge_exact": True,
            "frozen_cdf_bridge_exact": True,
        },
        "surface_facts": {
            "snapshot_rows": len(surface_snapshot),
            "regular_rows": int(sum(surface_counts.values())),
            "class_counts": surface_counts,
            "model_sha256": surface_sha,
            "probability_max_abs_residual": surface_probability_residual,
            "cdf_max_abs_residual": surface_cdf_residual,
            "snapshot_reloadable": True,
        },
        "hazard_facts": {
            "provider_total": len(provider),
            "provider_by_category": categories,
            "absolute_provider_ceiling": 8_192,
            "provider_ceiling_headroom": 8_192 - len(provider),
            "provider_logical_sha256": provider_logical_sha,
            "donor_panel_sha256": donor_panel_sha,
            "resolver_sha256": resolver_sha,
            "resolver_params": params,
            "source_bridge": hazard_source_bridge,
            "independent_distribution_replay": hazard_replay,
            "snapshot_reloadable": True,
        },
        "reference_facts": {
            "files": dict(EXPECTED_REFERENCE_FILES),
            "model_sha256": EXPECTED_REFERENCE_MODEL_SHA256,
            "putting_curve_sha256": EXPECTED_PUTTING_SHA256,
            "knots": len(reference_knots),
            "snapshot_reloadable": True,
        },
        "derived_expansion_audit_without_expansion": expansion,
        "candidate_closure": candidate_closure,
        "parent_closures": {
            "h8_proposal": h8_closure,
            "normalized_chain_50k": chain_closure,
            "fixed_reference": {
                "root": REFERENCE_PARENT.relative_to(REPO).as_posix(),
                "files": dict(EXPECTED_REFERENCE_FILES),
                "closed_exact": True,
            },
        },
        "logical_seals": {
            "candidate_prepared_logical_sha256": candidate[
                "prepared_logical_sha256"
            ],
            "h8_parent_logical_map_exact": True,
            "provider_logical_replay_exact": True,
        },
        "source_sha256_start": source_start,
        "source_sha256_end": source_end,
        "prepared_input_sha256_start": prepared_start,
        "prepared_input_sha256_end": prepared_end,
        "artifact_input_sha256_start": artifact_start,
        "artifact_input_sha256_end": artifact_end,
        "mutation_checks": mutation_checks,
        "current_seam_contains_candidate_authority": False,
        "candidate_non_authorizing_exact": True,
        "candidate_generation_mutated_sources_or_parents": False,
        "prepare_event_run": False,
        "complete_atom_expansion_run": False,
        "simulation_run": False,
        "seam_edited": False,
        "authority_minted": False,
        "verdict": "GO_DATA_ONLY_AUTHORITY_RECORD_REVIEW_EVIDENCE",
        "verdict_scope": (
            "manual source-pinned Lowry R1 H8 data record only; no expansion, "
            "simulation, anchor, or production authorization"
        ),
    }


def _audit_id(body: Mapping[str, object]) -> str:
    return _stable_sha(
        {
            "contract": CONTRACT,
            "role": "detached_non_authorizing_independent_review",
            "payload": body,
        }
    )


def _report(audit: Mapping[str, object]) -> str:
    expansion = audit["derived_expansion_audit_without_expansion"]
    hazard = audit["hazard_facts"]
    assert isinstance(expansion, Mapping)
    assert isinstance(hazard, Mapping)
    return (
        "# Independent Lowry R1 H8 candidate audit\n\n"
        "Status: **GO for one manual data-only authority record; this artifact "
        "is not authority**.\n\n"
        f"- Audit ID: `{audit['audit_id']}`\n"
        f"- Candidate ID: `{audit['candidate_id']}`\n"
        f"- Candidate output index: `{audit['candidate_output_index_sha256']}`\n"
        f"- Reviewed builder: `{audit['reviewed_builder_sha256']}`\n"
        f"- Reviewed pre-admission seam: `{audit['reviewed_pre_admission_seam_sha256']}`\n"
        f"- Query key: `{audit['query_key']}`\n"
        f"- Source inventory: `{audit['source_inventory_sha256']}`\n"
        f"- Recomputed complete atoms: {expansion['complete_atoms']}\n"
        f"- Recomputed terminal atoms: {expansion['terminal_atoms']}\n"
        f"- Recomputed imputed complete atoms: {expansion['imputed_complete_atoms']}\n"
        f"- Hazard provider: {hazard['provider_total']} total "
        f"({hazard['provider_by_category']})\n"
        "- Preparation, expansion, simulation, seam edits, and authority minting "
        "were not performed.\n"
        "- A separate source-coded lock must pin this evidence before any bounded "
        "runner can consume the query.\n"
    )


def build_audit_artifact() -> Path:
    if OUTPUT.exists():
        raise FileExistsError(f"Fixed audit output already exists: {OUTPUT}")
    body = collect_independent_audit()
    audit_id = _audit_id(body)
    audit = {"audit_id": audit_id, **body}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    _write_json(OUTPUT / "audit.json", audit)
    _write_json(
        OUTPUT / "source_input_seals.json",
        {
            "contract": "independent_candidate_audit_source_input_seals_v1",
            "audit_id": audit_id,
            "source_sha256_start": body["source_sha256_start"],
            "source_sha256_end": body["source_sha256_end"],
            "prepared_input_sha256_start": body["prepared_input_sha256_start"],
            "prepared_input_sha256_end": body["prepared_input_sha256_end"],
            "artifact_input_sha256_start": body["artifact_input_sha256_start"],
            "artifact_input_sha256_end": body["artifact_input_sha256_end"],
        },
    )
    (OUTPUT / "REPORT.md").write_text(_report(audit), encoding="utf-8")
    payload_files = {
        path.name: {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in sorted(OUTPUT.iterdir())
        if path.is_file()
    }
    manifest = {
        "contract": MANIFEST_CONTRACT,
        "audit_id": audit_id,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": STATUS,
        "expected_output_files_before_index": sorted(
            [*payload_files, "manifest.json"]
        ),
        "payload_files": payload_files,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_output_index_sha256": EXPECTED_CANDIDATE_INDEX_SHA256,
        "source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
        "query_key": EXPECTED_QUERY_KEY,
        "note": (
            "The recursive output index seals independent review evidence; "
            "neither the audit ID nor output index is an authority lock."
        ),
    }
    _write_json(OUTPUT / "manifest.json", manifest)
    _write_output_index(OUTPUT)
    validate_audit_artifact(deep=True)
    return OUTPUT


def validate_audit_artifact(*, deep: bool = True) -> dict[str, object]:
    closure = _audit_closed_tree(OUTPUT)
    audit = _read_json(OUTPUT / "audit.json")
    manifest = _read_json(OUTPUT / "manifest.json")
    if audit.get("authority_id") is not None or audit.get("authority_ready") is not False:
        raise AuditError("Detached audit became authorizing")
    if audit.get("production_enabled") is not False or audit.get("review_lock_sha256") is not None:
        raise AuditError("Detached audit became production-capable")
    if audit.get("admission_status") != STATUS or audit.get("audit_digest_is_not_authority") is not True:
        raise AuditError("Detached audit status changed")
    body = dict(audit)
    audit_id = str(body.pop("audit_id", ""))
    _require_exact(_audit_id(body), audit_id, "audit identity")
    _require_exact(manifest.get("audit_id"), audit_id, "manifest audit identity")
    _require_exact(manifest.get("authority_ready"), False, "manifest authority readiness")
    _require_exact(manifest.get("production_enabled"), False, "manifest production readiness")
    entries = _read_output_index(OUTPUT)
    expected_files = manifest.get("expected_output_files_before_index")
    if not isinstance(expected_files, list) or set(map(str, expected_files)) != set(entries):
        raise AuditError("Audit manifest does not enumerate the exact closed tree")
    payload = manifest.get("payload_files")
    if not isinstance(payload, Mapping):
        raise AuditError("Audit manifest payload map is absent")
    expected_payload = {
        name: metadata for name, metadata in entries.items() if name != "manifest.json"
    }
    _require_exact(dict(payload), expected_payload, "audit payload-file map")
    seals = _read_json(OUTPUT / "source_input_seals.json")
    for field in (
        "source_sha256_start", "source_sha256_end",
        "prepared_input_sha256_start", "prepared_input_sha256_end",
        "artifact_input_sha256_start", "artifact_input_sha256_end",
    ):
        _require_exact(seals.get(field), audit.get(field), f"seal file {field}")
    if deep:
        observed_body = collect_independent_audit()
        _require_exact(observed_body, body, "deep independent audit replay")
    return {"audit": audit, "closure": closure}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate the fixed non-authorizing independent audit of "
            "the Lowry R1 H8 source-authority candidate."
        )
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the one fixed audit output instead of building it",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        result = validate_audit_artifact(deep=True)
        audit = result["audit"]
        print(f"Validated non-authorizing audit: {OUTPUT}")
        print(f"Audit ID: {audit['audit_id']}")
        return
    output = build_audit_artifact()
    audit = _read_json(output / "audit.json")
    print(f"Wrote non-authorizing independent audit: {output}")
    print(f"Audit ID: {audit['audit_id']}")
    print("GO is limited to a manual data-only authority record.")


if __name__ == "__main__":
    main()

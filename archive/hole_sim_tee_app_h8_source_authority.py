"""Build a review-only source-inventory candidate for Lowry R1 H8.

This executable is deliberately one-way and non-authorizing.  It has no
source, query, cap, digest, or admission overrides.  Its only build argument
is a fresh output directory.  It snapshots the exact inputs needed to replay
the complete TEE/APP expansion, independently recomputes the ordered source
inventories exposed by the frozen seam, and always emits
``authority_ready=false``.

Nothing here edits or admits the complete-atom seam.  An independent review
must hard-pin a candidate in a separate lock before a separate bounded runner
can consume it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_fixed_reference_value as fixed_reference  # noqa: E402
import hole_sim_hazard_resolution as hazard  # noqa: E402
import hole_sim_normalized_atom_precompute as precompute  # noqa: E402
import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_tee_app_complete_atom_support as complete  # noqa: E402
import hole_sim_tee_app_integrated_v2 as spatial  # noqa: E402


CONTRACT = "lowry_r1h8_tee_app_source_authority_candidate_v1"
MANIFEST_CONTRACT = "lowry_r1h8_tee_app_source_authority_manifest_v1"
OUTPUT_INDEX = "OUTPUT_SHA256.json"
ADMISSION_STATUS = "candidate_external_review_required"
INVENTORY_SHARD_ROWS = 8_192
MAX_BATCH_ROWS = 8_192
MAX_HAZARD_DONORS_PER_SURFACE = 8_192
MAX_COMPLETE_ATOMS_PER_PROPOSAL = 16_384
ABSOLUTE_HAZARD_PROVIDER_CEILING = 8_192

EXPECTED_PROVIDER_TOTAL = 1_046
EXPECTED_PROVIDER_BY_CATEGORY = {"app": 364, "ott": 682}
EXPECTED_PROPOSAL_RECORDS = 904
EXPECTED_SURFACE_RECORDS = 903
EXPECTED_HAZARD_RECORDS = 30_702
EXPECTED_COMPLETE_ATOMS = 35_218
EXPECTED_TERMINAL_ATOMS = 1
EXPECTED_IMPUTED_COMPLETE_ATOMS = 1_989
EXPECTED_MISSING_PROPOSAL_ABILITIES = 51

EXPECTED_SEAM_SHA256 = (
    "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
)
EXPECTED_COMPLETE_DEPENDENCY_SHA256 = {
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
}
EXPECTED_STREAMING_DEPENDENCY_SHA256 = {
    "precompute_source": "3bd580b135f343cf2dbb0f9347b8678dd9c6eace66218826852f411e02d6e773",
    "full_chain_source": "10d8971b45843bc74d4dbd7b5c53d94cdb3f35e0e8621bcba357acce77a20c89",
    "batched_population_source": "0da3a934b671c3ed74a24c8c754a8404e3a329eb4f54f3b429152e45cd7fca23",
    "hazard_source": "8a3d8e019c39156299ebde6e7e8a344ba2edf5b55dc8602981ab75e4466163e5",
    "spatial_v2_source": "dc85b38b5601ba5f6afbe2ce4d900e109243f60cb6a9dd7a03426b436443ea97",
    "tee_source": "006f1de02d98be47dfd621a5e46d14672466ba6ee673cf7e0f56bc94daf56b43",
    "app_source": "6acceff91e9130e3584d02d4d113daecc5e84831dfe3fff6f7190d40f5450e1f",
    "asof_source": "9463aa4b0c2fd67adec11ccd109126d0bbf144088dca8095ad5965216482df3c",
    "normalizer_source": "57beb668622f2635dd47bc60b05de323ee05cae63247f0afdafb71e6e4b75e17",
    "hook_source": "311523564b178227ade614ffa598ed9787142b1a23136da9ea71f5c8b37e03a7",
    "markov_source": "6740b3c2ce6119084bbb5371a3ea84ab159f5eeeb1db746188fe25d5ab67a4cf",
    "state_foundations_source": "3f499f4a4746f896f3cf8ae39465cc6be4ab5cd5d7688151e99886bd4a381eaf",
    "tail_policy_source": "1f01e4719febe527cf064c3e5e0ff7281958c1d8adcd7027352d83dd9d95f4de",
}
EXPECTED_H8_RUNTIME_DEPENDENCY_SHA256 = {
    **EXPECTED_STREAMING_DEPENDENCY_SHA256,
    "chain_adapter_source": "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    "fixed_reference_source": "39df1d57a6623b49f90f3ae82cb9867cb77a3f7cff740754794cdef324ba8b8b",
    "frozen_v6_ledger_source": "11b14109b938cda5da4b4eaae4d00eb43f9d44074fdbd9e48328a4f889d7ab97",
    "production_target_source": "d9823f03c613b1985a73aa9d32fa0d68153f670924c16d297870ac19ebe5dc89",
    "streaming_source": "0d8c2be5027c5a5325d78eea6205691091455104624f6a9acb3006de5993f807",
    "traced_population_source": "3967ab682b28b3848481450ffc405c18b111e4ab9806384d51a4f8f168114a42",
}

PROPOSAL_PARENT = (
    REPO
    / "permanent_data"
    / "hole_sim_streaming_transition_ledger_lowry_r1h8_raw_app_holeout_5_v1"
)
NUMERIC_PARENT = (
    REPO / "permanent_data" / "hole_sim_normalized_chain_50k_travelers_tail_v1"
)
REFERENCE_PARENT = (
    REPO / "permanent_data" / "hole_sim_fixed_reference_travelers_v2"
)
PROPOSAL_SOURCE_FILE = PROPOSAL_PARENT / "proposal_support_atoms.parquet"
NUMERIC_SOURCE_RELATIVE = (
    "support_cache/raw/"
    "a62c2ca292847a871501078399d41fdeba3ea29a8170a2de5d792b68c723e37c.npz"
)
NUMERIC_SOURCE_FILE = NUMERIC_PARENT / NUMERIC_SOURCE_RELATIVE

EXPECTED_PARENT_CLOSURES = {
    "h8_proposal": {
        "path": "permanent_data/hole_sim_streaming_transition_ledger_lowry_r1h8_raw_app_holeout_5_v1",
        "output_index_sha256": "3e646b49fb6cf7732b72577b6496ab4aa208b038f4cae73dc2185f9b6ac0dade",
        "manifest_sha256": "0012e317da92d2cba0c77dd842f525b36906b1747c120a83b83c4aa9e5e08212",
    },
    "normalized_chain_50k": {
        "path": "permanent_data/hole_sim_normalized_chain_50k_travelers_tail_v1",
        "output_index_sha256": "b50f50c9a420726d50c30acf3b04023cc2b6ec580125f8f989fef09729de2b70",
        "manifest_sha256": "a8f688bbef83abb833a8ad4414b7fb4c274cd38e854a1277ba36a3eef07c9759",
    },
}
EXPECTED_PROPOSAL_PARQUET_SHA256 = (
    "4b411b8b2da9e880c4a8434cdfa8a73b0b40439a68604a4b92894a0367333758"
)
EXPECTED_PROPOSAL_INVENTORY_SHA256 = (
    "8f98678c91c8f1cbb3395263498f5287653ada69ee0bb1e0da185123f2fec755"
)
EXPECTED_NUMERIC_NPZ_SHA256 = (
    "249241d2c08b6b587eed7559873734fedb5fe8f2d3d2833096551366b3c803b5"
)
EXPECTED_PARENT_NUMERIC_LOCAL_PAYLOAD_SHA256 = (
    "b993873d1b0f8c8fb0fc9c21be5c33a10c75c3f2c60f683bd2e68a1ccd58a6cd"
)
EXPECTED_SOURCE_CACHE_SHA256 = (
    "5adde84d8fc7b5ac89c5a20a08e321c393d32acb452b3dac5390afc1e44b70d2"
)
EXPECTED_NUMERIC_PAYLOAD_SHA256 = (
    "44bde6f9a6943c5bb256b549e8de6bed9fa5c23ab9e91e78efb4f8ae2372a0d7"
)
EXPECTED_PROPOSAL_MEASURE_SHA256 = (
    "8a646ebc165f86eb9859ea1208fd11bfd1b8efba45128b8a63ea1c7c9a1495f6"
)
EXPECTED_ABILITY_POLICY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)
EXPECTED_REFERENCE_MODEL_SHA256 = (
    "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
)
EXPECTED_PUTTING_SHA256 = (
    "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
)
EXPECTED_REFERENCE_FILES = {
    "REPORT.md": "00b5b9e4033f97db8a3b6fe08291a6850a3b58ceb9539874d90922fc29ef85d1",
    "manifest.json": "ea37a25af0301e0613078a1d4d837024904e89645cfd0d97967fa1d402d35db4",
    "reference_value_knots.parquet": "55c0c3d2559d018fe50fc7eb1f6c00337bb6c57787af9d9beb6cb5365b462347",
}

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
EXPECTED_QUERY_CONTEXT_SHA256 = (
    "3737e0d82354591d3395204b35fd434299f43e6ced66d817fa1e8032ef276fbd"
)


class SourceAuthorityError(ValueError):
    """Raised when a candidate cannot be sealed without self-authorization."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_scalar(value: object) -> object:
    """Independent copy of the frozen seam's scalar identity contract."""

    if value is None or value is pd.NA:
        return {"type": "null"}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            return {"type": "null"}
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
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
            raise SourceAuthorityError("Identity hashes reject non-finite floats")
        if number == 0.0:
            number = 0.0
        return {"type": "float", "value": number.hex()}
    return {
        "type": "str",
        "value": unicodedata.normalize("NFKC", str(value)).strip(),
    }


class _OrderedInventoryHasher:
    """Independent ordered-chain implementation of the seam inventory hash."""

    def __init__(self, contract: str, columns: Sequence[str]):
        self.contract = str(contract)
        self.columns = tuple(map(str, columns))
        self.records = 0
        self.chain_sha256 = _stable_sha(
            {"contract": self.contract, "genesis": True}
        )

    def add(self, values: Mapping[str, object]) -> None:
        missing = [column for column in self.columns if column not in values]
        if missing:
            raise SourceAuthorityError(f"Inventory record misses fields={missing}")
        record = {
            column: _canonical_scalar(values[column]) for column in self.columns
        }
        self.chain_sha256 = _stable_sha(
            {
                "contract": self.contract,
                "prior_sha256": self.chain_sha256,
                "record_order": int(self.records),
                "record": record,
            }
        )
        self.records += 1

    def hexdigest(self) -> str:
        return _stable_sha(
            {
                "contract": self.contract,
                "records": int(self.records),
                "chain_sha256": self.chain_sha256,
            }
        )


# These tuples are copied literally rather than imported from the seam.  The
# candidate therefore proves that an independent projection produces the same
# inventory identity; a seam change cannot silently redefine the builder.
PROPOSAL_SOURCE_INVENTORY_COLUMNS = (
    "query_key",
    "proposal_support_key",
    "proposal_row_order",
    "proposal_atom_id",
    "category",
    "proposal_support_measure_sha256",
    "numeric_payload_sha256",
    "source_cache_sha256",
    "proposal_base_hierarchy_mass",
    "proposal_spatial_model_weight",
    "proposal_probability",
    "proposal_branch",
    "proposal_branch_conditional_probability",
    "proposal_cdf_lower",
    "proposal_cdf_upper",
    "proposal_branch_cdf_lower",
    "proposal_branch_cdf_upper",
    "frozen_spatial_cdf_lower",
    "frozen_spatial_cdf_upper",
    "frozen_terminal_residual_cdf_lower",
    "frozen_terminal_residual_cdf_upper",
    "terminal",
    "holed",
    "holeout_representation",
    "declared_app_base_holeout_probability",
    "implicit_holeout_probability",
    "proposal_app_holeout_equivalence_sha256",
    "donor_event_key",
    "donor_player_id",
    "donor_player_name",
    "donor_round_no",
    "donor_hole_no",
    "donor_stroke_no",
    "donor_event_start",
    "donor_skill_date",
    "bridge_row_id",
    "target_event_key",
    "target_event_start",
    "donor_ability_source_column",
    "donor_ability_available",
    "donor_ability_raw_strict_prior",
    "donor_ability_value_kind",
    "donor_ability_sg",
    "donor_ability_was_clipped",
    "donor_ability_policy_sha256",
    "donor_ability_row_sha256",
    "donor_ability_snapshot_kind",
    "proposal_policy_binding_sha256",
)
SURFACE_SOURCE_INVENTORY_COLUMNS = (
    "query_key",
    "proposal_row_order",
    "proposal_atom_id",
    "proposal_endpoint_long_yd",
    "proposal_endpoint_lat_yd",
    "surface_model_sha256",
    "surface_class_count",
    "surface_full_probability_json",
    "surface_full_cdf_upper_json",
    "surface_selection_boundary",
)
HAZARD_SOURCE_INVENTORY_COLUMNS = (
    "query_key",
    "proposal_row_order",
    "proposal_atom_id",
    "surface_class_index",
    "surface_class_label",
    "hazard_resolver_sha256",
    "hazard_donor_order",
    "hazard_donor_key",
    "hazard_probability",
    "hazard_cdf_lower",
    "hazard_cdf_upper",
    "hazard_donor_event_key",
    "hazard_donor_season",
    "hazard_donor_round_no",
    "hazard_donor_hole_no",
    "hazard_donor_category",
    "hazard_donor_causal_endpoint_x",
    "hazard_donor_causal_endpoint_y",
    "hazard_penalty_count",
    "hazard_penalty_stroke_cost",
    "hazard_resolution_shot_key",
    "hazard_resolution_event_type",
    "hazard_resolution_x",
    "hazard_resolution_y",
    "hazard_resolution_lie",
    "hazard_donor_physical_course_id",
    "hazard_foundation_version",
    "hazard_coordinate_reference_year",
    "hazard_coordinate_reference_event_key",
    "hazard_coordinate_reference_mode",
    "hazard_data_role",
    "hazard_support_source",
    "hazard_fallback_used",
    "hazard_support_setup",
    "hazard_support_hole",
    "hazard_support_category",
    "hazard_component_masses",
    "hazard_nearest_category_endpoint_distance_yd",
    "hazard_nearest_same_hole_endpoint_distance_yd",
    "hazard_support_radius_limit_yd",
    "hazard_causal_endpoint_x",
    "hazard_causal_endpoint_y",
    "penalty_strokes",
    "end_lie",
    "end_distance_yd",
    "end_long_yd",
    "end_lat_yd",
)
INVENTORY_SPECS = {
    "proposal": (
        complete.PROPOSAL_SOURCE_INVENTORY_CONTRACT,
        PROPOSAL_SOURCE_INVENTORY_COLUMNS,
        EXPECTED_PROPOSAL_RECORDS,
    ),
    "surface": (
        complete.SURFACE_SOURCE_INVENTORY_CONTRACT,
        SURFACE_SOURCE_INVENTORY_COLUMNS,
        EXPECTED_SURFACE_RECORDS,
    ),
    "hazard": (
        complete.HAZARD_SOURCE_INVENTORY_CONTRACT,
        HAZARD_SOURCE_INVENTORY_COLUMNS,
        EXPECTED_HAZARD_RECORDS,
    ),
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
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise SourceAuthorityError("JSON payload contains a non-finite float")
        return number
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SourceAuthorityError(f"Expected a JSON object: {path}")
    return payload


def _require_equal_maps(
    start: Mapping[str, object], end: Mapping[str, object], label: str
) -> None:
    if dict(start) != dict(end):
        keys = sorted(set(start) | set(end))
        mismatches = {
            key: {"start": start.get(key), "end": end.get(key)}
            for key in keys
            if start.get(key) != end.get(key)
        }
        raise SourceAuthorityError(f"{label} changed start-to-end: {mismatches}")


def _assert_exact_map(
    observed: Mapping[str, object], expected: Mapping[str, object], label: str
) -> None:
    if dict(observed) != dict(expected):
        keys = sorted(set(observed) | set(expected))
        mismatch = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in keys
            if observed.get(key) != expected.get(key)
        }
        raise SourceAuthorityError(f"{label} is not frozen: {mismatch}")


def _current_source_state() -> dict[str, object]:
    seam_sha = _sha256_file(Path(complete.__file__))
    if seam_sha != EXPECTED_SEAM_SHA256:
        raise SourceAuthorityError(
            f"Frozen seam changed: {seam_sha} != {EXPECTED_SEAM_SHA256}"
        )
    complete_dependencies = complete._dependency_sha256()
    _assert_exact_map(
        complete_dependencies,
        EXPECTED_COMPLETE_DEPENDENCY_SHA256,
        "complete dependency closure",
    )
    streaming_dependencies = {
        name: _sha256_file(path)
        for name, path in streaming.dependency_paths().items()
    }
    _assert_exact_map(
        streaming_dependencies,
        EXPECTED_STREAMING_DEPENDENCY_SHA256,
        "streaming dependency closure",
    )
    return {
        "builder_source": _sha256_file(Path(__file__)),
        "complete_seam": seam_sha,
        "complete_dependencies": complete_dependencies,
        "streaming_dependencies": streaming_dependencies,
    }


def _safe_index_entry(name: object) -> str:
    relative = str(name)
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or relative == OUTPUT_INDEX
        or pure.name == OUTPUT_INDEX
    ):
        raise SourceAuthorityError(f"Unsafe or reserved output path: {relative!r}")
    return relative


def _tree_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SourceAuthorityError(f"Artifact closure contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _validate_index_entries(root: Path, entries: Mapping[str, object]) -> None:
    for raw_name, raw_metadata in entries.items():
        relative = _safe_index_entry(raw_name)
        if not isinstance(raw_metadata, Mapping):
            raise SourceAuthorityError(f"Invalid checksum metadata for {relative}")
        if set(raw_metadata) != {"bytes", "sha256"}:
            raise SourceAuthorityError(f"Checksum schema changed for {relative}")
        path = root / PurePosixPath(relative)
        if not path.is_file() or path.is_symlink():
            raise SourceAuthorityError(f"Indexed artifact file is absent: {relative}")
        size = int(path.stat().st_size)
        digest = _sha256_file(path)
        if size != int(raw_metadata["bytes"]) or digest != str(
            raw_metadata["sha256"]
        ):
            raise SourceAuthorityError(f"Artifact checksum mismatch: {relative}")


def _validate_closed_output_index(root: Path) -> dict[str, object]:
    """Verify a non-circular output index and reject additions or symlinks."""

    root = Path(root)
    index_path = root / OUTPUT_INDEX
    payload = _read_json(index_path)
    if set(payload) == {"contract", "files"}:
        if payload.get("contract") != "non_circular_output_sha256_v1" or not isinstance(
            payload.get("files"), Mapping
        ):
            raise SourceAuthorityError("Wrapped output index contract changed")
        entries = dict(payload["files"])
    else:
        entries = payload
    indexed = {_safe_index_entry(name) for name in entries}
    actual = _tree_files(root)
    expected = indexed | {OUTPUT_INDEX}
    if actual != expected:
        raise SourceAuthorityError(
            "Artifact tree is not closed: "
            f"missing={sorted(expected - actual)}, added={sorted(actual - expected)}"
        )
    _validate_index_entries(root, entries)
    return entries


def _write_output_index(root: Path) -> Path:
    root = Path(root)
    index_path = root / OUTPUT_INDEX
    if index_path.exists():
        raise FileExistsError(index_path)
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SourceAuthorityError(f"Candidate contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = _safe_index_entry(path.relative_to(root).as_posix())
        entries[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    _write_json(index_path, entries)
    return index_path


def _verify_external_artifact_closure(
    root: Path, *, output_index_sha256: str, manifest_sha256: str
) -> dict[str, object]:
    root = Path(root)
    if _sha256_file(root / OUTPUT_INDEX) != output_index_sha256:
        raise SourceAuthorityError(f"Parent output index changed: {root}")
    if _sha256_file(root / "manifest.json") != manifest_sha256:
        raise SourceAuthorityError(f"Parent manifest changed: {root}")
    return _validate_closed_output_index(root)


def _verify_reference_closure(root: Path = REFERENCE_PARENT) -> None:
    actual = _tree_files(root)
    if actual != set(EXPECTED_REFERENCE_FILES):
        raise SourceAuthorityError(
            f"Fixed reference closure changed: {sorted(actual)}"
        )
    for relative, expected in EXPECTED_REFERENCE_FILES.items():
        if _sha256_file(root / relative) != expected:
            raise SourceAuthorityError(f"Fixed reference changed: {relative}")


def _query_payload(query: complete.CompleteAtomQuery) -> dict[str, object]:
    payload = asdict(query)
    payload["target_event_start"] = pd.Timestamp(
        payload["target_event_start"]
    ).tz_convert("UTC").isoformat()
    return _jsonable(payload)  # type: ignore[return-value]


def _assert_exact_query(query: complete.CompleteAtomQuery) -> None:
    query.validated()
    observed = _query_payload(query)
    if observed != EXPECTED_QUERY:
        mismatch = {
            key: {"expected": EXPECTED_QUERY.get(key), "observed": observed.get(key)}
            for key in sorted(set(observed) | set(EXPECTED_QUERY))
            if observed.get(key) != EXPECTED_QUERY.get(key)
        }
        raise SourceAuthorityError(f"Lowry R1 H8 query changed: {mismatch}")
    context_sha = _stable_sha(
        {
            "contract": complete.QUERY_CONTRACT,
            "query": complete._query_identity_payload(query),
        }
    )
    if context_sha != EXPECTED_QUERY_CONTEXT_SHA256:
        raise SourceAuthorityError("Canonical Lowry R1 H8 query identity changed")


def _select_exact_query(prepared: streaming.PreparedStreamingEvent) -> complete.CompleteAtomQuery:
    required = {
        "event_key",
        "source_player_id",
        "round_no",
        "hole_no",
        "par",
        "forecast_tee_x",
        "forecast_tee_y",
        "forecast_pin_x",
        "forecast_pin_y",
        "forecast_pin_long_yd",
    }
    missing = sorted(required.difference(prepared.grid.columns))
    if missing:
        raise SourceAuthorityError(f"Prepared grid misses query columns={missing}")
    selected = prepared.grid.loc[
        prepared.grid["event_key"].astype(str).eq(EXPECTED_QUERY["target_event_key"])
        & prepared.grid["source_player_id"].astype(str).eq(
            EXPECTED_QUERY["target_player_id"]
        )
        & prepared.grid["round_no"].eq(EXPECTED_QUERY["round_no"])
        & prepared.grid["hole_no"].eq(EXPECTED_QUERY["hole_no"])
    ]
    if len(selected) != 1:
        raise SourceAuthorityError(
            f"Lowry R1 H8 grid row is absent or duplicated: rows={len(selected)}"
        )
    row = next(selected.itertuples(index=False))
    query = complete.CompleteAtomQuery(
        arm="raw",
        round_no=int(row.round_no),
        hole_no=int(row.hole_no),
        par=int(row.par),
        target_event_key=str(row.event_key),
        target_event_start=prepared.atoms.target_event_start,
        target_player_id=str(row.source_player_id),
        start_lie="tee",
        start_distance_yd=float(row.forecast_pin_long_yd),
        start_long_yd=0.0,
        start_lat_yd=0.0,
        forecast_tee_x=float(row.forecast_tee_x),
        forecast_tee_y=float(row.forecast_tee_y),
        forecast_pin_x=float(row.forecast_pin_x),
        forecast_pin_y=float(row.forecast_pin_y),
        forecast_pin_long_yd=float(row.forecast_pin_long_yd),
        physical_course_id=str(prepared.atoms.physical_course_id),
    )
    _assert_exact_query(query)
    return query


def _normalized_provider_categories(provider: pd.DataFrame) -> dict[str, int]:
    aliases = {
        "sg_ott": "ott",
        "tee": "ott",
        "drive": "ott",
        "driving": "ott",
        "sg_app": "app",
        "approach": "app",
    }
    categories = (
        provider["category"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map(lambda value: aliases.get(value, value))
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {str(key): int(value) for key, value in categories.items()}


def _preflight_hazard_provider(
    resolver: object,
    *,
    expected_total: int = EXPECTED_PROVIDER_TOTAL,
    expected_by_category: Mapping[str, int] = EXPECTED_PROVIDER_BY_CATEGORY,
    absolute_ceiling: int = ABSOLUTE_HAZARD_PROVIDER_CEILING,
) -> dict[str, object]:
    """Check provider shape before any builder-owned copy, hash, or expansion."""

    if int(absolute_ceiling) != ABSOLUTE_HAZARD_PROVIDER_CEILING:
        raise SourceAuthorityError("The authority builder ceiling is not fixed at 8192")
    if complete.ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS != int(absolute_ceiling):
        raise SourceAuthorityError("The frozen seam's absolute provider ceiling changed")
    provider = getattr(resolver, "donors", None)
    if not isinstance(provider, pd.DataFrame):
        raise SourceAuthorityError("Hazard resolver lacks a DataFrame provider")
    total = int(len(provider))
    # This branch intentionally precedes category distribution, hashing, and
    # copying.  A provider over the absolute ceiling is rejected immediately.
    if total > int(absolute_ceiling):
        raise SourceAuthorityError(
            f"Hazard provider exceeds absolute ceiling: {total} > {absolute_ceiling}"
        )
    if total != int(expected_total):
        raise SourceAuthorityError(
            f"Hazard provider count changed: {total} != {expected_total}"
        )
    if "category" not in provider.columns:
        raise SourceAuthorityError("Hazard provider has no category column")
    by_category = _normalized_provider_categories(provider)
    expected_categories = {
        str(key): int(value) for key, value in expected_by_category.items()
    }
    if by_category != expected_categories:
        raise SourceAuthorityError(
            f"Hazard provider categories changed: {by_category} != {expected_categories}"
        )
    return {
        "provider_donors": total,
        "by_category": by_category,
        "absolute_ceiling": int(absolute_ceiling),
        "ceiling_headroom": int(absolute_ceiling) - total,
    }


def _hazard_resolver_params(resolver: object) -> dict[str, object]:
    return {
        "setup_prior": float(getattr(resolver, "setup_prior")),
        "hole_prior": float(getattr(resolver, "hole_prior")),
        "spatial_bandwidth_yd": float(
            getattr(resolver, "spatial_bandwidth_yd")
        ),
        "max_nearest_distance_yd": float(
            getattr(resolver, "max_nearest_distance_yd")
        ),
        "same_hole_fallback_max_distance_yd": getattr(
            resolver, "same_hole_fallback_max_distance_yd"
        ),
        "minimum_category_support": int(
            getattr(resolver, "minimum_category_support")
        ),
        "physical_course_id": str(getattr(resolver, "physical_course_id")),
        "coordinate_reference_year": int(
            getattr(resolver, "coordinate_reference_year")
        ),
        "coordinate_reference_event_key": str(
            getattr(resolver, "coordinate_reference_event_key")
        ),
        "coordinate_reference_mode": "latest_prior",
        "foundation_version": hazard.FOUNDATION_VERSION,
        "resolver_version": hazard.RESOLVER_VERSION,
    }


def _capture_hazard_snapshot(
    resolver: object,
    *,
    expected_total: int = EXPECTED_PROVIDER_TOTAL,
    expected_by_category: Mapping[str, int] = EXPECTED_PROVIDER_BY_CATEGORY,
    absolute_ceiling: int = ABSOLUTE_HAZARD_PROVIDER_CEILING,
    hash_fn: Callable[..., str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], str, dict[str, object]]:
    preflight = _preflight_hazard_provider(
        resolver,
        expected_total=expected_total,
        expected_by_category=expected_by_category,
        absolute_ceiling=absolute_ceiling,
    )
    resolver_hash = (hash_fn or complete.hazard_resolver_sha256)(
        resolver, max_donor_count=int(absolute_ceiling)
    )
    provider = getattr(resolver, "donors")
    snapshot = provider.copy(deep=True).reset_index(drop=True)
    snapshot.insert(
        0, "hazard_provider_row_order", np.arange(len(snapshot), dtype=np.int32)
    )
    return snapshot, _hazard_resolver_params(resolver), resolver_hash, preflight


SURFACE_SOURCE_COLUMNS = (
    "hole_no",
    "spatial_valid",
    "is_holeout",
    "common_end_long_yd",
    "common_end_lat_yd",
    "end_lie_class",
)


def _surface_snapshot(raw_course_app: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SURFACE_SOURCE_COLUMNS).difference(raw_course_app.columns))
    if missing:
        raise SourceAuthorityError(f"Raw APP surface misses fields={missing}")
    selected = raw_course_app.loc[
        raw_course_app["hole_no"].eq(EXPECTED_QUERY["hole_no"]),
        list(SURFACE_SOURCE_COLUMNS),
    ].copy()
    if selected.empty:
        raise SourceAuthorityError("The H8 APP surface source is empty")
    selected.reset_index(drop=True, inplace=True)
    selected.insert(
        0, "surface_row_order", np.arange(len(selected), dtype=np.int32)
    )
    return selected


def _surface_from_snapshot(snapshot: pd.DataFrame) -> object:
    required = ("surface_row_order", *SURFACE_SOURCE_COLUMNS)
    if tuple(snapshot.columns) != required:
        raise SourceAuthorityError("Surface snapshot columns or order changed")
    if not np.array_equal(
        snapshot["surface_row_order"].to_numpy(np.int64),
        np.arange(len(snapshot), dtype=np.int64),
    ):
        raise SourceAuthorityError("Surface snapshot row order is not contiguous")
    rows = snapshot.drop(columns="surface_row_order").copy()
    if not rows["hole_no"].eq(EXPECTED_QUERY["hole_no"]).all():
        raise SourceAuthorityError("Surface snapshot includes another hole")
    rows["end_x"] = rows["common_end_long_yd"].to_numpy(float) * 3.0
    rows["end_y"] = rows["common_end_lat_yd"].to_numpy(float) * 3.0
    return complete.app.AbsoluteSurfaceMap(rows)


def _load_and_seal_proposal() -> tuple[pd.DataFrame, pd.DataFrame]:
    if _sha256_file(PROPOSAL_SOURCE_FILE) != EXPECTED_PROPOSAL_PARQUET_SHA256:
        raise SourceAuthorityError("The real H8 proposal bytes changed")
    raw = pd.read_parquet(PROPOSAL_SOURCE_FILE)
    if len(raw) != EXPECTED_PROPOSAL_RECORDS:
        raise SourceAuthorityError("The real H8 proposal row count changed")
    if not np.array_equal(
        raw["proposal_row_order"].to_numpy(np.int64),
        np.arange(EXPECTED_PROPOSAL_RECORDS, dtype=np.int64),
    ):
        raise SourceAuthorityError("H8 proposal order is not contiguous")
    exact = {
        "support_key": EXPECTED_SUPPORT_KEY,
        "proposal_support_measure_sha256": EXPECTED_PROPOSAL_MEASURE_SHA256,
        "numeric_payload_sha256": EXPECTED_NUMERIC_PAYLOAD_SHA256,
        "source_cache_sha256": EXPECTED_SOURCE_CACHE_SHA256,
    }
    for column, expected in exact.items():
        values = set(raw[column].astype(str))
        if values != {expected}:
            raise SourceAuthorityError(f"H8 proposal {column} changed: {values}")
    if int(raw["terminal"].astype(bool).sum()) != EXPECTED_TERMINAL_ATOMS:
        raise SourceAuthorityError("H8 terminal proposal count changed")
    sealed = complete.apply_frozen_policy_to_proposal(raw)
    missing_ability = int((~sealed["donor_ability_available"].astype(bool)).sum())
    if missing_ability != EXPECTED_MISSING_PROPOSAL_ABILITIES:
        raise SourceAuthorityError(
            f"H8 missing-ability count changed: {missing_ability}"
        )
    policy_ids = set(sealed["donor_ability_policy_sha256"].astype(str))
    if policy_ids != {EXPECTED_ABILITY_POLICY_SHA256}:
        raise SourceAuthorityError(f"H8 ability policy changed: {policy_ids}")
    return raw, sealed


def _physical_atom_ids(proposal: pd.DataFrame) -> np.ndarray:
    ordered = proposal.sort_values("proposal_row_order", kind="mergesort")
    spatial_ids = ordered.loc[
        ~ordered["terminal"].astype(bool), "atom_id"
    ].astype(str)
    # The exact fixed-width dtype is part of the streaming payload identity.
    return np.asarray(spatial_ids.tolist(), dtype="<U64")


def _load_physical_numeric_support(
    proposal: pd.DataFrame,
) -> streaming.NumericSupport:
    if _sha256_file(NUMERIC_SOURCE_FILE) != EXPECTED_NUMERIC_NPZ_SHA256:
        raise SourceAuthorityError("The selected raw numeric cache bytes changed")
    with np.load(NUMERIC_SOURCE_FILE, allow_pickle=False) as data:
        expected_members = {
            "atom_ids",
            "points_yd",
            "probability",
            "cdf",
            "holeout_probability",
            "source_cache_sha256",
            "payload_sha256",
        }
        if set(data.files) != expected_members:
            raise SourceAuthorityError("The selected raw numeric NPZ schema changed")
        local_ids = np.asarray(data["atom_ids"])
        points = np.ascontiguousarray(data["points_yd"], dtype=float)
        probability = np.ascontiguousarray(data["probability"], dtype=float)
        cdf = np.ascontiguousarray(data["cdf"], dtype=float)
        holeout = float(np.asarray(data["holeout_probability"], float)[0])
        source_cache = str(np.asarray(data["source_cache_sha256"]).astype(str)[0])
        local_payload = str(np.asarray(data["payload_sha256"]).astype(str)[0])
    if not np.array_equal(local_ids, np.arange(EXPECTED_SURFACE_RECORDS)):
        raise SourceAuthorityError("Parent numeric cache local IDs changed")
    if source_cache != EXPECTED_SOURCE_CACHE_SHA256:
        raise SourceAuthorityError("Parent numeric source-cache identity changed")
    if local_payload != EXPECTED_PARENT_NUMERIC_LOCAL_PAYLOAD_SHA256:
        raise SourceAuthorityError("Parent numeric local payload identity changed")
    atom_ids = _physical_atom_ids(proposal)
    if len(atom_ids) != EXPECTED_SURFACE_RECORDS:
        raise SourceAuthorityError("Physical numeric atom count changed")
    payload = streaming.support_payload_sha256(
        EXPECTED_SUPPORT_KEY,
        atom_ids,
        points,
        probability,
        cdf,
        holeout,
    )
    if payload != EXPECTED_NUMERIC_PAYLOAD_SHA256:
        raise SourceAuthorityError(
            f"Physical numeric payload changed: {payload}"
        )
    return streaming.NumericSupport(
        key=EXPECTED_SUPPORT_KEY,
        atom_ids=atom_ids,
        points=points,
        probability=probability,
        cdf=cdf,
        holeout_probability=holeout,
        cache_sha256=source_cache,
        payload_sha256=payload,
    )


def _inventory_frame_hash(
    frame: pd.DataFrame, contract: str, columns: Sequence[str]
) -> str:
    expected_columns = ("inventory_record_order", *tuple(columns))
    if tuple(frame.columns) != expected_columns:
        raise SourceAuthorityError("Inventory projection schema or order changed")
    order = frame["inventory_record_order"].to_numpy(np.int64)
    if not np.array_equal(order, np.arange(len(frame), dtype=np.int64)):
        raise SourceAuthorityError("Inventory projection order is not contiguous")
    hasher = _OrderedInventoryHasher(contract, columns)
    for row in frame.loc[:, list(columns)].to_dict(orient="records"):
        hasher.add(row)
    return hasher.hexdigest()


def _source_inventory_sha256(inventory: Sequence[Mapping[str, object]]) -> str:
    records = []
    fields = (
        "query_key",
        "proposal_source_inventory_records",
        "proposal_source_inventory_sha256",
        "surface_source_inventory_records",
        "surface_source_inventory_sha256",
        "hazard_source_inventory_records",
        "hazard_source_inventory_sha256",
    )
    for row in sorted(inventory, key=lambda item: str(item["query_key"])):
        records.append(
            {field: _canonical_scalar(row[field]) for field in fields}
        )
    return _stable_sha(
        {"contract": complete.SOURCE_INVENTORY_CONTRACT, "queries": records}
    )


def _collect_independent_inventories(
    expansion: complete.CompleteAtomExpansion,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    parts: dict[str, list[pd.DataFrame]] = {
        "proposal": [],
        "surface": [],
        "hazard": [],
    }
    complete_rows = 0
    for batch in expansion.iter_batches():
        if len(batch) > MAX_BATCH_ROWS:
            raise SourceAuthorityError("Expansion yielded an over-cap batch")
        complete_rows += int(len(batch))
        first_branch = batch["mechanics_cdf_lower"].eq(0.0)
        selectors = {
            "proposal": first_branch,
            "surface": first_branch & ~batch["terminal"].astype(bool),
            "hazard": batch["hazard_resolution_status"].eq(
                "resolved_complete_atom"
            ),
        }
        for name, selector in selectors.items():
            columns = INVENTORY_SPECS[name][1]
            missing = [column for column in columns if column not in batch.columns]
            if missing:
                raise SourceAuthorityError(
                    f"Expansion inventory {name} misses columns={missing}"
                )
            projected = batch.loc[selector, list(columns)].copy()
            if not projected.empty:
                parts[name].append(projected)
    audit = asdict(expansion.audit)
    if complete_rows != EXPECTED_COMPLETE_ATOMS:
        raise SourceAuthorityError(
            f"Complete-atom count changed: {complete_rows} != {EXPECTED_COMPLETE_ATOMS}"
        )
    inventories: dict[str, pd.DataFrame] = {}
    for name, (_, columns, expected_records) in INVENTORY_SPECS.items():
        frame = (
            pd.concat(parts[name], ignore_index=True)
            if parts[name]
            else pd.DataFrame(columns=columns)
        )
        frame = frame.reindex(columns=list(columns))
        frame.insert(
            0,
            "inventory_record_order",
            np.arange(len(frame), dtype=np.int64),
        )
        if len(frame) != int(expected_records):
            raise SourceAuthorityError(
                f"{name} inventory count changed: {len(frame)} != {expected_records}"
            )
        observed_hash = _inventory_frame_hash(frame, *INVENTORY_SPECS[name][:2])
        audit_count = int(audit[f"{name}_source_inventory_records"])
        audit_hash = str(audit[f"{name}_source_inventory_sha256"])
        if audit_count != len(frame) or audit_hash != observed_hash:
            raise SourceAuthorityError(
                f"Independent {name} inventory disagrees with frozen seam audit"
            )
        inventories[name] = frame
    expected_audit = {
        "proposal_atoms": EXPECTED_PROPOSAL_RECORDS,
        "complete_atoms": EXPECTED_COMPLETE_ATOMS,
        "terminal_atoms": EXPECTED_TERMINAL_ATOMS,
        "imputed_ability_atoms": EXPECTED_IMPUTED_COMPLETE_ATOMS,
        "proposal_source_inventory_records": EXPECTED_PROPOSAL_RECORDS,
        "surface_source_inventory_records": EXPECTED_SURFACE_RECORDS,
        "hazard_source_inventory_records": EXPECTED_HAZARD_RECORDS,
    }
    for field, expected in expected_audit.items():
        if int(audit[field]) != expected:
            raise SourceAuthorityError(
                f"Expansion audit {field} changed: {audit[field]} != {expected}"
            )
    if not math.isclose(float(audit["quality_mass"]), 1.0, abs_tol=3e-13):
        raise SourceAuthorityError("Complete expansion mass does not close")
    if not math.isclose(float(audit["terminal_mass"]), 0.002, abs_tol=3e-13):
        raise SourceAuthorityError("Complete expansion terminal mass changed")
    return inventories, audit


def _parent_closure_state() -> dict[str, object]:
    proposal_index = _verify_external_artifact_closure(
        PROPOSAL_PARENT,
        output_index_sha256=str(
            EXPECTED_PARENT_CLOSURES["h8_proposal"]["output_index_sha256"]
        ),
        manifest_sha256=str(
            EXPECTED_PARENT_CLOSURES["h8_proposal"]["manifest_sha256"]
        ),
    )
    numeric_index = _verify_external_artifact_closure(
        NUMERIC_PARENT,
        output_index_sha256=str(
            EXPECTED_PARENT_CLOSURES["normalized_chain_50k"][
                "output_index_sha256"
            ]
        ),
        manifest_sha256=str(
            EXPECTED_PARENT_CLOSURES["normalized_chain_50k"]["manifest_sha256"]
        ),
    )
    proposal_entry = proposal_index.get("proposal_support_atoms.parquet")
    numeric_entry = numeric_index.get(NUMERIC_SOURCE_RELATIVE)
    if not isinstance(proposal_entry, Mapping) or str(
        proposal_entry.get("sha256")
    ) != EXPECTED_PROPOSAL_PARQUET_SHA256:
        raise SourceAuthorityError("Proposal parent index does not pin H8 atoms")
    proposal_inventory = proposal_index.get("proposal_support_inventory.parquet")
    if not isinstance(proposal_inventory, Mapping) or str(
        proposal_inventory.get("sha256")
    ) != EXPECTED_PROPOSAL_INVENTORY_SHA256:
        raise SourceAuthorityError("Proposal parent index does not pin H8 inventory")
    if not isinstance(numeric_entry, Mapping) or str(
        numeric_entry.get("sha256")
    ) != EXPECTED_NUMERIC_NPZ_SHA256:
        raise SourceAuthorityError("50k parent index does not pin the raw H8 cache")
    _verify_reference_closure()
    return {
        "h8_proposal": {
            **EXPECTED_PARENT_CLOSURES["h8_proposal"],
            "files": int(len(proposal_index)),
            "proposal_support_atoms_sha256": EXPECTED_PROPOSAL_PARQUET_SHA256,
            "proposal_support_inventory_sha256": (
                EXPECTED_PROPOSAL_INVENTORY_SHA256
            ),
        },
        "normalized_chain_50k": {
            **EXPECTED_PARENT_CLOSURES["normalized_chain_50k"],
            "files": int(len(numeric_index)),
            "selected_numeric_relative_path": NUMERIC_SOURCE_RELATIVE,
            "selected_numeric_sha256": EXPECTED_NUMERIC_NPZ_SHA256,
        },
        "fixed_reference": {
            "path": "permanent_data/hole_sim_fixed_reference_travelers_v2",
            "files": dict(EXPECTED_REFERENCE_FILES),
            "reference_model_sha256": EXPECTED_REFERENCE_MODEL_SHA256,
            "putting_curve_sha256": EXPECTED_PUTTING_SHA256,
        },
    }


def _validate_h8_parent_manifest(manifest: Mapping[str, object]) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SourceAuthorityError("H8 parent manifest has no inputs object")
    dependency_start = inputs.get("dependency_sha256_start")
    dependency_end = inputs.get("dependency_sha256_end")
    if not isinstance(dependency_start, Mapping) or not isinstance(
        dependency_end, Mapping
    ):
        raise SourceAuthorityError("H8 parent lacks start/end dependency maps")
    _require_equal_maps(dependency_start, dependency_end, "H8 parent dependencies")
    _assert_exact_map(
        dependency_start,
        EXPECTED_H8_RUNTIME_DEPENDENCY_SHA256,
        "H8 parent dependency closure",
    )
    prepared_inputs = inputs.get("prepared_input_sha256")
    prepared_logical = inputs.get("prepared_logical_sha256")
    if not isinstance(prepared_inputs, Mapping) or not isinstance(
        prepared_logical, Mapping
    ):
        raise SourceAuthorityError("H8 parent lacks prepared input identities")
    if str(manifest.get("contract")) != (
        "streaming_real_app_holeout_r1h8_raw_diagnostic_v1"
    ):
        raise SourceAuthorityError("H8 parent diagnostic contract changed")


def _verify_prepared_against_parent(
    prepared: streaming.PreparedStreamingEvent,
    parent_manifest: Mapping[str, object],
) -> None:
    inputs = parent_manifest["inputs"]
    assert isinstance(inputs, Mapping)
    expected_inputs = inputs["prepared_input_sha256"]
    expected_logical = inputs["prepared_logical_sha256"]
    assert isinstance(expected_inputs, Mapping)
    assert isinstance(expected_logical, Mapping)
    _assert_exact_map(
        prepared.input_hashes,
        expected_inputs,
        "prepared byte input closure",
    )
    _assert_exact_map(
        prepared.logical_hashes,
        expected_logical,
        "prepared logical input closure",
    )


def _source_inventory_record(audit: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "query_key",
        "proposal_source_inventory_records",
        "proposal_source_inventory_sha256",
        "surface_source_inventory_records",
        "surface_source_inventory_sha256",
        "hazard_source_inventory_records",
        "hazard_source_inventory_sha256",
    )
    return {field: audit[field] for field in fields}


def _frame_snapshot_sha256(
    frame: pd.DataFrame, *, contract: str
) -> str:
    columns = tuple(map(str, frame.columns))
    hasher = _OrderedInventoryHasher(contract, columns)
    for row in frame.to_dict(orient="records"):
        hasher.add(row)
    return hasher.hexdigest()


def _write_inventory_shards(
    output_dir: Path, inventories: Mapping[str, pd.DataFrame]
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for name, frame in inventories.items():
        directory = output_dir / "inventories" / name
        directory.mkdir(parents=True, exist_ok=False)
        shards = []
        for shard_index, start in enumerate(range(0, len(frame), INVENTORY_SHARD_ROWS)):
            stop = min(start + INVENTORY_SHARD_ROWS, len(frame))
            relative = (
                Path("inventories") / name / f"part-{shard_index:05d}.parquet"
            )
            path = output_dir / relative
            frame.iloc[start:stop].to_parquet(path, index=False)
            shards.append(
                {
                    "path": relative.as_posix(),
                    "records": int(stop - start),
                    "first_inventory_record_order": int(start),
                    "last_inventory_record_order": int(stop - 1),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
        contract, columns, expected_records = INVENTORY_SPECS[name]
        metadata[name] = {
            "contract": contract,
            "columns": list(columns),
            "records": int(len(frame)),
            "expected_records": int(expected_records),
            "inventory_sha256": _inventory_frame_hash(frame, contract, columns),
            "shard_rows_cap": INVENTORY_SHARD_ROWS,
            "shards": shards,
        }
    return metadata


def _write_physical_numeric(path: Path, numeric: streaming.NumericSupport) -> None:
    np.savez_compressed(
        path,
        support_key=np.asarray([numeric.key], dtype="<U128"),
        atom_ids=np.asarray(numeric.atom_ids, dtype="<U64"),
        points_yd=np.asarray(numeric.points, dtype=float),
        probability=np.asarray(numeric.probability, dtype=float),
        cdf=np.asarray(numeric.cdf, dtype=float),
        holeout_probability=np.asarray([numeric.holeout_probability], dtype=float),
        source_cache_sha256=np.asarray([numeric.cache_sha256], dtype="<U64"),
        payload_sha256=np.asarray([numeric.payload_sha256], dtype="<U64"),
    )


def _copy_candidate_sources(
    output_dir: Path,
    *,
    raw_proposal: pd.DataFrame,
    sealed_proposal: pd.DataFrame,
    numeric: streaming.NumericSupport,
    surface_snapshot: pd.DataFrame,
    hazard_snapshot: pd.DataFrame,
    resolver_params: Mapping[str, object],
    query: complete.CompleteAtomQuery,
) -> dict[str, object]:
    parents = output_dir / "parents"
    reference_dir = output_dir / "reference_model"
    parents.mkdir(parents=True, exist_ok=False)
    reference_dir.mkdir(parents=True, exist_ok=False)
    exact_copies = {
        "parents/h8_manifest.json": PROPOSAL_PARENT / "manifest.json",
        "parents/h8_OUTPUT_SHA256.json": PROPOSAL_PARENT / OUTPUT_INDEX,
        "parents/h8_proposal_support_atoms.parquet": PROPOSAL_SOURCE_FILE,
        "parents/normalized_chain_50k_manifest.json": NUMERIC_PARENT
        / "manifest.json",
        "parents/normalized_chain_50k_OUTPUT_SHA256.json": NUMERIC_PARENT
        / OUTPUT_INDEX,
        "parents/numeric_support_parent_raw.npz": NUMERIC_SOURCE_FILE,
        "reference_model/manifest.json": REFERENCE_PARENT / "manifest.json",
        "reference_model/reference_value_knots.parquet": REFERENCE_PARENT
        / "reference_value_knots.parquet",
        "reference_model/REPORT.md": REFERENCE_PARENT / "REPORT.md",
    }
    for relative, source in exact_copies.items():
        shutil.copyfile(source, output_dir / relative)
        if _sha256_file(source) != _sha256_file(output_dir / relative):
            raise SourceAuthorityError(f"Exact candidate copy failed: {relative}")
    # Re-reading the copied proposal proves its serialized source is the frame
    # on which the frozen policy was applied.
    copied_raw = pd.read_parquet(
        output_dir / "parents/h8_proposal_support_atoms.parquet"
    )
    if not copied_raw.equals(raw_proposal):
        raise SourceAuthorityError("Copied H8 proposal changed logical content")
    sealed_proposal.to_parquet(
        output_dir / "proposal_policy_v2.parquet", index=False
    )
    _write_physical_numeric(output_dir / "numeric_support_physical.npz", numeric)
    surface_snapshot.to_parquet(output_dir / "surface_source_h8.parquet", index=False)
    hazard_snapshot.to_parquet(output_dir / "hazard_provider.parquet", index=False)
    _write_json(output_dir / "hazard_resolver_params.json", resolver_params)
    _write_json(
        output_dir / "query.json",
        {
            "contract": complete.QUERY_CONTRACT,
            "query": _query_payload(query),
            "query_context_sha256": EXPECTED_QUERY_CONTEXT_SHA256,
        },
    )
    snapshot_paths = (
        "proposal_policy_v2.parquet",
        "numeric_support_physical.npz",
        "surface_source_h8.parquet",
        "hazard_provider.parquet",
        "hazard_resolver_params.json",
        "query.json",
        *exact_copies.keys(),
    )
    return {
        relative: {
            "bytes": int((output_dir / relative).stat().st_size),
            "sha256": _sha256_file(output_dir / relative),
        }
        for relative in snapshot_paths
    }


def _candidate_identifier(payload_without_id: Mapping[str, object]) -> str:
    return _stable_sha(
        {
            "contract": CONTRACT,
            "role": "review_only_non_authorizing_candidate",
            "payload": payload_without_id,
        }
    )


def _assert_candidate_non_authorizing(candidate: Mapping[str, object]) -> None:
    required = {
        "authority_id": None,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": ADMISSION_STATUS,
        "review_lock_sha256": None,
        "candidate_digest_is_not_authority": True,
    }
    for field, expected in required.items():
        if candidate.get(field) != expected:
            raise SourceAuthorityError(
                f"Candidate is not fail-closed: {field}={candidate.get(field)!r}"
            )


def _candidate_report(candidate: Mapping[str, object]) -> str:
    preflight = candidate["hazard_provider_preflight"]
    audit = candidate["expansion_audit"]
    assert isinstance(preflight, Mapping)
    assert isinstance(audit, Mapping)
    return (
        "# Lowry R1 H8 source-authority candidate\n\n"
        "Status: **external review required; not an authority**.\n\n"
        f"- Candidate ID: `{candidate['candidate_id']}`\n"
        f"- Frozen seam: `{EXPECTED_SEAM_SHA256}`\n"
        f"- Query key: `{audit['query_key']}`\n"
        f"- Source inventory: `{candidate['source_inventory_sha256']}`\n"
        f"- Provider preflight: {preflight['provider_donors']} total "
        f"({preflight['by_category']}); ceiling "
        f"{preflight['absolute_ceiling']}\n"
        f"- Complete atoms observed: {audit['complete_atoms']}\n"
        "- The candidate cannot admit the seam or enable production.\n"
        "- A different reviewer must pin the closed artifact and source "
        "inventory in a separate immutable lock.\n"
    )


def build_candidate(output_dir: Path) -> Path:
    """Build the one fixed real-H8 candidate into a fresh directory."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Candidate output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"Candidate parent directory does not exist: {output_dir.parent}"
        )
    source_start = _current_source_state()
    parent_start = _parent_closure_state()
    h8_manifest = _read_json(PROPOSAL_PARENT / "manifest.json")
    _validate_h8_parent_manifest(h8_manifest)

    # This is the sole heavy phase.  Every source/configuration argument is a
    # hard-coded project default; the CLI cannot replace any of them.
    prepared = streaming.prepare_streaming_event(
        export_root=precompute.DEFAULT_EXPORT,
        history_db=streaming.precompute.asof.DEFAULT_HISTORY_DB,
        profile_dir=precompute.DEFAULT_PROFILES,
        baseline_cases=precompute.DEFAULT_BASELINES,
        reference_dir=streaming.DEFAULT_REFERENCE,
        event_name=precompute.DEFAULT_EVENT,
        holdout_year=precompute.DEFAULT_HOLDOUT,
        course_start_year=2020,
        roster_seed=streaming.DEFAULT_SEED,
    )

    # The first builder-owned operation on the prepared scalar provider is the
    # 1046/682/364/8192 preflight.  Only after it passes may we hash or copy it.
    (
        hazard_snapshot,
        resolver_params,
        hazard_resolver_sha,
        provider_preflight,
    ) = _capture_hazard_snapshot(prepared.scalar_hazard)
    _verify_prepared_against_parent(prepared, h8_manifest)
    query = _select_exact_query(prepared)

    surface_snapshot = _surface_snapshot(prepared.atoms.raw_course_app)
    rebuilt_surface = _surface_from_snapshot(surface_snapshot)
    prepared_surface = spatial._surface_in_common_frame(
        prepared.atoms.raw_course_app
    ).get(int(EXPECTED_QUERY["hole_no"]))
    if prepared_surface is None:
        raise SourceAuthorityError("Prepared event has no H8 APP surface model")
    surface_sha = complete.surface_model_sha256(
        prepared_surface, "app", int(EXPECTED_QUERY["hole_no"])
    )
    rebuilt_surface_sha = complete.surface_model_sha256(
        rebuilt_surface, "app", int(EXPECTED_QUERY["hole_no"])
    )
    if rebuilt_surface_sha != surface_sha:
        raise SourceAuthorityError("Surface snapshot does not recreate H8 model")

    raw_proposal, sealed_proposal = _load_and_seal_proposal()
    numeric = _load_physical_numeric_support(sealed_proposal)
    _verify_reference_closure()
    reference_model = fixed_reference.ReferenceValueModel.load(REFERENCE_PARENT)
    if reference_model.model_sha256 != EXPECTED_REFERENCE_MODEL_SHA256:
        raise SourceAuthorityError("Fixed reference model identity changed")
    if reference_model.putting_curve_sha256 != EXPECTED_PUTTING_SHA256:
        raise SourceAuthorityError("Fixed putting identity changed")

    request = complete.CompleteAtomRequest(
        proposal_atoms=sealed_proposal,
        numeric_support=numeric,
        query=query,
        surface_map=prepared_surface,
        hazard_resolver=prepared.scalar_hazard,
        reference_model=reference_model,
    )
    expansion = complete.CompleteAtomExpansion(
        request,
        max_batch_rows=MAX_BATCH_ROWS,
        max_hazard_donors_per_surface=MAX_HAZARD_DONORS_PER_SURFACE,
        max_complete_atoms_per_proposal=MAX_COMPLETE_ATOMS_PER_PROPOSAL,
    )
    if expansion.hazard_sha != hazard_resolver_sha:
        raise SourceAuthorityError("Expansion hazard identity changed after preflight")
    if expansion.surface_sha != surface_sha:
        raise SourceAuthorityError("Expansion surface identity changed")
    inventories, expansion_audit = _collect_independent_inventories(expansion)

    source_inventory = _source_inventory_record(expansion_audit)
    source_inventory_sha = _source_inventory_sha256([source_inventory])
    input_end = streaming.verify_input_hashes_unchanged(prepared)
    _assert_exact_map(
        input_end,
        prepared.input_hashes,
        "prepared input end closure",
    )
    source_end = _current_source_state()
    _require_equal_maps(source_start, source_end, "builder source closure")
    parent_end = _parent_closure_state()
    _require_equal_maps(parent_start, parent_end, "parent artifact closure")

    output_dir.mkdir(parents=False, exist_ok=False)
    snapshot_files = _copy_candidate_sources(
        output_dir,
        raw_proposal=raw_proposal,
        sealed_proposal=sealed_proposal,
        numeric=numeric,
        surface_snapshot=surface_snapshot,
        hazard_snapshot=hazard_snapshot,
        resolver_params=resolver_params,
        query=query,
    )
    inventory_metadata = _write_inventory_shards(output_dir, inventories)

    candidate_body: dict[str, object] = {
        "contract": CONTRACT,
        "authority_id": None,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": ADMISSION_STATUS,
        "review_lock_sha256": None,
        "candidate_digest_is_not_authority": True,
        "scope": "lowry_r1h8_raw_app_exact_query_only",
        "query_contract": complete.QUERY_CONTRACT,
        "query": _query_payload(query),
        "query_context_sha256": EXPECTED_QUERY_CONTEXT_SHA256,
        "caps": {
            "max_batch_rows": MAX_BATCH_ROWS,
            "max_hazard_donors_per_surface": MAX_HAZARD_DONORS_PER_SURFACE,
            "absolute_hazard_provider_ceiling": ABSOLUTE_HAZARD_PROVIDER_CEILING,
            "max_complete_atoms_per_proposal": (
                MAX_COMPLETE_ATOMS_PER_PROPOSAL
            ),
            "inventory_shard_rows": INVENTORY_SHARD_ROWS,
        },
        "hazard_provider_preflight": provider_preflight,
        "hazard_resolver_sha256": hazard_resolver_sha,
        "surface_model_sha256": surface_sha,
        "reference_model_sha256": EXPECTED_REFERENCE_MODEL_SHA256,
        "putting_curve_sha256": EXPECTED_PUTTING_SHA256,
        "proposal_support_measure_sha256": EXPECTED_PROPOSAL_MEASURE_SHA256,
        "numeric_payload_sha256": EXPECTED_NUMERIC_PAYLOAD_SHA256,
        "source_cache_sha256": EXPECTED_SOURCE_CACHE_SHA256,
        "ability_policy_sha256": EXPECTED_ABILITY_POLICY_SHA256,
        "source_inventory_contract": complete.SOURCE_INVENTORY_CONTRACT,
        "source_inventory": [source_inventory],
        "source_inventory_sha256": source_inventory_sha,
        "expansion_audit": expansion_audit,
        "inventories": inventory_metadata,
        "snapshot_files": snapshot_files,
        "parent_artifact_closure_start": parent_start,
        "parent_artifact_closure_end": parent_end,
        "source_sha256_start": source_start,
        "source_sha256_end": source_end,
        "prepared_input_sha256_start": prepared.input_hashes,
        "prepared_input_sha256_end": input_end,
        "prepared_logical_sha256": prepared.logical_hashes,
        "external_review_requirements": {
            "reviewer_independent_of_builder": True,
            "reviewer_must_pin_candidate_output_index_sha256": True,
            "reviewer_must_pin_candidate_id": True,
            "reviewer_must_pin_source_inventory_sha256": True,
            "reviewer_must_pin_query_key": True,
            "reviewer_must_not_accept_runtime_or_candidate_supplied_digests": True,
            "separate_bounded_runner_required": True,
        },
    }
    candidate_id = _candidate_identifier(candidate_body)
    candidate = {"candidate_id": candidate_id, **candidate_body}
    _assert_candidate_non_authorizing(candidate)
    _write_json(output_dir / "authority_candidate.json", candidate)
    (output_dir / "REPORT.md").write_text(
        _candidate_report(candidate), encoding="utf-8"
    )

    files_before_manifest = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    expected_output_files = sorted(
        [*files_before_manifest, "manifest.json"]
    )
    manifest = {
        "contract": MANIFEST_CONTRACT,
        "candidate_id": candidate_id,
        "authority_ready": False,
        "production_enabled": False,
        "admission_status": ADMISSION_STATUS,
        "expected_output_files_before_index": expected_output_files,
        "payload_files": files_before_manifest,
        "source_inventory_sha256": source_inventory_sha,
        "query_key": expansion_audit["query_key"],
        "complete_atom_contract": complete.CONTRACT,
        "persistence_contract": complete.PERSISTENCE_CONTRACT,
        "note": (
            "OUTPUT_SHA256.json closes this candidate after manifest write; "
            "neither this manifest nor its checksum index is an authority lock."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_output_index(output_dir)
    validate_candidate_artifact(output_dir, verify_current_sources=True)
    return output_dir


def _read_inventory_shards(
    root: Path, name: str, metadata: Mapping[str, object]
) -> pd.DataFrame:
    contract, columns, expected_records = INVENTORY_SPECS[name]
    if str(metadata.get("contract")) != contract:
        raise SourceAuthorityError(f"{name} inventory contract changed")
    if metadata.get("columns") != list(columns):
        raise SourceAuthorityError(f"{name} inventory columns changed")
    if int(metadata.get("records", -1)) != expected_records:
        raise SourceAuthorityError(f"{name} inventory record count changed")
    if int(metadata.get("shard_rows_cap", -1)) != INVENTORY_SHARD_ROWS:
        raise SourceAuthorityError(f"{name} inventory shard cap changed")
    raw_shards = metadata.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise SourceAuthorityError(f"{name} inventory has no shards")
    parts = []
    next_order = 0
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            raise SourceAuthorityError(f"{name} inventory shard metadata is invalid")
        relative = _safe_index_entry(shard.get("path"))
        path = root / PurePosixPath(relative)
        if int(path.stat().st_size) != int(shard.get("bytes", -1)):
            raise SourceAuthorityError(f"{name} inventory shard byte size changed")
        if _sha256_file(path) != str(shard.get("sha256")):
            raise SourceAuthorityError(f"{name} inventory shard hash changed")
        part = pd.read_parquet(path)
        if len(part) > INVENTORY_SHARD_ROWS:
            raise SourceAuthorityError(f"{name} inventory shard exceeds cap")
        if tuple(part.columns) != ("inventory_record_order", *columns):
            raise SourceAuthorityError(f"{name} inventory shard schema changed")
        observed = part["inventory_record_order"].to_numpy(np.int64)
        expected = np.arange(next_order, next_order + len(part), dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise SourceAuthorityError(f"{name} inventory shard order changed")
        if int(shard.get("first_inventory_record_order", -1)) != next_order:
            raise SourceAuthorityError(f"{name} shard first order changed")
        if int(shard.get("last_inventory_record_order", -1)) != (
            next_order + len(part) - 1
        ):
            raise SourceAuthorityError(f"{name} shard last order changed")
        if int(shard.get("records", -1)) != len(part):
            raise SourceAuthorityError(f"{name} shard row count changed")
        next_order += len(part)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    observed_hash = _inventory_frame_hash(frame, contract, columns)
    if observed_hash != str(metadata.get("inventory_sha256")):
        raise SourceAuthorityError(f"{name} inventory identity changed")
    return frame


def _query_from_candidate(payload: Mapping[str, object]) -> complete.CompleteAtomQuery:
    expected_fields = set(EXPECTED_QUERY)
    if set(payload) != expected_fields:
        raise SourceAuthorityError("Candidate query schema changed")
    query = complete.CompleteAtomQuery(**dict(payload))
    _assert_exact_query(query)
    return query


def _load_candidate_numeric(
    path: Path, proposal: pd.DataFrame
) -> streaming.NumericSupport:
    with np.load(path, allow_pickle=False) as data:
        expected = {
            "support_key",
            "atom_ids",
            "points_yd",
            "probability",
            "cdf",
            "holeout_probability",
            "source_cache_sha256",
            "payload_sha256",
        }
        if set(data.files) != expected:
            raise SourceAuthorityError("Physical numeric snapshot schema changed")
        key = str(np.asarray(data["support_key"]).astype(str)[0])
        atom_ids = np.asarray(data["atom_ids"])
        points = np.ascontiguousarray(data["points_yd"], dtype=float)
        probability = np.ascontiguousarray(data["probability"], dtype=float)
        cdf = np.ascontiguousarray(data["cdf"], dtype=float)
        holeout = float(np.asarray(data["holeout_probability"], float)[0])
        cache_sha = str(np.asarray(data["source_cache_sha256"]).astype(str)[0])
        declared_payload = str(np.asarray(data["payload_sha256"]).astype(str)[0])
    if key != EXPECTED_SUPPORT_KEY:
        raise SourceAuthorityError("Physical numeric support key changed")
    if atom_ids.dtype != np.dtype("<U64"):
        raise SourceAuthorityError(
            f"Physical numeric atom dtype changed: {atom_ids.dtype}"
        )
    if not np.array_equal(atom_ids, _physical_atom_ids(proposal)):
        raise SourceAuthorityError("Physical numeric atom IDs changed")
    observed_payload = streaming.support_payload_sha256(
        key, atom_ids, points, probability, cdf, holeout
    )
    if declared_payload != observed_payload or observed_payload != (
        EXPECTED_NUMERIC_PAYLOAD_SHA256
    ):
        raise SourceAuthorityError("Physical numeric payload identity changed")
    if cache_sha != EXPECTED_SOURCE_CACHE_SHA256:
        raise SourceAuthorityError("Physical numeric source-cache identity changed")
    if len(atom_ids) != EXPECTED_SURFACE_RECORDS:
        raise SourceAuthorityError("Physical numeric record count changed")
    if not math.isclose(holeout, 0.002, rel_tol=0.0, abs_tol=0.0):
        raise SourceAuthorityError("Physical numeric holeout mass changed")
    if cdf.shape != probability.shape or not math.isclose(
        float(cdf[-1]), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise SourceAuthorityError("Physical numeric CDF does not close exactly")
    return streaming.NumericSupport(
        key=key,
        atom_ids=atom_ids,
        points=points,
        probability=probability,
        cdf=cdf,
        holeout_probability=holeout,
        cache_sha256=cache_sha,
        payload_sha256=observed_payload,
    )


def _parquet_metadata_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pandas parquet needs it here
        raise SourceAuthorityError("pyarrow is required for pre-load preflight") from exc
    metadata = pq.ParquetFile(path).metadata
    return int(metadata.num_rows)


def validate_candidate_artifact(
    output_dir: Path, *, verify_current_sources: bool = True
) -> dict[str, object]:
    """Validate a closed candidate without converting it into an authority."""

    output_dir = Path(output_dir).resolve()
    entries = _validate_closed_output_index(output_dir)
    if "authority_lock.json" in entries:
        raise SourceAuthorityError("A builder candidate cannot contain an authority lock")
    manifest = _read_json(output_dir / "manifest.json")
    if manifest.get("contract") != MANIFEST_CONTRACT:
        raise SourceAuthorityError("Candidate manifest contract changed")
    if manifest.get("authority_ready") is not False or manifest.get(
        "production_enabled"
    ) is not False:
        raise SourceAuthorityError("Candidate manifest is not fail-closed")
    if manifest.get("admission_status") != ADMISSION_STATUS:
        raise SourceAuthorityError("Candidate manifest admission state changed")
    expected_files = manifest.get("expected_output_files_before_index")
    if not isinstance(expected_files, list) or set(map(str, expected_files)) != set(
        entries
    ):
        raise SourceAuthorityError("Manifest does not enumerate the closed artifact")
    payload_files = manifest.get("payload_files")
    if not isinstance(payload_files, Mapping):
        raise SourceAuthorityError("Manifest payload-file map is absent")
    expected_payload_files = {
        name: metadata for name, metadata in entries.items() if name != "manifest.json"
    }
    if dict(payload_files) != expected_payload_files:
        raise SourceAuthorityError("Manifest payload-file map changed")

    candidate = _read_json(output_dir / "authority_candidate.json")
    _assert_candidate_non_authorizing(candidate)
    if candidate.get("contract") != CONTRACT:
        raise SourceAuthorityError("Authority candidate contract changed")
    body = dict(candidate)
    candidate_id = str(body.pop("candidate_id", ""))
    if candidate_id != _candidate_identifier(body):
        raise SourceAuthorityError("Authority candidate identity changed")
    if manifest.get("candidate_id") != candidate_id:
        raise SourceAuthorityError("Manifest/candidate identities disagree")

    caps = candidate.get("caps")
    expected_caps = {
        "max_batch_rows": MAX_BATCH_ROWS,
        "max_hazard_donors_per_surface": MAX_HAZARD_DONORS_PER_SURFACE,
        "absolute_hazard_provider_ceiling": ABSOLUTE_HAZARD_PROVIDER_CEILING,
        "max_complete_atoms_per_proposal": MAX_COMPLETE_ATOMS_PER_PROPOSAL,
        "inventory_shard_rows": INVENTORY_SHARD_ROWS,
    }
    if caps != expected_caps:
        raise SourceAuthorityError(f"Candidate caps changed: {caps}")
    query_payload = candidate.get("query")
    if not isinstance(query_payload, Mapping):
        raise SourceAuthorityError("Candidate query is absent")
    query = _query_from_candidate(query_payload)
    if candidate.get("query_context_sha256") != EXPECTED_QUERY_CONTEXT_SHA256:
        raise SourceAuthorityError("Candidate query context identity changed")
    query_file = _read_json(output_dir / "query.json")
    if query_file.get("contract") != complete.QUERY_CONTRACT or query_file.get(
        "query"
    ) != dict(query_payload):
        raise SourceAuthorityError("Candidate query file disagrees with candidate")

    source_start = candidate.get("source_sha256_start")
    source_end = candidate.get("source_sha256_end")
    if not isinstance(source_start, Mapping) or not isinstance(source_end, Mapping):
        raise SourceAuthorityError("Candidate source start/end maps are absent")
    _require_equal_maps(source_start, source_end, "candidate source closure")
    if verify_current_sources:
        _require_equal_maps(
            source_start, _current_source_state(), "candidate/current source closure"
        )
    parent_start = candidate.get("parent_artifact_closure_start")
    parent_end = candidate.get("parent_artifact_closure_end")
    if not isinstance(parent_start, Mapping) or not isinstance(parent_end, Mapping):
        raise SourceAuthorityError("Candidate parent start/end closures are absent")
    _require_equal_maps(parent_start, parent_end, "candidate parent closure")
    if dict(parent_start) != _parent_closure_state():
        raise SourceAuthorityError("Candidate parent closure is no longer exact")
    prepared_start = candidate.get("prepared_input_sha256_start")
    prepared_end = candidate.get("prepared_input_sha256_end")
    if not isinstance(prepared_start, Mapping) or not isinstance(
        prepared_end, Mapping
    ):
        raise SourceAuthorityError("Candidate prepared start/end maps are absent")
    _require_equal_maps(prepared_start, prepared_end, "candidate prepared inputs")

    snapshot_files = candidate.get("snapshot_files")
    if not isinstance(snapshot_files, Mapping):
        raise SourceAuthorityError("Candidate snapshot file map is absent")
    for relative, metadata in snapshot_files.items():
        if entries.get(str(relative)) != metadata:
            raise SourceAuthorityError(f"Snapshot metadata changed: {relative}")
    if _sha256_file(
        output_dir / "parents" / "h8_proposal_support_atoms.parquet"
    ) != EXPECTED_PROPOSAL_PARQUET_SHA256:
        raise SourceAuthorityError("Copied raw proposal changed")
    if _sha256_file(
        output_dir / "parents" / "numeric_support_parent_raw.npz"
    ) != EXPECTED_NUMERIC_NPZ_SHA256:
        raise SourceAuthorityError("Copied parent numeric cache changed")
    _verify_reference_closure(output_dir / "reference_model")

    proposal = pd.read_parquet(output_dir / "proposal_policy_v2.parquet")
    if len(proposal) != EXPECTED_PROPOSAL_RECORDS:
        raise SourceAuthorityError("Candidate proposal record count changed")
    complete.apply_frozen_policy_to_proposal(proposal)
    if set(proposal["proposal_support_measure_sha256"].astype(str)) != {
        EXPECTED_PROPOSAL_MEASURE_SHA256
    }:
        raise SourceAuthorityError("Candidate proposal measure changed")
    if set(proposal["donor_ability_policy_sha256"].astype(str)) != {
        EXPECTED_ABILITY_POLICY_SHA256
    }:
        raise SourceAuthorityError("Candidate proposal ability policy changed")
    numeric = _load_candidate_numeric(
        output_dir / "numeric_support_physical.npz", proposal
    )
    if numeric.payload_sha256 != candidate.get("numeric_payload_sha256"):
        raise SourceAuthorityError("Candidate numeric identity disagrees")

    surface_source = pd.read_parquet(output_dir / "surface_source_h8.parquet")
    surface_model = _surface_from_snapshot(surface_source)
    surface_sha = complete.surface_model_sha256(
        surface_model, "app", int(query.hole_no)
    )
    if surface_sha != candidate.get("surface_model_sha256"):
        raise SourceAuthorityError("Candidate surface identity changed")

    provider_path = output_dir / "hazard_provider.parquet"
    # Metadata count is checked before pandas materializes or copies provider rows.
    if _parquet_metadata_rows(provider_path) != EXPECTED_PROVIDER_TOTAL:
        raise SourceAuthorityError("Hazard provider Parquet row count changed")
    provider_snapshot = pd.read_parquet(provider_path)
    expected_provider_columns = ("hazard_provider_row_order",)
    if tuple(provider_snapshot.columns[:1]) != expected_provider_columns:
        raise SourceAuthorityError("Hazard provider envelope changed")
    if not np.array_equal(
        provider_snapshot["hazard_provider_row_order"].to_numpy(np.int64),
        np.arange(EXPECTED_PROVIDER_TOTAL, dtype=np.int64),
    ):
        raise SourceAuthorityError("Hazard provider order changed")
    provider = provider_snapshot.drop(columns="hazard_provider_row_order")
    _preflight_hazard_provider(SimpleNamespace(donors=provider))
    resolver_params = _read_json(output_dir / "hazard_resolver_params.json")
    constructor_fields = (
        "setup_prior",
        "hole_prior",
        "spatial_bandwidth_yd",
        "max_nearest_distance_yd",
        "same_hole_fallback_max_distance_yd",
        "minimum_category_support",
    )
    rebuilt_resolver = hazard.EndpointLinkedHazardResolver(
        provider,
        **{field: resolver_params[field] for field in constructor_fields},
    )
    _preflight_hazard_provider(rebuilt_resolver)
    if _hazard_resolver_params(rebuilt_resolver) != resolver_params:
        raise SourceAuthorityError("Hazard resolver parameters changed")
    resolver_sha = complete.hazard_resolver_sha256(
        rebuilt_resolver,
        max_donor_count=MAX_HAZARD_DONORS_PER_SURFACE,
    )
    if resolver_sha != candidate.get("hazard_resolver_sha256"):
        raise SourceAuthorityError("Candidate hazard resolver identity changed")

    reference_model = fixed_reference.ReferenceValueModel.load(
        output_dir / "reference_model"
    )
    if reference_model.model_sha256 != EXPECTED_REFERENCE_MODEL_SHA256:
        raise SourceAuthorityError("Candidate reference model changed")

    raw_inventory_metadata = candidate.get("inventories")
    if not isinstance(raw_inventory_metadata, Mapping):
        raise SourceAuthorityError("Candidate inventory metadata is absent")
    inventory_frames: dict[str, pd.DataFrame] = {}
    for name in INVENTORY_SPECS:
        metadata = raw_inventory_metadata.get(name)
        if not isinstance(metadata, Mapping):
            raise SourceAuthorityError(f"Candidate {name} inventory is absent")
        inventory_frames[name] = _read_inventory_shards(output_dir, name, metadata)

    audit = candidate.get("expansion_audit")
    if not isinstance(audit, Mapping):
        raise SourceAuthorityError("Candidate expansion audit is absent")
    expected_audit_counts = {
        "proposal_atoms": EXPECTED_PROPOSAL_RECORDS,
        "complete_atoms": EXPECTED_COMPLETE_ATOMS,
        "terminal_atoms": EXPECTED_TERMINAL_ATOMS,
        "imputed_ability_atoms": EXPECTED_IMPUTED_COMPLETE_ATOMS,
    }
    for field, expected in expected_audit_counts.items():
        if int(audit.get(field, -1)) != expected:
            raise SourceAuthorityError(f"Candidate expansion {field} changed")
    for name, frame in inventory_frames.items():
        contract, columns, expected_records = INVENTORY_SPECS[name]
        digest = _inventory_frame_hash(frame, contract, columns)
        if int(audit.get(f"{name}_source_inventory_records", -1)) != (
            expected_records
        ) or str(audit.get(f"{name}_source_inventory_sha256")) != digest:
            raise SourceAuthorityError(
                f"Candidate {name} inventory disagrees with expansion audit"
            )
    inventory_record = _source_inventory_record(audit)
    source_inventory_sha = _source_inventory_sha256([inventory_record])
    if candidate.get("source_inventory") != [inventory_record] or candidate.get(
        "source_inventory_sha256"
    ) != source_inventory_sha:
        raise SourceAuthorityError("Candidate global source inventory changed")
    if manifest.get("source_inventory_sha256") != source_inventory_sha:
        raise SourceAuthorityError("Manifest source inventory changed")
    if manifest.get("query_key") != audit.get("query_key"):
        raise SourceAuthorityError("Manifest query key changed")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed, non-authorizing Lowry R1 H8 TEE/APP source "
            "inventory candidate. No source/query/cap overrides exist."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh versioned directory for the review-only candidate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_candidate(args.output_dir)
    candidate = _read_json(output / "authority_candidate.json")
    print(f"Wrote non-authorizing candidate: {output}")
    print(f"Candidate ID: {candidate['candidate_id']}")
    print("HOLD: external source review is required before any authority lock.")


if __name__ == "__main__":
    main()





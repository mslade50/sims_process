"""Independent byte and vectorized-invariant audit for the fixed H8 candidate.

Canonical selector digests are sealed external evidence. This module does not
import or call the builder, selector, or their validators, and does not
reimplement their canonical scalar hashing. A separately reviewed launcher
captures and executes these exact source bytes.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = Path(__file__).resolve()
AUDITOR_TEST = REPO / "test_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
LAUNCHER_SOURCE = REPO / "archive" / "run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
LAUNCHER_TEST = REPO / "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"

SCOPE = "independent_byte_and_vectorized_invariant_audit_v2"
AUDIT_CONTRACT = "fixed_lowry_r1h8_real_app_selector_candidate_detached_audit_v2"
MANIFEST_CONTRACT = "fixed_lowry_r1h8_real_app_selector_candidate_audit_manifest_v2"
INDEX_CONTRACT = "non_circular_output_sha256_v1"
AUDIT_STATUS = "candidate_byte_and_vectorized_audit_passed_record_review_required"
DECISION = "GO_for_separate_data_only_authority_record_review_only"

FIXED_CANDIDATE = REPO / "permanent_data" / "hole_sim_real_app_h8_selector_authority_candidate_v1"
FIXED_ORIGINAL = REPO / "permanent_data" / "hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1"
FIXED_OUTPUT = REPO / "permanent_data" / "hole_sim_real_app_h8_selector_authority_candidate_audit_v2"

EXPECTED_CANDIDATE_ID = "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942"
EXPECTED_QUERY_KEY = "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
EXPECTED_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
REFERENCE_MODEL = "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
PUTTING_MODEL = "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
POLICY_IDENTITY = "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
AUDITOR_TEST_SHA256 = "11f9a55e89a22e1bb77addf09dc9fb395fd645cae6c6337c0d4a242147017eb9"

EXTERNAL_CANONICAL_EVIDENCE = {
    "full_schema_sha256": "113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d",
    "full_support_sha256": "b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7",
    "projection_schema_sha256": "855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2",
    "projection_54_sha256": "a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889",
    "selector_native_sha256": "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c",
    "topology_30_sha256": "9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484",
}

CANDIDATE_FILES = {
    "OUTPUT_SHA256.json": (474, "ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43"),
    "REPORT.md": (519, "5489db51378a60035b7d090036abcfe43f34878cf7ba2e6ce2fa4d1bbeacf879"),
    "authority_candidate.json": (5_283, "689b5c0ae5827ea99fd603a7b3af39163685d498dd7c1ac85c0777758e2d6e00"),
    "manifest.json": (927, "b5b5f07740dd43e8933aca21674351b435918b803a60fdfc5ad7dad78364c040"),
}
PARTS = (
    ("complete_atom_support/parts/part-000000.parquet", 8_192, 1_751_670, "3f3905bc2874e81baade5618a26b316e7a97da17dc3966766e22bfef602e38c7", "0f225adf7b7074c5ba7f27a948756bf708c876628498716856627f9aa93bfaf8"),
    ("complete_atom_support/parts/part-000001.parquet", 8_192, 1_767_025, "4c1c349977834aeab644b1a0b37fe99512c66fbe3332495eaf871788f1131ed6", "1c56da0e51b3627246d62bf0a065ff4799475bc6b2b900fdf6ae8487f9c27c6a"),
    ("complete_atom_support/parts/part-000002.parquet", 8_192, 1_748_952, "bcac8a464d13dd52eaaa3a9bc3e135d9022843c3c52613f0feca491aa72bb306", "96c75b4165824b28928081ced2711d3954cd132bb177694616e1d95268cd81be"),
    ("complete_atom_support/parts/part-000003.parquet", 8_192, 1_762_480, "9fd805ad604790e366f99f2435742866e2e72bf771a5a97bfbf6327d14ef33ca", "9bfd7255baa22069417df9303261cc307e13b66c94dc8be1263bbcb3ea3d598d"),
    ("complete_atom_support/parts/part-000004.parquet", 2_450, 600_833, "ef93b0721901b106f20e6bb6518c2875c5224bb545f988b934fd254e5b23e0a2", "00b8393e43c95aef677532b1cc79d2815be32736c3eda828214406cd30376a09"),
)
ORIGINAL_FILES = {
    "OUTPUT_SHA256.json": (1_642, "5a502f524ada60bb91e85f1e18474056d4b2a883be14c0e1f7df3450393093b2"),
    "REPORT.md": (395, "c56b043fd88dae28f4c4af23fe4b0c6d5d8f4139115acb3a9ae495b700f62dc7"),
    "complete_atom_support/OUTPUT_SHA256.json": (974, "e300bf8b00066ec02e03128ca8e0e6b7a89c4e5a772a2ac5f2dd5794dbe2693f"),
    "complete_atom_support/manifest.json": (6_495, "4dec00946a085fedb489aa51fef672188d4205c6a26c85faeb7156e5234da7f2"),
    "manifest.json": (7_267, "c96a8e8333c8e535a82e9e265f74cde41d6a5a717e36c68c9dea9894e39f5464"),
    **{path: (size, byte_sha) for path, _rows, size, byte_sha, _logical in PARTS},
}

SOURCE_PINS = {
    "archive/hole_sim_real_app_h8_selector_authority_candidate.py": "b3fdbd12f2e466bdac92f71f2d60a3e9fd0aec18a58f309248dc1c783046fdd7",
    "test_hole_sim_real_app_h8_selector_authority_candidate.py": "ebea0b3f4f4e196931908f89eeb3c7189b79784cfb5eb53a2f8027a342773db4",
    "archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": "f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0",
    "archive/hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": "ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460",
    "archive/hole_sim_tee_app_complete_atom_support.py": "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498",
    "archive/hole_sim_donor_ability_policy.py": "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339",
    "archive/hole_sim_transition_mean_anchor.py": "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28",
    "archive/hole_sim_transition_mean_chain_adapter.py": "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    "archive/hole_sim_tee_app_h8_source_authority.py": "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182",
    "archive/audit_hole_sim_tee_app_h8_source_authority.py": "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06",
    "test_hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": "bd3875f4de9e9082597a065f719e5b09bc053a5722e30f461f219e6dd1a1f4c9",
    "test_hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": "61103a4f410fb2f2fcbdcec2ad20dc40c13c2c37a8975bae4587aa839b78153f",
    "test_hole_sim_tee_app_complete_atom_support.py": "eb648d94447d8a60497dcf95b34dcbbe1a64c21cd5458bb27ad87c1673898397",
    "archive/hole_sim_compact_app_whole_atom_selector_v1.py": "e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7",
    "test_hole_sim_compact_app_whole_atom_selector_v1.py": "21bd3aa8f0f1d2340bdefb32b3fce4bae59fdbdc87c23de3129710c7287b77dd",
    "test_hole_sim_tee_app_h8_source_authority.py": "93966e84c85d7d3440516f0c80492e8c7d9145448aeaf88859a8cd833f115813",
    "test_audit_hole_sim_tee_app_h8_source_authority.py": "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c",
}
V1_DIAGNOSTIC_PIN = "e39041d173b42b609d3b37936cb210c57a946d03a905a85b484249dd156810ac"

FAIL_CLOSED = {
    "authority_ready": False, "real_support_admitted": False,
    "selector_support_ready": False, "weighting_enabled": False,
    "selection_enabled": False, "chain_ready": False, "anchor_ready": False,
    "callback_enabled": False, "callback_launched": False,
    "simulation_launched": False, "mean_solve_launched": False,
    "production_enabled": False,
}
NEXT_REVIEW = {
    "audit_id_is_not_authority": True, "candidate_id_is_not_authority": True,
    "separate_data_only_authority_record_required": True,
    "data_only_authority_record_source_review_required": True,
    "data_only_authority_record_artifact_review_required": True,
}

PROJECTION_COLUMNS = (
    "query_key", "complete_row_order", "atom_id", "category", "complete_atom",
    "boundary_only", "proposal_row_order", "proposal_probability",
    "proposal_branch", "terminal", "mechanics_conditional_probability",
    "quality_base_mass", "fixed_reference_reward", "donor_quality_group_id",
    "start_state_key", "declared_app_base_holeout_probability",
    "frozen_spatial_cdf_lower", "frozen_spatial_cdf_upper",
    "frozen_terminal_residual_cdf_lower", "frozen_terminal_residual_cdf_upper",
    "surface_class_count", "surface_full_probability_json",
    "surface_full_cdf_upper_json", "surface_class_index",
    "surface_class_label", "surface_probability", "surface_cdf_lower",
    "surface_cdf_upper", "hazard_resolution_status", "hazard_donor_order",
    "hazard_donor_key", "hazard_probability", "hazard_cdf_lower",
    "hazard_cdf_upper", "stroke_cost", "end_state_key", "next_support_key",
    "reference_model_sha256", "reference_putting_sha256", "bridge_row_id",
    "donor_player_id", "donor_event_key", "donor_event_start",
    "target_event_key", "target_event_start", "donor_skill_date",
    "donor_ability_source_column", "donor_ability_available",
    "donor_ability_raw_strict_prior", "donor_ability_value_kind",
    "donor_ability_sg", "donor_ability_was_clipped",
    "donor_ability_policy_sha256", "donor_ability_row_sha256",
)
EXTRA_COLUMNS = (
    "atom_kind", "complete_cdf_lower", "complete_cdf_upper",
    "mechanics_cdf_lower", "mechanics_cdf_upper", "proposal_cdf_lower",
    "proposal_cdf_upper", "proposal_branch_conditional_probability",
    "proposal_branch_cdf_lower", "proposal_branch_cdf_upper",
    "proposal_base_hierarchy_mass", "base_hierarchy_mass",
    "start_reference_value", "end_reference_value", "reference_descriptor",
    "reference_version", "source_cache_sha256", "surface_model_sha256",
    "hazard_resolver_sha256", "target_player_id", "proposal_atom_id",
    "proposal_spatial_model_weight", "spatial_model_weight",
    "donor_player_name", "donor_ability_snapshot_kind",
    "proposal_policy_binding_sha256", "causal_strokes", "penalty_strokes",
    "quality_reward_available", "holed", "next_category",
)
READ_COLUMNS = tuple(dict.fromkeys(PROJECTION_COLUMNS + EXTRA_COLUMNS))

EXPECTED_NDJSON = {
    "schema": {
        "contract": "lowry_r1h8_v2_schema_ndjson_sha256",
        "sha256": "f711d7c44747222b7a5abde522996b8e4d341c694f0667acf936024d54cee944",
        "records": 151, "framed_bytes": 14_888,
    },
    "topology": {
        "contract": "lowry_r1h8_v2_topology_ndjson_sha256",
        "sha256": "f20b53bf6cb5310685281f4519fbeeaa6ae6937122c13079c7a6175020924d97",
        "records": 905, "framed_bytes": 2_968_153,
    },
    "mass_cdf": {
        "contract": "lowry_r1h8_v2_mass_cdf_ndjson_sha256",
        "sha256": "bcd7acafb2fed51320e96e0e9dbf2b30ca4e50a964d5ddaefa1c9428d5df19be",
        "records": 905, "framed_bytes": 13_115_551,
    },
    "group": {
        "contract": "lowry_r1h8_v2_group_ndjson_sha256",
        "sha256": "cebf146bf965e293e2b5be8cb29895493c4cddba08530f8764f78b021aa33484",
        "records": 7, "framed_bytes": 5_653,
    },
    "ability_policy": {
        "contract": "lowry_r1h8_v2_ability_policy_ndjson_sha256",
        "sha256": "acda28fae2bdf86a038b369f5bfcb6f711ca26204acfc474ed22b8421fafba55",
        "records": 905, "framed_bytes": 887_469,
    },
    "reference_reward": {
        "contract": "lowry_r1h8_v2_reference_reward_ndjson_sha256",
        "sha256": "45cf3527ee448cb2d008ea155ec6220d2a0a80391e970d5a48842d75f953e88c",
        "records": 35_219, "framed_bytes": 16_878_023,
    },
    "semantic_root": {
        "contract": "lowry_r1h8_v2_semantic_root_ndjson_sha256",
        "sha256": "b8312616d491bdce5e97fa438e95a8b9ead36ce1cb0fb9d7274605cfe3675dae",
        "records": 1, "framed_bytes": 3_379,
    },
}
EXPECTED_SEMANTIC_ROOT = EXPECTED_NDJSON["semantic_root"]["sha256"]
EXPECTED_SCHEMA = "arm@string;atom_id@string;atom_kind@string;base_hierarchy_mass@double;boundary_only@bool;bridge_row_id@string;category@string;causal_strokes@int64;complete_atom@bool;complete_cdf_lower@double;complete_cdf_upper@double;complete_holeout_equivalence_status@string;complete_row_order@int64;contract@string;declared_app_base_holeout_probability@double;donor_ability_available@bool;donor_ability_policy_sha256@string;donor_ability_raw_strict_prior@double;donor_ability_row_sha256@string;donor_ability_sg@double;donor_ability_snapshot_kind@string;donor_ability_source_column@string;donor_ability_value_kind@string;donor_ability_was_clipped@bool;donor_event_key@string;donor_event_start@timestamp[ns, tz=UTC];donor_hole_no@int64;donor_player_id@string;donor_player_name@string;donor_quality_group_id@string;donor_round_no@int64;donor_skill_date@timestamp[ns, tz=UTC];donor_stroke_no@int64;end_distance_yd@double;end_lat_yd@double;end_lie@string;end_long_yd@double;end_reference_value@double;end_state_key@string;fixed_reference_reward@double;forecast_pin_long_yd@double;forecast_pin_x@double;forecast_pin_y@double;forecast_tee_x@double;forecast_tee_y@double;frozen_spatial_cdf_lower@double;frozen_spatial_cdf_upper@double;frozen_terminal_residual_cdf_lower@double;frozen_terminal_residual_cdf_upper@double;hazard_causal_endpoint_x@double;hazard_causal_endpoint_y@double;hazard_cdf_lower@double;hazard_cdf_upper@double;hazard_component_masses@string;hazard_coordinate_reference_event_key@string;hazard_coordinate_reference_mode@string;hazard_coordinate_reference_year@int64;hazard_data_role@string;hazard_donor_category@string;hazard_donor_causal_endpoint_x@double;hazard_donor_causal_endpoint_y@double;hazard_donor_event_key@string;hazard_donor_hole_no@int64;hazard_donor_key@string;hazard_donor_order@int64;hazard_donor_physical_course_id@string;hazard_donor_round_no@int64;hazard_donor_season@int64;hazard_fallback_used@bool;hazard_foundation_version@string;hazard_nearest_category_endpoint_distance_yd@double;hazard_nearest_same_hole_endpoint_distance_yd@double;hazard_penalty_count@int64;hazard_penalty_stroke_cost@int64;hazard_probability@double;hazard_resolution_event_type@string;hazard_resolution_lie@string;hazard_resolution_shot_key@string;hazard_resolution_status@string;hazard_resolution_x@double;hazard_resolution_y@double;hazard_resolver_sha256@string;hazard_support_category@int64;hazard_support_hole@int64;hazard_support_radius_limit_yd@double;hazard_support_setup@int64;hazard_support_source@string;hole_no@int64;holed@bool;holeout_representation@string;implicit_holeout_probability@double;mechanics_cdf_lower@double;mechanics_cdf_upper@double;mechanics_conditional_probability@double;next_category@string;next_support_key@string;numeric_payload_sha256@string;par@int64;penalty_strokes@int64;physical_course_id@string;proposal_app_holeout_equivalence_sha256@string;proposal_atom_id@string;proposal_base_hierarchy_mass@double;proposal_branch@string;proposal_branch_cdf_lower@double;proposal_branch_cdf_upper@double;proposal_branch_conditional_probability@double;proposal_cdf_lower@double;proposal_cdf_upper@double;proposal_endpoint_lat_yd@double;proposal_endpoint_long_yd@double;proposal_policy_binding_sha256@string;proposal_probability@double;proposal_row_order@int64;proposal_spatial_model_weight@double;proposal_support_key@string;proposal_support_measure_sha256@string;quality_base_mass@double;quality_reward_available@bool;query_key@string;reference_descriptor@string;reference_model_sha256@string;reference_putting_sha256@string;reference_version@string;round_no@int64;selection_contract@string;source_cache_sha256@string;spatial_model_weight@double;start_distance_yd@double;start_lat_yd@double;start_lie@string;start_long_yd@double;start_reference_value@double;start_state_key@string;stroke_cost@int64;support_key@string;surface_cdf_lower@double;surface_cdf_upper@double;surface_class_count@int64;surface_class_index@int64;surface_class_label@string;surface_full_cdf_upper_json@string;surface_full_probability_json@string;surface_model_sha256@string;surface_probability@double;surface_selection_boundary@string;target_event_key@string;target_event_start@timestamp[ns, tz=UTC];target_player_id@string;terminal@bool"

GROUP_EXPECTATIONS = {
    "03b7d145875af9889ce925a6e93718b1d5cdad75af6187bd91daafa4ffc2d2d8":
        (7_332, 188, 6_825, 507, 175, 13, "pga:R2022034", "2022-06-23T00:00:00+00:00", 0.22, (1, 2)),
    "0625e8c3727ddf1b2539f5b74f486543e2238f22b8915d13e622d1bbff039816":
        (7_566, 194, 6_942, 624, 178, 16, "pga:R2021034", "2021-06-24T00:00:00+00:00", 0.22199999999999998, (0, 1)),
    "3954437bb398899d7be29d91971ee8ab9ccb3e446563acc13080890f3c897e36":
        (7_059, 181, 6_435, 624, 165, 16, "pga:R2020034", "2020-06-25T00:00:00+00:00", 0.15633333333333332, (0,)),
    "5039b929b7f50dd0d1458855704a44a81366eef05fd92dcebf0d0d80f9d9d2ea":
        (6_670, 172, 6_436, 234, 166, 6, "pga:R2023034", "2023-06-22T00:00:00+00:00", 0.20500000000000002, (2, 3, 4)),
    "7cbb0a044f976a3e2e57309b3abfb41fea7d8eb392b1e3d4ff6cc70b0f01ab98":
        (3_432, 88, 3_432, 0, 88, 0, "pga:R2024034", "2024-06-20T00:00:00+00:00", 0.10033333333333332, (3,)),
    "edff72cd2b781856b6faf5833f1b10dea7f1a362683b3c66764c4aca7c0a37ac":
        (3_159, 81, 3_159, 0, 81, 0, "pga:R2025034", "2025-06-19T00:00:00+00:00", 0.09633333333333333, (3, 4)),
}
ABILITY_DATE_COUNTS = {
    ("pga:R2020034", "2020-06-24T00:00:00+00:00"): 170,
    ("pga:R2020034", None): 11,
    ("pga:R2021034", "2021-06-23T00:00:00+00:00"): 184,
    ("pga:R2021034", None): 10,
    ("pga:R2022034", "2021-09-14T00:00:00+00:00"): 2,
    ("pga:R2022034", "2022-06-22T00:00:00+00:00"): 177,
    ("pga:R2022034", None): 9,
    ("pga:R2023034", "2023-06-21T00:00:00+00:00"): 170,
    ("pga:R2023034", None): 2,
    ("pga:R2024034", "2024-06-19T00:00:00+00:00"): 88,
    ("pga:R2025034", "2025-06-18T00:00:00+00:00"): 81,
}
TOLERANCE = 3e-13


class AuditV2Error(ValueError):
    pass


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _json_digest(value: object, domain: str) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise AuditV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise AuditV2Error(f"non-finite JSON constant: {value}")


def _strict_json(data: bytes, label: str) -> dict[str, object]:
    try:
        result = json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditV2Error(f"{label} is not strict JSON") from exc
    if type(result) is not dict:
        raise AuditV2Error(f"{label} is not an object")
    return result


def _same(observed: object, expected: object, label: str) -> None:
    if type(observed) is not type(expected):
        raise AuditV2Error(f"{label} type changed")
    if type(expected) is dict:
        if set(observed) != set(expected):  # type: ignore[arg-type]
            raise AuditV2Error(f"{label} keys changed")
        for key in expected:
            _same(observed[key], expected[key], f"{label}.{key}")  # type: ignore[index]
    elif type(expected) in (list, tuple):
        if len(observed) != len(expected):  # type: ignore[arg-type]
            raise AuditV2Error(f"{label} length changed")
        for index, item in enumerate(expected):
            _same(observed[index], item, f"{label}[{index}]")  # type: ignore[index]
    elif observed != expected:
        raise AuditV2Error(f"{label} changed")


def _entries(root: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    if not root.is_dir() or root.is_symlink():
        raise AuditV2Error(f"not a plain directory: {root}")
    files: dict[str, Path] = {}
    directories: list[str] = []

    def visit(folder: Path) -> None:
        for entry in sorted(os.scandir(folder), key=lambda item: item.name):
            path = Path(entry.path)
            info = os.lstat(path)
            reparse = bool(
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if entry.is_symlink() or reparse:
                raise AuditV2Error(f"link/reparse entry rejected: {path}")
            relative = path.relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                directories.append(relative)
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                files[relative] = path
            else:
                raise AuditV2Error(f"unsupported entry: {path}")

    visit(root)
    return files, tuple(sorted(directories))


def _capture_pinned(
    root: Path,
    pins: Mapping[str, tuple[int, str]],
    directories: Sequence[str],
    label: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    files, observed_directories = _entries(root)
    if set(files) != set(pins) or observed_directories != tuple(directories):
        raise AuditV2Error(f"{label} closure changed")
    captured: dict[str, bytes] = {}
    inventory: list[dict[str, object]] = []
    for relative in sorted(files):
        raw = files[relative].read_bytes()
        size, expected_sha = pins[relative]
        if len(raw) != size or _digest(raw) != expected_sha:
            raise AuditV2Error(f"{label} byte pin changed: {relative}")
        captured[relative] = raw
        inventory.append({"path": relative, "bytes": size, "sha256": expected_sha})
    state = {
        "files": len(inventory), "directories": len(observed_directories),
        "bytes": sum(item["bytes"] for item in inventory),
        "inventory_sha256": _json_digest(
            {"directories": list(observed_directories), "files": inventory},
            "audit_v2_input_tree",
        ),
    }
    return captured, state


def _source_state() -> dict[str, object]:
    observed = {
        relative: _digest((REPO / PurePosixPath(relative)).read_bytes())
        for relative in SOURCE_PINS
    }
    _same(observed, SOURCE_PINS, "bound source pins")
    v1_sha = _digest(
        (
            REPO
            / "archive"
            / "audit_hole_sim_real_app_h8_selector_authority_candidate.py"
        ).read_bytes()
    )
    _same(v1_sha, V1_DIAGNOSTIC_PIN, "preserved v1 diagnostic")
    test_sha = _digest(AUDITOR_TEST.read_bytes())
    _same(test_sha, AUDITOR_TEST_SHA256, "auditor focused test pin")
    return {
        "auditor_source_sha256": _digest(AUDITOR_SOURCE.read_bytes()),
        "auditor_test_sha256": test_sha,
        "bound_source_files": len(observed),
        "bound_sources_sha256": _json_digest(observed, "audit_v2_bound_sources"),
        "v1_diagnostic_sha256": v1_sha,
    }


def _launcher_state(attestation: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "contract", "launcher_relative_path", "launcher_test_relative_path",
        "launcher_sha256", "launcher_test_sha256",
    }
    if set(attestation) != expected_fields:
        raise AuditV2Error("launcher attestation fields changed")
    _same(
        attestation["contract"], "fixed_one_read_launcher_attestation_v2",
        "launcher contract",
    )
    _same(
        attestation["launcher_relative_path"],
        "archive/run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py",
        "launcher path",
    )
    _same(
        attestation["launcher_test_relative_path"],
        "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py",
        "launcher test path",
    )
    observed = {
        "launcher_sha256": _digest(LAUNCHER_SOURCE.read_bytes()),
        "launcher_test_sha256": _digest(LAUNCHER_TEST.read_bytes()),
    }
    for field, value in observed.items():
        supplied = attestation[field]
        if (
            type(supplied) is not str or len(supplied) != 64
            or any(char not in "0123456789abcdef" for char in supplied)
            or supplied != value
        ):
            raise AuditV2Error(f"{field} changed")
    return dict(attestation)


def _validate_candidate(captured: Mapping[str, bytes]) -> dict[str, object]:
    candidate = _strict_json(captured["authority_candidate.json"], "candidate")
    fields = {
        "candidate_id", "contract", "admission_status", "authority_id",
        "authority_lock_sha256", *FAIL_CLOSED, "research_only",
        "fixed_input_identity", "authority_evidence_sha256",
        "source_bindings_sha256", "fixed_caps", "artifact_sha256_start",
        "artifact_sha256_end", "validation", "source_sha256_start",
        "source_sha256_end", "external_review_requirements",
    }
    if set(candidate) != fields:
        raise AuditV2Error("candidate fields changed")
    for field, expected in {
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "contract": "fixed_lowry_r1h8_real_app_selector_support_candidate_v1",
        "admission_status": "candidate_external_source_and_artifact_review_required",
        "authority_id": None, "authority_lock_sha256": None,
        "research_only": True,
    }.items():
        _same(candidate[field], expected, f"candidate {field}")
    for field, expected in FAIL_CLOSED.items():
        _same(candidate[field], expected, f"candidate {field}")
    _same(
        candidate["fixed_input_identity"],
        {
            "outer_output_index_sha256": ORIGINAL_FILES["OUTPUT_SHA256.json"][1],
            "outer_manifest_sha256": ORIGINAL_FILES["manifest.json"][1],
            "inner_output_index_sha256": ORIGINAL_FILES[
                "complete_atom_support/OUTPUT_SHA256.json"
            ][1],
            "inner_manifest_sha256": ORIGINAL_FILES[
                "complete_atom_support/manifest.json"
            ][1],
        },
        "candidate input identity",
    )
    _same(
        candidate["fixed_caps"],
        {
            "parts": 5, "rows_per_part": 8_192, "complete_atoms": 35_218,
            "proposals": 904, "surface_classes": 9,
            "hazards_per_surface": 34, "complete_atoms_per_proposal": 16_384,
        },
        "candidate caps",
    )
    _same(
        candidate["external_review_requirements"],
        {
            "independent_source_review_required": True,
            "independent_artifact_review_required": True,
            "candidate_id_is_not_authority": True,
            "separate_data_only_authority_record_required": True,
        },
        "candidate review requirements",
    )
    _same(
        candidate["authority_evidence_sha256"],
        "77111d48c92e57498dbcbc66286c1f3f235b35bb9377495eb01637a5a23f9e11",
        "candidate authority evidence",
    )
    _same(
        candidate["source_bindings_sha256"],
        "26dfd00bc0d42921c7a383f9df055af05760d8dd026faa54529e0b63e90ed21e",
        "candidate source bindings",
    )
    expected_artifact_state = {
        "bytes": 7_647_733,
        "closure_sha256": "5580dd21763ad5e4eac39057ef7264a4d748313a55033dc8e950ce97bd522f60",
        "directories": 2,
        "directory_state_sha256": "47adb231640339bd8b7bb8d593d3d243ef0f0bc0c832fe232c020caa9478b645",
        "file_state_sha256": "93f52860c532288419b4bb6bb57a78082aa89bb25d2a69eedfe2101d171e17c6",
        "files": 10,
    }
    expected_source_state = {
        "bound_source_files": 16,
        "bound_sources_sha256": "26dfd00bc0d42921c7a383f9df055af05760d8dd026faa54529e0b63e90ed21e",
        "builder_source_sha256": "b3fdbd12f2e466bdac92f71f2d60a3e9fd0aec18a58f309248dc1c783046fdd7",
        "source_closure_sha256": "e1783adaa9a03552e72d6e97d29cf97a5d3c720cc0dd22dab9630378ccf28dcc",
    }
    _same(candidate["artifact_sha256_start"], expected_artifact_state, "candidate artifact start")
    _same(candidate["artifact_sha256_end"], expected_artifact_state, "candidate artifact end")
    _same(candidate["source_sha256_start"], expected_source_state, "candidate source start")
    _same(candidate["source_sha256_end"], expected_source_state, "candidate source end")
    validation = candidate["validation"]
    if type(validation) is not dict:
        raise AuditV2Error("candidate validation changed")
    evidence_fields = {
        "canonical_full_schema_sha256": "full_schema_sha256",
        "canonical_full_support_sha256": "full_support_sha256",
        "canonical_projection_schema_sha256": "projection_schema_sha256",
        "canonical_projection_sha256": "projection_54_sha256",
        "canonical_selector_support_sha256": "selector_native_sha256",
        "canonical_topology_sha256": "topology_30_sha256",
    }
    for candidate_field, external_field in evidence_fields.items():
        _same(
            validation.get(candidate_field),
            EXTERNAL_CANONICAL_EVIDENCE[external_field],
            f"external canonical evidence {candidate_field}",
        )
    _same(
        validation.get("counts"),
        {
            "complete_atoms": 35_218, "donor_quality_groups": 6,
            "hazard_atoms": 30_702, "imputed_ability_atoms": 1_989,
            "materialized_surface_branches": 5_418, "mixed_policy_groups": 4,
            "proposals": 904, "spatial_proposals": 903,
            "surface_records": 903, "terminal_atoms": 1,
        },
        "candidate counts",
    )
    expected_validation_tail = {
        "cross_part_group_counts_sha256": "169d3dbad803cf78cead00ce48e2341fc4be551f9a7ff55b20fb1d4116efe2cd",
        "cross_part_proposals_sha256": "ad429cc5a44d72cdf7ce711e5b9b5299043e3ba9ce049de5b031f12c1d98e623",
        "full_schema_columns": 150,
        "full_schema_sha256": "90f5176de0cbf1c52118a94a47ba35299bcdf9755e77c6350e88bc0342c65855",
        "group_policy_topology_sha256": "b13eeb67a48bffbbc2cc8c728b3767f86bb032c85d26b873307f424bdbb02985",
        "inner_output_index_sha256": ORIGINAL_FILES[
            "complete_atom_support/OUTPUT_SHA256.json"
        ][1],
        "outer_closure_sha256": expected_artifact_state["closure_sha256"],
        "outer_output_index_sha256": ORIGINAL_FILES["OUTPUT_SHA256.json"][1],
        "semantic_projection_columns": 54,
        "semantic_projection_schema_sha256": "83b0b77a1894d6949d9236ca40bafe9fd9cb779d975dfb511933a82f7047d2a0",
        "semantic_projection_sha256": "1fa34f66cc3925ebc7d8ab12f6c8039629dfa3671f9369903a85ee1cf2217a4b",
        "topology_sha256": "7a7f6847c029dd1382084e1bcfc43c8454c1a09e93572a426e1bc19c237df9e2",
    }
    expected_validation_fields = {
        *evidence_fields, "counts", *expected_validation_tail,
    }
    if set(validation) != expected_validation_fields:
        raise AuditV2Error("candidate validation fields changed")
    for field, expected in expected_validation_tail.items():
        _same(validation[field], expected, f"candidate validation {field}")
    expected_index = {
        "contract": INDEX_CONTRACT,
        "files": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in CANDIDATE_FILES.items()
            if name != "OUTPUT_SHA256.json"
        },
    }
    _same(
        _strict_json(captured["OUTPUT_SHA256.json"], "candidate index"),
        expected_index,
        "candidate index",
    )
    manifest = _strict_json(captured["manifest.json"], "candidate manifest")
    _same(
        manifest,
        {
            "contract": "fixed_lowry_r1h8_real_app_selector_candidate_manifest_v1",
            "candidate_id": EXPECTED_CANDIDATE_ID,
            "admission_status": "candidate_external_source_and_artifact_review_required",
            **FAIL_CLOSED,
            "research_only": True,
            "payload_files": {
                "REPORT.md": {
                    "bytes": CANDIDATE_FILES["REPORT.md"][0],
                    "sha256": CANDIDATE_FILES["REPORT.md"][1],
                },
                "authority_candidate.json": {
                    "bytes": CANDIDATE_FILES["authority_candidate.json"][0],
                    "sha256": CANDIDATE_FILES["authority_candidate.json"][1],
                },
            },
        },
        "candidate manifest",
    )
    return candidate


def _validate_original(captured: Mapping[str, bytes]) -> list[bytes]:
    outer_index = _strict_json(captured["OUTPUT_SHA256.json"], "outer index")
    _same(
        outer_index,
        {
            "contract": "fixed_lowry_r1h8_outer_output_sha256_v1",
            "directories": ["complete_atom_support", "complete_atom_support/parts"],
            "files": {
                path: {"bytes": size, "sha256": digest}
                for path, (size, digest) in ORIGINAL_FILES.items()
                if path != "OUTPUT_SHA256.json"
            },
        },
        "outer output index",
    )
    inner_index = _strict_json(
        captured["complete_atom_support/OUTPUT_SHA256.json"], "inner index"
    )
    inner_files = {
        "manifest.json": {
            "bytes": ORIGINAL_FILES["complete_atom_support/manifest.json"][0],
            "sha256": ORIGINAL_FILES["complete_atom_support/manifest.json"][1],
        },
        **{
            path.removeprefix("complete_atom_support/"): {
                "bytes": size, "sha256": byte_sha,
            }
            for path, _rows, size, byte_sha, _logical in PARTS
        },
    }
    _same(
        inner_index,
        {"contract": INDEX_CONTRACT, "files": inner_files},
        "inner output index",
    )
    outer = _strict_json(captured["manifest.json"], "outer manifest")
    for field, expected in {
        "contract": "fixed_lowry_r1h8_complete_atom_materialization_v1",
        "complete_atoms": 35_218, "terminal_atoms": 1,
        "imputed_ability_atoms": 1_989, "query_key": EXPECTED_QUERY_KEY,
        "support_key": EXPECTED_SUPPORT_KEY, "research_only": True,
        "production_enabled": False, "simulation_launched": False,
        "mean_solve_launched": False, "anchor_ready": False,
    }.items():
        _same(outer.get(field), expected, f"outer {field}")
    inner = _strict_json(
        captured["complete_atom_support/manifest.json"], "inner manifest"
    )
    for field, expected in {
        "contract": "bounded_sharded_complete_atom_persistence_v4",
        "complete_atom_contract": "complete_tee_app_physical_atom_support_v3",
        "complete_atoms": 35_218, "expansion_batch_rows": 8_192,
        "max_rows_per_part": 8_192,
        "max_complete_atoms_per_proposal": 16_384,
        "peak_buffered_rows": 8_192, "research_only": True,
        "production_enabled": False,
    }.items():
        _same(inner.get(field), expected, f"inner {field}")
    raw_parts = inner.get("parts")
    if type(raw_parts) is not list or len(raw_parts) != 5:
        raise AuditV2Error("part inventory changed")
    buffers: list[bytes] = []
    for entry, expected in zip(raw_parts, PARTS):
        path, rows, size, byte_sha, logical_sha = expected
        inner_path = path.removeprefix("complete_atom_support/")
        _same(
            entry,
            {
                "byte_sha256": byte_sha, "bytes": size,
                "logical_sha256": logical_sha, "path": inner_path, "rows": rows,
            },
            f"part inventory {path}",
        )
        buffers.append(captured[path])
    return buffers


def _schema_tuple(schema: pa.Schema) -> tuple[tuple[str, str, bool], ...]:
    return tuple((field.name, str(field.type), field.nullable) for field in schema)


def _close(left: object, right: object, label: str) -> None:
    observed = np.asarray(left, dtype=np.float64)
    expected = np.asarray(right, dtype=np.float64)
    if observed.shape != expected.shape or not np.allclose(
        observed, expected, rtol=0.0, atol=TOLERANCE, equal_nan=False
    ):
        raise AuditV2Error(f"{label} changed")


def _cdf_chain(
    lower: np.ndarray,
    upper: np.ndarray,
    mass: np.ndarray,
    label: str,
) -> None:
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise AuditV2Error(f"{label} has non-finite bounds")
    _close(lower[:1], [0.0], f"{label} start")
    _close(upper[-1:], [1.0], f"{label} end")
    _close(lower[1:], upper[:-1], f"{label} continuity")
    _close(upper - lower, mass, f"{label} widths")


def _load_from_captures(
    part_buffers: Sequence[bytes],
) -> tuple[pd.DataFrame, pa.Schema]:
    expected_schema = tuple(
        (name, kind, True)
        for name, kind in (
            tuple(item.split("@", 1) for item in EXPECTED_SCHEMA.split(";"))
        )
    )
    tables: list[pa.Table] = []
    common_schema: pa.Schema | None = None
    for fragment_index, (raw, part) in enumerate(zip(part_buffers, PARTS)):
        parquet = pq.ParquetFile(pa.BufferReader(raw))
        if parquet.metadata.num_rows != part[1]:
            raise AuditV2Error(f"part {fragment_index} row metadata changed")
        if _schema_tuple(parquet.schema_arrow) != expected_schema:
            raise AuditV2Error(f"part {fragment_index} Arrow schema/order changed")
        if common_schema is None:
            common_schema = parquet.schema_arrow
        elif not parquet.schema_arrow.equals(common_schema, check_metadata=True):
            raise AuditV2Error("part schemas differ")
        table = parquet.read(columns=list(READ_COLUMNS), use_threads=False)
        table = table.append_column(
            "__fragment_index",
            pa.array(
                np.full(table.num_rows, fragment_index, dtype=np.int32),
                type=pa.int32(),
            ),
        )
        tables.append(table)
    if common_schema is None:
        raise AuditV2Error("no Parquet schema")
    combined = pa.concat_tables(tables, promote_options="none")
    frame = combined.to_pandas()
    if tuple(frame.columns) != READ_COLUMNS + ("__fragment_index",):
        raise AuditV2Error("vector read order changed")
    return frame, common_schema


def _group_inventory(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    representatives = frame.drop_duplicates("proposal_row_order")
    atom_groups = frame.groupby("donor_quality_group_id", sort=True)
    proposal_groups = representatives.groupby("donor_quality_group_id", sort=True)
    if set(atom_groups.groups) != set(GROUP_EXPECTATIONS):
        raise AuditV2Error("donor group IDs changed")
    inventory: dict[str, dict[str, object]] = {}
    for group_id, expected in GROUP_EXPECTATIONS.items():
        atoms = atom_groups.get_group(group_id)
        proposals = proposal_groups.get_group(group_id)
        population = (
            len(atoms), len(proposals),
            int(atoms["donor_ability_available"].sum()),
            int((~atoms["donor_ability_available"]).sum()),
            int(proposals["donor_ability_available"].sum()),
            int((~proposals["donor_ability_available"]).sum()),
        )
        if population != expected[:6]:
            raise AuditV2Error(f"group population changed: {group_id}")
        _close(
            [proposals["proposal_probability"].sum()], [expected[8]],
            f"group proposal mass {group_id}",
        )
        if (
            set(atoms["donor_event_key"]) != {expected[6]}
            or set(value.isoformat() for value in atoms["donor_event_start"])
            != {expected[7]}
            or tuple(sorted(atoms["__fragment_index"].unique())) != expected[9]
        ):
            raise AuditV2Error(f"group event/date/part tuple changed: {group_id}")
        policy_states = (
            {"observed_strict_prior"}
            if expected[5] == 0
            else {"observed_strict_prior", "imputed_frozen_neutral_zero"}
        )
        if set(atoms["donor_ability_value_kind"]) != policy_states:
            raise AuditV2Error(f"group policy states changed: {group_id}")
        inventory[group_id] = {
            "atoms": population[0], "proposals": population[1],
            "observed_atoms": population[2],
            "imputed_atoms": population[3],
            "observed_proposals": population[4],
            "imputed_proposals": population[5],
            "event_key": expected[6], "event_start": expected[7],
            "proposal_mass": format(expected[8], ".17g"),
            "parts": list(expected[9]), "policy_states": sorted(policy_states),
        }
    return inventory


def _float_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        raise AuditV2Error("non-finite binary64 in NDJSON record")
    return number.hex()


def _time_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise AuditV2Error("naive timestamp in NDJSON record")
    return timestamp.tz_convert("UTC").isoformat()


def _text_or_none(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _ndjson(
    contract: str,
    records: Sequence[Mapping[str, object]] | object,
) -> dict[str, object]:
    digest = sha256()
    contract_line = contract.encode("ascii") + b"\n"
    digest.update(contract_line)
    framed_bytes = len(contract_line)
    record_count = 0
    for record in records:  # type: ignore[union-attr]
        line = (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        digest.update(line)
        framed_bytes += len(line)
        record_count += 1
    return {
        "contract": contract, "sha256": digest.hexdigest(),
        "records": record_count, "framed_bytes": framed_bytes,
    }


def _schema_records(schema: pa.Schema) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "kind": "inventory", "part_rows": [8_192, 8_192, 8_192, 8_192, 2_450],
            "parts": 5, "rows": 35_218,
        }
    ]
    records.extend(
        {
            "kind": "field", "ordinal": ordinal, "name": field.name,
            "arrow_type": str(field.type), "nullable": bool(field.nullable),
        }
        for ordinal, field in enumerate(schema)
    )
    return records


def _topology_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "kind": "inventory", "atoms": 35_218, "proposals": 904,
            "spatial_proposals": 903, "terminal_atoms": 1,
            "materialized_surface_branches": 5_418,
            "resolved_hazard_atoms": 30_702,
        }
    ]
    for proposal, rows in frame.groupby("proposal_row_order", sort=False):
        first_surface = rows.drop_duplicates("surface_class_index")
        surface_counts = rows.groupby("surface_class_index", sort=False).size()
        surface_runs = [
            {
                "index": int(item.surface_class_index),
                "label": str(item.surface_class_label),
                "rows": int(surface_counts.loc[item.surface_class_index]),
            }
            for item in first_surface.itertuples(index=False)
        ]
        hazard_orders = rows.loc[
            rows["hazard_resolution_status"] == "resolved_complete_atom",
            "hazard_donor_order",
        ].astype(int).tolist()
        records.append(
            {
                "kind": "proposal", "proposal": int(proposal),
                "first_complete": int(rows["complete_row_order"].iloc[0]),
                "last_complete": int(rows["complete_row_order"].iloc[-1]),
                "rows": len(rows), "branch": str(rows["proposal_branch"].iloc[0]),
                "terminal_rows": int(rows["terminal"].sum()),
                "proposal_atom_id": str(rows["proposal_atom_id"].iloc[0]),
                "bridge_row_id": str(rows["bridge_row_id"].iloc[0]),
                "surface_runs": surface_runs,
                "resolved_hazard_orders": hazard_orders,
                "atom_ids": rows["atom_id"].astype(str).tolist(),
            }
        )
    return records


def _mass_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "kind": "inventory", "absolute_tolerance": float(TOLERANCE).hex(),
            "atoms": 35_218, "proposals": 904,
            "quality_mass": float(frame["quality_base_mass"].sum()).hex(),
        }
    ]
    for proposal, rows in frame.groupby("proposal_row_order", sort=False):
        one = rows.iloc[0]
        first_surface = rows.drop_duplicates("surface_class_index")
        surface_counts = rows.groupby("surface_class_index", sort=False).size()
        surface_records = [
            {
                "index": int(item.surface_class_index),
                "probability": _float_text(item.surface_probability),
                "lower": _float_text(item.surface_cdf_lower),
                "upper": _float_text(item.surface_cdf_upper),
                "rows": int(surface_counts.loc[item.surface_class_index]),
            }
            for item in first_surface.itertuples(index=False)
        ]
        full_probability = json.loads(
            one["surface_full_probability_json"], parse_constant=_reject_constant
        )
        full_cdf = json.loads(
            one["surface_full_cdf_upper_json"], parse_constant=_reject_constant
        )
        array_fields = {
            "surface_probability": "surface_probability",
            "surface_lower": "surface_cdf_lower",
            "surface_upper": "surface_cdf_upper",
            "hazard_probability": "hazard_probability",
            "hazard_lower": "hazard_cdf_lower",
            "hazard_upper": "hazard_cdf_upper",
            "mechanics_probability": "mechanics_conditional_probability",
            "mechanics_lower": "mechanics_cdf_lower",
            "mechanics_upper": "mechanics_cdf_upper",
            "quality_mass": "quality_base_mass",
            "base_hierarchy_mass": "base_hierarchy_mass",
            "spatial_model_weight": "spatial_model_weight",
            "complete_lower": "complete_cdf_lower",
            "complete_upper": "complete_cdf_upper",
        }
        record: dict[str, object] = {
            "kind": "proposal", "proposal": int(proposal),
            "proposal_probability": _float_text(one["proposal_probability"]),
            "proposal_base_hierarchy_mass": _float_text(
                one["proposal_base_hierarchy_mass"]
            ),
            "proposal_spatial_model_weight": _float_text(
                one["proposal_spatial_model_weight"]
            ),
            "proposal_cdf": [
                _float_text(one["proposal_cdf_lower"]),
                _float_text(one["proposal_cdf_upper"]),
            ],
            "branch_probability": _float_text(
                one["proposal_branch_conditional_probability"]
            ),
            "branch_cdf": [
                _float_text(one["proposal_branch_cdf_lower"]),
                _float_text(one["proposal_branch_cdf_upper"]),
            ],
            "frozen_spatial_cdf": [
                _float_text(one["frozen_spatial_cdf_lower"]),
                _float_text(one["frozen_spatial_cdf_upper"]),
            ],
            "frozen_terminal_cdf": [
                _float_text(one["frozen_terminal_residual_cdf_lower"]),
                _float_text(one["frozen_terminal_residual_cdf_upper"]),
            ],
            "full_surface_probability": [
                _float_text(value) for value in full_probability
            ],
            "full_surface_cdf": [_float_text(value) for value in full_cdf],
            "surface_records": surface_records,
        }
        record.update(
            {
                output_name: [_float_text(value) for value in rows[column]]
                for output_name, column in array_fields.items()
            }
        )
        records.append(record)
    return records


def _group_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    representatives = frame.drop_duplicates("proposal_row_order")
    records: list[dict[str, object]] = [
        {
            "kind": "inventory", "groups": 6, "mixed_groups": 4,
            "imputed_rows": 1_989, "observed_rows": 33_229,
            "imputed_proposals": 51, "observed_proposals": 853,
        }
    ]
    for group_id, rows in frame.groupby("donor_quality_group_id", sort=True):
        proposal_rows = representatives.loc[
            representatives["donor_quality_group_id"] == group_id
        ]
        kinds = sorted(rows["donor_ability_value_kind"].unique().tolist())
        counts = rows["donor_ability_value_kind"].value_counts()
        records.append(
            {
                "kind": "group", "group_id": str(group_id), "rows": len(rows),
                "proposals": len(proposal_rows),
                "available_rows": int(rows["donor_ability_available"].sum()),
                "unavailable_rows": int((~rows["donor_ability_available"]).sum()),
                "value_kinds": kinds,
                "rows_by_kind": {kind: int(counts.loc[kind]) for kind in kinds},
                "mixed": len(kinds) == 2,
                "proposal_orders": proposal_rows["proposal_row_order"].astype(int).tolist(),
            }
        )
    return records


def _ability_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "kind": "inventory", "policy_sha256": POLICY_IDENTITY,
            "proposals": 904, "observed_proposals": 853,
            "imputed_proposals": 51, "source_column": "sg_app",
        }
    ]
    for proposal, rows in frame.groupby("proposal_row_order", sort=False):
        one = rows.iloc[0]
        records.append(
            {
                "kind": "proposal", "proposal": int(proposal),
                "group_id": str(one["donor_quality_group_id"]),
                "bridge_row_id": str(one["bridge_row_id"]),
                "category": str(one["category"]),
                "donor_player_id": str(one["donor_player_id"]),
                "donor_player_name": str(one["donor_player_name"]),
                "donor_event_key": str(one["donor_event_key"]),
                "donor_event_start": _time_text(one["donor_event_start"]),
                "target_event_key": str(one["target_event_key"]),
                "target_event_start": _time_text(one["target_event_start"]),
                "donor_skill_date": _time_text(one["donor_skill_date"]),
                "source_column": str(one["donor_ability_source_column"]),
                "available": bool(one["donor_ability_available"]),
                "raw_strict_prior": _float_text(
                    one["donor_ability_raw_strict_prior"]
                ),
                "value_kind": str(one["donor_ability_value_kind"]),
                "applied_sg": _float_text(one["donor_ability_sg"]),
                "was_clipped": bool(one["donor_ability_was_clipped"]),
                "policy_sha256": str(one["donor_ability_policy_sha256"]),
                "row_sha256": str(one["donor_ability_row_sha256"]),
                "snapshot_kind": str(one["donor_ability_snapshot_kind"]),
                "proposal_policy_binding_sha256": str(
                    one["proposal_policy_binding_sha256"]
                ),
            }
        )
    return records


def _reference_records(frame: pd.DataFrame) -> object:
    yield {
        "kind": "inventory", "atoms": 35_218,
        "reference_model_sha256": REFERENCE_MODEL,
        "reference_putting_sha256": PUTTING_MODEL,
        "reference_descriptor":
            f"frozen_tour_expected_strokes_reference_v1:{REFERENCE_MODEL}",
        "reference_version": "broadie_2020_2025_plus_2020_prior_putting_v1",
    }
    columns = (
        "complete_row_order", "atom_id", "start_reference_value",
        "causal_strokes", "penalty_strokes", "stroke_cost",
        "end_reference_value", "fixed_reference_reward",
        "quality_reward_available", "terminal", "holed", "end_state_key",
        "next_category", "next_support_key",
    )
    for values in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        (
            row, atom_id, start_reference, causal_strokes, penalty_strokes,
            stroke_cost, end_reference, fixed_reward, reward_available,
            terminal, holed, end_state_key, next_category, next_support_key,
        ) = values
        yield {
            "kind": "atom", "row": int(row), "atom_id": str(atom_id),
            "start_reference": _float_text(start_reference),
            "causal_strokes": int(causal_strokes),
            "penalty_strokes": int(penalty_strokes),
            "stroke_cost": int(stroke_cost),
            "end_reference": _float_text(end_reference),
            "fixed_reward": _float_text(fixed_reward),
            "reward_available": bool(reward_available),
            "terminal": bool(terminal), "holed": bool(holed),
            "end_state_key": str(end_state_key),
            "next_category": _text_or_none(next_category),
            "next_support_key": str(next_support_key),
        }


def _file_inventory(pins: Mapping[str, tuple[int, str]]) -> list[dict[str, object]]:
    return [
        {"path": path, "bytes": pins[path][0], "sha256": pins[path][1]}
        for path in sorted(pins)
    ]


def _ndjson_components(
    frame: pd.DataFrame,
    schema: pa.Schema,
) -> dict[str, dict[str, object]]:
    components = {
        "schema": _ndjson(
            EXPECTED_NDJSON["schema"]["contract"], _schema_records(schema)
        ),
        "topology": _ndjson(
            EXPECTED_NDJSON["topology"]["contract"], _topology_records(frame)
        ),
        "mass_cdf": _ndjson(
            EXPECTED_NDJSON["mass_cdf"]["contract"], _mass_records(frame)
        ),
        "group": _ndjson(
            EXPECTED_NDJSON["group"]["contract"], _group_records(frame)
        ),
        "ability_policy": _ndjson(
            EXPECTED_NDJSON["ability_policy"]["contract"],
            _ability_records(frame),
        ),
        "reference_reward": _ndjson(
            EXPECTED_NDJSON["reference_reward"]["contract"],
            _reference_records(frame),
        ),
    }
    for name, result in components.items():
        _same(result, EXPECTED_NDJSON[name], f"{name} NDJSON evidence")
    counts = {
        "atoms": 35_218, "proposals": 904, "spatial_proposals": 903,
        "terminal_atoms": 1, "materialized_surface_branches": 5_418,
        "resolved_hazard_atoms": 30_702, "groups": 6, "mixed_groups": 4,
        "imputed_rows": 1_989, "observed_rows": 33_229,
        "imputed_proposals": 51, "observed_proposals": 853,
    }
    root_record = {
        "kind": "semantic_root", "candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_files": _file_inventory(CANDIDATE_FILES),
        "original_files": _file_inventory(ORIGINAL_FILES), "counts": counts,
        "components": components,
    }
    root = _ndjson(
        EXPECTED_NDJSON["semantic_root"]["contract"], [root_record]
    )
    _same(root, EXPECTED_NDJSON["semantic_root"], "semantic-root NDJSON evidence")
    return {**components, "semantic_root": root}


def _vector_checks(frame: pd.DataFrame, schema: pa.Schema) -> dict[str, object]:
    if len(frame) != 35_218 or len(schema) != 150:
        raise AuditV2Error("complete row/schema count changed")
    complete_order = frame["complete_row_order"].to_numpy(np.int64)
    proposal_order = frame["proposal_row_order"].to_numpy(np.int64)
    if not np.array_equal(complete_order, np.arange(35_218, dtype=np.int64)):
        raise AuditV2Error("complete row order changed")
    if frame["atom_id"].isna().any() or frame["atom_id"].nunique() != 35_218:
        raise AuditV2Error("atom IDs changed")
    fixed_values = {
        "query_key": EXPECTED_QUERY_KEY, "category": "app",
        "complete_atom": True, "boundary_only": False,
        "declared_app_base_holeout_probability": 0.002,
        "start_state_key": "tee|204.541715yd",
    }
    for column, expected in fixed_values.items():
        if set(frame[column].tolist()) != {expected}:
            raise AuditV2Error(f"fixed row identity changed: {column}")
    if (
        frame["atom_kind"].value_counts().to_dict()
        != {
            "app_spatial_hazard_resolved": 30_702,
            "app_spatial_surface": 4_515,
            "app_holeout": 1,
        }
        or set(frame["causal_strokes"]) != {1}
        or set(frame["quality_reward_available"]) != {True}
        or not np.array_equal(
            frame["holed"].to_numpy(bool), frame["terminal"].to_numpy(bool)
        )
    ):
        raise AuditV2Error("atom/reward state population changed")

    proposal_sizes = np.bincount(proposal_order, minlength=904)
    if (
        len(proposal_sizes) != 904
        or not np.array_equal(proposal_sizes[:903], np.full(903, 39))
        or proposal_sizes[903] != 1
    ):
        raise AuditV2Error("39-atom spatial topology changed")
    representatives = frame.drop_duplicates("proposal_row_order")
    if not np.array_equal(
        representatives["proposal_row_order"].to_numpy(np.int64),
        np.arange(904, dtype=np.int64),
    ):
        raise AuditV2Error("proposal ordering changed")
    constant_by_proposal = (
        "proposal_probability", "proposal_cdf_lower", "proposal_cdf_upper",
        "proposal_branch", "proposal_branch_conditional_probability",
        "proposal_branch_cdf_lower", "proposal_branch_cdf_upper",
        "proposal_base_hierarchy_mass", "donor_quality_group_id", "bridge_row_id",
        "category", "donor_player_id", "donor_player_name", "donor_event_key",
        "donor_event_start", "target_event_key", "target_event_start",
        "donor_skill_date", "donor_ability_source_column",
        "donor_ability_available", "donor_ability_raw_strict_prior",
        "donor_ability_value_kind", "donor_ability_sg",
        "donor_ability_was_clipped", "donor_ability_policy_sha256",
        "donor_ability_row_sha256", "donor_ability_snapshot_kind",
        "proposal_policy_binding_sha256",
    )
    grouped = frame.groupby("proposal_row_order")
    for column in constant_by_proposal:
        if grouped[column].nunique(dropna=False).max() != 1:
            raise AuditV2Error(f"proposal split: {column}")

    terminal = frame.loc[frame["terminal"]]
    if len(terminal) != 1:
        raise AuditV2Error("terminal population changed")
    terminal_row = terminal.iloc[0]
    terminal_expected = {
        "complete_row_order": 35_217, "proposal_row_order": 903,
        "atom_id": "62118466441c40a23026610a15dcdb7106931319fe5409fd930ee89dd716666a",
        "proposal_branch": "terminal", "proposal_probability": 0.002,
        "mechanics_conditional_probability": 1.0, "quality_base_mass": 0.002,
        "surface_class_count": 0, "surface_class_index": -1,
        "surface_class_label": "explicit_app_holeout",
        "hazard_resolution_status": "none", "hazard_donor_order": -1,
        "hazard_donor_key": "not_applicable", "stroke_cost": 1,
        "end_state_key": "terminal", "next_support_key": "",
        "donor_player_id": "28237", "donor_event_key": "pga:R2023034",
    }
    for column, expected in terminal_expected.items():
        if terminal_row[column] != expected:
            raise AuditV2Error(f"terminal sentinel changed: {column}")
    _close(
        [terminal_row["fixed_reference_reward"]], [2.131354288195138],
        "terminal reward",
    )
    terminal_cdf = {
        "proposal_cdf_lower": 0.9979999999999989,
        "proposal_cdf_upper": 1.0,
        "proposal_branch_conditional_probability": 1.0,
        "proposal_branch_cdf_lower": 0.0,
        "proposal_branch_cdf_upper": 1.0,
        "mechanics_cdf_lower": 0.0, "mechanics_cdf_upper": 1.0,
        "surface_probability": 1.0, "surface_cdf_lower": 0.0,
        "surface_cdf_upper": 1.0, "hazard_probability": 1.0,
        "hazard_cdf_lower": 0.0, "hazard_cdf_upper": 1.0,
    }
    for column, expected in terminal_cdf.items():
        _close([terminal_row[column]], [expected], f"terminal {column}")
    if (
        terminal_row["surface_full_probability_json"] != "[]"
        or terminal_row["surface_full_cdf_upper_json"] != "[]"
    ):
        raise AuditV2Error("terminal full-surface sentinels changed")

    spatial = frame.loc[~frame["terminal"]]
    if len(spatial) != 35_217 or set(spatial["proposal_branch"]) != {"spatial"}:
        raise AuditV2Error("spatial population changed")
    if set(spatial["surface_class_count"]) != {9}:
        raise AuditV2Error("surface class count changed")
    spatial_representatives = spatial.drop_duplicates("proposal_row_order")
    if len(spatial_representatives) != 903:
        raise AuditV2Error("spatial proposal count changed")

    proposal_mass = representatives["proposal_probability"].to_numpy(float)
    _cdf_chain(
        representatives["proposal_cdf_lower"].to_numpy(float),
        representatives["proposal_cdf_upper"].to_numpy(float),
        proposal_mass, "proposal CDF",
    )
    _close(
        representatives["proposal_base_hierarchy_mass"].to_numpy(float),
        proposal_mass, "proposal base mass",
    )
    _close([proposal_mass.sum()], [1.0], "proposal mass")
    spatial_mass = spatial_representatives["proposal_probability"].to_numpy(float)
    _close([spatial_mass.sum()], [0.998], "spatial proposal mass")
    branch_probability = spatial_representatives[
        "proposal_branch_conditional_probability"
    ].to_numpy(float)
    _cdf_chain(
        spatial_representatives["proposal_branch_cdf_lower"].to_numpy(float),
        spatial_representatives["proposal_branch_cdf_upper"].to_numpy(float),
        branch_probability, "spatial proposal-branch CDF",
    )
    _close(branch_probability, spatial_mass / 0.998, "spatial branch factor")
    _close(
        spatial_representatives["frozen_spatial_cdf_lower"].to_numpy(float),
        spatial_representatives["proposal_branch_cdf_lower"].to_numpy(float),
        "frozen spatial lower",
    )
    _close(
        spatial_representatives["frozen_spatial_cdf_upper"].to_numpy(float),
        spatial_representatives["proposal_branch_cdf_upper"].to_numpy(float),
        "frozen spatial upper",
    )
    if not (
        np.isnan(terminal_row["frozen_spatial_cdf_lower"])
        and np.isnan(terminal_row["frozen_spatial_cdf_upper"])
    ):
        raise AuditV2Error("terminal frozen-spatial sentinels changed")
    _close(
        [terminal_row["frozen_terminal_residual_cdf_lower"]], [0.0],
        "terminal residual lower",
    )
    _close(
        [terminal_row["frozen_terminal_residual_cdf_upper"]], [1.0],
        "terminal residual upper",
    )
    if spatial[
        ["frozen_terminal_residual_cdf_lower", "frozen_terminal_residual_cdf_upper"]
    ].notna().any(axis=None):
        raise AuditV2Error("spatial terminal-residual sentinels changed")

    mechanics_lower = frame["mechanics_cdf_lower"].to_numpy(float)
    mechanics_upper = frame["mechanics_cdf_upper"].to_numpy(float)
    mechanics_mass = frame["mechanics_conditional_probability"].to_numpy(float)
    first = np.r_[True, proposal_order[1:] != proposal_order[:-1]]
    last = np.r_[proposal_order[1:] != proposal_order[:-1], True]
    _close(mechanics_lower[first], np.zeros(904), "mechanics CDF starts")
    _close(mechanics_upper[last], np.ones(904), "mechanics CDF ends")
    same_proposal = proposal_order[1:] == proposal_order[:-1]
    _close(
        mechanics_lower[1:][same_proposal],
        mechanics_upper[:-1][same_proposal],
        "mechanics CDF continuity",
    )
    _close(
        mechanics_upper - mechanics_lower, mechanics_mass,
        "mechanics CDF widths",
    )

    surface_branches = spatial.drop_duplicates(
        ["proposal_row_order", "surface_class_index"]
    )
    surface_indices = np.array([0, 1, 2, 3, 4, 7])
    expected_indices = np.tile(surface_indices, 903)
    expected_labels = np.tile(
        np.array(["green", "fairway", "rough", "sand", "recovery", "penalty"]),
        903,
    )
    if (
        len(surface_branches) != 5_418
        or not np.array_equal(
            surface_branches["surface_class_index"].to_numpy(np.int64),
            expected_indices,
        )
        or not np.array_equal(
            surface_branches["surface_class_label"].to_numpy(str),
            expected_labels,
        )
    ):
        raise AuditV2Error("six-surface topology changed")
    if (
        spatial.groupby("proposal_row_order")[
            "surface_full_probability_json"
        ].nunique().max() != 1
        or spatial.groupby("proposal_row_order")[
            "surface_full_cdf_upper_json"
        ].nunique().max() != 1
    ):
        raise AuditV2Error("surface vector split")
    probability_vectors = np.asarray(
        [
            json.loads(text, parse_constant=_reject_constant)
            for text in spatial_representatives["surface_full_probability_json"]
        ],
        dtype=np.float64,
    )
    cdf_vectors = np.asarray(
        [
            json.loads(text, parse_constant=_reject_constant)
            for text in spatial_representatives["surface_full_cdf_upper_json"]
        ],
        dtype=np.float64,
    )
    if probability_vectors.shape != (903, 9) or cdf_vectors.shape != (903, 9):
        raise AuditV2Error("surface vector dimensions changed")
    if (
        not (probability_vectors[:, [5, 6, 8]] == 0.0).all()
        or not (probability_vectors[:, surface_indices] > 0.0).all()
        or spatial_representatives["surface_full_probability_json"].nunique() != 902
    ):
        raise AuditV2Error("surface support changed")
    _close(probability_vectors.sum(axis=1), np.ones(903), "surface mass")
    _close(np.cumsum(probability_vectors, axis=1), cdf_vectors, "surface CDF")
    expected_surface_mass = probability_vectors[:, surface_indices].reshape(-1)
    expected_surface_lower = np.column_stack(
        (
            np.zeros(903), cdf_vectors[:, 0], cdf_vectors[:, 1],
            cdf_vectors[:, 2], cdf_vectors[:, 3], cdf_vectors[:, 6],
        )
    ).reshape(-1)
    expected_surface_upper = cdf_vectors[:, surface_indices].reshape(-1)
    _close(
        surface_branches["surface_probability"].to_numpy(float),
        expected_surface_mass, "materialized surface mass",
    )
    _close(
        surface_branches["surface_cdf_lower"].to_numpy(float),
        expected_surface_lower, "materialized surface lower CDF",
    )
    _close(
        surface_branches["surface_cdf_upper"].to_numpy(float),
        expected_surface_upper, "materialized surface upper CDF",
    )

    resolved = spatial.loc[
        spatial["hazard_resolution_status"] == "resolved_complete_atom"
    ]
    no_hazard = spatial.loc[spatial["hazard_resolution_status"] == "none"]
    if (
        len(resolved) != 30_702 or len(no_hazard) != 4_515
        or not np.array_equal(
            resolved["hazard_donor_order"].to_numpy(np.int64),
            np.tile(np.arange(34, dtype=np.int64), 903),
        )
        or set(resolved["surface_class_index"]) != {7}
        or set(resolved["stroke_cost"]) != {2}
        or set(no_hazard["surface_class_index"]) != {0, 1, 2, 3, 4}
        or set(no_hazard["hazard_donor_order"]) != {-1}
        or set(no_hazard["hazard_donor_key"]) != {"not_applicable"}
        or set(no_hazard["stroke_cost"]) != {1}
    ):
        raise AuditV2Error("34-hazard/5-no-hazard topology changed")
    hazard_lower = resolved["hazard_cdf_lower"].to_numpy(float).reshape(903, 34)
    hazard_upper = resolved["hazard_cdf_upper"].to_numpy(float).reshape(903, 34)
    hazard_mass = resolved["hazard_probability"].to_numpy(float).reshape(903, 34)
    _close(hazard_lower[:, 0], np.zeros(903), "hazard CDF starts")
    _close(hazard_upper[:, -1], np.ones(903), "hazard CDF ends")
    _close(hazard_lower[:, 1:], hazard_upper[:, :-1], "hazard CDF continuity")
    _close(hazard_upper - hazard_lower, hazard_mass, "hazard CDF widths")
    _close(
        no_hazard["hazard_probability"].to_numpy(float),
        np.ones(4_515), "no-hazard mass",
    )

    quality = frame["quality_base_mass"].to_numpy(float)
    _close(
        quality,
        frame["proposal_probability"].to_numpy(float) * mechanics_mass,
        "proposal-mechanics factorization",
    )
    _close(
        mechanics_mass,
        frame["surface_probability"].to_numpy(float)
        * frame["hazard_probability"].to_numpy(float),
        "surface-hazard factorization",
    )
    _close(
        frame["base_hierarchy_mass"].to_numpy(float),
        frame["proposal_base_hierarchy_mass"].to_numpy(float) * mechanics_mass,
        "base hierarchy factorization",
    )
    _close(
        frame["proposal_spatial_model_weight"].to_numpy(float),
        frame["spatial_model_weight"].to_numpy(float),
        "spatial model-weight identity",
    )
    _close(
        frame["spatial_model_weight"].to_numpy(float),
        np.ones(35_218), "unit spatial model weight",
    )
    complete_lower = frame["complete_cdf_lower"].to_numpy(float)
    complete_upper = frame["complete_cdf_upper"].to_numpy(float)
    _cdf_chain(complete_lower, complete_upper, quality, "complete CDF")
    _close([quality.sum()], [1.0], "complete quality mass")

    groups = _group_inventory(frame)
    if (
        grouped["donor_quality_group_id"].nunique().max() != 1
        or grouped["donor_ability_available"].nunique().max() != 1
    ):
        raise AuditV2Error("proposal group/policy split")
    crossing = grouped["__fragment_index"].nunique()
    if crossing.loc[crossing > 1].index.tolist() != [210, 420, 630, 840]:
        raise AuditV2Error("cross-part proposal topology changed")

    available_mask = frame["donor_ability_available"].to_numpy(bool)
    observed = frame.loc[available_mask]
    imputed = frame.loc[~available_mask]
    proposal_available = representatives["donor_ability_available"].to_numpy(bool)
    if (
        len(observed) != 33_229 or len(imputed) != 1_989
        or int(proposal_available.sum()) != 853
        or int((~proposal_available).sum()) != 51
        or set(imputed["donor_ability_value_kind"])
        != {"imputed_frozen_neutral_zero"}
        or not imputed["donor_ability_raw_strict_prior"].isna().all()
        or not (imputed["donor_ability_sg"].to_numpy(float) == 0.0).all()
        or set(observed["donor_ability_value_kind"]) != {"observed_strict_prior"}
        or set(frame["donor_ability_source_column"]) != {"sg_app"}
        or set(frame["donor_ability_policy_sha256"]) != {POLICY_IDENTITY}
        or representatives["donor_ability_row_sha256"].nunique() != 904
    ):
        raise AuditV2Error("ability policy population changed")
    raw_ability = observed["donor_ability_raw_strict_prior"].to_numpy(float)
    final_ability = observed["donor_ability_sg"].to_numpy(float)
    clipped_flag = observed["donor_ability_was_clipped"].to_numpy(bool)
    clipped_expected = np.clip(raw_ability, -2.0, 2.0)
    _close(final_ability, clipped_expected, "observed ability clipping")
    if not np.array_equal(clipped_flag, raw_ability != clipped_expected):
        raise AuditV2Error("ability clipped flag changed")
    clipped_rows = frame.loc[frame["donor_ability_was_clipped"]]
    if (
        len(clipped_rows) != 39
        or set(clipped_rows["proposal_row_order"]) != {137}
        or set(clipped_rows["donor_ability_raw_strict_prior"]) != {-2.12}
        or set(clipped_rows["donor_ability_sg"]) != {-2.0}
    ):
        raise AuditV2Error("single clipped proposal changed")
    date_counts: dict[tuple[str, str | None], int] = {}
    for row in representatives[["donor_event_key", "donor_skill_date"]].itertuples(
        index=False, name=None
    ):
        key = (str(row[0]), _time_text(row[1]))
        date_counts[key] = date_counts.get(key, 0) + 1
    nonnull_skill = representatives["donor_skill_date"].notna()
    if (
        set(frame["target_player_id"]) != {"33204"}
        or set(frame["target_event_key"]) != {"pga:R2026034"}
        or set(value.isoformat() for value in frame["target_event_start"])
        != {"2026-06-25T00:00:00+00:00"}
        or representatives.loc[
            representatives["donor_ability_available"], "donor_skill_date"
        ].isna().any()
        or not (
            representatives.loc[nonnull_skill, "donor_skill_date"]
            < representatives.loc[nonnull_skill, "donor_event_start"]
        ).all()
        or not (
            representatives["donor_event_start"]
            < representatives["target_event_start"]
        ).all()
        or date_counts != ABILITY_DATE_COUNTS
    ):
        raise AuditV2Error("strict-prior target/date relation changed")

    if (
        set(frame["reference_model_sha256"]) != {REFERENCE_MODEL}
        or set(frame["reference_putting_sha256"]) != {PUTTING_MODEL}
        or set(frame["reference_descriptor"])
        != {f"frozen_tour_expected_strokes_reference_v1:{REFERENCE_MODEL}"}
        or set(frame["reference_version"])
        != {"broadie_2020_2025_plus_2020_prior_putting_v1"}
        or set(frame["source_cache_sha256"])
        != {"5adde84d8fc7b5ac89c5a20a08e321c393d32acb452b3dac5390afc1e44b70d2"}
        or set(frame["surface_model_sha256"])
        != {"b5f8a57a4c328e1b30506f899bed3f8ff68705f46ab52b357b89c4c3928c6ae1"}
        or set(frame["hazard_resolver_sha256"])
        != {"f7a674bb282afea5281657feb0f58c88ad4fad444f41c758be3192bde58ebce6"}
    ):
        raise AuditV2Error("reference/model identities changed")
    _close(
        frame["fixed_reference_reward"].to_numpy(float),
        frame["start_reference_value"].to_numpy(float)
        - frame["end_reference_value"].to_numpy(float)
        - frame["stroke_cost"].to_numpy(float),
        "fixed-reference reward equation",
    )
    _close(
        frame["start_reference_value"].to_numpy(float),
        np.full(35_218, 3.131354288195138), "start reference value",
    )

    ndjson = _ndjson_components(frame, schema)

    return {
        "scope": SCOPE,
        "schema": {
            "columns": 150, "all_nullable": True,
            "ndjson": ndjson["schema"],
        },
        "counts": {
            "complete_atoms": 35_218, "proposals": 904,
            "spatial_proposals": 903, "terminal_atoms": 1,
            "atoms_per_spatial_proposal": 39, "surface_records": 903,
            "materialized_surface_branches": 5_418,
            "resolved_hazard_atoms": 30_702,
            "no_hazard_spatial_atoms": 4_515,
            "observed_ability_atoms": 33_229,
            "imputed_ability_atoms": 1_989,
            "observed_ability_proposals": 853,
            "imputed_ability_proposals": 51,
            "clipped_ability_atoms": 39,
            "donor_quality_groups": 6, "mixed_policy_groups": 4,
        },
        "topology": {
            "surface_indices": [0, 1, 2, 3, 4, 7],
            "surface_labels": [
                "green", "fairway", "rough", "sand", "recovery", "penalty"
            ],
            "hazards_per_spatial_proposal": 34,
            "cross_part_proposals": [210, 420, 630, 840],
            "unique_surface_vectors": 902,
        },
        "mass_cdf": {
            "proposal": True, "proposal_branch": True,
            "frozen_spatial": True, "terminal_residual": True,
            "surface": True, "hazard": True, "mechanics": True,
            "complete": True, "all_factorizations": True,
        },
        "groups": groups,
        "ability_policy": {
            "policy_sha256": POLICY_IDENTITY,
            "observed_state": "observed_strict_prior",
            "imputed_state": "imputed_frozen_neutral_zero",
            "clip_bounds": [-2.0, 2.0], "clipped_proposal": 137,
            "date_relation": (
                "observed_skill_date_present;non_null_skill_date<donor_event_start;"
                "exact_11_bucket_population"
            ),
        },
        "reference_reward": {
            "reference_model_sha256": REFERENCE_MODEL,
            "putting_model_sha256": PUTTING_MODEL,
            "reward_equation": "start_reference-end_reference-stroke_cost",
            "start_reference_value": 3.131354288195138,
        },
        "ndjson_components": ndjson,
        "audit_only_semantic_root_sha256": ndjson["semantic_root"]["sha256"],
    }


def _collect(attestation: Mapping[str, object]) -> dict[str, object]:
    candidate_bytes, candidate_start = _capture_pinned(
        FIXED_CANDIDATE, CANDIDATE_FILES, (), "candidate"
    )
    original_bytes, original_start = _capture_pinned(
        FIXED_ORIGINAL, ORIGINAL_FILES,
        ("complete_atom_support", "complete_atom_support/parts"),
        "original support",
    )
    _same(
        {
            "files": candidate_start["files"],
            "directories": candidate_start["directories"],
            "bytes": candidate_start["bytes"],
        },
        {"files": 4, "directories": 0, "bytes": 7_203},
        "candidate byte inventory",
    )
    _same(
        {
            "files": original_start["files"],
            "directories": original_start["directories"],
            "bytes": original_start["bytes"],
        },
        {"files": 10, "directories": 2, "bytes": 7_647_733},
        "original byte inventory",
    )
    source_start = _source_state()
    launcher_start = _launcher_state(attestation)
    _validate_candidate(candidate_bytes)
    part_buffers = _validate_original(original_bytes)
    frame, schema = _load_from_captures(part_buffers)
    vectorized = _vector_checks(frame, schema)

    _, candidate_end = _capture_pinned(
        FIXED_CANDIDATE, CANDIDATE_FILES, (), "candidate"
    )
    _, original_end = _capture_pinned(
        FIXED_ORIGINAL, ORIGINAL_FILES,
        ("complete_atom_support", "complete_atom_support/parts"),
        "original support",
    )
    source_end = _source_state()
    launcher_end = _launcher_state(attestation)
    _same(candidate_start, candidate_end, "candidate start/end")
    _same(original_start, original_end, "original start/end")
    _same(source_start, source_end, "auditor/source start/end")
    _same(launcher_start, launcher_end, "launcher start/end")
    return {
        "candidate_bytes_start": candidate_start,
        "candidate_bytes_end": candidate_end,
        "original_bytes_start": original_start,
        "original_bytes_end": original_end,
        "source_start": source_start, "source_end": source_end,
        "launcher_start": launcher_start, "launcher_end": launcher_end,
        "external_canonical_evidence": dict(EXTERNAL_CANONICAL_EVIDENCE),
        "vectorized_invariants": vectorized,
    }


def _audit_id(body: Mapping[str, object]) -> str:
    return _json_digest(
        {"role": "non_authorizing_detached_candidate_audit_v2", "payload": body},
        AUDIT_CONTRACT,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)


def _report(audit: Mapping[str, object]) -> str:
    root_sha = audit["vectorized_invariants"][
        "audit_only_semantic_root_sha256"
    ]
    return (
        "# Lowry R1 H8 selector candidate byte/vectorized audit v2\n\n"
        "Status: GO for separate data-only authority-record review only.\n\n"
        f"- Scope: {SCOPE}\n"
        f"- Audit ID: {audit['audit_id']}\n"
        f"- Candidate ID: {EXPECTED_CANDIDATE_ID}\n"
        f"- Audit-only NDJSON semantic root: {root_sha}\n"
        "- Canonical digests are sealed external evidence, not rederived here.\n"
        "- Authority, admission, weighting, selection, callback, solve, simulation, "
        "chain, anchor, and production remain disabled.\n"
    )


def _capture_output(root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    files, directories = _entries(root)
    required = {"OUTPUT_SHA256.json", "REPORT.md", "audit.json", "manifest.json"}
    if directories or set(files) != required:
        raise AuditV2Error("audit output closure changed")
    captured: dict[str, bytes] = {}
    inventory: list[dict[str, object]] = []
    for relative in sorted(files):
        raw = files[relative].read_bytes()
        captured[relative] = raw
        inventory.append(
            {"path": relative, "bytes": len(raw), "sha256": _digest(raw)}
        )
    state = {
        "files": 4, "directories": 0,
        "bytes": sum(item["bytes"] for item in inventory),
        "inventory_sha256": _json_digest(inventory, "audit_v2_output_tree"),
    }
    return captured, state


def _validate_output(
    root: Path,
    expected_audit: Mapping[str, object],
) -> dict[str, object]:
    captured, state = _capture_output(root)
    audit = _strict_json(captured["audit.json"], "written audit")
    _same(audit, dict(expected_audit), "written audit payload")
    body = dict(audit)
    observed_audit_id = body.pop("audit_id")
    _same(observed_audit_id, _audit_id(body), "audit identity")
    for field, expected in FAIL_CLOSED.items():
        _same(audit[field], expected, f"audit {field}")
    _same(audit["authority_id"], None, "audit authority id")
    _same(audit["authority_lock_sha256"], None, "audit authority lock")
    _same(
        audit["data_only_authority_record_ready"], False,
        "data-only authority readiness",
    )
    _same(audit["next_review_requirements"], NEXT_REVIEW, "next review")
    _same(captured["REPORT.md"].decode("utf-8"), _report(audit), "audit report")

    payload = {
        name: {"bytes": len(captured[name]), "sha256": _digest(captured[name])}
        for name in ("REPORT.md", "audit.json")
    }
    expected_manifest = {
        "contract": MANIFEST_CONTRACT, "audit_id": observed_audit_id,
        "scope": SCOPE, "decision": DECISION, "audit_status": AUDIT_STATUS,
        "reviewed_candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_artifact_review_passed": True,
        "data_only_authority_record_ready": False,
        **FAIL_CLOSED, "research_only": True, "payload_files": payload,
    }
    _same(
        _strict_json(captured["manifest.json"], "audit manifest"),
        expected_manifest,
        "audit manifest",
    )
    expected_index = {
        "contract": INDEX_CONTRACT,
        "files": {
            name: {
                "bytes": len(captured[name]), "sha256": _digest(captured[name])
            }
            for name in ("REPORT.md", "audit.json", "manifest.json")
        },
    }
    _same(
        _strict_json(captured["OUTPUT_SHA256.json"], "audit index"),
        expected_index,
        "audit output index",
    )
    return state


def _final_source_recheck(
    evidence: Mapping[str, object],
    attestation: Mapping[str, object],
) -> None:
    _, candidate_state = _capture_pinned(
        FIXED_CANDIDATE, CANDIDATE_FILES, (), "final candidate"
    )
    _, original_state = _capture_pinned(
        FIXED_ORIGINAL, ORIGINAL_FILES,
        ("complete_atom_support", "complete_atom_support/parts"),
        "final original support",
    )
    source_state = _source_state()
    launcher_state = _launcher_state(attestation)
    _same(
        candidate_state, evidence["candidate_bytes_end"], "final candidate state"
    )
    _same(
        original_state, evidence["original_bytes_end"], "final original state"
    )
    _same(source_state, evidence["source_end"], "final source state")
    _same(launcher_state, evidence["launcher_end"], "final launcher state")


def _build(attestation: Mapping[str, object]) -> dict[str, object]:
    if FIXED_OUTPUT.exists():
        raise FileExistsError(f"fixed v2 audit output must be absent: {FIXED_OUTPUT}")
    if not FIXED_OUTPUT.parent.is_dir():
        raise FileNotFoundError(FIXED_OUTPUT.parent)
    evidence = _collect(attestation)
    body: dict[str, object] = {
        "contract": AUDIT_CONTRACT, "scope": SCOPE, "decision": DECISION,
        "audit_status": AUDIT_STATUS,
        "reviewed_candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_artifact_review_passed": True,
        "original_support_review_passed": True,
        "data_only_authority_record_ready": False,
        "authority_id": None, "authority_lock_sha256": None,
        **FAIL_CLOSED, "research_only": True, **evidence,
        "next_review_requirements": dict(NEXT_REVIEW),
    }
    audit = {"audit_id": _audit_id(body), **body}
    audit_raw = _json_bytes(audit)
    report_raw = _report(audit).encode("utf-8")
    FIXED_OUTPUT.mkdir(parents=False, exist_ok=False)
    _write_exclusive(FIXED_OUTPUT / "audit.json", audit_raw)
    _write_exclusive(FIXED_OUTPUT / "REPORT.md", report_raw)
    payload = {
        "REPORT.md": {"bytes": len(report_raw), "sha256": _digest(report_raw)},
        "audit.json": {"bytes": len(audit_raw), "sha256": _digest(audit_raw)},
    }
    manifest = {
        "contract": MANIFEST_CONTRACT, "audit_id": audit["audit_id"],
        "scope": SCOPE, "decision": DECISION, "audit_status": AUDIT_STATUS,
        "reviewed_candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_artifact_review_passed": True,
        "data_only_authority_record_ready": False,
        **FAIL_CLOSED, "research_only": True, "payload_files": payload,
    }
    manifest_raw = _json_bytes(manifest)
    _write_exclusive(FIXED_OUTPUT / "manifest.json", manifest_raw)
    index = {
        "contract": INDEX_CONTRACT,
        "files": {
            **payload,
            "manifest.json": {
                "bytes": len(manifest_raw), "sha256": _digest(manifest_raw)
            },
        },
    }
    _write_exclusive(
        FIXED_OUTPUT / "OUTPUT_SHA256.json", _json_bytes(index)
    )
    validated_tree = _validate_output(FIXED_OUTPUT, audit)
    _final_source_recheck(evidence, attestation)
    # The final auditor filesystem operation is this audit-tree capture.
    _captured, final_tree = _capture_output(FIXED_OUTPUT)
    _same(final_tree, validated_tree, "final audit tree")
    return {
        "mode": "audit", "scope": SCOPE, "decision": DECISION,
        "audit_status": AUDIT_STATUS, "audit_id": audit["audit_id"],
        "output": str(FIXED_OUTPUT), "final_audit_tree": final_tree,
    }


def final_audit_tree_v2() -> dict[str, object]:
    _captured, state = _capture_output(FIXED_OUTPUT)
    return state


def run_fixed_audit_v2(
    mode: str,
    launcher_attestation: Mapping[str, object],
) -> dict[str, object]:
    if type(launcher_attestation) is not dict:
        raise AuditV2Error("launcher attestation must be an exact built-in dict")
    if mode == "validate-only":
        if FIXED_OUTPUT.exists():
            raise AuditV2Error("fixed v2 output exists during source-only validation")
        evidence = _collect(launcher_attestation)
        return {
            "mode": "validate-only", "scope": SCOPE,
            "audit_only_semantic_root_sha256": evidence[
                "vectorized_invariants"
            ]["audit_only_semantic_root_sha256"],
            "status": "HOLD_source_reviews_required_no_audit_emitted",
        }
    if mode == "audit":
        return _build(launcher_attestation)
    raise AuditV2Error("mode must be exactly validate-only or audit")

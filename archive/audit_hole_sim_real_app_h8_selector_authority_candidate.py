"""Independently audit the fixed Lowry R1 H8 real-APP selector candidate.

This detached auditor does not import or trust the candidate builder or any
selector validator.  It independently closes and rehashes the candidate and
the original five-part materialized support, reconstructs the exact schema,
typed-scalar digests, topology, groups, policy, and reference identities, and
can emit only a non-authorizing review artifact at one fixed path.

Passing this audit is permission only to review a later, separate data-only
authority record.  Weighting, selection, callbacks, solves, simulations, and
production remain fail-closed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent

AUDITOR_SOURCE = Path(__file__).resolve()
AUDITOR_TEST = REPO / "test_audit_hole_sim_real_app_h8_selector_authority_candidate.py"

CANDIDATE_CONTRACT = "fixed_lowry_r1h8_real_app_selector_support_candidate_v1"
CANDIDATE_MANIFEST_CONTRACT = "fixed_lowry_r1h8_real_app_selector_candidate_manifest_v1"
AUDIT_CONTRACT = "detached_lowry_r1h8_real_app_selector_candidate_audit_v1"
AUDIT_MANIFEST_CONTRACT = "detached_lowry_r1h8_real_app_selector_candidate_audit_manifest_v1"
OUTPUT_INDEX_CONTRACT = "non_circular_output_sha256_v1"
OUTPUT_INDEX = "OUTPUT_SHA256.json"
ADMISSION_STATUS = "candidate_external_source_and_artifact_review_required"
AUDIT_DECISION = "GO_for_separate_data_only_authority_record_review_only"
AUDIT_STATUS = "candidate_artifact_passed_data_only_authority_record_review_required"

_FIXED_SOURCE_ROOT = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1"
)
_FIXED_CANDIDATE_ROOT = (
    REPO
    / "permanent_data"
    / "hole_sim_real_app_h8_selector_authority_candidate_v1"
)
_FIXED_AUDIT_ROOT = (
    REPO
    / "permanent_data"
    / "hole_sim_real_app_h8_selector_authority_candidate_audit_v1"
)
_INNER = "complete_atom_support"

EXPECTED_CANDIDATE_ID = (
    "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942"
)
EXPECTED_CANDIDATE_INDEX_SHA256 = (
    "ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43"
)
EXPECTED_CANDIDATE_JSON_SHA256 = (
    "689b5c0ae5827ea99fd603a7b3af39163685d498dd7c1ac85c0777758e2d6e00"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "b5b5f07740dd43e8933aca21674351b435918b803a60fdfc5ad7dad78364c040"
)
EXPECTED_CANDIDATE_REPORT_SHA256 = (
    "5489db51378a60035b7d090036abcfe43f34878cf7ba2e6ce2fa4d1bbeacf879"
)
EXPECTED_CANDIDATE_BUILDER_SHA256 = (
    "b3fdbd12f2e466bdac92f71f2d60a3e9fd0aec18a58f309248dc1c783046fdd7"
)
EXPECTED_CANDIDATE_TEST_SHA256 = (
    "ebea0b3f4f4e196931908f89eeb3c7189b79784cfb5eb53a2f8027a342773db4"
)
EXPECTED_AUDITOR_TEST_SHA256 = (
    "443f725b77a2b5b6fe8815655d9f5f90bc5a2cd0ab7803b1253f8565030b2b46"
)

EXPECTED_OUTER_INDEX_SHA256 = (
    "5a502f524ada60bb91e85f1e18474056d4b2a883be14c0e1f7df3450393093b2"
)
EXPECTED_OUTER_MANIFEST_SHA256 = (
    "c96a8e8333c8e535a82e9e265f74cde41d6a5a717e36c68c9dea9894e39f5464"
)
EXPECTED_INNER_INDEX_SHA256 = (
    "e300bf8b00066ec02e03128ca8e0e6b7a89c4e5a772a2ac5f2dd5794dbe2693f"
)
EXPECTED_INNER_MANIFEST_SHA256 = (
    "4dec00946a085fedb489aa51fef672188d4205c6a26c85faeb7156e5234da7f2"
)

EXPECTED_QUERY_KEY = (
    "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
)
EXPECTED_SUPPORT_KEY = "app|raw|R1|H8|tee|B14|33204"
EXPECTED_REFERENCE_MODEL_SHA256 = (
    "35ffdc1aa21ccc209ceb56a8b8e78b3118aa1cc8bd4c7372a92a0dc01b0bc68c"
)
EXPECTED_PUTTING_SHA256 = (
    "aace8326debb5c2af07b16e319730601a56d9ddd978d31ce4a0d3ee1d82f7c93"
)
EXPECTED_POLICY_IDENTITY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)
EXPECTED_CANONICAL_FULL_SUPPORT_SHA256 = (
    "b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7"
)
EXPECTED_CANONICAL_PROJECTION_SHA256 = (
    "a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889"
)
EXPECTED_CANONICAL_SELECTOR_SUPPORT_SHA256 = (
    "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
)
EXPECTED_CANONICAL_TOPOLOGY_SHA256 = (
    "9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484"
)

EXPECTED_AUTHORITY_EVIDENCE = {
    "authority_id": "lowry_r1h8_raw_app_exact_query_v1",
    "authority_lock_sha256": (
        "edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c"
    ),
    "authority_record_sha256": (
        "efaf969daea1ac9a7680b0fd8d534ffa700430523aa7cb13cec6fabaed9c347c"
    ),
    "candidate_id": (
        "341ce3818fcc5301179c8858fcba02a94778384e2e554e81fd762d2f9626b7ff"
    ),
    "candidate_output_index_sha256": (
        "28bc60ca52ec42ee1abba4eaa26d5d10a6cc50f6df1fdb4b0e1631594c977aa5"
    ),
    "candidate_manifest_sha256": (
        "d0be4068daf67c05243eeed4eb2963168ec8555ba2dc5842d72ade434e09bfc2"
    ),
    "authority_candidate_sha256": (
        "f046624aa48601a37b7ad6e8b2ab94f8558e857fca6220e286b9611dc010d5d4"
    ),
    "audit_id": (
        "16c79fb1fd622db1ee7f4b8cb638ec0b72b746d0c2370541c23957d12d23b6e1"
    ),
    "audit_output_index_sha256": (
        "cdfb7b50a958ddd397aaf5297767d0f1d5e11ca80b946eb0a13475a04f8f3431"
    ),
    "audit_json_sha256": (
        "57231a27cbb7df00fe4af95d957efccdcbdad67d5e202340a3ff2f4cd0c43402"
    ),
    "audit_report_sha256": (
        "405428f00644bb53984dbc89ec2394d6f54441865d55f971324329b5de2f5368"
    ),
}

EXPECTED_SOURCE_BINDINGS = {
    "archive/hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": (
        "f47c35d0345410bf165885e81afd8c0a0320f4950118889d2c4ac3fce6148bd0"
    ),
    "archive/hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": (
        "ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460"
    ),
    "archive/hole_sim_tee_app_complete_atom_support.py": (
        "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
    ),
    "archive/hole_sim_donor_ability_policy.py": (
        "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
    ),
    "archive/hole_sim_transition_mean_anchor.py": (
        "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
    ),
    "archive/hole_sim_transition_mean_chain_adapter.py": (
        "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff"
    ),
    "archive/hole_sim_tee_app_h8_source_authority.py": (
        "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
    ),
    "archive/audit_hole_sim_tee_app_h8_source_authority.py": (
        "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06"
    ),
    "test_hole_sim_real_app_whole_atom_selector_lowry_r1h8.py": (
        "bd3875f4de9e9082597a065f719e5b09bc053a5722e30f461f219e6dd1a1f4c9"
    ),
    "test_hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py": (
        "61103a4f410fb2f2fcbdcec2ad20dc40c13c2c37a8975bae4587aa839b78153f"
    ),
    "test_hole_sim_tee_app_complete_atom_support.py": (
        "eb648d94447d8a60497dcf95b34dcbbe1a64c21cd5458bb27ad87c1673898397"
    ),
    "archive/hole_sim_compact_app_whole_atom_selector_v1.py": (
        "e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7"
    ),
    "test_hole_sim_compact_app_whole_atom_selector_v1.py": (
        "21bd3aa8f0f1d2340bdefb32b3fce4bae59fdbdc87c23de3129710c7287b77dd"
    ),
    "test_hole_sim_tee_app_h8_source_authority.py": (
        "93966e84c85d7d3440516f0c80492e8c7d9145448aeaf88859a8cd833f115813"
    ),
    "test_audit_hole_sim_tee_app_h8_source_authority.py": (
        "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c"
    ),
    "test_hole_sim_real_app_h8_selector_authority_candidate.py": (
        "ebea0b3f4f4e196931908f89eeb3c7189b79784cfb5eb53a2f8027a342773db4"
    ),
}

MAX_PARTS = 5
MAX_ROWS_PER_PART = 8_192
MAX_COMPLETE_ATOMS = 35_218
MAX_PROPOSALS = 904
MAX_SPATIAL_PROPOSALS = 903
MAX_SURFACE_CLASSES = 9
MAX_MATERIALIZED_SURFACE_BRANCHES = 6
MAX_HAZARDS_PER_SURFACE = 34
MAX_COMPLETE_ATOMS_PER_PROPOSAL = 16_384
EXPECTED_TERMINAL_ATOMS = 1
EXPECTED_IMPUTED_ATOMS = 1_989
EXPECTED_GROUPS = 6
EXPECTED_MIXED_GROUPS = 4
MASS_TOLERANCE = 3e-13

_FIXED_INPUT_IDENTITY_PAYLOAD = {
    "outer_output_index_sha256": EXPECTED_OUTER_INDEX_SHA256,
    "outer_manifest_sha256": EXPECTED_OUTER_MANIFEST_SHA256,
    "inner_output_index_sha256": EXPECTED_INNER_INDEX_SHA256,
    "inner_manifest_sha256": EXPECTED_INNER_MANIFEST_SHA256,
}

_FIXED_CAPS_PAYLOAD = {
    "parts": MAX_PARTS,
    "rows_per_part": MAX_ROWS_PER_PART,
    "complete_atoms": MAX_COMPLETE_ATOMS,
    "proposals": MAX_PROPOSALS,
    "surface_classes": MAX_SURFACE_CLASSES,
    "hazards_per_surface": MAX_HAZARDS_PER_SURFACE,
    "complete_atoms_per_proposal": MAX_COMPLETE_ATOMS_PER_PROPOSAL,
}

_EXTERNAL_REVIEW_REQUIREMENTS = {
    "independent_source_review_required": True,
    "independent_artifact_review_required": True,
    "candidate_id_is_not_authority": True,
    "separate_data_only_authority_record_required": True,
}

_EXPECTED_PARTS = (
    (
        "parts/part-000000.parquet",
        8_192,
        1_751_670,
        "3f3905bc2874e81baade5618a26b316e7a97da17dc3966766e22bfef602e38c7",
        "0f225adf7b7074c5ba7f27a948756bf708c876628498716856627f9aa93bfaf8",
    ),
    (
        "parts/part-000001.parquet",
        8_192,
        1_767_025,
        "4c1c349977834aeab644b1a0b37fe99512c66fbe3332495eaf871788f1131ed6",
        "1c56da0e51b3627246d62bf0a065ff4799475bc6b2b900fdf6ae8487f9c27c6a",
    ),
    (
        "parts/part-000002.parquet",
        8_192,
        1_748_952,
        "bcac8a464d13dd52eaaa3a9bc3e135d9022843c3c52613f0feca491aa72bb306",
        "96c75b4165824b28928081ced2711d3954cd132bb177694616e1d95268cd81be",
    ),
    (
        "parts/part-000003.parquet",
        8_192,
        1_762_480,
        "9fd805ad604790e366f99f2435742866e2e72bf771a5a97bfbf6327d14ef33ca",
        "9bfd7255baa22069417df9303261cc307e13b66c94dc8be1263bbcb3ea3d598d",
    ),
    (
        "parts/part-000004.parquet",
        2_450,
        600_833,
        "ef93b0721901b106f20e6bb6518c2875c5224bb545f988b934fd254e5b23e0a2",
        "00b8393e43c95aef677532b1cc79d2815be32736c3eda828214406cd30376a09",
    ),
)

_OUTER_INDEX_FILES = {
    "REPORT.md": (395, "c56b043fd88dae28f4c4af23fe4b0c6d5d8f4139115acb3a9ae495b700f62dc7"),
    "complete_atom_support/OUTPUT_SHA256.json": (974, EXPECTED_INNER_INDEX_SHA256),
    "complete_atom_support/manifest.json": (6_495, EXPECTED_INNER_MANIFEST_SHA256),
    "complete_atom_support/parts/part-000000.parquet": (1_751_670, _EXPECTED_PARTS[0][3]),
    "complete_atom_support/parts/part-000001.parquet": (1_767_025, _EXPECTED_PARTS[1][3]),
    "complete_atom_support/parts/part-000002.parquet": (1_748_952, _EXPECTED_PARTS[2][3]),
    "complete_atom_support/parts/part-000003.parquet": (1_762_480, _EXPECTED_PARTS[3][3]),
    "complete_atom_support/parts/part-000004.parquet": (600_833, _EXPECTED_PARTS[4][3]),
    "manifest.json": (7_267, EXPECTED_OUTER_MANIFEST_SHA256),
}

_ARROW_SCHEMA_TEXT = """\
arm|string
atom_id|string
atom_kind|string
base_hierarchy_mass|double
boundary_only|bool
bridge_row_id|string
category|string
causal_strokes|int64
complete_atom|bool
complete_cdf_lower|double
complete_cdf_upper|double
complete_holeout_equivalence_status|string
complete_row_order|int64
contract|string
declared_app_base_holeout_probability|double
donor_ability_available|bool
donor_ability_policy_sha256|string
donor_ability_raw_strict_prior|double
donor_ability_row_sha256|string
donor_ability_sg|double
donor_ability_snapshot_kind|string
donor_ability_source_column|string
donor_ability_value_kind|string
donor_ability_was_clipped|bool
donor_event_key|string
donor_event_start|timestamp[ns, tz=UTC]
donor_hole_no|int64
donor_player_id|string
donor_player_name|string
donor_quality_group_id|string
donor_round_no|int64
donor_skill_date|timestamp[ns, tz=UTC]
donor_stroke_no|int64
end_distance_yd|double
end_lat_yd|double
end_lie|string
end_long_yd|double
end_reference_value|double
end_state_key|string
fixed_reference_reward|double
forecast_pin_long_yd|double
forecast_pin_x|double
forecast_pin_y|double
forecast_tee_x|double
forecast_tee_y|double
frozen_spatial_cdf_lower|double
frozen_spatial_cdf_upper|double
frozen_terminal_residual_cdf_lower|double
frozen_terminal_residual_cdf_upper|double
hazard_causal_endpoint_x|double
hazard_causal_endpoint_y|double
hazard_cdf_lower|double
hazard_cdf_upper|double
hazard_component_masses|string
hazard_coordinate_reference_event_key|string
hazard_coordinate_reference_mode|string
hazard_coordinate_reference_year|int64
hazard_data_role|string
hazard_donor_category|string
hazard_donor_causal_endpoint_x|double
hazard_donor_causal_endpoint_y|double
hazard_donor_event_key|string
hazard_donor_hole_no|int64
hazard_donor_key|string
hazard_donor_order|int64
hazard_donor_physical_course_id|string
hazard_donor_round_no|int64
hazard_donor_season|int64
hazard_fallback_used|bool
hazard_foundation_version|string
hazard_nearest_category_endpoint_distance_yd|double
hazard_nearest_same_hole_endpoint_distance_yd|double
hazard_penalty_count|int64
hazard_penalty_stroke_cost|int64
hazard_probability|double
hazard_resolution_event_type|string
hazard_resolution_lie|string
hazard_resolution_shot_key|string
hazard_resolution_status|string
hazard_resolution_x|double
hazard_resolution_y|double
hazard_resolver_sha256|string
hazard_support_category|int64
hazard_support_hole|int64
hazard_support_radius_limit_yd|double
hazard_support_setup|int64
hazard_support_source|string
hole_no|int64
holed|bool
holeout_representation|string
implicit_holeout_probability|double
mechanics_cdf_lower|double
mechanics_cdf_upper|double
mechanics_conditional_probability|double
next_category|string
next_support_key|string
numeric_payload_sha256|string
par|int64
penalty_strokes|int64
physical_course_id|string
proposal_app_holeout_equivalence_sha256|string
proposal_atom_id|string
proposal_base_hierarchy_mass|double
proposal_branch|string
proposal_branch_cdf_lower|double
proposal_branch_cdf_upper|double
proposal_branch_conditional_probability|double
proposal_cdf_lower|double
proposal_cdf_upper|double
proposal_endpoint_lat_yd|double
proposal_endpoint_long_yd|double
proposal_policy_binding_sha256|string
proposal_probability|double
proposal_row_order|int64
proposal_spatial_model_weight|double
proposal_support_key|string
proposal_support_measure_sha256|string
quality_base_mass|double
quality_reward_available|bool
query_key|string
reference_descriptor|string
reference_model_sha256|string
reference_putting_sha256|string
reference_version|string
round_no|int64
selection_contract|string
source_cache_sha256|string
spatial_model_weight|double
start_distance_yd|double
start_lat_yd|double
start_lie|string
start_long_yd|double
start_reference_value|double
start_state_key|string
stroke_cost|int64
support_key|string
surface_cdf_lower|double
surface_cdf_upper|double
surface_class_count|int64
surface_class_index|int64
surface_class_label|string
surface_full_cdf_upper_json|string
surface_full_probability_json|string
surface_model_sha256|string
surface_probability|double
surface_selection_boundary|string
target_event_key|string
target_event_start|timestamp[ns, tz=UTC]
target_player_id|string
terminal|bool"""

EXPECTED_ARROW_SCHEMA = tuple(
    tuple(line.split("|", 1)) for line in _ARROW_SCHEMA_TEXT.splitlines()
)

# The selector source declares 55 entries because ``category`` occurs in both
# its mechanics and policy lists.  The lossless physical projection is these
# 54 unique columns in first-occurrence order.
SEMANTIC_SELECTOR_COLUMNS = (
    "query_key",
    "complete_row_order",
    "atom_id",
    "category",
    "complete_atom",
    "boundary_only",
    "proposal_row_order",
    "proposal_probability",
    "proposal_branch",
    "terminal",
    "mechanics_conditional_probability",
    "quality_base_mass",
    "fixed_reference_reward",
    "donor_quality_group_id",
    "start_state_key",
    "declared_app_base_holeout_probability",
    "frozen_spatial_cdf_lower",
    "frozen_spatial_cdf_upper",
    "frozen_terminal_residual_cdf_lower",
    "frozen_terminal_residual_cdf_upper",
    "surface_class_count",
    "surface_full_probability_json",
    "surface_full_cdf_upper_json",
    "surface_class_index",
    "surface_class_label",
    "surface_probability",
    "surface_cdf_lower",
    "surface_cdf_upper",
    "hazard_resolution_status",
    "hazard_donor_order",
    "hazard_donor_key",
    "hazard_probability",
    "hazard_cdf_lower",
    "hazard_cdf_upper",
    "stroke_cost",
    "end_state_key",
    "next_support_key",
    "reference_model_sha256",
    "reference_putting_sha256",
    "bridge_row_id",
    "donor_player_id",
    "donor_event_key",
    "donor_event_start",
    "target_event_key",
    "target_event_start",
    "donor_skill_date",
    "donor_ability_source_column",
    "donor_ability_available",
    "donor_ability_raw_strict_prior",
    "donor_ability_value_kind",
    "donor_ability_sg",
    "donor_ability_was_clipped",
    "donor_ability_policy_sha256",
    "donor_ability_row_sha256",
)

TOPOLOGY_COLUMNS = (
    "complete_row_order",
    "atom_id",
    "proposal_row_order",
    "proposal_probability",
    "proposal_branch",
    "terminal",
    "mechanics_conditional_probability",
    "quality_base_mass",
    "declared_app_base_holeout_probability",
    "frozen_spatial_cdf_lower",
    "frozen_spatial_cdf_upper",
    "frozen_terminal_residual_cdf_lower",
    "frozen_terminal_residual_cdf_upper",
    "surface_class_count",
    "surface_full_probability_json",
    "surface_full_cdf_upper_json",
    "surface_class_index",
    "surface_class_label",
    "surface_probability",
    "surface_cdf_lower",
    "surface_cdf_upper",
    "hazard_resolution_status",
    "hazard_donor_order",
    "hazard_donor_key",
    "hazard_probability",
    "hazard_cdf_lower",
    "hazard_cdf_upper",
    "stroke_cost",
    "end_state_key",
    "next_support_key",
)

_GROUP_EXPECTATIONS = {
    "03b7d145875af9889ce925a6e93718b1d5cdad75af6187bd91daafa4ffc2d2d8": (7_332, 188, 6_825, 2),
    "0625e8c3727ddf1b2539f5b74f486543e2238f22b8915d13e622d1bbff039816": (7_566, 194, 6_942, 2),
    "3954437bb398899d7be29d91971ee8ab9ccb3e446563acc13080890f3c897e36": (7_059, 181, 6_435, 2),
    "5039b929b7f50dd0d1458855704a44a81366eef05fd92dcebf0d0d80f9d9d2ea": (6_670, 172, 6_436, 2),
    "7cbb0a044f976a3e2e57309b3abfb41fea7d8eb392b1e3d4ff6cc70b0f01ab98": (3_432, 88, 3_432, 1),
    "edff72cd2b781856b6faf5833f1b10dea7f1a362683b3c66764c4aca7c0a37ac": (3_159, 81, 3_159, 1),
}


class SelectorAuthorityCandidateError(ValueError):
    """Raised for any deviation from the fixed review-only contract."""


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
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise SelectorAuthorityCandidateError(f"Non-finite JSON constant: {value}")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SelectorAuthorityCandidateError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectorAuthorityCandidateError(f"Invalid JSON object: {path}") from exc
    if type(value) is not dict:
        raise SelectorAuthorityCandidateError(f"Expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _require_exact(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise SelectorAuthorityCandidateError(
            f"{label} changed: expected {expected!r}, observed {value!r}"
        )


def _require_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise SelectorAuthorityCandidateError(
            f"{label} must be the exact integer {expected}"
        )


def _require_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise SelectorAuthorityCandidateError(
            f"{label} must be the exact boolean {expected}"
        )


def _require_float(value: object, expected: float, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise SelectorAuthorityCandidateError(
            f"{label} must be the exact finite float {expected}"
        )


def _assert_exact_json(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected):
        raise SelectorAuthorityCandidateError(f"{label} JSON type changed")
    if type(expected) is dict:
        if set(value) != set(expected):  # type: ignore[arg-type]
            raise SelectorAuthorityCandidateError(f"{label} JSON keys changed")
        for key in expected:
            _assert_exact_json(value[key], expected[key], f"{label}.{key}")  # type: ignore[index]
        return
    if type(expected) is list:
        if len(value) != len(expected):  # type: ignore[arg-type]
            raise SelectorAuthorityCandidateError(f"{label} JSON length changed")
        for index, item in enumerate(expected):
            _assert_exact_json(value[index], item, f"{label}[{index}]")  # type: ignore[index]
        return
    if value != expected:
        raise SelectorAuthorityCandidateError(f"{label} JSON value changed")


def _is_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise SelectorAuthorityCandidateError(f"Cannot inspect path: {path}") from exc
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _closed_tree(root: Path) -> tuple[dict[str, Path], tuple[str, ...]]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink() or _is_reparse(root):
        raise SelectorAuthorityCandidateError(f"Artifact root is not a plain directory: {root}")
    files: dict[str, Path] = {}
    directories: list[str] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise SelectorAuthorityCandidateError(f"Cannot enumerate {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_reparse(path):
                raise SelectorAuthorityCandidateError(f"Artifact contains a link: {path}")
            relative = path.relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                directories.append(relative)
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                files[relative] = path
            else:
                raise SelectorAuthorityCandidateError(f"Unsupported artifact entry: {path}")

    visit(root)
    return files, tuple(sorted(directories))


def _metadata_map(specification: Mapping[str, tuple[int, str]]) -> dict[str, object]:
    return {
        name: {"bytes": size, "sha256": digest}
        for name, (size, digest) in specification.items()
    }


def _artifact_closure_state(root: Path, *, enforce_fixed: bool) -> dict[str, object]:
    root = Path(root)
    files, directories = _closed_tree(root)
    expected_files = set(_OUTER_INDEX_FILES) | {OUTPUT_INDEX}
    expected_directories = (_INNER, f"{_INNER}/parts")
    if set(files) != expected_files or directories != expected_directories:
        raise SelectorAuthorityCandidateError(
            "Fixed artifact closure changed: "
            f"missing={sorted(expected_files - set(files))}, "
            f"added={sorted(set(files) - expected_files)}, dirs={directories}"
        )
    state_files = {
        relative: {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(files.items())
    }
    if enforce_fixed:
        _require_exact(
            state_files[OUTPUT_INDEX]["sha256"],
            EXPECTED_OUTER_INDEX_SHA256,
            "outer output index SHA-256",
        )
        for relative, (size, digest) in _OUTER_INDEX_FILES.items():
            _require_int(state_files[relative]["bytes"], size, f"{relative} bytes")
            _require_exact(state_files[relative]["sha256"], digest, f"{relative} SHA-256")
    payload = {
        "directories": len(directories),
        "files": len(state_files),
        "bytes": sum(item["bytes"] for item in state_files.values()),
        "file_state_sha256": _stable_sha(state_files),
        "directory_state_sha256": _stable_sha(list(directories)),
    }
    payload["closure_sha256"] = _stable_sha(payload)
    return {"summary": payload, "files": state_files}


def _pinned_builder_source_state() -> dict[str, object]:
    observed = {
        relative: _sha256_file(REPO / PurePosixPath(relative))
        for relative in EXPECTED_SOURCE_BINDINGS
    }
    _assert_exact_json(observed, EXPECTED_SOURCE_BINDINGS, "frozen source bindings")
    builder_path = REPO / "archive" / "hole_sim_real_app_h8_selector_authority_candidate.py"
    _require_exact(
        _sha256_file(builder_path),
        EXPECTED_CANDIDATE_BUILDER_SHA256,
        "candidate builder source SHA-256",
    )
    _require_exact(
        observed["test_hole_sim_real_app_h8_selector_authority_candidate.py"],
        EXPECTED_CANDIDATE_TEST_SHA256,
        "candidate builder focused-test SHA-256",
    )
    payload = {
        "builder_source_sha256": EXPECTED_CANDIDATE_BUILDER_SHA256,
        "bound_sources_sha256": _stable_sha(observed),
        "bound_source_files": len(observed),
    }
    payload["source_closure_sha256"] = _stable_sha(payload)
    return payload


def _auditor_source_state() -> dict[str, object]:
    test_sha = _sha256_file(AUDITOR_TEST)
    _require_exact(
        test_sha,
        EXPECTED_AUDITOR_TEST_SHA256,
        "detached auditor focused-test SHA-256",
    )
    payload = {
        "auditor_source_sha256": _sha256_file(AUDITOR_SOURCE),
        "auditor_test_sha256": test_sha,
    }
    payload["source_closure_sha256"] = _stable_sha(payload)
    return payload


def _validate_outer_and_inner_indexes(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    outer = _read_json(root / OUTPUT_INDEX)
    expected_outer = {
        "contract": "fixed_lowry_r1h8_outer_output_sha256_v1",
        "directories": [_INNER, f"{_INNER}/parts"],
        "files": _metadata_map(_OUTER_INDEX_FILES),
    }
    _assert_exact_json(outer, expected_outer, "outer output index")
    inner = _read_json(root / _INNER / OUTPUT_INDEX)
    expected_inner_files = {
        "manifest.json": {
            "bytes": 6_495,
            "sha256": EXPECTED_INNER_MANIFEST_SHA256,
        },
        **{
            path: {"bytes": size, "sha256": byte_hash}
            for path, _rows, size, byte_hash, _logical_hash in _EXPECTED_PARTS
        },
    }
    expected_inner = {
        "contract": OUTPUT_INDEX_CONTRACT,
        "files": expected_inner_files,
    }
    _assert_exact_json(inner, expected_inner, "inner output index")
    return outer, inner


def _validate_manifests(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    outer = _read_json(root / "manifest.json")
    _require_exact(outer.get("contract"), "fixed_lowry_r1h8_complete_atom_materialization_v1", "outer contract")
    _require_int(outer.get("complete_atoms"), MAX_COMPLETE_ATOMS, "outer complete atoms")
    _require_int(outer.get("terminal_atoms"), EXPECTED_TERMINAL_ATOMS, "outer terminal atoms")
    _require_int(outer.get("imputed_ability_atoms"), EXPECTED_IMPUTED_ATOMS, "outer imputed atoms")
    _require_exact(outer.get("query_key"), EXPECTED_QUERY_KEY, "outer query key")
    _require_exact(outer.get("support_key"), EXPECTED_SUPPORT_KEY, "outer support key")
    _require_exact(outer.get("inner_output_index_sha256"), EXPECTED_INNER_INDEX_SHA256, "outer inner index pin")
    for field in (
        "research_only",
        "production_enabled",
        "simulation_launched",
        "mean_solve_launched",
        "anchor_ready",
    ):
        _require_bool(outer.get(field), field == "research_only", f"outer {field}")
    for field, expected in EXPECTED_AUTHORITY_EVIDENCE.items():
        if field.startswith("candidate_") or field.startswith("authority_candidate"):
            source = outer["artifact_sha256_start"]
            key = field.removeprefix("candidate_") if field.startswith("candidate_") else field
            if field == "candidate_output_index_sha256":
                key = "candidate_output_index"
            elif field == "candidate_manifest_sha256":
                key = "candidate_manifest"
            elif field == "authority_candidate_sha256":
                key = "authority_candidate"
            elif field == "candidate_id":
                source = outer
                key = "candidate_id"
        elif field.startswith("audit_"):
            source = outer["artifact_sha256_start"]
            key = {
                "audit_output_index_sha256": "audit_output_index",
                "audit_json_sha256": "audit_json",
                "audit_report_sha256": "audit_report",
                "audit_id": "audit_id",
            }[field]
            if field == "audit_id":
                source = outer
        else:
            source = outer
            key = field
        if type(source) is not dict:
            raise SelectorAuthorityCandidateError("Outer authority evidence map changed")
        _require_exact(source.get(key), expected, f"authority evidence {field}")
    start = outer.get("artifact_sha256_start")
    end = outer.get("artifact_sha256_end")
    if type(start) is not dict or type(end) is not dict:
        raise SelectorAuthorityCandidateError("Outer artifact start/end maps are absent")
    _assert_exact_json(start, end, "outer embedded artifact start/end")
    source_start = outer.get("source_sha256_start")
    source_end = outer.get("source_sha256_end")
    if type(source_start) is not dict or type(source_end) is not dict:
        raise SelectorAuthorityCandidateError("Outer source start/end maps are absent")
    _assert_exact_json(source_start, source_end, "outer embedded source start/end")

    inner = _read_json(root / _INNER / "manifest.json")
    _require_exact(inner.get("contract"), "bounded_sharded_complete_atom_persistence_v4", "inner contract")
    _require_exact(inner.get("complete_atom_contract"), "complete_tee_app_physical_atom_support_v3", "complete atom contract")
    _require_int(inner.get("complete_atoms"), MAX_COMPLETE_ATOMS, "inner complete atoms")
    _require_int(inner.get("expansion_batch_rows"), MAX_ROWS_PER_PART, "expansion cap")
    _require_int(inner.get("max_rows_per_part"), MAX_ROWS_PER_PART, "row cap")
    _require_int(inner.get("max_hazard_donors_per_surface"), 8_192, "materializer hazard cap")
    _require_int(inner.get("absolute_max_hazard_provider_donors"), 8_192, "absolute hazard cap")
    _require_int(inner.get("max_complete_atoms_per_proposal"), MAX_COMPLETE_ATOMS_PER_PROPOSAL, "proposal cap")
    _require_int(inner.get("peak_buffered_rows"), MAX_ROWS_PER_PART, "peak buffered rows")
    _require_bool(inner.get("research_only"), True, "inner research_only")
    _require_bool(inner.get("production_enabled"), False, "inner production_enabled")
    _require_exact(inner.get("source_sha256_start"), EXPECTED_SOURCE_BINDINGS["archive/hole_sim_tee_app_complete_atom_support.py"], "inner seam start")
    _require_exact(inner.get("source_sha256_end"), EXPECTED_SOURCE_BINDINGS["archive/hole_sim_tee_app_complete_atom_support.py"], "inner seam end")
    _require_exact(inner.get("source_inventory_authority_lock_sha256"), EXPECTED_AUTHORITY_EVIDENCE["authority_lock_sha256"], "inner authority lock")
    _require_exact(inner.get("source_inventory_authority_id"), EXPECTED_AUTHORITY_EVIDENCE["authority_id"], "inner authority id")
    _require_bool(inner.get("source_inventory_authority_ready"), True, "inner source authority ready")
    dependencies = inner.get("dependency_sha256_start")
    dependency_end = inner.get("dependency_sha256_end")
    if type(dependencies) is not dict or type(dependency_end) is not dict:
        raise SelectorAuthorityCandidateError("Inner dependency maps are absent")
    _assert_exact_json(dependencies, dependency_end, "inner dependency start/end")
    for key, relative in {
        "ability_policy": "archive/hole_sim_donor_ability_policy.py",
        "chain_adapter": "archive/hole_sim_transition_mean_chain_adapter.py",
    }.items():
        _require_exact(dependencies.get(key), EXPECTED_SOURCE_BINDINGS[relative], f"inner dependency {key}")
    inventory = inner.get("inventory")
    if type(inventory) is not list or len(inventory) != 1 or type(inventory[0]) is not dict:
        raise SelectorAuthorityCandidateError("Inner inventory envelope changed")
    item = inventory[0]
    for field, expected in {
        "complete_atoms": MAX_COMPLETE_ATOMS,
        "proposal_atoms": MAX_PROPOSALS,
        "terminal_atoms": EXPECTED_TERMINAL_ATOMS,
        "imputed_ability_atoms": EXPECTED_IMPUTED_ATOMS,
        "hazard_atoms": 30_702,
    }.items():
        _require_int(item.get(field), expected, f"inner inventory {field}")
    _require_exact(item.get("query_key"), EXPECTED_QUERY_KEY, "inner inventory query")
    _require_exact(item.get("support_key"), EXPECTED_SUPPORT_KEY, "inner inventory support")
    _require_exact(item.get("reference_model_sha256"), EXPECTED_REFERENCE_MODEL_SHA256, "inner reference model")
    _require_float(item.get("declared_app_holeout_probability"), 0.002, "inner holeout probability")
    raw_parts = inner.get("parts")
    if type(raw_parts) is not list or len(raw_parts) != MAX_PARTS:
        raise SelectorAuthorityCandidateError("Inner part manifest changed")
    for index, expected in enumerate(_EXPECTED_PARTS):
        entry = raw_parts[index]
        if type(entry) is not dict:
            raise SelectorAuthorityCandidateError(f"Inner part {index} is not an object")
        path, rows, size, byte_hash, logical_hash = expected
        expected_entry = {
            "byte_sha256": byte_hash,
            "bytes": size,
            "logical_sha256": logical_hash,
            "path": path,
            "rows": rows,
        }
        _assert_exact_json(entry, expected_entry, f"inner part {index}")
    return outer, inner


def _schema_payload(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _assert_schema(schema: pa.Schema, label: str) -> None:
    observed = tuple((field.name, str(field.type)) for field in schema)
    if observed != EXPECTED_ARROW_SCHEMA:
        raise SelectorAuthorityCandidateError(f"{label} 150-column schema changed")
    if len(schema) != 150 or any(type(field.nullable) is not bool or not field.nullable for field in schema):
        raise SelectorAuthorityCandidateError(f"{label} nullability changed")


def _hash_arrow_tables(tables: Sequence[pa.Table], columns: Sequence[str], contract: str) -> str:
    digest = sha256()
    digest.update(contract.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(columns), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    for table in tables:
        selected = table.select(list(columns)).combine_chunks()
        for batch in selected.to_batches(max_chunksize=1_024):
            for row in range(batch.num_rows):
                for column in range(batch.num_columns):
                    scalar = batch.column(column)[row]
                    if not scalar.is_valid:
                        digest.update(b"N")
                        continue
                    value = scalar.as_py()
                    if type(value) is bool:
                        digest.update(b"B1" if value else b"B0")
                    elif type(value) is int:
                        raw = str(value).encode("ascii")
                        digest.update(b"I" + len(raw).to_bytes(4, "big") + raw)
                    elif type(value) is float:
                        if not math.isfinite(value):
                            raise SelectorAuthorityCandidateError("Semantic support contains a non-finite float")
                        digest.update(b"F" + struct.pack(">d", value))
                    elif type(value) is str:
                        raw = value.encode("utf-8")
                        digest.update(b"S" + len(raw).to_bytes(4, "big") + raw)
                    elif isinstance(value, pd.Timestamp) or hasattr(value, "timestamp"):
                        raw = value.isoformat().encode("ascii")
                        digest.update(b"T" + len(raw).to_bytes(4, "big") + raw)
                    else:
                        raise SelectorAuthorityCandidateError(
                            f"Unsupported semantic scalar type: {type(value).__name__}"
                        )
    return digest.hexdigest()


def _canonical_support_scalar(value: object) -> dict[str, object]:
    """Mirror the reviewed selector's lossless scalar envelope independently."""

    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing"}
    value_type = type(value)
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        return {"type": "integer", "value": str(value)}
    if value_type is float:
        if math.isnan(value):
            return {"type": "missing"}
        if not math.isfinite(value):
            raise SelectorAuthorityCandidateError(
                "Canonical support cannot contain infinity"
            )
        return {"type": "binary64", "value": value.hex()}
    if value_type is pd.Timestamp:
        if value.tzinfo is None:
            raise SelectorAuthorityCandidateError(
                "Canonical support cannot contain a naive timestamp"
            )
        return {
            "type": "timestamp_utc",
            "value": value.tz_convert("UTC").isoformat(),
        }
    if value_type is str:
        return {"type": "string", "value": value}
    raise SelectorAuthorityCandidateError(
        f"Canonical support scalar type changed: {value_type.__name__}"
    )


def _canonical_support_sha(
    tables: Sequence[pa.Table], columns: Sequence[str], contract: str
) -> str:
    """Stream the exact sorted compact-JSON framing used by the selector."""

    digest = sha256()
    dump_options = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": True,
        "allow_nan": False,
    }
    # With sorted keys the top-level order is columns, contract, records.
    digest.update(b'{"columns":')
    digest.update(json.dumps(list(columns), **dump_options).encode("ascii"))
    digest.update(b',"contract":')
    digest.update(json.dumps(contract, **dump_options).encode("ascii"))
    digest.update(b',"records":[')
    first_record = True
    for table in tables:
        selected = table.select(list(columns)).combine_chunks()
        arrays = [selected.column(index) for index in range(selected.num_columns)]
        for row in range(selected.num_rows):
            record = {
                column: _canonical_support_scalar(arrays[index][row].as_py())
                for index, column in enumerate(columns)
            }
            if first_record:
                first_record = False
            else:
                digest.update(b",")
            digest.update(json.dumps(record, **dump_options).encode("ascii"))
    digest.update(b"]}")
    return digest.hexdigest()


def _decode_number_vector(value: object, length: int, label: str) -> np.ndarray:
    if type(value) is not str:
        raise SelectorAuthorityCandidateError(f"{label} must be a JSON string")
    try:
        decoded = json.loads(value, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SelectorAuthorityCandidateError(f"{label} is invalid") from exc
    if type(decoded) is not list or len(decoded) != length:
        raise SelectorAuthorityCandidateError(f"{label} vector length changed")
    if any(type(item) not in (int, float) for item in decoded):
        raise SelectorAuthorityCandidateError(f"{label} contains a non-number")
    vector = np.asarray(decoded, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise SelectorAuthorityCandidateError(f"{label} contains a non-finite number")
    return vector


def _require_close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise SelectorAuthorityCandidateError(f"{label} changed: {left} != {right}")


def _validate_frame(frame: pd.DataFrame) -> dict[str, object]:
    if tuple(frame.columns) != SEMANTIC_SELECTOR_COLUMNS:
        raise SelectorAuthorityCandidateError("Semantic selector projection order changed")
    if len(frame) != MAX_COMPLETE_ATOMS:
        raise SelectorAuthorityCandidateError("Complete support row count changed")
    if not np.array_equal(frame["complete_row_order"].to_numpy(), np.arange(MAX_COMPLETE_ATOMS, dtype=np.int64)):
        raise SelectorAuthorityCandidateError("Complete row order changed")
    if frame["atom_id"].isna().any() or not frame["atom_id"].is_unique:
        raise SelectorAuthorityCandidateError("Physical atom IDs are not complete and unique")
    for column, expected in (("complete_atom", True), ("boundary_only", False)):
        if set(frame[column].tolist()) != {expected}:
            raise SelectorAuthorityCandidateError(f"{column} changed")
    for column, expected in {
        "query_key": EXPECTED_QUERY_KEY,
        "category": "app",
        "reference_model_sha256": EXPECTED_REFERENCE_MODEL_SHA256,
        "reference_putting_sha256": EXPECTED_PUTTING_SHA256,
        "donor_ability_policy_sha256": EXPECTED_POLICY_IDENTITY_SHA256,
    }.items():
        if set(frame[column].tolist()) != {expected}:
            raise SelectorAuthorityCandidateError(f"{column} identity changed")
    if set(frame["declared_app_base_holeout_probability"].tolist()) != {0.002}:
        raise SelectorAuthorityCandidateError("Declared APP holeout probability changed")

    proposal_orders = frame["proposal_row_order"].drop_duplicates().to_numpy()
    if not np.array_equal(proposal_orders, np.arange(MAX_PROPOSALS, dtype=np.int64)):
        raise SelectorAuthorityCandidateError("Proposal order changed")
    proposal_sizes = frame.groupby("proposal_row_order", sort=False).size().to_numpy()
    if not np.array_equal(proposal_sizes, np.asarray([39] * MAX_SPATIAL_PROPOSALS + [1], dtype=np.int64)):
        raise SelectorAuthorityCandidateError("Proposal whole-atom topology changed")
    proposal_mass = frame.drop_duplicates("proposal_row_order")["proposal_probability"].to_numpy(np.float64)
    _require_close(float(proposal_mass.sum()), 1.0, "proposal probability mass")
    terminal = frame.loc[frame["terminal"]]
    if len(terminal) != 1 or int(terminal.iloc[0]["complete_row_order"]) != 35_217:
        raise SelectorAuthorityCandidateError("Explicit APP terminal atom changed")
    terminal_row = terminal.iloc[0]
    terminal_expected = {
        "proposal_row_order": 903,
        "proposal_branch": "terminal",
        "surface_class_count": 0,
        "surface_class_index": -1,
        "surface_class_label": "explicit_app_holeout",
        "hazard_resolution_status": "none",
        "hazard_donor_order": -1,
        "hazard_donor_key": "not_applicable",
        "stroke_cost": 1,
        "end_state_key": "terminal",
        "next_support_key": "",
    }
    for field, expected in terminal_expected.items():
        if terminal_row[field] != expected or type(terminal_row[field]) not in (
            type(expected), np.int64, np.bool_,
        ):
            raise SelectorAuthorityCandidateError(f"Terminal {field} changed")
    for field in ("proposal_probability", "quality_base_mass"):
        _require_close(float(terminal_row[field]), 0.002, f"terminal {field}")
    for field in ("mechanics_conditional_probability", "surface_probability", "hazard_probability"):
        _require_close(float(terminal_row[field]), 1.0, f"terminal {field}")
    for field in ("surface_cdf_lower", "hazard_cdf_lower"):
        _require_close(float(terminal_row[field]), 0.0, f"terminal {field}")
    for field in ("surface_cdf_upper", "hazard_cdf_upper"):
        _require_close(float(terminal_row[field]), 1.0, f"terminal {field}")
    if not pd.isna(terminal_row["frozen_spatial_cdf_lower"]) or not pd.isna(
        terminal_row["frozen_spatial_cdf_upper"]
    ):
        raise SelectorAuthorityCandidateError("Terminal spatial CDF sentinels changed")
    _require_close(
        float(terminal_row["frozen_terminal_residual_cdf_lower"]),
        0.0,
        "terminal residual CDF start",
    )
    _require_close(
        float(terminal_row["frozen_terminal_residual_cdf_upper"]),
        1.0,
        "terminal residual CDF end",
    )

    spatial = frame.loc[~frame["terminal"]]
    if len(spatial) != 35_217 or set(spatial["proposal_branch"].tolist()) != {"spatial"}:
        raise SelectorAuthorityCandidateError("Spatial proposal population changed")
    if set(spatial["surface_class_count"].tolist()) != {MAX_SURFACE_CLASSES}:
        raise SelectorAuthorityCandidateError("Declared surface-class count changed")
    if spatial[
        [
            "frozen_terminal_residual_cdf_lower",
            "frozen_terminal_residual_cdf_upper",
        ]
    ].notna().any(axis=None):
        raise SelectorAuthorityCandidateError("Spatial terminal-CDF sentinels changed")
    spatial_proposals = spatial.drop_duplicates("proposal_row_order")
    if (
        spatial.groupby("proposal_row_order")["frozen_spatial_cdf_lower"]
        .nunique(dropna=False)
        .max()
        != 1
        or spatial.groupby("proposal_row_order")["frozen_spatial_cdf_upper"]
        .nunique(dropna=False)
        .max()
        != 1
    ):
        raise SelectorAuthorityCandidateError("Frozen spatial CDF split within proposal")
    frozen_lower = spatial_proposals["frozen_spatial_cdf_lower"].to_numpy(
        np.float64
    )
    frozen_upper = spatial_proposals["frozen_spatial_cdf_upper"].to_numpy(
        np.float64
    )
    _require_close(float(frozen_lower[0]), 0.0, "spatial CDF start")
    _require_close(float(frozen_upper[-1]), 1.0, "spatial CDF end")
    if (
        not np.isfinite(frozen_lower).all()
        or not np.isfinite(frozen_upper).all()
        or not np.allclose(
            frozen_lower[1:], frozen_upper[:-1], rtol=0.0, atol=MASS_TOLERANCE
        )
        or not np.allclose(
            frozen_upper - frozen_lower,
            spatial_proposals["proposal_probability"].to_numpy(np.float64)
            / (1.0 - 0.002),
            rtol=0.0,
            atol=MASS_TOLERANCE,
        )
    ):
        raise SelectorAuthorityCandidateError("Frozen spatial CDF chain changed")
    surface_rows = spatial.drop_duplicates(["proposal_row_order", "surface_class_index"])
    if len(surface_rows) != MAX_SPATIAL_PROPOSALS * MAX_MATERIALIZED_SURFACE_BRANCHES:
        raise SelectorAuthorityCandidateError("Materialized surface topology changed")
    expected_indices = (0, 1, 2, 3, 4, 7)
    expected_labels = ("green", "fairway", "rough", "sand", "recovery", "penalty")
    for proposal_order, rows in spatial.groupby("proposal_row_order", sort=False):
        by_surface = rows.groupby("surface_class_index", sort=False)
        observed_indices = tuple(int(value) for value in by_surface.size().index.tolist())
        if observed_indices != expected_indices:
            raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface indices changed")
        one = rows.iloc[0]
        probabilities = _decode_number_vector(one["surface_full_probability_json"], 9, "surface probability")
        cdf = _decode_number_vector(one["surface_full_cdf_upper_json"], 9, "surface CDF")
        if not (probabilities[[5, 6, 8]] == 0.0).all() or not (probabilities[list(expected_indices)] > 0.0).all():
            raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface support changed")
        if not np.allclose(np.cumsum(probabilities), cdf, rtol=0.0, atol=MASS_TOLERANCE):
            raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface CDF changed")
        _require_close(float(probabilities.sum()), 1.0, f"proposal {proposal_order} surface mass")
        if rows["surface_full_probability_json"].nunique(dropna=False) != 1 or rows["surface_full_cdf_upper_json"].nunique(dropna=False) != 1:
            raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface vector split")
        for index, label in zip(expected_indices, expected_labels):
            branch = by_surface.get_group(index)
            if set(branch["surface_class_label"].tolist()) != {label}:
                raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface label changed")
            if not np.allclose(branch["surface_probability"].to_numpy(np.float64), probabilities[index], rtol=0.0, atol=MASS_TOLERANCE):
                raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} surface probability changed")
            expected_lower = 0.0 if index == 0 else float(cdf[index - 1])
            expected_upper = float(cdf[index])
            if not np.allclose(
                branch["surface_cdf_lower"].to_numpy(np.float64),
                expected_lower,
                rtol=0.0,
                atol=MASS_TOLERANCE,
            ) or not np.allclose(
                branch["surface_cdf_upper"].to_numpy(np.float64),
                expected_upper,
                rtol=0.0,
                atol=MASS_TOLERANCE,
            ):
                raise SelectorAuthorityCandidateError(
                    f"Proposal {proposal_order} surface CDF bounds changed"
                )
            if index != 7:
                if len(branch) != 1 or branch.iloc[0]["hazard_resolution_status"] != "none" or int(branch.iloc[0]["hazard_donor_order"]) != -1:
                    raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} non-hazard branch changed")
                _require_close(float(branch.iloc[0]["hazard_probability"]), 1.0, "non-hazard mass")
                if branch.iloc[0]["hazard_donor_key"] != "not_applicable" or int(
                    branch.iloc[0]["stroke_cost"]
                ) != 1:
                    raise SelectorAuthorityCandidateError(
                        f"Proposal {proposal_order} non-hazard sentinels changed"
                    )
            else:
                if len(branch) != MAX_HAZARDS_PER_SURFACE or set(branch["hazard_resolution_status"].tolist()) != {"resolved_complete_atom"}:
                    raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} hazard support changed")
                if not np.array_equal(branch["hazard_donor_order"].to_numpy(), np.arange(MAX_HAZARDS_PER_SURFACE, dtype=np.int64)):
                    raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} hazard order changed")
                if (
                    branch["hazard_donor_key"].isna().any()
                    or not branch["hazard_donor_key"].is_unique
                    or "not_applicable" in set(branch["hazard_donor_key"].tolist())
                    or set(branch["stroke_cost"].tolist()) != {2}
                ):
                    raise SelectorAuthorityCandidateError(
                        f"Proposal {proposal_order} hazard keys/stroke cost changed"
                    )
                lower = branch["hazard_cdf_lower"].to_numpy(np.float64)
                upper = branch["hazard_cdf_upper"].to_numpy(np.float64)
                mass = branch["hazard_probability"].to_numpy(np.float64)
                _require_close(float(lower[0]), 0.0, "hazard CDF start")
                _require_close(float(upper[-1]), 1.0, "hazard CDF end")
                if not np.allclose(lower[1:], upper[:-1], rtol=0.0, atol=MASS_TOLERANCE) or not np.allclose(upper - lower, mass, rtol=0.0, atol=MASS_TOLERANCE):
                    raise SelectorAuthorityCandidateError(f"Proposal {proposal_order} hazard CDF changed")

    group_records: dict[str, tuple[int, int, int, int]] = {}
    if frame.groupby("proposal_row_order")["donor_quality_group_id"].nunique().max() != 1:
        raise SelectorAuthorityCandidateError("A whole proposal was split across donor groups")
    for group_id, rows in frame.groupby("donor_quality_group_id", sort=True):
        kinds = set(rows["donor_ability_value_kind"].tolist())
        group_records[str(group_id)] = (
            len(rows),
            rows["proposal_row_order"].nunique(),
            int(rows["donor_ability_available"].sum()),
            len(kinds),
        )
    if group_records != _GROUP_EXPECTATIONS:
        raise SelectorAuthorityCandidateError("Six-group policy-v2 topology changed")
    availability_pairs = frame.groupby("donor_quality_group_id")["donor_ability_available"].nunique()
    if int((availability_pairs == 2).sum()) != EXPECTED_MIXED_GROUPS:
        raise SelectorAuthorityCandidateError("Mixed observed/imputed groups changed")
    unavailable = frame.loc[~frame["donor_ability_available"]]
    if len(unavailable) != EXPECTED_IMPUTED_ATOMS:
        raise SelectorAuthorityCandidateError("Imputed ability count changed")
    if set(unavailable["donor_ability_value_kind"].tolist()) != {"imputed_frozen_neutral_zero"} or not (unavailable["donor_ability_sg"].to_numpy(np.float64) == 0.0).all():
        raise SelectorAuthorityCandidateError("Policy-v2 neutral imputation changed")
    observed = frame.loc[frame["donor_ability_available"]]
    if set(observed["donor_ability_value_kind"].tolist()) != {"observed_strict_prior"}:
        raise SelectorAuthorityCandidateError("Observed strict-prior policy state changed")

    quality_mass = frame["quality_base_mass"].to_numpy(np.float64)
    if not np.isfinite(quality_mass).all() or (quality_mass <= 0.0).any():
        raise SelectorAuthorityCandidateError("Quality base mass is not finite and positive")
    _require_close(float(quality_mass.sum()), 1.0, "quality base mass")
    factorized = (
        frame["proposal_probability"].to_numpy(np.float64)
        * frame["mechanics_conditional_probability"].to_numpy(np.float64)
    )
    if not np.allclose(quality_mass, factorized, rtol=0.0, atol=MASS_TOLERANCE):
        raise SelectorAuthorityCandidateError(
            "Whole-atom quality/proposal/mechanics factorization changed"
        )
    return {
        "complete_atoms": len(frame),
        "proposals": int(frame["proposal_row_order"].nunique()),
        "spatial_proposals": int(spatial["proposal_row_order"].nunique()),
        "terminal_atoms": len(terminal),
        "surface_records": int(len(surface_rows) // MAX_MATERIALIZED_SURFACE_BRANCHES),
        "materialized_surface_branches": len(surface_rows),
        "hazard_atoms": int((spatial["hazard_resolution_status"] == "resolved_complete_atom").sum()),
        "imputed_ability_atoms": len(unavailable),
        "donor_quality_groups": len(group_records),
        "mixed_policy_groups": int((availability_pairs == 2).sum()),
    }


def _read_and_validate_parts(root: Path, inner_manifest: Mapping[str, object]) -> tuple[list[pa.Table], pd.DataFrame]:
    raw_parts = inner_manifest.get("parts")
    if type(raw_parts) is not list or len(raw_parts) != MAX_PARTS:
        raise SelectorAuthorityCandidateError("Part manifest envelope changed")
    tables: list[pa.Table] = []
    for index, expected in enumerate(_EXPECTED_PARTS):
        entry = raw_parts[index]
        if type(entry) is not dict:
            raise SelectorAuthorityCandidateError(f"Part {index} entry changed")
        relative, rows, _size, _byte_hash, _logical_hash = expected
        _require_exact(entry.get("path"), relative, f"part {index} path")
        path = root / _INNER / PurePosixPath(relative)
        parquet = pq.ParquetFile(path)
        _require_int(parquet.metadata.num_rows, rows, f"part {index} metadata rows")
        _assert_schema(parquet.schema_arrow, f"part {index}")
        # Read the complete 150-column physical row, then derive the 54-column
        # selector view.  Weighting groups and proposals are never partitioned
        # or normalized at a Parquet boundary.
        table = parquet.read()
        _require_int(table.num_rows, rows, f"part {index} loaded rows")
        tables.append(table)
    combined = pa.concat_tables(
        [table.select(list(SEMANTIC_SELECTOR_COLUMNS)) for table in tables],
        promote_options="none",
    )
    if tuple(combined.column_names) != SEMANTIC_SELECTOR_COLUMNS:
        raise SelectorAuthorityCandidateError("Combined semantic projection changed")
    frame = combined.to_pandas()
    frame = frame.loc[:, list(SEMANTIC_SELECTOR_COLUMNS)]
    return tables, frame


def _validation_summary(root: Path) -> dict[str, object]:
    root = Path(root)
    artifact_start = _artifact_closure_state(root, enforce_fixed=True)
    _validate_outer_and_inner_indexes(root)
    _outer_manifest, inner_manifest = _validate_manifests(root)
    tables, frame = _read_and_validate_parts(root, inner_manifest)
    tables, frame = _read_and_validate_parts(root, inner_manifest)
    counts = _validate_frame(frame)
    part_frames = [
        table.select(["proposal_row_order", "donor_quality_group_id"]).to_pandas()
        for table in tables
    ]
    crossing_proposals = sorted(
        proposal
        for proposal in range(MAX_PROPOSALS)
        if sum(
            proposal in set(part["proposal_row_order"].tolist())
            for part in part_frames
        )
        > 1
    )
    if crossing_proposals != [210, 420, 630, 840]:
        raise SelectorAuthorityCandidateError(
            "Cross-part whole-proposal continuity changed"
        )
    group_part_counts = {
        group: sum(
            group in set(part["donor_quality_group_id"].tolist())
            for part in part_frames
        )
        for group in _GROUP_EXPECTATIONS
    }
    expected_cross_part_groups = {
        "03b7d145875af9889ce925a6e93718b1d5cdad75af6187bd91daafa4ffc2d2d8",
        "0625e8c3727ddf1b2539f5b74f486543e2238f22b8915d13e622d1bbff039816",
        "5039b929b7f50dd0d1458855704a44a81366eef05fd92dcebf0d0d80f9d9d2ea",
        "edff72cd2b781856b6faf5833f1b10dea7f1a362683b3c66764c4aca7c0a37ac",
    }
    if {group for group, parts in group_part_counts.items() if parts > 1} != (
        expected_cross_part_groups
    ):
        raise SelectorAuthorityCandidateError(
            "Cross-part donor-group continuity changed"
        )
    schema = pq.ParquetFile(root / _INNER / _EXPECTED_PARTS[0][0]).schema_arrow
    schema_payload = _schema_payload(schema)
    semantic_schema = [item for item in schema_payload if item["name"] in SEMANTIC_SELECTOR_COLUMNS]
    semantic_schema.sort(key=lambda item: SEMANTIC_SELECTOR_COLUMNS.index(item["name"]))
    group_columns = (
        "complete_row_order",
        "proposal_row_order",
        "donor_quality_group_id",
        "donor_ability_available",
        "donor_ability_value_kind",
        "donor_ability_sg",
        "donor_ability_policy_sha256",
    )
    pandas_dtype = {
        "string": "object",
        "double": "float64",
        "bool": "bool",
        "int64": "int64",
        "timestamp[ns, tz=UTC]": "datetime64[ns, UTC]",
    }
    full_pandas_schema = {
        "contract": "lowry_r1h8_real_app_parquet_schema_v1",
        "columns": [
            {
                "order": order,
                "name": name,
                "pandas_dtype": pandas_dtype[dtype],
            }
            for order, (name, dtype) in enumerate(EXPECTED_ARROW_SCHEMA)
        ],
    }
    source_order = {
        name: order for order, (name, _dtype) in enumerate(EXPECTED_ARROW_SCHEMA)
    }
    type_by_name = dict(EXPECTED_ARROW_SCHEMA)
    projection_pandas_schema = {
        "contract": "lowry_r1h8_real_app_selector_projection_schema_v1",
        "columns": [
            {
                "order": order,
                "source_order": source_order[name],
                "name": name,
                "pandas_dtype": pandas_dtype[type_by_name[name]],
            }
            for order, name in enumerate(SEMANTIC_SELECTOR_COLUMNS)
        ],
    }
    full_pandas_schema_sha = _stable_sha(full_pandas_schema)
    projection_pandas_schema_sha = _stable_sha(projection_pandas_schema)
    _require_exact(
        full_pandas_schema_sha,
        "113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d",
        "canonical full schema SHA-256",
    )
    _require_exact(
        projection_pandas_schema_sha,
        "855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2",
        "canonical projection schema SHA-256",
    )
    canonical_full = _canonical_support_sha(
        tables,
        tuple(name for name, _dtype in EXPECTED_ARROW_SCHEMA),
        "lowry_r1h8_real_app_complete_support_full_150_v1",
    )
    canonical_projection = _canonical_support_sha(
        tables,
        SEMANTIC_SELECTOR_COLUMNS,
        "lowry_r1h8_real_app_selector_projection_54_v1",
    )
    canonical_selector = _canonical_support_sha(
        tables,
        SEMANTIC_SELECTOR_COLUMNS,
        "real_shaped_app_logic_fixture_full_contract_digest_v2",
    )
    canonical_topology = _canonical_support_sha(
        tables,
        TOPOLOGY_COLUMNS,
        "lowry_r1h8_real_app_selector_topology_30_v1",
    )
    for observed, expected, label in (
        (
            canonical_full,
            EXPECTED_CANONICAL_FULL_SUPPORT_SHA256,
            "canonical full-150 support",
        ),
        (
            canonical_projection,
            EXPECTED_CANONICAL_PROJECTION_SHA256,
            "canonical selector projection",
        ),
        (
            canonical_selector,
            EXPECTED_CANONICAL_SELECTOR_SUPPORT_SHA256,
            "canonical selector-native support",
        ),
        (
            canonical_topology,
            EXPECTED_CANONICAL_TOPOLOGY_SHA256,
            "canonical four-uniform topology",
        ),
    ):
        _require_exact(observed, expected, label)
    summary = {
        "counts": counts,
        "full_schema_sha256": _stable_sha(schema_payload),
        "full_schema_columns": len(schema_payload),
        "semantic_projection_sha256": _hash_arrow_tables(
            tables, SEMANTIC_SELECTOR_COLUMNS, "real_app_h8_semantic_projection_v1"
        ),
        "semantic_projection_schema_sha256": _stable_sha(semantic_schema),
        "semantic_projection_columns": len(SEMANTIC_SELECTOR_COLUMNS),
        "canonical_full_schema_sha256": full_pandas_schema_sha,
        "canonical_projection_schema_sha256": projection_pandas_schema_sha,
        "canonical_full_support_sha256": canonical_full,
        "canonical_projection_sha256": canonical_projection,
        "canonical_selector_support_sha256": canonical_selector,
        "canonical_topology_sha256": canonical_topology,
        "topology_sha256": _hash_arrow_tables(tables, TOPOLOGY_COLUMNS, "real_app_h8_topology_v1"),
        "group_policy_topology_sha256": _hash_arrow_tables(tables, group_columns, "real_app_h8_group_policy_topology_v1"),
        "cross_part_proposals_sha256": _stable_sha(crossing_proposals),
        "cross_part_group_counts_sha256": _stable_sha(group_part_counts),
        "outer_closure_sha256": artifact_start["summary"]["closure_sha256"],
        "inner_output_index_sha256": EXPECTED_INNER_INDEX_SHA256,
        "outer_output_index_sha256": EXPECTED_OUTER_INDEX_SHA256,
    }
    artifact_end = _artifact_closure_state(root, enforce_fixed=True)
    _assert_exact_json(artifact_start, artifact_end, "artifact start/end rehash")
    return {
        "artifact_sha256_start": artifact_start["summary"],
        "artifact_sha256_end": artifact_end["summary"],
        "validation": summary,
    }


_FAIL_CLOSED_FLAGS = {
    "authority_ready": False,
    "real_support_admitted": False,
    "selector_support_ready": False,
    "weighting_enabled": False,
    "selection_enabled": False,
    "chain_ready": False,
    "anchor_ready": False,
    "simulation_launched": False,
    "mean_solve_launched": False,
    "callback_enabled": False,
    "callback_launched": False,
    "production_enabled": False,
}

_EXPECTED_ORIGINAL_SUMMARY = {
    "bytes": 7_647_733,
    "closure_sha256": "5580dd21763ad5e4eac39057ef7264a4d748313a55033dc8e950ce97bd522f60",
    "directories": 2,
    "directory_state_sha256": "47adb231640339bd8b7bb8d593d3d243ef0f0bc0c832fe232c020caa9478b645",
    "file_state_sha256": "93f52860c532288419b4bb6bb57a78082aa89bb25d2a69eedfe2101d171e17c6",
    "files": 10,
}

_EXPECTED_BUILDER_SOURCE_STATE = {
    "bound_source_files": 16,
    "bound_sources_sha256": "26dfd00bc0d42921c7a383f9df055af05760d8dd026faa54529e0b63e90ed21e",
    "builder_source_sha256": EXPECTED_CANDIDATE_BUILDER_SHA256,
    "source_closure_sha256": "e1783adaa9a03552e72d6e97d29cf97a5d3c720cc0dd22dab9630378ccf28dcc",
}

_EXPECTED_VALIDATION = {
    "canonical_full_schema_sha256": "113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d",
    "canonical_full_support_sha256": EXPECTED_CANONICAL_FULL_SUPPORT_SHA256,
    "canonical_projection_schema_sha256": "855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2",
    "canonical_projection_sha256": EXPECTED_CANONICAL_PROJECTION_SHA256,
    "canonical_selector_support_sha256": EXPECTED_CANONICAL_SELECTOR_SUPPORT_SHA256,
    "canonical_topology_sha256": EXPECTED_CANONICAL_TOPOLOGY_SHA256,
    "counts": {
        "complete_atoms": 35_218,
        "donor_quality_groups": 6,
        "hazard_atoms": 30_702,
        "imputed_ability_atoms": 1_989,
        "materialized_surface_branches": 5_418,
        "mixed_policy_groups": 4,
        "proposals": 904,
        "spatial_proposals": 903,
        "surface_records": 903,
        "terminal_atoms": 1,
    },
    "cross_part_group_counts_sha256": "169d3dbad803cf78cead00ce48e2341fc4be551f9a7ff55b20fb1d4116efe2cd",
    "cross_part_proposals_sha256": "ad429cc5a44d72cdf7ce711e5b9b5299043e3ba9ce049de5b031f12c1d98e623",
    "full_schema_columns": 150,
    "full_schema_sha256": "90f5176de0cbf1c52118a94a47ba35299bcdf9755e77c6350e88bc0342c65855",
    "group_policy_topology_sha256": "b13eeb67a48bffbbc2cc8c728b3767f86bb032c85d26b873307f424bdbb02985",
    "inner_output_index_sha256": EXPECTED_INNER_INDEX_SHA256,
    "outer_closure_sha256": _EXPECTED_ORIGINAL_SUMMARY["closure_sha256"],
    "outer_output_index_sha256": EXPECTED_OUTER_INDEX_SHA256,
    "semantic_projection_columns": 54,
    "semantic_projection_schema_sha256": "83b0b77a1894d6949d9236ca40bafe9fd9cb779d975dfb511933a82f7047d2a0",
    "semantic_projection_sha256": "1fa34f66cc3925ebc7d8ab12f6c8039629dfa3671f9369903a85ee1cf2217a4b",
    "topology_sha256": "7a7f6847c029dd1382084e1bcfc43c8454c1a09e93572a426e1bc19c237df9e2",
}

_EXPECTED_CANDIDATE_FILES = {
    "REPORT.md": (519, EXPECTED_CANDIDATE_REPORT_SHA256),
    "authority_candidate.json": (5_283, EXPECTED_CANDIDATE_JSON_SHA256),
    "manifest.json": (927, EXPECTED_CANDIDATE_MANIFEST_SHA256),
    OUTPUT_INDEX: (474, EXPECTED_CANDIDATE_INDEX_SHA256),
}

_CANDIDATE_FIELDS = {
    "candidate_id",
    "contract",
    "admission_status",
    "authority_id",
    "authority_lock_sha256",
    *_FAIL_CLOSED_FLAGS,
    "research_only",
    "fixed_input_identity",
    "authority_evidence_sha256",
    "source_bindings_sha256",
    "fixed_caps",
    "artifact_sha256_start",
    "artifact_sha256_end",
    "validation",
    "source_sha256_start",
    "source_sha256_end",
    "external_review_requirements",
}

_CANDIDATE_MANIFEST_FIELDS = {
    "contract",
    "candidate_id",
    "admission_status",
    *_FAIL_CLOSED_FLAGS,
    "research_only",
    "payload_files",
}

_AUDIT_REQUIREMENTS = {
    "audit_id_is_not_authority": True,
    "candidate_id_is_not_authority": True,
    "separate_data_only_authority_record_required": True,
    "data_only_authority_record_source_review_required": True,
    "data_only_authority_record_artifact_review_required": True,
}

_AUDIT_FIELDS = {
    "audit_id",
    "contract",
    "decision",
    "audit_status",
    "reviewed_candidate_id",
    "candidate_artifact_review_passed",
    "original_support_review_passed",
    "data_only_authority_record_ready",
    "authority_id",
    "authority_lock_sha256",
    *_FAIL_CLOSED_FLAGS,
    "research_only",
    "candidate_sha256_start",
    "candidate_sha256_end",
    "original_support_sha256_start",
    "original_support_sha256_end",
    "builder_source_sha256_start",
    "builder_source_sha256_end",
    "auditor_source_sha256_start",
    "auditor_source_sha256_end",
    "independent_validation",
    "next_review_requirements",
}

_AUDIT_MANIFEST_FIELDS = {
    "contract",
    "audit_id",
    "decision",
    "audit_status",
    "reviewed_candidate_id",
    "candidate_artifact_review_passed",
    "data_only_authority_record_ready",
    *_FAIL_CLOSED_FLAGS,
    "research_only",
    "payload_files",
}


def _candidate_id(body: Mapping[str, object]) -> str:
    return _stable_sha(
        {
            "contract": CANDIDATE_CONTRACT,
            "role": "review_only_non_authorizing_selector_support_candidate",
            "payload": dict(body),
        }
    )


def _audit_id(body: Mapping[str, object]) -> str:
    return _stable_sha(
        {
            "contract": AUDIT_CONTRACT,
            "role": "detached_candidate_audit_non_authorizing",
            "payload": dict(body),
        }
    )


def _candidate_tree_state(root: Path, *, enforce_fixed: bool) -> dict[str, object]:
    files, directories = _closed_tree(root)
    if directories or set(files) != set(_EXPECTED_CANDIDATE_FILES):
        raise SelectorAuthorityCandidateError("Candidate closure changed")
    file_state = {
        relative: {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(files.items())
    }
    if enforce_fixed:
        for relative, (size, digest) in _EXPECTED_CANDIDATE_FILES.items():
            _require_int(file_state[relative]["bytes"], size, f"candidate {relative} bytes")
            _require_exact(
                file_state[relative]["sha256"],
                digest,
                f"candidate {relative} SHA-256",
            )
    payload = {
        "files": len(file_state),
        "bytes": sum(item["bytes"] for item in file_state.values()),
        "file_state_sha256": _stable_sha(file_state),
    }
    payload["closure_sha256"] = _stable_sha(payload)
    return payload


def _validate_candidate_index(root: Path) -> dict[str, object]:
    _require_exact(
        _sha256_file(root / OUTPUT_INDEX),
        EXPECTED_CANDIDATE_INDEX_SHA256,
        "candidate output-index SHA-256",
    )
    index = _read_json(root / OUTPUT_INDEX)
    expected = {
        "contract": OUTPUT_INDEX_CONTRACT,
        "files": {
            relative: {"bytes": size, "sha256": digest}
            for relative, (size, digest) in _EXPECTED_CANDIDATE_FILES.items()
            if relative != OUTPUT_INDEX
        },
    }
    _assert_exact_json(index, expected, "candidate output index")
    return index


def _assert_candidate(candidate: Mapping[str, object]) -> None:
    if set(candidate) != _CANDIDATE_FIELDS:
        raise SelectorAuthorityCandidateError("Candidate field set changed")
    _require_exact(candidate.get("contract"), CANDIDATE_CONTRACT, "candidate contract")
    _require_exact(candidate.get("admission_status"), ADMISSION_STATUS, "candidate admission status")
    _require_exact(candidate.get("authority_id"), None, "candidate authority id")
    _require_exact(candidate.get("authority_lock_sha256"), None, "candidate authority lock")
    for field, expected in _FAIL_CLOSED_FLAGS.items():
        _require_bool(candidate.get(field), expected, f"candidate {field}")
    _require_bool(candidate.get("research_only"), True, "candidate research_only")
    _assert_exact_json(
        candidate.get("fixed_input_identity"),
        _FIXED_INPUT_IDENTITY_PAYLOAD,
        "candidate fixed-input identity",
    )
    _require_exact(
        candidate.get("authority_evidence_sha256"),
        _stable_sha(EXPECTED_AUTHORITY_EVIDENCE),
        "candidate authority-evidence SHA-256",
    )
    _require_exact(
        candidate.get("source_bindings_sha256"),
        _stable_sha(EXPECTED_SOURCE_BINDINGS),
        "candidate source-bindings SHA-256",
    )
    _assert_exact_json(candidate.get("fixed_caps"), _FIXED_CAPS_PAYLOAD, "candidate fixed caps")
    _assert_exact_json(
        candidate.get("external_review_requirements"),
        _EXTERNAL_REVIEW_REQUIREMENTS,
        "candidate external-review requirements",
    )
    _assert_exact_json(
        candidate.get("artifact_sha256_start"),
        _EXPECTED_ORIGINAL_SUMMARY,
        "candidate original-support start",
    )
    _assert_exact_json(
        candidate.get("artifact_sha256_end"),
        _EXPECTED_ORIGINAL_SUMMARY,
        "candidate original-support end",
    )
    _assert_exact_json(
        candidate.get("source_sha256_start"),
        _EXPECTED_BUILDER_SOURCE_STATE,
        "candidate builder-source start",
    )
    _assert_exact_json(
        candidate.get("source_sha256_end"),
        _EXPECTED_BUILDER_SOURCE_STATE,
        "candidate builder-source end",
    )
    _assert_exact_json(candidate.get("validation"), _EXPECTED_VALIDATION, "candidate validation")
    body = dict(candidate)
    candidate_id = body.pop("candidate_id", None)
    _require_exact(candidate_id, EXPECTED_CANDIDATE_ID, "candidate fixed identity")
    _require_exact(candidate_id, _candidate_id(body), "candidate reproduced identity")


def _validate_candidate_artifact(root: Path) -> dict[str, object]:
    _candidate_tree_state(root, enforce_fixed=True)
    index = _validate_candidate_index(root)
    candidate = _read_json(root / "authority_candidate.json")
    _assert_candidate(candidate)
    manifest = _read_json(root / "manifest.json")
    if set(manifest) != _CANDIDATE_MANIFEST_FIELDS:
        raise SelectorAuthorityCandidateError("Candidate manifest field set changed")
    _require_exact(
        manifest.get("contract"),
        CANDIDATE_MANIFEST_CONTRACT,
        "candidate manifest contract",
    )
    _require_exact(manifest.get("candidate_id"), EXPECTED_CANDIDATE_ID, "manifest candidate identity")
    _require_exact(manifest.get("admission_status"), ADMISSION_STATUS, "manifest admission status")
    for field, expected in _FAIL_CLOSED_FLAGS.items():
        _require_bool(manifest.get(field), expected, f"candidate manifest {field}")
    _require_bool(manifest.get("research_only"), True, "candidate manifest research_only")
    expected_payload = {
        name: metadata
        for name, metadata in index["files"].items()
        if name != "manifest.json"
    }
    _assert_exact_json(
        manifest.get("payload_files"),
        expected_payload,
        "candidate manifest payload files",
    )
    return candidate


def collect_independent_audit() -> dict[str, object]:
    """Re-derive the fixed candidate and original support without trusted validators."""

    candidate_start = _candidate_tree_state(_FIXED_CANDIDATE_ROOT, enforce_fixed=True)
    auditor_source_start = _auditor_source_state()
    builder_source_start = _pinned_builder_source_state()
    _assert_exact_json(
        builder_source_start,
        _EXPECTED_BUILDER_SOURCE_STATE,
        "current candidate-builder source state",
    )
    candidate = _validate_candidate_artifact(_FIXED_CANDIDATE_ROOT)
    original = _validation_summary(_FIXED_SOURCE_ROOT)
    _assert_exact_json(
        original["artifact_sha256_start"],
        _EXPECTED_ORIGINAL_SUMMARY,
        "independent original-support start",
    )
    _assert_exact_json(
        original["artifact_sha256_end"],
        _EXPECTED_ORIGINAL_SUMMARY,
        "independent original-support end",
    )
    _assert_exact_json(
        original["validation"],
        _EXPECTED_VALIDATION,
        "independent source validation",
    )
    _assert_exact_json(
        candidate["artifact_sha256_start"],
        original["artifact_sha256_start"],
        "candidate/independent original-support start",
    )
    _assert_exact_json(
        candidate["artifact_sha256_end"],
        original["artifact_sha256_end"],
        "candidate/independent original-support end",
    )
    _assert_exact_json(
        candidate["validation"],
        original["validation"],
        "candidate/independent validation",
    )
    builder_source_end = _pinned_builder_source_state()
    auditor_source_end = _auditor_source_state()
    candidate_end = _candidate_tree_state(_FIXED_CANDIDATE_ROOT, enforce_fixed=True)
    _assert_exact_json(builder_source_start, builder_source_end, "builder source start/end")
    _assert_exact_json(auditor_source_start, auditor_source_end, "auditor source start/end")
    _assert_exact_json(candidate_start, candidate_end, "candidate tree start/end")
    return {
        "candidate_sha256_start": candidate_start,
        "candidate_sha256_end": candidate_end,
        "original_support_sha256_start": original["artifact_sha256_start"],
        "original_support_sha256_end": original["artifact_sha256_end"],
        "builder_source_sha256_start": builder_source_start,
        "builder_source_sha256_end": builder_source_end,
        "auditor_source_sha256_start": auditor_source_start,
        "auditor_source_sha256_end": auditor_source_end,
        "independent_validation": original["validation"],
    }


def _assert_audit(audit: Mapping[str, object]) -> None:
    if set(audit) != _AUDIT_FIELDS:
        raise SelectorAuthorityCandidateError("Audit field set changed")
    _require_exact(audit.get("contract"), AUDIT_CONTRACT, "audit contract")
    _require_exact(audit.get("decision"), AUDIT_DECISION, "audit decision")
    _require_exact(audit.get("audit_status"), AUDIT_STATUS, "audit status")
    _require_exact(
        audit.get("reviewed_candidate_id"),
        EXPECTED_CANDIDATE_ID,
        "reviewed candidate identity",
    )
    _require_bool(
        audit.get("candidate_artifact_review_passed"),
        True,
        "candidate artifact review result",
    )
    _require_bool(
        audit.get("original_support_review_passed"),
        True,
        "original support review result",
    )
    _require_bool(
        audit.get("data_only_authority_record_ready"),
        False,
        "data-only authority-record readiness",
    )
    _require_exact(audit.get("authority_id"), None, "audit authority id")
    _require_exact(audit.get("authority_lock_sha256"), None, "audit authority lock")
    for field, expected in _FAIL_CLOSED_FLAGS.items():
        _require_bool(audit.get(field), expected, f"audit {field}")
    _require_bool(audit.get("research_only"), True, "audit research_only")
    _assert_exact_json(
        audit.get("candidate_sha256_start"),
        audit.get("candidate_sha256_end"),
        "audit candidate start/end",
    )
    expected_candidate_state = _candidate_tree_state(
        _FIXED_CANDIDATE_ROOT,
        enforce_fixed=True,
    )
    _assert_exact_json(
        audit.get("candidate_sha256_start"),
        expected_candidate_state,
        "audit candidate closure",
    )
    _assert_exact_json(
        audit.get("original_support_sha256_start"),
        _EXPECTED_ORIGINAL_SUMMARY,
        "audit original-support start",
    )
    _assert_exact_json(
        audit.get("original_support_sha256_end"),
        _EXPECTED_ORIGINAL_SUMMARY,
        "audit original-support end",
    )
    _assert_exact_json(
        audit.get("builder_source_sha256_start"),
        _EXPECTED_BUILDER_SOURCE_STATE,
        "audit builder-source start",
    )
    _assert_exact_json(
        audit.get("builder_source_sha256_end"),
        _EXPECTED_BUILDER_SOURCE_STATE,
        "audit builder-source end",
    )
    current_auditor = _auditor_source_state()
    _assert_exact_json(
        audit.get("auditor_source_sha256_start"),
        current_auditor,
        "audit auditor-source start",
    )
    _assert_exact_json(
        audit.get("auditor_source_sha256_end"),
        current_auditor,
        "audit auditor-source end",
    )
    _assert_exact_json(
        audit.get("independent_validation"),
        _EXPECTED_VALIDATION,
        "audit independent validation",
    )
    _assert_exact_json(
        audit.get("next_review_requirements"),
        _AUDIT_REQUIREMENTS,
        "audit next-review requirements",
    )
    body = dict(audit)
    audit_id = body.pop("audit_id", None)
    _require_exact(audit_id, _audit_id(body), "audit reproduced identity")


def _write_output_index(root: Path) -> None:
    files, directories = _closed_tree(root)
    if directories:
        raise SelectorAuthorityCandidateError("Audit output must have a flat closure")
    if OUTPUT_INDEX in files:
        raise FileExistsError(root / OUTPUT_INDEX)
    payload = {
        "contract": OUTPUT_INDEX_CONTRACT,
        "files": {
            relative: {
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for relative, path in sorted(files.items())
        },
    }
    _write_json_exclusive(root / OUTPUT_INDEX, payload)


def _audit_report(audit: Mapping[str, object]) -> str:
    validation = audit["independent_validation"]
    return (
        "# Lowry R1 H8 real APP selector-candidate detached audit\n\n"
        "Status: GO for separate data-only authority-record review only.\n\n"
        f"- Audit ID: {audit['audit_id']}\n"
        f"- Reviewed candidate ID: {audit['reviewed_candidate_id']}\n"
        f"- Complete atoms independently rederived: {validation['counts']['complete_atoms']}\n"
        f"- Canonical selector projection: {validation['canonical_projection_sha256']}\n"
        f"- Canonical topology: {validation['canonical_topology_sha256']}\n"
        "- No authority, admission, selector support, weighting, selection, callback, "
        "simulation, mean solve, chain, anchor, or production path is enabled.\n"
    )


def _audit_tree_state(root: Path) -> dict[str, object]:
    files, directories = _closed_tree(root)
    expected_files = {OUTPUT_INDEX, "REPORT.md", "audit.json", "manifest.json"}
    if directories or set(files) != expected_files:
        raise SelectorAuthorityCandidateError("Audit output closure changed")
    file_state = {
        relative: {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(files.items())
    }
    payload = {
        "files": len(file_state),
        "bytes": sum(item["bytes"] for item in file_state.values()),
        "file_state_sha256": _stable_sha(file_state),
    }
    payload["closure_sha256"] = _stable_sha(payload)
    return payload


def _validate_audit_index(root: Path) -> dict[str, object]:
    files, directories = _closed_tree(root)
    expected_files = {OUTPUT_INDEX, "REPORT.md", "audit.json", "manifest.json"}
    if directories or set(files) != expected_files:
        raise SelectorAuthorityCandidateError("Audit output closure changed")
    index = _read_json(root / OUTPUT_INDEX)
    if set(index) != {"contract", "files"}:
        raise SelectorAuthorityCandidateError("Audit index envelope changed")
    _require_exact(index.get("contract"), OUTPUT_INDEX_CONTRACT, "audit index contract")
    entries = index.get("files")
    if type(entries) is not dict or set(entries) != expected_files - {OUTPUT_INDEX}:
        raise SelectorAuthorityCandidateError("Audit index file set changed")
    for relative, metadata in entries.items():
        if type(metadata) is not dict or set(metadata) != {"bytes", "sha256"}:
            raise SelectorAuthorityCandidateError(f"Audit index metadata changed: {relative}")
        path = root / PurePosixPath(relative)
        _require_int(metadata.get("bytes"), int(path.stat().st_size), f"audit {relative} bytes")
        _require_exact(metadata.get("sha256"), _sha256_file(path), f"audit {relative} SHA-256")
    return index


def validate_audit_artifact(
    root: Path | None = None,
    *,
    deep: bool = True,
) -> dict[str, object]:
    """Validate one audit artifact without granting authority."""

    audit_root = _FIXED_AUDIT_ROOT if root is None else Path(root)
    tree_start = _audit_tree_state(audit_root)
    index = _validate_audit_index(audit_root)
    audit = _read_json(audit_root / "audit.json")
    _assert_audit(audit)
    manifest = _read_json(audit_root / "manifest.json")
    if set(manifest) != _AUDIT_MANIFEST_FIELDS:
        raise SelectorAuthorityCandidateError("Audit manifest field set changed")
    _require_exact(
        manifest.get("contract"),
        AUDIT_MANIFEST_CONTRACT,
        "audit manifest contract",
    )
    for field in (
        "audit_id",
        "decision",
        "audit_status",
        "reviewed_candidate_id",
        "candidate_artifact_review_passed",
        "data_only_authority_record_ready",
    ):
        _require_exact(manifest.get(field), audit.get(field), f"audit manifest {field}")
    for field, expected in _FAIL_CLOSED_FLAGS.items():
        _require_bool(manifest.get(field), expected, f"audit manifest {field}")
    _require_bool(manifest.get("research_only"), True, "audit manifest research_only")
    expected_payload = {
        name: metadata
        for name, metadata in index["files"].items()
        if name != "manifest.json"
    }
    _assert_exact_json(
        manifest.get("payload_files"),
        expected_payload,
        "audit manifest payload files",
    )
    _require_exact(
        (audit_root / "REPORT.md").read_text(encoding="utf-8"),
        _audit_report(audit),
        "audit report",
    )
    if deep:
        current = collect_independent_audit()
        for field, value in current.items():
            _assert_exact_json(audit.get(field), value, f"audit/current {field}")
    tree_end = _audit_tree_state(audit_root)
    _assert_exact_json(tree_start, tree_end, "audit tree start/end")
    return audit


def build_audit_artifact() -> Path:
    """Write only a non-authorizing detached audit at the fixed absent path."""

    if _FIXED_AUDIT_ROOT.exists():
        raise FileExistsError(f"Fixed audit output must be absent: {_FIXED_AUDIT_ROOT}")
    if not _FIXED_AUDIT_ROOT.parent.is_dir():
        raise FileNotFoundError(f"Audit parent is absent: {_FIXED_AUDIT_ROOT.parent}")
    evidence = collect_independent_audit()
    body: dict[str, object] = {
        "contract": AUDIT_CONTRACT,
        "decision": AUDIT_DECISION,
        "audit_status": AUDIT_STATUS,
        "reviewed_candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_artifact_review_passed": True,
        "original_support_review_passed": True,
        "data_only_authority_record_ready": False,
        "authority_id": None,
        "authority_lock_sha256": None,
        **_FAIL_CLOSED_FLAGS,
        "research_only": True,
        **evidence,
        "next_review_requirements": dict(_AUDIT_REQUIREMENTS),
    }
    audit = {"audit_id": _audit_id(body), **body}
    _assert_audit(audit)
    _FIXED_AUDIT_ROOT.mkdir(parents=False, exist_ok=False)
    _write_json_exclusive(_FIXED_AUDIT_ROOT / "audit.json", audit)
    with (_FIXED_AUDIT_ROOT / "REPORT.md").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(_audit_report(audit))
    manifest = {
        "contract": AUDIT_MANIFEST_CONTRACT,
        "audit_id": audit["audit_id"],
        "decision": AUDIT_DECISION,
        "audit_status": AUDIT_STATUS,
        "reviewed_candidate_id": EXPECTED_CANDIDATE_ID,
        "candidate_artifact_review_passed": True,
        "data_only_authority_record_ready": False,
        **_FAIL_CLOSED_FLAGS,
        "research_only": True,
        "payload_files": {
            name: {
                "bytes": int((_FIXED_AUDIT_ROOT / name).stat().st_size),
                "sha256": _sha256_file(_FIXED_AUDIT_ROOT / name),
            }
            for name in ("REPORT.md", "audit.json")
        },
    }
    _write_json_exclusive(_FIXED_AUDIT_ROOT / "manifest.json", manifest)
    _write_output_index(_FIXED_AUDIT_ROOT)
    validate_audit_artifact(_FIXED_AUDIT_ROOT, deep=False)
    post_candidate = _candidate_tree_state(
        _FIXED_CANDIDATE_ROOT,
        enforce_fixed=True,
    )
    _assert_exact_json(
        evidence["candidate_sha256_end"],
        post_candidate,
        "post-write candidate closure",
    )
    post_original = _artifact_closure_state(
        _FIXED_SOURCE_ROOT,
        enforce_fixed=True,
    )["summary"]
    _assert_exact_json(
        evidence["original_support_sha256_end"],
        post_original,
        "post-write original-support closure",
    )
    _assert_exact_json(
        evidence["builder_source_sha256_end"],
        _pinned_builder_source_state(),
        "post-write builder-source closure",
    )
    _assert_exact_json(
        evidence["auditor_source_sha256_end"],
        _auditor_source_state(),
        "post-write auditor-source closure",
    )
    return _FIXED_AUDIT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or audit the one fixed Lowry R1 H8 real-APP selector "
            "candidate. No path, digest, cap, parameter, or runtime override exists."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        result = collect_independent_audit()
        print(json.dumps(result["independent_validation"], indent=2, sort_keys=True))
        print("HOLD: inputs validate; no detached audit artifact was emitted.")
        return
    output = build_audit_artifact()
    audit = _read_json(output / "audit.json")
    print(f"Wrote non-authorizing detached audit: {output}")
    print(f"Audit ID: {audit['audit_id']}")
    print(
        "GO only for separate data-only authority-record review; "
        "all runtime paths remain disabled."
    )


if __name__ == "__main__":
    main()

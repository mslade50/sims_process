"""Archive-only complete TEE/APP transition supports.

The streaming shot chain deliberately caches only the first stage of a
TEE/APP draw: a spatial donor and its endpoint/miss vector.  A quality tilt
cannot be applied to that row because the generated surface lie and any
endpoint-linked penalty/drop resolution are still random.  This module keeps
those mechanics inseparable by expanding one *exact query state* into:

``proposal donor mass x surface probability x hazard-donor probability``.

APP hole-outs are already explicit proposal donors.  They bypass the surface
map, retain their exact empirical mass, and end at the hole with one causal
stroke.  A spatial hazard is expanded through the exact private distribution
used by :class:`hole_sim_hazard_resolution.EndpointLinkedHazardResolver`;
there is no representative or expected penalty shortcut.

This is a research seam, not a simulator.  It neither edits nor is imported
by the frozen streaming/full-chain runners.  Broad persistence is sharded and
bounded: callers supply one support/query at a time and no cross-support
branch table is materialized in memory.  Persisted admission is intentionally
closed to two source-pinned synthetic fixtures; no real-H8 authority has been
minted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import sys
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_batched_population_transition as batched_population  # noqa: E402
import hole_sim_donor_ability_policy as ability_policy  # noqa: E402
import hole_sim_fixed_reference_value as fixed_reference  # noqa: E402
import hole_sim_full_chain as full_chain  # noqa: E402
import hole_sim_hazard_resolution as hazard  # noqa: E402
import hole_sim_normalized_chain_streaming as streaming  # noqa: E402
import hole_sim_streaming_transition_ledger as proposal_ledger  # noqa: E402
import hole_sim_tee_app_integrated_v2 as spatial  # noqa: E402
import hole_sim_tee_transition as tee  # noqa: E402
import hole_sim_transition_mean_chain_adapter as chain_adapter  # noqa: E402


def _freeze_authority_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_authority_tree(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_authority_tree(item) for item in value)
    return value


def _plain_authority_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_authority_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_plain_authority_tree(item) for item in value)
    return value


CONTRACT = "complete_tee_app_physical_atom_support_v3"
QUERY_CONTRACT = "exact_tee_app_generated_start_state_v2"
SELECTION_CONTRACT = "hierarchical_complete_atom_crn_v2"
PERSISTENCE_CONTRACT = "bounded_sharded_complete_atom_persistence_v4"
MASS_TOLERANCE = 3e-13
STATE_TOLERANCE_YD = 2e-8
ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS = 8_192
DEFAULT_MAX_HAZARD_DONORS_PER_SURFACE = ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
DEFAULT_MAX_COMPLETE_ATOMS_PER_PROPOSAL = 16_384
PROPOSAL_SOURCE_INVENTORY_CONTRACT = "ordered_tee_app_proposal_source_inventory_v1"
SURFACE_SOURCE_INVENTORY_CONTRACT = "ordered_tee_app_surface_source_inventory_v1"
HAZARD_SOURCE_INVENTORY_CONTRACT = "ordered_tee_app_hazard_source_inventory_v1"
SOURCE_INVENTORY_CONTRACT = "externally_anchored_tee_app_source_inventory_v1"
SOURCE_INVENTORY_AUTHORITY_CONTRACT = (
    "source_pinned_closed_tee_app_inventory_authority_v1"
)
UNMINTED_SOURCE_INVENTORY_AUTHORITY_ID = "unminted_fail_closed"
_SOURCE_PINNED_AUTHORITY_RECORDS = tuple(
    _freeze_authority_tree(record)
    for record in (
        {
            "authority_id": "synthetic_app_complete_fixture_v1",
            "scope": "synthetic_test_fixture_only",
            "source_inventory_sha256": (
                "ff107a9c3ea4b177355be97b43308143c7a276c0539589c87ae5835af79aa7c5"
            ),
            "queries": (
                {
                    "query_key": (
                        "fa2e8aa472f3a09df003dcbda79222bc58a8d7549770c78e28a6859b7e54985d"
                    ),
                    "proposal_source_inventory_records": 2,
                    "proposal_source_inventory_sha256": (
                        "f54d6f609d92f7c1fe73db3d29781ede6e7ae83e04201e20e4f194aa753238c9"
                    ),
                    "surface_source_inventory_records": 1,
                    "surface_source_inventory_sha256": (
                        "3ea4d0d5c9338c5b42a6e38bf4545ab8bde3ae3d2b91c48f2fbe28ae7d6ed8c7"
                    ),
                    "hazard_source_inventory_records": 0,
                    "hazard_source_inventory_sha256": (
                        "2ff33a10bd618c020905413415fb79616029a1c41dcb05ef3e08f47a7c22bac8"
                    ),
                },
            ),
        },
        {
            "authority_id": "synthetic_ott_hazard_complete_fixture_v1",
            "scope": "synthetic_test_fixture_only",
            "source_inventory_sha256": (
                "dc55008ba2c16a2a90732fc62afe7e8f4ea7303e02914a895b9cebcf4f6d07b5"
            ),
            "queries": (
                {
                    "query_key": (
                        "e783a6a3321a70a49bbb66dd26c69bde1d703a90d4c8e026b50a8ffb4b237ae0"
                    ),
                    "proposal_source_inventory_records": 1,
                    "proposal_source_inventory_sha256": (
                        "545d1d3915db4d3c9679a7c7edbb75c4afdd624938044efcf65e594bd6c04148"
                    ),
                    "surface_source_inventory_records": 1,
                    "surface_source_inventory_sha256": (
                        "ee9db3347a446e981c6472d5cf2945bf8eea232b1d3f2e6b85f77b5062c7dc49"
                    ),
                    "hazard_source_inventory_records": 12,
                    "hazard_source_inventory_sha256": (
                        "246e7105f9b9ac9accfa0874f55a5563a2b3bd485a146ac3eda8b973281415bf"
                    ),
                },
            ),
        },
        {
            "authority_id": "lowry_r1h8_raw_app_exact_query_v1",
            "scope": "lowry_r1h8_raw_app_exact_query_only",
            "review_evidence": {
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
                "candidate_builder_sha256": (
                    "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
                ),
                "reviewed_pre_admission_seam_sha256": (
                    "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
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
                "auditor_source_sha256": (
                    "8d9bb4ae1f3e3c8c0cca875af0aa827afe237545dc657f86f1ed8963787aac06"
                ),
                "auditor_test_sha256": (
                    "aa44f738e85db98a3faf287f553fec7cb9651266961cc1e6fee4c77eef1a350c"
                ),
            },
            "source_inventory_sha256": (
                "37a91ecae53d7945132a9eca7871490250b57f978ec95df4f06e477550c9f92c"
            ),
            "queries": (
                {
                    "query_key": (
                        "943f67faa5e345d395bf706e29c1a526e69879f80e14e81f55e255fd81dbf43e"
                    ),
                    "proposal_source_inventory_records": 904,
                    "proposal_source_inventory_sha256": (
                        "1f3d370647885ac3e9917fc1ebac81a7594b6a11c592b92e86ab65cda362d7a8"
                    ),
                    "surface_source_inventory_records": 903,
                    "surface_source_inventory_sha256": (
                        "2fda69c1b3a666f23d9041023a75e68efe36ed0f68816fde4fa90fd7cbc9ae38"
                    ),
                    "hazard_source_inventory_records": 30702,
                    "hazard_source_inventory_sha256": (
                        "55b324ee433f0e2841e896f0862a5e829c4d40a37358e7fb14fd4d4f93bbb465"
                    ),
                },
            ),
        },
    )
)
_SOURCE_PINNED_AUTHORITY_LOCK_SHA256 = (
    "edbadf6498b1fd5166c6711f0ac20e87bb7d691ec58f7f511f660cd38b978a4c"
)


class CompleteAtomSupportError(ValueError):
    """Raised when a proposal cannot become an honest complete support."""


def _anchor_hold_reason(imputed_ability_atoms: int) -> str:
    return (
        "current_chain_adapter_v3_rejects_policy_v2_imputed_rows;"
        "complete_reachable_support_graph_not_yet_sealed"
        if int(imputed_ability_atoms) > 0
        else "complete_reachable_support_graph_not_yet_sealed"
    )


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _timestamp(value: object, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise CompleteAtomSupportError(f"{label} is not a valid UTC timestamp")
    return pd.Timestamp(parsed)


def _text(value: object, label: str) -> str:
    if value is None or value is pd.NA:
        raise CompleteAtomSupportError(f"{label} is missing")
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        raise CompleteAtomSupportError(f"{label} is blank")
    return text


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompleteAtomSupportError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise CompleteAtomSupportError(f"{label} must be finite")
    return number


def _canonical_scalar(value: object) -> object:
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
            raise CompleteAtomSupportError("Identity hashes cannot contain non-finite floats")
        if number == 0.0:
            number = 0.0
        return {"type": "float", "value": number.hex()}
    return {"type": "str", "value": unicodedata.normalize("NFKC", str(value)).strip()}


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_sha256() -> dict[str, str]:
    """Bind every implementation that participates in complete mechanics."""

    return {
        "proposal_ledger": _sha256_file(Path(proposal_ledger.__file__)),
        "streaming": _sha256_file(Path(streaming.__file__)),
        "full_chain": _sha256_file(Path(full_chain.__file__)),
        "batched_population": _sha256_file(Path(batched_population.__file__)),
        "hazard": _sha256_file(Path(hazard.__file__)),
        "spatial": _sha256_file(Path(spatial.__file__)),
        "tee": _sha256_file(Path(tee.__file__)),
        "app": _sha256_file(Path(app.__file__)),
        "fixed_reference": _sha256_file(Path(fixed_reference.__file__)),
        "ability_policy": _sha256_file(Path(ability_policy.__file__)),
        "chain_adapter": _sha256_file(Path(chain_adapter.__file__)),
    }


def _frame_sha(frame: pd.DataFrame, columns: Sequence[str], contract: str) -> str:
    records = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        records.append(
            {
                str(column): _canonical_scalar(value)
                for column, value in zip(columns, row)
            }
        )
    return _stable_sha({"contract": contract, "records": records})


class _OrderedInventoryHasher:
    """Hash ordered source records without retaining an unbounded record list."""

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
            raise CompleteAtomSupportError(
                f"Source inventory record misses fields={missing}"
            )
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


_PROPOSAL_POLICY_BINDING_COLUMNS = (
    "proposal_support_key",
    "proposal_row_order",
    "proposal_atom_id",
    "proposal_support_measure_sha256",
    *ability_policy.INPUT_COLUMNS,
    *ability_policy.OUTPUT_COLUMNS,
    "donor_ability_snapshot_kind",
)

_PROPOSAL_SOURCE_INVENTORY_COLUMNS = (
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
    *ability_policy.OUTPUT_COLUMNS,
    "donor_ability_snapshot_kind",
    "proposal_policy_binding_sha256",
)

_SURFACE_SOURCE_INVENTORY_COLUMNS = (
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

_HAZARD_SOURCE_INVENTORY_COLUMNS = (
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


def _proposal_policy_binding_sha256(values: Mapping[str, object]) -> str:
    return _stable_sha(
        {
            "contract": "source_bound_tee_app_proposal_policy_bridge_v1",
            "fields": {
                field: _canonical_scalar(values[field])
                for field in _PROPOSAL_POLICY_BINDING_COLUMNS
            },
        }
    )


def _proposal_bridge_row_id(
    support_key: object, proposal_row_order: object, proposal_atom_id: object
) -> str:
    return _stable_sha(
        {
            "contract": "tee_app_proposal_ability_bridge_row_v1",
            "support_key": str(support_key),
            "proposal_row_order": int(proposal_row_order),
            "atom_id": str(proposal_atom_id),
        }
    )


def _source_inventory_sha256(inventory: Sequence[Mapping[str, object]]) -> str:
    records = []
    for row in sorted(inventory, key=lambda item: str(item["query_key"])):
        records.append(
            {
                field: _canonical_scalar(row[field])
                for field in (
                    "query_key",
                    "proposal_source_inventory_records",
                    "proposal_source_inventory_sha256",
                    "surface_source_inventory_records",
                    "surface_source_inventory_sha256",
                    "hazard_source_inventory_records",
                    "hazard_source_inventory_sha256",
                )
            }
        )
    return _stable_sha(
        {"contract": SOURCE_INVENTORY_CONTRACT, "queries": records}
    )


def _source_authority_projection(
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    queries = []
    for row in sorted(inventory, key=lambda item: str(item["query_key"])):
        queries.append(
            {
                field: row[field]
                for field in (
                    "query_key",
                    "proposal_source_inventory_records",
                    "proposal_source_inventory_sha256",
                    "surface_source_inventory_records",
                    "surface_source_inventory_sha256",
                    "hazard_source_inventory_records",
                    "hazard_source_inventory_sha256",
                )
            }
        )
    return {
        "source_inventory_sha256": _source_inventory_sha256(inventory),
        "queries": tuple(queries),
    }


def _closed_source_inventory_authority_records() -> tuple[Mapping[str, object], ...]:
    observed_lock = _stable_sha(
        {
            "contract": SOURCE_INVENTORY_AUTHORITY_CONTRACT,
            "records": _plain_authority_tree(_SOURCE_PINNED_AUTHORITY_RECORDS),
        }
    )
    if observed_lock != _SOURCE_PINNED_AUTHORITY_LOCK_SHA256:
        raise CompleteAtomSupportError(
            "Closed source-inventory authority lock changed"
        )
    identifiers = [str(row["authority_id"]) for row in _SOURCE_PINNED_AUTHORITY_RECORDS]
    if len(identifiers) != len(set(identifiers)):
        raise CompleteAtomSupportError(
            "Closed source-inventory authority identifiers are duplicated"
        )
    return _SOURCE_PINNED_AUTHORITY_RECORDS  # type: ignore[return-value]


def source_inventory_authority_records() -> tuple[Mapping[str, object], ...]:
    """Expose the closed authority as a recursively read-only view."""

    return _closed_source_inventory_authority_records()


def _source_inventory_authority(
    inventory: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    observed = _source_authority_projection(inventory)
    for record in _closed_source_inventory_authority_records():
        expected = {
            "source_inventory_sha256": record["source_inventory_sha256"],
            "queries": record["queries"],
        }
        if observed == expected:
            return record
    return None


@dataclass(frozen=True)
class CompleteAtomQuery:
    """Exact generated start state and target setup for one proposal support."""

    arm: str
    round_no: int
    hole_no: int
    par: int
    target_event_key: str
    target_event_start: object
    target_player_id: str
    start_lie: str
    start_distance_yd: float
    start_long_yd: float
    start_lat_yd: float
    forecast_tee_x: float
    forecast_tee_y: float
    forecast_pin_x: float
    forecast_pin_y: float
    forecast_pin_long_yd: float
    physical_course_id: str

    def validated(self) -> "CompleteAtomQuery":
        arm = _text(self.arm, "query arm")
        if arm not in {"raw", "normalized"}:
            raise CompleteAtomSupportError(f"Unsupported query arm {arm!r}")
        if int(self.round_no) not in {1, 2, 3, 4}:
            raise CompleteAtomSupportError("query round_no must be in 1..4")
        if not 1 <= int(self.hole_no) <= 18:
            raise CompleteAtomSupportError("query hole_no must be in 1..18")
        if int(self.par) not in {3, 4, 5}:
            raise CompleteAtomSupportError("query par must be 3, 4, or 5")
        fixed_reference.canonical_lie(self.start_lie)
        start_distance = _finite(self.start_distance_yd, "query start_distance_yd")
        if start_distance < 0.0:
            raise CompleteAtomSupportError("query start distance cannot be negative")
        for name in (
            "start_long_yd",
            "start_lat_yd",
            "forecast_tee_x",
            "forecast_tee_y",
            "forecast_pin_x",
            "forecast_pin_y",
            "forecast_pin_long_yd",
        ):
            _finite(getattr(self, name), f"query {name}")
        _text(self.target_event_key, "query target_event_key")
        _timestamp(self.target_event_start, "query target_event_start")
        _text(self.target_player_id, "query target_player_id")
        _text(self.physical_course_id, "query physical_course_id")
        return self


def _query_identity_payload(query: CompleteAtomQuery) -> dict[str, object]:
    """Canonicalize the semantic query, independent of caller scalar types."""

    normalized = {
        "arm": _text(query.arm, "query arm"),
        "round_no": int(query.round_no),
        "hole_no": int(query.hole_no),
        "par": int(query.par),
        "target_event_key": _text(query.target_event_key, "query target_event_key"),
        "target_event_start": _timestamp(
            query.target_event_start, "query target_event_start"
        ),
        "target_player_id": _text(
            query.target_player_id, "query target_player_id"
        ),
        "start_lie": fixed_reference.canonical_lie(query.start_lie),
        "start_distance_yd": float(query.start_distance_yd),
        "start_long_yd": float(query.start_long_yd),
        "start_lat_yd": float(query.start_lat_yd),
        "forecast_tee_x": float(query.forecast_tee_x),
        "forecast_tee_y": float(query.forecast_tee_y),
        "forecast_pin_x": float(query.forecast_pin_x),
        "forecast_pin_y": float(query.forecast_pin_y),
        "forecast_pin_long_yd": float(query.forecast_pin_long_yd),
        "physical_course_id": _text(
            query.physical_course_id, "query physical_course_id"
        ),
    }
    return {
        key: _canonical_scalar(value) for key, value in normalized.items()
    }


@dataclass(frozen=True)
class CompleteAtomRequest:
    proposal_atoms: pd.DataFrame
    numeric_support: object
    query: CompleteAtomQuery
    surface_map: object
    hazard_resolver: hazard.EndpointLinkedHazardResolver
    reference_model: fixed_reference.ReferenceValueModel


@dataclass(frozen=True)
class CompleteAtomExpansionAudit:
    query_key: str
    support_key: str
    category: str
    proposal_atoms: int
    complete_atoms: int
    terminal_atoms: int
    hazard_atoms: int
    imputed_ability_atoms: int
    quality_mass: float
    terminal_mass: float
    declared_app_holeout_probability: float
    max_proposal_mass_residual: float
    current_chain_adapter_v3_ability_ready: bool
    required_adapter_contract_bump: bool
    anchor_ready: bool
    anchor_hold_reason: str
    proposal_support_measure_sha256: str
    numeric_payload_sha256: str
    surface_model_sha256: str
    hazard_resolver_sha256: str
    reference_model_sha256: str
    proposal_source_inventory_records: int
    proposal_source_inventory_sha256: str
    surface_source_inventory_records: int
    surface_source_inventory_sha256: str
    hazard_source_inventory_records: int
    hazard_source_inventory_sha256: str


def apply_frozen_policy_to_proposal(proposal: pd.DataFrame) -> pd.DataFrame:
    """Upgrade legacy proposal ability columns to the sealed v2 policy.

    A genuinely absent observation becomes the policy's explicit neutral-zero
    value while its availability flag remains false.  The helper never treats
    an unannotated NaN as usable: downstream expansion requires and revalidates
    every v2 row hash.
    """

    required = {
        "support_key",
        "proposal_row_order",
        "category",
        "donor_player_id",
        "donor_event_key",
        "donor_event_start",
        "target_event_key",
        "target_event_start",
        "donor_skill_date",
        "donor_ability_sg",
        "donor_ability_available",
    }
    missing = sorted(required.difference(proposal.columns))
    if missing:
        raise CompleteAtomSupportError(
            f"Proposal lacks raw ability-policy inputs={missing}"
        )
    out = proposal.copy(deep=True).reset_index(drop=True)
    already = set(ability_policy.OUTPUT_COLUMNS).issubset(out.columns)
    if already and "donor_ability_raw_strict_prior" in out.columns:
        ability_policy.validate_applied_frame(
            out.loc[:, [*ability_policy.INPUT_COLUMNS, *ability_policy.OUTPUT_COLUMNS]]
        )
        return out

    raw_ability = pd.to_numeric(out["donor_ability_sg"], errors="coerce")
    available_values = out["donor_ability_available"].to_numpy(object)
    if not all(isinstance(value, (bool, np.bool_)) for value in available_values):
        raise CompleteAtomSupportError(
            "Legacy donor_ability_available must contain explicit booleans"
        )
    available = np.asarray(available_values, bool)
    if raw_ability[available].isna().any():
        raise CompleteAtomSupportError("Available legacy donor ability is missing")
    if raw_ability[~available].notna().any():
        raise CompleteAtomSupportError(
            "Unavailable legacy donor ability unexpectedly has a numeric value"
        )
    categories = out["category"].astype(str).str.casefold()
    if not categories.isin(ability_policy.CATEGORY_ABILITY_MAPPING).all():
        raise CompleteAtomSupportError("Proposal has an unsupported ability category")
    bridge_ids = [
        _proposal_bridge_row_id(support_key, row_order, atom_id)
        for support_key, row_order, atom_id in out[
            ["support_key", "proposal_row_order", "atom_id"]
        ].itertuples(index=False, name=None)
    ]
    raw = pd.DataFrame(
        {
            "bridge_row_id": bridge_ids,
            "category": categories,
            "donor_player_id": out["donor_player_id"].astype(str),
            "donor_event_key": out["donor_event_key"].astype(str),
            "donor_event_start": out["donor_event_start"],
            "target_event_key": out["target_event_key"].astype(str),
            "target_event_start": out["target_event_start"],
            "donor_skill_date": out["donor_skill_date"],
            "donor_ability_source_column": categories.map(
                ability_policy.CATEGORY_ABILITY_MAPPING
            ),
            "donor_ability_available": available,
            "donor_ability_raw_strict_prior": raw_ability.where(available, None),
        }
    )
    evaluated = ability_policy.apply_frozen_policy(raw)
    ability_policy.validate_applied_frame(evaluated)
    for column in (
        "bridge_row_id",
        "donor_ability_source_column",
        "donor_ability_raw_strict_prior",
        *ability_policy.OUTPUT_COLUMNS,
    ):
        out[column] = evaluated[column].to_numpy()
    out["donor_ability_snapshot_kind"] = np.where(
        out["donor_ability_available"].to_numpy(bool),
        chain_adapter.DONOR_SNAPSHOT_CONTRACT,
        "strict_prior_missing_imputed_frozen_neutral_zero_v2",
    )
    return out


def _validate_policy_rows(proposal: pd.DataFrame) -> None:
    required = [*ability_policy.INPUT_COLUMNS, *ability_policy.OUTPUT_COLUMNS]
    missing = [column for column in required if column not in proposal.columns]
    if missing:
        raise CompleteAtomSupportError(
            "Proposal ability is unsealed; apply the frozen policy first: "
            f"missing={missing}"
        )
    try:
        ability_policy.validate_applied_frame(proposal.loc[:, required])
    except ability_policy.DonorAbilityPolicyError as exc:
        raise CompleteAtomSupportError(
            f"Proposal donor ability policy is invalid: {exc}"
        ) from exc
    applied = pd.to_numeric(proposal["donor_ability_sg"], errors="coerce")
    if not np.isfinite(applied.to_numpy(float)).all():
        raise CompleteAtomSupportError("Applied donor ability is still missing")
    expected_bridge = [
        _proposal_bridge_row_id(support_key, row_order, atom_id)
        for support_key, row_order, atom_id in proposal[
            ["support_key", "proposal_row_order", "atom_id"]
        ].itertuples(index=False, name=None)
    ]
    if not np.array_equal(
        proposal["bridge_row_id"].astype(str).to_numpy(),
        np.asarray(expected_bridge, dtype=object),
    ):
        raise CompleteAtomSupportError(
            "Proposal donor ability bridge is not source-bound"
        )


def _validate_proposal(
    proposal: pd.DataFrame,
    numeric_support: object,
    query: CompleteAtomQuery,
) -> tuple[pd.DataFrame, str, str, float]:
    required = set(proposal_ledger.PROPOSAL_SUPPORT_DIGEST_COLUMNS) | {
        "proposal_support_measure_sha256",
        "donor_event_start",
        "donor_skill_date",
        "donor_player_name",
        "donor_round_no",
        "donor_hole_no",
        "donor_stroke_no",
        "donor_ability_snapshot_kind",
    }
    missing = sorted(required.difference(proposal.columns))
    if missing:
        raise CompleteAtomSupportError(f"Proposal support misses columns={missing}")
    frame = proposal.copy(deep=True).sort_values(
        "proposal_row_order", kind="mergesort"
    ).reset_index(drop=True)
    if frame.empty:
        raise CompleteAtomSupportError("Proposal support is empty")
    support_values = frame["support_key"].astype(str).unique().tolist()
    if len(support_values) != 1:
        raise CompleteAtomSupportError("Expansion accepts exactly one proposal support")
    support_key = _text(support_values[0], "support_key")
    order = pd.to_numeric(frame["proposal_row_order"], errors="coerce")
    if not np.array_equal(order.to_numpy(float), np.arange(len(frame), dtype=float)):
        raise CompleteAtomSupportError("Proposal row order is not complete and contiguous")
    if frame["atom_id"].astype(str).duplicated().any():
        raise CompleteAtomSupportError("Proposal atom identities are duplicated")
    for column in (
        "atom_id",
        "source_shot_key",
        "donor_event_key",
        "donor_player_id",
        "source_cache_sha256",
        "numeric_payload_sha256",
    ):
        for value in frame[column]:
            _text(value, f"proposal {column}")
    if not frame["atom_id"].astype(str).eq(frame["source_shot_key"].astype(str)).all():
        raise CompleteAtomSupportError("Proposal physical shot identity changed")

    declared_hashes = frame["proposal_support_measure_sha256"].astype(str).unique()
    if len(declared_hashes) != 1 or not _is_sha256(declared_hashes[0]):
        raise CompleteAtomSupportError("Proposal measure identity is absent or inconsistent")
    observed_measure = proposal_ledger._frame_digest(  # noqa: SLF001 - frozen seam
        frame, proposal_ledger.PROPOSAL_SUPPORT_DIGEST_COLUMNS
    )
    if observed_measure != str(declared_hashes[0]):
        raise CompleteAtomSupportError("Proposal measure digest changed")

    probability = pd.to_numeric(frame["proposal_probability"], errors="coerce").to_numpy(float)
    mass = pd.to_numeric(frame["quality_base_mass"], errors="coerce").to_numpy(float)
    if not np.isfinite(mass).all() or (mass <= 0.0).any():
        raise CompleteAtomSupportError("Proposal masses must be finite and positive")
    hierarchy = pd.to_numeric(
        frame["base_hierarchy_mass"], errors="coerce"
    ).to_numpy(float)
    model_weight = pd.to_numeric(
        frame["spatial_model_weight"], errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(hierarchy).all()
        or not np.isfinite(model_weight).all()
        or (hierarchy <= 0.0).any()
        or (model_weight <= 0.0).any()
        or not np.allclose(
            mass, hierarchy * model_weight, atol=2e-15, rtol=2e-15
        )
    ):
        raise CompleteAtomSupportError("Proposal hierarchy/model mass factorization changed")
    if not math.isclose(float(mass.sum()), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise CompleteAtomSupportError("Proposal support mass does not sum to one")
    expected_probability = mass / mass.sum()
    if not np.allclose(probability, expected_probability, atol=2e-15, rtol=2e-15):
        raise CompleteAtomSupportError("Proposal probability changed from quality mass")
    upper = np.cumsum(expected_probability)
    upper[-1] = 1.0
    lower = np.concatenate(([0.0], upper[:-1]))
    if not np.allclose(
        frame["proposal_cdf_lower"].to_numpy(float), lower, atol=2e-15, rtol=2e-15
    ) or not np.allclose(
        frame["proposal_cdf_upper"].to_numpy(float), upper, atol=2e-15, rtol=2e-15
    ):
        raise CompleteAtomSupportError("Proposal right-CDF intervals changed")

    for branch_name, branch_frame in frame.groupby(
        "proposal_branch", sort=False, observed=True
    ):
        branch_probability = pd.to_numeric(
            branch_frame["branch_conditional_probability"], errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(branch_probability).all()
            or (branch_probability <= 0.0).any()
            or not math.isclose(
                float(branch_probability.sum()),
                1.0,
                rel_tol=0.0,
                abs_tol=MASS_TOLERANCE,
            )
        ):
            raise CompleteAtomSupportError(
                f"Proposal branch {branch_name!r} has invalid conditional mass"
            )
        branch_upper = np.cumsum(branch_probability)
        branch_upper[-1] = 1.0
        branch_lower = np.concatenate(([0.0], branch_upper[:-1]))
        if not np.array_equal(
            branch_frame["branch_cdf_lower"].to_numpy(float), branch_lower
        ) or not np.array_equal(
            branch_frame["branch_cdf_upper"].to_numpy(float), branch_upper
        ):
            raise CompleteAtomSupportError(
                f"Proposal branch {branch_name!r} frozen right-CDF changed"
            )

    categories = frame["category"].astype(str).str.casefold().unique().tolist()
    if len(categories) != 1 or categories[0] not in {"ott", "app"}:
        raise CompleteAtomSupportError("Proposal category must be exactly OTT or APP")
    category = str(categories[0])
    expected_prefix = "tee" if category == "ott" else "app"
    if not support_key.startswith(f"{expected_prefix}|{query.arm}|"):
        raise CompleteAtomSupportError("Proposal support key/category/query arm disagree")
    expected_bucket = max(1, int(np.rint(float(query.start_distance_yd) / 15.0)))
    expected_key = (
        streaming.StreamingSupportRepository.tee_key(
            query.arm,
            int(query.round_no),
            int(query.hole_no),
            int(query.par),
            str(query.target_player_id),
        )
        if category == "ott"
        else streaming.StreamingSupportRepository.app_key(
            query.arm,
            int(query.round_no),
            int(query.hole_no),
            fixed_reference.canonical_lie(query.start_lie),
            expected_bucket,
            str(query.target_player_id),
        )
    )
    if support_key != expected_key:
        raise CompleteAtomSupportError("Proposal support key is not the exact query key")
    checks = {
        "target_event_key": str(query.target_event_key),
        "target_player_id": str(query.target_player_id),
        "target_round_no": int(query.round_no),
        "target_hole_no": int(query.hole_no),
    }
    for column, expected in checks.items():
        values = frame[column].unique().tolist()
        if len(values) != 1 or str(values[0]) != str(expected):
            raise CompleteAtomSupportError(
                f"Proposal {column}={values!r} disagrees with query {expected!r}"
            )
    proposal_target_start = _timestamp(
        frame["target_event_start"].iloc[0], "proposal target_event_start"
    )
    if proposal_target_start != _timestamp(query.target_event_start, "query target_event_start"):
        raise CompleteAtomSupportError("Proposal and query target-event starts differ")
    _validate_policy_rows(frame)

    if _text(getattr(numeric_support, "key", ""), "numeric support key") != support_key:
        raise CompleteAtomSupportError("Numeric and proposal support keys differ")
    numeric_ids = np.asarray(getattr(numeric_support, "atom_ids", []))
    points = np.asarray(getattr(numeric_support, "points", []), dtype=float)
    numeric_probability = np.asarray(
        getattr(numeric_support, "probability", getattr(numeric_support, "weights", [])),
        dtype=float,
    )
    numeric_cdf = np.asarray(getattr(numeric_support, "cdf", []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise CompleteAtomSupportError("Numeric support endpoints must be finite Nx2")
    spatial_rows = frame.loc[~frame["terminal"].astype(bool)].copy()
    if not np.array_equal(
        numeric_ids.astype(str), spatial_rows["atom_id"].astype(str).to_numpy()
    ):
        raise CompleteAtomSupportError("Numeric endpoint rows no longer align to proposal donors")
    if len(points) != len(spatial_rows) or len(numeric_probability) != len(spatial_rows):
        raise CompleteAtomSupportError("Numeric spatial arrays have inconsistent lengths")
    expected_spatial_probability = spatial_rows["proposal_probability"].to_numpy(float)
    expected_spatial_probability /= expected_spatial_probability.sum()
    if not np.allclose(
        numeric_probability, expected_spatial_probability, atol=2e-15, rtol=2e-15
    ):
        raise CompleteAtomSupportError("Numeric donor probability changed")
    expected_numeric_cdf = np.cumsum(numeric_probability)
    expected_numeric_cdf[-1] = 1.0
    if not np.allclose(numeric_cdf, expected_numeric_cdf, atol=2e-15, rtol=2e-15):
        raise CompleteAtomSupportError("Numeric donor right-CDF changed")
    cache_hashes = frame["source_cache_sha256"].astype(str).unique().tolist()
    payload_hashes = frame["numeric_payload_sha256"].astype(str).unique().tolist()
    numeric_cache = str(getattr(numeric_support, "cache_sha256", ""))
    numeric_payload = str(getattr(numeric_support, "payload_sha256", ""))
    if cache_hashes != [numeric_cache] or not _is_sha256(numeric_cache):
        raise CompleteAtomSupportError("Numeric source-cache provenance changed")
    if payload_hashes != [numeric_payload] or not _is_sha256(numeric_payload):
        raise CompleteAtomSupportError("Numeric payload provenance changed")
    recomputed_payload = streaming.support_payload_sha256(
        support_key,
        numeric_ids,
        points,
        numeric_probability,
        numeric_cdf,
        float(getattr(numeric_support, "holeout_probability")),
    )
    if recomputed_payload != numeric_payload:
        raise CompleteAtomSupportError("Numeric payload hash does not bind its arrays")

    terminal = frame["terminal"].astype(bool).to_numpy()
    holed = frame["holed"].astype(bool).to_numpy()
    if not np.array_equal(terminal, holed):
        raise CompleteAtomSupportError("Proposal terminal and holed flags differ")
    declared_values = frame["declared_app_base_holeout_probability"].to_numpy(float)
    if not np.allclose(declared_values, declared_values[0], atol=0.0, rtol=0.0):
        raise CompleteAtomSupportError("Declared APP holeout mass varies within support")
    declared = float(declared_values[0])
    observed_terminal = float(mass[terminal].sum())
    if category == "ott":
        if terminal.any() or declared != 0.0:
            raise CompleteAtomSupportError("TEE proposal cannot contain terminal mass")
        if not frame["atom_kind"].astype(str).eq("tee_spatial").all():
            raise CompleteAtomSupportError("TEE proposal atom kind changed")
        if not frame["holeout_representation"].astype(str).eq("not_applicable").all():
            raise CompleteAtomSupportError("TEE proposal declares APP holeout semantics")
        if fixed_reference.canonical_lie(query.start_lie) != "tee":
            raise CompleteAtomSupportError("TEE proposal requires an exact tee start state")
    else:
        if not math.isclose(observed_terminal, declared, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
            raise CompleteAtomSupportError("Explicit APP terminal mass changed")
        if not math.isclose(
            float(getattr(numeric_support, "holeout_probability")),
            declared,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CompleteAtomSupportError("Numeric scalar APP holeout declaration changed")
        expected_kinds = np.where(terminal, "app_holeout", "app_spatial")
        if not np.array_equal(frame["atom_kind"].astype(str).to_numpy(), expected_kinds):
            raise CompleteAtomSupportError("APP proposal atom kind/terminal semantics changed")
        if not frame["holeout_representation"].astype(str).eq(
            proposal_ledger.EXPLICIT_APP_HOLEOUT_VERSION
        ).all():
            raise CompleteAtomSupportError("APP proposal is not the explicit holeout measure")
        if not frame["implicit_holeout_probability"].astype(float).eq(0.0).all():
            raise CompleteAtomSupportError("APP proposal retains implicit holeout mass")
        equivalence = frame["app_holeout_equivalence_sha256"].astype(str).unique()
        if len(equivalence) != 1 or not _is_sha256(equivalence[0]):
            raise CompleteAtomSupportError("APP proposal equivalence identity is invalid")
        start_lie = fixed_reference.canonical_lie(query.start_lie)
        if start_lie not in {"tee", "fairway", "rough", "sand", "recovery"}:
            raise CompleteAtomSupportError("APP query start lie is not playable")
        proposal_buckets = frame["target_distance_bucket"].astype(int).unique().tolist()
        if proposal_buckets != [expected_bucket]:
            raise CompleteAtomSupportError(
                "APP proposal bucket does not match the exact generated start distance"
            )

    # The parent TEE/APP spatial selector consumes the numeric support CDF,
    # not a CDF reconstructed from the enriched proposal table.  The two are
    # mathematically equivalent but need not be bit-identical.  Persist the
    # authoritative intervals on every proposal donor so beta-zero replay can
    # use them without another floating-point normalization.
    frame["frozen_spatial_cdf_lower"] = np.nan
    frame["frozen_spatial_cdf_upper"] = np.nan
    numeric_lower = np.concatenate(([0.0], numeric_cdf[:-1]))
    frame.loc[spatial_rows.index, "frozen_spatial_cdf_lower"] = numeric_lower
    frame.loc[spatial_rows.index, "frozen_spatial_cdf_upper"] = numeric_cdf
    frame["frozen_terminal_residual_cdf_lower"] = np.nan
    frame["frozen_terminal_residual_cdf_upper"] = np.nan
    if terminal.any():
        terminal_index = frame.index[terminal]
        frame.loc[terminal_index, "frozen_terminal_residual_cdf_lower"] = frame.loc[
            terminal_index, "branch_cdf_lower"
        ].to_numpy(float)
        frame.loc[terminal_index, "frozen_terminal_residual_cdf_upper"] = frame.loc[
            terminal_index, "branch_cdf_upper"
        ].to_numpy(float)
    return frame, support_key, category, declared


_HAZARD_IDENTITY_COLUMNS = (
    "donor_key",
    "event_key",
    "season",
    "round_no",
    "hole_no",
    "category",
    "causal_endpoint_x",
    "causal_endpoint_y",
    "penalty_count",
    "penalty_stroke_cost",
    "resolution_shot_key",
    "resolution_event_type",
    "resolution_x",
    "resolution_y",
    "resolution_lie",
    "physical_course_id",
    "foundation_version",
    "coordinate_reference_year",
    "coordinate_reference_event_key",
    "coordinate_reference_mode",
    "data_role",
)


def _preflight_hazard_provider_count(
    resolver: object, *, max_donor_count: int
) -> int:
    """Bound the resolver's entire provider before any candidate copy."""

    bound = int(max_donor_count)
    if bound <= 0:
        raise CompleteAtomSupportError("Hazard provider bound must be positive")
    if bound > ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS:
        raise CompleteAtomSupportError(
            "Hazard provider bound exceeds internal absolute ceiling: "
            f"requested={bound}, ceiling={ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS}"
        )
    if not isinstance(resolver, hazard.EndpointLinkedHazardResolver):
        raise CompleteAtomSupportError(
            "Hazard expansion requires the audited EndpointLinkedHazardResolver"
        )
    provider = getattr(resolver, "donors", None)
    if not isinstance(provider, pd.DataFrame):
        raise CompleteAtomSupportError(
            "Hazard resolver lacks the bounded DataFrame provider contract"
        )
    provider_count = int(len(provider))
    if provider_count <= 0:
        raise CompleteAtomSupportError("Hazard donor provider is empty")
    if provider_count > bound:
        raise CompleteAtomSupportError(
            "Hazard donor fanout provider preflight exceeded bound: "
            f"provider={provider_count}, bound={bound}"
        )
    return provider_count


def hazard_resolver_sha256(
    resolver: object,
    *,
    max_donor_count: int = DEFAULT_MAX_HAZARD_DONORS_PER_SURFACE,
) -> str:
    _preflight_hazard_provider_count(
        resolver, max_donor_count=int(max_donor_count)
    )
    if not isinstance(resolver, hazard.EndpointLinkedHazardResolver):
        raise CompleteAtomSupportError(
            "Hazard expansion requires the audited EndpointLinkedHazardResolver"
        )
    donors = resolver.donors.copy().sort_values("donor_key", kind="mergesort").reset_index(drop=True)
    missing = sorted(set(_HAZARD_IDENTITY_COLUMNS).difference(donors.columns))
    if missing:
        raise CompleteAtomSupportError(f"Hazard donor provenance misses columns={missing}")
    if donors.empty or donors["donor_key"].astype(str).duplicated().any():
        raise CompleteAtomSupportError("Hazard donor identity is empty or duplicated")
    donor_sha = _frame_sha(donors, _HAZARD_IDENTITY_COLUMNS, "hazard_donor_panel_v1")
    return _stable_sha(
        {
            "contract": "exact_endpoint_linked_hazard_distribution_v1",
            "resolver_version": hazard.RESOLVER_VERSION,
            "foundation_version": hazard.FOUNDATION_VERSION,
            "donor_panel_sha256": donor_sha,
            "physical_course_id": str(resolver.physical_course_id),
            "coordinate_reference_year": int(resolver.coordinate_reference_year),
            "coordinate_reference_event_key": str(
                resolver.coordinate_reference_event_key
            ),
            "setup_prior": float(resolver.setup_prior).hex(),
            "hole_prior": float(resolver.hole_prior).hex(),
            "spatial_bandwidth_yd": float(resolver.spatial_bandwidth_yd).hex(),
            "max_nearest_distance_yd": float(
                resolver.max_nearest_distance_yd
            ).hex(),
            "same_hole_fallback_max_distance_yd": (
                None
                if resolver.same_hole_fallback_max_distance_yd is None
                else float(resolver.same_hole_fallback_max_distance_yd).hex()
            ),
            "minimum_category_support": int(resolver.minimum_category_support),
        }
    )


def surface_model_sha256(surface_map: object, category: str, hole_no: int) -> str:
    required = ("points", "codes", "prior", "prior_strength")
    missing = [name for name in required if not hasattr(surface_map, name)]
    if missing:
        raise CompleteAtomSupportError(f"Surface model provenance misses fields={missing}")
    points = np.asarray(surface_map.points, dtype=float)
    codes = np.asarray(surface_map.codes, dtype=int)
    prior = np.asarray(surface_map.prior, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(codes):
        raise CompleteAtomSupportError("Surface provenance arrays have invalid shapes")
    if not np.isfinite(points).all() or not np.isfinite(prior).all():
        raise CompleteAtomSupportError("Surface provenance arrays are non-finite")
    classes = tee.LIE_CLASSES if category == "ott" else app.END_LIES
    if prior.shape != (len(classes),) or (prior < 0.0).any():
        raise CompleteAtomSupportError("Surface prior has an invalid class shape")
    if (codes < 0).any() or (codes >= len(classes)).any():
        raise CompleteAtomSupportError("Surface empirical class codes are invalid")
    if not math.isclose(float(prior.sum()), 1.0, rel_tol=0.0, abs_tol=2e-14):
        raise CompleteAtomSupportError("Surface prior mass does not sum to one")
    digest = sha256()
    digest.update(np.ascontiguousarray(points).tobytes())
    digest.update(np.ascontiguousarray(codes).tobytes())
    digest.update(np.ascontiguousarray(prior).tobytes())
    implementation = tee if category == "ott" else app
    return _stable_sha(
        {
            "contract": "empirical_endpoint_surface_model_v2",
            "category": str(category),
            "hole_no": int(hole_no),
            "class_labels": list(classes),
            "prior_strength": float(surface_map.prior_strength).hex(),
            "array_sha256": digest.hexdigest(),
            "surface_type": f"{type(surface_map).__module__}.{type(surface_map).__qualname__}",
            "surface_implementation_sha256": _sha256_file(
                Path(implementation.__file__)
            ),
            "surface_support_mask_sha256": _sha256_file(Path(spatial.__file__)),
        }
    )


def _reference_identity(model: object) -> tuple[str, str]:
    if not isinstance(model, fixed_reference.ReferenceValueModel):
        raise CompleteAtomSupportError(
            "Complete rewards require the frozen ReferenceValueModel"
        )
    model_sha = str(model.model_sha256)
    putting_sha = str(model.putting_curve_sha256)
    if not _is_sha256(model_sha) or not _is_sha256(putting_sha):
        raise CompleteAtomSupportError("Reference model provenance is invalid")
    # Reconstructing the dataclass reruns its content digest check without
    # relying on a mutable caller assertion.
    fixed_reference.ReferenceValueModel(model.table, putting_sha, model_sha)
    return model_sha, putting_sha


def _surface_probabilities(
    *,
    surface_map: object,
    endpoints: np.ndarray,
    category: str,
    query: CompleteAtomQuery,
    resolver: hazard.EndpointLinkedHazardResolver,
) -> np.ndarray:
    frame = full_chain.CourseFrame(
        float(query.forecast_tee_x),
        float(query.forecast_tee_y),
        float(query.forecast_pin_x),
        float(query.forecast_pin_y),
    )
    absolute = frame.local_yards_to_absolute_feet_many(endpoints)

    def supported(points: np.ndarray) -> np.ndarray:
        # The surface helper passes back the same local points.  Recompute the
        # absolute frame here so the predicate remains identical to the parent.
        absolute_points = frame.local_yards_to_absolute_feet_many(points)
        return resolver.has_support_many(
            absolute_points[:, 0],
            absolute_points[:, 1],
            hole_no=int(query.hole_no),
            round_no=int(query.round_no),
            category=category,
        )

    del absolute
    if category == "ott":
        return spatial._supported_surface_probabilities(  # noqa: SLF001
            surface_map.probabilities(endpoints),
            np.asarray(surface_map.codes, int),
            class_count=len(tee.LIE_CLASSES),
            enforce_empirical_class_support=True,
            endpoints=endpoints,
            hazard_class_indices=tuple(
                tee.LIE_CLASSES.index(name) for name in ("water", "other")
            ),
            hazard_support_checker=supported,
        )
    return spatial._supported_surface_probabilities(  # noqa: SLF001
        surface_map.probabilities(endpoints),
        np.asarray(surface_map.codes, int),
        class_count=len(app.END_LIES),
        enforce_empirical_class_support=True,
        endpoints=endpoints,
        hazard_class_indices=(app.END_LIES.index("penalty"),),
        hazard_support_checker=supported,
    )


def _spatial_endpoints(
    numeric_support: object,
    proposal: pd.DataFrame,
    query: CompleteAtomQuery,
    category: str,
) -> dict[int, np.ndarray]:
    points = np.asarray(numeric_support.points, dtype=float)
    spatial_rows = proposal.loc[~proposal["terminal"].astype(bool)]
    if category == "ott":
        endpoints = points.copy()
    else:
        pin = np.asarray([float(query.forecast_pin_long_yd), 0.0])
        start = np.asarray([float(query.start_long_yd), float(query.start_lat_yd)])
        vector = pin - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-8 or not math.isclose(
            length,
            float(query.start_distance_yd),
            rel_tol=0.0,
            abs_tol=STATE_TOLERANCE_YD,
        ):
            raise CompleteAtomSupportError(
                "APP exact start endpoint and remaining distance are inconsistent"
            )
        unit = vector / length
        perpendicular = np.asarray([-unit[1], unit[0]])
        endpoints = (
            pin[None, :]
            + unit[None, :] * points[:, 0, None]
            + perpendicular[None, :] * points[:, 1, None]
        )
    return {
        int(row_order): endpoints[position].copy()
        for position, row_order in enumerate(
            spatial_rows["proposal_row_order"].astype(int).to_numpy()
        )
    }


def _next_category(end_lie: str, end_distance_yd: float, par: int) -> str:
    lie = fixed_reference.canonical_lie(end_lie)
    if lie == "hole":
        return "terminal"
    if lie == "green":
        return "putt"
    if lie == "tee":
        return "app" if int(par) == 3 else "ott"
    return "arg" if float(end_distance_yd) <= 50.0 else "app"


def _next_support_locator(
    *,
    query: CompleteAtomQuery,
    end_state_key: str,
    category: str,
    end_lie: str,
    end_distance_yd: float,
    end_endpoint: np.ndarray,
) -> str:
    if category == "terminal":
        return ""
    endpoint = np.asarray(end_endpoint, dtype=float)
    if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
        raise CompleteAtomSupportError("Child support locator needs one finite endpoint")
    return _stable_sha(
        {
            "contract": "exact_next_transition_locator_v2",
            "arm": query.arm,
            "round_no": int(query.round_no),
            "hole_no": int(query.hole_no),
            "par": int(query.par),
            "target_event_key": str(query.target_event_key),
            "target_event_start": _canonical_scalar(
                _timestamp(query.target_event_start, "query target_event_start")
            ),
            "target_player_id": str(query.target_player_id),
            "physical_course_id": str(query.physical_course_id),
            "forecast_tee_x": _canonical_scalar(float(query.forecast_tee_x)),
            "forecast_tee_y": _canonical_scalar(float(query.forecast_tee_y)),
            "forecast_pin_x": _canonical_scalar(float(query.forecast_pin_x)),
            "forecast_pin_y": _canonical_scalar(float(query.forecast_pin_y)),
            "forecast_pin_long_yd": _canonical_scalar(
                float(query.forecast_pin_long_yd)
            ),
            "category": str(category),
            "start_state_key": str(end_state_key),
            "start_lie": fixed_reference.canonical_lie(end_lie),
            "start_distance_yd": _canonical_scalar(float(end_distance_yd)),
            "start_long_yd": _canonical_scalar(float(endpoint[0])),
            "start_lat_yd": _canonical_scalar(float(endpoint[1])),
        }
    )


def _hazard_distribution(
    *,
    resolver: hazard.EndpointLinkedHazardResolver,
    endpoint_local: np.ndarray,
    query: CompleteAtomQuery,
    category: str,
    max_donor_count: int,
) -> tuple[pd.DataFrame, np.ndarray, Mapping[str, object]]:
    _preflight_hazard_provider_count(
        resolver, max_donor_count=int(max_donor_count)
    )
    frame = full_chain.CourseFrame(
        float(query.forecast_tee_x),
        float(query.forecast_tee_y),
        float(query.forecast_pin_x),
        float(query.forecast_pin_y),
    )
    absolute = frame.local_yards_to_absolute_feet_many(
        np.asarray(endpoint_local, float).reshape(1, 2)
    )[0]
    try:
        (
            candidates,
            weights,
            support,
            component_masses,
            nearest_category,
            nearest_same_hole,
            fallback_used,
            support_source,
            radius_limit,
        ) = resolver._distribution(  # noqa: SLF001 - exact frozen resolver measure
            float(absolute[0]),
            float(absolute[1]),
            int(query.hole_no),
            int(query.round_no),
            str(category),
        )
    except hazard.HazardResolutionUnavailable as exc:
        raise CompleteAtomSupportError(
            "Surface exposed hazard mass without a complete resolver distribution"
        ) from exc
    order = np.argsort(candidates["donor_key"].astype(str).to_numpy())
    candidates = candidates.iloc[order].reset_index(drop=True)
    weights = np.asarray(weights, float)[order]
    if candidates.empty or not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise CompleteAtomSupportError("Hazard resolver returned invalid donor mass")
    if not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE
    ):
        raise CompleteAtomSupportError("Hazard resolver donor mass does not close")
    if candidates["donor_key"].astype(str).duplicated().any():
        raise CompleteAtomSupportError("Hazard resolver donor keys are duplicated")
    return candidates, weights, {
        "support_setup": int(support["setup"]),
        "support_hole": int(support["hole"]),
        "support_category": int(support["category"]),
        "component_masses": ",".join(
            f"{name}={float(value):.17g}"
            for name, value in component_masses.items()
        ),
        "nearest_category_endpoint_distance_yd": float(nearest_category),
        "nearest_same_hole_endpoint_distance_yd": float(nearest_same_hole),
        "fallback_used": bool(fallback_used),
        "support_source": str(support_source),
        "support_radius_limit_yd": float(radius_limit),
        "causal_endpoint_x": float(absolute[0]),
        "causal_endpoint_y": float(absolute[1]),
    }


def _complete_atom_id(payload: Mapping[str, object]) -> str:
    identity_fields = (
        "query_key",
        "category",
        "proposal_atom_id",
        "surface_class_index",
        "surface_class_label",
        "hazard_donor_key",
        "stroke_cost",
        "end_lie",
        "end_distance_yd",
        "end_long_yd",
        "end_lat_yd",
    )
    return _stable_sha(
        {
            "contract": "inseparable_complete_tee_app_atom_identity_v1",
            "fields": {
                field: _canonical_scalar(payload[field]) for field in identity_fields
            },
        }
    )


def _not_applicable_hazard_source_fields() -> dict[str, object]:
    return {
        "hazard_donor_order": -1,
        "hazard_donor_event_key": "not_applicable",
        "hazard_donor_season": -1,
        "hazard_donor_round_no": -1,
        "hazard_donor_hole_no": -1,
        "hazard_donor_category": "not_applicable",
        "hazard_donor_causal_endpoint_x": np.nan,
        "hazard_donor_causal_endpoint_y": np.nan,
        "hazard_penalty_stroke_cost": 0,
        "hazard_resolution_event_type": "not_applicable",
        "hazard_resolution_x": np.nan,
        "hazard_resolution_y": np.nan,
        "hazard_resolution_lie": "not_applicable",
        "hazard_donor_physical_course_id": "not_applicable",
        "hazard_foundation_version": "not_applicable",
        "hazard_coordinate_reference_year": -1,
        "hazard_coordinate_reference_event_key": "not_applicable",
        "hazard_coordinate_reference_mode": "not_applicable",
        "hazard_data_role": "not_applicable",
    }


class CompleteAtomExpansion:
    """Validated one-support expansion that yields bounded row batches."""

    def __init__(
        self,
        request: CompleteAtomRequest,
        *,
        max_batch_rows: int = 8_192,
        max_hazard_donors_per_surface: int = (
            DEFAULT_MAX_HAZARD_DONORS_PER_SURFACE
        ),
        max_complete_atoms_per_proposal: int = (
            DEFAULT_MAX_COMPLETE_ATOMS_PER_PROPOSAL
        ),
    ):
        if int(max_batch_rows) <= 0:
            raise ValueError("max_batch_rows must be positive")
        if int(max_hazard_donors_per_surface) <= 0:
            raise ValueError("max_hazard_donors_per_surface must be positive")
        if (
            int(max_hazard_donors_per_surface)
            > ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
        ):
            raise ValueError(
                "max_hazard_donors_per_surface exceeds internal absolute ceiling"
            )
        if int(max_complete_atoms_per_proposal) <= 0:
            raise ValueError("max_complete_atoms_per_proposal must be positive")
        self.request = request
        self.max_batch_rows = int(max_batch_rows)
        self.max_hazard_donors_per_surface = int(
            max_hazard_donors_per_surface
        )
        self.max_complete_atoms_per_proposal = int(
            max_complete_atoms_per_proposal
        )
        self.query = request.query.validated()
        course_axis_yd = float(
            np.hypot(
                float(self.query.forecast_pin_x) - float(self.query.forecast_tee_x),
                float(self.query.forecast_pin_y) - float(self.query.forecast_tee_y),
            )
            / hazard.FEET_PER_YARD
        )
        if not math.isclose(
            course_axis_yd,
            float(self.query.forecast_pin_long_yd),
            rel_tol=0.0,
            abs_tol=STATE_TOLERANCE_YD,
        ):
            raise CompleteAtomSupportError(
                "Forecast pin-long distance disagrees with the absolute course frame"
            )
        (
            self.proposal,
            self.support_key,
            self.category,
            self.declared_holeout,
        ) = _validate_proposal(
            request.proposal_atoms, request.numeric_support, self.query
        )
        if not isinstance(
            request.hazard_resolver, hazard.EndpointLinkedHazardResolver
        ):
            raise CompleteAtomSupportError(
                "Hazard expansion requires the audited EndpointLinkedHazardResolver"
            )
        if str(request.hazard_resolver.physical_course_id) != str(
            self.query.physical_course_id
        ):
            raise CompleteAtomSupportError(
                "Hazard resolver and exact query use different physical courses"
            )
        _preflight_hazard_provider_count(
            request.hazard_resolver,
            max_donor_count=self.max_hazard_donors_per_surface,
        )
        self.hazard_sha = hazard_resolver_sha256(
            request.hazard_resolver,
            max_donor_count=self.max_hazard_donors_per_surface,
        )
        self.surface_sha = surface_model_sha256(
            request.surface_map, self.category, int(self.query.hole_no)
        )
        self.reference_sha, self.putting_sha = _reference_identity(
            request.reference_model
        )
        self.query_key = _stable_sha(
            {
                "contract": QUERY_CONTRACT,
                "query": _query_identity_payload(self.query),
                "proposal_support_measure_sha256": str(
                    self.proposal["proposal_support_measure_sha256"].iloc[0]
                ),
                "numeric_payload_sha256": str(
                    self.proposal["numeric_payload_sha256"].iloc[0]
                ),
                "surface_model_sha256": self.surface_sha,
                "hazard_resolver_sha256": self.hazard_sha,
                "reference_model_sha256": self.reference_sha,
                "ability_policy_sha256": str(
                    self.proposal["donor_ability_policy_sha256"].iloc[0]
                ),
            }
        )
        self._audit: CompleteAtomExpansionAudit | None = None
        self._iterated = False

    @property
    def audit(self) -> CompleteAtomExpansionAudit:
        if self._audit is None:
            raise RuntimeError("Expansion audit is available only after full iteration")
        return self._audit

    def _base_fields(self, proposal_row: pd.Series) -> dict[str, object]:
        start_lie = fixed_reference.canonical_lie(self.query.start_lie)
        start_distance = float(self.query.start_distance_yd)
        start_value = self.request.reference_model.value(start_lie, start_distance)
        start_key = self.request.reference_model.state_key(start_lie, start_distance)
        donor_event_start = _timestamp(
            proposal_row["donor_event_start"], "donor_event_start"
        )
        target_event_start = _timestamp(
            proposal_row["target_event_start"], "target_event_start"
        )
        if not donor_event_start < target_event_start:
            raise CompleteAtomSupportError(
                "Proposal donor event is not strictly prior to target event"
            )
        donor_skill = pd.to_datetime(
            proposal_row["donor_skill_date"], errors="coerce", utc=True
        )
        if not pd.isna(donor_skill) and not pd.Timestamp(donor_skill) < donor_event_start:
            raise CompleteAtomSupportError(
                "Proposal donor skill date is not strictly prior to donor event"
            )
        fields = {
            "contract": CONTRACT,
            "selection_contract": SELECTION_CONTRACT,
            "query_key": self.query_key,
            "proposal_support_key": self.support_key,
            "support_key": self.query_key,
            "proposal_row_order": int(proposal_row["proposal_row_order"]),
            "proposal_atom_id": str(proposal_row["atom_id"]),
            "category": self.category,
            "arm": self.query.arm,
            "round_no": int(self.query.round_no),
            "hole_no": int(self.query.hole_no),
            "par": int(self.query.par),
            "target_event_key": str(self.query.target_event_key),
            "target_event_start": target_event_start,
            "target_player_id": str(self.query.target_player_id),
            "physical_course_id": str(self.query.physical_course_id),
            "forecast_tee_x": float(self.query.forecast_tee_x),
            "forecast_tee_y": float(self.query.forecast_tee_y),
            "forecast_pin_x": float(self.query.forecast_pin_x),
            "forecast_pin_y": float(self.query.forecast_pin_y),
            "forecast_pin_long_yd": float(self.query.forecast_pin_long_yd),
            "start_lie": start_lie,
            "start_distance_yd": start_distance,
            "start_long_yd": float(self.query.start_long_yd),
            "start_lat_yd": float(self.query.start_lat_yd),
            "start_state_key": start_key,
            "start_reference_value": float(start_value),
            "causal_strokes": 1,
            "proposal_base_hierarchy_mass": float(
                proposal_row["base_hierarchy_mass"]
            ),
            "proposal_spatial_model_weight": float(
                proposal_row["spatial_model_weight"]
            ),
            "proposal_probability": float(proposal_row["proposal_probability"]),
            "proposal_branch": str(proposal_row["proposal_branch"]),
            "proposal_branch_conditional_probability": float(
                proposal_row["branch_conditional_probability"]
            ),
            "proposal_cdf_lower": float(proposal_row["proposal_cdf_lower"]),
            "proposal_cdf_upper": float(proposal_row["proposal_cdf_upper"]),
            "proposal_branch_cdf_lower": float(proposal_row["branch_cdf_lower"]),
            "proposal_branch_cdf_upper": float(proposal_row["branch_cdf_upper"]),
            "frozen_spatial_cdf_lower": (
                np.nan
                if pd.isna(proposal_row["frozen_spatial_cdf_lower"])
                else float(proposal_row["frozen_spatial_cdf_lower"])
            ),
            "frozen_spatial_cdf_upper": (
                np.nan
                if pd.isna(proposal_row["frozen_spatial_cdf_upper"])
                else float(proposal_row["frozen_spatial_cdf_upper"])
            ),
            "frozen_terminal_residual_cdf_lower": (
                np.nan
                if pd.isna(proposal_row["frozen_terminal_residual_cdf_lower"])
                else float(proposal_row["frozen_terminal_residual_cdf_lower"])
            ),
            "frozen_terminal_residual_cdf_upper": (
                np.nan
                if pd.isna(proposal_row["frozen_terminal_residual_cdf_upper"])
                else float(proposal_row["frozen_terminal_residual_cdf_upper"])
            ),
            "donor_event_key": str(proposal_row["donor_event_key"]),
            "donor_player_id": str(proposal_row["donor_player_id"]),
            "donor_player_name": str(proposal_row["donor_player_name"]),
            "donor_round_no": int(proposal_row["donor_round_no"]),
            "donor_hole_no": int(proposal_row["donor_hole_no"]),
            "donor_stroke_no": int(proposal_row["donor_stroke_no"]),
            "donor_event_start": donor_event_start,
            "donor_skill_date": (
                pd.NaT if pd.isna(donor_skill) else pd.Timestamp(donor_skill)
            ),
            "donor_ability_available": bool(
                proposal_row["donor_ability_available"]
            ),
            "bridge_row_id": str(proposal_row["bridge_row_id"]),
            "donor_ability_source_column": str(
                proposal_row["donor_ability_source_column"]
            ),
            "donor_ability_raw_strict_prior": (
                np.nan
                if pd.isna(proposal_row["donor_ability_raw_strict_prior"])
                else float(proposal_row["donor_ability_raw_strict_prior"])
            ),
            "donor_ability_value_kind": str(
                proposal_row["donor_ability_value_kind"]
            ),
            "donor_ability_sg": float(proposal_row["donor_ability_sg"]),
            "donor_ability_was_clipped": bool(
                proposal_row["donor_ability_was_clipped"]
            ),
            "donor_ability_snapshot_kind": str(
                proposal_row["donor_ability_snapshot_kind"]
            ),
            "donor_ability_policy_sha256": str(
                proposal_row["donor_ability_policy_sha256"]
            ),
            "donor_ability_row_sha256": str(
                proposal_row["donor_ability_row_sha256"]
            ),
            "holeout_representation": (
                chain_adapter.APP_HOLEOUT_CONTRACT
                if self.category == "app"
                else "not_applicable"
            ),
            "declared_app_base_holeout_probability": float(
                self.declared_holeout
            ),
            "implicit_holeout_probability": 0.0,
            "proposal_app_holeout_equivalence_sha256": str(
                proposal_row["app_holeout_equivalence_sha256"]
            ),
            "complete_holeout_equivalence_status": (
                "explicit_proposal_mass_preserved"
                if self.category == "app"
                else "not_applicable"
            ),
            "proposal_support_measure_sha256": str(
                proposal_row["proposal_support_measure_sha256"]
            ),
            "numeric_payload_sha256": str(proposal_row["numeric_payload_sha256"]),
            "source_cache_sha256": str(proposal_row["source_cache_sha256"]),
            "surface_model_sha256": self.surface_sha,
            "hazard_resolver_sha256": self.hazard_sha,
            "reference_model_sha256": self.reference_sha,
            "reference_putting_sha256": self.putting_sha,
            "reference_descriptor": self.request.reference_model.descriptor,
            "reference_version": fixed_reference.VERSION,
        }
        fields["proposal_policy_binding_sha256"] = (
            _proposal_policy_binding_sha256(fields)
        )
        return fields

    def _finalize_branch(
        self,
        fields: dict[str, object],
        *,
        mechanics_probability: float,
        end_lie: str,
        end_distance_yd: float,
        end_endpoint: np.ndarray,
        penalty_strokes: int,
        terminal: bool,
        atom_kind: str,
        boundary_only: bool = False,
    ) -> dict[str, object]:
        probability = _finite(mechanics_probability, "mechanics probability")
        boundary_only = bool(boundary_only)
        if probability < 0.0 or (probability == 0.0 and not boundary_only):
            raise CompleteAtomSupportError(
                "Complete mechanics branch has inadmissible mass"
            )
        if probability > 0.0 and boundary_only:
            raise CompleteAtomSupportError(
                "Boundary-only surface atom unexpectedly has positive mass"
            )
        end_lie = fixed_reference.canonical_lie(end_lie)
        end_distance = _finite(end_distance_yd, "complete end distance")
        if end_distance < 0.0:
            raise CompleteAtomSupportError("Complete end distance is negative")
        terminal = bool(terminal)
        if terminal != (end_lie == "hole"):
            raise CompleteAtomSupportError("Terminal complete atom must end at hole")
        if terminal:
            end_distance = 0.0
        penalty = int(penalty_strokes)
        if penalty < 0:
            raise CompleteAtomSupportError("Complete atom penalty cannot be negative")
        stroke_cost = 1 + penalty
        end_value = self.request.reference_model.value(end_lie, end_distance)
        end_key = self.request.reference_model.state_key(end_lie, end_distance)
        reward = (
            float(fields["start_reference_value"])
            - float(stroke_cost)
            - float(end_value)
        )
        next_category = _next_category(end_lie, end_distance, int(self.query.par))
        fields.update(
            {
                "mechanics_conditional_probability": probability,
                "base_hierarchy_mass": float(
                    fields["proposal_base_hierarchy_mass"]
                )
                * probability,
                "spatial_model_weight": float(
                    fields["proposal_spatial_model_weight"]
                ),
                "quality_base_mass": float(fields["proposal_probability"])
                * probability,
                "penalty_strokes": penalty,
                "stroke_cost": stroke_cost,
                "terminal": terminal,
                "holed": terminal,
                "complete_atom": True,
                "boundary_only": boundary_only,
                "atom_kind": str(atom_kind),
                "end_lie": end_lie,
                "end_distance_yd": end_distance,
                "end_long_yd": float(end_endpoint[0]),
                "end_lat_yd": float(end_endpoint[1]),
                "end_state_key": end_key,
                "end_reference_value": float(end_value),
                "fixed_reference_reward": float(reward),
                "quality_reward_available": True,
                "next_category": next_category,
                "next_support_key": _next_support_locator(
                    query=self.query,
                    end_state_key=end_key,
                    category=next_category,
                    end_lie=end_lie,
                    end_distance_yd=end_distance,
                    end_endpoint=end_endpoint,
                ),
            }
        )
        fields["donor_quality_group_id"] = (
            chain_adapter.canonical_donor_quality_group_id(fields)
        )
        fields["atom_id"] = _complete_atom_id(fields)
        return fields

    def _proposal_branches(
        self,
        proposal_row: pd.Series,
        endpoint_lookup: Mapping[int, np.ndarray],
        surface_probability: Mapping[int, np.ndarray],
    ) -> Iterator[dict[str, object]]:
        """Yield one proposal's complete atoms without building a branch list."""

        base = self._base_fields(proposal_row)
        proposal_order = int(proposal_row["proposal_row_order"])

        def raw_branches() -> Iterator[dict[str, object]]:
            emitted = 0

            def bounded(row: dict[str, object]) -> dict[str, object]:
                nonlocal emitted
                emitted += 1
                if emitted > self.max_complete_atoms_per_proposal:
                    raise CompleteAtomSupportError(
                        "Complete-atom proposal fanout exceeded bound: "
                        f"proposal={proposal_order}, bound="
                        f"{self.max_complete_atoms_per_proposal}"
                    )
                return row

            if bool(proposal_row["terminal"]):
                if self.category != "app" or not bool(proposal_row["holed"]):
                    raise CompleteAtomSupportError(
                        "Only explicit APP holeout proposals terminate"
                    )
                pin = np.asarray([float(self.query.forecast_pin_long_yd), 0.0])
                terminal_base = dict(base)
                terminal_base.update(
                    {
                        "proposal_endpoint_long_yd": np.nan,
                        "proposal_endpoint_lat_yd": np.nan,
                        "surface_class_count": 0,
                        "surface_full_probability_json": "[]",
                        "surface_full_cdf_upper_json": "[]",
                        "surface_class_index": -1,
                        "surface_class_label": "explicit_app_holeout",
                        "surface_probability": 1.0,
                        "surface_cdf_lower": 0.0,
                        "surface_cdf_upper": 1.0,
                        "surface_selection_boundary": "not_applicable_terminal",
                        "hazard_donor_key": "not_applicable",
                        "hazard_probability": 1.0,
                        "hazard_cdf_lower": 0.0,
                        "hazard_cdf_upper": 1.0,
                        "hazard_resolution_status": "none",
                        "hazard_penalty_count": 0,
                        "hazard_resolution_shot_key": "not_applicable",
                        "hazard_support_source": "not_applicable",
                        "hazard_fallback_used": False,
                        "hazard_support_setup": 0,
                        "hazard_support_hole": 0,
                        "hazard_support_category": 0,
                        "hazard_component_masses": "not_applicable",
                        "hazard_nearest_category_endpoint_distance_yd": np.nan,
                        "hazard_nearest_same_hole_endpoint_distance_yd": np.nan,
                        "hazard_support_radius_limit_yd": np.nan,
                        "hazard_causal_endpoint_x": np.nan,
                        "hazard_causal_endpoint_y": np.nan,
                        **_not_applicable_hazard_source_fields(),
                    }
                )
                yield bounded(
                    self._finalize_branch(
                        terminal_base,
                        mechanics_probability=1.0,
                        end_lie="hole",
                        end_distance_yd=0.0,
                        end_endpoint=pin,
                        penalty_strokes=0,
                        terminal=True,
                        atom_kind="app_holeout",
                    )
                )
                return

            endpoint = np.asarray(endpoint_lookup[proposal_order], float)
            probabilities = np.asarray(surface_probability[proposal_order], float)
            classes = tee.LIE_CLASSES if self.category == "ott" else app.END_LIES
            mapping = (
                streaming.TEE_SURFACE_TO_LIE
                if self.category == "ott"
                else streaming.APP_SURFACE_TO_LIE
            )
            if probabilities.shape != (len(classes),):
                raise CompleteAtomSupportError("Surface probability vector changed shape")
            if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
                raise CompleteAtomSupportError("Surface probability vector is invalid")
            surface_upper = np.cumsum(probabilities)
            if not math.isclose(
                float(surface_upper[-1]),
                1.0,
                rel_tol=0.0,
                abs_tol=MASS_TOLERANCE,
            ):
                raise CompleteAtomSupportError("Surface probability vector does not close")
            surface_lower = np.concatenate(([0.0], surface_upper[:-1]))
            full_probability_json = json.dumps(
                [float(value) for value in probabilities], separators=(",", ":")
            )
            full_cdf_json = json.dumps(
                [float(value) for value in surface_upper], separators=(",", ":")
            )
            pin_local = np.asarray([float(self.query.forecast_pin_long_yd), 0.0])
            geometric = float(np.linalg.norm(endpoint - pin_local))
            for class_index, surface_mass in enumerate(probabilities):
                # With the parent's strict `u > CDF` rule a zero-mass class is
                # selectable only when it is the leading class and u == 0.
                # Keep that one boundary atom at exact zero mass; all other
                # zero classes can never be selected by a valid uniform.
                boundary_only = bool(class_index == 0 and surface_mass == 0.0)
                if surface_mass == 0.0 and not boundary_only:
                    continue
                surface_label = str(classes[class_index])
                mapped_lie = str(streaming.LIE_NAMES[int(mapping[class_index])])
                common = dict(base)
                common.update(
                    {
                        "proposal_endpoint_long_yd": float(endpoint[0]),
                        "proposal_endpoint_lat_yd": float(endpoint[1]),
                        "surface_class_count": int(len(classes)),
                        "surface_full_probability_json": full_probability_json,
                        "surface_full_cdf_upper_json": full_cdf_json,
                        "surface_class_index": int(class_index),
                        "surface_class_label": surface_label,
                        "surface_probability": float(surface_mass),
                        "surface_cdf_lower": float(surface_lower[class_index]),
                        "surface_cdf_upper": float(surface_upper[class_index]),
                        "surface_selection_boundary": "strict_greater_than_cdf",
                    }
                )
                if mapped_lie != "penalty":
                    common.update(
                        {
                            "hazard_donor_key": "not_applicable",
                            "hazard_probability": 1.0,
                            "hazard_cdf_lower": 0.0,
                            "hazard_cdf_upper": 1.0,
                            "hazard_resolution_status": "none",
                            "hazard_penalty_count": 0,
                            "hazard_resolution_shot_key": "not_applicable",
                            "hazard_support_source": "not_applicable",
                            "hazard_fallback_used": False,
                            "hazard_support_setup": 0,
                            "hazard_support_hole": 0,
                            "hazard_support_category": 0,
                            "hazard_component_masses": "not_applicable",
                            "hazard_nearest_category_endpoint_distance_yd": np.nan,
                            "hazard_nearest_same_hole_endpoint_distance_yd": np.nan,
                            "hazard_support_radius_limit_yd": np.nan,
                            "hazard_causal_endpoint_x": np.nan,
                            "hazard_causal_endpoint_y": np.nan,
                            **_not_applicable_hazard_source_fields(),
                        }
                    )
                    yield bounded(
                        self._finalize_branch(
                            common,
                            mechanics_probability=float(surface_mass),
                            end_lie=mapped_lie,
                            end_distance_yd=geometric,
                            end_endpoint=endpoint,
                            penalty_strokes=0,
                            terminal=False,
                            atom_kind=(
                                "tee_spatial_surface_boundary"
                                if boundary_only and self.category == "ott"
                                else (
                                    "app_spatial_surface_boundary"
                                    if boundary_only
                                    else (
                                        "tee_spatial_surface"
                                        if self.category == "ott"
                                        else "app_spatial_surface"
                                    )
                                )
                            ),
                            boundary_only=boundary_only,
                        )
                    )
                    continue
                if boundary_only:
                    raise CompleteAtomSupportError(
                        "Leading zero-mass hazard class cannot form a complete atom"
                    )

                donors, weights, support = _hazard_distribution(
                    resolver=self.request.hazard_resolver,
                    endpoint_local=endpoint,
                    query=self.query,
                    category=self.category,
                    max_donor_count=self.max_hazard_donors_per_surface,
                )
                if len(donors) > self.max_hazard_donors_per_surface:
                    raise CompleteAtomSupportError(
                        "Hazard donor fanout exceeded bound: "
                        f"proposal={proposal_order}, donors={len(donors)}, "
                        f"bound={self.max_hazard_donors_per_surface}"
                    )
                hazard_upper = np.cumsum(weights)
                hazard_lower = np.concatenate(([0.0], hazard_upper[:-1]))
                course_frame = full_chain.CourseFrame(
                    float(self.query.forecast_tee_x),
                    float(self.query.forecast_tee_y),
                    float(self.query.forecast_pin_x),
                    float(self.query.forecast_pin_y),
                )
                for donor_index, donor in donors.iterrows():
                    resolution_absolute = np.asarray(
                        [float(donor["resolution_x"]), float(donor["resolution_y"])]
                    )
                    delta = resolution_absolute - np.asarray(
                        [course_frame.tee_x, course_frame.tee_y]
                    )
                    resolution_local = np.asarray(
                        [
                            float(delta @ course_frame.unit / hazard.FEET_PER_YARD),
                            float(
                                delta
                                @ course_frame.perpendicular
                                / hazard.FEET_PER_YARD
                            ),
                        ]
                    )
                    resolution_distance = float(
                        np.hypot(
                            float(donor["resolution_x"])
                            - float(self.query.forecast_pin_x),
                            float(donor["resolution_y"])
                            - float(self.query.forecast_pin_y),
                        )
                        / hazard.FEET_PER_YARD
                    )
                    if not 0.0 <= resolution_distance <= 700.0:
                        raise CompleteAtomSupportError(
                            "A hazard donor has an unplayable target-pin distance"
                        )
                    normalized_lie = str(
                        batched_population.normalize_population_lies(
                            np.asarray([donor["resolution_lie"]], dtype=object)
                        )[0]
                    )
                    if normalized_lie not in fixed_reference.NON_GREEN_LIES:
                        raise CompleteAtomSupportError(
                            "Hazard donor does not resolve to a playable non-green state"
                        )
                    penalty_cost = int(donor["penalty_stroke_cost"])
                    penalty_count = int(donor["penalty_count"])
                    if (
                        penalty_cost < 1
                        or penalty_count < 1
                        or penalty_cost < penalty_count
                    ):
                        raise CompleteAtomSupportError(
                            "Hazard donor has incoherent penalty accounting"
                        )
                    branch = dict(common)
                    branch.update(
                        {
                            "hazard_donor_order": int(donor_index),
                            "hazard_donor_key": str(donor["donor_key"]),
                            "hazard_probability": float(weights[donor_index]),
                            "hazard_cdf_lower": float(hazard_lower[donor_index]),
                            "hazard_cdf_upper": float(hazard_upper[donor_index]),
                            "hazard_resolution_status": "resolved_complete_atom",
                            "hazard_penalty_count": penalty_count,
                            "hazard_resolution_shot_key": str(
                                donor["resolution_shot_key"]
                            ),
                            "hazard_donor_event_key": str(donor["event_key"]),
                            "hazard_donor_season": int(donor["season"]),
                            "hazard_donor_round_no": int(donor["round_no"]),
                            "hazard_donor_hole_no": int(donor["hole_no"]),
                            "hazard_donor_category": str(donor["category"]),
                            "hazard_donor_causal_endpoint_x": float(
                                donor["causal_endpoint_x"]
                            ),
                            "hazard_donor_causal_endpoint_y": float(
                                donor["causal_endpoint_y"]
                            ),
                            "hazard_penalty_stroke_cost": penalty_cost,
                            "hazard_resolution_event_type": str(
                                donor["resolution_event_type"]
                            ),
                            "hazard_resolution_x": float(donor["resolution_x"]),
                            "hazard_resolution_y": float(donor["resolution_y"]),
                            "hazard_resolution_lie": str(donor["resolution_lie"]),
                            "hazard_donor_physical_course_id": str(
                                donor["physical_course_id"]
                            ),
                            "hazard_foundation_version": str(
                                donor["foundation_version"]
                            ),
                            "hazard_coordinate_reference_year": int(
                                donor["coordinate_reference_year"]
                            ),
                            "hazard_coordinate_reference_event_key": str(
                                donor["coordinate_reference_event_key"]
                            ),
                            "hazard_coordinate_reference_mode": str(
                                donor["coordinate_reference_mode"]
                            ),
                            "hazard_data_role": str(donor["data_role"]),
                            "hazard_support_source": str(support["support_source"]),
                            "hazard_fallback_used": bool(support["fallback_used"]),
                            "hazard_support_setup": int(support["support_setup"]),
                            "hazard_support_hole": int(support["support_hole"]),
                            "hazard_support_category": int(
                                support["support_category"]
                            ),
                            "hazard_component_masses": str(
                                support["component_masses"]
                            ),
                            "hazard_nearest_category_endpoint_distance_yd": float(
                                support["nearest_category_endpoint_distance_yd"]
                            ),
                            "hazard_nearest_same_hole_endpoint_distance_yd": float(
                                support["nearest_same_hole_endpoint_distance_yd"]
                            ),
                            "hazard_support_radius_limit_yd": float(
                                support["support_radius_limit_yd"]
                            ),
                            "hazard_causal_endpoint_x": float(
                                support["causal_endpoint_x"]
                            ),
                            "hazard_causal_endpoint_y": float(
                                support["causal_endpoint_y"]
                            ),
                        }
                    )
                    yield bounded(
                        self._finalize_branch(
                            branch,
                            mechanics_probability=float(surface_mass)
                            * float(weights[donor_index]),
                            end_lie=normalized_lie,
                            end_distance_yd=resolution_distance,
                            end_endpoint=resolution_local,
                            penalty_strokes=penalty_cost,
                            terminal=False,
                            atom_kind=(
                                "tee_spatial_hazard_resolved"
                                if self.category == "ott"
                                else "app_spatial_hazard_resolved"
                            ),
                        )
                    )

        running_probability = 0.0
        pending: dict[str, object] | None = None
        for branch in raw_branches():
            if pending is not None:
                probability = float(pending["mechanics_conditional_probability"])
                pending["mechanics_cdf_lower"] = float(running_probability)
                running_probability += probability
                pending["mechanics_cdf_upper"] = float(running_probability)
                yield pending
            pending = branch
        if pending is None:
            raise CompleteAtomSupportError(
                f"Proposal row {proposal_order} produced no complete branches"
            )
        probability = float(pending["mechanics_conditional_probability"])
        pending["mechanics_cdf_lower"] = float(running_probability)
        running_probability += probability
        pending["mechanics_cdf_upper"] = 1.0
        yield pending
        if not math.isclose(
            running_probability, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE
        ):
            raise CompleteAtomSupportError(
                "Complete mechanics branches do not close for proposal row "
                f"{proposal_order}"
            )

    def iter_batches(self) -> Iterator[pd.DataFrame]:
        if self._iterated:
            raise RuntimeError("A complete-atom expansion is single-pass")
        self._iterated = True
        endpoint_lookup = _spatial_endpoints(
            self.request.numeric_support,
            self.proposal,
            self.query,
            self.category,
        )
        spatial_orders = sorted(endpoint_lookup)
        if spatial_orders:
            endpoints = np.vstack([endpoint_lookup[index] for index in spatial_orders])
            probabilities = _surface_probabilities(
                surface_map=self.request.surface_map,
                endpoints=endpoints,
                category=self.category,
                query=self.query,
                resolver=self.request.hazard_resolver,
            )
            surface_probability = {
                row_order: probabilities[position]
                for position, row_order in enumerate(spatial_orders)
            }
        else:
            surface_probability = {}

        buffered: list[dict[str, object]] = []
        complete_order = 0
        running_mass = 0.0
        complete_atoms = 0
        terminal_atoms = 0
        hazard_atoms = 0
        imputed_atoms = 0
        max_proposal_residual = 0.0
        proposal_inventory = _OrderedInventoryHasher(
            PROPOSAL_SOURCE_INVENTORY_CONTRACT,
            _PROPOSAL_SOURCE_INVENTORY_COLUMNS,
        )
        surface_inventory = _OrderedInventoryHasher(
            SURFACE_SOURCE_INVENTORY_CONTRACT,
            _SURFACE_SOURCE_INVENTORY_COLUMNS,
        )
        hazard_inventory = _OrderedInventoryHasher(
            HAZARD_SOURCE_INVENTORY_CONTRACT,
            _HAZARD_SOURCE_INVENTORY_COLUMNS,
        )
        for proposal_position, proposal_row in self.proposal.iterrows():
            observed_proposal_mass = 0.0
            branch_count = 0
            for row in self._proposal_branches(
                proposal_row, endpoint_lookup, surface_probability
            ):
                mass = float(row["quality_base_mass"])
                observed_proposal_mass += mass
                row["complete_row_order"] = int(complete_order)
                row["complete_cdf_lower"] = float(running_mass)
                running_mass += mass
                is_last = (
                    proposal_position == len(self.proposal) - 1
                    and math.isclose(
                        float(row["mechanics_cdf_upper"]),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                )
                row["complete_cdf_upper"] = 1.0 if is_last else float(running_mass)
                if branch_count == 0:
                    proposal_inventory.add(row)
                    if not bool(row["terminal"]):
                        surface_inventory.add(row)
                if row["hazard_resolution_status"] == "resolved_complete_atom":
                    hazard_inventory.add(row)
                complete_order += 1
                complete_atoms += 1
                branch_count += 1
                terminal_atoms += int(bool(row["terminal"]))
                hazard_atoms += int(row["hazard_resolution_status"] == "resolved_complete_atom")
                imputed_atoms += int(
                    row["donor_ability_value_kind"]
                    == ability_policy.IMPUTED_VALUE_KIND
                )
                buffered.append(row)
                if len(buffered) >= self.max_batch_rows:
                    batch = pd.DataFrame(buffered)
                    yield batch.reindex(columns=sorted(batch.columns))
                    buffered = []
            if branch_count == 0:
                raise CompleteAtomSupportError(
                    f"Proposal row {proposal_position} produced no branches"
                )
            expected_proposal_mass = float(proposal_row["proposal_probability"])
            max_proposal_residual = max(
                max_proposal_residual,
                abs(observed_proposal_mass - expected_proposal_mass),
            )
            if abs(observed_proposal_mass - expected_proposal_mass) > MASS_TOLERANCE:
                raise CompleteAtomSupportError(
                    f"Complete branches changed proposal mass at row {proposal_position}"
                )
        if buffered:
            batch = pd.DataFrame(buffered)
            yield batch.reindex(columns=sorted(batch.columns))
        if not math.isclose(running_mass, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
            raise CompleteAtomSupportError(
                f"Complete support mass is {running_mass:.17g}, not one"
            )
        terminal_mass = float(
            self.proposal.loc[
                self.proposal["terminal"].astype(bool), "proposal_probability"
            ].sum()
        )
        if self.category == "app" and not math.isclose(
            terminal_mass,
            self.declared_holeout,
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ):
            raise CompleteAtomSupportError("Complete APP holeout mass changed")
        self._audit = CompleteAtomExpansionAudit(
            query_key=self.query_key,
            support_key=self.support_key,
            category=self.category,
            proposal_atoms=int(len(self.proposal)),
            complete_atoms=int(complete_atoms),
            terminal_atoms=int(terminal_atoms),
            hazard_atoms=int(hazard_atoms),
            imputed_ability_atoms=int(imputed_atoms),
            quality_mass=float(running_mass),
            terminal_mass=float(terminal_mass),
            declared_app_holeout_probability=float(self.declared_holeout),
            max_proposal_mass_residual=float(max_proposal_residual),
            current_chain_adapter_v3_ability_ready=bool(
                self.proposal["donor_ability_available"].astype(bool).all()
            ),
            required_adapter_contract_bump=bool(imputed_atoms > 0),
            # This seam is only a conditional TEE/APP support.  It does not
            # claim that all reachable next supports form a sealed graph.
            anchor_ready=False,
            anchor_hold_reason=_anchor_hold_reason(imputed_atoms),
            proposal_support_measure_sha256=str(
                self.proposal["proposal_support_measure_sha256"].iloc[0]
            ),
            numeric_payload_sha256=str(
                self.proposal["numeric_payload_sha256"].iloc[0]
            ),
            surface_model_sha256=self.surface_sha,
            hazard_resolver_sha256=self.hazard_sha,
            reference_model_sha256=self.reference_sha,
            proposal_source_inventory_records=int(proposal_inventory.records),
            proposal_source_inventory_sha256=proposal_inventory.hexdigest(),
            surface_source_inventory_records=int(surface_inventory.records),
            surface_source_inventory_sha256=surface_inventory.hexdigest(),
            hazard_source_inventory_records=int(hazard_inventory.records),
            hazard_source_inventory_sha256=hazard_inventory.hexdigest(),
        )


def _require_uniform(value: float, label: str) -> float:
    uniform = _finite(value, label)
    if not 0.0 <= uniform < 1.0:
        raise CompleteAtomSupportError(f"{label} must satisfy 0 <= u < 1")
    return uniform


def _select_right(mass: np.ndarray, uniform: float) -> int:
    values = np.asarray(mass, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise CompleteAtomSupportError("Selection mass is invalid")
    if (values <= 0.0).any():
        raise CompleteAtomSupportError("Selection mass must be strictly positive")
    values = values / values.sum()
    cdf = np.cumsum(values)
    cdf[-1] = 1.0
    return min(int(np.searchsorted(cdf, uniform, side="right")), len(values) - 1)


def _select_frozen_right(cdf_upper: np.ndarray, uniform: float) -> int:
    """Replay an already-frozen parent right-CDF without normalization."""

    cdf = np.asarray(cdf_upper, float)
    if cdf.ndim != 1 or not len(cdf) or not np.isfinite(cdf).all():
        raise CompleteAtomSupportError("Frozen selection CDF is invalid")
    if (np.diff(cdf) < 0.0).any() or cdf[0] < 0.0:
        raise CompleteAtomSupportError("Frozen selection CDF is not monotone")
    if not math.isclose(float(cdf[-1]), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE):
        raise CompleteAtomSupportError("Frozen selection CDF does not close")
    return min(int(np.searchsorted(cdf, uniform, side="right")), len(cdf) - 1)


def _select_surface(
    mass: np.ndarray, uniform: float, *, normalize: bool
) -> int:
    values = np.asarray(mass, float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or float(values.sum()) <= 0.0
    ):
        raise CompleteAtomSupportError("Surface selection mass is invalid")
    if normalize:
        values = values / values.sum()
    elif not math.isclose(
        float(values.sum()), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE
    ):
        raise CompleteAtomSupportError("Frozen surface mass does not close")
    cdf = np.cumsum(values)
    # Exact parent behavior: count u > cdf.  This differs from proposal and
    # hazard right-CDF selection only when u is exactly a surface boundary.
    return min(int(np.count_nonzero(uniform > cdf)), len(values) - 1)


def _json_float_vector(value: object, label: str) -> np.ndarray:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CompleteAtomSupportError(f"{label} is not valid JSON") from exc
    result = np.asarray(decoded, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise CompleteAtomSupportError(f"{label} is not a finite vector")
    return result


def select_complete_atom_crn(
    complete_atoms: pd.DataFrame,
    *,
    proposal_uniform: float,
    surface_uniform: float,
    hazard_uniform: float,
    holeout_uniform: float,
    probability_column: str = "quality_base_mass",
) -> pd.Series:
    """Select one complete atom with the streaming chain's CRN hierarchy.

    At beta/lambda zero, APP first uses ``holeout_uniform`` for both the
    Bernoulli decision and terminal-donor residual, spatial proposal donors
    use ``proposal_uniform`` with right-CDF semantics, surfaces retain the
    parent's strict-boundary categorical rule, and hazards use their original
    right-CDF uniform.  For tilted whole-atom probabilities the same uniforms
    drive the corresponding Rosenblatt conditionals.
    """

    required = {
        "query_key",
        "category",
        "complete_row_order",
        "proposal_row_order",
        "proposal_probability",
        "terminal",
        "boundary_only",
        "declared_app_base_holeout_probability",
        "frozen_spatial_cdf_upper",
        "frozen_terminal_residual_cdf_upper",
        "surface_class_count",
        "surface_full_probability_json",
        "surface_full_cdf_upper_json",
        "surface_class_index",
        "surface_class_label",
        "surface_probability",
        "hazard_donor_key",
        "hazard_probability",
        "hazard_cdf_upper",
        "hazard_resolution_status",
        probability_column,
    }
    missing = sorted(required.difference(complete_atoms.columns))
    if missing:
        raise CompleteAtomSupportError(f"Complete selection frame misses={missing}")
    frame = complete_atoms.copy().sort_values("complete_row_order", kind="mergesort")
    if frame.empty or frame["query_key"].astype(str).nunique() != 1:
        raise CompleteAtomSupportError("Selection accepts one non-empty complete support")
    probability = pd.to_numeric(frame[probability_column], errors="coerce")
    probability_values = probability.to_numpy(float)
    boundary_only = frame["boundary_only"].astype(bool).to_numpy()
    if (
        not np.isfinite(probability_values).all()
        or (probability_values < 0.0).any()
        or ((probability_values == 0.0) != boundary_only).any()
    ):
        raise CompleteAtomSupportError("Complete selection probabilities are invalid")
    frame["_selection_mass"] = probability_values
    proposal_u = _require_uniform(proposal_uniform, "proposal_uniform")
    surface_u = _require_uniform(surface_uniform, "surface_uniform")
    hazard_u = _require_uniform(hazard_uniform, "hazard_uniform")
    holeout_u = _require_uniform(holeout_uniform, "holeout_uniform")
    categories = frame["category"].astype(str).str.casefold().unique().tolist()
    if len(categories) != 1 or categories[0] not in {"ott", "app"}:
        raise CompleteAtomSupportError("Selection support category is invalid")
    category = categories[0]
    beta_zero = probability_column == "quality_base_mass"
    proposal_rows: list[dict[str, object]] = []
    for proposal_order, rows in frame.groupby(
        "proposal_row_order", sort=True, observed=True
    ):
        terminal_values = rows["terminal"].astype(bool).unique().tolist()
        proposal_values = pd.to_numeric(
            rows["proposal_probability"], errors="coerce"
        ).unique()
        if len(terminal_values) != 1 or len(proposal_values) != 1:
            raise CompleteAtomSupportError(
                "Proposal selection metadata varies across mechanics branches"
            )
        proposal_rows.append(
            {
                "proposal_row_order": int(proposal_order),
                "terminal": bool(terminal_values[0]),
                "mass": (
                    float(proposal_values[0])
                    if beta_zero
                    else float(rows["_selection_mass"].sum())
                ),
                "frozen_spatial_cdf_upper": rows[
                    "frozen_spatial_cdf_upper"
                ].iloc[0],
                "frozen_terminal_residual_cdf_upper": rows[
                    "frozen_terminal_residual_cdf_upper"
                ].iloc[0],
            }
        )
    proposal_mass = pd.DataFrame(proposal_rows).sort_values(
        "proposal_row_order", kind="mergesort"
    )
    masses = proposal_mass["mass"].to_numpy(float)
    if not np.isfinite(masses).all() or (masses <= 0.0).any():
        raise CompleteAtomSupportError("Proposal selection masses are invalid")
    total = float(masses.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise CompleteAtomSupportError("Complete selection mass does not close")
    if category == "app":
        declared_holeout = pd.to_numeric(
            frame["declared_app_base_holeout_probability"], errors="coerce"
        ).unique()
        if len(declared_holeout) != 1 or not math.isfinite(
            float(declared_holeout[0])
        ):
            raise CompleteAtomSupportError("APP holeout declaration varies")
        terminal_mass = (
            float(declared_holeout[0])
            if beta_zero
            else float(
                proposal_mass.loc[proposal_mass["terminal"], "mass"].sum()
                / total
            )
        )
        if not 0.0 <= terminal_mass <= 1.0:
            raise CompleteAtomSupportError("APP holeout probability is invalid")
        if holeout_u < terminal_mass:
            candidates = proposal_mass.loc[proposal_mass["terminal"]].copy().reset_index(
                drop=True
            )
            if candidates.empty or terminal_mass <= 0.0:
                raise CompleteAtomSupportError("Tilted APP holeout branch lacks donors")
            residual = holeout_u / terminal_mass
            if beta_zero:
                terminal_cdf = pd.to_numeric(
                    candidates["frozen_terminal_residual_cdf_upper"],
                    errors="coerce",
                ).to_numpy(float)
                selected = _select_frozen_right(terminal_cdf, residual)
            else:
                selected = _select_right(
                    candidates["mass"].to_numpy(float), residual
                )
            proposal_order = int(candidates.iloc[selected]["proposal_row_order"])
            result = frame.loc[frame["proposal_row_order"].eq(proposal_order)]
            if len(result) != 1 or not bool(result.iloc[0]["terminal"]):
                raise CompleteAtomSupportError("APP terminal proposal is not one whole atom")
            return result.iloc[0].drop(labels="_selection_mass")
        candidates = proposal_mass.loc[~proposal_mass["terminal"]].copy().reset_index(
            drop=True
        )
    else:
        if proposal_mass["terminal"].any():
            raise CompleteAtomSupportError("TEE complete support contains terminal proposals")
        candidates = proposal_mass.reset_index(drop=True)
    if candidates.empty:
        raise CompleteAtomSupportError("Spatial branch lacks proposal donors")
    if beta_zero:
        spatial_cdf = pd.to_numeric(
            candidates["frozen_spatial_cdf_upper"], errors="coerce"
        ).to_numpy(float)
        selected = _select_frozen_right(spatial_cdf, proposal_u)
    else:
        selected = _select_right(candidates["mass"].to_numpy(float), proposal_u)
    proposal_order = int(candidates.iloc[selected]["proposal_row_order"])
    proposal_rows = frame.loc[frame["proposal_row_order"].eq(proposal_order)].copy()
    if beta_zero:
        counts = pd.to_numeric(
            proposal_rows["surface_class_count"], errors="coerce"
        ).unique()
        probability_json = proposal_rows["surface_full_probability_json"].astype(
            str
        ).unique()
        cdf_json = proposal_rows["surface_full_cdf_upper_json"].astype(str).unique()
        if len(counts) != 1 or len(probability_json) != 1 or len(cdf_json) != 1:
            raise CompleteAtomSupportError(
                "Frozen surface selection metadata varies within proposal"
            )
        full_probability = _json_float_vector(
            probability_json[0], "frozen surface probability"
        )
        full_cdf = _json_float_vector(cdf_json[0], "frozen surface CDF")
        if (
            int(counts[0]) != len(full_probability)
            or len(full_probability) != len(full_cdf)
            or not len(full_probability)
            or (full_probability < 0.0).any()
            or not np.array_equal(full_cdf, np.cumsum(full_probability))
            or not math.isclose(
                float(full_cdf[-1]), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE
            )
        ):
            raise CompleteAtomSupportError("Frozen surface CDF is invalid")
        # Parent surface sampling counts strict comparisons, not a
        # searchsorted right-CDF.  In particular, u=0 selects a leading
        # zero-mass class; its explicit boundary-only atom is retained.
        surface_index = min(
            int(np.count_nonzero(surface_u > full_cdf)), len(full_cdf) - 1
        )
    else:
        counts = pd.to_numeric(
            proposal_rows["surface_class_count"], errors="coerce"
        ).unique()
        if len(counts) != 1 or int(counts[0]) <= 0:
            raise CompleteAtomSupportError("Tilted surface class count is invalid")
        surface_mass = np.zeros(int(counts[0]), dtype=float)
        for class_index, rows in proposal_rows.groupby(
            "surface_class_index", sort=True, observed=True
        ):
            index = int(class_index)
            if not 0 <= index < len(surface_mass):
                raise CompleteAtomSupportError("Surface class index is invalid")
            surface_mass[index] = float(rows["_selection_mass"].sum())
        surface_index = _select_surface(surface_mass, surface_u, normalize=True)
    surface_rows = proposal_rows.loc[
        proposal_rows["surface_class_index"].eq(surface_index)
    ].copy().sort_values("complete_row_order", kind="mergesort")
    if surface_rows.empty:
        raise CompleteAtomSupportError(
            "Selected surface class lacks its required complete atom"
        )
    if surface_rows["surface_class_label"].astype(str).nunique() != 1:
        raise CompleteAtomSupportError("Surface class label varies within branch")
    if surface_rows["hazard_resolution_status"].eq("resolved_complete_atom").all():
        if beta_zero:
            selected_hazard = _select_frozen_right(
                pd.to_numeric(
                    surface_rows["hazard_cdf_upper"], errors="coerce"
                ).to_numpy(float),
                hazard_u,
            )
        else:
            selected_hazard = _select_right(
                surface_rows["_selection_mass"].to_numpy(float), hazard_u
            )
        result = surface_rows.iloc[selected_hazard]
    elif surface_rows["hazard_resolution_status"].eq("none").all() and len(surface_rows) == 1:
        result = surface_rows.iloc[0]
    else:
        raise CompleteAtomSupportError("Surface branch mixes incomplete hazard semantics")
    return result.drop(labels="_selection_mass")


def _persisted_frame_sha(frame: pd.DataFrame) -> str:
    columns = tuple(map(str, frame.columns))
    return _frame_sha(frame, columns, "persisted_complete_atom_part_v3")


_OUTPUT_INDEX_NAME = "OUTPUT_SHA256.json"


def _safe_artifact_relative(value: object, label: str) -> str:
    relative = str(value)
    parsed = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative
    ):
        raise CompleteAtomSupportError(f"{label} is not a safe relative path")
    return relative


def _artifact_regular_files(
    output: Path, *, exclude_root_index: bool
) -> dict[str, Path]:
    """Close the tree over regular files, reserving only the root index."""

    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise CompleteAtomSupportError("Persisted artifact root is not a regular directory")
    files: dict[str, Path] = {}
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            raise CompleteAtomSupportError(
                f"Persisted artifact contains a symlink: {relative}"
            )
        if path.name == _OUTPUT_INDEX_NAME and relative != _OUTPUT_INDEX_NAME:
            raise CompleteAtomSupportError(
                f"Persisted artifact contains nested reserved name: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CompleteAtomSupportError(
                f"Persisted artifact contains a non-regular node: {relative}"
            )
        if exclude_root_index and relative == _OUTPUT_INDEX_NAME:
            continue
        files[relative] = path
    return files


def persist_complete_atom_requests(
    requests: Iterable[CompleteAtomRequest],
    output_dir: Path,
    *,
    expansion_batch_rows: int = 8_192,
    max_rows_per_part: int = 32_768,
    max_hazard_donors_per_surface: int = DEFAULT_MAX_HAZARD_DONORS_PER_SURFACE,
    max_complete_atoms_per_proposal: int = DEFAULT_MAX_COMPLETE_ATOMS_PER_PROPOSAL,
) -> dict[str, object]:
    """Persist requests without materializing the cross-support branch table."""

    output = Path(output_dir)
    if (
        int(expansion_batch_rows) <= 0
        or int(max_rows_per_part) <= 0
        or int(max_hazard_donors_per_surface) <= 0
        or int(max_complete_atoms_per_proposal) <= 0
    ):
        raise ValueError("Persistence row bounds must be positive")
    if (
        int(max_hazard_donors_per_surface)
        > ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
    ):
        raise ValueError(
            "Persistence hazard-provider bound exceeds internal absolute ceiling"
        )
    source_sha_start = _sha256_file(Path(__file__))
    dependency_sha_start = _dependency_sha256()
    if output.exists():
        raise FileExistsError(f"Complete-atom output already exists: {output}")
    output.mkdir(parents=True)
    parts_dir = output / "parts"
    parts_dir.mkdir()
    part_rows: list[pd.DataFrame] = []
    buffered_rows = 0
    peak_buffered_rows = 0
    part_index = 0
    part_manifest: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    seen_query_keys: set[str] = set()

    def flush() -> None:
        nonlocal part_rows, buffered_rows, part_index
        if not part_rows:
            return
        combined = pd.concat(part_rows, ignore_index=True, sort=False)
        while len(combined):
            take = min(len(combined), int(max_rows_per_part))
            piece = combined.iloc[:take].copy().reset_index(drop=True)
            combined = combined.iloc[take:].copy().reset_index(drop=True)
            path = parts_dir / f"part-{part_index:06d}.parquet"
            piece.to_parquet(path, index=False)
            persisted = pd.read_parquet(path)
            pd.testing.assert_frame_equal(
                piece,
                persisted,
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
                obj=path.name,
            )
            part_manifest.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "rows": int(len(persisted)),
                    "bytes": int(path.stat().st_size),
                    "byte_sha256": _sha256_file(path),
                    "logical_sha256": _persisted_frame_sha(persisted),
                }
            )
            part_index += 1
        part_rows = []
        buffered_rows = 0

    request_count = 0
    for request in requests:
        expansion = CompleteAtomExpansion(
            request,
            max_batch_rows=int(expansion_batch_rows),
            max_hazard_donors_per_surface=int(max_hazard_donors_per_surface),
            max_complete_atoms_per_proposal=int(max_complete_atoms_per_proposal),
        )
        if expansion.query_key in seen_query_keys:
            raise CompleteAtomSupportError(
                "Persistence received a duplicated exact query request"
            )
        seen_query_keys.add(expansion.query_key)
        for batch in expansion.iter_batches():
            if batch.empty:
                raise CompleteAtomSupportError("Expansion yielded an empty batch")
            start = 0
            while start < len(batch):
                capacity = int(max_rows_per_part) - buffered_rows
                stop = min(len(batch), start + capacity)
                part_rows.append(batch.iloc[start:stop].copy())
                buffered_rows += stop - start
                peak_buffered_rows = max(peak_buffered_rows, buffered_rows)
                start = stop
                if buffered_rows >= int(max_rows_per_part):
                    flush()
        inventory.append(asdict(expansion.audit))
        request_count += 1
    flush()
    if request_count == 0 or not part_manifest:
        raise CompleteAtomSupportError("No complete-atom requests were persisted")
    source_sha_end = _sha256_file(Path(__file__))
    dependency_sha_end = _dependency_sha256()
    if source_sha_start != source_sha_end:
        raise CompleteAtomSupportError(
            "Complete-atom implementation changed during persistence"
        )
    if dependency_sha_start != dependency_sha_end:
        raise CompleteAtomSupportError(
            "A complete-atom dependency changed during persistence"
        )
    source_inventory_sha = _source_inventory_sha256(inventory)
    source_authority = _source_inventory_authority(inventory)
    manifest: dict[str, object] = {
        "contract": PERSISTENCE_CONTRACT,
        "complete_atom_contract": CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "requests": int(request_count),
        "complete_atoms": int(sum(item["complete_atoms"] for item in inventory)),
        "expansion_batch_rows": int(expansion_batch_rows),
        "max_rows_per_part": int(max_rows_per_part),
        "max_hazard_donors_per_surface": int(max_hazard_donors_per_surface),
        "absolute_max_hazard_provider_donors": (
            ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
        ),
        "max_complete_atoms_per_proposal": int(max_complete_atoms_per_proposal),
        "peak_buffered_rows": int(peak_buffered_rows),
        "parts": part_manifest,
        "inventory": inventory,
        "source_inventory_contract": SOURCE_INVENTORY_CONTRACT,
        "source_inventory_sha256": source_inventory_sha,
        "source_inventory_authority_contract": (
            SOURCE_INVENTORY_AUTHORITY_CONTRACT
        ),
        "source_inventory_authority_id": (
            str(source_authority["authority_id"])
            if source_authority is not None
            else UNMINTED_SOURCE_INVENTORY_AUTHORITY_ID
        ),
        "source_inventory_authority_scope": (
            str(source_authority["scope"])
            if source_authority is not None
            else "none"
        ),
        "source_inventory_authority_ready": bool(source_authority is not None),
        "source_inventory_authority_lock_sha256": (
            _SOURCE_PINNED_AUTHORITY_LOCK_SHA256
        ),
        "source_sha256_start": source_sha_start,
        "source_sha256_end": source_sha_end,
        "dependency_sha256_start": dependency_sha_start,
        "dependency_sha256_end": dependency_sha_end,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    artifact_files = _artifact_regular_files(output, exclude_root_index=True)
    output_index = {
        "contract": "non_circular_output_sha256_v1",
        "files": {
            relative: {
                "sha256": _sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for relative, path in sorted(artifact_files.items())
        },
    }
    (output / _OUTPUT_INDEX_NAME).write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_persisted_complete_atom_artifact(
    output_dir: Path,
    *,
    require_current_source: bool = True,
) -> dict[str, object]:
    """Stream-validate a sharded artifact and every policy/physical row.

    Validation holds only the current proposal's scalar accounting state.  It
    therefore remains bounded even when a query spans many Parquet parts.
    Admission comes only from the source-pinned closed authority records in
    this implementation.  Candidate manifests cannot supply an unlock digest.
    """

    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise CompleteAtomSupportError("Persisted artifact root is invalid")
    manifest_path = output / "manifest.json"
    index_path = output / _OUTPUT_INDEX_NAME
    if (
        manifest_path.is_symlink()
        or index_path.is_symlink()
        or not manifest_path.is_file()
        or not index_path.is_file()
    ):
        raise CompleteAtomSupportError("Persisted artifact lacks regular root metadata")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != PERSISTENCE_CONTRACT:
        raise CompleteAtomSupportError("Persisted artifact contract changed")
    if manifest.get("complete_atom_contract") != CONTRACT:
        raise CompleteAtomSupportError("Persisted complete-atom contract changed")
    if manifest.get("production_enabled") is not False:
        raise CompleteAtomSupportError("Research artifact unexpectedly enables production")
    declared_source_inventory = str(manifest.get("source_inventory_sha256", ""))
    if (
        manifest.get("source_inventory_contract") != SOURCE_INVENTORY_CONTRACT
        or not _is_sha256(declared_source_inventory)
    ):
        raise CompleteAtomSupportError("Persisted source inventory attestation is invalid")
    if (
        manifest.get("source_inventory_authority_contract")
        != SOURCE_INVENTORY_AUTHORITY_CONTRACT
        or manifest.get("source_inventory_authority_lock_sha256")
        != _SOURCE_PINNED_AUTHORITY_LOCK_SHA256
    ):
        raise CompleteAtomSupportError(
            "Persisted source-inventory authority identity changed"
        )

    source_start = str(manifest.get("source_sha256_start", ""))
    source_end = str(manifest.get("source_sha256_end", ""))
    dependency_start = manifest.get("dependency_sha256_start")
    dependency_end = manifest.get("dependency_sha256_end")
    if not _is_sha256(source_start) or source_start != source_end:
        raise CompleteAtomSupportError("Complete-atom source was not stable during persistence")
    if (
        not isinstance(dependency_start, dict)
        or not isinstance(dependency_end, dict)
        or dependency_start != dependency_end
        or set(dependency_start) != set(_dependency_sha256())
        or not all(_is_sha256(value) for value in dependency_start.values())
    ):
        raise CompleteAtomSupportError("Complete-atom dependency attestation is invalid")
    if require_current_source:
        if source_end != _sha256_file(Path(__file__)):
            raise CompleteAtomSupportError("Complete-atom source changed after persistence")
        if dependency_end != _dependency_sha256():
            raise CompleteAtomSupportError("Complete-atom dependency changed after persistence")
    try:
        max_part_rows = int(manifest["max_rows_per_part"])
        max_hazard_donors = int(manifest["max_hazard_donors_per_surface"])
        absolute_hazard_ceiling = int(
            manifest["absolute_max_hazard_provider_donors"]
        )
        max_atoms_per_proposal = int(manifest["max_complete_atoms_per_proposal"])
        expansion_batch_rows = int(manifest["expansion_batch_rows"])
        peak_buffered_rows = int(manifest["peak_buffered_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompleteAtomSupportError("Persisted row bounds are malformed") from exc
    if (
        min(
            max_part_rows,
            max_hazard_donors,
            max_atoms_per_proposal,
            expansion_batch_rows,
        )
        <= 0
        or not 0 < peak_buffered_rows <= max_part_rows
    ):
        raise CompleteAtomSupportError("Persisted row bounds are invalid")
    if (
        absolute_hazard_ceiling != ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
        or max_hazard_donors > ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
    ):
        raise CompleteAtomSupportError(
            "Persisted hazard-provider bound exceeds internal absolute ceiling"
        )

    output_index = json.loads(index_path.read_text(encoding="utf-8"))
    if output_index.get("contract") != "non_circular_output_sha256_v1":
        raise CompleteAtomSupportError("Persisted output-index contract changed")
    indexed = output_index.get("files")
    if not isinstance(indexed, dict):
        raise CompleteAtomSupportError("Persisted output index is malformed")
    observed_files = _artifact_regular_files(output, exclude_root_index=True)
    normalized_index: dict[str, Mapping[str, object]] = {}
    for raw_relative, metadata in indexed.items():
        relative = _safe_artifact_relative(raw_relative, "output-index path")
        if relative == _OUTPUT_INDEX_NAME or not isinstance(metadata, Mapping):
            raise CompleteAtomSupportError("Persisted output index is malformed")
        normalized_index[relative] = metadata
    if set(observed_files) != set(normalized_index):
        raise CompleteAtomSupportError("Persisted artifact file closure changed")
    for relative, metadata in normalized_index.items():
        path = observed_files[relative]
        if _sha256_file(path) != str(metadata.get("sha256", "")):
            raise CompleteAtomSupportError(f"Persisted byte hash changed: {relative}")
        try:
            declared_bytes = int(metadata["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompleteAtomSupportError("Persisted output index is malformed") from exc
        if path.stat().st_size != declared_bytes:
            raise CompleteAtomSupportError(f"Persisted byte size changed: {relative}")

    inventory_rows = manifest.get("inventory")
    part_rows = manifest.get("parts")
    if not isinstance(inventory_rows, list) or not inventory_rows:
        raise CompleteAtomSupportError("Persisted support inventory is empty")
    if not isinstance(part_rows, list) or not part_rows:
        raise CompleteAtomSupportError("Persisted part inventory is empty")
    inventory = {str(row["query_key"]): row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise CompleteAtomSupportError("Persisted query inventory is duplicated")
    try:
        declared_requests = int(manifest["requests"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompleteAtomSupportError("Persisted request count is malformed") from exc
    if declared_requests != len(inventory_rows):
        raise CompleteAtomSupportError("Persisted request inventory count changed")
    part_paths: list[str] = []
    for index, part in enumerate(part_rows):
        if not isinstance(part, Mapping):
            raise CompleteAtomSupportError("Persisted part inventory is malformed")
        relative = _safe_artifact_relative(part.get("path"), "part path")
        expected = f"parts/part-{index:06d}.parquet"
        if relative != expected:
            raise CompleteAtomSupportError("Persisted part order/path changed")
        part_paths.append(relative)
    if len(set(part_paths)) != len(part_paths):
        raise CompleteAtomSupportError("Persisted part paths are duplicated")
    if set(observed_files) != {"manifest.json", *part_paths}:
        raise CompleteAtomSupportError("Persisted artifact contains undeclared files")

    required = {
        "contract",
        "selection_contract",
        "query_key",
        "complete_row_order",
        "complete_cdf_lower",
        "complete_cdf_upper",
        "proposal_row_order",
        "proposal_atom_id",
        "proposal_probability",
        "proposal_base_hierarchy_mass",
        "proposal_spatial_model_weight",
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
        "surface_class_count",
        "surface_full_probability_json",
        "surface_full_cdf_upper_json",
        "surface_class_index",
        "surface_class_label",
        "surface_probability",
        "surface_cdf_lower",
        "surface_cdf_upper",
        "hazard_donor_key",
        "hazard_probability",
        "hazard_cdf_lower",
        "hazard_cdf_upper",
        "hazard_resolution_status",
        "hazard_penalty_count",
        "mechanics_conditional_probability",
        "mechanics_cdf_lower",
        "mechanics_cdf_upper",
        "quality_base_mass",
        "base_hierarchy_mass",
        "spatial_model_weight",
        "boundary_only",
        "fixed_reference_reward",
        "start_reference_value",
        "end_reference_value",
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "terminal",
        "holed",
        "end_lie",
        "end_distance_yd",
        "end_long_yd",
        "end_lat_yd",
        "end_state_key",
        "next_category",
        "next_support_key",
        "atom_id",
        "donor_quality_group_id",
        "arm",
        "round_no",
        "hole_no",
        "par",
        "target_event_key",
        "target_event_start",
        "target_player_id",
        "start_lie",
        "start_distance_yd",
        "start_long_yd",
        "start_lat_yd",
        "start_state_key",
        "forecast_tee_x",
        "forecast_tee_y",
        "forecast_pin_x",
        "forecast_pin_y",
        "forecast_pin_long_yd",
        "physical_course_id",
        "declared_app_base_holeout_probability",
        "proposal_support_measure_sha256",
        "numeric_payload_sha256",
        "surface_model_sha256",
        "hazard_resolver_sha256",
        "reference_model_sha256",
        "proposal_policy_binding_sha256",
        "donor_ability_snapshot_kind",
        *_SURFACE_SOURCE_INVENTORY_COLUMNS,
        *_HAZARD_SOURCE_INVENTORY_COLUMNS,
        *ability_policy.INPUT_COLUMNS,
        *ability_policy.OUTPUT_COLUMNS,
    }
    finite_columns = (
        "proposal_probability",
        "proposal_base_hierarchy_mass",
        "proposal_spatial_model_weight",
        "proposal_branch_conditional_probability",
        "proposal_cdf_lower",
        "proposal_cdf_upper",
        "proposal_branch_cdf_lower",
        "proposal_branch_cdf_upper",
        "surface_probability",
        "surface_cdf_lower",
        "surface_cdf_upper",
        "hazard_probability",
        "hazard_cdf_lower",
        "hazard_cdf_upper",
        "mechanics_conditional_probability",
        "mechanics_cdf_lower",
        "mechanics_cdf_upper",
        "quality_base_mass",
        "base_hierarchy_mass",
        "spatial_model_weight",
        "complete_cdf_lower",
        "complete_cdf_upper",
        "start_reference_value",
        "end_reference_value",
        "fixed_reference_reward",
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "end_distance_yd",
        "end_long_yd",
        "end_lat_yd",
    )
    query_identity_columns = (
        "arm",
        "round_no",
        "hole_no",
        "par",
        "target_event_key",
        "target_event_start",
        "target_player_id",
        "start_lie",
        "start_distance_yd",
        "start_long_yd",
        "start_lat_yd",
        "forecast_tee_x",
        "forecast_tee_y",
        "forecast_pin_x",
        "forecast_pin_y",
        "forecast_pin_long_yd",
        "physical_course_id",
    )
    proposal_identity_columns = (
        *_PROPOSAL_SOURCE_INVENTORY_COLUMNS,
        "proposal_endpoint_long_yd",
        "proposal_endpoint_lat_yd",
        "surface_class_count",
        "surface_full_probability_json",
        "surface_full_cdf_upper_json",
    )

    seen_queries: set[str] = set()
    qstate: dict[str, object] = {}
    pstate: dict[str, object] = {}
    sstate: dict[str, object] = {}
    total_rows = 0
    observed_source_inventories: list[dict[str, object]] = []

    def validate_cdf_step(
        lower: float,
        probability: float,
        upper: float,
        label: str,
        *,
        allow_forced_one: bool,
    ) -> None:
        expected = float(lower + probability)
        if upper == expected:
            return
        if (
            allow_forced_one
            and upper == 1.0
            and math.isclose(expected, 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE)
        ):
            return
        raise CompleteAtomSupportError(f"Persisted {label} CDF width changed")

    def finish_surface() -> None:
        if not sstate:
            return
        if int(sstate["rows"]) > max_hazard_donors:
            raise CompleteAtomSupportError("Persisted hazard fanout exceeds its bound")
        if not math.isclose(
            float(sstate["hazard_mass"]),
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ) or not math.isclose(
            float(sstate["hazard_upper"]),
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ):
            raise CompleteAtomSupportError("Persisted hazard donor CDF does not close")
        if not math.isclose(
            float(sstate["mechanics_mass"]),
            float(sstate["surface_probability"]),
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ):
            raise CompleteAtomSupportError("Persisted surface×hazard mass changed")
        sstate.clear()

    def finish_proposal() -> None:
        if not pstate:
            return
        finish_surface()
        if int(pstate["rows"]) > max_atoms_per_proposal:
            raise CompleteAtomSupportError("Persisted proposal fanout exceeds its bound")
        if not math.isclose(
            float(pstate["mechanics_mass"]),
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ) or not math.isclose(
            float(pstate["mechanics_upper"]),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CompleteAtomSupportError("Persisted mechanics CDF does not close")
        residual = abs(float(pstate["quality_mass"]) - float(pstate["proposal_probability"]))
        qstate["max_proposal_residual"] = max(
            float(qstate["max_proposal_residual"]), residual
        )
        if residual > MASS_TOLERANCE:
            raise CompleteAtomSupportError("Persisted proposal×mechanics mass changed")
        expected_indices = set(pstate["expected_surface_indices"])
        observed_indices = set(pstate["observed_surface_indices"])
        if expected_indices != observed_indices:
            raise CompleteAtomSupportError("Persisted surface class closure changed")
        pstate.clear()

    def finish_query() -> None:
        if not qstate:
            return
        finish_proposal()
        query_key = str(qstate["query_key"])
        declared = inventory.get(query_key)
        if declared is None:
            raise CompleteAtomSupportError("Persisted rows lack a query inventory entry")
        if not math.isclose(
            float(qstate["proposal_upper"]), 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise CompleteAtomSupportError("Persisted proposal CDF does not close")
        for branch, upper in dict(qstate["branch_upper"]).items():
            if not math.isclose(float(upper), 1.0, rel_tol=0.0, abs_tol=0.0):
                raise CompleteAtomSupportError(
                    f"Persisted {branch} proposal branch CDF does not close"
                )
        if not math.isclose(
            float(qstate["frozen_spatial_upper"]),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CompleteAtomSupportError("Persisted frozen numeric CDF does not close")
        if bool(qstate["has_terminal"]) and not math.isclose(
            float(qstate["frozen_terminal_upper"]),
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CompleteAtomSupportError("Persisted terminal residual CDF does not close")
        observed = {
            "proposal_atoms": int(qstate["proposal_atoms"]),
            "complete_atoms": int(qstate["complete_atoms"]),
            "terminal_atoms": int(qstate["terminal_atoms"]),
            "hazard_atoms": int(qstate["hazard_atoms"]),
            "imputed_ability_atoms": int(qstate["imputed_atoms"]),
        }
        for name, value in observed.items():
            if int(declared[name]) != value:
                raise CompleteAtomSupportError(f"Persisted query {query_key} {name} changed")
        for name, value in (
            ("quality_mass", qstate["quality_mass"]),
            ("terminal_mass", qstate["terminal_mass"]),
            ("max_proposal_mass_residual", qstate["max_proposal_residual"]),
        ):
            if not math.isclose(
                float(declared[name]), float(value), rel_tol=0.0, abs_tol=MASS_TOLERANCE
            ):
                raise CompleteAtomSupportError(f"Persisted query {query_key} {name} changed")
        if not math.isclose(
            float(qstate["quality_mass"]), 1.0, rel_tol=0.0, abs_tol=MASS_TOLERANCE
        ) or not math.isclose(
            float(qstate["complete_upper"]), 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise CompleteAtomSupportError("Persisted complete support CDF does not close")
        if not math.isclose(
            float(qstate["terminal_mass"]),
            float(declared["declared_app_holeout_probability"]),
            rel_tol=0.0,
            abs_tol=MASS_TOLERANCE,
        ):
            raise CompleteAtomSupportError("Persisted APP terminal mass changed")
        source_observed = {
            "query_key": query_key,
            "proposal_source_inventory_records": int(
                qstate["proposal_source_inventory"].records
            ),
            "proposal_source_inventory_sha256": qstate[
                "proposal_source_inventory"
            ].hexdigest(),
            "surface_source_inventory_records": int(
                qstate["surface_source_inventory"].records
            ),
            "surface_source_inventory_sha256": qstate[
                "surface_source_inventory"
            ].hexdigest(),
            "hazard_source_inventory_records": int(
                qstate["hazard_source_inventory"].records
            ),
            "hazard_source_inventory_sha256": qstate[
                "hazard_source_inventory"
            ].hexdigest(),
        }
        for field, value in source_observed.items():
            if field.endswith("_records"):
                equal = int(declared[field]) == int(value)
            else:
                equal = str(declared[field]) == str(value)
            if not equal:
                raise CompleteAtomSupportError(
                    f"Persisted query {query_key} source inventory changed"
                )
        observed_source_inventories.append(source_observed)

        readiness = {
            "current_chain_adapter_v3_ability_ready": bool(
                qstate["all_ability_available"]
            ),
            "required_adapter_contract_bump": bool(
                int(qstate["imputed_atoms"]) > 0
            ),
            "anchor_ready": False,
        }
        for field, expected in readiness.items():
            if type(declared.get(field)) is not bool or declared[field] is not expected:
                raise CompleteAtomSupportError(
                    f"Persisted query {query_key} readiness changed"
                )
        if str(declared.get("anchor_hold_reason", "")) != _anchor_hold_reason(
            int(qstate["imputed_atoms"])
        ):
            raise CompleteAtomSupportError(
                f"Persisted query {query_key} hold reason changed"
            )
        provenance = dict(qstate["provenance"])
        for field in (
            "proposal_support_measure_sha256",
            "numeric_payload_sha256",
            "surface_model_sha256",
            "hazard_resolver_sha256",
            "reference_model_sha256",
        ):
            audit_field = (
                "proposal_support_measure_sha256" if field == "proposal_support_measure_sha256"
                else field
            )
            if str(declared[audit_field]) != str(provenance[field]):
                raise CompleteAtomSupportError("Persisted query provenance changed")
        qstate.clear()

    expected_columns: list[str] | None = None
    for part_index, (part, relative) in enumerate(zip(part_rows, part_paths)):
        path = observed_files[relative]
        if path.suffix.casefold() not in {".parquet", ".pq"}:
            raise CompleteAtomSupportError(f"Persisted part path is invalid: {relative}")
        if _sha256_file(path) != str(part.get("byte_sha256", "")):
            raise CompleteAtomSupportError(f"Persisted part byte hash changed: {relative}")
        frame = pd.read_parquet(path)
        try:
            declared_rows = int(part["rows"])
            declared_bytes = int(part["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompleteAtomSupportError("Persisted part inventory is malformed") from exc
        if (
            len(frame) != declared_rows
            or not 0 < len(frame) <= max_part_rows
            or path.stat().st_size != declared_bytes
        ):
            raise CompleteAtomSupportError(f"Persisted part row/byte bound changed: {relative}")
        if _persisted_frame_sha(frame) != str(part.get("logical_sha256", "")):
            raise CompleteAtomSupportError(f"Persisted part logical hash changed: {relative}")
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            raise CompleteAtomSupportError("Persisted part schema/order changed")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise CompleteAtomSupportError(f"Persisted parts miss columns={missing}")
        if not frame["contract"].astype(str).eq(CONTRACT).all() or not frame[
            "selection_contract"
        ].astype(str).eq(SELECTION_CONTRACT).all():
            raise CompleteAtomSupportError("Persisted row contract changed")
        numeric = frame.loc[:, finite_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise CompleteAtomSupportError("Persisted complete-atom numeric content is non-finite")
        boundary = frame["boundary_only"].astype(bool).to_numpy()
        quality = numeric["quality_base_mass"].to_numpy(float)
        mechanics = numeric["mechanics_conditional_probability"].to_numpy(float)
        if (
            (quality < 0.0).any()
            or (mechanics < 0.0).any()
            or ((quality == 0.0) != boundary).any()
            or ((mechanics == 0.0) != boundary).any()
        ):
            raise CompleteAtomSupportError("Persisted boundary/positive mass semantics changed")
        proposal_probability = numeric["proposal_probability"].to_numpy(float)
        proposal_hierarchy = numeric["proposal_base_hierarchy_mass"].to_numpy(float)
        proposal_model = numeric["proposal_spatial_model_weight"].to_numpy(float)
        surface_probability = numeric["surface_probability"].to_numpy(float)
        hazard_probability = numeric["hazard_probability"].to_numpy(float)
        if (
            (proposal_probability <= 0.0).any()
            or (proposal_hierarchy <= 0.0).any()
            or (proposal_model <= 0.0).any()
            or (surface_probability < 0.0).any()
            or (hazard_probability <= 0.0).any()
        ):
            raise CompleteAtomSupportError("Persisted factor mass is invalid")
        if not np.allclose(
            mechanics,
            surface_probability * hazard_probability,
            atol=2e-15,
            rtol=2e-15,
        ) or not np.allclose(
            quality,
            proposal_probability * mechanics,
            atol=2e-15,
            rtol=2e-15,
        ) or not np.allclose(
            numeric["base_hierarchy_mass"].to_numpy(float),
            proposal_hierarchy * mechanics,
            atol=2e-15,
            rtol=2e-15,
        ) or not np.array_equal(
            numeric["spatial_model_weight"].to_numpy(float), proposal_model
        ) or not np.allclose(
            quality,
            numeric["base_hierarchy_mass"].to_numpy(float)
            * numeric["spatial_model_weight"].to_numpy(float),
            atol=2e-15,
            rtol=2e-15,
        ):
            raise CompleteAtomSupportError(
                "Persisted proposal×surface×hazard factorization changed"
            )
        if not np.allclose(
            numeric["fixed_reference_reward"],
            numeric["start_reference_value"]
            - numeric["stroke_cost"]
            - numeric["end_reference_value"],
            atol=2e-15,
            rtol=2e-15,
        ):
            raise CompleteAtomSupportError("Persisted fixed-reference reward changed")
        if not numeric["causal_strokes"].eq(1).all() or not numeric["stroke_cost"].eq(
            numeric["causal_strokes"] + numeric["penalty_strokes"]
        ).all():
            raise CompleteAtomSupportError("Persisted physical stroke accounting changed")

        for row in frame.itertuples(index=False):
            values = row._asdict()
            if not all(
                isinstance(values[field], (bool, np.bool_))
                for field in ("boundary_only", "terminal", "holed")
            ):
                raise CompleteAtomSupportError("Persisted boolean semantics changed")
            query_key = str(values["query_key"])
            proposal_order = int(values["proposal_row_order"])
            if not qstate or query_key != str(qstate["query_key"]):
                if query_key in seen_queries:
                    raise CompleteAtomSupportError("Persisted query rows are interleaved")
                if qstate:
                    seen_queries.add(str(qstate["query_key"]))
                finish_query()
                query = CompleteAtomQuery(
                    arm=str(values["arm"]),
                    round_no=int(values["round_no"]),
                    hole_no=int(values["hole_no"]),
                    par=int(values["par"]),
                    target_event_key=str(values["target_event_key"]),
                    target_event_start=values["target_event_start"],
                    target_player_id=str(values["target_player_id"]),
                    start_lie=str(values["start_lie"]),
                    start_distance_yd=float(values["start_distance_yd"]),
                    start_long_yd=float(values["start_long_yd"]),
                    start_lat_yd=float(values["start_lat_yd"]),
                    forecast_tee_x=float(values["forecast_tee_x"]),
                    forecast_tee_y=float(values["forecast_tee_y"]),
                    forecast_pin_x=float(values["forecast_pin_x"]),
                    forecast_pin_y=float(values["forecast_pin_y"]),
                    forecast_pin_long_yd=float(values["forecast_pin_long_yd"]),
                    physical_course_id=str(values["physical_course_id"]),
                ).validated()
                provenance = {
                    field: str(values[field])
                    for field in (
                        "proposal_support_measure_sha256",
                        "numeric_payload_sha256",
                        "surface_model_sha256",
                        "hazard_resolver_sha256",
                        "reference_model_sha256",
                    )
                }
                expected_query_key = _stable_sha(
                    {
                        "contract": QUERY_CONTRACT,
                        "query": _query_identity_payload(query),
                        **provenance,
                        "ability_policy_sha256": str(
                            values["donor_ability_policy_sha256"]
                        ),
                    }
                )
                if query_key != expected_query_key:
                    raise CompleteAtomSupportError("Persisted query identity changed")
                qstate.update(
                    {
                        "query_key": query_key,
                        "query": query,
                        "provenance": provenance,
                        "query_identity_sha256": _stable_sha(
                            {
                                field: _canonical_scalar(values[field])
                                for field in query_identity_columns
                            }
                        ),
                        "expected_complete_order": 0,
                        "complete_upper": 0.0,
                        "proposal_upper": 0.0,
                        "branch_upper": {},
                        "frozen_spatial_upper": 0.0,
                        "frozen_terminal_upper": 0.0,
                        "has_terminal": False,
                        "expected_proposal_order": 0,
                        "proposal_atoms": 0,
                        "complete_atoms": 0,
                        "terminal_atoms": 0,
                        "hazard_atoms": 0,
                        "imputed_atoms": 0,
                        "quality_mass": 0.0,
                        "terminal_mass": 0.0,
                        "max_proposal_residual": 0.0,
                        "all_ability_available": True,
                        "proposal_source_inventory": _OrderedInventoryHasher(
                            PROPOSAL_SOURCE_INVENTORY_CONTRACT,
                            _PROPOSAL_SOURCE_INVENTORY_COLUMNS,
                        ),
                        "surface_source_inventory": _OrderedInventoryHasher(
                            SURFACE_SOURCE_INVENTORY_CONTRACT,
                            _SURFACE_SOURCE_INVENTORY_COLUMNS,
                        ),
                        "hazard_source_inventory": _OrderedInventoryHasher(
                            HAZARD_SOURCE_INVENTORY_CONTRACT,
                            _HAZARD_SOURCE_INVENTORY_COLUMNS,
                        ),
                    }
                )
            query = qstate["query"]
            if str(qstate["query_identity_sha256"]) != _stable_sha(
                {
                    field: _canonical_scalar(values[field])
                    for field in query_identity_columns
                }
            ):
                raise CompleteAtomSupportError("Persisted exact query state varies")
            for field, expected in dict(qstate["provenance"]).items():
                if str(values[field]) != str(expected):
                    raise CompleteAtomSupportError("Persisted query provenance varies")
            if str(values["bridge_row_id"]) != _proposal_bridge_row_id(
                values["proposal_support_key"],
                values["proposal_row_order"],
                values["proposal_atom_id"],
            ):
                raise CompleteAtomSupportError(
                    "Persisted donor ability bridge is not source-bound"
                )
            policy_frame = pd.DataFrame(
                [
                    {
                        column: values[column]
                        for column in (
                            *ability_policy.INPUT_COLUMNS,
                            *ability_policy.OUTPUT_COLUMNS,
                        )
                    }
                ]
            )
            try:
                ability_policy.validate_applied_frame(policy_frame)
            except ability_policy.DonorAbilityPolicyError as exc:
                raise CompleteAtomSupportError(
                    "Persisted donor ability attestation changed"
                ) from exc
            if str(values["proposal_policy_binding_sha256"]) != (
                _proposal_policy_binding_sha256(values)
            ):
                raise CompleteAtomSupportError(
                    "Persisted proposal policy origin binding changed"
                )
            if int(values["complete_row_order"]) != int(
                qstate["expected_complete_order"]
            ):
                raise CompleteAtomSupportError("Persisted complete row order changed")
            lower = float(values["complete_cdf_lower"])
            upper = float(values["complete_cdf_upper"])
            mass = float(values["quality_base_mass"])
            if lower != float(qstate["complete_upper"]):
                raise CompleteAtomSupportError("Persisted complete CDF is discontinuous")
            validate_cdf_step(
                lower, mass, upper, "complete", allow_forced_one=True
            )
            qstate["complete_upper"] = upper
            qstate["expected_complete_order"] = int(qstate["expected_complete_order"]) + 1

            if not pstate or proposal_order != int(pstate["proposal_order"]):
                finish_proposal()
                if proposal_order != int(qstate["expected_proposal_order"]):
                    raise CompleteAtomSupportError("Persisted proposal order changed")
                proposal_lower = float(values["proposal_cdf_lower"])
                proposal_upper = float(values["proposal_cdf_upper"])
                proposal_probability_value = float(values["proposal_probability"])
                if proposal_lower != float(qstate["proposal_upper"]):
                    raise CompleteAtomSupportError("Persisted proposal CDF is discontinuous")
                validate_cdf_step(
                    proposal_lower,
                    proposal_probability_value,
                    proposal_upper,
                    "proposal",
                    allow_forced_one=True,
                )
                qstate["proposal_upper"] = proposal_upper
                branch = str(values["proposal_branch"])
                branch_prior = float(dict(qstate["branch_upper"]).get(branch, 0.0))
                branch_lower = float(values["proposal_branch_cdf_lower"])
                branch_upper = float(values["proposal_branch_cdf_upper"])
                branch_probability = float(
                    values["proposal_branch_conditional_probability"]
                )
                if branch_lower != branch_prior:
                    raise CompleteAtomSupportError(
                        "Persisted proposal branch CDF is discontinuous"
                    )
                validate_cdf_step(
                    branch_lower,
                    branch_probability,
                    branch_upper,
                    "proposal branch",
                    allow_forced_one=True,
                )
                qstate["branch_upper"][branch] = branch_upper
                terminal = bool(values["terminal"])
                if terminal:
                    frozen_lower = float(values["frozen_terminal_residual_cdf_lower"])
                    frozen_upper = float(values["frozen_terminal_residual_cdf_upper"])
                    if (
                        frozen_lower != float(qstate["frozen_terminal_upper"])
                        or frozen_lower != branch_lower
                        or frozen_upper != branch_upper
                    ):
                        raise CompleteAtomSupportError("Persisted terminal residual CDF changed")
                    qstate["frozen_terminal_upper"] = frozen_upper
                    qstate["has_terminal"] = True
                else:
                    frozen_lower = float(values["frozen_spatial_cdf_lower"])
                    frozen_upper = float(values["frozen_spatial_cdf_upper"])
                    if frozen_lower != float(qstate["frozen_spatial_upper"]) or not math.isclose(
                        frozen_upper - frozen_lower,
                        branch_probability,
                        rel_tol=0.0,
                        abs_tol=MASS_TOLERANCE,
                    ):
                        raise CompleteAtomSupportError("Persisted frozen numeric CDF changed")
                    qstate["frozen_spatial_upper"] = frozen_upper
                if terminal:
                    expected_surface_indices: set[int] = {-1}
                    surface_probability_vector = np.asarray([], dtype=float)
                    surface_cdf_vector = np.asarray([], dtype=float)
                else:
                    surface_probability_vector = _json_float_vector(
                        values["surface_full_probability_json"],
                        "persisted full surface probability",
                    )
                    surface_cdf_vector = _json_float_vector(
                        values["surface_full_cdf_upper_json"],
                        "persisted full surface CDF",
                    )
                    if (
                        int(values["surface_class_count"])
                        != len(surface_probability_vector)
                        or not len(surface_probability_vector)
                        or (surface_probability_vector < 0.0).any()
                        or not np.array_equal(
                            surface_cdf_vector, np.cumsum(surface_probability_vector)
                        )
                        or not math.isclose(
                            float(surface_cdf_vector[-1]),
                            1.0,
                            rel_tol=0.0,
                            abs_tol=MASS_TOLERANCE,
                        )
                    ):
                        raise CompleteAtomSupportError("Persisted full surface CDF changed")
                    expected_surface_indices = {
                        index
                        for index, value in enumerate(surface_probability_vector)
                        if value > 0.0 or index == 0
                    }
                qstate["proposal_source_inventory"].add(values)
                if not terminal:
                    qstate["surface_source_inventory"].add(values)
                pstate.update(
                    {
                        "proposal_order": proposal_order,
                        "proposal_atom_id": str(values["proposal_atom_id"]),
                        "proposal_probability": proposal_probability_value,
                        "proposal_hierarchy": float(values["proposal_base_hierarchy_mass"]),
                        "proposal_model": float(values["proposal_spatial_model_weight"]),
                        "category": str(values["category"]).casefold(),
                        "terminal": terminal,
                        "mechanics_upper": 0.0,
                        "mechanics_mass": 0.0,
                        "quality_mass": 0.0,
                        "rows": 0,
                        "surface_probability_vector": surface_probability_vector,
                        "surface_cdf_vector": surface_cdf_vector,
                        "surface_probability_json": str(values["surface_full_probability_json"]),
                        "surface_cdf_json": str(values["surface_full_cdf_upper_json"]),
                        "expected_surface_indices": expected_surface_indices,
                        "observed_surface_indices": set(),
                        "proposal_identity_sha256": _stable_sha(
                            {
                                field: _canonical_scalar(values[field])
                                for field in proposal_identity_columns
                            }
                        ),
                    }
                )
                qstate["expected_proposal_order"] = int(qstate["expected_proposal_order"]) + 1
                qstate["proposal_atoms"] = int(qstate["proposal_atoms"]) + 1
            else:
                if str(pstate["proposal_identity_sha256"]) != _stable_sha(
                    {
                        field: _canonical_scalar(values[field])
                        for field in proposal_identity_columns
                    }
                ):
                    raise CompleteAtomSupportError("Persisted proposal attestation varies")
                constants = {
                    "proposal_atom_id": str(values["proposal_atom_id"]),
                    "proposal_probability": float(values["proposal_probability"]),
                    "proposal_hierarchy": float(values["proposal_base_hierarchy_mass"]),
                    "proposal_model": float(values["proposal_spatial_model_weight"]),
                    "category": str(values["category"]).casefold(),
                    "terminal": bool(values["terminal"]),
                    "surface_probability_json": str(values["surface_full_probability_json"]),
                    "surface_cdf_json": str(values["surface_full_cdf_upper_json"]),
                }
                if any(constants[name] != pstate[name] for name in constants):
                    raise CompleteAtomSupportError("Persisted proposal metadata varies")

            mechanics_lower = float(values["mechanics_cdf_lower"])
            mechanics_upper = float(values["mechanics_cdf_upper"])
            mechanics_mass = float(values["mechanics_conditional_probability"])
            if mechanics_lower != float(pstate["mechanics_upper"]):
                raise CompleteAtomSupportError("Persisted mechanics CDF is discontinuous")
            validate_cdf_step(
                mechanics_lower,
                mechanics_mass,
                mechanics_upper,
                "mechanics",
                allow_forced_one=True,
            )
            pstate["mechanics_upper"] = mechanics_upper
            pstate["mechanics_mass"] = float(pstate["mechanics_mass"]) + mechanics_mass
            pstate["quality_mass"] = float(pstate["quality_mass"]) + mass
            pstate["rows"] = int(pstate["rows"]) + 1

            surface_index = int(values["surface_class_index"])
            if bool(values["terminal"]):
                if (
                    str(pstate["category"]) != "app"
                    or int(values["surface_class_count"]) != 0
                    or surface_index != -1
                    or str(values["surface_full_probability_json"]) != "[]"
                    or str(values["surface_full_cdf_upper_json"]) != "[]"
                    or float(values["surface_probability"]) != 1.0
                    or bool(values["boundary_only"])
                ):
                    raise CompleteAtomSupportError("Persisted terminal surface semantics changed")
            else:
                vector = np.asarray(pstate["surface_probability_vector"], float)
                cdf = np.asarray(pstate["surface_cdf_vector"], float)
                category = str(pstate["category"])
                if category == "ott":
                    classes = tee.LIE_CLASSES
                    mapping = streaming.TEE_SURFACE_TO_LIE
                elif category == "app":
                    classes = app.END_LIES
                    mapping = streaming.APP_SURFACE_TO_LIE
                else:
                    raise CompleteAtomSupportError("Persisted category changed")
                if not 0 <= surface_index < len(vector):
                    raise CompleteAtomSupportError("Persisted surface class index changed")
                expected_lower = 0.0 if surface_index == 0 else float(cdf[surface_index - 1])
                expected_upper = float(cdf[surface_index])
                expected_boundary = surface_index == 0 and float(vector[0]) == 0.0
                if (
                    float(values["surface_probability"]) != float(vector[surface_index])
                    or int(values["surface_class_count"]) != len(vector)
                    or str(values["surface_class_label"]) != str(classes[surface_index])
                    or float(values["surface_cdf_lower"]) != expected_lower
                    or float(values["surface_cdf_upper"]) != expected_upper
                    or bool(values["boundary_only"]) != expected_boundary
                ):
                    raise CompleteAtomSupportError("Persisted surface boundary/CDF changed")
            pstate["observed_surface_indices"].add(surface_index)

            surface_key = (proposal_order, surface_index, str(values["surface_class_label"]))
            if not sstate or surface_key != sstate["surface_key"]:
                finish_surface()
                sstate.update(
                    {
                        "surface_key": surface_key,
                        "surface_probability": float(values["surface_probability"]),
                        "hazard_upper": 0.0,
                        "hazard_mass": 0.0,
                        "mechanics_mass": 0.0,
                        "rows": 0,
                        "status": str(values["hazard_resolution_status"]),
                        "prior_donor_key": "",
                    }
                )
                if pstate.get("last_surface_index") is not None and surface_index <= int(
                    pstate["last_surface_index"]
                ):
                    raise CompleteAtomSupportError("Persisted surface class order changed")
                pstate["last_surface_index"] = surface_index
            elif (
                float(values["surface_probability"])
                != float(sstate["surface_probability"])
                or str(values["hazard_resolution_status"]) != str(sstate["status"])
            ):
                raise CompleteAtomSupportError("Persisted hazard surface metadata varies")
            hazard_lower = float(values["hazard_cdf_lower"])
            hazard_upper = float(values["hazard_cdf_upper"])
            hazard_mass = float(values["hazard_probability"])
            if hazard_lower != float(sstate["hazard_upper"]):
                raise CompleteAtomSupportError("Persisted raw hazard CDF is discontinuous")
            validate_cdf_step(
                hazard_lower,
                hazard_mass,
                hazard_upper,
                "raw hazard",
                allow_forced_one=False,
            )
            sstate["hazard_upper"] = hazard_upper
            sstate["hazard_mass"] = float(sstate["hazard_mass"]) + hazard_mass
            sstate["mechanics_mass"] = float(sstate["mechanics_mass"]) + mechanics_mass
            sstate["rows"] = int(sstate["rows"]) + 1
            status = str(values["hazard_resolution_status"])
            if status == "none":
                if (
                    int(sstate["rows"]) != 1
                    or hazard_lower != 0.0
                    or hazard_upper != 1.0
                    or int(values["hazard_donor_order"]) != -1
                ):
                    raise CompleteAtomSupportError("Persisted nonhazard branch changed")
            elif status == "resolved_complete_atom":
                donor_key = str(values["hazard_donor_key"])
                if (
                    int(values["penalty_strokes"]) <= 0
                    or int(values["hazard_donor_order"]) != int(sstate["rows"]) - 1
                    or donor_key <= str(sstate["prior_donor_key"])
                ):
                    raise CompleteAtomSupportError("Persisted hazard resolution order changed")
                sstate["prior_donor_key"] = donor_key
                qstate["hazard_source_inventory"].add(values)
            else:
                raise CompleteAtomSupportError("Persisted hazard resolution semantics changed")

            if not bool(values["terminal"]):
                mapped_lie = str(streaming.LIE_NAMES[int(mapping[surface_index])])
                if (mapped_lie == "penalty") != (status == "resolved_complete_atom"):
                    raise CompleteAtomSupportError("Persisted surface/hazard linkage changed")
                if mapped_lie != "penalty" and str(values["end_lie"]) != mapped_lie:
                    raise CompleteAtomSupportError("Persisted surface end lie changed")

            terminal = bool(values["terminal"])
            if terminal != bool(values["holed"]) or terminal != (
                str(values["end_lie"]) == "hole"
            ):
                raise CompleteAtomSupportError("Persisted terminal/holed equivalence changed")
            end_lie = fixed_reference.canonical_lie(values["end_lie"])
            end_distance = float(values["end_distance_yd"])
            end_long = float(values["end_long_yd"])
            end_lat = float(values["end_lat_yd"])
            geometric_distance = float(
                np.hypot(
                    end_long - float(values["forecast_pin_long_yd"]), end_lat
                )
            )
            if not math.isclose(
                end_distance,
                geometric_distance,
                rel_tol=0.0,
                abs_tol=STATE_TOLERANCE_YD,
            ):
                raise CompleteAtomSupportError("Persisted endpoint distance changed")
            start_lie = fixed_reference.canonical_lie(values["start_lie"])
            expected_start_key = (
                "terminal"
                if start_lie == "hole"
                else f"{start_lie}|{float(values['start_distance_yd']):.6f}yd"
            )
            expected_end_key = (
                "terminal" if terminal else f"{end_lie}|{end_distance:.6f}yd"
            )
            if (
                str(values["start_state_key"]) != expected_start_key
                or str(values["end_state_key"]) != expected_end_key
            ):
                raise CompleteAtomSupportError("Persisted reference state key changed")
            expected_next_category = _next_category(
                end_lie, end_distance, int(values["par"])
            )
            if str(values["next_category"]) != expected_next_category:
                raise CompleteAtomSupportError("Persisted next category changed")
            penalty_strokes = int(values["penalty_strokes"])
            penalty_count = int(values["hazard_penalty_count"])
            if status == "none":
                if penalty_strokes != 0 or penalty_count != 0:
                    raise CompleteAtomSupportError("Persisted nonhazard penalty changed")
            else:
                if (
                    not 1 <= penalty_count <= penalty_strokes
                    or int(values["hazard_penalty_stroke_cost"])
                    != penalty_strokes
                    or str(values["hazard_donor_category"]).casefold()
                    != str(values["category"]).casefold()
                    or str(values["hazard_donor_physical_course_id"])
                    != str(values["physical_course_id"])
                    or str(values["hazard_foundation_version"])
                    != hazard.FOUNDATION_VERSION
                ):
                    raise CompleteAtomSupportError(
                        "Persisted linked hazard donor content changed"
                    )
                resolution_lie = str(
                    batched_population.normalize_population_lies(
                        np.asarray([values["hazard_resolution_lie"]], dtype=object)
                    )[0]
                )
                course_frame = full_chain.CourseFrame(
                    float(values["forecast_tee_x"]),
                    float(values["forecast_tee_y"]),
                    float(values["forecast_pin_x"]),
                    float(values["forecast_pin_y"]),
                )
                resolution_local = course_frame.absolute_feet_to_local_yards(
                    np.asarray(
                        [
                            _finite(
                                values["hazard_resolution_x"],
                                "persisted hazard resolution_x",
                            ),
                            _finite(
                                values["hazard_resolution_y"],
                                "persisted hazard resolution_y",
                            ),
                        ]
                    )
                )
                if resolution_lie != end_lie or not np.allclose(
                    resolution_local,
                    np.asarray([end_long, end_lat]),
                    atol=STATE_TOLERANCE_YD,
                    rtol=0.0,
                ):
                    raise CompleteAtomSupportError(
                        "Persisted linked hazard resolution changed"
                    )
            if terminal and (
                str(values["end_state_key"]) != "terminal"
                or str(values["next_support_key"]) != ""
                or float(values["end_reference_value"]) != 0.0
            ):
                raise CompleteAtomSupportError("Persisted terminal state changed")
            expected_next = _next_support_locator(
                query=query,
                end_state_key=str(values["end_state_key"]),
                category=str(values["next_category"]),
                end_lie=str(values["end_lie"]),
                end_distance_yd=float(values["end_distance_yd"]),
                end_endpoint=np.asarray(
                    [float(values["end_long_yd"]), float(values["end_lat_yd"])]
                ),
            )
            if str(values["next_support_key"]) != expected_next:
                raise CompleteAtomSupportError("Persisted child query identity changed")
            if str(values["atom_id"]) != _complete_atom_id(values):
                raise CompleteAtomSupportError("Persisted physical atom identity changed")
            if str(values["donor_quality_group_id"]) != (
                chain_adapter.canonical_donor_quality_group_id(values)
            ):
                raise CompleteAtomSupportError("Persisted donor quality group changed")

            qstate["quality_mass"] = float(qstate["quality_mass"]) + mass
            qstate["terminal_mass"] = float(qstate["terminal_mass"]) + (
                mass if terminal else 0.0
            )
            qstate["complete_atoms"] = int(qstate["complete_atoms"]) + 1
            qstate["terminal_atoms"] = int(qstate["terminal_atoms"]) + int(terminal)
            qstate["hazard_atoms"] = int(qstate["hazard_atoms"]) + int(
                status == "resolved_complete_atom"
            )
            qstate["imputed_atoms"] = int(qstate["imputed_atoms"]) + int(
                str(values["donor_ability_value_kind"])
                == ability_policy.IMPUTED_VALUE_KIND
            )
            qstate["all_ability_available"] = bool(
                qstate["all_ability_available"]
            ) and bool(values["donor_ability_available"])
            total_rows += 1
    if qstate:
        seen_queries.add(str(qstate["query_key"]))
    finish_query()
    if seen_queries != set(inventory):
        raise CompleteAtomSupportError("Persisted query inventory closure changed")
    if total_rows != int(manifest["complete_atoms"]):
        raise CompleteAtomSupportError("Persisted global complete-atom count changed")
    observed_source_inventory = _source_inventory_sha256(
        observed_source_inventories
    )
    if observed_source_inventory != declared_source_inventory:
        raise CompleteAtomSupportError(
            "Persisted source inventory digest changed"
        )
    source_authority = _source_inventory_authority(observed_source_inventories)
    if source_authority is None:
        raise CompleteAtomSupportError(
            "Source inventory is not admitted by the closed authority"
        )
    authority_fields = {
        "source_inventory_authority_id": str(source_authority["authority_id"]),
        "source_inventory_authority_scope": str(source_authority["scope"]),
    }
    for field, expected in authority_fields.items():
        if str(manifest.get(field, "")) != expected:
            raise CompleteAtomSupportError(
                "Persisted source-inventory authority record changed"
            )
    if manifest.get("source_inventory_authority_ready") is not True:
        raise CompleteAtomSupportError(
            "Persisted source-inventory authority readiness changed"
        )
    return {
        "contract": "persisted_complete_tee_app_atom_validation_v4",
        "queries": int(len(seen_queries)),
        "parts": int(len(part_rows)),
        "complete_atoms": int(total_rows),
        "all_byte_hashes_exact": True,
        "all_logical_hashes_exact": True,
        "all_policy_rows_exact": True,
        "all_physical_atom_ids_exact": True,
        "all_factorizations_and_cdfs_exact": True,
        "all_mass_and_reward_identities_exact": True,
        "all_dependency_attestations_exact": True,
        "closed_source_inventory_authority_exact": True,
        "all_readiness_fields_rederived": True,
        "artifact_tree_closed": True,
    }


__all__ = [
    "ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS",
    "CONTRACT",
    "DEFAULT_MAX_COMPLETE_ATOMS_PER_PROPOSAL",
    "DEFAULT_MAX_HAZARD_DONORS_PER_SURFACE",
    "PERSISTENCE_CONTRACT",
    "SOURCE_INVENTORY_CONTRACT",
    "SOURCE_INVENTORY_AUTHORITY_CONTRACT",
    "UNMINTED_SOURCE_INVENTORY_AUTHORITY_ID",
    "QUERY_CONTRACT",
    "SELECTION_CONTRACT",
    "CompleteAtomExpansion",
    "CompleteAtomExpansionAudit",
    "CompleteAtomQuery",
    "CompleteAtomRequest",
    "CompleteAtomSupportError",
    "apply_frozen_policy_to_proposal",
    "hazard_resolver_sha256",
    "persist_complete_atom_requests",
    "select_complete_atom_crn",
    "source_inventory_authority_records",
    "surface_model_sha256",
    "validate_persisted_complete_atom_artifact",
]

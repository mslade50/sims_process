"""Research-only policy-v2 donor admission for the v3 mean-chain atoms.

This module is intentionally separate from
``hole_sim_transition_mean_chain_adapter.py``.  The v3 adapter and its sealed
artifacts remain byte-for-byte unchanged.  V4 changes only the donor-ability
admission contract:

* an observed value must be an attested, strictly-prior policy-v2 value;
* an unavailable value may be exactly the policy-v2 neutral-zero imputation;
* unavailable rows remain unavailable and retain their actual nullable skill
  date; and
* every row binds the frozen policy identity, policy source, policy row hash,
  original availability, and original skill date.

The current mean-anchor API does not explicitly advertise support for the
policy-v2 imputed state.  Consequently a valid frame containing even one
imputed row is *not anchor ready*.  Validation never changes
``donor_ability_available`` to true, and this module exposes no simulation
runner.  A future anchor can opt in only by publishing the exact mapping::

    EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT = {
        "frozen_strict_prior_donor_ability_policy_v2":
            "f30e0161...ba5bc7",
    }

This is a research admission/readiness layer, not a production or streaming
integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_donor_ability_policy as donor_policy  # noqa: E402
import hole_sim_transition_mean_anchor as anchor  # noqa: E402
import hole_sim_transition_mean_chain_adapter as v3  # noqa: E402


ATOM_SCHEMA_CONTRACT = "complete_physical_category_sg_atom_v4_policy2"
DONOR_SOURCE_CONTRACT = "canonical_policy_v2_donor_source_v2"
DONOR_SNAPSHOT_CONTRACT = "policy_v2_donor_snapshot_content_verified_v1"
SUPPORT_INVENTORY_CONTRACT = "complete_policy_v2_support_inventory_v1"
OBSERVED_SNAPSHOT_KIND = v3.DONOR_SNAPSHOT_CONTRACT
IMPUTED_SNAPSHOT_KIND = "policy_v2_neutral_zero_attested_v1"

FROZEN_POLICY_CONTRACT = "frozen_strict_prior_donor_ability_policy_v2"
FROZEN_POLICY_STATUS = "policy_evidence_only_not_real_chain_ready"
FROZEN_POLICY_IDENTITY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)
FROZEN_POLICY_SOURCE_DESCRIPTOR = "archive/hole_sim_donor_ability_policy.py"
FROZEN_POLICY_SOURCE_SHA256 = (
    "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
)

POLICY_ATOM_COLUMNS = (
    "atom_schema_contract",
    "donor_ability_bridge_row_id",
    "donor_ability_source_column",
    "donor_ability_raw_strict_prior",
    "donor_ability_value_kind",
    "donor_ability_was_clipped",
    "donor_ability_policy_contract",
    "donor_ability_policy_status",
    "donor_ability_policy_sha256",
    "donor_ability_policy_source_descriptor",
    "donor_ability_policy_source_sha256",
    "donor_ability_row_sha256",
    "donor_ability_original_available",
    "donor_ability_original_skill_date",
    "support_inventory_sha256",
)

REQUIRED_ATOM_COLUMNS = tuple(dict.fromkeys((*v3.REQUIRED_ATOM_COLUMNS, *POLICY_ATOM_COLUMNS)))
QUALITY_HASH_COLUMNS = tuple(dict.fromkeys((*v3.QUALITY_HASH_COLUMNS, *POLICY_ATOM_COLUMNS)))
SUPPORT_INVENTORY_HASH_COLUMNS = tuple(
    column
    for column in REQUIRED_ATOM_COLUMNS
    if column not in {"quality_input_sha256", "support_inventory_sha256"}
)

POLICY_FRAME_COLUMNS = (
    "bridge_row_id",
    "category",
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

_OPTIONAL_TIMESTAMPS = {
    "donor_skill_date",
    "donor_ability_original_skill_date",
}
_OPTIONAL_NUMBERS = {"donor_ability_raw_strict_prior"}
_POLICY_BOOLEANS = {
    "donor_ability_was_clipped",
    "donor_ability_original_available",
}
_POLICY_TEXT = {
    "atom_schema_contract",
    "donor_ability_bridge_row_id",
    "donor_ability_source_column",
    "donor_ability_value_kind",
    "donor_ability_policy_contract",
    "donor_ability_policy_status",
    "donor_ability_policy_sha256",
    "donor_ability_policy_source_descriptor",
    "donor_ability_policy_source_sha256",
    "donor_ability_row_sha256",
    "support_inventory_sha256",
}
_DONOR_IDENTITY_COLUMNS = ("donor_event_key", "donor_player_id", "category")


class PolicyV2ContractUnavailable(v3.QualitySupportUnavailable):
    """Raised when a v4 donor-policy contract is missing or forged."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_timestamp(value: object, label: str) -> str | None:
    if _is_missing(value):
        return None
    return v3._canonical_timestamp(value, label)


def _optional_number(value: object, label: str) -> str | None:
    if _is_missing(value):
        return None
    return v3._canonical_number(value, label)


def _canonical_v4_value(column: str, value: object) -> object:
    if column in _OPTIONAL_TIMESTAMPS:
        return _optional_timestamp(value, column)
    if column in _OPTIONAL_NUMBERS:
        return _optional_number(value, column)
    if column in _POLICY_BOOLEANS:
        return v3._canonical_bool(value, column)
    if column in _POLICY_TEXT:
        return v3._canonical_text(value, column)
    return v3._canonical_hash_value(column, value)


def _canonical_record(row: Mapping[str, object], columns: Sequence[str]) -> dict[str, object]:
    return {column: _canonical_v4_value(column, row[column]) for column in columns}


def _records_sha(frame: pd.DataFrame, columns: Sequence[str], contract: str) -> str:
    ordered = frame.sort_values(["support_key", "atom_id"]).reset_index(drop=True)
    records = [
        _canonical_record(row, columns)
        for row in ordered[list(columns)].to_dict("records")
    ]
    return _stable_sha({"contract": contract, "records": records})


def validate_frozen_policy_runtime() -> dict[str, str]:
    """Verify the imported policy is the exact sealed v2 source and identity."""

    if donor_policy.POLICY_CONTRACT != FROZEN_POLICY_CONTRACT:
        raise PolicyV2ContractUnavailable("Frozen donor policy contract changed")
    if donor_policy.POLICY_STATUS != FROZEN_POLICY_STATUS:
        raise PolicyV2ContractUnavailable("Frozen donor policy status changed")
    identity_sha = str(donor_policy.policy_payload().get("policy_identity_sha256"))
    if identity_sha != FROZEN_POLICY_IDENTITY_SHA256:
        raise PolicyV2ContractUnavailable("Frozen donor policy identity changed")
    source_sha = _file_sha256(donor_policy.POLICY_SOURCE_PATH)
    if source_sha != FROZEN_POLICY_SOURCE_SHA256:
        raise PolicyV2ContractUnavailable("Frozen donor policy source changed")
    return {
        "contract": FROZEN_POLICY_CONTRACT,
        "status": FROZEN_POLICY_STATUS,
        "identity_sha256": identity_sha,
        "source_descriptor": FROZEN_POLICY_SOURCE_DESCRIPTOR,
        "source_sha256": source_sha,
    }


def _policy_frame_from_atoms(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={"donor_ability_bridge_row_id": "bridge_row_id"})
    evidence_columns = list(POLICY_FRAME_COLUMNS)
    canonical_rows: dict[str, dict[str, object]] = {}
    identity_to_bridge: dict[tuple[str, str, str], str] = {}
    identity_to_kind: dict[tuple[str, str, str], str] = {}
    for row in renamed[evidence_columns].to_dict("records"):
        bridge = v3._canonical_text(row["bridge_row_id"], "bridge_row_id", casefold=False)
        identity = tuple(
            v3._canonical_text(row[column], column) for column in _DONOR_IDENTITY_COLUMNS
        )
        kind = v3._canonical_text(row["donor_ability_value_kind"], "donor_ability_value_kind")
        if identity in identity_to_bridge and identity_to_bridge[identity] != bridge:
            raise PolicyV2ContractUnavailable(
                "One donor event/player/category maps to multiple policy bridge rows"
            )
        if identity in identity_to_kind and identity_to_kind[identity] != kind:
            raise PolicyV2ContractUnavailable(
                "One donor event/player/category mixes observed and imputed states"
            )
        identity_to_bridge[identity] = bridge
        identity_to_kind[identity] = kind
        canonical = {
            column: _canonical_v4_value(
                "donor_ability_bridge_row_id" if column == "bridge_row_id" else column,
                value,
            )
            for column, value in row.items()
        }
        if bridge in canonical_rows and canonical_rows[bridge] != canonical:
            prior_kind = canonical_rows[bridge].get("donor_ability_value_kind")
            if prior_kind != canonical.get("donor_ability_value_kind"):
                raise PolicyV2ContractUnavailable(
                    "One policy bridge row mixes observed and imputed states"
                )
            raise PolicyV2ContractUnavailable(
                "Repeated policy bridge row has conflicting attested content"
            )
        canonical_rows[bridge] = canonical

    positions: dict[str, int] = {}
    unique_rows: list[dict[str, object]] = []
    for row in renamed[evidence_columns].to_dict("records"):
        bridge = str(row["bridge_row_id"])
        if bridge not in positions:
            positions[bridge] = len(unique_rows)
            unique_rows.append(row)
    policy_frame = pd.DataFrame(unique_rows, columns=evidence_columns)
    try:
        donor_policy.validate_applied_frame(policy_frame)
    except donor_policy.DonorAbilityPolicyError as exc:
        raise PolicyV2ContractUnavailable(
            f"Policy-v2 donor row attestation failed: {exc}"
        ) from exc
    return policy_frame


def _validate_policy_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    runtime = validate_frozen_policy_runtime()
    exact_fields = {
        "atom_schema_contract": ATOM_SCHEMA_CONTRACT,
        "donor_ability_policy_contract": runtime["contract"],
        "donor_ability_policy_status": runtime["status"],
        "donor_ability_policy_sha256": runtime["identity_sha256"],
        "donor_ability_policy_source_descriptor": runtime["source_descriptor"],
        "donor_ability_policy_source_sha256": runtime["source_sha256"],
    }
    for column, expected in exact_fields.items():
        observed = {
            v3._canonical_text(value, column) for value in frame[column]
        }
        if observed != {v3._canonical_text(expected, column)}:
            raise PolicyV2ContractUnavailable(
                f"{column} does not bind the exact frozen policy-v2 value"
            )

    available = v3._coerce_bool(
        frame["donor_ability_available"], "donor_ability_available"
    )
    original_available = v3._coerce_bool(
        frame["donor_ability_original_available"],
        "donor_ability_original_available",
    )
    if not np.array_equal(available, original_available):
        raise PolicyV2ContractUnavailable(
            "Policy adaptation changed original donor ability availability"
        )
    for current, original in zip(
        frame["donor_skill_date"], frame["donor_ability_original_skill_date"]
    ):
        if _optional_timestamp(current, "donor_skill_date") != _optional_timestamp(
            original, "donor_ability_original_skill_date"
        ):
            raise PolicyV2ContractUnavailable(
                "Policy adaptation erased or invented the original donor skill date"
            )

    _policy_frame_from_atoms(frame)
    kinds = frame["donor_ability_value_kind"].astype(str)
    allowed = {donor_policy.OBSERVED_VALUE_KIND, donor_policy.IMPUTED_VALUE_KIND}
    if not set(kinds).issubset(allowed):
        raise PolicyV2ContractUnavailable(
            "Donor ability state is neither observed strict-prior nor policy-v2 neutral zero"
        )
    observed = kinds.eq(donor_policy.OBSERVED_VALUE_KIND).to_numpy(bool)
    imputed = kinds.eq(donor_policy.IMPUTED_VALUE_KIND).to_numpy(bool)
    ability = pd.to_numeric(frame["donor_ability_sg"], errors="coerce").to_numpy(float)
    if not np.isfinite(ability).all():
        raise PolicyV2ContractUnavailable("Applied donor abilities must be finite")
    if observed.any():
        if not available[observed].all():
            raise PolicyV2ContractUnavailable("Observed strict-prior ability is unavailable")
        if any(_is_missing(value) for value in frame.loc[observed, "donor_skill_date"]):
            raise PolicyV2ContractUnavailable(
                "Observed strict-prior ability requires its original skill date"
            )
        if not frame.loc[observed, "donor_ability_snapshot_kind"].eq(
            OBSERVED_SNAPSHOT_KIND
        ).all():
            raise PolicyV2ContractUnavailable("Observed donor snapshot kind changed")
    if imputed.any():
        if available[imputed].any():
            raise PolicyV2ContractUnavailable(
                "Policy-v2 neutral-zero imputation changed availability to true"
            )
        if not np.equal(ability[imputed], donor_policy.IMPUTED_NEUTRAL_VALUE).all():
            raise PolicyV2ContractUnavailable(
                "Policy-v2 missing donor ability is not exact neutral zero"
            )
        if not frame.loc[imputed, "donor_ability_snapshot_kind"].eq(
            IMPUTED_SNAPSHOT_KIND
        ).all():
            raise PolicyV2ContractUnavailable("Imputed donor snapshot kind changed")
    return frame


def _donor_source_payload(frame: pd.DataFrame) -> dict[str, object]:
    _validate_policy_evidence(frame)
    descriptors = {
        v3._canonical_text(value, "donor_source_descriptor")
        for value in frame["donor_source_descriptor"]
    }
    versions = {
        v3._canonical_text(value, "donor_source_version")
        for value in frame["donor_source_version"]
    }
    if len(descriptors) != 1 or len(versions) != 1:
        raise PolicyV2ContractUnavailable(
            "Donor source must have one descriptor and version"
        )
    event_starts = v3._canonical_event_start_map(frame)
    record_columns = (
        "donor_ability_bridge_row_id",
        "donor_event_key",
        "donor_event_start",
        "donor_player_id",
        "category",
        "donor_skill_date",
        "donor_ability_source_column",
        "donor_ability_available",
        "donor_ability_original_available",
        "donor_ability_original_skill_date",
        "donor_ability_raw_strict_prior",
        "donor_ability_value_kind",
        "donor_ability_sg",
        "donor_ability_was_clipped",
        "donor_ability_policy_contract",
        "donor_ability_policy_status",
        "donor_ability_policy_sha256",
        "donor_ability_policy_source_descriptor",
        "donor_ability_policy_source_sha256",
        "donor_ability_row_sha256",
        "target_event_key",
        "target_event_start",
    )
    records_by_identity: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in frame[list(record_columns)].to_dict("records"):
        identity = tuple(
            v3._canonical_text(row[column], column)
            for column in _DONOR_IDENTITY_COLUMNS
        )
        canonical = _canonical_record(row, record_columns)
        if identity in records_by_identity and records_by_identity[identity] != canonical:
            raise PolicyV2ContractUnavailable(
                "One donor event/player/category has conflicting policy-v2 content"
            )
        records_by_identity[identity] = canonical
    return {
        "contract": DONOR_SOURCE_CONTRACT,
        "descriptor": next(iter(descriptors)),
        "version": next(iter(versions)),
        "policy_contract": FROZEN_POLICY_CONTRACT,
        "policy_identity_sha256": FROZEN_POLICY_IDENTITY_SHA256,
        "policy_source_descriptor": FROZEN_POLICY_SOURCE_DESCRIPTOR,
        "policy_source_sha256": FROZEN_POLICY_SOURCE_SHA256,
        "event_start_mapping": [
            {"event_key": key, "event_start": event_starts[key]}
            for key in sorted(event_starts)
        ],
        "snapshot_records": [records_by_identity[key] for key in sorted(records_by_identity)],
    }


def donor_source_sha256(frame: pd.DataFrame) -> str:
    """Digest original and applied policy-v2 donor evidence as one source."""

    return _stable_sha(_donor_source_payload(frame))


def _donor_snapshot_digest(row: Mapping[str, object], source_sha: str) -> str:
    columns = (
        "donor_ability_bridge_row_id",
        "donor_event_key",
        "donor_event_start",
        "donor_player_id",
        "category",
        "donor_skill_date",
        "donor_ability_available",
        "donor_ability_original_available",
        "donor_ability_original_skill_date",
        "donor_ability_value_kind",
        "donor_ability_sg",
        "donor_ability_policy_sha256",
        "donor_ability_policy_source_sha256",
        "donor_ability_row_sha256",
    )
    return _stable_sha(
        {
            "contract": DONOR_SNAPSHOT_CONTRACT,
            "donor_source_sha256": source_sha,
            "snapshot": _canonical_record(row, columns),
        }
    )


def quality_input_sha256(frame: pd.DataFrame) -> str:
    """Hash one support including the complete policy-v2 donor attestation."""

    missing = set(QUALITY_HASH_COLUMNS).difference(frame.columns)
    if missing:
        raise PolicyV2ContractUnavailable(
            f"Cannot hash v4 quality support; missing fields={sorted(missing)}"
        )
    if frame["support_key"].astype(str).nunique() != 1:
        raise PolicyV2ContractUnavailable(
            "quality_input_sha256 accepts exactly one support"
        )
    return _records_sha(frame, QUALITY_HASH_COLUMNS, ATOM_SCHEMA_CONTRACT)


def support_inventory_sha256(frame: pd.DataFrame) -> str:
    """Hash the complete canonical atom/support inventory without self-reference."""

    missing = set(SUPPORT_INVENTORY_HASH_COLUMNS).difference(frame.columns)
    if missing:
        raise PolicyV2ContractUnavailable(
            f"Cannot hash v4 support inventory; missing fields={sorted(missing)}"
        )
    return _records_sha(
        frame,
        SUPPORT_INVENTORY_HASH_COLUMNS,
        SUPPORT_INVENTORY_CONTRACT,
    )


def seal_policy_v2_contracts(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate then seal policy-v2 source, inventory, and support hashes."""

    missing = set(REQUIRED_ATOM_COLUMNS).difference(frame.columns)
    missing.discard("quality_input_sha256")
    missing.discard("support_inventory_sha256")
    if missing:
        raise PolicyV2ContractUnavailable(
            f"Cannot seal v4 atoms; missing fields={sorted(missing)}"
        )
    out = frame.copy(deep=True)
    _validate_policy_evidence(out)
    out = _validate_base_atom_semantics(out)
    source_sha = donor_source_sha256(out)
    out["donor_source_sha256"] = source_sha
    out["donor_snapshot_sha256"] = [
        _donor_snapshot_digest(row, source_sha) for row in out.to_dict("records")
    ]
    out["donor_quality_group_id"] = [
        v3.canonical_donor_quality_group_id(row) for row in out.to_dict("records")
    ]
    out["support_inventory_sha256"] = ""
    inventory_sha = support_inventory_sha256(out)
    out["support_inventory_sha256"] = inventory_sha
    out["quality_input_sha256"] = ""
    for _, index in out.groupby("support_key", observed=True, sort=False).groups.items():
        positions = list(index)
        out.loc[positions, "quality_input_sha256"] = quality_input_sha256(
            out.loc[positions]
        )
    return out


def _validate_base_atom_semantics(
    frame: pd.DataFrame,
    *,
    expected_reward_reference_sha256: str | None = None,
    expected_target_event_key: str | None = None,
    expected_target_event_start: object | None = None,
) -> pd.DataFrame:
    """Apply every ability-independent v3 atom, support, and APP invariant."""

    out = frame.copy(deep=True)
    if out.empty:
        raise PolicyV2ContractUnavailable("Complete atom graph cannot be empty")
    for column in (
        "support_key",
        "atom_id",
        "physical_atom_sha256",
        "category",
        "start_state_key",
        "end_state_key",
        "atom_kind",
        "hazard_resolution_status",
        "reward_reference_sha256",
        "donor_quality_group_id",
        "donor_event_key",
        "donor_player_id",
        "donor_ability_snapshot_kind",
        "donor_source_descriptor",
        "donor_source_version",
        "donor_source_sha256",
        "donor_snapshot_sha256",
        "target_event_key",
        "holeout_representation",
        "app_holeout_equivalence_sha256",
        "reward_reference_descriptor",
        "reward_reference_version",
    ):
        out[column] = [v3._canonical_text(value, column) for value in out[column]]
    out["next_support_key"] = [
        ""
        if value is None or (not isinstance(value, str) and pd.isna(value))
        else unicodedata.normalize("NFKC", str(value)).strip().casefold()
        for value in out["next_support_key"]
    ]
    if out.duplicated(["support_key", "atom_id"]).any():
        raise PolicyV2ContractUnavailable("Atom IDs are duplicated inside a support")
    if not out["category"].isin(v3.CATEGORIES).all():
        raise PolicyV2ContractUnavailable("Atom category is outside OTT/APP/ARG/PUTT")
    for column in (
        "physical_atom_sha256",
        "reward_reference_sha256",
        "donor_ability_policy_sha256",
        "donor_ability_policy_source_sha256",
        "donor_ability_row_sha256",
    ):
        if not out[column].map(v3._is_sha256).all():
            raise PolicyV2ContractUnavailable(f"{column} is invalid")
    reward_hashes = set(out["reward_reference_sha256"])
    if len(reward_hashes) != 1:
        raise PolicyV2ContractUnavailable("Complete atoms use multiple reward references")
    if expected_reward_reference_sha256 is not None:
        if not v3._is_sha256(expected_reward_reference_sha256):
            raise PolicyV2ContractUnavailable("Expected reward reference digest is invalid")
        if reward_hashes != {str(expected_reward_reference_sha256)}:
            raise PolicyV2ContractUnavailable("Complete atom reward reference changed")

    terminal = v3._coerce_bool(out["terminal"], "terminal")
    holed = v3._coerce_bool(out["holed"], "holed")
    complete = v3._coerce_bool(out["complete_atom"], "complete_atom")
    reward_available = v3._coerce_bool(
        out["quality_reward_available"], "quality_reward_available"
    )
    ability_available = v3._coerce_bool(
        out["donor_ability_available"], "donor_ability_available"
    )
    if not complete.all():
        raise PolicyV2ContractUnavailable("Every quality atom must be complete")
    if not reward_available.all():
        raise PolicyV2ContractUnavailable(
            "Missing fixed-reference reward cannot be treated as zero"
        )
    out["terminal"] = terminal
    out["holed"] = holed
    out["complete_atom"] = complete
    out["quality_reward_available"] = reward_available
    out["donor_ability_available"] = ability_available

    numeric_columns = (
        "causal_strokes",
        "penalty_strokes",
        "stroke_cost",
        "start_reference_value",
        "end_reference_value",
        "fixed_reference_reward",
        "donor_ability_sg",
        "base_hierarchy_mass",
        "spatial_model_weight",
        "quality_base_mass",
        "implicit_holeout_probability",
        "declared_app_base_holeout_probability",
    )
    numeric = out[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise PolicyV2ContractUnavailable("Complete atom numeric fields must be finite")
    out[list(numeric_columns)] = numeric
    for column in ("causal_strokes", "penalty_strokes", "stroke_cost"):
        values = numeric[column].to_numpy(float)
        if not np.array_equal(values, np.rint(values)):
            raise PolicyV2ContractUnavailable(
                f"{column} must be integer physical strokes"
            )
        out[column] = np.rint(values).astype(np.int16)
    if not out["causal_strokes"].eq(1).all():
        raise PolicyV2ContractUnavailable(
            "One complete atom must own exactly one causal shot"
        )
    if out["penalty_strokes"].lt(0).any():
        raise PolicyV2ContractUnavailable("Penalty strokes cannot be negative")
    if not out["stroke_cost"].eq(
        out["causal_strokes"] + out["penalty_strokes"]
    ).all():
        raise PolicyV2ContractUnavailable(
            "stroke_cost must equal causal shot plus linked penalty strokes"
        )

    observed_physical = pd.Series(
        [
            v3._physical_atom_digest(row)
            for row in out[list(v3.PHYSICAL_ATOM_HASH_COLUMNS)].to_dict("records")
        ],
        index=out.index,
    )
    if not out["physical_atom_sha256"].equals(observed_physical):
        raise PolicyV2ContractUnavailable(
            "physical_atom_sha256 is stale or not bound to canonical physical content"
        )
    penalty = out["penalty_strokes"].gt(0)
    if not out.loc[penalty, "hazard_resolution_status"].isin(
        ["resolved_complete_atom", "collapsed_exact_transition"]
    ).all():
        raise PolicyV2ContractUnavailable("Penalty atoms are not coherently resolved")
    if not out.loc[~penalty, "hazard_resolution_status"].eq("none").all():
        raise PolicyV2ContractUnavailable(
            "Non-penalty atoms have an invalid hazard status"
        )
    if out["base_hierarchy_mass"].le(0.0).any():
        raise PolicyV2ContractUnavailable(
            "Every structural atom needs positive hierarchy mass"
        )
    if out["spatial_model_weight"].le(0.0).any():
        raise PolicyV2ContractUnavailable("Spatial model weights must be positive")
    expected_mass = (
        out["base_hierarchy_mass"].to_numpy(float)
        * out["spatial_model_weight"].to_numpy(float)
    )
    if not np.allclose(
        out["quality_base_mass"].to_numpy(float),
        expected_mass,
        atol=1e-15,
        rtol=1e-12,
    ):
        raise PolicyV2ContractUnavailable(
            "quality_base_mass must be hierarchy mass times spatial model weight"
        )

    observed_reward_sha = v3.reward_reference_sha256(out)
    if not out["reward_reference_sha256"].eq(observed_reward_sha).all():
        raise PolicyV2ContractUnavailable(
            "reward_reference_sha256 is not bound to the canonical full value table"
        )
    if (
        expected_reward_reference_sha256 is not None
        and observed_reward_sha != str(expected_reward_reference_sha256)
    ):
        raise PolicyV2ContractUnavailable("Canonical reward reference changed")
    expected_reward = (
        out["start_reference_value"].to_numpy(float)
        - out["stroke_cost"].to_numpy(float)
        - out["end_reference_value"].to_numpy(float)
    )
    residual = out["fixed_reference_reward"].to_numpy(float) - expected_reward
    if np.max(np.abs(residual), initial=0.0) > v3.ATOM_REWARD_TOLERANCE:
        raise PolicyV2ContractUnavailable(
            "Fixed-reference atom reward fails exact stroke/value accounting"
        )

    terminal_rows = out["terminal"]
    if not np.array_equal(
        terminal_rows.to_numpy(bool), out["holed"].to_numpy(bool)
    ):
        raise PolicyV2ContractUnavailable(
            "Terminal and holed semantics must be exactly equivalent per atom"
        )
    if not out.loc[terminal_rows, "end_state_key"].eq(v3.TERMINAL_STATE_KEY).all():
        raise PolicyV2ContractUnavailable("Terminal atoms must end at terminal state")
    if not np.allclose(
        out.loc[terminal_rows, "end_reference_value"], 0.0, atol=0.0, rtol=0.0
    ):
        raise PolicyV2ContractUnavailable("Terminal reference value must be exactly zero")
    if not out.loc[terminal_rows, "next_support_key"].eq("").all():
        raise PolicyV2ContractUnavailable("Terminal atoms cannot point to another support")
    if out.loc[~terminal_rows, "next_support_key"].eq("").any():
        raise PolicyV2ContractUnavailable("Nonterminal atoms need a next support")
    if out.loc[terminal_rows, "penalty_strokes"].gt(0).any():
        raise PolicyV2ContractUnavailable(
            "A linked penalty/drop sequence must continue to a playable state"
        )

    app_rows = out["category"].eq("app")
    if not out.loc[app_rows, "holeout_representation"].eq(
        v3.APP_HOLEOUT_CONTRACT
    ).all():
        raise PolicyV2ContractUnavailable(
            "APP support does not declare explicit holeout atoms"
        )
    if not np.allclose(
        out.loc[app_rows, "implicit_holeout_probability"],
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise PolicyV2ContractUnavailable("Independent APP holeout probability is forbidden")
    app_terminal = app_rows & terminal_rows
    app_holeout = out["atom_kind"].eq("app_holeout")
    if not np.array_equal(app_holeout.to_numpy(bool), app_terminal.to_numpy(bool)):
        raise PolicyV2ContractUnavailable(
            "APP holeout atom must be present iff category is APP and atom is terminal/holed"
        )
    if out.loc[~app_rows, "atom_kind"].eq("app_holeout").any():
        raise PolicyV2ContractUnavailable("APP holeout atom is assigned to another category")
    if not out.loc[~app_rows, "holeout_representation"].eq("not_applicable").all():
        raise PolicyV2ContractUnavailable(
            "Non-APP holeout representation must be not_applicable"
        )
    if not np.allclose(
        out.loc[~app_rows, "declared_app_base_holeout_probability"],
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise PolicyV2ContractUnavailable(
            "Non-APP declared holeout probability must be exactly zero"
        )
    if not out.loc[~app_rows, "app_holeout_equivalence_sha256"].eq(
        "not_applicable"
    ).all():
        raise PolicyV2ContractUnavailable(
            "Non-APP holeout equivalence must be not_applicable"
        )
    for support_key, group in out.loc[app_rows].groupby(
        "support_key", observed=True, sort=False
    ):
        if group["declared_app_base_holeout_probability"].nunique(
            dropna=False
        ) != 1:
            raise PolicyV2ContractUnavailable(
                f"APP support {support_key!r} has multiple declared holeout probabilities"
            )
        declared_holeout = float(
            group["declared_app_base_holeout_probability"].iloc[0]
        )
        if not 0.0 <= declared_holeout <= 1.0:
            raise PolicyV2ContractUnavailable(
                f"APP support {support_key!r} declared holeout probability is outside [0,1]"
            )
        terminal_mask = group["terminal"].to_numpy(bool)
        if (declared_holeout > 0.0) != bool(terminal_mask.any()):
            raise PolicyV2ContractUnavailable(
                f"APP support {support_key!r} terminal holeout coverage must exist iff "
                "declared expected mass is positive"
            )
        base_mass = group["quality_base_mass"].to_numpy(float)
        actual_holeout = float(base_mass[terminal_mask].sum() / base_mass.sum())
        if not math.isclose(
            actual_holeout,
            declared_holeout,
            rel_tol=0.0,
            abs_tol=v3.APP_HOLEOUT_MASS_TOLERANCE,
        ):
            raise PolicyV2ContractUnavailable(
                f"APP support {support_key!r} actual base holeout mass "
                f"{actual_holeout:.17g} does not equal declared "
                f"{declared_holeout:.17g}"
            )
        observed_equivalence = v3._app_holeout_equivalence_digest(group)
        if set(group["app_holeout_equivalence_sha256"]) != {observed_equivalence}:
            raise PolicyV2ContractUnavailable(
                f"APP support {support_key!r} holeout semantics/equivalence digest is stale"
            )

    skill_date = pd.to_datetime(out["donor_skill_date"], errors="coerce", utc=True)
    event_start = pd.to_datetime(out["donor_event_start"], errors="coerce", utc=True)
    target_start = pd.to_datetime(out["target_event_start"], errors="coerce", utc=True)
    supplied_skill_date = ~out["donor_skill_date"].map(_is_missing)
    if event_start.isna().any() or target_start.isna().any():
        raise PolicyV2ContractUnavailable("Donor event/target timestamps are invalid")
    if skill_date[supplied_skill_date].isna().any():
        raise PolicyV2ContractUnavailable("Supplied donor skill timestamps are invalid")
    if not (skill_date[supplied_skill_date] < event_start[supplied_skill_date]).all():
        raise PolicyV2ContractUnavailable("Donor ability is not strictly prior to donor event")
    if not (event_start < target_start).all():
        raise PolicyV2ContractUnavailable(
            "Donor event is not strictly prior to the target-event cutoff"
        )
    out["donor_skill_date"] = skill_date
    out["donor_event_start"] = event_start
    out["target_event_start"] = target_start

    target_keys = set(out["target_event_key"])
    target_start_texts = {
        v3._canonical_timestamp(value, "target_event_start")
        for value in out["target_event_start"]
    }
    if len(target_keys) != 1 or len(target_start_texts) != 1:
        raise PolicyV2ContractUnavailable("Atom target event identity is not singular")
    if expected_target_event_key is not None:
        expected_key = v3._canonical_text(
            expected_target_event_key, "expected_target_event_key"
        )
        if target_keys != {expected_key}:
            raise PolicyV2ContractUnavailable("Atom target event identity changed")
    if expected_target_event_start is not None:
        expected_start = v3._canonical_timestamp(
            expected_target_event_start, "expected_target_event_start"
        )
        if target_start_texts != {expected_start}:
            raise PolicyV2ContractUnavailable("Atom target event start changed")
    v3._canonical_event_start_map(out)

    supports: dict[str, pd.DataFrame] = {}
    for support_key, group in out.groupby("support_key", observed=True, sort=False):
        for column in ("category", "start_state_key", "start_reference_value"):
            if group[column].nunique(dropna=False) != 1:
                raise PolicyV2ContractUnavailable(
                    f"Support {support_key!r} does not have one {column}"
                )
        supports[str(support_key)] = group

    state_values: dict[str, float] = {v3.TERMINAL_STATE_KEY: 0.0}
    for row in out.itertuples(index=False):
        for state_key, value in (
            (str(row.start_state_key), float(row.start_reference_value)),
            (str(row.end_state_key), float(row.end_reference_value)),
        ):
            if state_key in state_values and not math.isclose(
                state_values[state_key], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise PolicyV2ContractUnavailable(
                    f"Reference value changes for state {state_key!r}"
                )
            state_values[state_key] = value

    support_start_state = {
        key: str(group["start_state_key"].iloc[0]) for key, group in supports.items()
    }
    for row in out.itertuples(index=False):
        if bool(row.terminal):
            continue
        next_key = str(row.next_support_key)
        if next_key not in supports:
            raise PolicyV2ContractUnavailable(
                f"Atom {row.atom_id!r} points to missing support {next_key!r}"
            )
        if str(row.end_state_key) != support_start_state[next_key]:
            raise PolicyV2ContractUnavailable(
                f"Atom {row.atom_id!r} end state does not match next support start"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise PolicyV2ContractUnavailable(
                "Support graph contains a cycle; unroll by transition/stroke budget first"
            )
        if key in visited:
            return
        visiting.add(key)
        group = supports[key]
        for child in sorted(set(group.loc[~group["terminal"], "next_support_key"])):
            visit(str(child))
        visiting.remove(key)
        visited.add(key)

    for support_key in sorted(supports):
        visit(support_key)
    return out.sort_values(["support_key", "atom_id"]).reset_index(drop=True)


def validate_policy_v2_atom_frame(
    frame: pd.DataFrame,
    *,
    expected_reward_reference_sha256: str,
    expected_donor_source_sha256: str,
    expected_support_inventory_sha256: str,
    expected_target_event_key: str,
    expected_target_event_start: object,
) -> pd.DataFrame:
    """Validate and canonicalize a research-only v4 atom frame."""

    if not isinstance(frame, pd.DataFrame):
        raise PolicyV2ContractUnavailable("V4 atom input must be a DataFrame")
    missing = set(REQUIRED_ATOM_COLUMNS).difference(frame.columns)
    if missing:
        raise PolicyV2ContractUnavailable(f"V4 atoms miss fields={sorted(missing)}")
    out = frame[list(REQUIRED_ATOM_COLUMNS)].copy(deep=True)
    _validate_policy_evidence(out)
    out = _validate_base_atom_semantics(
        out,
        expected_reward_reference_sha256=expected_reward_reference_sha256,
        expected_target_event_key=expected_target_event_key,
        expected_target_event_start=expected_target_event_start,
    )
    observed_source = donor_source_sha256(out)
    if not v3._is_sha256(expected_donor_source_sha256):
        raise PolicyV2ContractUnavailable("Expected v4 donor source digest is invalid")
    if observed_source != str(expected_donor_source_sha256):
        raise PolicyV2ContractUnavailable("Canonical policy-v2 donor source changed")
    if set(out["donor_source_sha256"].astype(str)) != {observed_source}:
        raise PolicyV2ContractUnavailable("donor_source_sha256 is stale")
    snapshots = pd.Series(
        [
            _donor_snapshot_digest(row, observed_source)
            for row in out.to_dict("records")
        ],
        index=out.index,
    )
    if not out["donor_snapshot_sha256"].astype(str).equals(snapshots):
        raise PolicyV2ContractUnavailable("donor_snapshot_sha256 is stale")
    groups = pd.Series(
        [v3.canonical_donor_quality_group_id(row) for row in out.to_dict("records")],
        index=out.index,
    )
    if not out["donor_quality_group_id"].astype(str).equals(groups):
        raise PolicyV2ContractUnavailable("donor_quality_group_id changed")
    if not v3._is_sha256(expected_support_inventory_sha256):
        raise PolicyV2ContractUnavailable(
            "Expected v4 support-inventory digest is invalid"
        )
    observed_inventory = support_inventory_sha256(out)
    if observed_inventory != str(expected_support_inventory_sha256):
        raise PolicyV2ContractUnavailable(
            "Canonical full support inventory changed"
        )
    if set(out["support_inventory_sha256"].astype(str)) != {observed_inventory}:
        raise PolicyV2ContractUnavailable("support_inventory_sha256 is stale")
    for support_key, group in out.groupby("support_key", observed=True, sort=False):
        declared = set(group["quality_input_sha256"].astype(str))
        observed = quality_input_sha256(group)
        if declared != {observed}:
            raise PolicyV2ContractUnavailable(
                f"Support {support_key!r} quality_input_sha256 is stale"
            )
    return out.sort_values(["support_key", "atom_id"]).reset_index(drop=True)


def _anchor_api_supports_policy_v2() -> bool:
    declaration = getattr(anchor, "EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT", None)
    if not isinstance(declaration, Mapping):
        return False
    return declaration.get(FROZEN_POLICY_CONTRACT) == FROZEN_POLICY_IDENTITY_SHA256


@dataclass(frozen=True)
class PolicyV2AnchorReadiness:
    contract: str
    policy_contract: str
    policy_identity_sha256: str
    rows: int
    observed_rows: int
    imputed_rows: int
    unavailable_rows: int
    nullable_skill_date_rows: int
    availability_preserved: bool
    anchor_api_explicit_policy_support: bool
    anchor_ready: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def policy_v2_anchor_readiness(
    frame: pd.DataFrame,
    *,
    expected_reward_reference_sha256: str,
    expected_donor_source_sha256: str,
    expected_support_inventory_sha256: str,
    expected_target_event_key: str,
    expected_target_event_start: object,
) -> PolicyV2AnchorReadiness:
    """Validate atoms and report fail-closed anchor readiness."""

    validated = validate_policy_v2_atom_frame(
        frame,
        expected_reward_reference_sha256=expected_reward_reference_sha256,
        expected_donor_source_sha256=expected_donor_source_sha256,
        expected_support_inventory_sha256=expected_support_inventory_sha256,
        expected_target_event_key=expected_target_event_key,
        expected_target_event_start=expected_target_event_start,
    )
    kinds = validated["donor_ability_value_kind"].astype(str)
    imputed = kinds.eq(donor_policy.IMPUTED_VALUE_KIND)
    observed = kinds.eq(donor_policy.OBSERVED_VALUE_KIND)
    available = v3._coerce_bool(
        validated["donor_ability_available"], "donor_ability_available"
    )
    original = v3._coerce_bool(
        validated["donor_ability_original_available"],
        "donor_ability_original_available",
    )
    explicit_support = _anchor_api_supports_policy_v2()
    blockers: list[str] = []
    if imputed.any() and not explicit_support:
        blockers.append(
            "anchor_api_lacks_explicit_frozen_policy_v2_neutral_zero_support"
        )
    return PolicyV2AnchorReadiness(
        contract=ATOM_SCHEMA_CONTRACT,
        policy_contract=FROZEN_POLICY_CONTRACT,
        policy_identity_sha256=FROZEN_POLICY_IDENTITY_SHA256,
        rows=int(len(validated)),
        observed_rows=int(observed.sum()),
        imputed_rows=int(imputed.sum()),
        unavailable_rows=int((~available).sum()),
        nullable_skill_date_rows=int(validated["donor_skill_date"].isna().sum()),
        availability_preserved=bool(np.array_equal(available, original)),
        anchor_api_explicit_policy_support=explicit_support,
        anchor_ready=not blockers,
        blockers=tuple(blockers),
    )


def assert_policy_v2_anchor_ready(
    frame: pd.DataFrame,
    **contract_kwargs: object,
) -> None:
    readiness = policy_v2_anchor_readiness(frame, **contract_kwargs)
    if not readiness.anchor_ready:
        raise PolicyV2ContractUnavailable(json.dumps(readiness.as_dict(), sort_keys=True))


def _canonical_join_timestamp(value: object, label: str) -> str:
    return v3._canonical_timestamp(value, label)


def attest_policy_v2_donor_rows(
    atoms: pd.DataFrame, applied_policy_rows: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact applied policy rows and seal a v4 atom frame.

    The input atom rows must already preserve the policy row's original
    availability and nullable date.  This function refuses to turn a present
    observation into an imputation or to erase/invent its date.
    """

    if not isinstance(atoms, pd.DataFrame) or atoms.empty:
        raise PolicyV2ContractUnavailable("Atoms must be a nonempty DataFrame")
    try:
        donor_policy.validate_applied_frame(applied_policy_rows)
    except donor_policy.DonorAbilityPolicyError as exc:
        raise PolicyV2ContractUnavailable(
            f"Cannot attest invalid policy-v2 rows: {exc}"
        ) from exc
    policy_rows = applied_policy_rows.copy(deep=True)
    if policy_rows.duplicated(list(_DONOR_IDENTITY_COLUMNS)).any():
        raise PolicyV2ContractUnavailable(
            "Applied policy rows duplicate donor event/player/category identity"
        )
    atom_identities = atoms[list(_DONOR_IDENTITY_COLUMNS)].drop_duplicates()
    policy_identities = policy_rows[list(_DONOR_IDENTITY_COLUMNS)].drop_duplicates()
    coverage = atom_identities.merge(
        policy_identities,
        on=list(_DONOR_IDENTITY_COLUMNS),
        how="outer",
        indicator=True,
    )
    if not coverage["_merge"].eq("both").all():
        raise PolicyV2ContractUnavailable(
            "Applied policy rows do not exactly cover atom donor identities"
        )

    attach = policy_rows.rename(
        columns={
            "bridge_row_id": "donor_ability_bridge_row_id",
            "donor_skill_date": "policy_donor_skill_date",
            "donor_event_start": "policy_donor_event_start",
            "target_event_key": "policy_target_event_key",
            "target_event_start": "policy_target_event_start",
            "donor_ability_available": "policy_donor_ability_available",
            "donor_ability_sg": "policy_donor_ability_sg",
        }
    )
    overlap = set(atoms.columns).intersection(attach.columns).difference(
        _DONOR_IDENTITY_COLUMNS
    )
    if overlap:
        attach = attach.drop(columns=sorted(overlap))
    out = atoms.merge(
        attach,
        on=list(_DONOR_IDENTITY_COLUMNS),
        how="left",
        validate="many_to_one",
    )

    current_available = v3._coerce_bool(
        out["donor_ability_available"], "donor_ability_available"
    )
    policy_available = v3._coerce_bool(
        out["policy_donor_ability_available"],
        "policy donor_ability_available",
    )
    if not np.array_equal(current_available, policy_available):
        raise PolicyV2ContractUnavailable(
            "Atom input does not preserve original donor availability"
        )
    for atom_date, policy_date in zip(
        out["donor_skill_date"], out["policy_donor_skill_date"]
    ):
        if _optional_timestamp(atom_date, "donor_skill_date") != _optional_timestamp(
            policy_date, "policy donor_skill_date"
        ):
            raise PolicyV2ContractUnavailable(
                "Atom input does not preserve original nullable donor skill date"
            )
    for atom_start, policy_start in zip(
        out["donor_event_start"], out["policy_donor_event_start"]
    ):
        if _canonical_join_timestamp(atom_start, "donor_event_start") != (
            _canonical_join_timestamp(policy_start, "policy donor_event_start")
        ):
            raise PolicyV2ContractUnavailable("Donor event start changed during attestation")
    for atom_key, policy_key in zip(
        out["target_event_key"], out["policy_target_event_key"]
    ):
        if v3._canonical_text(atom_key, "target_event_key") != v3._canonical_text(
            policy_key, "policy target_event_key"
        ):
            raise PolicyV2ContractUnavailable("Target event key changed during attestation")
    for atom_start, policy_start in zip(
        out["target_event_start"], out["policy_target_event_start"]
    ):
        if _canonical_join_timestamp(atom_start, "target_event_start") != (
            _canonical_join_timestamp(policy_start, "policy target_event_start")
        ):
            raise PolicyV2ContractUnavailable("Target event start changed during attestation")

    current_ability = pd.to_numeric(out["donor_ability_sg"], errors="coerce")
    applied_ability = pd.to_numeric(out["policy_donor_ability_sg"], errors="coerce")
    observed = policy_available
    imputed = ~policy_available
    if observed.any() and not np.equal(
        current_ability.to_numpy(float)[observed],
        applied_ability.to_numpy(float)[observed],
    ).all():
        raise PolicyV2ContractUnavailable(
            "Observed atom donor ability differs from policy-v2 applied value"
        )
    imputed_input = current_ability.to_numpy(float)[imputed]
    if imputed.any() and not np.logical_or(
        np.isnan(imputed_input), imputed_input == 0.0
    ).all():
        raise PolicyV2ContractUnavailable(
            "Unavailable atom carries a nonzero donor ability before policy attestation"
        )

    out["donor_ability_sg"] = applied_ability.to_numpy(float)
    out["donor_ability_available"] = policy_available
    out["donor_skill_date"] = out["policy_donor_skill_date"]
    out["donor_ability_original_available"] = policy_available
    out["donor_ability_original_skill_date"] = out["policy_donor_skill_date"]
    out["donor_ability_bridge_row_id"] = out["donor_ability_bridge_row_id"]
    out["donor_ability_snapshot_kind"] = np.where(
        policy_available, OBSERVED_SNAPSHOT_KIND, IMPUTED_SNAPSHOT_KIND
    )
    out["atom_schema_contract"] = ATOM_SCHEMA_CONTRACT
    out["donor_ability_policy_contract"] = FROZEN_POLICY_CONTRACT
    out["donor_ability_policy_status"] = FROZEN_POLICY_STATUS
    out["donor_ability_policy_source_descriptor"] = (
        FROZEN_POLICY_SOURCE_DESCRIPTOR
    )
    out["donor_ability_policy_source_sha256"] = FROZEN_POLICY_SOURCE_SHA256
    out = out.drop(
        columns=[
            "policy_donor_skill_date",
            "policy_donor_event_start",
            "policy_target_event_key",
            "policy_target_event_start",
            "policy_donor_ability_available",
            "policy_donor_ability_sg",
        ]
    )
    return seal_policy_v2_contracts(out)


def build_policy_rows_from_atoms(atoms: pd.DataFrame) -> pd.DataFrame:
    """Build and apply frozen v2 rows from honest atom availability/date fields."""

    rows: list[dict[str, object]] = []
    for identity, group in atoms.groupby(
        list(_DONOR_IDENTITY_COLUMNS), observed=True, sort=True
    ):
        if group["donor_ability_available"].nunique(dropna=False) != 1:
            raise PolicyV2ContractUnavailable(
                "One donor identity mixes availability before policy application"
            )
        if group["donor_skill_date"].map(
            lambda value: _optional_timestamp(value, "donor_skill_date")
        ).nunique(dropna=False) != 1:
            raise PolicyV2ContractUnavailable(
                "One donor identity has conflicting original skill dates"
            )
        available = v3._canonical_bool(
            group["donor_ability_available"].iloc[0], "donor_ability_available"
        )
        ability_values = pd.to_numeric(group["donor_ability_sg"], errors="coerce")
        if available:
            finite = ability_values[np.isfinite(ability_values)]
            if len(finite) != len(group) or finite.nunique() != 1:
                raise PolicyV2ContractUnavailable(
                    "Observed donor identity has conflicting applied abilities"
                )
            raw: float | None = float(finite.iloc[0])
        else:
            values = ability_values.to_numpy(float)
            if not np.logical_or(np.isnan(values), values == 0.0).all():
                raise PolicyV2ContractUnavailable(
                    "Unavailable donor identity carries a nonzero source ability"
                )
            raw = None
        event_key, player_id, category = map(str, identity)
        bridge_digest = _stable_sha(
            {
                "contract": "policy_v2_bridge_identity_v1",
                "donor_event_key": event_key,
                "donor_player_id": player_id,
                "category": category,
            }
        )
        rows.append(
            {
                "bridge_row_id": f"policy-v2:{bridge_digest}",
                "category": category,
                "donor_player_id": player_id,
                "donor_event_key": event_key,
                "donor_event_start": group["donor_event_start"].iloc[0],
                "target_event_key": group["target_event_key"].iloc[0],
                "target_event_start": group["target_event_start"].iloc[0],
                "donor_skill_date": group["donor_skill_date"].iloc[0],
                "donor_ability_source_column": donor_policy.CATEGORY_ABILITY_MAPPING[
                    category
                ],
                "donor_ability_available": available,
                "donor_ability_raw_strict_prior": raw,
            }
        )
    try:
        return donor_policy.apply_frozen_policy(pd.DataFrame(rows))
    except donor_policy.DonorAbilityPolicyError as exc:
        raise PolicyV2ContractUnavailable(
            f"Atom donor evidence violates frozen policy v2: {exc}"
        ) from exc


def load_policy_v2_atom_contract(
    path: Path,
    **contract_kwargs: object,
) -> pd.DataFrame:
    """Load an authoritative Parquet v4 contract; CSV is always rejected."""

    contract_path = Path(path)
    if contract_path.suffix.casefold() not in {".parquet", ".pq"}:
        raise PolicyV2ContractUnavailable(
            "V4 policy-v2 atom contracts require Parquet; CSV is disabled"
        )
    return validate_policy_v2_atom_frame(
        pd.read_parquet(contract_path), **contract_kwargs
    )


__all__ = [
    "ATOM_SCHEMA_CONTRACT",
    "DONOR_SOURCE_CONTRACT",
    "FROZEN_POLICY_CONTRACT",
    "FROZEN_POLICY_IDENTITY_SHA256",
    "FROZEN_POLICY_SOURCE_SHA256",
    "IMPUTED_SNAPSHOT_KIND",
    "OBSERVED_SNAPSHOT_KIND",
    "PolicyV2AnchorReadiness",
    "PolicyV2ContractUnavailable",
    "REQUIRED_ATOM_COLUMNS",
    "SUPPORT_INVENTORY_CONTRACT",
    "assert_policy_v2_anchor_ready",
    "attest_policy_v2_donor_rows",
    "build_policy_rows_from_atoms",
    "donor_source_sha256",
    "load_policy_v2_atom_contract",
    "policy_v2_anchor_readiness",
    "quality_input_sha256",
    "seal_policy_v2_contracts",
    "support_inventory_sha256",
    "validate_frozen_policy_runtime",
    "validate_policy_v2_atom_frame",
]

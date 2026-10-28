"""Research-only policy-v2 admission seam for whole-atom quality weighting.

This module does not solve a mean anchor and does not run a physical chain.
It gates one existing numerical operation, ``whole_atom_probabilities``, behind
all of the following independently checked evidence:

* the exact frozen mean-anchor, adapter-v4, v3 physical-adapter, and
  donor-policy source hashes;
* the sole supported frozen policy contract-to-identity mapping;
* adapter-v4 validation against one source-pinned synthetic inventory hash;
* one source-pinned synthetic ordered root-support list and digest; and
* full reachability plus homogeneous observed/imputed donor quality groups.

The approval-lock identity is closed inside this reviewed source.  Admission
accepts no caller-supplied expected digest.  Any fresh, resealed, or real-chain
manifest fails until a future review explicitly changes the pinned lock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys as _sys
from types import MappingProxyType
from typing import Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in _sys.path:
    _sys.path.insert(0, str(ARCHIVE))

from hole_sim_donor_ability_policy import (  # noqa: E402
    IMPUTED_VALUE_KIND as _IMPUTED_VALUE_KIND,
    OBSERVED_VALUE_KIND as _OBSERVED_VALUE_KIND,
    POLICY_CONTRACT as _POLICY_CONTRACT,
    policy_payload as _policy_payload,
)
from hole_sim_transition_mean_anchor import (  # noqa: E402
    ANCHOR_CONTRACT as _ANCHOR_CONTRACT,
    ATOM_WEIGHT_CONTRACT as _ATOM_WEIGHT_CONTRACT,
    AtomMassResult as _AtomMassResult,
    TransitionQualityParameters as _TransitionQualityParameters,
    whole_atom_probabilities as _whole_atom_probabilities,
)
from hole_sim_transition_mean_chain_adapter import (  # noqa: E402
    ATOM_SCHEMA_CONTRACT as _V3_ADAPTER_CONTRACT,
    _canonical_text,
    _canonical_timestamp,
    _coerce_bool,
)
from hole_sim_transition_mean_chain_adapter_v4 import (  # noqa: E402
    ATOM_SCHEMA_CONTRACT as _V4_ADAPTER_CONTRACT,
    FROZEN_POLICY_CONTRACT as _FROZEN_POLICY_CONTRACT,
    FROZEN_POLICY_IDENTITY_SHA256 as _FROZEN_POLICY_IDENTITY_SHA256,
    FROZEN_POLICY_SOURCE_DESCRIPTOR as _FROZEN_POLICY_SOURCE_DESCRIPTOR,
    FROZEN_POLICY_SOURCE_SHA256 as _FROZEN_POLICY_SOURCE_SHA256,
    PolicyV2ContractUnavailable as _V4ContractUnavailable,
    SUPPORT_INVENTORY_CONTRACT as _SUPPORT_INVENTORY_CONTRACT,
    validate_frozen_policy_runtime as _validate_frozen_policy_runtime,
    validate_policy_v2_atom_frame as _validate_policy_v2_atom_frame,
)


SEAM_CONTRACT = "trusted_policy_v2_whole_atom_weighting_seam_v1"
AUTHORITY_CONTRACT = "externally_attested_policy_v2_graph_authority_v1"
ROOT_SUPPORT_CONTRACT = "ordered_policy_v2_root_support_inventory_v1"
RUNTIME_BINDING_CONTRACT = "frozen_policy_v2_weighting_runtime_binding_v1"

ANCHOR_SOURCE_DESCRIPTOR = "archive/hole_sim_transition_mean_anchor.py"
ANCHOR_SOURCE_SHA256 = (
    "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
)
ADAPTER_SOURCE_DESCRIPTOR = "archive/hole_sim_transition_mean_chain_adapter_v4.py"
ADAPTER_SOURCE_SHA256 = (
    "e6ab1aa587149af2143ed122ae1307994c9d3277694e153f444f07f30daa711c"
)
V3_ADAPTER_SOURCE_DESCRIPTOR = "archive/hole_sim_transition_mean_chain_adapter.py"
V3_ADAPTER_SOURCE_SHA256 = (
    "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff"
)
POLICY_SOURCE_DESCRIPTOR = _FROZEN_POLICY_SOURCE_DESCRIPTOR
POLICY_SOURCE_SHA256 = _FROZEN_POLICY_SOURCE_SHA256

EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT = MappingProxyType(
    {_FROZEN_POLICY_CONTRACT: _FROZEN_POLICY_IDENTITY_SHA256}
)

APPROVAL_LOCK_CONTRACT = "closed_synthetic_policy_v2_weighting_approval_lock_v1"
APPROVAL_LOCK_STATUS = "synthetic_fixture_only_real_chain_hold"
APPROVED_AUTHORITY_SOURCE_DESCRIPTOR = (
    "closed/synthetic-policy-v2-weighting-fixture-v1"
)
APPROVED_SUPPORT_INVENTORY_SHA256 = (
    "33bf857bcfe7e3dd3bd8f5b6237d41a8bf4f629c1dabce7ac44d6135320f26f8"
)
APPROVED_ROOT_SUPPORT_SHA256 = (
    "b0bba4a8150b8363491692f8636cb6dbb8b0bab9d94559c0ff8b55643b233d12"
)
# Closed approval identities; these are reports, not caller inputs.
APPROVED_AUTHORITY_MANIFEST_SHA256 = (
    "e6177dc0ed5f7545af06d4f61a23dd8458abdcbb98f2cda1f52e981f0330562c"
)
APPROVAL_LOCK_SHA256 = (
    "626bedaa62c23604bc51e3fb0cc04e40b349f8f8697745ff55450816bebe52ab"
)


def _closed_approval_identity() -> dict[str, str]:
    """Return source literals used by admission, separate from report constants."""

    return {
        "authority_source_descriptor": (
            "closed/synthetic-policy-v2-weighting-fixture-v1"
        ),
        "authority_manifest_sha256": (
            "e6177dc0ed5f7545af06d4f61a23dd8458abdcbb98f2cda1f52e981f0330562c"
        ),
        "support_inventory_sha256": (
            "33bf857bcfe7e3dd3bd8f5b6237d41a8bf4f629c1dabce7ac44d6135320f26f8"
        ),
        "root_support_sha256": (
            "b0bba4a8150b8363491692f8636cb6dbb8b0bab9d94559c0ff8b55643b233d12"
        ),
        "approval_lock_sha256": (
            "626bedaa62c23604bc51e3fb0cc04e40b349f8f8697745ff55450816bebe52ab"
        ),
    }

AUTHORITY_MANIFEST_FIELDS = (
    "contract",
    "research_only",
    "production_enabled",
    "authority_source_descriptor",
    "policy_support_mapping",
    "anchor_contract",
    "anchor_source_descriptor",
    "anchor_source_sha256",
    "adapter_contract",
    "adapter_source_descriptor",
    "adapter_source_sha256",
    "v3_adapter_contract",
    "v3_adapter_source_descriptor",
    "v3_adapter_source_sha256",
    "policy_contract",
    "policy_identity_sha256",
    "policy_source_descriptor",
    "policy_source_sha256",
    "support_inventory_contract",
    "support_inventory_sha256",
    "root_support_contract",
    "root_support_sha256",
    "start_support_keys",
    "expected_round_holes",
    "reward_reference_sha256",
    "donor_source_sha256",
    "target_event_key",
    "target_event_start",
)


class PolicyV2AnchorContractUnavailable(ValueError):
    """Raised when policy-v2 weighting lacks independently trusted evidence."""


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
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _descriptor(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PolicyV2AnchorContractUnavailable(f"{label} must be text")
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        raise PolicyV2AnchorContractUnavailable(f"{label} cannot be empty")
    return text


def _canonical_root_keys(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PolicyV2AnchorContractUnavailable(
            "start_support_keys must be an ordered sequence"
        )
    roots = tuple(
        _canonical_text(value, "start_support_key") for value in values
    )
    if not roots:
        raise PolicyV2AnchorContractUnavailable(
            "Trusted root-support list cannot be empty"
        )
    if len(set(roots)) != len(roots):
        raise PolicyV2AnchorContractUnavailable(
            "Trusted root-support list contains duplicates"
        )
    return roots


def _root_support_sha256(
    *,
    support_inventory_sha256: str,
    start_support_keys: Sequence[str],
    expected_round_holes: int,
    target_event_key: str,
    target_event_start: str,
) -> str:
    return _stable_sha(
        {
            "contract": ROOT_SUPPORT_CONTRACT,
            "support_inventory_sha256": support_inventory_sha256,
            "start_support_keys": list(start_support_keys),
            "expected_round_holes": int(expected_round_holes),
            "target_event_key": target_event_key,
            "target_event_start": target_event_start,
        }
    )


def _approval_lock_payload() -> dict[str, object]:
    closed = _closed_approval_identity()
    return {
        "contract": APPROVAL_LOCK_CONTRACT,
        "status": APPROVAL_LOCK_STATUS,
        "research_only": True,
        "production_enabled": False,
        "authority_source_descriptor": closed["authority_source_descriptor"],
        "authority_manifest_sha256": closed["authority_manifest_sha256"],
        "support_inventory_sha256": closed["support_inventory_sha256"],
        "root_support_sha256": closed["root_support_sha256"],
        "anchor_source_sha256": ANCHOR_SOURCE_SHA256,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "v3_adapter_source_sha256": V3_ADAPTER_SOURCE_SHA256,
        "policy_source_sha256": POLICY_SOURCE_SHA256,
    }


def _validate_closed_approval_lock() -> str:
    closed = _closed_approval_identity()
    reports = {
        "authority_source_descriptor": APPROVED_AUTHORITY_SOURCE_DESCRIPTOR,
        "authority_manifest_sha256": APPROVED_AUTHORITY_MANIFEST_SHA256,
        "support_inventory_sha256": APPROVED_SUPPORT_INVENTORY_SHA256,
        "root_support_sha256": APPROVED_ROOT_SUPPORT_SHA256,
        "approval_lock_sha256": APPROVAL_LOCK_SHA256,
    }
    if reports != closed:
        raise PolicyV2AnchorContractUnavailable(
            "Public approval-lock report constants changed"
        )
    if not _is_sha256(closed["authority_manifest_sha256"]):
        raise PolicyV2AnchorContractUnavailable(
            "No closed synthetic authority manifest is pinned"
        )
    if not _is_sha256(closed["approval_lock_sha256"]):
        raise PolicyV2AnchorContractUnavailable("Closed approval-lock SHA is invalid")
    observed = _stable_sha(_approval_lock_payload())
    if observed != closed["approval_lock_sha256"]:
        raise PolicyV2AnchorContractUnavailable(
            "Closed approval-lock identity does not match its frozen payload"
        )
    return observed


def _runtime_binding_payload() -> dict[str, object]:
    """Fail if any frozen numerical or admission source changed."""

    if _ANCHOR_CONTRACT != "joint_recursive_production_category_mean_anchor_v1":
        raise PolicyV2AnchorContractUnavailable("Frozen mean-anchor contract changed")
    if _ATOM_WEIGHT_CONTRACT != "whole_atom_donor_detilt_target_reapply_lambda_v1":
        raise PolicyV2AnchorContractUnavailable("Frozen atom-weight contract changed")
    if _V4_ADAPTER_CONTRACT != "complete_physical_category_sg_atom_v4_policy2":
        raise PolicyV2AnchorContractUnavailable("Adapter-v4 atom contract changed")
    if _V3_ADAPTER_CONTRACT != "complete_physical_category_sg_atom_v3":
        raise PolicyV2AnchorContractUnavailable("V3 physical-adapter contract changed")
    if _SUPPORT_INVENTORY_CONTRACT != "complete_policy_v2_support_inventory_v1":
        raise PolicyV2AnchorContractUnavailable("Adapter-v4 inventory contract changed")
    if dict(EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT) != {
        _FROZEN_POLICY_CONTRACT: _FROZEN_POLICY_IDENTITY_SHA256
    }:
        raise PolicyV2AnchorContractUnavailable(
            "Policy-v2 support declaration is not the sole frozen mapping"
        )
    if _POLICY_CONTRACT != _FROZEN_POLICY_CONTRACT:
        raise PolicyV2AnchorContractUnavailable("Frozen donor-policy contract changed")
    if (
        str(_policy_payload().get("policy_identity_sha256"))
        != _FROZEN_POLICY_IDENTITY_SHA256
    ):
        raise PolicyV2AnchorContractUnavailable("Frozen donor-policy identity changed")

    observed = {
        "anchor_source_sha256": _file_sha256(
            ARCHIVE / "hole_sim_transition_mean_anchor.py"
        ),
        "adapter_source_sha256": _file_sha256(
            ARCHIVE / "hole_sim_transition_mean_chain_adapter_v4.py"
        ),
        "v3_adapter_source_sha256": _file_sha256(
            ARCHIVE / "hole_sim_transition_mean_chain_adapter.py"
        ),
        "policy_source_sha256": _file_sha256(
            ARCHIVE / "hole_sim_donor_ability_policy.py"
        ),
    }
    expected = {
        "anchor_source_sha256": ANCHOR_SOURCE_SHA256,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "v3_adapter_source_sha256": V3_ADAPTER_SOURCE_SHA256,
        "policy_source_sha256": POLICY_SOURCE_SHA256,
    }
    for label, expected_sha in expected.items():
        if observed[label] != expected_sha:
            raise PolicyV2AnchorContractUnavailable(
                f"Frozen runtime {label} changed: expected={expected_sha}, "
                f"observed={observed[label]}"
            )
    try:
        _validate_frozen_policy_runtime()
    except _V4ContractUnavailable as exc:
        raise PolicyV2AnchorContractUnavailable(
            f"Adapter-v4 rejected frozen policy runtime: {exc}"
        ) from exc
    return {
        "contract": RUNTIME_BINDING_CONTRACT,
        "anchor_contract": _ANCHOR_CONTRACT,
        "atom_weight_contract": _ATOM_WEIGHT_CONTRACT,
        "anchor_source_descriptor": ANCHOR_SOURCE_DESCRIPTOR,
        "anchor_source_sha256": observed["anchor_source_sha256"],
        "adapter_contract": _V4_ADAPTER_CONTRACT,
        "adapter_source_descriptor": ADAPTER_SOURCE_DESCRIPTOR,
        "adapter_source_sha256": observed["adapter_source_sha256"],
        "v3_adapter_contract": _V3_ADAPTER_CONTRACT,
        "v3_adapter_source_descriptor": V3_ADAPTER_SOURCE_DESCRIPTOR,
        "v3_adapter_source_sha256": observed["v3_adapter_source_sha256"],
        "policy_support_mapping": dict(EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT),
        "policy_source_descriptor": POLICY_SOURCE_DESCRIPTOR,
        "policy_source_sha256": observed["policy_source_sha256"],
        "approval_lock_sha256": _validate_closed_approval_lock(),
    }


def validate_frozen_runtime_bindings() -> dict[str, object]:
    """Return the exact runtime-source binding after content verification."""

    payload = _runtime_binding_payload()
    return {**payload, "runtime_binding_sha256": _stable_sha(payload)}


def _canonical_authority_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(manifest, Mapping):
        raise PolicyV2AnchorContractUnavailable("Graph authority manifest must be an object")
    supplied = set(manifest)
    required = set(AUTHORITY_MANIFEST_FIELDS)
    if supplied != required:
        raise PolicyV2AnchorContractUnavailable(
            "Graph authority manifest fields differ: "
            f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
        )
    if manifest["contract"] != AUTHORITY_CONTRACT:
        raise PolicyV2AnchorContractUnavailable("Wrong graph-authority contract")
    if manifest["research_only"] is not True:
        raise PolicyV2AnchorContractUnavailable("Graph authority is not research-only")
    if manifest["production_enabled"] is not False:
        raise PolicyV2AnchorContractUnavailable("Graph authority enables production")
    mapping = manifest["policy_support_mapping"]
    if not isinstance(mapping, Mapping) or dict(mapping) != dict(
        EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT
    ):
        raise PolicyV2AnchorContractUnavailable(
            "Wrong or expanded donor-ability policy support mapping"
        )

    exact = {
        "anchor_contract": _ANCHOR_CONTRACT,
        "anchor_source_descriptor": ANCHOR_SOURCE_DESCRIPTOR,
        "anchor_source_sha256": ANCHOR_SOURCE_SHA256,
        "adapter_contract": _V4_ADAPTER_CONTRACT,
        "adapter_source_descriptor": ADAPTER_SOURCE_DESCRIPTOR,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "v3_adapter_contract": _V3_ADAPTER_CONTRACT,
        "v3_adapter_source_descriptor": V3_ADAPTER_SOURCE_DESCRIPTOR,
        "v3_adapter_source_sha256": V3_ADAPTER_SOURCE_SHA256,
        "policy_contract": _FROZEN_POLICY_CONTRACT,
        "policy_identity_sha256": _FROZEN_POLICY_IDENTITY_SHA256,
        "policy_source_descriptor": POLICY_SOURCE_DESCRIPTOR,
        "policy_source_sha256": POLICY_SOURCE_SHA256,
        "support_inventory_contract": _SUPPORT_INVENTORY_CONTRACT,
        "root_support_contract": ROOT_SUPPORT_CONTRACT,
    }
    for field_name, expected in exact.items():
        if manifest[field_name] != expected:
            raise PolicyV2AnchorContractUnavailable(
                f"Graph authority has wrong {field_name}"
            )
    for field_name in (
        "support_inventory_sha256",
        "root_support_sha256",
        "reward_reference_sha256",
        "donor_source_sha256",
    ):
        if not _is_sha256(manifest[field_name]):
            raise PolicyV2AnchorContractUnavailable(
                f"Graph authority {field_name} is not a lowercase SHA-256"
            )
    roots = _canonical_root_keys(manifest["start_support_keys"])
    holes = manifest["expected_round_holes"]
    if isinstance(holes, bool) or not isinstance(holes, (int, np.integer)):
        raise PolicyV2AnchorContractUnavailable("expected_round_holes must be an integer")
    holes = int(holes)
    if holes <= 0 or holes != len(roots):
        raise PolicyV2AnchorContractUnavailable(
            "Expected round-hole count does not equal the trusted root list"
        )
    target_key = _canonical_text(manifest["target_event_key"], "target_event_key")
    target_start = _canonical_timestamp(
        manifest["target_event_start"], "target_event_start"
    )
    root_sha = _root_support_sha256(
        support_inventory_sha256=str(manifest["support_inventory_sha256"]),
        start_support_keys=roots,
        expected_round_holes=holes,
        target_event_key=target_key,
        target_event_start=target_start,
    )
    if manifest["root_support_sha256"] != root_sha:
        raise PolicyV2AnchorContractUnavailable(
            "Root-support digest is stale or not bound to ordered roots/inventory"
        )
    return {
        "contract": AUTHORITY_CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "authority_source_descriptor": _descriptor(
            manifest["authority_source_descriptor"], "authority_source_descriptor"
        ),
        "policy_support_mapping": dict(EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT),
        **exact,
        "support_inventory_sha256": str(manifest["support_inventory_sha256"]),
        "root_support_sha256": root_sha,
        "start_support_keys": list(roots),
        "expected_round_holes": holes,
        "reward_reference_sha256": str(manifest["reward_reference_sha256"]),
        "donor_source_sha256": str(manifest["donor_source_sha256"]),
        "target_event_key": target_key,
        "target_event_start": target_start,
    }


@dataclass(frozen=True)
class _TrustedGraphAuthority:
    manifest_sha256: str
    authority_source_descriptor: str
    support_inventory_sha256: str
    root_support_sha256: str
    start_support_keys: tuple[str, ...]
    expected_round_holes: int
    reward_reference_sha256: str
    donor_source_sha256: str
    target_event_key: str
    target_event_start: str
    runtime_binding_sha256: str


def _authority_content_sha256(authority: object) -> str | None:
    """Reconstruct the full manifest so a forged private object cannot confer trust."""

    if not isinstance(authority, _TrustedGraphAuthority):
        return None
    payload = {
        "contract": AUTHORITY_CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "authority_source_descriptor": authority.authority_source_descriptor,
        "policy_support_mapping": dict(EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT),
        "anchor_contract": _ANCHOR_CONTRACT,
        "anchor_source_descriptor": ANCHOR_SOURCE_DESCRIPTOR,
        "anchor_source_sha256": ANCHOR_SOURCE_SHA256,
        "adapter_contract": _V4_ADAPTER_CONTRACT,
        "adapter_source_descriptor": ADAPTER_SOURCE_DESCRIPTOR,
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "v3_adapter_contract": _V3_ADAPTER_CONTRACT,
        "v3_adapter_source_descriptor": V3_ADAPTER_SOURCE_DESCRIPTOR,
        "v3_adapter_source_sha256": V3_ADAPTER_SOURCE_SHA256,
        "policy_contract": _FROZEN_POLICY_CONTRACT,
        "policy_identity_sha256": _FROZEN_POLICY_IDENTITY_SHA256,
        "policy_source_descriptor": POLICY_SOURCE_DESCRIPTOR,
        "policy_source_sha256": POLICY_SOURCE_SHA256,
        "support_inventory_contract": _SUPPORT_INVENTORY_CONTRACT,
        "support_inventory_sha256": authority.support_inventory_sha256,
        "root_support_contract": ROOT_SUPPORT_CONTRACT,
        "root_support_sha256": authority.root_support_sha256,
        "start_support_keys": list(authority.start_support_keys),
        "expected_round_holes": authority.expected_round_holes,
        "reward_reference_sha256": authority.reward_reference_sha256,
        "donor_source_sha256": authority.donor_source_sha256,
        "target_event_key": authority.target_event_key,
        "target_event_start": authority.target_event_start,
    }
    return _stable_sha(payload)


def _is_closed_authority(authority: object) -> bool:
    closed_sha = _closed_approval_identity()["authority_manifest_sha256"]
    return bool(
        isinstance(authority, _TrustedGraphAuthority)
        and authority.manifest_sha256 == closed_sha
        and _authority_content_sha256(authority) == closed_sha
    )


def _trusted_graph_authority(
    manifest: Mapping[str, object],
) -> _TrustedGraphAuthority:
    runtime = validate_frozen_runtime_bindings()
    canonical = _canonical_authority_manifest(manifest)
    observed_sha = _stable_sha(canonical)
    closed = _closed_approval_identity()
    closed_values = {
        "authority_source_descriptor": closed["authority_source_descriptor"],
        "support_inventory_sha256": closed["support_inventory_sha256"],
        "root_support_sha256": closed["root_support_sha256"],
    }
    for field_name, expected in closed_values.items():
        if canonical[field_name] != expected:
            raise PolicyV2AnchorContractUnavailable(
                f"Candidate graph is outside the closed synthetic approval lock: "
                f"{field_name}"
            )
    if observed_sha != closed["authority_manifest_sha256"]:
        raise PolicyV2AnchorContractUnavailable(
            "Candidate authority manifest is not named by the closed synthetic "
            "approval lock"
        )
    return _TrustedGraphAuthority(
        manifest_sha256=observed_sha,
        authority_source_descriptor=str(canonical["authority_source_descriptor"]),
        support_inventory_sha256=str(canonical["support_inventory_sha256"]),
        root_support_sha256=str(canonical["root_support_sha256"]),
        start_support_keys=tuple(canonical["start_support_keys"]),
        expected_round_holes=int(canonical["expected_round_holes"]),
        reward_reference_sha256=str(canonical["reward_reference_sha256"]),
        donor_source_sha256=str(canonical["donor_source_sha256"]),
        target_event_key=str(canonical["target_event_key"]),
        target_event_start=str(canonical["target_event_start"]),
        runtime_binding_sha256=str(runtime["runtime_binding_sha256"]),
    )


def _validate_roots_and_policy_groups(
    atoms: pd.DataFrame,
    authority: _TrustedGraphAuthority,
) -> pd.DataFrame:
    if not _is_closed_authority(authority):
        raise PolicyV2AnchorContractUnavailable("Untrusted graph authority")
    try:
        out = _validate_policy_v2_atom_frame(
            atoms,
            expected_reward_reference_sha256=authority.reward_reference_sha256,
            expected_donor_source_sha256=authority.donor_source_sha256,
            expected_support_inventory_sha256=authority.support_inventory_sha256,
            expected_target_event_key=authority.target_event_key,
            expected_target_event_start=authority.target_event_start,
        )
    except _V4ContractUnavailable as exc:
        raise PolicyV2AnchorContractUnavailable(
            f"Adapter-v4 admission failed: {exc}"
        ) from exc

    kinds = out["donor_ability_value_kind"].astype(str)
    imputed = kinds.eq(_IMPUTED_VALUE_KIND)
    observed = kinds.eq(_OBSERVED_VALUE_KIND)
    if not (imputed | observed).all():
        raise PolicyV2AnchorContractUnavailable("Unexpected donor-ability value kind")
    available = _coerce_bool(
        out["donor_ability_available"], "donor_ability_available"
    )
    original = _coerce_bool(
        out["donor_ability_original_available"],
        "donor_ability_original_available",
    )
    if not np.array_equal(available, original):
        raise PolicyV2AnchorContractUnavailable(
            "Donor availability changed after policy application"
        )
    if available[imputed.to_numpy(bool)].any():
        raise PolicyV2AnchorContractUnavailable(
            "Imputed donor rows became available"
        )
    if not available[observed.to_numpy(bool)].all():
        raise PolicyV2AnchorContractUnavailable(
            "Observed donor rows became unavailable"
        )
    abilities = pd.to_numeric(out["donor_ability_sg"], errors="coerce").to_numpy(float)
    imputed_values = abilities[imputed.to_numpy(bool)]
    if imputed_values.size and (
        not np.equal(imputed_values, 0.0).all() or np.signbit(imputed_values).any()
    ):
        raise PolicyV2AnchorContractUnavailable(
            "Imputed donor rows do not carry exact positive numeric zero"
        )
    for group_id, group in out.groupby(
        "donor_quality_group_id", observed=True, sort=False
    ):
        if group["donor_ability_value_kind"].nunique(dropna=False) != 1:
            raise PolicyV2AnchorContractUnavailable(
                f"Donor quality group {group_id!r} mixes observed and imputed rows"
            )

    supports = {
        str(key): group
        for key, group in out.groupby("support_key", observed=True, sort=False)
    }
    missing_roots = sorted(set(authority.start_support_keys).difference(supports))
    if missing_roots:
        raise PolicyV2AnchorContractUnavailable(
            f"Trusted root supports are missing: {missing_roots}"
        )
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visited:
            return
        visited.add(key)
        group = supports[key]
        children = set(group.loc[~group["terminal"], "next_support_key"].astype(str))
        for child in children:
            visit(child)

    for root in authority.start_support_keys:
        visit(root)
    unreachable = sorted(set(supports).difference(visited))
    if unreachable:
        raise PolicyV2AnchorContractUnavailable(
            f"Authorized atom inventory has unreachable supports: {unreachable[:8]}"
        )
    return out


@dataclass(frozen=True)
class PolicyV2WeightingReadiness:
    contract: str
    approval_lock_sha256: str
    approval_lock_status: str
    authority_manifest_sha256: str
    runtime_binding_sha256: str
    support_inventory_sha256: str
    root_support_sha256: str
    rows: int
    supports: int
    roots: int
    observed_rows: int
    imputed_rows: int
    unavailable_rows: int
    nullable_skill_date_rows: int
    availability_preserved: bool
    synthetic_fixture_only: bool
    weighting_ready: bool
    real_chain_ready: bool
    integration_ready: bool
    integration_blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PolicyV2WeightingSession:
    """Opaque validated capability for analytical whole-atom weighting only."""

    __slots__ = ("_atoms", "_authority", "_readiness")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise PolicyV2AnchorContractUnavailable(
            "Weighting sessions must be minted by admit_policy_v2_weighting"
        )

    @property
    def readiness(self) -> PolicyV2WeightingReadiness:
        return self._readiness

    @property
    def atoms(self) -> pd.DataFrame:
        """Return a defensive copy; availability remains evidence, not a mask."""

        return self._atoms.copy(deep=True)

    def weight_support(
        self,
        support_key: str,
        parameters: _TransitionQualityParameters,
    ) -> _AtomMassResult:
        """Weight one validated support through the unchanged frozen primitive."""

        if not _is_closed_authority(self._authority):
            raise PolicyV2AnchorContractUnavailable("Unvalidated weighting session")
        runtime = validate_frozen_runtime_bindings()
        if runtime["runtime_binding_sha256"] != self._authority.runtime_binding_sha256:
            raise PolicyV2AnchorContractUnavailable(
                "Runtime binding changed after graph admission"
            )
        atoms = _validate_roots_and_policy_groups(self._atoms, self._authority)
        key = _canonical_text(support_key, "support_key")
        group = atoms.loc[atoms["support_key"].eq(key)]
        if group.empty:
            raise PolicyV2AnchorContractUnavailable(
                f"Requested support is not authorized: {key!r}"
            )
        if not isinstance(parameters, _TransitionQualityParameters):
            raise TypeError("parameters must be TransitionQualityParameters")
        if parameters.reward_reference_sha256 != self._authority.reward_reference_sha256:
            raise PolicyV2AnchorContractUnavailable(
                "Quality parameters use an unauthorized reward reference"
            )
        imputed = group["donor_ability_value_kind"].eq(
            _IMPUTED_VALUE_KIND
        ).to_numpy(bool)
        abilities = group["donor_ability_sg"].to_numpy(float)
        if imputed.any() and (
            not np.equal(abilities[imputed], 0.0).all()
            or np.signbit(abilities[imputed]).any()
        ):
            raise PolicyV2AnchorContractUnavailable(
                "Imputed donor values changed before numerical weighting"
            )
        result = _whole_atom_probabilities(
            base_mass=group["quality_base_mass"].to_numpy(float),
            fixed_reference_reward=group["fixed_reference_reward"].to_numpy(float),
            donor_ability_sg=abilities,
            donor_group_id=group["donor_quality_group_id"].to_numpy(str),
            category=str(group["category"].iloc[0]),
            parameters=parameters,
        )
        if not np.allclose(
            result.donor_group_mass_before,
            result.donor_group_mass_after,
            atol=2e-15,
            rtol=2e-15,
        ):
            raise AssertionError("Frozen weighting did not preserve donor-group mass")
        return result


def _mint_policy_v2_weighting_session(
    atoms: pd.DataFrame,
    authority: _TrustedGraphAuthority,
    readiness: PolicyV2WeightingReadiness,
) -> PolicyV2WeightingSession:
    session = object.__new__(PolicyV2WeightingSession)
    session._atoms = atoms.copy(deep=True)
    session._authority = authority
    session._readiness = readiness
    return session


def admit_policy_v2_weighting(
    atoms: pd.DataFrame,
    *,
    authority_manifest: Mapping[str, object],
) -> PolicyV2WeightingSession:
    """Mint weighting only for the exact source-pinned synthetic approval lock."""

    if not isinstance(atoms, pd.DataFrame):
        raise PolicyV2AnchorContractUnavailable("Candidate atoms must be a DataFrame")
    authority = _trusted_graph_authority(authority_manifest)
    validated = _validate_roots_and_policy_groups(atoms, authority)
    kinds = validated["donor_ability_value_kind"].astype(str)
    available = _coerce_bool(
        validated["donor_ability_available"], "donor_ability_available"
    )
    original = _coerce_bool(
        validated["donor_ability_original_available"],
        "donor_ability_original_available",
    )
    readiness = PolicyV2WeightingReadiness(
        contract=SEAM_CONTRACT,
        approval_lock_sha256=APPROVAL_LOCK_SHA256,
        approval_lock_status=APPROVAL_LOCK_STATUS,
        authority_manifest_sha256=authority.manifest_sha256,
        runtime_binding_sha256=authority.runtime_binding_sha256,
        support_inventory_sha256=authority.support_inventory_sha256,
        root_support_sha256=authority.root_support_sha256,
        rows=int(len(validated)),
        supports=int(validated["support_key"].nunique()),
        roots=len(authority.start_support_keys),
        observed_rows=int(kinds.eq(_OBSERVED_VALUE_KIND).sum()),
        imputed_rows=int(kinds.eq(_IMPUTED_VALUE_KIND).sum()),
        unavailable_rows=int((~available).sum()),
        nullable_skill_date_rows=int(validated["donor_skill_date"].isna().sum()),
        availability_preserved=bool(np.array_equal(available, original)),
        synthetic_fixture_only=True,
        weighting_ready=True,
        real_chain_ready=False,
        integration_ready=False,
        integration_blockers=(
            "no_policy_v2_physical_chain_expectation_callback",
            "no_policy_v2_mean_solve_or_simulation_integration",
        ),
    )
    return _mint_policy_v2_weighting_session(
        validated, authority, readiness
    )


__all__ = [
    "ADAPTER_SOURCE_SHA256",
    "ANCHOR_SOURCE_SHA256",
    "APPROVAL_LOCK_SHA256",
    "APPROVAL_LOCK_STATUS",
    "APPROVED_AUTHORITY_MANIFEST_SHA256",
    "APPROVED_AUTHORITY_SOURCE_DESCRIPTOR",
    "APPROVED_ROOT_SUPPORT_SHA256",
    "APPROVED_SUPPORT_INVENTORY_SHA256",
    "AUTHORITY_CONTRACT",
    "EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT",
    "POLICY_SOURCE_SHA256",
    "PolicyV2AnchorContractUnavailable",
    "PolicyV2WeightingReadiness",
    "PolicyV2WeightingSession",
    "ROOT_SUPPORT_CONTRACT",
    "SEAM_CONTRACT",
    "V3_ADAPTER_SOURCE_DESCRIPTOR",
    "V3_ADAPTER_SOURCE_SHA256",
    "admit_policy_v2_weighting",
    "validate_frozen_runtime_bindings",
]

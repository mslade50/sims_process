"""Synthetic-only compact APP whole-atom weighting and selector seam.

The selector owns the four APP semantic uniforms explicitly: holeout,
proposal, surface, and hazard.  It accepts one closed synthetic complete-atom
fixture, privately applies the frozen whole-atom weighting primitive, and
selects a terminal donor or a spatial proposal without separating any atom's
endpoint, lie, hazard, cost, reward, or donor ability.

This module is not an expectation callback, solver, simulator, real-support
authority, or production integration.  Candidate digests cannot widen its
single source-pinned synthetic authority.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass, fields as _dataclass_fields
from fractions import Fraction as _Fraction
from hashlib import sha256 as _sha256
import json as _json
import math as _math
from pathlib import Path as _Path
from typing import Mapping as _Mapping, Sequence as _Sequence
import unicodedata as _unicodedata

import numpy as _np
import pandas as _pd

from hole_sim_donor_ability_policy import (  # noqa: E402
    CATEGORY_ABILITY_MAPPING as _CATEGORY_ABILITY_MAPPING,
    IMPUTED_NEUTRAL_VALUE as _IMPUTED_NEUTRAL_VALUE,
    IMPUTED_VALUE_KIND as _IMPUTED_VALUE_KIND,
    OBSERVED_VALUE_KIND as _OBSERVED_VALUE_KIND,
    POLICY_CONTRACT as _POLICY_CONTRACT,
    POLICY_STATUS as _POLICY_STATUS,
    policy_payload as _policy_payload,
    validate_applied_frame as _validate_applied_policy_frame,
)
from hole_sim_transition_mean_anchor import (  # noqa: E402
    ATOM_WEIGHT_CONTRACT as _ATOM_WEIGHT_CONTRACT,
    TransitionQualityParameters as _TransitionQualityParameters,
    whole_atom_probabilities as _whole_atom_probabilities,
)


COMPACT_APP_SUPPORT_CONTRACT = "synthetic_compact_complete_app_support_v1"
SYNTHETIC_APP_AUTHORITY_CONTRACT = (
    "closed_synthetic_compact_app_support_authority_v1"
)
APP_WEIGHTING_SELECTOR_CONTRACT = (
    "whole_atom_weighted_app_holeout_proposal_surface_hazard_selector_v2"
)
APP_ZERO_FAST_PATH_CONTRACT = "exact_frozen_app_semantic_cdfs_zero_tilt_v1"
APP_SELECTION_BOUNDARY_CONTRACT = (
    "holeout_strict_terminal_unconditional_prefix_proposal_right_surface_strict_"
    "hazard_right_binary64_pathwise_v2"
)
INTEGRATION_STATUS = "synthetic_app_fixture_only_integration_hold"

_ANCHOR_DESCRIPTOR = "archive/hole_sim_transition_mean_anchor.py"
_POLICY_DESCRIPTOR = "archive/hole_sim_donor_ability_policy.py"
_COMPLETE_DESCRIPTOR = "archive/hole_sim_tee_app_complete_atom_support.py"
_TEE_SELECTOR_DESCRIPTOR = "archive/hole_sim_compact_physical_support_selector_v1.py"
_ANCHOR_SOURCE_SHA256 = (
    "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
)
_POLICY_SOURCE_SHA256 = (
    "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
)
_COMPLETE_SOURCE_SHA256 = (
    "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
)
_TEE_SELECTOR_SOURCE_SHA256 = (
    "a5043c1fe1d32331ab8484d4faf99bb06a30c4c7ade1d0e5ed52b31ecbd3aff6"
)
_POLICY_IDENTITY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)

_MAX_ATOMS = 20
_MAX_TERMINAL_DONORS = 4
_MAX_SPATIAL_PROPOSALS = 4
_MAX_SURFACES_PER_PROPOSAL = 4
_MAX_HAZARDS_PER_SURFACE = 4
_MAX_ABS_LOG_WEIGHT = 30.0
_MASS_TOLERANCE = 3e-13

_SEQUENCE_FIELDS = (
    "atom_id",
    "base_mass",
    "fixed_reference_reward",
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
    "original_donor_group_id",
    "terminal",
    "terminal_donor_order",
    "spatial_proposal_order",
    "surface_order",
    "hazard_order",
    "frozen_terminal_residual_cdf_upper",
    "frozen_spatial_proposal_cdf_upper",
    "frozen_surface_cdf_upper",
    "frozen_hazard_cdf_upper",
    "boundary_only",
    "complete_atom",
    "stroke_cost",
    "next_state_key",
)
_POLICY_FRAME_COLUMNS = (
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


class CompactAppSupportUnavailable(ValueError):
    """Raised when APP weighting or selection is not research-admissible."""


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_sha(value: object) -> str:
    return _sha256(
        _json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: _Path) -> str:
    digest = _sha256()
    with _Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise CompactAppSupportUnavailable(f"{label} must be a plain string")
    result = _unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise CompactAppSupportUnavailable(f"{label} must be populated")
    return result


def _timestamp(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None:
        if optional:
            return None
        raise CompactAppSupportUnavailable(f"{label} must be populated")
    if type(value) not in (str, _pd.Timestamp):
        raise CompactAppSupportUnavailable(
            f"{label} must be a plain timestamp string or pandas Timestamp"
        )
    try:
        stamp = _pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CompactAppSupportUnavailable(f"{label} is not a timestamp") from exc
    if bool(_pd.isna(stamp)):
        if optional:
            return None
        raise CompactAppSupportUnavailable(f"{label} must be populated")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


_PLAIN_NUMBER_TYPES = (int, float, _np.float64)


def _plain_finite_number(value: object, label: str) -> float:
    if type(value) not in _PLAIN_NUMBER_TYPES:
        raise CompactAppSupportUnavailable(f"{label} must be a plain numeric scalar")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CompactAppSupportUnavailable(f"{label} must be finite") from exc
    if not _math.isfinite(number):
        raise CompactAppSupportUnavailable(f"{label} must be finite")
    return number


def _optional_finite(value: object, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in _PLAIN_NUMBER_TYPES:
        raise CompactAppSupportUnavailable(f"{label} must be a plain numeric scalar")
    if bool(_pd.isna(value)):
        return None
    return _plain_finite_number(value, label)


def _boolean(value: object, label: str) -> bool:
    if type(value) not in (bool, _np.bool_):
        raise CompactAppSupportUnavailable(f"{label} must be boolean")
    return bool(value)


def _integer(value: object, label: str) -> int:
    if type(value) not in (int, _np.int64, _np.int32, _np.int16):
        raise CompactAppSupportUnavailable(f"{label} must be a plain integer")
    return int(value)


def _float_hex(value: float) -> str:
    return float(value).hex()


def _optional_float_hex(value: float | None) -> str | None:
    return None if value is None else _float_hex(value)


def _source_bindings() -> tuple[dict[str, str], ...]:
    return (
        {
            "role": "whole_atom_weighting_primitive",
            "descriptor": _ANCHOR_DESCRIPTOR,
            "source_sha256": _ANCHOR_SOURCE_SHA256,
        },
        {
            "role": "frozen_policy_v2_attestation",
            "descriptor": _POLICY_DESCRIPTOR,
            "source_sha256": _POLICY_SOURCE_SHA256,
        },
        {
            "role": "complete_app_semantic_reference",
            "descriptor": _COMPLETE_DESCRIPTOR,
            "source_sha256": _COMPLETE_SOURCE_SHA256,
        },
        {
            "role": "patched_tee_selector_nonregression_reference",
            "descriptor": _TEE_SELECTOR_DESCRIPTOR,
            "source_sha256": _TEE_SELECTOR_SOURCE_SHA256,
        },
    )


def _runtime_binding_payload() -> dict[str, object]:
    return {
        "atom_weight_contract": _ATOM_WEIGHT_CONTRACT,
        "policy_contract": _POLICY_CONTRACT,
        "policy_status": _POLICY_STATUS,
        "policy_identity_sha256": _POLICY_IDENTITY_SHA256,
        "policy_category_mapping": dict(_CATEGORY_ABILITY_MAPPING),
        "category": "app",
        "semantic_uniforms": ["holeout", "proposal", "surface", "hazard"],
        "selection_boundary_contract": APP_SELECTION_BOUNDARY_CONTRACT,
        "tilted_probability_policy": "unchanged_primitive_binary64_atom_vector_v1",
        "tilted_terminal_policy": "ordered_unconditional_atom_prefix_v1",
        "tilted_ratio_projection": (
            "exact_fraction_right_ceil_surface_strict_floor_binary64_v1"
        ),
        "tilted_oracle": "independent_direct_flat_probability_fraction_oracle_v1",
        "source_bindings": list(_source_bindings()),
    }


def _closed_authority_envelope() -> dict[str, object]:
    manifest = {
        "authority_contract": SYNTHETIC_APP_AUTHORITY_CONTRACT,
        "authority_id": "compact_app_whole_atom_selector_fixture_v1",
        "scope": "exact_source_pinned_synthetic_app_fixture_only",
        "research_only": True,
        "production_enabled": False,
        "real_chain_ready": False,
        "integration_ready": False,
        "support_contract": COMPACT_APP_SUPPORT_CONTRACT,
        "weighting_selector_contract": APP_WEIGHTING_SELECTOR_CONTRACT,
        "zero_fast_path_contract": APP_ZERO_FAST_PATH_CONTRACT,
        "selection_boundary_contract": APP_SELECTION_BOUNDARY_CONTRACT,
        "category": "app",
        "supported_categories": ["app"],
        "semantic_uniforms": ["holeout", "proposal", "surface", "hazard"],
        "terminal_contract": (
            "zero_frozen_residual_tilt_unconditional_q_prefix_v2"
        ),
        "spatial_contract": (
            "exact_mass_ratio_proposal_right_surface_strict_hazard_right_"
            "binary64_lattice_v2"
        ),
        "tilted_probability_policy": "unchanged_primitive_binary64_atom_vector_v1",
        "tilted_oracle": "independent_direct_flat_probability_fraction_oracle_v1",
        "terminal_atoms_allowed": True,
        "multiple_terminal_donors_required": True,
        "candidate_derived_digests_accepted": False,
        "support_sha256": "ef1e5ba126213f1079b533e2fbdd989febd07c77f56cb1cded78cc32c45bf092",
        "frozen_cdf_sha256": "d8e10749af594fd0797f12792a60e76100ee6bb7fee155556618285b834aaade",
        "reward_reference_sha256": "a70ebc4aa3ec0dab67eb4d69722e48ec3f1fd6ec390643b53f21a63d6c45219b",
        "policy_identity_sha256": _POLICY_IDENTITY_SHA256,
        "runtime_binding_sha256": "6e3d680f55bbd6a4de25cb8710e709da111da68450369f492a3dd8d7651d80b5",
        "source_bindings": list(_source_bindings()),
        "limits": {
            "max_atoms": _MAX_ATOMS,
            "max_terminal_donors": _MAX_TERMINAL_DONORS,
            "max_spatial_proposals": _MAX_SPATIAL_PROPOSALS,
            "max_surfaces_per_proposal": _MAX_SURFACES_PER_PROPOSAL,
            "max_hazards_per_surface": _MAX_HAZARDS_PER_SURFACE,
            "max_abs_log_weight": _MAX_ABS_LOG_WEIGHT,
        },
    }
    return {
        "manifest": manifest,
        "manifest_sha256": "47d8fb193443161d6199d2007975ead282bd4708817a2d9927c533a2519e53ec",
    }


def synthetic_app_authority_manifest() -> dict[str, object]:
    """Return a defensive reporting copy of the closed synthetic authority."""

    return _json.loads(_json.dumps(_closed_authority_envelope()))


def _validate_runtime() -> str:
    if _ATOM_WEIGHT_CONTRACT != "whole_atom_donor_detilt_target_reapply_lambda_v1":
        raise CompactAppSupportUnavailable("Frozen whole-atom contract changed")
    if _POLICY_CONTRACT != "frozen_strict_prior_donor_ability_policy_v2":
        raise CompactAppSupportUnavailable("Frozen donor-ability policy changed")
    if _policy_payload().get("policy_identity_sha256") != _POLICY_IDENTITY_SHA256:
        raise CompactAppSupportUnavailable("Frozen policy identity changed")
    archive = _Path(__file__).resolve().parent
    filenames = {
        _ANCHOR_DESCRIPTOR: "hole_sim_transition_mean_anchor.py",
        _POLICY_DESCRIPTOR: "hole_sim_donor_ability_policy.py",
        _COMPLETE_DESCRIPTOR: "hole_sim_tee_app_complete_atom_support.py",
        _TEE_SELECTOR_DESCRIPTOR: "hole_sim_compact_physical_support_selector_v1.py",
    }
    observed = {
        descriptor: _sha256_file(archive / filename)
        for descriptor, filename in filenames.items()
    }
    expected = {
        binding["descriptor"]: binding["source_sha256"]
        for binding in _source_bindings()
    }
    if observed != expected:
        changed = sorted(key for key in expected if observed.get(key) != expected[key])
        raise CompactAppSupportUnavailable(f"Frozen source binding changed: {changed}")
    return _stable_sha(_runtime_binding_payload())


@_dataclass(frozen=True)
class CompactAppSupport:
    support_key: str
    reward_reference_sha256: str
    frozen_holeout_probability: object
    atom_id: tuple[object, ...]
    base_mass: tuple[object, ...]
    fixed_reference_reward: tuple[object, ...]
    bridge_row_id: tuple[object, ...]
    category: tuple[object, ...]
    donor_player_id: tuple[object, ...]
    donor_event_key: tuple[object, ...]
    donor_event_start: tuple[object, ...]
    target_event_key: tuple[object, ...]
    target_event_start: tuple[object, ...]
    donor_skill_date: tuple[object, ...]
    donor_ability_source_column: tuple[object, ...]
    donor_ability_available: tuple[object, ...]
    donor_ability_raw_strict_prior: tuple[object, ...]
    donor_ability_value_kind: tuple[object, ...]
    donor_ability_sg: tuple[object, ...]
    donor_ability_was_clipped: tuple[object, ...]
    donor_ability_policy_sha256: tuple[object, ...]
    donor_ability_row_sha256: tuple[object, ...]
    original_donor_group_id: tuple[object, ...]
    terminal: tuple[object, ...]
    terminal_donor_order: tuple[object, ...]
    spatial_proposal_order: tuple[object, ...]
    surface_order: tuple[object, ...]
    hazard_order: tuple[object, ...]
    frozen_terminal_residual_cdf_upper: tuple[object, ...]
    frozen_spatial_proposal_cdf_upper: tuple[object, ...]
    frozen_surface_cdf_upper: tuple[object, ...]
    frozen_hazard_cdf_upper: tuple[object, ...]
    boundary_only: tuple[object, ...]
    complete_atom: tuple[object, ...]
    stroke_cost: tuple[object, ...]
    next_state_key: tuple[object, ...]


def _canonicalize_support(candidate: CompactAppSupport) -> CompactAppSupport:
    if type(candidate) is not CompactAppSupport:
        raise CompactAppSupportUnavailable("APP support has the wrong exact type")
    values: dict[str, object] = {
        "support_key": _plain_text(candidate.support_key, "support_key"),
        "reward_reference_sha256": _plain_text(
            candidate.reward_reference_sha256, "reward_reference_sha256"
        ),
        "frozen_holeout_probability": _plain_finite_number(
            candidate.frozen_holeout_probability, "frozen_holeout_probability"
        ),
    }
    if not _is_sha256(values["reward_reference_sha256"]):
        raise CompactAppSupportUnavailable("reward_reference_sha256 is invalid")
    raw_sequences: dict[str, tuple[object, ...]] = {}
    for name in _SEQUENCE_FIELDS:
        raw = getattr(candidate, name)
        if type(raw) is not tuple:
            raise CompactAppSupportUnavailable(f"{name} must be an exact tuple")
        raw_sequences[name] = raw
    lengths = {name: len(value) for name, value in raw_sequences.items()}
    if len(set(lengths.values())) != 1:
        raise CompactAppSupportUnavailable(f"APP support arrays do not align: {lengths}")
    rows = next(iter(lengths.values()))
    if rows <= 0 or rows > _MAX_ATOMS:
        raise CompactAppSupportUnavailable(
            f"APP support atom cap invalid: rows={rows}, cap={_MAX_ATOMS}"
        )
    text_fields = (
        "atom_id",
        "bridge_row_id",
        "category",
        "donor_player_id",
        "donor_event_key",
        "target_event_key",
        "donor_ability_source_column",
        "donor_ability_value_kind",
        "donor_ability_policy_sha256",
        "donor_ability_row_sha256",
        "original_donor_group_id",
    )
    for name in text_fields:
        values[name] = tuple(_plain_text(item, name) for item in raw_sequences[name])
    values["category"] = tuple(str(item).casefold() for item in values["category"])
    values["next_state_key"] = tuple(
        _plain_text(item, "next_state_key", allow_empty=True)
        for item in raw_sequences["next_state_key"]
    )
    for name in ("donor_event_start", "target_event_start"):
        values[name] = tuple(_timestamp(item, name) for item in raw_sequences[name])
    values["donor_skill_date"] = tuple(
        _timestamp(item, "donor_skill_date", optional=True)
        for item in raw_sequences["donor_skill_date"]
    )
    for name in ("base_mass", "fixed_reference_reward", "donor_ability_sg"):
        values[name] = tuple(
            _plain_finite_number(item, name) for item in raw_sequences[name]
        )
    values["donor_ability_raw_strict_prior"] = tuple(
        _optional_finite(item, "donor_ability_raw_strict_prior")
        for item in raw_sequences["donor_ability_raw_strict_prior"]
    )
    for name in (
        "terminal",
        "donor_ability_available",
        "donor_ability_was_clipped",
        "boundary_only",
        "complete_atom",
    ):
        values[name] = tuple(_boolean(item, name) for item in raw_sequences[name])
    for name in (
        "terminal_donor_order",
        "spatial_proposal_order",
        "surface_order",
        "hazard_order",
        "stroke_cost",
    ):
        values[name] = tuple(_integer(item, name) for item in raw_sequences[name])
    for name in (
        "frozen_terminal_residual_cdf_upper",
        "frozen_spatial_proposal_cdf_upper",
        "frozen_surface_cdf_upper",
        "frozen_hazard_cdf_upper",
    ):
        values[name] = tuple(
            _plain_finite_number(item, name) for item in raw_sequences[name]
        )
    canonical = CompactAppSupport(**values)  # type: ignore[arg-type]
    _validate_support_semantics(canonical)
    return canonical


def _node_cdf(
    keys: _Sequence[tuple[int, ...]],
    uppers: _Sequence[float],
    label: str,
    *,
    allow_leading_zero: bool = False,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    nodes: list[tuple[int, ...]] = []
    values: list[float] = []
    seen: dict[tuple[int, ...], float] = {}
    for key, upper in zip(keys, uppers, strict=True):
        if key in seen:
            if seen[key] != float(upper):
                raise CompactAppSupportUnavailable(f"{label} varies inside a node")
            continue
        seen[key] = float(upper)
        nodes.append(key)
        values.append(float(upper))
    vector = _np.asarray(values, dtype=float)
    lower_invalid = (vector < 0.0).any() if allow_leading_zero else (vector <= 0.0).any()
    if (
        not len(vector)
        or not _np.isfinite(vector).all()
        or lower_invalid
        or (vector > 1.0).any()
        or (_np.diff(vector) <= 0.0).any()
        or float(vector[-1]) != 1.0
    ):
        raise CompactAppSupportUnavailable(f"{label} is not an exact closing ordered CDF")
    return tuple(nodes), tuple(map(float, vector))


def _validate_support_semantics(support: CompactAppSupport) -> None:
    rows = len(support.atom_id)
    if len(set(support.atom_id)) != rows:
        raise CompactAppSupportUnavailable("APP atom IDs are duplicated")
    if set(support.category) != {"app"}:
        raise CompactAppSupportUnavailable("Compact APP support must contain only APP atoms")
    if set(support.donor_ability_source_column) != {
        _CATEGORY_ABILITY_MAPPING["app"]
    }:
        raise CompactAppSupportUnavailable("APP ability source column changed")
    if not all(support.complete_atom):
        raise CompactAppSupportUnavailable("Every APP row must be a complete atom")
    if any(value <= 0 for value in support.stroke_cost):
        raise CompactAppSupportUnavailable("Every APP atom needs positive stroke cost")

    base = _np.asarray(support.base_mass, dtype=float)
    boundary = _np.asarray(support.boundary_only, dtype=bool)
    if (
        (base < 0.0).any()
        or not _np.array_equal(base == 0.0, boundary)
        or not _math.isclose(
            float(base.sum()), 1.0, rel_tol=0.0, abs_tol=_MASS_TOLERANCE
        )
    ):
        raise CompactAppSupportUnavailable(
            "APP base mass must be positive except exact surface sentinels and close"
        )
    holeout = float(support.frozen_holeout_probability)
    if not 0.0 < holeout < 1.0:
        raise CompactAppSupportUnavailable(
            "Frozen APP holeout probability must be strictly between zero and one"
        )
    terminal_indices = [index for index, value in enumerate(support.terminal) if value]
    spatial_indices = [index for index, value in enumerate(support.terminal) if not value]
    if not terminal_indices or not spatial_indices:
        raise CompactAppSupportUnavailable("APP support needs terminal and spatial atoms")
    if len(terminal_indices) < 2:
        raise CompactAppSupportUnavailable(
            "Synthetic APP support requires multiple terminal donors"
        )
    if terminal_indices != list(range(len(terminal_indices))):
        raise CompactAppSupportUnavailable("Terminal donors must precede spatial atoms")
    if len(terminal_indices) > _MAX_TERMINAL_DONORS:
        raise CompactAppSupportUnavailable("Terminal donor cap exceeded")
    if tuple(support.terminal_donor_order[i] for i in terminal_indices) != tuple(
        range(len(terminal_indices))
    ):
        raise CompactAppSupportUnavailable("Terminal donor order must be contiguous")
    for index in terminal_indices:
        if (
            support.spatial_proposal_order[index] != -1
            or support.surface_order[index] != -1
            or support.hazard_order[index] != -1
            or support.stroke_cost[index] != 1
            or support.boundary_only[index]
            or support.next_state_key[index] != ""
            or support.frozen_spatial_proposal_cdf_upper[index] != 0.0
            or support.frozen_surface_cdf_upper[index] != 0.0
            or support.frozen_hazard_cdf_upper[index] != 0.0
        ):
            raise CompactAppSupportUnavailable(
                "Terminal APP donors require exact one-stroke holeout sentinels"
            )
    for index in spatial_indices:
        if (
            support.terminal_donor_order[index] != -1
            or support.frozen_terminal_residual_cdf_upper[index] != 0.0
            or support.spatial_proposal_order[index] < 0
            or support.surface_order[index] < 0
            or support.hazard_order[index] < 0
            or support.next_state_key[index] == ""
        ):
            raise CompactAppSupportUnavailable("Spatial APP hierarchy sentinels changed")
    for terminal, next_state in zip(
        support.terminal, support.next_state_key, strict=True
    ):
        if terminal != (next_state == ""):
            raise CompactAppSupportUnavailable(
                "Terminal atoms need empty next state; spatial atoms need live state"
            )

    _, terminal_upper = _node_cdf(
        tuple((support.terminal_donor_order[i],) for i in terminal_indices),
        tuple(support.frozen_terminal_residual_cdf_upper[i] for i in terminal_indices),
        "Frozen terminal donor residual right-CDF",
    )
    spatial_order = tuple(
        (
            support.spatial_proposal_order[i],
            support.surface_order[i],
            support.hazard_order[i],
        )
        for i in spatial_indices
    )
    if spatial_order != tuple(sorted(spatial_order)) or len(set(spatial_order)) != len(
        spatial_order
    ):
        raise CompactAppSupportUnavailable(
            "Spatial APP atoms must be unique and hierarchy ordered"
        )
    proposals = tuple(
        dict.fromkeys(support.spatial_proposal_order[i] for i in spatial_indices)
    )
    if proposals != tuple(range(len(proposals))):
        raise CompactAppSupportUnavailable("Spatial proposal order must be contiguous")
    if len(proposals) > _MAX_SPATIAL_PROPOSALS:
        raise CompactAppSupportUnavailable("Spatial proposal cap exceeded")
    proposal_keys = tuple(
        (support.spatial_proposal_order[i],) for i in spatial_indices
    )
    _node_cdf(
        proposal_keys,
        tuple(support.frozen_spatial_proposal_cdf_upper[i] for i in spatial_indices),
        "Frozen spatial proposal right-CDF",
    )
    for proposal in proposals:
        p_indices = [
            i for i in spatial_indices if support.spatial_proposal_order[i] == proposal
        ]
        surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
        if surfaces != tuple(range(len(surfaces))):
            raise CompactAppSupportUnavailable("Surface order must be contiguous")
        if len(surfaces) > _MAX_SURFACES_PER_PROPOSAL:
            raise CompactAppSupportUnavailable("Surface cap exceeded")
        surface_keys = tuple(
            (proposal, support.surface_order[i]) for i in p_indices
        )
        _node_cdf(
            surface_keys,
            tuple(support.frozen_surface_cdf_upper[i] for i in p_indices),
            f"Frozen surface strict CDF for proposal {proposal}",
            allow_leading_zero=True,
        )
        for surface in surfaces:
            s_indices = [i for i in p_indices if support.surface_order[i] == surface]
            hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
            if hazards != tuple(range(len(hazards))):
                raise CompactAppSupportUnavailable("Hazard order must be contiguous")
            if len(hazards) > _MAX_HAZARDS_PER_SURFACE:
                raise CompactAppSupportUnavailable("Hazard cap exceeded")
            _node_cdf(
                tuple((proposal, surface, support.hazard_order[i]) for i in s_indices),
                tuple(support.frozen_hazard_cdf_upper[i] for i in s_indices),
                f"Frozen hazard right-CDF for {proposal}/{surface}",
            )
            branch_boundary = {support.boundary_only[i] for i in s_indices}
            if len(branch_boundary) != 1:
                raise CompactAppSupportUnavailable(
                    "Boundary-only status varies within a surface"
                )
            if True in branch_boundary and (
                surface != 0
                or len(s_indices) != 1
                or support.hazard_order[s_indices[0]] != 0
                or support.frozen_surface_cdf_upper[s_indices[0]] != 0.0
                or support.frozen_hazard_cdf_upper[s_indices[0]] != 1.0
            ):
                raise CompactAppSupportUnavailable(
                    "Zero-mass surface sentinel must be the one leading surface atom"
                )

    policy_frame = _pd.DataFrame(
        {column: getattr(support, column) for column in _POLICY_FRAME_COLUMNS}
    )
    try:
        _validate_applied_policy_frame(policy_frame)
    except Exception as exc:
        raise CompactAppSupportUnavailable(
            f"Policy-v2 row attestation failed: {exc}"
        ) from exc
    kinds = _np.asarray(support.donor_ability_value_kind, dtype=str)
    available = _np.asarray(support.donor_ability_available, dtype=bool)
    ability = _np.asarray(support.donor_ability_sg, dtype=float)
    imputed = kinds == _IMPUTED_VALUE_KIND
    observed = kinds == _OBSERVED_VALUE_KIND
    if not _np.logical_or(imputed, observed).all():
        raise CompactAppSupportUnavailable("Unexpected policy-v2 ability state")
    if available[imputed].any() or not available[observed].all():
        raise CompactAppSupportUnavailable("Policy-v2 availability changed")
    if (
        not _np.equal(ability[imputed], _IMPUTED_NEUTRAL_VALUE).all()
        or _np.signbit(ability[imputed]).any()
    ):
        raise CompactAppSupportUnavailable(
            "Neutral-imputed APP ability must be exact positive +0.0"
        )

    terminal_expected: list[float] = []
    prior = 0.0
    for upper in terminal_upper:
        terminal_expected.append(holeout * (upper - prior))
        prior = upper
    expected = _np.zeros(rows, dtype=float)
    expected[terminal_indices] = terminal_expected
    prior_proposal = 0.0
    for proposal in proposals:
        p_indices = [
            i for i in spatial_indices if support.spatial_proposal_order[i] == proposal
        ]
        p_upper = support.frozen_spatial_proposal_cdf_upper[p_indices[0]]
        p_width = p_upper - prior_proposal
        prior_proposal = p_upper
        prior_surface = 0.0
        surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
        for surface in surfaces:
            s_indices = [i for i in p_indices if support.surface_order[i] == surface]
            s_upper = support.frozen_surface_cdf_upper[s_indices[0]]
            s_width = s_upper - prior_surface
            prior_surface = s_upper
            prior_hazard = 0.0
            for index in s_indices:
                h_upper = support.frozen_hazard_cdf_upper[index]
                h_width = h_upper - prior_hazard
                prior_hazard = h_upper
                expected[index] = (1.0 - holeout) * p_width * s_width * h_width
    if not _np.allclose(base, expected, atol=2e-15, rtol=2e-15):
        raise CompactAppSupportUnavailable(
            "Frozen APP holeout/spatial hierarchy does not factor base mass"
        )


def _support_payload(support: CompactAppSupport) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index in range(len(support.atom_id)):
        records.append(
            {
                "atom_id": support.atom_id[index],
                "base_mass": _float_hex(support.base_mass[index]),
                "fixed_reference_reward": _float_hex(
                    support.fixed_reference_reward[index]
                ),
                "bridge_row_id": support.bridge_row_id[index],
                "category": support.category[index],
                "donor_player_id": support.donor_player_id[index],
                "donor_event_key": support.donor_event_key[index],
                "donor_event_start": support.donor_event_start[index],
                "target_event_key": support.target_event_key[index],
                "target_event_start": support.target_event_start[index],
                "donor_skill_date": support.donor_skill_date[index],
                "donor_ability_source_column": support.donor_ability_source_column[index],
                "donor_ability_available": support.donor_ability_available[index],
                "donor_ability_raw_strict_prior": _optional_float_hex(
                    support.donor_ability_raw_strict_prior[index]
                ),
                "donor_ability_value_kind": support.donor_ability_value_kind[index],
                "donor_ability_sg": _float_hex(support.donor_ability_sg[index]),
                "donor_ability_was_clipped": support.donor_ability_was_clipped[index],
                "donor_ability_policy_sha256": support.donor_ability_policy_sha256[index],
                "donor_ability_row_sha256": support.donor_ability_row_sha256[index],
                "original_donor_group_id": support.original_donor_group_id[index],
                "terminal": support.terminal[index],
                "terminal_donor_order": support.terminal_donor_order[index],
                "spatial_proposal_order": support.spatial_proposal_order[index],
                "surface_order": support.surface_order[index],
                "hazard_order": support.hazard_order[index],
                "boundary_only": support.boundary_only[index],
                "complete_atom": support.complete_atom[index],
                "stroke_cost": support.stroke_cost[index],
                "next_state_key": support.next_state_key[index],
            }
        )
    return {
        "contract": COMPACT_APP_SUPPORT_CONTRACT,
        "support_key": support.support_key,
        "reward_reference_sha256": support.reward_reference_sha256,
        "frozen_holeout_probability": _float_hex(
            support.frozen_holeout_probability
        ),
        "records": records,
    }


def _support_sha256(support: CompactAppSupport) -> str:
    return _stable_sha(_support_payload(support))


def _frozen_cdf_sha256(support: CompactAppSupport) -> str:
    return _stable_sha(
        {
            "contract": APP_ZERO_FAST_PATH_CONTRACT,
            "support_key": support.support_key,
            "support_sha256": _support_sha256(support),
            "atom_id": list(support.atom_id),
            "holeout_probability": _float_hex(support.frozen_holeout_probability),
            "terminal_residual_cdf_upper": [
                _float_hex(value)
                for value in support.frozen_terminal_residual_cdf_upper
            ],
            "spatial_proposal_cdf_upper": [
                _float_hex(value)
                for value in support.frozen_spatial_proposal_cdf_upper
            ],
            "surface_cdf_upper": [
                _float_hex(value) for value in support.frozen_surface_cdf_upper
            ],
            "hazard_cdf_upper": [
                _float_hex(value) for value in support.frozen_hazard_cdf_upper
            ],
            "terminal": list(support.terminal),
            "terminal_donor_order": list(support.terminal_donor_order),
            "spatial_proposal_order": list(support.spatial_proposal_order),
            "surface_order": list(support.surface_order),
            "hazard_order": list(support.hazard_order),
            "boundary_only": list(support.boundary_only),
        }
    )


@_dataclass(frozen=True)
class AppWeightingParameters:
    transfer_beta: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    lambda_correction: tuple[float, float, float, float]
    reward_reference_sha256: str
    max_abs_log_weight: float = _MAX_ABS_LOG_WEIGHT


@_dataclass(frozen=True)
class AppSemanticUniforms:
    holeout: float
    proposal: float
    surface: float
    hazard: float


def _plain_four_vector(value: object, label: str) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise CompactAppSupportUnavailable(f"{label} must be an immutable four-vector")
    return tuple(
        _plain_finite_number(item, label) for item in value
    )  # type: ignore[return-value]


def _validated_weighting_parameters(
    parameters: object,
) -> tuple[_TransitionQualityParameters, bool]:
    if type(parameters) is not AppWeightingParameters:
        raise CompactAppSupportUnavailable(
            "APP weighting parameters have the wrong exact type"
        )
    beta = _plain_four_vector(
        object.__getattribute__(parameters, "transfer_beta"), "transfer_beta"
    )
    target = _plain_four_vector(
        object.__getattribute__(parameters, "target_ability_sg"),
        "target_ability_sg",
    )
    correction = _plain_four_vector(
        object.__getattribute__(parameters, "lambda_correction"),
        "lambda_correction",
    )
    if any(value < 0.0 for value in beta):
        raise CompactAppSupportUnavailable("transfer_beta cannot be negative")
    reward = _plain_text(
        object.__getattribute__(parameters, "reward_reference_sha256"),
        "reward_reference_sha256",
    )
    if not _is_sha256(reward):
        raise CompactAppSupportUnavailable("Weighting reward reference is invalid")
    cap = _plain_finite_number(
        object.__getattribute__(parameters, "max_abs_log_weight"),
        "max_abs_log_weight",
    )
    if cap != _MAX_ABS_LOG_WEIGHT:
        raise CompactAppSupportUnavailable("Weighting log cap differs from closed cap")
    try:
        anchor_parameters = _TransitionQualityParameters(
            transfer_beta=beta,
            target_ability_sg=target,
            lambda_correction=correction,
            reward_reference_sha256=reward,
            max_abs_log_weight=cap,
        )
    except (TypeError, ValueError) as exc:
        raise CompactAppSupportUnavailable(
            f"APP weighting parameters are invalid: {exc}"
        ) from exc
    beta_lambda_zero = all(value == 0.0 for value in (*beta, *correction))
    return anchor_parameters, beta_lambda_zero


def _validated_semantic_uniforms(uniforms: object) -> tuple[float, float, float, float]:
    if type(uniforms) is not AppSemanticUniforms:
        raise CompactAppSupportUnavailable(
            "APP semantic uniforms have the wrong exact type"
        )
    retained = []
    for label in ("holeout", "proposal", "surface", "hazard"):
        number = _plain_finite_number(
            object.__getattribute__(uniforms, label), f"{label}_uniform"
        )
        if not 0.0 <= number < 1.0:
            raise CompactAppSupportUnavailable(
                f"{label}_uniform must satisfy 0 <= u < 1"
            )
        retained.append(number)
    return tuple(retained)  # type: ignore[return-value]


@_dataclass(frozen=True)
class WeightedAppSupport:
    support_sha256: str
    parameter_sha256: str
    atom_id: tuple[str, ...]
    probability: tuple[float, ...]
    neutral_probability_mass: tuple[float, ...]
    terminal_probability: float
    spatial_probability: float
    spatial_probability_exact_ratio: str | None
    terminal_conditional_probability: tuple[float, ...]
    terminal_unconditional_cdf_lower: tuple[float, ...]
    terminal_unconditional_cdf_upper: tuple[float, ...]
    spatial_proposal_conditional_probability: tuple[float, ...]
    surface_conditional_probability: tuple[float, ...]
    hazard_conditional_probability: tuple[float, ...]
    spatial_proposal_conditional_exact_ratio: tuple[str | None, ...]
    surface_conditional_exact_ratio: tuple[str | None, ...]
    hazard_conditional_exact_ratio: tuple[str | None, ...]
    terminal_cdf_lower: tuple[float, ...]
    terminal_cdf_upper: tuple[float, ...]
    spatial_proposal_cdf_lower: tuple[float, ...]
    spatial_proposal_cdf_upper: tuple[float, ...]
    surface_cdf_lower: tuple[float, ...]
    surface_cdf_upper: tuple[float, ...]
    hazard_cdf_lower: tuple[float, ...]
    hazard_cdf_upper: tuple[float, ...]
    donor_group_mass_before: tuple[float, ...]
    donor_group_mass_after: tuple[float, ...]
    clipped_donor_logs: int
    clipped_target_logs: int
    selection_mode: str
    factorization_from_final_probability: bool
    exact_spatial_mass_reconstruction: bool
    binary64_pathwise_exact: bool


@_dataclass(frozen=True)
class AppSelectionResult:
    atom_index: int
    atom_id: str
    terminal: bool
    terminal_donor_order: int
    spatial_proposal_order: int
    surface_order: int
    hazard_order: int
    stroke_cost: int
    next_state_key: str
    boundary_only: bool
    terminal_probability: float
    terminal_residual_uniform: float | None
    selection_mode: str
    hierarchical_atom_id: str
    flat_atom_id: str
    flat_hierarchical_equal: bool
    semantic_uniforms: AppSemanticUniforms


@_dataclass(frozen=True)
class AppSelectorReadiness:
    contract: str
    authority_manifest_sha256: str
    runtime_binding_sha256: str
    support_sha256: str
    frozen_cdf_sha256: str
    rows: int
    terminal_donors: int
    spatial_proposals: int
    original_donor_groups: int
    mixed_policy_state_groups: int
    observed_rows: int
    neutral_imputed_rows: int
    availability_preserved: bool
    positive_zero_preserved: bool
    synthetic_fixture_only: bool
    weighting_ready: bool
    selector_ready: bool
    real_chain_ready: bool
    integration_ready: bool
    blockers: tuple[str, ...]


@_dataclass(frozen=True)
class _TrustedSyntheticAppAuthority:
    manifest_sha256: str
    runtime_binding_sha256: str
    support_sha256: str
    frozen_cdf_sha256: str
    reward_reference_sha256: str


_AUTHORITY_FIELDS = (
    "manifest_sha256",
    "runtime_binding_sha256",
    "support_sha256",
    "frozen_cdf_sha256",
    "reward_reference_sha256",
)


def _exact_authority_fields(authority: object) -> tuple[str, ...] | None:
    if type(authority) is not _TrustedSyntheticAppAuthority:
        return None
    values = []
    try:
        for field in _AUTHORITY_FIELDS:
            value = object.__getattribute__(authority, field)
            if not _is_sha256(value):
                return None
            values.append(value)
    except AttributeError:
        return None
    return tuple(values)


def _closed_authority_value(runtime_sha: str) -> _TrustedSyntheticAppAuthority | None:
    closed = _closed_authority_envelope()
    manifest = closed.get("manifest")
    manifest_sha = closed.get("manifest_sha256")
    if (
        type(manifest) is not dict
        or not _is_sha256(manifest_sha)
        or _stable_sha(manifest) != manifest_sha
        or type(runtime_sha) is not str
        or manifest.get("runtime_binding_sha256") != runtime_sha
    ):
        return None
    values = tuple(manifest.get(field) for field in (
        "support_sha256", "frozen_cdf_sha256", "reward_reference_sha256"
    ))
    if not all(_is_sha256(value) for value in values):
        return None
    return _TrustedSyntheticAppAuthority(
        manifest_sha256=manifest_sha,
        runtime_binding_sha256=runtime_sha,
        support_sha256=values[0],  # type: ignore[arg-type]
        frozen_cdf_sha256=values[1],  # type: ignore[arg-type]
        reward_reference_sha256=values[2],  # type: ignore[arg-type]
    )


def _is_closed_authority(authority: object, runtime_sha: str) -> bool:
    observed = _exact_authority_fields(authority)
    expected = _exact_authority_fields(_closed_authority_value(runtime_sha))
    if observed is None or expected is None:
        return False
    return all(
        str.__eq__(left, right) is True
        for left, right in zip(observed, expected, strict=True)
    )


def _trusted_authority(
    envelope: _Mapping[str, object], runtime_sha: str
) -> _TrustedSyntheticAppAuthority:
    if type(envelope) is not dict:
        raise CompactAppSupportUnavailable("Synthetic APP authority must be an exact dict")

    active_containers: set[int] = set()
    visited_nodes = 0

    def exact_json_tree(value: object, label: str, depth: int = 0) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if depth > 16 or visited_nodes > 1024:
            raise CompactAppSupportUnavailable(
                "Synthetic APP authority exceeds the closed complexity cap"
            )
        if type(value) is dict:
            identity = id(value)
            if identity in active_containers:
                raise CompactAppSupportUnavailable(
                    "Synthetic APP authority contains a container cycle"
                )
            active_containers.add(identity)
            try:
                for key, child in value.items():
                    if type(key) is not str:
                        raise CompactAppSupportUnavailable(
                            f"{label} keys must be plain strings"
                        )
                    exact_json_tree(child, f"{label}.{key}", depth + 1)
            finally:
                active_containers.discard(identity)
            return
        if type(value) is list:
            identity = id(value)
            if identity in active_containers:
                raise CompactAppSupportUnavailable(
                    "Synthetic APP authority contains a container cycle"
                )
            active_containers.add(identity)
            try:
                for index, child in enumerate(value):
                    exact_json_tree(child, f"{label}[{index}]", depth + 1)
            finally:
                active_containers.discard(identity)
            return
        if type(value) in (str, bool, int, float) or value is None:
            if type(value) is float and not _math.isfinite(value):
                raise CompactAppSupportUnavailable(f"{label} must be finite")
            return
        raise CompactAppSupportUnavailable(
            f"{label} contains a non-plain authority value"
        )

    # Reject hostile containers and scalar subclasses before comparison or JSON.
    exact_json_tree(envelope, "authority")
    manifest = envelope.get("manifest")
    manifest_sha = envelope.get("manifest_sha256")
    if type(manifest) is not dict or not _is_sha256(manifest_sha):
        raise CompactAppSupportUnavailable("Synthetic APP authority schema changed")
    closed = _closed_authority_envelope()
    if envelope != closed:
        raise CompactAppSupportUnavailable(
            "Candidate is outside the closed source-pinned APP authority"
        )
    if _stable_sha(manifest) != manifest_sha:
        raise CompactAppSupportUnavailable("Closed synthetic APP authority is unminted")
    if manifest.get("runtime_binding_sha256") != runtime_sha:
        raise CompactAppSupportUnavailable("Synthetic APP runtime binding changed")
    trusted = _closed_authority_value(runtime_sha)
    if trusted is None:
        raise CompactAppSupportUnavailable("Closed APP authority is inconsistent")
    return trusted


def _fraction_mass(probability: _np.ndarray, indices: _Sequence[int]) -> _Fraction:
    return sum(
        (_Fraction.from_float(float(probability[index])) for index in indices),
        _Fraction(0, 1),
    )


def _fraction_text(value: _Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _binary64_ratio_cut(
    numerator: _Fraction,
    denominator: _Fraction,
    *,
    boundary: str,
) -> float:
    """Project one exact mass ratio onto the binary64 uniform lattice.

    Right-CDF nodes use the least binary64 value greater than or equal to the
    exact ratio. Strict-surface nodes use the greatest value less than or equal
    to it. These projections preserve the exact rational comparison for every
    admissible binary64 semantic uniform.
    """

    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise CompactAppSupportUnavailable("APP conditional mass ratio is invalid")
    if numerator == 0:
        return 0.0
    if numerator == denominator:
        return 1.0
    ratio = numerator / denominator
    candidate = float(ratio)
    retained = _Fraction.from_float(candidate)
    if boundary == "right":
        if retained < ratio:
            candidate = float(_np.nextafter(candidate, 1.0))
    elif boundary == "strict":
        if retained > ratio:
            candidate = float(_np.nextafter(candidate, 0.0))
    else:
        raise CompactAppSupportUnavailable("Unknown APP binary64 boundary rule")
    if not 0.0 <= candidate <= 1.0:
        raise CompactAppSupportUnavailable("APP binary64 CDF cut is out of range")
    return candidate


def _ordered_terminal_prefix(
    support: CompactAppSupport, probability: _np.ndarray
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    lower = [0.0] * len(support.atom_id)
    upper = [0.0] * len(support.atom_id)
    running = 0.0
    for index, terminal in enumerate(support.terminal):
        if not terminal:
            break
        mass = float(probability[index])
        retained = float(running + mass)
        if mass <= 0.0 or not retained > running:
            raise CompactAppSupportUnavailable(
                "Positive terminal APP atom cannot form a binary64 prefix interval"
            )
        lower[index] = running
        upper[index] = retained
        running = retained
    if not 0.0 < running < 1.0:
        raise CompactAppSupportUnavailable(
            "Canonical terminal APP prefix must lie strictly inside (0, 1)"
        )
    return running, tuple(lower), tuple(upper)


def _hierarchy_from_probability(
    support: CompactAppSupport, probability: _np.ndarray
) -> dict[str, object]:
    """Build pathwise binary64 cuts from the unchanged weighted atom vector."""

    rows = len(support.atom_id)
    probability = _np.asarray(probability, dtype=float)
    terminal = _np.asarray(support.terminal, dtype=bool)
    boundary = _np.asarray(support.boundary_only, dtype=bool)
    if (
        probability.shape != (rows,)
        or not _np.isfinite(probability).all()
        or (probability < 0.0).any()
        or not _math.isclose(
            float(probability.sum()), 1.0, rel_tol=0.0, abs_tol=_MASS_TOLERANCE
        )
        or not _np.array_equal(probability == 0.0, boundary)
    ):
        raise CompactAppSupportUnavailable(
            "Weighting changed positive versus boundary-only APP atom support"
        )

    terminal_indices = [i for i, value in enumerate(support.terminal) if value]
    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    if terminal_indices != list(range(len(terminal_indices))):
        raise CompactAppSupportUnavailable(
            "Terminal APP atoms must remain one declared contiguous prefix"
        )
    terminal_probability, terminal_unconditional_lower, terminal_unconditional_upper = (
        _ordered_terminal_prefix(support, probability)
    )
    terminal_exact_mass = _fraction_mass(probability, terminal_indices)
    spatial_exact_mass = _fraction_mass(probability, spatial_indices)
    if terminal_exact_mass <= 0 or spatial_exact_mass <= 0:
        raise CompactAppSupportUnavailable(
            "Weighting collapsed the terminal or spatial APP branch"
        )
    spatial_probability = float(spatial_exact_mass)
    if not 0.0 < spatial_probability < 1.0:
        raise CompactAppSupportUnavailable(
            "Canonical spatial APP mass is not representable inside (0, 1)"
        )

    terminal_conditional = _np.zeros(rows, dtype=float)
    proposal_conditional = _np.zeros(rows, dtype=float)
    surface_conditional = _np.zeros(rows, dtype=float)
    hazard_conditional = _np.zeros(rows, dtype=float)
    proposal_conditional_exact: list[str | None] = [None] * rows
    surface_conditional_exact: list[str | None] = [None] * rows
    hazard_conditional_exact: list[str | None] = [None] * rows
    terminal_lower = _np.zeros(rows, dtype=float)
    terminal_upper = _np.zeros(rows, dtype=float)
    proposal_lower = _np.zeros(rows, dtype=float)
    proposal_upper = _np.zeros(rows, dtype=float)
    surface_lower = _np.zeros(rows, dtype=float)
    surface_upper = _np.zeros(rows, dtype=float)
    hazard_lower = _np.zeros(rows, dtype=float)
    hazard_upper = _np.zeros(rows, dtype=float)

    running_terminal_exact = _Fraction(0, 1)
    running_terminal_cut = 0.0
    for index in terminal_indices:
        running_terminal_exact += _Fraction.from_float(float(probability[index]))
        cut = _binary64_ratio_cut(
            running_terminal_exact, terminal_exact_mass, boundary="right"
        )
        if not cut > running_terminal_cut:
            raise CompactAppSupportUnavailable(
                "Positive terminal donor mass has no distinct binary64 residual cut"
            )
        terminal_lower[index] = running_terminal_cut
        terminal_upper[index] = cut
        terminal_conditional[index] = cut - running_terminal_cut
        running_terminal_cut = cut

    proposals = tuple(
        dict.fromkeys(support.spatial_proposal_order[i] for i in spatial_indices)
    )
    proposal_masses: dict[int, _Fraction] = {}
    running_proposal_exact = _Fraction(0, 1)
    running_proposal_cut = 0.0
    for proposal in proposals:
        p_indices = [
            i
            for i in spatial_indices
            if support.spatial_proposal_order[i] == proposal
        ]
        p_mass = _fraction_mass(probability, p_indices)
        if p_mass <= 0:
            raise CompactAppSupportUnavailable("Weighting collapsed a spatial proposal")
        proposal_masses[proposal] = p_mass
        running_proposal_exact += p_mass
        p_cut = _binary64_ratio_cut(
            running_proposal_exact, spatial_exact_mass, boundary="right"
        )
        if not p_cut > running_proposal_cut:
            raise CompactAppSupportUnavailable(
                "Positive spatial proposal mass has no distinct binary64 right-CDF cut"
            )
        for index in p_indices:
            proposal_lower[index] = running_proposal_cut
            proposal_upper[index] = p_cut
            proposal_conditional[index] = p_cut - running_proposal_cut
            proposal_conditional_exact[index] = _fraction_text(
                p_mass / spatial_exact_mass
            )
        running_proposal_cut = p_cut

        surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
        positive_surface_masses: dict[int, _Fraction] = {}
        for surface in surfaces:
            s_indices = [i for i in p_indices if support.surface_order[i] == surface]
            s_mass = _fraction_mass(probability, s_indices)
            if any(support.boundary_only[i] for i in s_indices):
                if (
                    s_mass != 0
                    or surface != 0
                    or len(s_indices) != 1
                    or not all(support.boundary_only[i] for i in s_indices)
                ):
                    raise CompactAppSupportUnavailable(
                        "Weighted APP boundary-only surface semantics changed"
                    )
                surface_conditional[s_indices] = 0.0
                surface_lower[s_indices] = 0.0
                surface_upper[s_indices] = 0.0
                hazard_conditional[s_indices] = 1.0
                hazard_lower[s_indices] = 0.0
                hazard_upper[s_indices] = 1.0
                continue
            if s_mass <= 0:
                raise CompactAppSupportUnavailable(
                    "Weighting collapsed a positive APP surface"
                )
            positive_surface_masses[surface] = s_mass
        if sum(positive_surface_masses.values(), _Fraction(0, 1)) != p_mass:
            raise CompactAppSupportUnavailable(
                "Canonical APP surface masses do not reconstruct their proposal"
            )

        running_surface_exact = _Fraction(0, 1)
        running_surface_cut = 0.0
        for surface in surfaces:
            if surface not in positive_surface_masses:
                continue
            s_indices = [i for i in p_indices if support.surface_order[i] == surface]
            s_mass = positive_surface_masses[surface]
            running_surface_exact += s_mass
            s_cut = _binary64_ratio_cut(
                running_surface_exact, p_mass, boundary="strict"
            )
            if not s_cut > running_surface_cut:
                raise CompactAppSupportUnavailable(
                    "Positive APP surface mass has no distinct binary64 strict-CDF cut"
                )
            surface_witness = (
                0.0
                if running_surface_cut == 0.0
                else float(_np.nextafter(running_surface_cut, 1.0))
            )
            if not surface_witness < 1.0 or surface_witness > s_cut:
                raise CompactAppSupportUnavailable(
                    "Positive APP surface has no admissible binary64 strict-CDF witness"
                )
            for index in s_indices:
                surface_lower[index] = running_surface_cut
                surface_upper[index] = s_cut
                surface_conditional[index] = s_cut - running_surface_cut
                surface_conditional_exact[index] = _fraction_text(s_mass / p_mass)
            running_surface_cut = s_cut

            hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
            running_hazard_exact = _Fraction(0, 1)
            running_hazard_cut = 0.0
            for hazard in hazards:
                h_indices = [
                    i for i in s_indices if support.hazard_order[i] == hazard
                ]
                if len(h_indices) != 1:
                    raise CompactAppSupportUnavailable(
                        "One APP proposal/surface/hazard leaf must name one whole atom"
                    )
                index = h_indices[0]
                h_mass = _Fraction.from_float(float(probability[index]))
                if h_mass <= 0:
                    raise CompactAppSupportUnavailable(
                        "Weighting collapsed a positive APP hazard atom"
                    )
                running_hazard_exact += h_mass
                h_cut = _binary64_ratio_cut(
                    running_hazard_exact, s_mass, boundary="right"
                )
                if not h_cut > running_hazard_cut:
                    raise CompactAppSupportUnavailable(
                        "Positive APP hazard mass has no distinct binary64 right-CDF cut"
                    )
                hazard_lower[index] = running_hazard_cut
                hazard_upper[index] = h_cut
                hazard_conditional[index] = h_cut - running_hazard_cut
                hazard_conditional_exact[index] = _fraction_text(h_mass / s_mass)
                running_hazard_cut = h_cut

                reconstructed = (
                    spatial_exact_mass
                    * (p_mass / spatial_exact_mass)
                    * (s_mass / p_mass)
                    * (h_mass / s_mass)
                )
                if reconstructed != h_mass:
                    raise CompactAppSupportUnavailable(
                        "Exact APP conditional masses do not reconstruct a flat atom"
                    )

    if running_proposal_exact != spatial_exact_mass or running_proposal_cut != 1.0:
        raise CompactAppSupportUnavailable(
            "Canonical spatial APP proposal hierarchy does not close"
        )
    return {
        "terminal_probability": terminal_probability,
        "spatial_probability": spatial_probability,
        "spatial_probability_exact_ratio": _fraction_text(spatial_exact_mass),
        "terminal_conditional_probability": tuple(map(float, terminal_conditional)),
        "terminal_unconditional_cdf_lower": terminal_unconditional_lower,
        "terminal_unconditional_cdf_upper": terminal_unconditional_upper,
        "spatial_proposal_conditional_probability": tuple(
            map(float, proposal_conditional)
        ),
        "surface_conditional_probability": tuple(map(float, surface_conditional)),
        "hazard_conditional_probability": tuple(map(float, hazard_conditional)),
        "spatial_proposal_conditional_exact_ratio": tuple(
            proposal_conditional_exact
        ),
        "surface_conditional_exact_ratio": tuple(surface_conditional_exact),
        "hazard_conditional_exact_ratio": tuple(hazard_conditional_exact),
        "terminal_cdf_lower": tuple(map(float, terminal_lower)),
        "terminal_cdf_upper": tuple(map(float, terminal_upper)),
        "spatial_proposal_cdf_lower": tuple(map(float, proposal_lower)),
        "spatial_proposal_cdf_upper": tuple(map(float, proposal_upper)),
        "surface_cdf_lower": tuple(map(float, surface_lower)),
        "surface_cdf_upper": tuple(map(float, surface_upper)),
        "hazard_cdf_lower": tuple(map(float, hazard_lower)),
        "hazard_cdf_upper": tuple(map(float, hazard_upper)),
    }


def _frozen_hierarchy(support: CompactAppSupport) -> dict[str, object]:
    """Retain the exact frozen scalar and semantic CDFs at beta=lambda=0."""

    rows = len(support.atom_id)
    terminal_conditional = _np.zeros(rows, dtype=float)
    proposal_conditional = _np.zeros(rows, dtype=float)
    surface_conditional = _np.zeros(rows, dtype=float)
    hazard_conditional = _np.zeros(rows, dtype=float)
    terminal_lower = _np.zeros(rows, dtype=float)
    terminal_unconditional_lower = _np.zeros(rows, dtype=float)
    terminal_unconditional_upper = _np.zeros(rows, dtype=float)
    proposal_lower = _np.zeros(rows, dtype=float)
    surface_lower = _np.zeros(rows, dtype=float)
    hazard_lower = _np.zeros(rows, dtype=float)

    terminal_indices = [i for i, value in enumerate(support.terminal) if value]
    prior_terminal = 0.0
    prior_terminal_unconditional = 0.0
    for index in terminal_indices:
        upper = support.frozen_terminal_residual_cdf_upper[index]
        terminal_lower[index] = prior_terminal
        terminal_conditional[index] = upper - prior_terminal
        prior_terminal = upper
        terminal_unconditional_lower[index] = prior_terminal_unconditional
        prior_terminal_unconditional = float(
            prior_terminal_unconditional + support.base_mass[index]
        )
        terminal_unconditional_upper[index] = prior_terminal_unconditional
    if prior_terminal_unconditional != float(support.frozen_holeout_probability):
        raise CompactAppSupportUnavailable(
            "Frozen terminal APP prefix differs from frozen holeout probability"
        )

    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    prior_proposal = 0.0
    for proposal in dict.fromkeys(
        support.spatial_proposal_order[i] for i in spatial_indices
    ):
        p_indices = [
            i
            for i in spatial_indices
            if support.spatial_proposal_order[i] == proposal
        ]
        p_upper = support.frozen_spatial_proposal_cdf_upper[p_indices[0]]
        for index in p_indices:
            proposal_lower[index] = prior_proposal
            proposal_conditional[index] = p_upper - prior_proposal
        prior_proposal = p_upper
        prior_surface = 0.0
        for surface in dict.fromkeys(support.surface_order[i] for i in p_indices):
            s_indices = [i for i in p_indices if support.surface_order[i] == surface]
            s_upper = support.frozen_surface_cdf_upper[s_indices[0]]
            for index in s_indices:
                surface_lower[index] = prior_surface
                surface_conditional[index] = s_upper - prior_surface
            prior_surface = s_upper
            prior_hazard = 0.0
            for hazard in dict.fromkeys(support.hazard_order[i] for i in s_indices):
                h_indices = [i for i in s_indices if support.hazard_order[i] == hazard]
                h_upper = support.frozen_hazard_cdf_upper[h_indices[0]]
                for index in h_indices:
                    hazard_lower[index] = prior_hazard
                    hazard_conditional[index] = h_upper - prior_hazard
                prior_hazard = h_upper
    return {
        "terminal_probability": float(support.frozen_holeout_probability),
        "spatial_probability": 1.0 - float(support.frozen_holeout_probability),
        "spatial_probability_exact_ratio": None,
        "terminal_conditional_probability": tuple(map(float, terminal_conditional)),
        "terminal_unconditional_cdf_lower": tuple(
            map(float, terminal_unconditional_lower)
        ),
        "terminal_unconditional_cdf_upper": tuple(
            map(float, terminal_unconditional_upper)
        ),
        "spatial_proposal_conditional_probability": tuple(
            map(float, proposal_conditional)
        ),
        "surface_conditional_probability": tuple(map(float, surface_conditional)),
        "hazard_conditional_probability": tuple(map(float, hazard_conditional)),
        "spatial_proposal_conditional_exact_ratio": (None,) * rows,
        "surface_conditional_exact_ratio": (None,) * rows,
        "hazard_conditional_exact_ratio": (None,) * rows,
        "terminal_cdf_lower": tuple(map(float, terminal_lower)),
        "terminal_cdf_upper": tuple(
            map(float, support.frozen_terminal_residual_cdf_upper)
        ),
        "spatial_proposal_cdf_lower": tuple(map(float, proposal_lower)),
        "spatial_proposal_cdf_upper": tuple(
            map(float, support.frozen_spatial_proposal_cdf_upper)
        ),
        "surface_cdf_lower": tuple(map(float, surface_lower)),
        "surface_cdf_upper": tuple(map(float, support.frozen_surface_cdf_upper)),
        "hazard_cdf_lower": tuple(map(float, hazard_lower)),
        "hazard_cdf_upper": tuple(map(float, support.frozen_hazard_cdf_upper)),
    }


def _select_right(upper: _Sequence[float], uniform: float) -> int:
    return min(
        int(_np.searchsorted(_np.asarray(upper, dtype=float), uniform, side="right")),
        len(upper) - 1,
    )


def _select_surface_strict(upper: _Sequence[float], uniform: float) -> int:
    return min(
        int(_np.count_nonzero(uniform > _np.asarray(upper, dtype=float))),
        len(upper) - 1,
    )


def _hierarchical_select(
    support: CompactAppSupport,
    hierarchy: _Mapping[str, object],
    uniforms: tuple[float, float, float, float],
    *,
    allow_boundary_only: bool,
) -> tuple[int, float | None]:
    holeout_u, proposal_u, surface_u, hazard_u = uniforms
    terminal_probability = float(hierarchy["terminal_probability"])
    if holeout_u < terminal_probability:
        residual = holeout_u / terminal_probability
        if residual >= 1.0:
            residual = float(_np.nextafter(1.0, 0.0))
        terminal_indices = [
            i for i, terminal in enumerate(support.terminal) if terminal
        ]
        terminal_key = (
            "terminal_cdf_upper"
            if allow_boundary_only
            else "terminal_unconditional_cdf_upper"
        )
        terminal_upper_all = hierarchy[terminal_key]
        assert isinstance(terminal_upper_all, tuple)
        terminal_upper = tuple(terminal_upper_all[i] for i in terminal_indices)
        selector_uniform = residual if allow_boundary_only else holeout_u
        return (
            terminal_indices[_select_right(terminal_upper, selector_uniform)],
            residual,
        )

    spatial_indices = [i for i, terminal in enumerate(support.terminal) if not terminal]
    proposals = tuple(
        dict.fromkeys(support.spatial_proposal_order[i] for i in spatial_indices)
    )
    proposal_upper_all = hierarchy["spatial_proposal_cdf_upper"]
    surface_lower_all = hierarchy["surface_cdf_lower"]
    surface_upper_all = hierarchy["surface_cdf_upper"]
    hazard_upper_all = hierarchy["hazard_cdf_upper"]
    assert isinstance(proposal_upper_all, tuple)
    assert isinstance(surface_lower_all, tuple)
    assert isinstance(surface_upper_all, tuple)
    assert isinstance(hazard_upper_all, tuple)
    proposal_upper = tuple(
        proposal_upper_all[
            next(i for i in spatial_indices if support.spatial_proposal_order[i] == p)
        ]
        for p in proposals
    )
    proposal = proposals[_select_right(proposal_upper, proposal_u)]
    p_indices = [
        i for i in spatial_indices if support.spatial_proposal_order[i] == proposal
    ]
    surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
    if not allow_boundary_only:
        surfaces = tuple(
            surface
            for surface in surfaces
            if (
                surface_upper_all[
                    next(i for i in p_indices if support.surface_order[i] == surface)
                ]
                > surface_lower_all[
                    next(i for i in p_indices if support.surface_order[i] == surface)
                ]
            )
        )
    if not surfaces:
        raise CompactAppSupportUnavailable("No reachable positive APP surface remains")
    surface_upper = tuple(
        surface_upper_all[
            next(i for i in p_indices if support.surface_order[i] == surface)
        ]
        for surface in surfaces
    )
    surface = surfaces[_select_surface_strict(surface_upper, surface_u)]
    s_indices = [i for i in p_indices if support.surface_order[i] == surface]
    hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
    hazard_upper = tuple(
        hazard_upper_all[
            next(i for i in s_indices if support.hazard_order[i] == hazard)
        ]
        for hazard in hazards
    )
    hazard = hazards[_select_right(hazard_upper, hazard_u)]
    matches = [i for i in s_indices if support.hazard_order[i] == hazard]
    if len(matches) != 1:
        raise CompactAppSupportUnavailable(
            "Hierarchical APP leaf is not one complete atom"
        )
    return matches[0], None


def _probability_direct_select(
    support: CompactAppSupport,
    probability: _Sequence[float],
    uniforms: tuple[float, float, float, float],
) -> int:
    """Independent tilted oracle using only the returned flat probability."""

    holeout_u, proposal_u, surface_u, hazard_u = uniforms
    retained_probability = _np.asarray(probability, dtype=float)
    if retained_probability.shape != (len(support.atom_id),):
        raise CompactAppSupportUnavailable("Direct APP probability oracle shape changed")
    terminal_upper = [0.0] * len(support.atom_id)
    terminal_probability = 0.0
    for index, is_terminal in enumerate(support.terminal):
        if not is_terminal:
            break
        mass = float(retained_probability[index])
        terminal_probability = float(terminal_probability + mass)
        terminal_upper[index] = terminal_probability
    if not 0.0 < terminal_probability < 1.0:
        raise CompactAppSupportUnavailable(
            "Direct APP probability oracle terminal prefix is invalid"
        )
    if holeout_u < terminal_probability:
        for index, is_terminal in enumerate(support.terminal):
            if is_terminal and holeout_u < terminal_upper[index]:
                return index
        raise CompactAppSupportUnavailable(
            "Direct APP probability oracle could not resolve terminal prefix"
        )

    def direct_mass(indices: _Sequence[int]) -> _Fraction:
        retained = _Fraction(0, 1)
        for index in indices:
            retained += _Fraction.from_float(float(retained_probability[index]))
        return retained

    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    spatial_mass = direct_mass(spatial_indices)
    proposal_fraction = _Fraction.from_float(proposal_u)
    proposals = tuple(
        dict.fromkeys(support.spatial_proposal_order[i] for i in spatial_indices)
    )
    running_proposal = _Fraction(0, 1)
    selected_proposal: int | None = None
    selected_p_indices: list[int] = []
    selected_p_mass = _Fraction(0, 1)
    for proposal in proposals:
        p_indices = [
            i
            for i in spatial_indices
            if support.spatial_proposal_order[i] == proposal
        ]
        p_mass = direct_mass(p_indices)
        running_proposal += p_mass
        if proposal_fraction < running_proposal / spatial_mass:
            selected_proposal = proposal
            selected_p_indices = p_indices
            selected_p_mass = p_mass
            break
    if selected_proposal is None:
        raise CompactAppSupportUnavailable(
            "Direct APP probability oracle could not resolve a proposal"
        )

    surface_fraction = _Fraction.from_float(surface_u)
    surfaces = tuple(
        dict.fromkeys(support.surface_order[i] for i in selected_p_indices)
    )
    running_surface = _Fraction(0, 1)
    selected_surface: int | None = None
    selected_s_indices: list[int] = []
    selected_s_mass = _Fraction(0, 1)
    for surface in surfaces:
        s_indices = [
            i for i in selected_p_indices if support.surface_order[i] == surface
        ]
        s_mass = direct_mass(s_indices)
        if s_mass == 0:
            continue
        running_surface += s_mass
        if surface_fraction <= running_surface / selected_p_mass:
            selected_surface = surface
            selected_s_indices = s_indices
            selected_s_mass = s_mass
            break
    if selected_surface is None:
        raise CompactAppSupportUnavailable(
            "Direct APP probability oracle could not resolve a surface"
        )

    hazard_fraction = _Fraction.from_float(hazard_u)
    hazards = tuple(
        dict.fromkeys(support.hazard_order[i] for i in selected_s_indices)
    )
    running_hazard = _Fraction(0, 1)
    for hazard in hazards:
        matches = [
            i for i in selected_s_indices if support.hazard_order[i] == hazard
        ]
        if len(matches) != 1:
            raise CompactAppSupportUnavailable(
                "Direct APP probability oracle leaf is not one whole atom"
            )
        index = matches[0]
        running_hazard += _Fraction.from_float(float(retained_probability[index]))
        if hazard_fraction < running_hazard / selected_s_mass:
            return index
    raise CompactAppSupportUnavailable(
        "Direct APP probability oracle could not resolve a hazard"
    )


def _frozen_direct_select(
    support: CompactAppSupport,
    uniforms: tuple[float, float, float, float],
) -> int:
    """Independent zero-tilt oracle reading the frozen source CDFs directly."""

    holeout_u, proposal_u, surface_u, hazard_u = uniforms
    holeout = float(support.frozen_holeout_probability)
    if holeout_u < holeout:
        residual = holeout_u / holeout
        if residual >= 1.0:
            residual = float(_np.nextafter(1.0, 0.0))
        terminal_indices = [i for i, value in enumerate(support.terminal) if value]
        uppers = tuple(
            support.frozen_terminal_residual_cdf_upper[i]
            for i in terminal_indices
        )
        return terminal_indices[_select_right(uppers, residual)]

    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    proposals = tuple(
        dict.fromkeys(support.spatial_proposal_order[i] for i in spatial_indices)
    )
    proposal_uppers = tuple(
        support.frozen_spatial_proposal_cdf_upper[
            next(i for i in spatial_indices if support.spatial_proposal_order[i] == p)
        ]
        for p in proposals
    )
    proposal = proposals[_select_right(proposal_uppers, proposal_u)]
    p_indices = [
        i for i in spatial_indices if support.spatial_proposal_order[i] == proposal
    ]
    surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
    surface_uppers = tuple(
        support.frozen_surface_cdf_upper[
            next(i for i in p_indices if support.surface_order[i] == surface)
        ]
        for surface in surfaces
    )
    surface = surfaces[_select_surface_strict(surface_uppers, surface_u)]
    s_indices = [i for i in p_indices if support.surface_order[i] == surface]
    hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
    hazard_uppers = tuple(
        support.frozen_hazard_cdf_upper[
            next(i for i in s_indices if support.hazard_order[i] == hazard)
        ]
        for hazard in hazards
    )
    hazard = hazards[_select_right(hazard_uppers, hazard_u)]
    matches = [i for i in s_indices if support.hazard_order[i] == hazard]
    if len(matches) != 1:
        raise CompactAppSupportUnavailable(
            "Frozen direct APP oracle leaf is not one whole atom"
        )
    return matches[0]


def _build_readiness(
    support: CompactAppSupport,
    authority: _TrustedSyntheticAppAuthority,
    runtime: str,
) -> AppSelectorReadiness:
    kinds = _np.asarray(support.donor_ability_value_kind, dtype=str)
    abilities = _np.asarray(support.donor_ability_sg, dtype=float)
    availability = _np.asarray(support.donor_ability_available, dtype=bool)
    imputed = kinds == _IMPUTED_VALUE_KIND
    observed = kinds == _OBSERVED_VALUE_KIND
    mixed_groups = 0
    for group_id in dict.fromkeys(support.original_donor_group_id):
        states = {
            support.donor_ability_value_kind[index]
            for index, value in enumerate(support.original_donor_group_id)
            if value == group_id
        }
        mixed_groups += int(len(states) > 1)
    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    return AppSelectorReadiness(
        contract=APP_WEIGHTING_SELECTOR_CONTRACT,
        authority_manifest_sha256=authority.manifest_sha256,
        runtime_binding_sha256=runtime,
        support_sha256=authority.support_sha256,
        frozen_cdf_sha256=authority.frozen_cdf_sha256,
        rows=len(support.atom_id),
        terminal_donors=int(_np.count_nonzero(support.terminal)),
        spatial_proposals=len(
            {support.spatial_proposal_order[i] for i in spatial_indices}
        ),
        original_donor_groups=len(set(support.original_donor_group_id)),
        mixed_policy_state_groups=mixed_groups,
        observed_rows=int(_np.count_nonzero(observed)),
        neutral_imputed_rows=int(_np.count_nonzero(imputed)),
        availability_preserved=bool(
            not availability[imputed].any() and availability[observed].all()
        ),
        positive_zero_preserved=bool(
            _np.equal(abilities[imputed], 0.0).all()
            and not _np.signbit(abilities[imputed]).any()
        ),
        synthetic_fixture_only=True,
        weighting_ready=True,
        selector_ready=True,
        real_chain_ready=False,
        integration_ready=False,
        blockers=(
            "no_externally_reviewed_real_app_support_authority",
            "no_app_selector_to_complete_seam_integration",
            "no_streaming_or_full_chain_integration",
            "no_mean_solve_or_simulation_integration",
        ),
    )


class CompactAppSelectorSession:
    """Opaque validated capability for synthetic APP weighting and selection."""

    __slots__ = ("_support", "_authority", "_readiness")

    def __init__(self, *_: object, **__: object) -> None:
        raise CompactAppSupportUnavailable(
            "APP sessions are minted only by admit_synthetic_app_support"
        )

    @property
    def readiness(self) -> AppSelectorReadiness:
        if type(self) is not CompactAppSelectorSession:
            raise CompactAppSupportUnavailable(
                "Synthetic APP session has the wrong exact type"
            )
        support = CompactAppSelectorSession._live_support(self)
        runtime = _validate_runtime()
        authority = object.__getattribute__(self, "_authority")
        if not _is_closed_authority(authority, runtime):
            raise CompactAppSupportUnavailable("Synthetic APP session authority changed")
        return _build_readiness(support, authority, runtime)

    def support_copy(self) -> CompactAppSupport:
        if type(self) is not CompactAppSelectorSession:
            raise CompactAppSupportUnavailable(
                "Synthetic APP session has the wrong exact type"
            )
        support = CompactAppSelectorSession._live_support(self)
        return CompactAppSupport(
            **{
                field.name: getattr(support, field.name)
                for field in _dataclass_fields(CompactAppSupport)
            }
        )

    def _live_support(self) -> CompactAppSupport:
        if type(self) is not CompactAppSelectorSession:
            raise CompactAppSupportUnavailable(
                "Synthetic APP session has the wrong exact type"
            )
        runtime = _validate_runtime()
        authority = object.__getattribute__(self, "_authority")
        if not _is_closed_authority(authority, runtime):
            raise CompactAppSupportUnavailable("Synthetic APP session authority changed")
        canonical = _canonicalize_support(object.__getattribute__(self, "_support"))
        if (
            _support_sha256(canonical) != authority.support_sha256
            or _frozen_cdf_sha256(canonical) != authority.frozen_cdf_sha256
        ):
            raise CompactAppSupportUnavailable("Synthetic APP session support changed")
        return canonical

    def weight(self, parameters: AppWeightingParameters) -> WeightedAppSupport:
        if type(self) is not CompactAppSelectorSession:
            raise CompactAppSupportUnavailable(
                "Synthetic APP session has the wrong exact type"
            )
        anchor_parameters, beta_lambda_zero = _validated_weighting_parameters(parameters)
        support = CompactAppSelectorSession._live_support(self)
        authority = object.__getattribute__(self, "_authority")
        if anchor_parameters.reward_reference_sha256 != authority.reward_reference_sha256:
            raise CompactAppSupportUnavailable("APP weighting reward reference changed")
        abilities = _np.asarray(support.donor_ability_sg, dtype=float)
        kinds = _np.asarray(support.donor_ability_value_kind, dtype=str)
        imputed = kinds == _IMPUTED_VALUE_KIND
        observed = kinds == _OBSERVED_VALUE_KIND
        available = _np.asarray(support.donor_ability_available, dtype=bool)
        if (
            available[imputed].any()
            or not available[observed].all()
            or not _np.equal(abilities[imputed], 0.0).all()
            or _np.signbit(abilities[imputed]).any()
        ):
            raise CompactAppSupportUnavailable(
                "APP policy-v2 availability or exact positive zero changed"
            )
        try:
            atom_mass = _whole_atom_probabilities(
                base_mass=support.base_mass,
                fixed_reference_reward=support.fixed_reference_reward,
                donor_ability_sg=tuple(map(float, abilities)),
                donor_group_id=support.original_donor_group_id,
                category="app",
                parameters=anchor_parameters,
            )
        except (RuntimeError, ValueError, AssertionError) as exc:
            raise CompactAppSupportUnavailable(
                f"Frozen whole-atom weighting rejected APP support: {exc}"
            ) from exc
        probability = _np.asarray(atom_mass.probability, dtype=float)
        neutral_probability = _np.asarray(
            atom_mass.neutral_probability_mass, dtype=float
        )
        rows = len(support.atom_id)
        if (
            probability.shape != (rows,)
            or neutral_probability.shape != (rows,)
            or not _np.isfinite(probability).all()
            or not _np.isfinite(neutral_probability).all()
        ):
            raise CompactAppSupportUnavailable(
                "Whole-atom APP weighting returned malformed arrays"
            )
        probability_total = float(probability.sum())
        if (
            probability_total <= 0.0
            or not _math.isclose(
                probability_total, 1.0, rel_tol=0.0, abs_tol=_MASS_TOLERANCE
            )
        ):
            raise CompactAppSupportUnavailable(
                "Unchanged whole-atom APP probability does not close within the lock"
            )
        probability = _np.ascontiguousarray(probability)
        if beta_lambda_zero:
            if not _np.allclose(
                probability,
                _np.asarray(support.base_mass, dtype=float),
                atol=2e-15,
                rtol=2e-15,
            ):
                raise CompactAppSupportUnavailable(
                    "Zero-tilt APP weighting did not replay frozen base mass"
                )
            hierarchy = _frozen_hierarchy(support)
            selection_mode = APP_ZERO_FAST_PATH_CONTRACT
            factorization_from_final = False
        else:
            hierarchy = _hierarchy_from_probability(support, probability)
            selection_mode = (
                "weighted_app_binary64_pathwise_from_unchanged_flat_probability_v2"
            )
            factorization_from_final = True
        return WeightedAppSupport(
            support_sha256=authority.support_sha256,
            parameter_sha256=anchor_parameters.parameter_sha256,
            atom_id=tuple(support.atom_id),
            probability=tuple(map(float, probability)),
            neutral_probability_mass=tuple(map(float, neutral_probability)),
            donor_group_mass_before=tuple(
                map(float, atom_mass.donor_group_mass_before)
            ),
            donor_group_mass_after=tuple(map(float, atom_mass.donor_group_mass_after)),
            clipped_donor_logs=int(atom_mass.clipped_donor_logs),
            clipped_target_logs=int(atom_mass.clipped_target_logs),
            selection_mode=selection_mode,
            factorization_from_final_probability=factorization_from_final,
            exact_spatial_mass_reconstruction=not beta_lambda_zero,
            binary64_pathwise_exact=True,
            **hierarchy,
        )

    def select(
        self,
        parameters: AppWeightingParameters,
        uniforms: AppSemanticUniforms,
    ) -> AppSelectionResult:
        if type(self) is not CompactAppSelectorSession:
            raise CompactAppSupportUnavailable(
                "Synthetic APP session has the wrong exact type"
            )
        retained = _validated_semantic_uniforms(uniforms)
        support = CompactAppSelectorSession._live_support(self)
        weighted = CompactAppSelectorSession.weight(self, parameters)
        hierarchy = {
            name: getattr(weighted, name)
            for name in (
                "terminal_probability",
                "terminal_cdf_lower",
                "terminal_cdf_upper",
                "terminal_unconditional_cdf_lower",
                "terminal_unconditional_cdf_upper",
                "spatial_proposal_cdf_lower",
                "spatial_proposal_cdf_upper",
                "surface_cdf_lower",
                "surface_cdf_upper",
                "hazard_cdf_lower",
                "hazard_cdf_upper",
            )
        }
        allow_boundary_only = weighted.selection_mode == APP_ZERO_FAST_PATH_CONTRACT
        hierarchical_index, terminal_residual = _hierarchical_select(
            support,
            hierarchy,
            retained,
            allow_boundary_only=allow_boundary_only,
        )
        if allow_boundary_only:
            flat_index = _frozen_direct_select(support, retained)
        else:
            flat_index = _probability_direct_select(
                support, weighted.probability, retained
            )
        if hierarchical_index != flat_index:
            raise CompactAppSupportUnavailable(
                "Flat and hierarchical retained APP uniform selections diverged"
            )
        index = hierarchical_index
        return AppSelectionResult(
            atom_index=index,
            atom_id=support.atom_id[index],
            terminal=support.terminal[index],
            terminal_donor_order=support.terminal_donor_order[index],
            spatial_proposal_order=support.spatial_proposal_order[index],
            surface_order=support.surface_order[index],
            hazard_order=support.hazard_order[index],
            stroke_cost=support.stroke_cost[index],
            next_state_key=support.next_state_key[index],
            boundary_only=support.boundary_only[index],
            terminal_probability=weighted.terminal_probability,
            terminal_residual_uniform=terminal_residual,
            selection_mode=weighted.selection_mode,
            hierarchical_atom_id=support.atom_id[hierarchical_index],
            flat_atom_id=support.atom_id[flat_index],
            flat_hierarchical_equal=True,
            semantic_uniforms=AppSemanticUniforms(*retained),
        )


def _mint_session(
    support: CompactAppSupport,
    authority: _TrustedSyntheticAppAuthority,
    readiness: AppSelectorReadiness,
) -> CompactAppSelectorSession:
    session = object.__new__(CompactAppSelectorSession)
    session._support = support
    session._authority = authority
    session._readiness = readiness
    return session


def admit_synthetic_app_support(
    support: CompactAppSupport,
    *,
    authority_manifest: _Mapping[str, object],
) -> CompactAppSelectorSession:
    """Admit only the exact source-pinned synthetic APP fixture."""

    runtime = _validate_runtime()
    authority = _trusted_authority(authority_manifest, runtime)
    canonical = _canonicalize_support(support)
    observed_support_sha = _support_sha256(canonical)
    observed_cdf_sha = _frozen_cdf_sha256(canonical)
    if observed_support_sha != authority.support_sha256:
        raise CompactAppSupportUnavailable(
            "Candidate APP support is outside the closed synthetic authority"
        )
    if observed_cdf_sha != authority.frozen_cdf_sha256:
        raise CompactAppSupportUnavailable(
            "Candidate APP CDFs are outside the closed synthetic authority"
        )
    if canonical.reward_reference_sha256 != authority.reward_reference_sha256:
        raise CompactAppSupportUnavailable("Synthetic APP reward reference changed")

    readiness = _build_readiness(canonical, authority, runtime)
    return _mint_session(canonical, authority, readiness)


__all__ = [
    "COMPACT_APP_SUPPORT_CONTRACT",
    "SYNTHETIC_APP_AUTHORITY_CONTRACT",
    "APP_WEIGHTING_SELECTOR_CONTRACT",
    "APP_ZERO_FAST_PATH_CONTRACT",
    "APP_SELECTION_BOUNDARY_CONTRACT",
    "INTEGRATION_STATUS",
    "CompactAppSupportUnavailable",
    "CompactAppSupport",
    "AppWeightingParameters",
    "AppSemanticUniforms",
    "WeightedAppSupport",
    "AppSelectionResult",
    "AppSelectorReadiness",
    "CompactAppSelectorSession",
    "synthetic_app_authority_manifest",
    "admit_synthetic_app_support",
]

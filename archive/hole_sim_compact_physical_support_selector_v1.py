"""Synthetic-only whole-atom weighting and compact TEE selection seam.

This archive module is deliberately narrower than an expectation callback.  It
accepts one already-expanded conditional support, preserves every complete
atom while applying the frozen whole-atom quality weighting primitive, and
selects with the proposal/surface/hazard uniforms already owned by the parent
chain.  APP is deliberately rejected until its separate holeout uniform and
terminal-residual donor replay are represented.  This module cannot solve,
simulate, or authorize a real-chain support.

Only one source-pinned synthetic fixture is admitted.  A caller-computed hash
is evidence about a candidate, not authority, so candidate digests can never
replace the closed approval lock embedded below.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass, fields as _dataclass_fields
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
    CATEGORIES as _CATEGORIES,
    TransitionQualityParameters as _TransitionQualityParameters,
    whole_atom_probabilities as _whole_atom_probabilities,
)


COMPACT_SUPPORT_CONTRACT = "synthetic_compact_complete_tee_support_v2"
SYNTHETIC_AUTHORITY_CONTRACT = "closed_synthetic_compact_tee_support_authority_v2"
WEIGHTING_SELECTOR_CONTRACT = (
    "whole_atom_weighted_tee_proposal_surface_hazard_selector_v2"
)
ZERO_FAST_PATH_CONTRACT = "exact_frozen_parent_cdf_beta_lambda_zero_v2"
SELECTION_BOUNDARY_CONTRACT = (
    "proposal_right_surface_strict_greater_hazard_right_v1"
)
INTEGRATION_STATUS = "synthetic_tee_fixture_only_callback_integration_hold"

_ANCHOR_DESCRIPTOR = "archive/hole_sim_transition_mean_anchor.py"
_POLICY_DESCRIPTOR = "archive/hole_sim_donor_ability_policy.py"
_PARENT_SELECTOR_DESCRIPTOR = "archive/hole_sim_tee_app_complete_atom_support.py"
_ANCHOR_SOURCE_SHA256 = (
    "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
)
_POLICY_SOURCE_SHA256 = (
    "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
)
_PARENT_SELECTOR_SOURCE_SHA256 = (
    "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
)
_POLICY_IDENTITY_SHA256 = (
    "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
)

_MAX_ATOMS = 16
_MAX_PROPOSALS = 4
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
    "proposal_order",
    "surface_order",
    "hazard_order",
    "frozen_proposal_cdf_upper",
    "frozen_surface_cdf_upper",
    "frozen_hazard_cdf_upper",
    "boundary_only",
    "complete_atom",
    "terminal",
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


class CompactSupportUnavailable(ValueError):
    """Raised when compact weighting or selection is not research-admissible."""


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _stable_sha(value: object) -> str:
    encoded = _json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded).hexdigest()


def _sha256_file(path: _Path) -> str:
    digest = _sha256()
    with _Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CompactSupportUnavailable(f"{label} must be text")
    result = _unicodedata.normalize("NFKC", value).strip()
    if not result and not allow_empty:
        raise CompactSupportUnavailable(f"{label} must be populated")
    return result


def _timestamp(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None or bool(_pd.isna(value)):
        if optional:
            return None
        raise CompactSupportUnavailable(f"{label} must be populated")
    try:
        stamp = _pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CompactSupportUnavailable(f"{label} is not a timestamp") from exc
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactSupportUnavailable(f"{label} must be numeric") from exc
    if not _math.isfinite(result):
        raise CompactSupportUnavailable(f"{label} must be finite")
    return result


def _optional_finite(value: object, label: str) -> float | None:
    if value is None or bool(_pd.isna(value)):
        return None
    return _finite(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, (bool, _np.bool_)):
        raise CompactSupportUnavailable(f"{label} must be boolean")
    return bool(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, (bool, _np.bool_)):
        raise CompactSupportUnavailable(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactSupportUnavailable(f"{label} must be an integer") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactSupportUnavailable(f"{label} must be an integer") from exc
    if not _math.isfinite(exact) or exact != float(result):
        raise CompactSupportUnavailable(f"{label} must be an integer")
    return result


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
            "role": "parent_semantic_selector_reference",
            "descriptor": _PARENT_SELECTOR_DESCRIPTOR,
            "source_sha256": _PARENT_SELECTOR_SOURCE_SHA256,
        },
    )


def _runtime_binding_payload() -> dict[str, object]:
    return {
        "atom_weight_contract": _ATOM_WEIGHT_CONTRACT,
        "policy_contract": _POLICY_CONTRACT,
        "policy_status": _POLICY_STATUS,
        "policy_identity_sha256": _POLICY_IDENTITY_SHA256,
        "policy_category_mapping": dict(_CATEGORY_ABILITY_MAPPING),
        "parent_selection_contract": "hierarchical_complete_atom_crn_v2",
        "selection_boundary_contract": SELECTION_BOUNDARY_CONTRACT,
        "supported_categories": ["ott"],
        "tee_terminal_contract": "all_complete_atoms_nonterminal_v1",
        "app_selection_contract": (
            "fail_closed_until_separate_holeout_uniform_and_terminal_residual_replay"
        ),
        "source_bindings": list(_source_bindings()),
    }


def _closed_authority_envelope() -> dict[str, object]:
    """Return the private source literal approval lock.

    The three digest placeholders are replaced only after the bounded fixture
    is independently generated and reviewed.  They are intentionally not
    derived from a candidate supplied to ``admit_synthetic_support``.
    """

    manifest = {
        "authority_contract": SYNTHETIC_AUTHORITY_CONTRACT,
        "authority_id": "compact_tee_weighting_selector_fixture_v2",
        "scope": "exact_source_pinned_synthetic_fixture_only",
        "research_only": True,
        "production_enabled": False,
        "real_chain_ready": False,
        "integration_ready": False,
        "support_contract": COMPACT_SUPPORT_CONTRACT,
        "weighting_selector_contract": WEIGHTING_SELECTOR_CONTRACT,
        "zero_fast_path_contract": ZERO_FAST_PATH_CONTRACT,
        "selection_boundary_contract": SELECTION_BOUNDARY_CONTRACT,
        "supported_categories": ["ott"],
        "tee_terminal_contract": "all_complete_atoms_nonterminal_v1",
        "tee_terminal_atoms_allowed": False,
        "app_selection_ready": False,
        "support_sha256": (
            "6d457cefad522e1e82e3ee4aba10bf3c8dd7c42197f8860644bd328d986bb85a"
        ),
        "frozen_cdf_sha256": (
            "c45497f114074fb45862efa6ba7776c25477d5ffa98cd40ba2962e300c7de7aa"
        ),
        "reward_reference_sha256": (
            "a70ebc4aa3ec0dab67eb4d69722e48ec3f1fd6ec390643b53f21a63d6c45219b"
        ),
        "policy_identity_sha256": _POLICY_IDENTITY_SHA256,
        "runtime_binding_sha256": (
            "fdbba2d6a778b17272cbfd9ad2e32502541904f10118d0cb592631532e9d1cb0"
        ),
        "source_bindings": list(_source_bindings()),
        "limits": {
            "max_atoms": _MAX_ATOMS,
            "max_proposals": _MAX_PROPOSALS,
            "max_surfaces_per_proposal": _MAX_SURFACES_PER_PROPOSAL,
            "max_hazards_per_surface": _MAX_HAZARDS_PER_SURFACE,
            "max_abs_log_weight": _MAX_ABS_LOG_WEIGHT,
        },
    }
    return {
        "manifest": manifest,
        "manifest_sha256": (
            "990e7507684b48d2a6301cb61b4e3947b57a216d18db23ddb21f96ec107933a3"
        ),
    }


def synthetic_authority_manifest() -> dict[str, object]:
    """Return a defensive reporting copy of the closed synthetic authority."""

    return _json.loads(_json.dumps(_closed_authority_envelope()))


def _validate_runtime() -> str:
    if _ATOM_WEIGHT_CONTRACT != "whole_atom_donor_detilt_target_reapply_lambda_v1":
        raise CompactSupportUnavailable("Frozen whole-atom contract changed")
    if _POLICY_CONTRACT != "frozen_strict_prior_donor_ability_policy_v2":
        raise CompactSupportUnavailable("Frozen donor-ability policy contract changed")
    payload = _policy_payload()
    if payload.get("policy_identity_sha256") != _POLICY_IDENTITY_SHA256:
        raise CompactSupportUnavailable("Frozen donor-ability policy identity changed")
    archive = _Path(__file__).resolve().parent
    observed = {
        _ANCHOR_DESCRIPTOR: _sha256_file(archive / "hole_sim_transition_mean_anchor.py"),
        _POLICY_DESCRIPTOR: _sha256_file(archive / "hole_sim_donor_ability_policy.py"),
        _PARENT_SELECTOR_DESCRIPTOR: _sha256_file(
            archive / "hole_sim_tee_app_complete_atom_support.py"
        ),
    }
    expected = {
        binding["descriptor"]: binding["source_sha256"]
        for binding in _source_bindings()
    }
    if observed != expected:
        changed = sorted(
            descriptor
            for descriptor in expected
            if observed.get(descriptor) != expected[descriptor]
        )
        raise CompactSupportUnavailable(f"Frozen source binding changed: {changed}")
    return _stable_sha(_runtime_binding_payload())


@_dataclass(frozen=True)
class CompactPhysicalSupport:
    """Ordered arrays for one complete conditional physical support."""

    support_key: str
    reward_reference_sha256: str
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
    proposal_order: tuple[object, ...]
    surface_order: tuple[object, ...]
    hazard_order: tuple[object, ...]
    frozen_proposal_cdf_upper: tuple[object, ...]
    frozen_surface_cdf_upper: tuple[object, ...]
    frozen_hazard_cdf_upper: tuple[object, ...]
    boundary_only: tuple[object, ...]
    complete_atom: tuple[object, ...]
    terminal: tuple[object, ...]
    stroke_cost: tuple[object, ...]
    next_state_key: tuple[object, ...]


def _canonicalize_support(candidate: CompactPhysicalSupport) -> CompactPhysicalSupport:
    if type(candidate) is not CompactPhysicalSupport:
        raise CompactSupportUnavailable("Compact support has the wrong type")
    values: dict[str, object] = {
        "support_key": _text(candidate.support_key, "support_key"),
        "reward_reference_sha256": str(candidate.reward_reference_sha256),
    }
    if not _is_sha256(values["reward_reference_sha256"]):
        raise CompactSupportUnavailable("reward_reference_sha256 is invalid")
    raw_sequences: dict[str, tuple[object, ...]] = {}
    for name in _SEQUENCE_FIELDS:
        raw = getattr(candidate, name)
        if isinstance(raw, (str, bytes)):
            raise CompactSupportUnavailable(f"{name} must be an ordered array")
        try:
            raw_sequences[name] = tuple(raw)
        except TypeError as exc:
            raise CompactSupportUnavailable(f"{name} must be an ordered array") from exc
    lengths = {name: len(sequence) for name, sequence in raw_sequences.items()}
    if len(set(lengths.values())) != 1:
        raise CompactSupportUnavailable(f"Compact support arrays do not align: {lengths}")
    rows = next(iter(lengths.values()))
    if rows <= 0:
        raise CompactSupportUnavailable("Compact support must contain atoms")
    if rows > _MAX_ATOMS:
        raise CompactSupportUnavailable(
            f"Compact support atom cap exceeded: {rows}>{_MAX_ATOMS}"
        )

    values["atom_id"] = tuple(_text(value, "atom_id") for value in raw_sequences["atom_id"])
    values["base_mass"] = tuple(
        _finite(value, "base_mass") for value in raw_sequences["base_mass"]
    )
    values["fixed_reference_reward"] = tuple(
        _finite(value, "fixed_reference_reward")
        for value in raw_sequences["fixed_reference_reward"]
    )
    for name in (
        "bridge_row_id",
        "donor_player_id",
        "donor_event_key",
        "target_event_key",
        "donor_ability_source_column",
        "donor_ability_value_kind",
        "donor_ability_policy_sha256",
        "donor_ability_row_sha256",
        "original_donor_group_id",
    ):
        values[name] = tuple(_text(value, name) for value in raw_sequences[name])
    values["category"] = tuple(
        _text(value, "category").casefold() for value in raw_sequences["category"]
    )
    for name in ("donor_event_start", "target_event_start"):
        values[name] = tuple(
            _timestamp(value, name) for value in raw_sequences[name]
        )
    values["donor_skill_date"] = tuple(
        _timestamp(value, "donor_skill_date", optional=True)
        for value in raw_sequences["donor_skill_date"]
    )
    values["donor_ability_available"] = tuple(
        _boolean(value, "donor_ability_available")
        for value in raw_sequences["donor_ability_available"]
    )
    values["donor_ability_raw_strict_prior"] = tuple(
        _optional_finite(value, "donor_ability_raw_strict_prior")
        for value in raw_sequences["donor_ability_raw_strict_prior"]
    )
    values["donor_ability_sg"] = tuple(
        _finite(value, "donor_ability_sg")
        for value in raw_sequences["donor_ability_sg"]
    )
    values["donor_ability_was_clipped"] = tuple(
        _boolean(value, "donor_ability_was_clipped")
        for value in raw_sequences["donor_ability_was_clipped"]
    )
    for name in ("proposal_order", "surface_order", "hazard_order", "stroke_cost"):
        values[name] = tuple(_integer(value, name) for value in raw_sequences[name])
    for name in (
        "frozen_proposal_cdf_upper",
        "frozen_surface_cdf_upper",
        "frozen_hazard_cdf_upper",
    ):
        values[name] = tuple(_finite(value, name) for value in raw_sequences[name])
    values["complete_atom"] = tuple(
        _boolean(value, "complete_atom") for value in raw_sequences["complete_atom"]
    )
    values["boundary_only"] = tuple(
        _boolean(value, "boundary_only") for value in raw_sequences["boundary_only"]
    )
    values["terminal"] = tuple(
        _boolean(value, "terminal") for value in raw_sequences["terminal"]
    )
    values["next_state_key"] = tuple(
        _text(value, "next_state_key", allow_empty=True)
        for value in raw_sequences["next_state_key"]
    )
    canonical = CompactPhysicalSupport(**values)  # type: ignore[arg-type]
    _validate_support_semantics(canonical)
    return canonical


def _node_cdf(
    keys: _Sequence[tuple[int, ...]],
    values: _Sequence[float],
    label: str,
    *,
    allow_leading_zero: bool = False,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    nodes: list[tuple[int, ...]] = []
    uppers: list[float] = []
    seen: dict[tuple[int, ...], float] = {}
    for key, value in zip(keys, values, strict=True):
        if key in seen:
            if float(value) != seen[key]:
                raise CompactSupportUnavailable(f"{label} varies inside hierarchy node")
            continue
        seen[key] = float(value)
        nodes.append(key)
        uppers.append(float(value))
    vector = _np.asarray(uppers, dtype=float)
    lower_invalid = (vector < 0.0).any() if allow_leading_zero else (vector <= 0.0).any()
    monotonic_invalid = (_np.diff(vector) <= 0.0).any()
    if (
        not len(vector)
        or not _np.isfinite(vector).all()
        or lower_invalid
        or (vector > 1.0).any()
        or monotonic_invalid
        or float(vector[-1]) != 1.0
    ):
        raise CompactSupportUnavailable(f"{label} is not an exact closing ordered CDF")
    return tuple(nodes), tuple(map(float, vector))


def _validate_support_semantics(support: CompactPhysicalSupport) -> None:
    rows = len(support.atom_id)
    if len(set(support.atom_id)) != rows:
        raise CompactSupportUnavailable("Atom IDs are duplicated")
    base = _np.asarray(support.base_mass, dtype=float)
    boundary = _np.asarray(support.boundary_only, dtype=bool)
    if (
        (base < 0.0).any()
        or not _np.array_equal(base == 0.0, boundary)
        or not _math.isclose(
        float(base.sum()), 1.0, rel_tol=0.0, abs_tol=_MASS_TOLERANCE
        )
    ):
        raise CompactSupportUnavailable(
            "Base mass must be positive except for exact boundary-only zeros and close exactly"
        )
    categories = set(support.category)
    if len(categories) != 1 or not categories.issubset(set(_CATEGORIES)):
        raise CompactSupportUnavailable("One compact support must have one valid category")
    category = support.category[0]
    if category == "app":
        raise CompactSupportUnavailable(
            "Compact selector is TEE-only until APP holeout and terminal-residual "
            "uniform semantics are represented"
        )
    if category != "ott":
        raise CompactSupportUnavailable(
            f"Compact selector does not support category={category!r}"
        )
    if any(support.terminal):
        raise CompactSupportUnavailable(
            "TEE compact support cannot contain terminal atoms"
        )
    if set(support.donor_ability_source_column) != {
        _CATEGORY_ABILITY_MAPPING[category]
    }:
        raise CompactSupportUnavailable("Ability source does not match support category")
    if not all(support.complete_atom):
        raise CompactSupportUnavailable("Every retained row must be a complete atom")
    if any(value < 0 for value in support.proposal_order):
        raise CompactSupportUnavailable("Proposal order cannot be negative")
    if any(value < 0 for value in support.surface_order):
        raise CompactSupportUnavailable("Surface order cannot be negative")
    if any(value < 0 for value in support.hazard_order):
        raise CompactSupportUnavailable("Hazard order cannot be negative")
    if any(value <= 0 for value in support.stroke_cost):
        raise CompactSupportUnavailable("Each complete atom needs positive stroke cost")
    for terminal, next_state in zip(support.terminal, support.next_state_key, strict=True):
        if terminal != (next_state == ""):
            raise CompactSupportUnavailable(
                "Terminal atoms need empty next state and nonterminal atoms need a next state"
            )

    order = tuple(
        zip(support.proposal_order, support.surface_order, support.hazard_order, strict=True)
    )
    if order != tuple(sorted(order)) or len(set(order)) != rows:
        raise CompactSupportUnavailable(
            "Atoms must be unique and ordered by proposal/surface/hazard"
        )
    proposals = tuple(dict.fromkeys(support.proposal_order))
    if proposals != tuple(range(len(proposals))):
        raise CompactSupportUnavailable("Proposal order must be contiguous from zero")
    if len(proposals) > _MAX_PROPOSALS:
        raise CompactSupportUnavailable("Proposal cap exceeded")
    for proposal in proposals:
        indices = [i for i, value in enumerate(support.proposal_order) if value == proposal]
        surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in indices))
        if surfaces != tuple(range(len(surfaces))):
            raise CompactSupportUnavailable("Surface order must be contiguous within proposal")
        if len(surfaces) > _MAX_SURFACES_PER_PROPOSAL:
            raise CompactSupportUnavailable("Surface cap exceeded")
        terminal_values = {support.terminal[i] for i in indices}
        if len(terminal_values) != 1:
            raise CompactSupportUnavailable("One proposal mixes terminal and live atoms")
        if True in terminal_values and (
            len(indices) != 1
            or support.surface_order[indices[0]] != 0
            or support.hazard_order[indices[0]] != 0
        ):
            raise CompactSupportUnavailable(
                "A terminal proposal must be one inseparable complete atom"
            )
        for surface in surfaces:
            branch = [i for i in indices if support.surface_order[i] == surface]
            hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in branch))
            if hazards != tuple(range(len(hazards))):
                raise CompactSupportUnavailable(
                    "Hazard order must be contiguous within proposal/surface"
                )
            if len(hazards) > _MAX_HAZARDS_PER_SURFACE:
                raise CompactSupportUnavailable("Hazard cap exceeded")
            boundary_values = {support.boundary_only[i] for i in branch}
            if len(boundary_values) != 1:
                raise CompactSupportUnavailable(
                    "Boundary-only status varies within one surface branch"
                )
            if True in boundary_values and (
                surface != 0
                or len(branch) != 1
                or support.hazard_order[branch[0]] != 0
                or support.terminal[branch[0]]
            ):
                raise CompactSupportUnavailable(
                    "Boundary-only mass must be one leading nonterminal surface atom"
                )

    policy_frame = _pd.DataFrame(
        {column: getattr(support, column) for column in _POLICY_FRAME_COLUMNS}
    )
    try:
        _validate_applied_policy_frame(policy_frame)
    except Exception as exc:
        raise CompactSupportUnavailable(
            f"Policy-v2 row attestation failed: {exc}"
        ) from exc
    kinds = _np.asarray(support.donor_ability_value_kind, dtype=str)
    available = _np.asarray(support.donor_ability_available, dtype=bool)
    abilities = _np.asarray(support.donor_ability_sg, dtype=float)
    imputed = kinds == _IMPUTED_VALUE_KIND
    observed = kinds == _OBSERVED_VALUE_KIND
    if not _np.logical_or(imputed, observed).all():
        raise CompactSupportUnavailable("Unexpected policy-v2 ability state")
    if available[imputed].any():
        raise CompactSupportUnavailable("Neutral-imputed rows cannot become available")
    if (
        not _np.equal(abilities[imputed], _IMPUTED_NEUTRAL_VALUE).all()
        or _np.signbit(abilities[imputed]).any()
    ):
        raise CompactSupportUnavailable("Neutral-imputed ability must be exact positive zero")
    if not available[observed].all():
        raise CompactSupportUnavailable("Observed strict-prior rows must remain available")

    proposal_keys = tuple((value,) for value in support.proposal_order)
    _node_cdf(
        proposal_keys,
        support.frozen_proposal_cdf_upper,
        "Frozen proposal right-CDF",
    )
    surface_keys = tuple(
        (proposal, surface)
        for proposal, surface in zip(
            support.proposal_order, support.surface_order, strict=True
        )
    )
    for proposal in proposals:
        mask = [i for i, value in enumerate(support.proposal_order) if value == proposal]
        _node_cdf(
            tuple(surface_keys[i] for i in mask),
            tuple(support.frozen_surface_cdf_upper[i] for i in mask),
            f"Frozen surface strict-boundary CDF for proposal {proposal}",
            allow_leading_zero=True,
        )
    for proposal, surface in dict.fromkeys(surface_keys):
        mask = [
            i
            for i, key in enumerate(surface_keys)
            if key == (proposal, surface)
        ]
        hazard_keys = tuple(
            (proposal, surface, support.hazard_order[i]) for i in mask
        )
        _node_cdf(
            hazard_keys,
            tuple(support.frozen_hazard_cdf_upper[i] for i in mask),
            f"Frozen hazard right-CDF for proposal {proposal}/surface {surface}",
        )

    for index, is_boundary in enumerate(support.boundary_only):
        if is_boundary:
            if (
                support.surface_order[index] != 0
                or support.frozen_surface_cdf_upper[index] != 0.0
                or support.frozen_hazard_cdf_upper[index] != 1.0
            ):
                raise CompactSupportUnavailable(
                    "Boundary-only atom does not preserve the leading zero surface CDF"
                )

    # The frozen parent measure is independently authoritative, but it must
    # still factor to the same support within the parent's sealing tolerance.
    proposal_upper_by_id: dict[int, float] = {}
    proposal_lower_by_id: dict[int, float] = {}
    prior = 0.0
    for proposal in proposals:
        index = support.proposal_order.index(proposal)
        upper = support.frozen_proposal_cdf_upper[index]
        proposal_lower_by_id[proposal] = prior
        proposal_upper_by_id[proposal] = upper
        prior = upper
    expected: list[float] = []
    for index, (proposal, surface, hazard) in enumerate(order):
        proposal_mass = proposal_upper_by_id[proposal] - proposal_lower_by_id[proposal]
        proposal_rows = [i for i, p in enumerate(support.proposal_order) if p == proposal]
        surface_ids = tuple(dict.fromkeys(support.surface_order[i] for i in proposal_rows))
        surface_position = surface_ids.index(surface)
        surface_upper = support.frozen_surface_cdf_upper[index]
        surface_lower = (
            0.0
            if surface_position == 0
            else support.frozen_surface_cdf_upper[
                next(
                    i
                    for i in proposal_rows
                    if support.surface_order[i] == surface_ids[surface_position - 1]
                )
            ]
        )
        branch_rows = [
            i
            for i in proposal_rows
            if support.surface_order[i] == surface
        ]
        hazard_ids = tuple(dict.fromkeys(support.hazard_order[i] for i in branch_rows))
        hazard_position = hazard_ids.index(hazard)
        hazard_upper = support.frozen_hazard_cdf_upper[index]
        hazard_lower = (
            0.0
            if hazard_position == 0
            else support.frozen_hazard_cdf_upper[
                next(
                    i
                    for i in branch_rows
                    if support.hazard_order[i] == hazard_ids[hazard_position - 1]
                )
            ]
        )
        expected.append(
            proposal_mass
            * (surface_upper - surface_lower)
            * (hazard_upper - hazard_lower)
        )
    if not _np.allclose(base / base.sum(), expected, atol=2e-15, rtol=2e-15):
        raise CompactSupportUnavailable(
            "Frozen proposal/surface/hazard factorization changed"
        )


def _support_payload(support: CompactPhysicalSupport) -> dict[str, object]:
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
                "proposal_order": support.proposal_order[index],
                "surface_order": support.surface_order[index],
                "hazard_order": support.hazard_order[index],
                "frozen_proposal_cdf_upper": _float_hex(
                    support.frozen_proposal_cdf_upper[index]
                ),
                "frozen_surface_cdf_upper": _float_hex(
                    support.frozen_surface_cdf_upper[index]
                ),
                "frozen_hazard_cdf_upper": _float_hex(
                    support.frozen_hazard_cdf_upper[index]
                ),
                "boundary_only": support.boundary_only[index],
                "complete_atom": support.complete_atom[index],
                "terminal": support.terminal[index],
                "stroke_cost": support.stroke_cost[index],
                "next_state_key": support.next_state_key[index],
            }
        )
    return {
        "contract": COMPACT_SUPPORT_CONTRACT,
        "support_key": support.support_key,
        "reward_reference_sha256": support.reward_reference_sha256,
        "rows": records,
    }


def _support_sha256(support: CompactPhysicalSupport) -> str:
    return _stable_sha(_support_payload(support))


def _frozen_cdf_sha256(support: CompactPhysicalSupport) -> str:
    return _stable_sha(
        {
            "contract": ZERO_FAST_PATH_CONTRACT,
            "support_key": support.support_key,
            "atom_id": list(support.atom_id),
            "proposal_order": list(support.proposal_order),
            "surface_order": list(support.surface_order),
            "hazard_order": list(support.hazard_order),
            "boundary_only": list(support.boundary_only),
            "frozen_proposal_cdf_upper": [
                _float_hex(value) for value in support.frozen_proposal_cdf_upper
            ],
            "frozen_surface_cdf_upper": [
                _float_hex(value) for value in support.frozen_surface_cdf_upper
            ],
            "frozen_hazard_cdf_upper": [
                _float_hex(value) for value in support.frozen_hazard_cdf_upper
            ],
        }
    )


@_dataclass(frozen=True)
class WeightingParameters:
    transfer_beta: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    lambda_correction: tuple[float, float, float, float]
    reward_reference_sha256: str
    max_abs_log_weight: float = _MAX_ABS_LOG_WEIGHT


@_dataclass(frozen=True)
class SemanticUniforms:
    proposal: float
    surface: float
    hazard: float


_PLAIN_NUMBER_TYPES = (int, float, _np.float64)


def _plain_finite_number(value: object, label: str) -> float:
    if type(value) not in _PLAIN_NUMBER_TYPES:
        raise CompactSupportUnavailable(f"{label} must be a plain numeric scalar")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise CompactSupportUnavailable(f"{label} must be finite") from exc
    if not _math.isfinite(number):
        raise CompactSupportUnavailable(f"{label} must be finite")
    return number


def _plain_four_vector(value: object, label: str) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise CompactSupportUnavailable(f"{label} must be an immutable four-vector")
    return tuple(
        _plain_finite_number(item, label) for item in value
    )  # type: ignore[return-value]


def _validated_weighting_parameters(
    parameters: object,
) -> tuple[_TransitionQualityParameters, bool]:
    if type(parameters) is not WeightingParameters:
        raise CompactSupportUnavailable("Weighting parameters have the wrong exact type")
    transfer_beta = _plain_four_vector(
        object.__getattribute__(parameters, "transfer_beta"), "transfer_beta"
    )
    target_ability = _plain_four_vector(
        object.__getattribute__(parameters, "target_ability_sg"), "target_ability_sg"
    )
    lambda_correction = _plain_four_vector(
        object.__getattribute__(parameters, "lambda_correction"), "lambda_correction"
    )
    if any(value < 0.0 for value in transfer_beta):
        raise CompactSupportUnavailable("transfer_beta cannot be negative")
    reward_reference = object.__getattribute__(
        parameters, "reward_reference_sha256"
    )
    if not _is_sha256(reward_reference):
        raise CompactSupportUnavailable("Weighting reward reference is invalid")
    log_cap = _plain_finite_number(
        object.__getattribute__(parameters, "max_abs_log_weight"),
        "max_abs_log_weight",
    )
    if log_cap != _MAX_ABS_LOG_WEIGHT:
        raise CompactSupportUnavailable(
            "Weighting log cap differs from the closed synthetic authority"
        )
    try:
        anchor_parameters = _TransitionQualityParameters(
            transfer_beta=transfer_beta,
            target_ability_sg=target_ability,
            lambda_correction=lambda_correction,
            reward_reference_sha256=reward_reference,
            max_abs_log_weight=log_cap,
        )
    except (TypeError, ValueError) as exc:
        raise CompactSupportUnavailable(f"Weighting parameters are invalid: {exc}") from exc
    beta_lambda_zero = all(value == 0.0 for value in transfer_beta) and all(
        value == 0.0 for value in lambda_correction
    )
    return anchor_parameters, beta_lambda_zero


def _validated_semantic_uniforms(uniforms: object) -> tuple[float, float, float]:
    if type(uniforms) is not SemanticUniforms:
        raise CompactSupportUnavailable("Semantic uniforms have the wrong exact type")
    result: list[float] = []
    for label in ("proposal", "surface", "hazard"):
        number = _plain_finite_number(
            object.__getattribute__(uniforms, label), f"{label}_uniform"
        )
        if not 0.0 <= number < 1.0:
            raise CompactSupportUnavailable(
                f"{label}_uniform must satisfy 0 <= u < 1"
            )
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


@_dataclass(frozen=True)
class WeightedSupport:
    support_sha256: str
    parameter_sha256: str
    atom_id: tuple[str, ...]
    probability: tuple[float, ...]
    neutral_probability_mass: tuple[float, ...]
    proposal_probability: tuple[float, ...]
    surface_conditional_probability: tuple[float, ...]
    hazard_conditional_probability: tuple[float, ...]
    proposal_cdf_lower: tuple[float, ...]
    proposal_cdf_upper: tuple[float, ...]
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


@_dataclass(frozen=True)
class SelectionResult:
    atom_index: int
    atom_id: str
    proposal_order: int
    surface_order: int
    hazard_order: int
    terminal: bool
    stroke_cost: int
    next_state_key: str
    boundary_only: bool
    selection_mode: str
    hierarchical_atom_id: str
    flat_atom_id: str
    flat_hierarchical_equal: bool
    semantic_uniforms: SemanticUniforms


@_dataclass(frozen=True)
class CompactSelectorReadiness:
    contract: str
    authority_manifest_sha256: str
    runtime_binding_sha256: str
    support_sha256: str
    frozen_cdf_sha256: str
    rows: int
    proposals: int
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
class _TrustedSyntheticAuthority:
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
    """Read only exact immutable authority fields without dispatched equality."""

    if type(authority) is not _TrustedSyntheticAuthority:
        return None
    observed: list[str] = []
    try:
        for field in _AUTHORITY_FIELDS:
            value = object.__getattribute__(authority, field)
            if not _is_sha256(value):
                return None
            observed.append(value)
    except AttributeError:
        return None
    return tuple(observed)


def _closed_authority_value(
    runtime_binding_sha256: str,
) -> _TrustedSyntheticAuthority | None:
    closed = _closed_authority_envelope()
    manifest = closed.get("manifest")
    manifest_sha = closed.get("manifest_sha256")
    if (
        type(manifest) is not dict
        or not _is_sha256(manifest_sha)
        or _stable_sha(manifest) != manifest_sha
        or type(runtime_binding_sha256) is not str
        or manifest.get("runtime_binding_sha256") != runtime_binding_sha256
    ):
        return None
    return _TrustedSyntheticAuthority(
        manifest_sha256=str(manifest_sha),
        runtime_binding_sha256=runtime_binding_sha256,
        support_sha256=str(manifest.get("support_sha256")),
        frozen_cdf_sha256=str(manifest.get("frozen_cdf_sha256")),
        reward_reference_sha256=str(manifest.get("reward_reference_sha256")),
    )


def _is_closed_authority(
    authority: object, runtime_binding_sha256: str
) -> bool:
    observed = _exact_authority_fields(authority)
    if observed is None or not _is_sha256(runtime_binding_sha256):
        return False
    expected = _closed_authority_value(runtime_binding_sha256)
    expected_fields = _exact_authority_fields(expected)
    if expected_fields is None:
        return False
    return all(
        str.__eq__(left, right) is True
        for left, right in zip(observed, expected_fields, strict=True)
    )


def _trusted_authority(
    envelope: _Mapping[str, object], runtime_binding_sha256: str
) -> _TrustedSyntheticAuthority:
    if type(envelope) is not dict:
        raise CompactSupportUnavailable("Synthetic authority must be an object")
    candidate = _json.loads(_json.dumps(dict(envelope)))
    closed = _closed_authority_envelope()
    manifest = closed["manifest"]
    expected_sha = str(closed["manifest_sha256"])
    if not _is_sha256(expected_sha) or _stable_sha(manifest) != expected_sha:
        raise CompactSupportUnavailable("Closed synthetic authority lock is not minted")
    if candidate != closed:
        raise CompactSupportUnavailable(
            "Candidate is outside the closed source-pinned synthetic authority"
        )
    assert isinstance(manifest, dict)
    if manifest["runtime_binding_sha256"] != runtime_binding_sha256:
        raise CompactSupportUnavailable("Synthetic authority runtime binding changed")
    trusted = _closed_authority_value(runtime_binding_sha256)
    if trusted is None:
        raise CompactSupportUnavailable("Closed synthetic authority is inconsistent")
    return trusted


def _hierarchy_from_probability(
    support: CompactPhysicalSupport, probability: _np.ndarray
) -> dict[str, tuple[float, ...]]:
    rows = len(support.atom_id)
    probability = _np.asarray(probability, dtype=float)
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
        raise CompactSupportUnavailable(
            "Weighting changed positive versus boundary-only atom support"
        )
    proposal_probability = _np.empty(rows, dtype=float)
    surface_conditional = _np.empty(rows, dtype=float)
    hazard_conditional = _np.empty(rows, dtype=float)
    proposal_lower = _np.empty(rows, dtype=float)
    proposal_upper = _np.empty(rows, dtype=float)
    surface_lower = _np.empty(rows, dtype=float)
    surface_upper = _np.empty(rows, dtype=float)
    hazard_lower = _np.empty(rows, dtype=float)
    hazard_upper = _np.empty(rows, dtype=float)

    running_proposal = 0.0
    proposals = tuple(dict.fromkeys(support.proposal_order))
    for proposal_index, proposal in enumerate(proposals):
        p_indices = _np.asarray(
            [i for i, value in enumerate(support.proposal_order) if value == proposal],
            dtype=int,
        )
        p_mass = float(probability[p_indices].sum())
        if p_mass <= 0.0:
            raise CompactSupportUnavailable("Weighting collapsed a proposal")
        p_upper = (
            1.0
            if proposal_index == len(proposals) - 1
            else running_proposal + p_mass
        )
        if not p_upper > running_proposal:
            raise CompactSupportUnavailable(
                "Positive proposal mass collapsed to a zero-width CDF interval"
            )
        p_width = p_upper - running_proposal
        proposal_probability[p_indices] = p_width
        proposal_lower[p_indices] = running_proposal
        proposal_upper[p_indices] = p_upper
        running_proposal = p_upper

        running_surface = 0.0
        surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
        for surface_index, surface in enumerate(surfaces):
            s_indices = _np.asarray(
                [i for i in p_indices if support.surface_order[i] == surface],
                dtype=int,
            )
            s_mass = float(probability[s_indices].sum())
            boundary_values = boundary[s_indices]
            if bool(boundary_values.any()):
                if (
                    not bool(boundary_values.all())
                    or s_mass != 0.0
                    or surface_index != 0
                    or running_surface != 0.0
                    or len(s_indices) != 1
                ):
                    raise CompactSupportUnavailable(
                        "Weighted boundary-only surface semantics changed"
                    )
                surface_conditional[s_indices] = 0.0
                surface_lower[s_indices] = 0.0
                surface_upper[s_indices] = 0.0
                hazard_conditional[s_indices] = 1.0
                hazard_lower[s_indices] = 0.0
                hazard_upper[s_indices] = 1.0
                continue
            if s_mass <= 0.0:
                raise CompactSupportUnavailable("Weighting collapsed a positive surface")
            s_conditional = s_mass / p_mass
            s_upper = (
                1.0
                if surface_index == len(surfaces) - 1
                else running_surface + s_conditional
            )
            if not s_upper > running_surface:
                raise CompactSupportUnavailable(
                    "Positive surface mass collapsed to a zero-width CDF interval"
                )
            s_width = s_upper - running_surface
            surface_conditional[s_indices] = s_width
            surface_lower[s_indices] = running_surface
            surface_upper[s_indices] = s_upper
            running_surface = s_upper

            running_hazard = 0.0
            hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
            for hazard_index, hazard in enumerate(hazards):
                h_indices = _np.asarray(
                    [i for i in s_indices if support.hazard_order[i] == hazard],
                    dtype=int,
                )
                if len(h_indices) != 1:
                    raise CompactSupportUnavailable(
                        "One proposal/surface/hazard leaf must name one whole atom"
                    )
                h_mass = float(probability[h_indices].sum())
                if h_mass <= 0.0:
                    raise CompactSupportUnavailable(
                        "Weighting collapsed a positive hazard atom"
                    )
                h_conditional = h_mass / s_mass
                h_upper = (
                    1.0
                    if hazard_index == len(hazards) - 1
                    else running_hazard + h_conditional
                )
                if not h_upper > running_hazard:
                    raise CompactSupportUnavailable(
                        "Positive hazard mass collapsed to a zero-width CDF interval"
                    )
                h_width = h_upper - running_hazard
                hazard_conditional[h_indices] = h_width
                hazard_lower[h_indices] = running_hazard
                hazard_upper[h_indices] = h_upper
                running_hazard = h_upper

    proposal_width = proposal_upper - proposal_lower
    surface_width = surface_upper - surface_lower
    hazard_width = hazard_upper - hazard_lower
    cdf_factorized = proposal_width * surface_width * hazard_width
    if not _np.allclose(cdf_factorized, probability, atol=0.0, rtol=2e-12):
        raise CompactSupportUnavailable(
            "Actual hierarchical CDF interval widths do not factor the final "
            "whole-atom probability"
        )
    return {
        "proposal_probability": tuple(map(float, proposal_probability)),
        "surface_conditional_probability": tuple(map(float, surface_conditional)),
        "hazard_conditional_probability": tuple(map(float, hazard_conditional)),
        "proposal_cdf_lower": tuple(map(float, proposal_lower)),
        "proposal_cdf_upper": tuple(map(float, proposal_upper)),
        "surface_cdf_lower": tuple(map(float, surface_lower)),
        "surface_cdf_upper": tuple(map(float, surface_upper)),
        "hazard_cdf_lower": tuple(map(float, hazard_lower)),
        "hazard_cdf_upper": tuple(map(float, hazard_upper)),
    }


def _frozen_hierarchy(support: CompactPhysicalSupport) -> dict[str, tuple[float, ...]]:
    rows = len(support.atom_id)
    proposal_lower = _np.empty(rows, dtype=float)
    surface_lower = _np.empty(rows, dtype=float)
    hazard_lower = _np.empty(rows, dtype=float)
    proposal_probability = _np.empty(rows, dtype=float)
    surface_conditional = _np.empty(rows, dtype=float)
    hazard_conditional = _np.empty(rows, dtype=float)
    prior_proposal = 0.0
    for proposal in dict.fromkeys(support.proposal_order):
        p_indices = [i for i, value in enumerate(support.proposal_order) if value == proposal]
        p_upper = support.frozen_proposal_cdf_upper[p_indices[0]]
        for index in p_indices:
            proposal_lower[index] = prior_proposal
            proposal_probability[index] = p_upper - prior_proposal
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
        "proposal_probability": tuple(map(float, proposal_probability)),
        "surface_conditional_probability": tuple(map(float, surface_conditional)),
        "hazard_conditional_probability": tuple(map(float, hazard_conditional)),
        "proposal_cdf_lower": tuple(map(float, proposal_lower)),
        "proposal_cdf_upper": tuple(map(float, support.frozen_proposal_cdf_upper)),
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
    return min(int(_np.count_nonzero(uniform > _np.asarray(upper, dtype=float))), len(upper) - 1)


def _hierarchical_select(
    support: CompactPhysicalSupport,
    hierarchy: _Mapping[str, tuple[float, ...]],
    uniforms: tuple[float, float, float],
    *,
    allow_boundary_only: bool,
) -> int:
    proposal_u, surface_u, hazard_u = uniforms
    proposals = tuple(dict.fromkeys(support.proposal_order))
    proposal_upper = tuple(
        hierarchy["proposal_cdf_upper"][support.proposal_order.index(proposal)]
        for proposal in proposals
    )
    proposal = proposals[_select_right(proposal_upper, proposal_u)]
    p_indices = [i for i, value in enumerate(support.proposal_order) if value == proposal]
    surfaces = tuple(dict.fromkeys(support.surface_order[i] for i in p_indices))
    if not allow_boundary_only:
        reachable: list[int] = []
        for surface in surfaces:
            node_index = next(
                i for i in p_indices if support.surface_order[i] == surface
            )
            if (
                hierarchy["surface_cdf_upper"][node_index]
                > hierarchy["surface_cdf_lower"][node_index]
            ):
                reachable.append(surface)
        surfaces = tuple(reachable)
    if not surfaces:
        raise CompactSupportUnavailable("No reachable positive surface remains")
    surface_upper = tuple(
        hierarchy["surface_cdf_upper"][
            next(i for i in p_indices if support.surface_order[i] == surface)
        ]
        for surface in surfaces
    )
    surface = surfaces[_select_surface_strict(surface_upper, surface_u)]
    s_indices = [i for i in p_indices if support.surface_order[i] == surface]
    hazards = tuple(dict.fromkeys(support.hazard_order[i] for i in s_indices))
    hazard_upper = tuple(
        hierarchy["hazard_cdf_upper"][
            next(i for i in s_indices if support.hazard_order[i] == hazard)
        ]
        for hazard in hazards
    )
    hazard = hazards[_select_right(hazard_upper, hazard_u)]
    matches = [i for i in s_indices if support.hazard_order[i] == hazard]
    if len(matches) != 1:
        raise CompactSupportUnavailable("Hierarchical leaf is not one complete atom")
    return matches[0]


def _flat_semantic_select(
    support: CompactPhysicalSupport,
    hierarchy: _Mapping[str, tuple[float, ...]],
    uniforms: tuple[float, float, float],
    *,
    allow_boundary_only: bool,
) -> int:
    proposal_u, surface_u, hazard_u = uniforms
    matches: list[int] = []
    for index in range(len(support.atom_id)):
        proposal_match = (
            hierarchy["proposal_cdf_lower"][index]
            <= proposal_u
            < hierarchy["proposal_cdf_upper"][index]
        )
        surface_lower = hierarchy["surface_cdf_lower"][index]
        surface_upper = hierarchy["surface_cdf_upper"][index]
        if support.boundary_only[index]:
            surface_match = bool(
                allow_boundary_only
                and surface_u == 0.0
                and surface_lower == 0.0
                and surface_upper == 0.0
            )
        else:
            surface_match = bool(
                surface_upper > surface_lower
                and (
                    (
                        (
                            support.surface_order[index] == 0
                            or not allow_boundary_only
                        )
                        and surface_lower == 0.0
                        and 0.0 <= surface_u
                    )
                    or surface_u > surface_lower
                )
                and surface_u <= surface_upper
            )
        hazard_match = (
            hierarchy["hazard_cdf_lower"][index]
            <= hazard_u
            < hierarchy["hazard_cdf_upper"][index]
        )
        if proposal_match and surface_match and hazard_match:
            matches.append(index)
    if len(matches) != 1:
        raise CompactSupportUnavailable(
            f"Retained semantic uniforms select {len(matches)} flattened atoms"
        )
    return matches[0]


class CompactWeightingSelectorSession:
    """Opaque validated capability for synthetic weighting and selection only."""

    __slots__ = ("_support", "_authority", "_readiness")

    def __init__(self, *_: object, **__: object) -> None:
        raise CompactSupportUnavailable(
            "Sessions are minted only by admit_synthetic_support"
        )

    @property
    def readiness(self) -> CompactSelectorReadiness:
        if type(self) is not CompactWeightingSelectorSession:
            raise CompactSupportUnavailable("Synthetic session has the wrong exact type")
        return object.__getattribute__(self, "_readiness")

    def support_copy(self) -> CompactPhysicalSupport:
        if type(self) is not CompactWeightingSelectorSession:
            raise CompactSupportUnavailable("Synthetic session has the wrong exact type")
        support = CompactWeightingSelectorSession._live_support(self)
        return CompactPhysicalSupport(
            **{
                field.name: getattr(support, field.name)
                for field in _dataclass_fields(CompactPhysicalSupport)
            }
        )

    def _live_support(self) -> CompactPhysicalSupport:
        if type(self) is not CompactWeightingSelectorSession:
            raise CompactSupportUnavailable("Synthetic session has the wrong exact type")
        runtime = _validate_runtime()
        authority = object.__getattribute__(self, "_authority")
        if not _is_closed_authority(authority, runtime):
            raise CompactSupportUnavailable("Synthetic session authority changed")
        canonical = _canonicalize_support(object.__getattribute__(self, "_support"))
        if (
            _support_sha256(canonical) != authority.support_sha256
            or _frozen_cdf_sha256(canonical) != authority.frozen_cdf_sha256
        ):
            raise CompactSupportUnavailable("Synthetic session support changed")
        return canonical

    def weight(self, parameters: WeightingParameters) -> WeightedSupport:
        if type(self) is not CompactWeightingSelectorSession:
            raise CompactSupportUnavailable("Synthetic session has the wrong exact type")
        anchor_parameters, beta_lambda_zero = _validated_weighting_parameters(parameters)
        support = CompactWeightingSelectorSession._live_support(self)
        authority = object.__getattribute__(self, "_authority")
        if anchor_parameters.reward_reference_sha256 != authority.reward_reference_sha256:
            raise CompactSupportUnavailable("Weighting reward reference changed")
        abilities = _np.asarray(support.donor_ability_sg, dtype=float)
        imputed = _np.asarray(support.donor_ability_value_kind, dtype=str) == _IMPUTED_VALUE_KIND
        available = _np.asarray(support.donor_ability_available, dtype=bool)
        if available[imputed].any() or _np.signbit(abilities[imputed]).any():
            raise CompactSupportUnavailable("Imputed availability or positive zero changed")
        try:
            atom_mass = _whole_atom_probabilities(
                base_mass=support.base_mass,
                fixed_reference_reward=support.fixed_reference_reward,
                donor_ability_sg=tuple(map(float, abilities)),
                donor_group_id=support.original_donor_group_id,
                category=support.category[0],
                parameters=anchor_parameters,
            )
        except (RuntimeError, ValueError, AssertionError) as exc:
            raise CompactSupportUnavailable(
                f"Frozen whole-atom weighting rejected the support: {exc}"
            ) from exc
        probability = _np.asarray(atom_mass.probability, dtype=float)
        if beta_lambda_zero:
            hierarchy = _frozen_hierarchy(support)
            selection_mode = ZERO_FAST_PATH_CONTRACT
            factorization_from_final = False
        else:
            hierarchy = _hierarchy_from_probability(support, probability)
            selection_mode = "weighted_rosenblatt_from_final_flat_probability_v1"
            factorization_from_final = True
        return WeightedSupport(
            support_sha256=self._authority.support_sha256,
            parameter_sha256=anchor_parameters.parameter_sha256,
            atom_id=tuple(support.atom_id),
            probability=tuple(map(float, probability)),
            neutral_probability_mass=tuple(
                map(float, atom_mass.neutral_probability_mass)
            ),
            donor_group_mass_before=tuple(
                map(float, atom_mass.donor_group_mass_before)
            ),
            donor_group_mass_after=tuple(
                map(float, atom_mass.donor_group_mass_after)
            ),
            clipped_donor_logs=int(atom_mass.clipped_donor_logs),
            clipped_target_logs=int(atom_mass.clipped_target_logs),
            selection_mode=selection_mode,
            factorization_from_final_probability=factorization_from_final,
            **hierarchy,
        )

    def select(
        self,
        parameters: WeightingParameters,
        uniforms: SemanticUniforms,
    ) -> SelectionResult:
        if type(self) is not CompactWeightingSelectorSession:
            raise CompactSupportUnavailable("Synthetic session has the wrong exact type")
        retained = _validated_semantic_uniforms(uniforms)
        support = CompactWeightingSelectorSession._live_support(self)
        weighted = CompactWeightingSelectorSession.weight(self, parameters)
        hierarchy = {
            name: getattr(weighted, name)
            for name in (
                "proposal_cdf_lower",
                "proposal_cdf_upper",
                "surface_cdf_lower",
                "surface_cdf_upper",
                "hazard_cdf_lower",
                "hazard_cdf_upper",
            )
        }
        allow_boundary_only = weighted.selection_mode == ZERO_FAST_PATH_CONTRACT
        hierarchical_index = _hierarchical_select(
            support,
            hierarchy,
            retained,
            allow_boundary_only=allow_boundary_only,
        )
        flat_index = _flat_semantic_select(
            support,
            hierarchy,
            retained,
            allow_boundary_only=allow_boundary_only,
        )
        if hierarchical_index != flat_index:
            raise CompactSupportUnavailable(
                "Flat and hierarchical retained-uniform selections diverged"
            )
        index = hierarchical_index
        return SelectionResult(
            atom_index=index,
            atom_id=support.atom_id[index],
            proposal_order=support.proposal_order[index],
            surface_order=support.surface_order[index],
            hazard_order=support.hazard_order[index],
            terminal=support.terminal[index],
            stroke_cost=support.stroke_cost[index],
            next_state_key=support.next_state_key[index],
            boundary_only=support.boundary_only[index],
            selection_mode=weighted.selection_mode,
            hierarchical_atom_id=support.atom_id[hierarchical_index],
            flat_atom_id=support.atom_id[flat_index],
            flat_hierarchical_equal=True,
            semantic_uniforms=SemanticUniforms(*retained),
        )


def _mint_session(
    support: CompactPhysicalSupport,
    authority: _TrustedSyntheticAuthority,
    readiness: CompactSelectorReadiness,
) -> CompactWeightingSelectorSession:
    session = object.__new__(CompactWeightingSelectorSession)
    session._support = support
    session._authority = authority
    session._readiness = readiness
    return session


def admit_synthetic_support(
    support: CompactPhysicalSupport,
    *,
    authority_manifest: _Mapping[str, object],
) -> CompactWeightingSelectorSession:
    """Admit only the exact closed synthetic fixture; real supports fail closed."""

    runtime = _validate_runtime()
    authority = _trusted_authority(authority_manifest, runtime)
    canonical = _canonicalize_support(support)
    observed_support_sha = _support_sha256(canonical)
    observed_cdf_sha = _frozen_cdf_sha256(canonical)
    if observed_support_sha != authority.support_sha256:
        raise CompactSupportUnavailable(
            "Candidate support is outside the closed synthetic authority"
        )
    if observed_cdf_sha != authority.frozen_cdf_sha256:
        raise CompactSupportUnavailable(
            "Candidate frozen CDF is outside the closed synthetic authority"
        )
    if canonical.reward_reference_sha256 != authority.reward_reference_sha256:
        raise CompactSupportUnavailable("Synthetic reward reference changed")
    kinds = _np.asarray(canonical.donor_ability_value_kind, dtype=str)
    availability = _np.asarray(canonical.donor_ability_available, dtype=bool)
    abilities = _np.asarray(canonical.donor_ability_sg, dtype=float)
    mixed_groups = 0
    for group_id in dict.fromkeys(canonical.original_donor_group_id):
        states = {
            canonical.donor_ability_value_kind[index]
            for index, value in enumerate(canonical.original_donor_group_id)
            if value == group_id
        }
        mixed_groups += int(len(states) > 1)
    readiness = CompactSelectorReadiness(
        contract=WEIGHTING_SELECTOR_CONTRACT,
        authority_manifest_sha256=authority.manifest_sha256,
        runtime_binding_sha256=runtime,
        support_sha256=observed_support_sha,
        frozen_cdf_sha256=observed_cdf_sha,
        rows=len(canonical.atom_id),
        proposals=len(set(canonical.proposal_order)),
        original_donor_groups=len(set(canonical.original_donor_group_id)),
        mixed_policy_state_groups=mixed_groups,
        observed_rows=int(_np.count_nonzero(kinds == _OBSERVED_VALUE_KIND)),
        neutral_imputed_rows=int(_np.count_nonzero(kinds == _IMPUTED_VALUE_KIND)),
        availability_preserved=bool(not availability[kinds == _IMPUTED_VALUE_KIND].any()),
        positive_zero_preserved=bool(
            _np.equal(abilities[kinds == _IMPUTED_VALUE_KIND], 0.0).all()
            and not _np.signbit(abilities[kinds == _IMPUTED_VALUE_KIND]).any()
        ),
        synthetic_fixture_only=True,
        weighting_ready=True,
        selector_ready=True,
        real_chain_ready=False,
        integration_ready=False,
        blockers=(
            "no_externally_reviewed_real_support_authority",
            "app_requires_separate_holeout_uniform_and_terminal_residual_replay",
            "no_compact_support_to_physical_chain_callback_integration",
            "no_mean_solve_or_simulation_integration",
        ),
    )
    return _mint_session(canonical, authority, readiness)


__all__ = [
    "COMPACT_SUPPORT_CONTRACT",
    "SYNTHETIC_AUTHORITY_CONTRACT",
    "WEIGHTING_SELECTOR_CONTRACT",
    "ZERO_FAST_PATH_CONTRACT",
    "SELECTION_BOUNDARY_CONTRACT",
    "INTEGRATION_STATUS",
    "CompactSupportUnavailable",
    "CompactPhysicalSupport",
    "WeightingParameters",
    "SemanticUniforms",
    "WeightedSupport",
    "SelectionResult",
    "CompactSelectorReadiness",
    "CompactWeightingSelectorSession",
    "synthetic_authority_manifest",
    "admit_synthetic_support",
]

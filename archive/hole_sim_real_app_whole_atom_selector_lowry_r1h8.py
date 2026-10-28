"""Logic-only real-shaped APP whole-atom weighting and selection seam.

This phase deliberately admits no real support.  It freezes and tests the
semantics needed by the one reviewed Lowry R1 H8 APP complete-atom support:

* one whole-atom weight over the entire conditional support;
* donor de-tilting by the original event/category/start-state group;
* APP holeout, proposal, surface, and hazard uniforms with their original
  boundary rules; and
* exact binary64 projections of tilted conditional mass ratios.

The public admission function has no path or candidate arguments and fails
before runtime source validation while the authority registry is empty.  The
private fixture helpers exist only for focused logic tests.  This module does
not read a materialized support, run a chain, expose a callback, or launch a
mean solve or simulation.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from fractions import Fraction as _Fraction
from hashlib import sha256 as _sha256
import json as _json
import math as _math
from pathlib import Path as _Path
import sys as _sys
from typing import Mapping as _Mapping, NoReturn as _NoReturn, Sequence as _Sequence

import numpy as _np
import pandas as _pd


_ARCHIVE = _Path(__file__).resolve().parent
if str(_ARCHIVE) not in _sys.path:
    _sys.path.insert(0, str(_ARCHIVE))

import hole_sim_compact_app_whole_atom_selector_v1 as _synthetic  # noqa: E402
import hole_sim_donor_ability_policy as _policy  # noqa: E402
import hole_sim_transition_mean_anchor as _anchor  # noqa: E402
import hole_sim_transition_mean_chain_adapter as _chain_v3  # noqa: E402

# ``from __future__ import annotations`` installs the feature name in this
# interpreter.  Keep the logic-only module's public namespace exactly equal to
# ``__all__`` rather than leaking that implementation detail.
globals().pop("annotations", None)


REAL_APP_SELECTOR_CONTRACT = "real_shaped_whole_atom_app_selector_logic_v1"
REAL_APP_AUTHORITY_CONTRACT = "closed_real_app_selector_authority_registry_v1"
REAL_APP_ZERO_PATH_CONTRACT = "frozen_real_app_four_uniform_zero_path_v1"
REAL_APP_TILTED_PATH_CONTRACT = (
    "fraction_projected_real_app_four_uniform_tilted_path_v1"
)
REAL_APP_BOUNDARY_CONTRACT = (
    "holeout_strict_proposal_right_surface_strict_hazard_right_binary64_v1"
)
INTEGRATION_STATUS = "logic_only_empty_authority_registry_hold"

_SYNTHETIC_DESCRIPTOR = "archive/hole_sim_compact_app_whole_atom_selector_v1.py"
_ANCHOR_DESCRIPTOR = "archive/hole_sim_transition_mean_anchor.py"
_POLICY_DESCRIPTOR = "archive/hole_sim_donor_ability_policy.py"
_COMPLETE_DESCRIPTOR = "archive/hole_sim_tee_app_complete_atom_support.py"
_RUNNER_DESCRIPTOR = (
    "archive/hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner.py"
)
_CHAIN_V3_DESCRIPTOR = "archive/hole_sim_transition_mean_chain_adapter.py"

_SOURCE_BINDINGS = (
    (
        "reviewed_synthetic_app_selector",
        _SYNTHETIC_DESCRIPTOR,
        "e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7",
    ),
    (
        "whole_atom_weighting_primitive",
        _ANCHOR_DESCRIPTOR,
        "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28",
    ),
    (
        "frozen_policy_v2",
        _POLICY_DESCRIPTOR,
        "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339",
    ),
    (
        "complete_atom_semantic_reference",
        _COMPLETE_DESCRIPTOR,
        "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498",
    ),
    (
        "fixed_h8_materializer_reference",
        _RUNNER_DESCRIPTOR,
        "ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460",
    ),
    (
        "canonical_group_derivation_v3",
        _CHAIN_V3_DESCRIPTOR,
        "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    ),
)

_MAX_ATOMS = 35_218
_MAX_PROPOSALS = 904
_MAX_SURFACE_CLASSES = 9
_MAX_HAZARDS_PER_SURFACE = 34
_MAX_ABS_LOG_WEIGHT = 30.0
_MASS_TOLERANCE = 3e-13
_APP_SURFACE_LABELS = (
    "green",
    "fairway",
    "rough",
    "sand",
    "recovery",
    "tee",
    "hole",
    "penalty",
    "other",
)
_FIXED_ZERO_SURFACE_SLOTS = (5, 6, 8)

_AUTHORITY_RECORDS: tuple[dict[str, object], ...] = ()
# Stable SHA-256 of {"contract": REAL_APP_AUTHORITY_CONTRACT, "records": []}.
_AUTHORITY_LOCK_SHA256 = (
    "2c538eb3cbc98314749833c1ad8888b0c00aa8be8ff66776e4f9d3ae3d937e7a"
)

_POLICY_COLUMNS = (*_policy.INPUT_COLUMNS, *_policy.OUTPUT_COLUMNS)
_INTEGRAL_SCALAR_COLUMNS = (
    "complete_row_order",
    "proposal_row_order",
    "surface_class_count",
    "surface_class_index",
    "hazard_donor_order",
    "stroke_cost",
)
_BOOLEAN_SCALAR_COLUMNS = (
    "complete_atom",
    "boundary_only",
    "terminal",
    "donor_ability_available",
    "donor_ability_was_clipped",
)
_NUMERIC_SCALAR_COLUMNS = (
    "proposal_probability",
    "mechanics_conditional_probability",
    "quality_base_mass",
    "fixed_reference_reward",
    "donor_ability_sg",
    "declared_app_base_holeout_probability",
    "surface_probability",
    "surface_cdf_lower",
    "surface_cdf_upper",
    "hazard_probability",
    "hazard_cdf_lower",
    "hazard_cdf_upper",
)
_NULLABLE_NUMERIC_SCALAR_COLUMNS = (
    "donor_ability_raw_strict_prior",
    "frozen_spatial_cdf_lower",
    "frozen_spatial_cdf_upper",
    "frozen_terminal_residual_cdf_lower",
    "frozen_terminal_residual_cdf_upper",
)
_STRING_SCALAR_COLUMNS = (
    "query_key",
    "atom_id",
    "category",
    "proposal_branch",
    "donor_quality_group_id",
    "start_state_key",
    "surface_full_probability_json",
    "surface_full_cdf_upper_json",
    "surface_class_label",
    "hazard_resolution_status",
    "hazard_donor_key",
    "end_state_key",
    "next_support_key",
    "reference_model_sha256",
    "reference_putting_sha256",
    "bridge_row_id",
    "donor_player_id",
    "donor_event_key",
    "target_event_key",
    "donor_ability_source_column",
    "donor_ability_value_kind",
    "donor_ability_policy_sha256",
    "donor_ability_row_sha256",
)
_REQUIRED_TEMPORAL_SCALAR_COLUMNS = (
    "donor_event_start",
    "target_event_start",
)
_NULLABLE_TEMPORAL_SCALAR_COLUMNS = ("donor_skill_date",)
_REQUIRED_COLUMNS = (
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
    *_POLICY_COLUMNS,
)


class RealAppSelectorUnavailable(ValueError):
    """Raised whenever this logic-only seam is not exactly admissible."""


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


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


_PLAIN_NUMBER_TYPES = (int, float, _np.float64)
_EXACT_BOOLEAN_TYPES = (bool, _np.bool_)
_EXACT_INTEGER_TYPES = tuple(
    dict.fromkeys(
        (
            int,
            _np.int8,
            _np.int16,
            _np.int32,
            _np.int64,
            _np.uint8,
            _np.uint16,
            _np.uint32,
            _np.uint64,
            _np.intp,
            _np.uintp,
            _np.longlong,
            _np.ulonglong,
        )
    )
)
_EXACT_FLOAT_TYPES = tuple(
    dict.fromkeys((float, _np.float16, _np.float32, _np.float64, _np.longdouble))
)
_EXACT_NUMERIC_TYPES = (*_EXACT_INTEGER_TYPES, *_EXACT_FLOAT_TYPES)
_EXACT_STRING_TYPES = (str, _np.str_)
_EXACT_TIMESTAMP_TYPES = (_pd.Timestamp, _np.datetime64)
_EXACT_NAT_TYPE = type(_pd.NaT)


def _plain_finite(value: object, label: str) -> float:
    if type(value) not in _PLAIN_NUMBER_TYPES:
        raise RealAppSelectorUnavailable(f"{label} must be a plain numeric scalar")
    number = float(value)
    if not _math.isfinite(number):
        raise RealAppSelectorUnavailable(f"{label} must be finite")
    return number


def _authority_payload() -> dict[str, object]:
    return {
        "contract": REAL_APP_AUTHORITY_CONTRACT,
        "records": list(_AUTHORITY_RECORDS),
    }


def _validate_authority_registry() -> None:
    if type(_AUTHORITY_RECORDS) is not tuple:
        raise RealAppSelectorUnavailable("Real APP authority registry type changed")
    if _stable_sha(_authority_payload()) != _AUTHORITY_LOCK_SHA256:
        raise RealAppSelectorUnavailable("Real APP authority registry lock changed")
    if _AUTHORITY_RECORDS:
        raise RealAppSelectorUnavailable(
            "Logic-only phase cannot contain an admitted real APP authority"
        )


def real_app_authority_manifest() -> dict[str, object]:
    """Report the closed empty registry without widening its authority."""

    _validate_authority_registry()
    return {
        "contract": REAL_APP_AUTHORITY_CONTRACT,
        "authority_lock_sha256": _AUTHORITY_LOCK_SHA256,
        "records": [],
        "research_only": True,
        "production_enabled": False,
        "real_support_admitted": False,
        "weighting_enabled": False,
        "selection_enabled": False,
        "chain_ready": False,
        "anchor_ready": False,
        "integration_status": INTEGRATION_STATUS,
    }


def admit_lowry_r1h8_app_support() -> _NoReturn:
    """Fail before any source or artifact read while the registry is empty."""

    _validate_authority_registry()
    raise RealAppSelectorUnavailable(
        "No real Lowry R1 H8 APP selector authority has been admitted"
    )


def _runtime_payload() -> dict[str, object]:
    return {
        "selector_contract": REAL_APP_SELECTOR_CONTRACT,
        "zero_path_contract": REAL_APP_ZERO_PATH_CONTRACT,
        "tilted_path_contract": REAL_APP_TILTED_PATH_CONTRACT,
        "boundary_contract": REAL_APP_BOUNDARY_CONTRACT,
        "atom_weight_contract": _anchor.ATOM_WEIGHT_CONTRACT,
        "policy_contract": _policy.POLICY_CONTRACT,
        "policy_identity_sha256": _policy.policy_payload()[
            "policy_identity_sha256"
        ],
        "semantic_uniforms": ["holeout", "proposal", "surface", "hazard"],
        "app_surface_labels": list(_APP_SURFACE_LABELS),
        "fixed_positive_zero_surface_slots": list(_FIXED_ZERO_SURFACE_SLOTS),
        "source_bindings": [
            {"role": role, "descriptor": descriptor, "source_sha256": digest}
            for role, descriptor, digest in _SOURCE_BINDINGS
        ],
        "limits": {
            "max_atoms": _MAX_ATOMS,
            "max_proposals": _MAX_PROPOSALS,
            "max_surface_classes": _MAX_SURFACE_CLASSES,
            "max_hazards_per_surface": _MAX_HAZARDS_PER_SURFACE,
            "max_abs_log_weight": _MAX_ABS_LOG_WEIGHT,
        },
    }


def _validate_runtime() -> str:
    if _anchor.ATOM_WEIGHT_CONTRACT != (
        "whole_atom_donor_detilt_target_reapply_lambda_v1"
    ):
        raise RealAppSelectorUnavailable("Whole-atom weighting contract changed")
    if _policy.POLICY_CONTRACT != "frozen_strict_prior_donor_ability_policy_v2":
        raise RealAppSelectorUnavailable("Donor-ability policy contract changed")
    if _synthetic.APP_SELECTION_BOUNDARY_CONTRACT != (
        "holeout_strict_terminal_unconditional_prefix_proposal_right_surface_"
        "strict_hazard_right_binary64_pathwise_v2"
    ):
        raise RealAppSelectorUnavailable("Reviewed APP boundary reference changed")
    if _chain_v3.canonical_donor_quality_group_id.__name__ != (
        "canonical_donor_quality_group_id"
    ):
        raise RealAppSelectorUnavailable("Canonical group derivation changed")
    observed = {
        descriptor: _sha256_file(_ARCHIVE / _Path(descriptor).name)
        for _, descriptor, _ in _SOURCE_BINDINGS
    }
    expected = {
        descriptor: digest for _, descriptor, digest in _SOURCE_BINDINGS
    }
    if observed != expected:
        changed = sorted(
            descriptor
            for descriptor, digest in expected.items()
            if observed.get(descriptor) != digest
        )
        raise RealAppSelectorUnavailable(f"Frozen real APP sources changed: {changed}")
    return _stable_sha(_runtime_payload())


def _finite_vector(values: object, label: str) -> _np.ndarray:
    try:
        vector = _np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RealAppSelectorUnavailable(f"{label} must be numeric") from exc
    if vector.ndim != 1 or not _np.isfinite(vector).all():
        raise RealAppSelectorUnavailable(f"{label} must be a finite vector")
    return vector


def _json_vector(value: object, label: str) -> _np.ndarray:
    if type(value) is not str:
        raise RealAppSelectorUnavailable(f"{label} must be a plain JSON string")
    try:
        decoded = _json.loads(value)
    except (TypeError, ValueError, _json.JSONDecodeError) as exc:
        raise RealAppSelectorUnavailable(f"{label} is invalid JSON") from exc
    if type(decoded) is not list or any(
        type(item) not in (int, float)
        or not _math.isfinite(float(item))
        for item in decoded
    ):
        raise RealAppSelectorUnavailable(
            f"{label} must contain only genuine finite JSON numbers"
        )
    return _finite_vector(decoded, label)


def _single(values: _pd.Series, label: str) -> object:
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise RealAppSelectorUnavailable(f"{label} varies within one node")
    return unique[0]


def _require_close(left: float, right: float, label: str) -> None:
    if not _math.isclose(
        float(left), float(right), rel_tol=2e-15, abs_tol=2e-15
    ):
        raise RealAppSelectorUnavailable(f"{label} changed")


def _require_mass_close(left: float, right: float, label: str) -> None:
    if not _math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=_MASS_TOLERANCE,
    ):
        raise RealAppSelectorUnavailable(f"{label} changed")


def _validate_scalar_families(frame: _pd.DataFrame) -> None:
    """Reject coercible impostors before any ``astype``/numeric conversion."""

    for column in _INTEGRAL_SCALAR_COLUMNS:
        for value in frame[column].array:
            if type(value) not in _EXACT_INTEGER_TYPES:
                raise RealAppSelectorUnavailable(
                    f"{column} must contain exact non-boolean integral scalars"
                )
    for column in _BOOLEAN_SCALAR_COLUMNS:
        for value in frame[column].array:
            if type(value) not in _EXACT_BOOLEAN_TYPES:
                raise RealAppSelectorUnavailable(
                    f"{column} must contain exact boolean scalars"
                )
    for column in _NUMERIC_SCALAR_COLUMNS:
        for value in frame[column].array:
            if (
                type(value) not in _EXACT_NUMERIC_TYPES
                or not _math.isfinite(float(value))
            ):
                raise RealAppSelectorUnavailable(
                    f"{column} must contain genuine finite numeric scalars"
                )
    for column in _NULLABLE_NUMERIC_SCALAR_COLUMNS:
        for value in frame[column].array:
            if value is None or value is _pd.NA:
                continue
            if type(value) not in _EXACT_NUMERIC_TYPES:
                raise RealAppSelectorUnavailable(
                    f"{column} must contain genuine numeric or missing scalars"
                )
            number = float(value)
            if not (_math.isfinite(number) or _math.isnan(number)):
                raise RealAppSelectorUnavailable(
                    f"{column} cannot contain infinite scalars"
                )
    for column in _STRING_SCALAR_COLUMNS:
        for value in frame[column].array:
            if type(value) not in _EXACT_STRING_TYPES:
                raise RealAppSelectorUnavailable(
                    f"{column} must contain genuine string scalars"
                )


def _exact_utc_timestamp(
    value: object, *, label: str, nullable: bool
) -> object:
    value_type = type(value)
    missing = value_type is _EXACT_NAT_TYPE or (
        value_type is _np.datetime64 and _np.isnat(value)
    )
    if missing:
        if nullable:
            return _pd.NaT
        raise RealAppSelectorUnavailable(f"{label} requires a real timestamp")
    if value_type not in _EXACT_TIMESTAMP_TYPES:
        suffix = " or exact NaT" if nullable else ""
        raise RealAppSelectorUnavailable(
            f"{label} must contain exact pandas/NumPy timestamps{suffix}"
        )
    try:
        timestamp = _pd.Timestamp(value)
        if value_type is _np.datetime64:
            timestamp = timestamp.tz_localize("UTC")
        elif timestamp.tzinfo is None:
            raise RealAppSelectorUnavailable(
                f"{label} pandas timestamps must be timezone-aware"
            )
        else:
            timestamp = timestamp.tz_convert("UTC")
    except RealAppSelectorUnavailable:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise RealAppSelectorUnavailable(f"{label} timestamp is invalid") from exc
    if _pd.isna(timestamp):
        if nullable:
            return _pd.NaT
        raise RealAppSelectorUnavailable(f"{label} requires a real timestamp")
    return timestamp


def _normalize_temporal_columns(frame: _pd.DataFrame) -> None:
    for column in _REQUIRED_TEMPORAL_SCALAR_COLUMNS:
        values = [
            _exact_utc_timestamp(value, label=column, nullable=False)
            for value in frame[column].array
        ]
        frame[column] = _pd.Series(
            values, index=frame.index, dtype="datetime64[ns, UTC]"
        )
    for column in _NULLABLE_TEMPORAL_SCALAR_COLUMNS:
        values = [
            _exact_utc_timestamp(value, label=column, nullable=True)
            for value in frame[column].array
        ]
        frame[column] = _pd.Series(
            values, index=frame.index, dtype="datetime64[ns, UTC]"
        )
    if not (frame["donor_event_start"] < frame["target_event_start"]).all():
        raise RealAppSelectorUnavailable(
            "Donor event start must be strictly prior to target event start"
        )
    dated_skill = frame["donor_skill_date"].notna()
    if not (
        frame.loc[dated_skill, "donor_skill_date"]
        < frame.loc[dated_skill, "donor_event_start"]
    ).all():
        raise RealAppSelectorUnavailable(
            "Donor skill date must be strictly prior to donor event start"
        )


def _validate_policy_and_groups(frame: _pd.DataFrame) -> None:
    # One proposal donor is repeated across all of its complete mechanics
    # branches.  Validate one exact policy row per bridge only after proving
    # that no bridge identifier names two different policy payloads.
    policy_frame = frame.loc[:, list(_POLICY_COLUMNS)].drop_duplicates()
    if policy_frame["bridge_row_id"].astype(str).duplicated().any():
        raise RealAppSelectorUnavailable(
            "One bridge row names multiple donor-ability policy payloads"
        )
    try:
        _policy.validate_applied_frame(policy_frame.reset_index(drop=True))
    except Exception as exc:
        raise RealAppSelectorUnavailable(
            f"Policy-v2 row attestation failed: {exc}"
        ) from exc
    kinds = frame["donor_ability_value_kind"].astype(str).to_numpy()
    ability = frame["donor_ability_sg"].to_numpy(float)
    available = frame["donor_ability_available"].astype(bool).to_numpy()
    imputed = kinds == _policy.IMPUTED_VALUE_KIND
    observed = kinds == _policy.OBSERVED_VALUE_KIND
    if not _np.logical_or(imputed, observed).all():
        raise RealAppSelectorUnavailable("Unexpected policy-v2 ability state")
    if available[imputed].any() or not available[observed].all():
        raise RealAppSelectorUnavailable("Policy-v2 availability changed")
    if (
        not _np.equal(ability[imputed], 0.0).all()
        or _np.signbit(ability[imputed]).any()
    ):
        raise RealAppSelectorUnavailable(
            "Neutral-imputed APP ability must remain exact positive +0.0"
        )
    expected_groups = _pd.Series(
        [
            _chain_v3.canonical_donor_quality_group_id(row)
            for row in frame.loc[
                :, ["donor_event_key", "category", "start_state_key"]
            ].to_dict("records")
        ],
        index=frame.index,
        dtype="object",
    )
    if not frame["donor_quality_group_id"].astype(str).equals(expected_groups):
        raise RealAppSelectorUnavailable(
            "Donor normalization group is not the canonical event/category/state group"
        )


def _validate_terminal_proposals(
    terminal_rows: _pd.DataFrame, holeout: float
) -> None:
    if len(terminal_rows) != 1 or terminal_rows["proposal_row_order"].nunique() != 1:
        raise RealAppSelectorUnavailable(
            "Fixed real H8 APP support requires exactly one terminal atom"
        )
    proposals = []
    for proposal, rows in terminal_rows.groupby(
        "proposal_row_order", sort=True, observed=True
    ):
        if len(rows) != 1:
            raise RealAppSelectorUnavailable(
                "Each terminal proposal must be one complete whole atom"
            )
        row = rows.iloc[0]
        if (
            str(row["proposal_branch"]) != "terminal"
            or int(row["surface_class_count"]) != 0
            or str(row["surface_full_probability_json"]) != "[]"
            or str(row["surface_full_cdf_upper_json"]) != "[]"
            or int(row["surface_class_index"]) != -1
            or str(row["surface_class_label"]) != "explicit_app_holeout"
            or float(row["surface_probability"]) != 1.0
            or float(row["surface_cdf_lower"]) != 0.0
            or float(row["surface_cdf_upper"]) != 1.0
            or int(row["hazard_donor_order"]) != -1
            or str(row["hazard_donor_key"]) != "not_applicable"
            or float(row["hazard_probability"]) != 1.0
            or float(row["hazard_cdf_lower"]) != 0.0
            or float(row["hazard_cdf_upper"]) != 1.0
            or str(row["hazard_resolution_status"]) != "none"
            or int(row["stroke_cost"]) != 1
            or str(row["end_state_key"]) != "terminal"
            or str(row["next_support_key"]) != ""
        ):
            raise RealAppSelectorUnavailable("Terminal APP sentinels changed")
        _require_close(
            float(row["mechanics_conditional_probability"]),
            1.0,
            "terminal mechanics probability",
        )
        proposals.append(
            (
                int(proposal),
                float(row["proposal_probability"]),
                float(row["frozen_terminal_residual_cdf_lower"]),
                float(row["frozen_terminal_residual_cdf_upper"]),
            )
        )
        if not (
            _pd.isna(row["frozen_spatial_cdf_lower"])
            and _pd.isna(row["frozen_spatial_cdf_upper"])
        ):
            raise RealAppSelectorUnavailable("Terminal proposal has a spatial CDF")
    if not proposals:
        raise RealAppSelectorUnavailable("APP support lacks a terminal atom")
    prior = 0.0
    terminal_mass = []
    for _, mass, lower, upper in proposals:
        if lower != prior or not upper > lower:
            raise RealAppSelectorUnavailable("Terminal residual CDF changed")
        _require_close(upper - lower, mass / holeout, "terminal CDF width")
        prior = upper
        terminal_mass.append(mass)
    if prior != 1.0:
        raise RealAppSelectorUnavailable("Terminal residual CDF does not close")
    _require_mass_close(
        _math.fsum(terminal_mass), holeout, "declared terminal mass"
    )


def _validate_surface_hazard_topology(rows: _pd.DataFrame) -> None:
    count = int(_single(rows["surface_class_count"], "surface class count"))
    if count != _MAX_SURFACE_CLASSES:
        raise RealAppSelectorUnavailable(
            "Real APP spatial proposals require exactly nine surface slots"
        )
    probability_json = str(
        _single(rows["surface_full_probability_json"], "surface probability JSON")
    )
    cdf_json = str(
        _single(rows["surface_full_cdf_upper_json"], "surface CDF JSON")
    )
    probability = _json_vector(probability_json, "surface probability JSON")
    cdf = _json_vector(cdf_json, "surface CDF JSON")
    if (
        len(probability) != count
        or len(cdf) != count
        or (probability < 0.0).any()
        or not _np.array_equal(cdf, _np.cumsum(probability))
        or not _math.isclose(
            float(cdf[-1]), 1.0, rel_tol=0.0, abs_tol=_MASS_TOLERANCE
        )
    ):
        raise RealAppSelectorUnavailable("Frozen full-surface topology changed")
    if not probability[0] > 0.0:
        raise RealAppSelectorUnavailable(
            "Real APP surface slot zero must have positive selectable mass"
        )
    fixed_zero = probability[list(_FIXED_ZERO_SURFACE_SLOTS)]
    if not _np.equal(fixed_zero, 0.0).all() or _np.signbit(fixed_zero).any():
        raise RealAppSelectorUnavailable(
            "Real H8 APP surface slots 5, 6, and 8 must remain exact positive zero"
        )
    surface_lower = _np.r_[0.0, cdf[:-1]]
    positive = probability > 0.0
    if not (cdf[positive] > surface_lower[positive]).all():
        raise RealAppSelectorUnavailable(
            "Positive surface slot lacks a distinct binary64 interval"
        )
    positive_classes = tuple(_np.flatnonzero(probability > 0.0).tolist())
    materialized_classes = tuple(
        sorted(rows["surface_class_index"].astype(int).unique().tolist())
    )
    if materialized_classes != positive_classes:
        raise RealAppSelectorUnavailable(
            "Materialized surfaces do not exactly match positive full-surface slots"
        )
    for surface, surface_rows in rows.groupby(
        "surface_class_index", sort=True, observed=True
    ):
        surface = int(surface)
        if str(_single(surface_rows["surface_class_label"], "surface label")) != (
            _APP_SURFACE_LABELS[surface]
        ):
            raise RealAppSelectorUnavailable(
                "APP surface index-to-label mapping changed"
            )
        end_states = surface_rows["end_state_key"].astype(str)
        next_support = surface_rows["next_support_key"].astype(str)
        if (
            end_states.eq("").any()
            or end_states.eq("terminal").any()
            or next_support.eq("").any()
        ):
            raise RealAppSelectorUnavailable(
                "Spatial APP atoms require live end-state and next-support keys"
            )
        expected_lower = 0.0 if surface == 0 else float(cdf[surface - 1])
        expected_upper = float(cdf[surface])
        _require_close(
            float(_single(surface_rows["surface_probability"], "surface probability")),
            float(probability[surface]),
            "surface probability",
        )
        _require_close(
            float(_single(surface_rows["surface_cdf_lower"], "surface CDF lower")),
            expected_lower,
            "surface CDF lower",
        )
        _require_close(
            float(_single(surface_rows["surface_cdf_upper"], "surface CDF upper")),
            expected_upper,
            "surface CDF upper",
        )
        statuses = surface_rows["hazard_resolution_status"].astype(str).unique()
        if len(statuses) != 1:
            raise RealAppSelectorUnavailable("A surface mixes hazard semantics")
        status = str(statuses[0])
        expected_status = (
            "resolved_complete_atom" if surface == 7 else "none"
        )
        if status != expected_status:
            raise RealAppSelectorUnavailable(
                "APP surface-to-hazard linkage changed"
            )
        if status == "none":
            if (
                len(surface_rows) != 1
                or int(surface_rows.iloc[0]["hazard_donor_order"]) != -1
                or str(surface_rows.iloc[0]["hazard_donor_key"]) != "not_applicable"
                or int(surface_rows.iloc[0]["stroke_cost"]) != 1
            ):
                raise RealAppSelectorUnavailable("No-hazard singleton changed")
            _require_close(
                float(surface_rows.iloc[0]["hazard_probability"]),
                1.0,
                "no-hazard probability",
            )
            _require_close(
                float(surface_rows.iloc[0]["hazard_cdf_lower"]),
                0.0,
                "no-hazard CDF lower",
            )
            _require_close(
                float(surface_rows.iloc[0]["hazard_cdf_upper"]),
                1.0,
                "no-hazard CDF upper",
            )
        elif status == "resolved_complete_atom":
            ordered = surface_rows.sort_values("hazard_donor_order", kind="mergesort")
            orders = ordered["hazard_donor_order"].astype(int).to_numpy()
            if (
                len(ordered) > _MAX_HAZARDS_PER_SURFACE
                or not _np.array_equal(orders, _np.arange(len(ordered)))
                or ordered["hazard_donor_key"].astype(str).duplicated().any()
            ):
                raise RealAppSelectorUnavailable("Resolved hazard order changed")
            hazard_probability = ordered["hazard_probability"].to_numpy(float)
            hazard_lower = ordered["hazard_cdf_lower"].to_numpy(float)
            hazard_upper = ordered["hazard_cdf_upper"].to_numpy(float)
            donor_keys = ordered["hazard_donor_key"].astype(str).tolist()
            if donor_keys != sorted(donor_keys) or (
                ordered["stroke_cost"].astype(int).to_numpy() < 2
            ).any():
                raise RealAppSelectorUnavailable(
                    "Resolved hazards require ordered donors and penalty stroke cost"
                )
            expected_lower = _np.r_[0.0, _np.cumsum(hazard_probability)[:-1]]
            expected_upper = _np.cumsum(hazard_probability)
            if not (hazard_upper > hazard_lower).all():
                raise RealAppSelectorUnavailable(
                    "Positive hazard branch lacks a distinct binary64 interval"
                )
            if (
                (hazard_probability <= 0.0).any()
                or not _np.array_equal(hazard_lower, expected_lower)
                or not _np.array_equal(hazard_upper, expected_upper)
                or not _math.isclose(
                    float(hazard_upper[-1]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=_MASS_TOLERANCE,
                )
            ):
                raise RealAppSelectorUnavailable("Resolved hazard CDF changed")
        else:
            raise RealAppSelectorUnavailable("Unknown hazard resolution status")
        observed_mechanics = surface_rows[
            "mechanics_conditional_probability"
        ].to_numpy(float)
        expected_mechanics = float(probability[surface]) * surface_rows[
            "hazard_probability"
        ].to_numpy(float)
        if not _np.allclose(
            observed_mechanics,
            expected_mechanics,
            rtol=2e-15,
            atol=2e-15,
        ):
            raise RealAppSelectorUnavailable(
                "Whole-atom mechanics no longer equal surface times hazard mass"
            )


def _validate_spatial_proposals(
    spatial_rows: _pd.DataFrame, holeout: float
) -> None:
    proposals = []
    for proposal, rows in spatial_rows.groupby(
        "proposal_row_order", sort=True, observed=True
    ):
        if str(_single(rows["proposal_branch"], "proposal branch")) != "spatial":
            raise RealAppSelectorUnavailable("Spatial proposal branch changed")
        probability = float(
            _single(rows["proposal_probability"], "proposal probability")
        )
        lower = float(_single(rows["frozen_spatial_cdf_lower"], "spatial CDF lower"))
        upper = float(_single(rows["frozen_spatial_cdf_upper"], "spatial CDF upper"))
        if (
            rows["frozen_terminal_residual_cdf_lower"].notna().any()
            or rows["frozen_terminal_residual_cdf_upper"].notna().any()
        ):
            raise RealAppSelectorUnavailable("Spatial proposal has a terminal CDF")
        proposals.append((int(proposal), probability, lower, upper))
        _validate_surface_hazard_topology(rows)
        _require_mass_close(
            _math.fsum(
                map(float, rows["mechanics_conditional_probability"].array)
            ),
            1.0,
            "spatial proposal mechanics mass",
        )
        _require_mass_close(
            _math.fsum(map(float, rows["quality_base_mass"].array)),
            probability,
            "spatial proposal base mass",
        )
    if not proposals:
        raise RealAppSelectorUnavailable("APP support lacks spatial proposals")
    if len(proposals) >= _MAX_PROPOSALS:
        # One proposal slot is reserved for the required terminal proposal.
        raise RealAppSelectorUnavailable("Spatial proposal cap exceeded")
    prior = 0.0
    spatial_mass = []
    for _, mass, lower, upper in proposals:
        if lower != prior or not upper > lower:
            raise RealAppSelectorUnavailable("Frozen spatial proposal CDF changed")
        _require_close(
            upper - lower, mass / (1.0 - holeout), "spatial proposal CDF width"
        )
        prior = upper
        spatial_mass.append(mass)
    if prior != 1.0:
        raise RealAppSelectorUnavailable("Frozen spatial proposal CDF does not close")
    _require_mass_close(
        _math.fsum(spatial_mass), 1.0 - holeout, "declared spatial mass"
    )


def _canonicalize_fixture(frame: object) -> _pd.DataFrame:
    """Validate one in-memory fixture; never load or admit a real artifact."""

    _validate_runtime()
    if type(frame) is not _pd.DataFrame:
        raise RealAppSelectorUnavailable("Real-shaped APP fixture must be a DataFrame")
    missing = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise RealAppSelectorUnavailable(f"Real-shaped APP fixture misses={missing}")
    out = frame.copy(deep=True).reset_index(drop=True)
    _validate_scalar_families(out)
    _normalize_temporal_columns(out)
    rows = len(out)
    if rows <= 0 or rows > _MAX_ATOMS:
        raise RealAppSelectorUnavailable("Real-shaped APP atom cap changed")
    complete_order = out["complete_row_order"].to_numpy()
    if not _np.array_equal(complete_order, _np.arange(rows, dtype=complete_order.dtype)):
        raise RealAppSelectorUnavailable("Complete source row order changed")
    if out["atom_id"].astype(str).duplicated().any():
        raise RealAppSelectorUnavailable("Complete APP atom IDs are duplicated")
    if out["query_key"].astype(str).nunique() != 1:
        raise RealAppSelectorUnavailable("Fixture must contain one exact query")
    if set(out["category"].astype(str).str.casefold()) != {"app"}:
        raise RealAppSelectorUnavailable("Fixture must contain only APP atoms")
    if not out["complete_atom"].astype(bool).all():
        raise RealAppSelectorUnavailable("Every fixture row must be a whole atom")
    if out["boundary_only"].astype(bool).any():
        raise RealAppSelectorUnavailable(
            "The fixed real H8 shape has no materialized boundary-only atoms"
        )
    proposal_order = out["proposal_row_order"].to_numpy(int)
    if (
        (proposal_order < 0).any()
        or not _np.array_equal(
            _np.unique(proposal_order), _np.arange(int(proposal_order.max()) + 1)
        )
        or (proposal_order[1:] < proposal_order[:-1]).any()
        or len(_np.unique(proposal_order)) > _MAX_PROPOSALS
    ):
        raise RealAppSelectorUnavailable("Proposal source order changed")
    numeric_columns = (
        "proposal_probability",
        "mechanics_conditional_probability",
        "quality_base_mass",
        "fixed_reference_reward",
        "donor_ability_sg",
    )
    numeric = out.loc[:, numeric_columns].apply(_pd.to_numeric, errors="coerce")
    if not _np.isfinite(numeric.to_numpy(float)).all():
        raise RealAppSelectorUnavailable("Fixture weighting columns are not finite")
    base = numeric["quality_base_mass"].to_numpy(float)
    if (
        (base <= 0.0).any()
        or not _math.isclose(
            _math.fsum(map(float, base)),
            1.0,
            rel_tol=0.0,
            abs_tol=_MASS_TOLERANCE,
        )
    ):
        raise RealAppSelectorUnavailable("Fixture base mass must be positive and close")
    expected_base = (
        numeric["proposal_probability"].to_numpy(float)
        * numeric["mechanics_conditional_probability"].to_numpy(float)
    )
    if not _np.allclose(base, expected_base, atol=2e-15, rtol=2e-15):
        raise RealAppSelectorUnavailable("Whole-atom mass factorization changed")
    references = out["reference_model_sha256"].astype(str).unique().tolist()
    putting = out["reference_putting_sha256"].astype(str).unique().tolist()
    if (
        len(references) != 1
        or len(putting) != 1
        or not _is_sha256(references[0])
        or not _is_sha256(putting[0])
    ):
        raise RealAppSelectorUnavailable("Fixed reference identity changed")
    if (out["stroke_cost"].to_numpy(int) <= 0).any():
        raise RealAppSelectorUnavailable("Every whole atom needs positive stroke cost")
    terminal = out["terminal"].astype(bool)
    if not terminal.any() or terminal.all():
        raise RealAppSelectorUnavailable("APP fixture needs terminal and spatial atoms")
    terminal_proposals = set(out.loc[terminal, "proposal_row_order"].astype(int))
    spatial_proposals = set(out.loc[~terminal, "proposal_row_order"].astype(int))
    if terminal_proposals & spatial_proposals:
        raise RealAppSelectorUnavailable("A proposal mixes terminal and spatial atoms")
    declared = out["declared_app_base_holeout_probability"].to_numpy(float)
    if (
        not _np.isfinite(declared).all()
        or not _np.equal(declared, declared[0]).all()
        or not 0.0 < float(declared[0]) < 1.0
    ):
        raise RealAppSelectorUnavailable("Declared APP holeout probability changed")
    _validate_policy_and_groups(out)
    _validate_terminal_proposals(out.loc[terminal], float(declared[0]))
    _validate_spatial_proposals(out.loc[~terminal], float(declared[0]))
    return out


@_dataclass(frozen=True)
class RealAppWeightingParameters:
    transfer_beta: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    lambda_correction: tuple[float, float, float, float]
    reward_reference_sha256: str
    max_abs_log_weight: float = _MAX_ABS_LOG_WEIGHT


@_dataclass(frozen=True)
class RealAppSemanticUniforms:
    holeout: float
    proposal: float
    surface: float
    hazard: float


@_dataclass(frozen=True)
class RealAppWeightedSupport:
    contract: str
    runtime_binding_sha256: str
    support_sha256: str
    parameter_sha256: str
    atom_id: tuple[str, ...]
    probability: tuple[float, ...]
    neutral_probability_mass: tuple[float, ...]
    donor_group_mass_before: tuple[float, ...]
    donor_group_mass_after: tuple[float, ...]
    terminal_probability: float
    clipped_donor_logs: int
    clipped_target_logs: int
    zero_fast_path: bool
    selection_mode: str
    whole_atom_vector_preserved: bool
    group_wide_weighting: bool


@_dataclass(frozen=True)
class RealAppSelectionResult:
    atom_index: int
    complete_row_order: int
    atom_id: str
    terminal: bool
    proposal_row_order: int
    surface_class_index: int
    hazard_donor_order: int
    stroke_cost: int
    end_state_key: str
    next_support_key: str
    terminal_probability: float
    terminal_residual_uniform: float | None
    selection_mode: str
    hierarchical_atom_id: str
    flat_atom_id: str
    flat_hierarchical_equal: bool
    semantic_uniforms: RealAppSemanticUniforms


@_dataclass(frozen=True)
class _Node:
    keys: tuple[int, ...]
    upper: tuple[float, ...]
    exact_ratio: tuple[str | None, ...]


@_dataclass(frozen=True)
class _Hierarchy:
    terminal_indices: tuple[int, ...]
    terminal_conditional: _Node
    terminal_unconditional_upper: tuple[float, ...]
    proposal: _Node
    surfaces: tuple[tuple[int, _Node], ...]
    hazards: tuple[tuple[tuple[int, int], _Node], ...]


@_dataclass(frozen=True)
class _WeightedFixtureState:
    frame: _pd.DataFrame
    weighted: RealAppWeightedSupport
    hierarchy: _Hierarchy


def _four_vector(value: object, label: str) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise RealAppSelectorUnavailable(f"{label} must be an immutable four-vector")
    result = tuple(_plain_finite(item, label) for item in value)
    return result  # type: ignore[return-value]


def _anchor_parameters(
    parameters: object, reference_sha256: str
) -> tuple[_anchor.TransitionQualityParameters, bool]:
    if type(parameters) is not RealAppWeightingParameters:
        raise RealAppSelectorUnavailable("Real APP weighting parameter type changed")
    beta = _four_vector(parameters.transfer_beta, "transfer_beta")
    target = _four_vector(parameters.target_ability_sg, "target_ability_sg")
    correction = _four_vector(parameters.lambda_correction, "lambda_correction")
    if any(item < 0.0 for item in beta):
        raise RealAppSelectorUnavailable("transfer_beta cannot be negative")
    parameter_reference = parameters.reward_reference_sha256
    if type(parameter_reference) is not str or not _is_sha256(parameter_reference):
        raise RealAppSelectorUnavailable(
            "Weighting reward reference must be an exact lowercase SHA-256 string"
        )
    if type(reference_sha256) is not str or not _is_sha256(reference_sha256):
        raise RealAppSelectorUnavailable(
            "Fixture reward reference must be an exact lowercase SHA-256 string"
        )
    if not str.__eq__(parameter_reference, reference_sha256):
        raise RealAppSelectorUnavailable("Weighting reward reference changed")
    if (
        _plain_finite(parameters.max_abs_log_weight, "max_abs_log_weight")
        != _MAX_ABS_LOG_WEIGHT
    ):
        raise RealAppSelectorUnavailable("Weighting log cap changed")
    try:
        result = _anchor.TransitionQualityParameters(
            transfer_beta=beta,
            target_ability_sg=target,
            lambda_correction=correction,
            reward_reference_sha256=reference_sha256,
            max_abs_log_weight=_MAX_ABS_LOG_WEIGHT,
        )
    except (TypeError, ValueError) as exc:
        raise RealAppSelectorUnavailable(
            f"Real APP weighting parameters are invalid: {exc}"
        ) from exc
    # This seam contains APP atoms only.  Category-isolated OTT/ARG/PUTT
    # coefficients cannot move an APP probability and therefore must retain
    # the exact frozen zero-path CDFs rather than rebuilding projected cuts.
    return result, beta[1] == 0.0 and correction[1] == 0.0


def _support_digest_scalar(value: object) -> dict[str, object]:
    """Canonicalize one validated scalar without lossy string coercion."""

    value_type = type(value)
    if value is None or value is _pd.NA or value_type is _EXACT_NAT_TYPE:
        return {"type": "missing"}
    if value_type in _EXACT_BOOLEAN_TYPES:
        return {"type": "bool", "value": bool(value)}
    if value_type in _EXACT_INTEGER_TYPES:
        return {"type": "integer", "value": str(int(value))}
    if value_type in _EXACT_FLOAT_TYPES:
        number = float(value)
        if _math.isnan(number):
            return {"type": "missing"}
        if not _math.isfinite(number):
            raise RealAppSelectorUnavailable(
                "Validated support digest cannot contain infinity"
            )
        return {"type": "binary64", "value": number.hex()}
    if value_type is _pd.Timestamp:
        if value.tzinfo is None:
            raise RealAppSelectorUnavailable(
                "Validated support digest received a naive timestamp"
            )
        return {
            "type": "timestamp_utc",
            "value": value.tz_convert("UTC").isoformat(),
        }
    if value_type is _np.datetime64:
        if _np.isnat(value):
            return {"type": "missing"}
        return {
            "type": "timestamp_utc",
            "value": _pd.Timestamp(value).tz_localize("UTC").isoformat(),
        }
    if value_type in _EXACT_STRING_TYPES:
        return {"type": "string", "value": str(value)}
    raise RealAppSelectorUnavailable(
        f"Validated support digest received unsupported scalar {type(value).__name__}"
    )


def _support_sha(frame: _pd.DataFrame) -> str:
    # Bind the entire closed input contract, including frozen CDF topology,
    # terminal sentinels, selected output payload, reference identities, and
    # policy attestations.  Atom IDs and masses alone cannot identify which
    # atom a retained semantic uniform reaches.
    digest_columns = tuple(dict.fromkeys(_REQUIRED_COLUMNS))
    records = [
        {
            column: _support_digest_scalar(frame.at[index, column])
            for column in digest_columns
        }
        for index in frame.index
    ]
    return _stable_sha(
        {
            "contract": "real_shaped_app_logic_fixture_full_contract_digest_v2",
            "columns": list(digest_columns),
            "records": records,
        }
    )


def _fraction_mass(probability: _Sequence[float], indices: _Sequence[int]) -> _Fraction:
    return sum(
        (_Fraction.from_float(float(probability[index])) for index in indices),
        _Fraction(0, 1),
    )


def _ratio_text(value: _Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction_node(
    keys: _Sequence[int],
    masses: _Sequence[_Fraction],
    *,
    boundary: str,
    allow_zero: bool,
) -> _Node:
    if len(keys) != len(masses) or not len(keys):
        raise RealAppSelectorUnavailable("Conditional hierarchy node is malformed")
    denominator = sum(masses, _Fraction(0, 1))
    if denominator <= 0 or any(mass < 0 for mass in masses):
        raise RealAppSelectorUnavailable("Conditional hierarchy mass is invalid")
    running = _Fraction(0, 1)
    prior_cut = 0.0
    upper = []
    exact = []
    for mass in masses:
        running += mass
        cut = _synthetic._binary64_ratio_cut(  # reviewed source-pinned primitive
            running, denominator, boundary=boundary
        )
        if mass > 0 and not cut > prior_cut:
            raise RealAppSelectorUnavailable(
                "Positive hierarchy branch has no binary64 selection interval"
            )
        if mass == 0 and not allow_zero:
            raise RealAppSelectorUnavailable("Unexpected zero hierarchy branch")
        upper.append(cut)
        exact.append(_ratio_text(running / denominator))
        prior_cut = cut
    if running != denominator or upper[-1] != 1.0:
        raise RealAppSelectorUnavailable("Conditional hierarchy node does not close")
    return _Node(tuple(map(int, keys)), tuple(upper), tuple(exact))


def _zero_hierarchy(frame: _pd.DataFrame) -> tuple[float, _Hierarchy]:
    terminal = frame["terminal"].astype(bool).to_numpy()
    terminal_indices = tuple(_np.flatnonzero(terminal).tolist())
    terminal_rows = frame.iloc[list(terminal_indices)].sort_values(
        "proposal_row_order", kind="mergesort"
    )
    terminal_indices = tuple(terminal_rows.index.tolist())
    terminal_conditional = _Node(
        tuple(terminal_rows["proposal_row_order"].astype(int)),
        tuple(terminal_rows["frozen_terminal_residual_cdf_upper"].astype(float)),
        (None,) * len(terminal_rows),
    )
    terminal_probability = float(
        frame["declared_app_base_holeout_probability"].iloc[0]
    )
    running = 0.0
    terminal_unconditional = []
    for index in terminal_indices:
        running = float(running + float(frame.iloc[index]["quality_base_mass"]))
        terminal_unconditional.append(running)
    spatial = frame.loc[~frame["terminal"].astype(bool)]
    proposal_rows = spatial.groupby(
        "proposal_row_order", sort=True, observed=True
    ).first()
    proposal = _Node(
        tuple(map(int, proposal_rows.index)),
        tuple(proposal_rows["frozen_spatial_cdf_upper"].astype(float)),
        (None,) * len(proposal_rows),
    )
    surfaces = []
    hazards = []
    for proposal_order, rows in spatial.groupby(
        "proposal_row_order", sort=True, observed=True
    ):
        count = int(rows["surface_class_count"].iloc[0])
        surface_upper = _json_vector(
            str(rows["surface_full_cdf_upper_json"].iloc[0]), "surface CDF JSON"
        )
        surfaces.append(
            (
                int(proposal_order),
                _Node(tuple(range(count)), tuple(map(float, surface_upper)), (None,) * count),
            )
        )
        for surface, surface_rows in rows.groupby(
            "surface_class_index", sort=True, observed=True
        ):
            ordered = surface_rows.sort_values("hazard_donor_order", kind="mergesort")
            if str(ordered["hazard_resolution_status"].iloc[0]) == "none":
                keys = (-1,)
            else:
                keys = tuple(ordered["hazard_donor_order"].astype(int))
            hazards.append(
                (
                    (int(proposal_order), int(surface)),
                    _Node(
                        keys,
                        tuple(ordered["hazard_cdf_upper"].astype(float)),
                        (None,) * len(ordered),
                    ),
                )
            )
    return terminal_probability, _Hierarchy(
        terminal_indices=terminal_indices,
        terminal_conditional=terminal_conditional,
        terminal_unconditional_upper=tuple(terminal_unconditional),
        proposal=proposal,
        surfaces=tuple(surfaces),
        hazards=tuple(hazards),
    )


def _tilted_hierarchy(
    frame: _pd.DataFrame, probability: _np.ndarray
) -> tuple[float, _Hierarchy]:
    terminal = frame["terminal"].astype(bool).to_numpy()
    terminal_indices = tuple(
        frame.loc[terminal].sort_values("proposal_row_order", kind="mergesort").index
    )
    terminal_mass = [_Fraction.from_float(float(probability[i])) for i in terminal_indices]
    terminal_conditional = _fraction_node(
        tuple(frame.loc[list(terminal_indices), "proposal_row_order"].astype(int)),
        terminal_mass,
        boundary="right",
        allow_zero=False,
    )
    running_terminal = 0.0
    terminal_unconditional = []
    for index in terminal_indices:
        retained = float(running_terminal + float(probability[index]))
        if not retained > running_terminal:
            raise RealAppSelectorUnavailable("Terminal prefix collapsed in binary64")
        terminal_unconditional.append(retained)
        running_terminal = retained
    terminal_probability = running_terminal
    if not 0.0 < terminal_probability < 1.0:
        raise RealAppSelectorUnavailable("Tilted terminal probability is invalid")

    spatial_indices = _np.flatnonzero(~terminal).tolist()
    proposal_keys = tuple(
        sorted(frame.loc[spatial_indices, "proposal_row_order"].astype(int).unique())
    )
    proposal_masses = []
    for proposal in proposal_keys:
        indices = frame.index[
            (~frame["terminal"].astype(bool))
            & frame["proposal_row_order"].eq(proposal)
        ].tolist()
        proposal_masses.append(_fraction_mass(probability, indices))
    proposal_node = _fraction_node(
        proposal_keys,
        proposal_masses,
        boundary="right",
        allow_zero=False,
    )

    surfaces = []
    hazards = []
    for proposal in proposal_keys:
        proposal_rows = frame.loc[
            (~frame["terminal"].astype(bool))
            & frame["proposal_row_order"].eq(proposal)
        ]
        count = int(proposal_rows["surface_class_count"].iloc[0])
        surface_masses = []
        for surface in range(count):
            indices = proposal_rows.index[
                proposal_rows["surface_class_index"].eq(surface)
            ].tolist()
            surface_masses.append(_fraction_mass(probability, indices))
        surfaces.append(
            (
                proposal,
                _fraction_node(
                    tuple(range(count)),
                    surface_masses,
                    boundary="strict",
                    allow_zero=True,
                ),
            )
        )
        for surface in _np.flatnonzero(
            _np.asarray([mass > 0 for mass in surface_masses], dtype=bool)
        ):
            rows = proposal_rows.loc[
                proposal_rows["surface_class_index"].eq(int(surface))
            ].sort_values("hazard_donor_order", kind="mergesort")
            if str(rows["hazard_resolution_status"].iloc[0]) == "none":
                keys = (-1,)
            else:
                keys = tuple(rows["hazard_donor_order"].astype(int))
            masses = [
                _Fraction.from_float(float(probability[index])) for index in rows.index
            ]
            hazards.append(
                (
                    (proposal, int(surface)),
                    _fraction_node(
                        keys,
                        masses,
                        boundary="right",
                        allow_zero=False,
                    ),
                )
            )
    return terminal_probability, _Hierarchy(
        terminal_indices=terminal_indices,
        terminal_conditional=terminal_conditional,
        terminal_unconditional_upper=tuple(terminal_unconditional),
        proposal=proposal_node,
        surfaces=tuple(surfaces),
        hazards=tuple(hazards),
    )


def _weight_fixture(
    fixture: object, parameters: object
) -> _WeightedFixtureState:
    """Exercise logic on an in-memory fixture; not a public admission path."""

    frame = _canonicalize_fixture(fixture)
    runtime = _validate_runtime()
    reference = str(frame["reference_model_sha256"].iloc[0])
    anchor_parameters, zero = _anchor_parameters(parameters, reference)
    try:
        atom_mass = _anchor.whole_atom_probabilities(
            base_mass=frame["quality_base_mass"].to_numpy(float),
            fixed_reference_reward=frame["fixed_reference_reward"].to_numpy(float),
            donor_ability_sg=frame["donor_ability_sg"].to_numpy(float),
            donor_group_id=frame["donor_quality_group_id"].astype(str).to_numpy(),
            category="app",
            parameters=anchor_parameters,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        raise RealAppSelectorUnavailable(
            f"Whole-atom APP fixture weighting failed: {exc}"
        ) from exc
    probability = _np.asarray(atom_mass.probability, dtype=float)
    neutral = _np.asarray(atom_mass.neutral_probability_mass, dtype=float)
    if (
        probability.shape != (len(frame),)
        or neutral.shape != probability.shape
        or not _np.isfinite(probability).all()
        or not _np.isfinite(neutral).all()
        or (probability <= 0.0).any()
        or not _math.isclose(
            _math.fsum(map(float, probability)),
            1.0,
            rel_tol=0.0,
            abs_tol=_MASS_TOLERANCE,
        )
        or atom_mass.clipped_donor_logs != 0
        or atom_mass.clipped_target_logs != 0
    ):
        raise RealAppSelectorUnavailable("Whole-atom APP fixture result is invalid")
    if zero:
        terminal_probability, hierarchy = _zero_hierarchy(frame)
        mode = REAL_APP_ZERO_PATH_CONTRACT
    else:
        terminal_probability, hierarchy = _tilted_hierarchy(frame, probability)
        mode = REAL_APP_TILTED_PATH_CONTRACT
    weighted = RealAppWeightedSupport(
        contract=REAL_APP_SELECTOR_CONTRACT,
        runtime_binding_sha256=runtime,
        support_sha256=_support_sha(frame),
        parameter_sha256=anchor_parameters.parameter_sha256,
        atom_id=tuple(frame["atom_id"].astype(str)),
        probability=tuple(map(float, probability)),
        neutral_probability_mass=tuple(map(float, neutral)),
        donor_group_mass_before=tuple(map(float, atom_mass.donor_group_mass_before)),
        donor_group_mass_after=tuple(map(float, atom_mass.donor_group_mass_after)),
        terminal_probability=float(terminal_probability),
        clipped_donor_logs=int(atom_mass.clipped_donor_logs),
        clipped_target_logs=int(atom_mass.clipped_target_logs),
        zero_fast_path=zero,
        selection_mode=mode,
        whole_atom_vector_preserved=True,
        group_wide_weighting=True,
    )
    return _WeightedFixtureState(frame=frame, weighted=weighted, hierarchy=hierarchy)


def _uniforms(value: object) -> tuple[float, float, float, float]:
    if type(value) is not RealAppSemanticUniforms:
        raise RealAppSelectorUnavailable("Real APP uniform type changed")
    result = tuple(
        _plain_finite(getattr(value, name), f"{name}_uniform")
        for name in ("holeout", "proposal", "surface", "hazard")
    )
    if not all(0.0 <= item < 1.0 for item in result):
        raise RealAppSelectorUnavailable("Every semantic uniform must satisfy 0 <= u < 1")
    return result  # type: ignore[return-value]


def _select_right(node: _Node, uniform: float) -> int:
    index = min(
        int(_np.searchsorted(_np.asarray(node.upper, dtype=float), uniform, side="right")),
        len(node.keys) - 1,
    )
    return node.keys[index]


def _select_surface(node: _Node, uniform: float) -> int:
    index = min(
        int(_np.count_nonzero(uniform > _np.asarray(node.upper, dtype=float))),
        len(node.keys) - 1,
    )
    return node.keys[index]


def _hierarchical_index(
    state: _WeightedFixtureState,
    retained: tuple[float, float, float, float],
) -> tuple[int, float | None]:
    holeout_u, proposal_u, surface_u, hazard_u = retained
    frame = state.frame
    hierarchy = state.hierarchy
    terminal_probability = state.weighted.terminal_probability
    if holeout_u < terminal_probability:
        residual = holeout_u / terminal_probability
        if state.weighted.zero_fast_path:
            proposal = _select_right(hierarchy.terminal_conditional, residual)
        else:
            slot = min(
                int(
                    _np.searchsorted(
                        _np.asarray(hierarchy.terminal_unconditional_upper),
                        holeout_u,
                        side="right",
                    )
                ),
                len(hierarchy.terminal_indices) - 1,
            )
            return hierarchy.terminal_indices[slot], residual
        matches = frame.index[
            frame["terminal"].astype(bool)
            & frame["proposal_row_order"].eq(proposal)
        ].tolist()
        if len(matches) != 1:
            raise RealAppSelectorUnavailable("Terminal selector leaf changed")
        return matches[0], residual
    proposal = _select_right(hierarchy.proposal, proposal_u)
    surface_nodes = dict(hierarchy.surfaces)
    surface = _select_surface(surface_nodes[proposal], surface_u)
    rows = frame.loc[
        (~frame["terminal"].astype(bool))
        & frame["proposal_row_order"].eq(proposal)
        & frame["surface_class_index"].eq(surface)
    ]
    if rows.empty:
        raise RealAppSelectorUnavailable(
            "Strict surface selection reached an unmaterialized zero-mass class"
        )
    hazard_nodes = dict(hierarchy.hazards)
    hazard = _select_right(hazard_nodes[(proposal, surface)], hazard_u)
    matches = rows.index[rows["hazard_donor_order"].eq(hazard)].tolist()
    if len(matches) != 1:
        raise RealAppSelectorUnavailable("Hazard selector leaf changed")
    return matches[0], None


def _direct_index(
    state: _WeightedFixtureState,
    retained: tuple[float, float, float, float],
) -> int:
    """Independent flat-mass oracle for retained binary64 uniforms."""

    holeout_u, proposal_u, surface_u, hazard_u = retained
    frame = state.frame
    probability = _np.asarray(state.weighted.probability, dtype=float)
    terminal_rows = frame.loc[frame["terminal"].astype(bool)].sort_values(
        "proposal_row_order", kind="mergesort"
    )
    terminal_probability = state.weighted.terminal_probability
    if holeout_u < terminal_probability:
        if state.weighted.zero_fast_path:
            residual = holeout_u / terminal_probability
            uppers = terminal_rows["frozen_terminal_residual_cdf_upper"].to_numpy(float)
            slot = min(int(_np.searchsorted(uppers, residual, side="right")), len(uppers) - 1)
            return int(terminal_rows.index[slot])
        running = 0.0
        for index in terminal_rows.index:
            running = float(running + probability[index])
            if holeout_u < running:
                return int(index)
        raise RealAppSelectorUnavailable("Direct terminal oracle did not resolve")

    spatial = frame.loc[~frame["terminal"].astype(bool)]
    if state.weighted.zero_fast_path:
        proposal_rows = spatial.groupby(
            "proposal_row_order", sort=True, observed=True
        ).first()
        proposal_slot = min(
            int(
                _np.searchsorted(
                    proposal_rows["frozen_spatial_cdf_upper"].to_numpy(float),
                    proposal_u,
                    side="right",
                )
            ),
            len(proposal_rows) - 1,
        )
        proposal = int(proposal_rows.index[proposal_slot])
        proposal_atoms = spatial.loc[spatial["proposal_row_order"].eq(proposal)]
        full_cdf = _json_vector(
            str(proposal_atoms["surface_full_cdf_upper_json"].iloc[0]),
            "surface CDF JSON",
        )
        surface = min(int(_np.count_nonzero(surface_u > full_cdf)), len(full_cdf) - 1)
        surface_atoms = proposal_atoms.loc[
            proposal_atoms["surface_class_index"].eq(surface)
        ].sort_values("hazard_donor_order", kind="mergesort")
        if surface_atoms.empty:
            raise RealAppSelectorUnavailable("Direct frozen surface oracle reached zero mass")
        hazard_slot = min(
            int(
                _np.searchsorted(
                    surface_atoms["hazard_cdf_upper"].to_numpy(float),
                    hazard_u,
                    side="right",
                )
            ),
            len(surface_atoms) - 1,
        )
        return int(surface_atoms.index[hazard_slot])

    spatial_indices = spatial.index.tolist()
    spatial_mass = _fraction_mass(probability, spatial_indices)
    proposal_fraction = _Fraction.from_float(proposal_u)
    proposal = None
    proposal_indices: list[int] = []
    proposal_mass = _Fraction(0, 1)
    running = _Fraction(0, 1)
    for key in sorted(spatial["proposal_row_order"].astype(int).unique()):
        indices = spatial.index[spatial["proposal_row_order"].eq(key)].tolist()
        mass = _fraction_mass(probability, indices)
        running += mass
        if proposal_fraction < running / spatial_mass:
            proposal = int(key)
            proposal_indices = indices
            proposal_mass = mass
            break
    if proposal is None:
        raise RealAppSelectorUnavailable("Direct proposal oracle did not resolve")
    proposal_rows = frame.loc[proposal_indices]
    count = int(proposal_rows["surface_class_count"].iloc[0])
    surface_fraction = _Fraction.from_float(surface_u)
    running = _Fraction(0, 1)
    surface = None
    surface_indices: list[int] = []
    surface_mass = _Fraction(0, 1)
    for key in range(count):
        indices = proposal_rows.index[
            proposal_rows["surface_class_index"].eq(key)
        ].tolist()
        mass = _fraction_mass(probability, indices)
        running += mass
        if surface_fraction <= running / proposal_mass:
            surface = key
            surface_indices = indices
            surface_mass = mass
            break
    if surface is None or not surface_indices:
        raise RealAppSelectorUnavailable("Direct surface oracle did not resolve")
    surface_rows = frame.loc[surface_indices].sort_values(
        "hazard_donor_order", kind="mergesort"
    )
    hazard_fraction = _Fraction.from_float(hazard_u)
    running = _Fraction(0, 1)
    for index in surface_rows.index:
        running += _Fraction.from_float(float(probability[index]))
        if hazard_fraction < running / surface_mass:
            return int(index)
    raise RealAppSelectorUnavailable("Direct hazard oracle did not resolve")


def _select_fixture(
    fixture: object,
    parameters: object,
    uniforms: object,
) -> RealAppSelectionResult:
    """Exercise retained-uniform selection only on an in-memory test fixture."""

    state = _weight_fixture(fixture, parameters)
    retained = _uniforms(uniforms)
    hierarchical, residual = _hierarchical_index(state, retained)
    direct = _direct_index(state, retained)
    if hierarchical != direct:
        raise RealAppSelectorUnavailable(
            "Hierarchical and independent flat real APP selectors diverged"
        )
    row = state.frame.iloc[hierarchical]
    return RealAppSelectionResult(
        atom_index=int(hierarchical),
        complete_row_order=int(row["complete_row_order"]),
        atom_id=str(row["atom_id"]),
        terminal=bool(row["terminal"]),
        proposal_row_order=int(row["proposal_row_order"]),
        surface_class_index=int(row["surface_class_index"]),
        hazard_donor_order=int(row["hazard_donor_order"]),
        stroke_cost=int(row["stroke_cost"]),
        end_state_key=str(row["end_state_key"]),
        next_support_key=str(row["next_support_key"]),
        terminal_probability=state.weighted.terminal_probability,
        terminal_residual_uniform=residual,
        selection_mode=state.weighted.selection_mode,
        hierarchical_atom_id=str(row["atom_id"]),
        flat_atom_id=str(state.frame.iloc[direct]["atom_id"]),
        flat_hierarchical_equal=True,
        semantic_uniforms=RealAppSemanticUniforms(*retained),
    )


__all__ = [
    "REAL_APP_SELECTOR_CONTRACT",
    "REAL_APP_AUTHORITY_CONTRACT",
    "REAL_APP_ZERO_PATH_CONTRACT",
    "REAL_APP_TILTED_PATH_CONTRACT",
    "REAL_APP_BOUNDARY_CONTRACT",
    "INTEGRATION_STATUS",
    "RealAppSelectorUnavailable",
    "RealAppWeightingParameters",
    "RealAppSemanticUniforms",
    "RealAppWeightedSupport",
    "RealAppSelectionResult",
    "real_app_authority_manifest",
    "admit_lowry_r1h8_app_support",
]

"""Frozen donor-ability policy for archive-only hole-simulation research.

This module closes one narrow provenance gap from the historical shot-quality
tilt evaluator.  It does not modify or enable the streaming chain, full-chain
runner, production simulator, or transition-mean anchor adapter.

The contract deliberately keeps availability separate from the numeric value
used by the research evaluator:

* a real strictly-prior observation remains available and is clipped to
  ``[-2, 2]`` only when applied;
* a genuinely missing observation remains unavailable, preserves its actual
  nullable skill date (strictly prior when present), and receives the frozen
  neutral value ``0``;
* no other missingness or imputation is accepted.

The small sealed artifact builder at the bottom of the file exists only to
demonstrate and audit this policy on bounded synthetic bridge rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


POLICY_CONTRACT = "frozen_strict_prior_donor_ability_policy_v2"
POLICY_STATUS = "policy_evidence_only_not_real_chain_ready"
OUTPUT_INDEX_CONTRACT = "content_derived_output_index_v2"

OBSERVED_VALUE_KIND = "observed_strict_prior"
IMPUTED_VALUE_KIND = "imputed_frozen_neutral_zero"
ABILITY_CLIP = 2.0
IMPUTED_NEUTRAL_VALUE = 0.0

CATEGORY_ABILITY_MAPPING = {
    "ott": "sg_ott",
    "app": "sg_app",
    "arg": "ag_arg",
    "putt": "ag_putt",
}

EVALUATOR_SOURCE_SHA256 = (
    "ad6e5d748400971482e317bb049c019f792c69bc080f2ef16d93507862e31040"
)
EVALUATOR_MANIFEST_SHA256 = (
    "3e9ac11496e46a08cbbe6764c34555156c89b639242b6b37f328bc8b542865a8"
)
COMPLETE_EVENT_SELECTION_MANIFEST_SHA256 = (
    "04a3543f65465303a4165dfa1940a97ac3b6e2bb70b90d06dcb7493235fef145"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_SOURCE_PATH = Path(__file__).resolve()
EVALUATOR_SOURCE_PATH = REPO_ROOT / "archive" / "evaluate_shot_quality_tilt.py"
EVALUATOR_MANIFEST_PATH = (
    REPO_ROOT / "permanent_data" / "shot_quality_tilt_eval_20260820_v2" / "manifest.json"
)
COMPLETE_EVENT_SELECTION_MANIFEST_PATH = (
    REPO_ROOT
    / "permanent_data"
    / "shot_quality_tilt_complete_events_20260821_v1"
    / "manifest.json"
)
V1_ARTIFACT_DIR = REPO_ROOT / "permanent_data" / "hole_sim_donor_ability_policy_v1"
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "permanent_data" / "hole_sim_donor_ability_policy_v2"
)

INPUT_COLUMNS = (
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
)
OUTPUT_COLUMNS = (
    "donor_ability_value_kind",
    "donor_ability_sg",
    "donor_ability_was_clipped",
    "donor_ability_policy_sha256",
    "donor_ability_row_sha256",
)
CANONICAL_IDENTIFIER_COLUMNS = (
    "donor_event_key",
    "donor_player_id",
    "target_event_key",
)
CSV_STRING_COLUMNS = {
    column: "string"
    for column in (
        "bridge_row_id",
        *CANONICAL_IDENTIFIER_COLUMNS,
        "category",
        "donor_ability_source_column",
        "donor_ability_value_kind",
        "donor_ability_policy_sha256",
        "donor_ability_row_sha256",
    )
}

BASE_ARTIFACT_FILES = (
    "policy.json",
    "bridge_fixture_input.csv",
    "bridge_fixture_evaluated.csv",
    "REPORT.md",
)
SEALED_ARTIFACT_FILES = (
    *BASE_ARTIFACT_FILES,
    "manifest.json",
    "manifest.sha256",
    "OUTPUT_SHA256.json",
)


class DonorAbilityPolicyError(ValueError):
    """Raised when donor-ability evidence violates the frozen contract."""


def sha256_file(path: Path) -> str:
    """Return the content digest of ``path`` without consulting metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _write_stable_json(path: Path, value: object) -> None:
    path.write_bytes(_stable_json_bytes(value))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DonorAbilityPolicyError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DonorAbilityPolicyError(f"{label} must be a JSON object")
    return value


def policy_identity() -> dict[str, object]:
    """Return the exact JSON-safe policy identity bound by every row."""

    return {
        "contract": POLICY_CONTRACT,
        "status": POLICY_STATUS,
        "research_only": True,
        "production_enabled": False,
        "real_chain_ready": False,
        "category_ability_mapping": dict(CATEGORY_ABILITY_MAPPING),
        "availability_semantics": {
            "preserve_donor_ability_available": True,
            "available_requires_finite_raw_value": True,
            "unavailable_requires_null_raw_value": True,
            "unavailable_preserves_nullable_actual_skill_date": True,
            "undeclared_missing": "fail_closed",
        },
        "identifier_admission": {
            "required": list(CANONICAL_IDENTIFIER_COLUMNS),
            "representation": "populated_string",
            "canonical_form": "NFKC_normalized_and_edge_trimmed_on_arrival",
            "missing_blank_nontext_or_noncanonical": "fail_closed",
        },
        "missing_predicate": {
            "expression": (
                "donor_ability_available == false AND "
                "donor_ability_raw_strict_prior IS NULL"
            ),
            "partial_missing": "fail_closed",
            "skill_date": (
                "preserve nullable actual donor_skill_date; never erase or invent"
            ),
            "dated_missing_cutoff": "donor_skill_date < donor_event_start",
            "undated_missing": "allowed only as the declared no-bridge value",
            "invented_skill_date": "forbidden",
        },
        "observed_value": {
            "value_kind": OBSERVED_VALUE_KIND,
            "raw_source": "strict_prior_category_ability",
            "applied_value": "clip(raw_strict_prior_ability,-2,+2)",
            "clip_min": -ABILITY_CLIP,
            "clip_max": ABILITY_CLIP,
            "required_cutoff": (
                "donor_skill_date < donor_event_start < target_event_start"
            ),
        },
        "imputed_value": {
            "value_kind": IMPUTED_VALUE_KIND,
            "applied_value": IMPUTED_NEUTRAL_VALUE,
            "skill_date": "preserve_actual_nullable_value",
            "nonnull_skill_date_cutoff": "donor_skill_date < donor_event_start",
            "null_skill_date_allowed": True,
            "nonzero_imputation": "fail_closed",
        },
        "beta_zero_identity": {
            "beta": 0.0,
            "donor_log_weight_delta": 0.0,
            "donor_weight_multiplier": 1.0,
            "independent_of_availability_and_applied_value": True,
        },
        "bound_evidence_sha256": {
            "evaluator_source": EVALUATOR_SOURCE_SHA256,
            "evaluator_manifest": EVALUATOR_MANIFEST_SHA256,
            "complete_event_selection_manifest": (
                COMPLETE_EVENT_SELECTION_MANIFEST_SHA256
            ),
        },
    }


def policy_payload() -> dict[str, object]:
    identity = policy_identity()
    return {
        "policy": identity,
        "policy_identity_sha256": _stable_sha256(identity),
    }


def validate_policy_payload(value: object, *, label: str = "policy") -> None:
    """Require exact policy identity; missing, extra, or changed values fail."""

    if not isinstance(value, Mapping):
        raise DonorAbilityPolicyError(f"{label} must be an object")
    observed = dict(value)
    expected = policy_payload()
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        changed = sorted(
            key
            for key in set(expected).intersection(observed)
            if observed[key] != expected[key]
        )
        raise DonorAbilityPolicyError(
            f"{label} differs from {POLICY_CONTRACT}: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _require_content_sha(path: Path, expected: str, *, label: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise DonorAbilityPolicyError(f"missing bound {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise DonorAbilityPolicyError(
            f"wrong {label} sha256: expected {expected}, observed {observed}"
        )
    return observed


def validate_bound_evidence(
    *,
    evaluator_source: Path = EVALUATOR_SOURCE_PATH,
    evaluator_manifest: Path = EVALUATOR_MANIFEST_PATH,
    complete_event_selection_manifest: Path = COMPLETE_EVENT_SELECTION_MANIFEST_PATH,
) -> dict[str, dict[str, object]]:
    """Validate the three immutable evidence inputs and selected semantics."""

    source_sha = _require_content_sha(
        evaluator_source,
        EVALUATOR_SOURCE_SHA256,
        label="evaluator source",
    )
    evaluator_manifest_sha = _require_content_sha(
        evaluator_manifest,
        EVALUATOR_MANIFEST_SHA256,
        label="evaluator manifest",
    )
    selection_manifest_sha = _require_content_sha(
        complete_event_selection_manifest,
        COMPLETE_EVENT_SELECTION_MANIFEST_SHA256,
        label="complete-event selection manifest",
    )

    evaluator = _read_json(evaluator_manifest, label="evaluator manifest")
    if evaluator.get("contract") != "strict_prior_whole_reward_atom_tilt_v1":
        raise DonorAbilityPolicyError("changed evaluator manifest contract")
    if evaluator.get("source_sha256") != source_sha:
        raise DonorAbilityPolicyError("evaluator manifest source hash mismatch")
    if evaluator.get("source_sha256_at_end") != source_sha:
        raise DonorAbilityPolicyError("evaluator source changed during bound run")
    run_config = evaluator.get("run_config")
    if not isinstance(run_config, Mapping):
        raise DonorAbilityPolicyError("evaluator run_config missing")
    if float(run_config.get("ability_clip", float("nan"))) != ABILITY_CLIP:
        raise DonorAbilityPolicyError("evaluator ability clip differs from 2")
    beta_grid = evaluator.get("beta_grid")
    if not isinstance(beta_grid, list) or not any(
        isinstance(value, (int, float)) and float(value) == 0.0
        for value in beta_grid
    ):
        raise DonorAbilityPolicyError("evaluator beta grid lacks exact beta zero")

    selection = _read_json(
        complete_event_selection_manifest,
        label="complete-event selection manifest",
    )
    if selection.get("contract") != "nested_quality_tilt_complete_event_gate_v1":
        raise DonorAbilityPolicyError("changed complete-event selection contract")
    if selection.get("research_only") is not True:
        raise DonorAbilityPolicyError("complete-event selection is not research-only")
    if selection.get("production_enabled") is not False:
        raise DonorAbilityPolicyError("complete-event selection enables production")

    return {
        "evaluator_source": {
            "descriptor": "archive/evaluate_shot_quality_tilt.py",
            "sha256": source_sha,
        },
        "evaluator_manifest": {
            "descriptor": (
                "permanent_data/shot_quality_tilt_eval_20260820_v2/manifest.json"
            ),
            "sha256": evaluator_manifest_sha,
        },
        "complete_event_selection_manifest": {
            "descriptor": (
                "permanent_data/shot_quality_tilt_complete_events_20260821_v1/"
                "manifest.json"
            ),
            "sha256": selection_manifest_sha,
        },
    }


def donor_weight_multiplier(beta: float, donor_ability_sg: float, reward: float) -> float:
    """Return the frozen evaluator's donor term, with exact beta-zero identity."""

    beta_value = float(beta)
    ability_value = float(donor_ability_sg)
    reward_value = float(reward)
    if not all(math.isfinite(value) for value in (beta_value, ability_value, reward_value)):
        raise DonorAbilityPolicyError("beta, donor ability, and reward must be finite")
    if ability_value < -ABILITY_CLIP or ability_value > ABILITY_CLIP:
        raise DonorAbilityPolicyError("applied donor ability is outside [-2, 2]")
    if beta_value == 0.0:
        return 1.0
    return math.exp(-beta_value * ability_value * reward_value)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require_canonical_identifier(value: object, *, label: str) -> str:
    """Require an already canonical NFKC, edge-trimmed identity string."""

    if _is_missing(value) or not isinstance(value, str):
        raise DonorAbilityPolicyError(f"{label} must be a populated string")
    canonical = unicodedata.normalize("NFKC", value).strip()
    if not canonical:
        raise DonorAbilityPolicyError(f"{label} must be nonblank")
    if canonical != value:
        raise DonorAbilityPolicyError(
            f"{label} must already be canonical NFKC text without edge whitespace"
        )
    return canonical


def _require_bool(value: object, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise DonorAbilityPolicyError(f"{label} must be an explicit boolean")


def _require_timestamp(value: object, *, label: str) -> pd.Timestamp:
    if _is_missing(value):
        raise DonorAbilityPolicyError(f"{label} is required")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise DonorAbilityPolicyError(f"{label} is not a valid timestamp")
    return pd.Timestamp(parsed)


def _optional_timestamp(value: object, *, label: str) -> pd.Timestamp | None:
    if _is_missing(value):
        return None
    return _require_timestamp(value, label=label)


def _canonical_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _canonical_scalar(value: object) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return _canonical_timestamp(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise DonorAbilityPolicyError("non-finite value cannot be hashed")
        return number
    return str(value)


def _row_hash_payload(
    row: Mapping[str, object],
    *,
    skill_date: pd.Timestamp | None,
    donor_event_start: pd.Timestamp,
    target_event_start: pd.Timestamp,
    available: bool,
    raw_ability: float | None,
    value_kind: str,
    applied_ability: float,
    was_clipped: bool,
) -> dict[str, object]:
    return {
        "bridge_row_id": str(row["bridge_row_id"]),
        "category": str(row["category"]),
        "donor_player_id": str(row["donor_player_id"]),
        "donor_event_key": str(row["donor_event_key"]),
        "donor_event_start": _canonical_timestamp(donor_event_start),
        "target_event_key": str(row["target_event_key"]),
        "target_event_start": _canonical_timestamp(target_event_start),
        "donor_skill_date": _canonical_timestamp(skill_date),
        "donor_ability_source_column": str(row["donor_ability_source_column"]),
        "donor_ability_available": available,
        "donor_ability_raw_strict_prior": raw_ability,
        "donor_ability_value_kind": value_kind,
        "donor_ability_sg": applied_ability,
        "donor_ability_was_clipped": was_clipped,
        "donor_ability_policy_sha256": policy_payload()[
            "policy_identity_sha256"
        ],
    }


def _validate_input_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise DonorAbilityPolicyError(f"donor ability frame missing columns: {missing}")
    if frame.empty:
        raise DonorAbilityPolicyError("donor ability frame must not be empty")
    identifiers = frame["bridge_row_id"].astype("string")
    if identifiers.isna().any() or identifiers.str.strip().eq("").any():
        raise DonorAbilityPolicyError("bridge_row_id must be populated")
    if identifiers.duplicated().any():
        raise DonorAbilityPolicyError("bridge_row_id must be unique")


def apply_frozen_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact policy to raw bridge rows and bind every result by hash."""

    if not isinstance(frame, pd.DataFrame):
        raise DonorAbilityPolicyError("donor ability input must be a DataFrame")
    _validate_input_columns(frame)
    preexisting = [column for column in OUTPUT_COLUMNS if column in frame.columns]
    if preexisting:
        raise DonorAbilityPolicyError(
            f"raw donor ability frame already contains policy outputs: {preexisting}"
        )

    output = frame.copy(deep=True).reset_index(drop=True)
    kinds: list[str] = []
    applied_values: list[float] = []
    clipped_values: list[bool] = []
    row_hashes: list[str] = []
    policy_sha = str(policy_payload()["policy_identity_sha256"])

    for position, row in output.iterrows():
        label = f"row {position} ({row['bridge_row_id']})"
        for identifier_column in CANONICAL_IDENTIFIER_COLUMNS:
            _require_canonical_identifier(
                row[identifier_column],
                label=f"{label} {identifier_column}",
            )
        category = str(row["category"])
        if category not in CATEGORY_ABILITY_MAPPING:
            raise DonorAbilityPolicyError(f"{label} has unsupported category {category!r}")
        expected_source = CATEGORY_ABILITY_MAPPING[category]
        if str(row["donor_ability_source_column"]) != expected_source:
            raise DonorAbilityPolicyError(
                f"{label} category mapping must use {expected_source!r}"
            )

        available = _require_bool(
            row["donor_ability_available"],
            label=f"{label} donor_ability_available",
        )
        donor_event_start = _require_timestamp(
            row["donor_event_start"],
            label=f"{label} donor_event_start",
        )
        target_event_start = _require_timestamp(
            row["target_event_start"],
            label=f"{label} target_event_start",
        )
        if not donor_event_start < target_event_start:
            raise DonorAbilityPolicyError(
                f"{label} requires donor_event_start < target_event_start"
            )

        raw_value = row["donor_ability_raw_strict_prior"]
        raw_missing = _is_missing(raw_value)
        skill_date = _optional_timestamp(
            row["donor_skill_date"],
            label=f"{label} donor_skill_date",
        )

        if available:
            if raw_missing:
                raise DonorAbilityPolicyError(
                    f"{label} has undeclared missing donor ability"
                )
            try:
                raw_ability = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise DonorAbilityPolicyError(
                    f"{label} donor ability must be numeric"
                ) from exc
            if not math.isfinite(raw_ability):
                raise DonorAbilityPolicyError(
                    f"{label} donor ability must be finite"
                )
            if skill_date is None:
                raise DonorAbilityPolicyError(
                    f"{label} observed ability requires its true skill date"
                )
            if not skill_date < donor_event_start:
                raise DonorAbilityPolicyError(
                    f"{label} requires donor_skill_date < donor_event_start"
                )
            value_kind = OBSERVED_VALUE_KIND
            applied = float(np.clip(raw_ability, -ABILITY_CLIP, ABILITY_CLIP))
            was_clipped = applied != raw_ability
        else:
            if not raw_missing:
                raise DonorAbilityPolicyError(
                    f"{label} marks a present raw ability unavailable"
                )
            if skill_date is not None and not skill_date < donor_event_start:
                raise DonorAbilityPolicyError(
                    f"{label} dated missing ability requires "
                    "donor_skill_date < donor_event_start"
                )
            raw_ability = None
            value_kind = IMPUTED_VALUE_KIND
            applied = IMPUTED_NEUTRAL_VALUE
            was_clipped = False

        row_payload = _row_hash_payload(
            row,
            skill_date=skill_date,
            donor_event_start=donor_event_start,
            target_event_start=target_event_start,
            available=available,
            raw_ability=raw_ability,
            value_kind=value_kind,
            applied_ability=applied,
            was_clipped=was_clipped,
        )
        kinds.append(value_kind)
        applied_values.append(applied)
        clipped_values.append(was_clipped)
        row_hashes.append(_stable_sha256(row_payload))

    output["donor_ability_value_kind"] = kinds
    output["donor_ability_sg"] = np.asarray(applied_values, dtype=float)
    output["donor_ability_was_clipped"] = np.asarray(clipped_values, dtype=bool)
    output["donor_ability_policy_sha256"] = policy_sha
    output["donor_ability_row_sha256"] = row_hashes
    return output


def _canonical_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {column: _canonical_scalar(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    ]


def validate_applied_frame(frame: pd.DataFrame) -> None:
    """Recompute every policy output and reject any altered value or hash."""

    if not isinstance(frame, pd.DataFrame):
        raise DonorAbilityPolicyError("applied donor ability value must be a DataFrame")
    missing = [column for column in (*INPUT_COLUMNS, *OUTPUT_COLUMNS) if column not in frame]
    if missing:
        raise DonorAbilityPolicyError(f"applied donor ability frame missing: {missing}")
    imputed = frame["donor_ability_value_kind"].eq(IMPUTED_VALUE_KIND)
    if not pd.to_numeric(
        frame.loc[imputed, "donor_ability_sg"], errors="coerce"
    ).eq(0.0).all():
        raise DonorAbilityPolicyError("nonzero missing-ability imputation")
    if frame.loc[imputed, "donor_ability_available"].astype(bool).any():
        raise DonorAbilityPolicyError("imputation changed honest availability")

    raw = frame.drop(columns=list(OUTPUT_COLUMNS))
    expected = apply_frozen_policy(raw)
    observed = frame.loc[:, expected.columns]
    if _canonical_records(observed) != _canonical_records(expected):
        raise DonorAbilityPolicyError("applied donor ability values or hashes changed")


def synthetic_bridge_fixture() -> pd.DataFrame:
    """Return a bounded real-shaped fixture covering all policy branches."""

    rows = [
        {
            "bridge_row_id": "observed_inside_clip",
            "category": "ott",
            "donor_player_id": "33948",
            "donor_event_key": "pga:R2020034",
            "donor_event_start": "2020-06-25T00:00:00Z",
            "target_event_key": "pga:R2026034",
            "target_event_start": "2026-08-20T00:00:00Z",
            "donor_skill_date": "2020-06-24T00:00:00Z",
            "donor_ability_source_column": "sg_ott",
            "donor_ability_available": True,
            "donor_ability_raw_strict_prior": 1.25,
        },
        {
            "bridge_row_id": "observed_clip_high",
            "category": "app",
            "donor_player_id": "22371",
            "donor_event_key": "pga:R2021034",
            "donor_event_start": "2021-08-26T00:00:00Z",
            "target_event_key": "pga:R2026034",
            "target_event_start": "2026-08-20T00:00:00Z",
            "donor_skill_date": "2021-08-25T00:00:00Z",
            "donor_ability_source_column": "sg_app",
            "donor_ability_available": True,
            "donor_ability_raw_strict_prior": 3.75,
        },
        {
            "bridge_row_id": "observed_clip_low",
            "category": "arg",
            "donor_player_id": "34076",
            "donor_event_key": "pga:R2022034",
            "donor_event_start": "2022-08-25T00:00:00Z",
            "target_event_key": "pga:R2026034",
            "target_event_start": "2026-08-20T00:00:00Z",
            "donor_skill_date": "2022-08-24T00:00:00Z",
            "donor_ability_source_column": "ag_arg",
            "donor_ability_available": True,
            "donor_ability_raw_strict_prior": -2.75,
        },
        {
            "bridge_row_id": "missing_dated_neutral_zero",
            "category": "putt",
            "donor_player_id": "45609",
            "donor_event_key": "pga:R2023034",
            "donor_event_start": "2023-08-24T00:00:00Z",
            "target_event_key": "pga:R2026034",
            "target_event_start": "2026-08-20T00:00:00Z",
            "donor_skill_date": "2023-08-23T00:00:00Z",
            "donor_ability_source_column": "ag_putt",
            "donor_ability_available": False,
            "donor_ability_raw_strict_prior": None,
        },
        {
            "bridge_row_id": "missing_undated_neutral_zero",
            "category": "putt",
            "donor_player_id": "29535",
            "donor_event_key": "pga:R2024034",
            "donor_event_start": "2024-08-22T00:00:00Z",
            "target_event_key": "pga:R2026034",
            "target_event_start": "2026-08-20T00:00:00Z",
            "donor_skill_date": None,
            "donor_ability_source_column": "ag_putt",
            "donor_ability_available": False,
            "donor_ability_raw_strict_prior": None,
        },
    ]
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def _artifact_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def _require_exact_recursive_file_set(
    directory: Path,
    expected_names: Sequence[str],
) -> list[str]:
    """Reject any unsealed file, directory, or link anywhere below the root."""

    directory = Path(directory)
    entries = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
    }
    expected = sorted(expected_names)
    observed = sorted(entries)
    if observed != expected:
        raise DonorAbilityPolicyError(
            "sealed artifact recursive output closure changed: "
            f"expected={expected}, observed={observed}"
        )
    invalid = sorted(
        name
        for name, path in entries.items()
        if not path.is_file() or path.is_symlink()
    )
    if invalid:
        raise DonorAbilityPolicyError(
            f"sealed artifact contains non-file or linked entries: {invalid}"
        )
    return expected


def _report_text() -> str:
    return """# Frozen donor-ability policy evidence

Status: **policy evidence only; not real-chain readiness**.

This bounded real-shaped synthetic bridge fixture demonstrates the archive
evaluator's frozen donor-ability semantics. Honest availability is retained. Strictly-prior
observations use their raw category ability clipped to `[-2, 2]`; genuinely
missing observations receive exactly `0`. Their true nullable skill date is
preserved: a present date must precede the donor event, while an undated
no-bridge row remains null. Beta zero therefore remains an exact multiplicative
identity for both value kinds.

The artifact binds the historical evaluator source, its sealed evaluation
manifest, and the complete-event selection manifest by SHA-256. It does not
modify or authorize the streaming chain, full-chain runner, production
simulator, or transition-mean anchor adapter.
"""


def build_sealed_artifact(output_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, object]:
    """Build a new bounded artifact; an existing target is never overwritten."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise DonorAbilityPolicyError(
            f"refusing to overwrite existing artifact directory: {output_dir}"
        )

    bound_start = validate_bound_evidence()
    policy_source_sha_start = sha256_file(POLICY_SOURCE_PATH)
    output_dir.mkdir(parents=True, exist_ok=False)

    _write_stable_json(output_dir / "policy.json", policy_payload())
    fixture = synthetic_bridge_fixture()
    evaluated = apply_frozen_policy(fixture)
    fixture.to_csv(
        output_dir / "bridge_fixture_input.csv",
        index=False,
        lineterminator="\n",
    )
    evaluated.to_csv(
        output_dir / "bridge_fixture_evaluated.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "REPORT.md").write_text(
        _report_text(),
        encoding="utf-8",
        newline="\n",
    )

    reloaded = pd.read_csv(
        output_dir / "bridge_fixture_evaluated.csv",
        dtype=CSV_STRING_COLUMNS,
    )
    validate_applied_frame(reloaded)
    artifact_start = _artifact_hashes(output_dir, BASE_ARTIFACT_FILES)

    bound_end = validate_bound_evidence()
    policy_source_sha_end = sha256_file(POLICY_SOURCE_PATH)
    artifact_end = _artifact_hashes(output_dir, BASE_ARTIFACT_FILES)
    if bound_start != bound_end:
        raise DonorAbilityPolicyError("bound evidence changed during artifact build")
    if policy_source_sha_start != policy_source_sha_end:
        raise DonorAbilityPolicyError("policy source changed during artifact build")
    if artifact_start != artifact_end:
        raise DonorAbilityPolicyError("artifact outputs changed during validation")

    manifest = {
        "contract": POLICY_CONTRACT,
        "status": POLICY_STATUS,
        "research_only": True,
        "production_enabled": False,
        "real_chain_ready": False,
        "fixture_kind": "bounded_real_shaped_synthetic_audited_bridge",
        "fixture_rows": int(len(evaluated)),
        "observed_rows": int(
            evaluated["donor_ability_value_kind"].eq(OBSERVED_VALUE_KIND).sum()
        ),
        "imputed_rows": int(
            evaluated["donor_ability_value_kind"].eq(IMPUTED_VALUE_KIND).sum()
        ),
        "imputed_rows_with_preserved_skill_date": int(
            (
                evaluated["donor_ability_value_kind"].eq(IMPUTED_VALUE_KIND)
                & evaluated["donor_skill_date"].notna()
            ).sum()
        ),
        "imputed_rows_without_skill_date": int(
            (
                evaluated["donor_ability_value_kind"].eq(IMPUTED_VALUE_KIND)
                & evaluated["donor_skill_date"].isna()
            ).sum()
        ),
        "policy_identity_sha256": policy_payload()["policy_identity_sha256"],
        "policy_source_sha256_start": policy_source_sha_start,
        "policy_source_sha256_end": policy_source_sha_end,
        "policy_source_start_equals_end": (
            policy_source_sha_start == policy_source_sha_end
        ),
        "bound_evidence_start": bound_start,
        "bound_evidence_end": bound_end,
        "bound_evidence_start_equals_end": bound_start == bound_end,
        "artifact_start": artifact_start,
        "artifact_end": artifact_end,
        "artifact_start_equals_end": artifact_start == artifact_end,
        "output_index": "OUTPUT_SHA256.json",
        "output_index_policy": (
            "exact recursive output closure; directories and links forbidden; "
            "content hashes and byte sizes cover every sealed file except the "
            "index itself"
        ),
    }
    _write_stable_json(output_dir / "manifest.json", manifest)
    manifest_sha = sha256_file(output_dir / "manifest.json")
    (output_dir / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n",
        encoding="ascii",
        newline="\n",
    )

    indexed_names = _require_exact_recursive_file_set(
        output_dir,
        [name for name in SEALED_ARTIFACT_FILES if name != "OUTPUT_SHA256.json"],
    )
    index = {
        "contract": OUTPUT_INDEX_CONTRACT,
        "content_derived": True,
        "index_self_excluded": True,
        "recursive_closure": "exact_files_only_no_directories_or_links",
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "size_bytes": int((output_dir / name).stat().st_size),
            }
            for name in indexed_names
        },
    }
    _write_stable_json(output_dir / "OUTPUT_SHA256.json", index)
    summary = validate_sealed_artifact(output_dir)
    return summary


def validate_sealed_artifact(output_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, object]:
    """Validate the full sealed artifact, dependencies, rows, and output index."""

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise DonorAbilityPolicyError(f"sealed artifact directory missing: {output_dir}")
    recursive_files = _require_exact_recursive_file_set(
        output_dir,
        SEALED_ARTIFACT_FILES,
    )

    index = _read_json(output_dir / "OUTPUT_SHA256.json", label="output index")
    expected_index_keys = {
        "contract",
        "content_derived",
        "index_self_excluded",
        "recursive_closure",
        "files",
    }
    if set(index) != expected_index_keys:
        raise DonorAbilityPolicyError("output index fields changed")
    if index.get("contract") != OUTPUT_INDEX_CONTRACT:
        raise DonorAbilityPolicyError("output index contract changed")
    if index.get("content_derived") is not True:
        raise DonorAbilityPolicyError("output index is not content-derived")
    if index.get("index_self_excluded") is not True:
        raise DonorAbilityPolicyError("output index self policy changed")
    if index.get("recursive_closure") != "exact_files_only_no_directories_or_links":
        raise DonorAbilityPolicyError("output index recursive closure changed")
    files = index.get("files")
    expected_indexed = sorted(
        name for name in recursive_files if name != "OUTPUT_SHA256.json"
    )
    if not isinstance(files, Mapping) or sorted(files) != expected_indexed:
        raise DonorAbilityPolicyError("output index file set changed")
    for name in expected_indexed:
        entry = files[name]
        if not isinstance(entry, Mapping):
            raise DonorAbilityPolicyError(f"invalid output index entry: {name}")
        if set(entry) != {"sha256", "size_bytes"}:
            raise DonorAbilityPolicyError(f"output index entry fields changed: {name}")
        actual_sha = sha256_file(output_dir / name)
        actual_size = int((output_dir / name).stat().st_size)
        if entry.get("sha256") != actual_sha or entry.get("size_bytes") != actual_size:
            raise DonorAbilityPolicyError(f"output changed after sealing: {name}")

    detached = (output_dir / "manifest.sha256").read_text(encoding="ascii")
    expected_detached = f"{sha256_file(output_dir / 'manifest.json')}  manifest.json\n"
    if detached != expected_detached:
        raise DonorAbilityPolicyError("detached manifest checksum changed")

    policy = _read_json(output_dir / "policy.json", label="policy")
    validate_policy_payload(policy)
    manifest = _read_json(output_dir / "manifest.json", label="manifest")
    if manifest.get("contract") != POLICY_CONTRACT:
        raise DonorAbilityPolicyError("manifest contract changed")
    if manifest.get("status") != POLICY_STATUS:
        raise DonorAbilityPolicyError("manifest readiness label changed")
    if manifest.get("research_only") is not True:
        raise DonorAbilityPolicyError("manifest is not research-only")
    if manifest.get("production_enabled") is not False:
        raise DonorAbilityPolicyError("manifest enables production")
    if manifest.get("real_chain_ready") is not False:
        raise DonorAbilityPolicyError("manifest claims real-chain readiness")
    if manifest.get("fixture_kind") != (
        "bounded_real_shaped_synthetic_audited_bridge"
    ):
        raise DonorAbilityPolicyError("manifest fixture kind changed")
    if manifest.get("output_index") != "OUTPUT_SHA256.json":
        raise DonorAbilityPolicyError("manifest output index path changed")
    if manifest.get("output_index_policy") != (
        "exact recursive output closure; directories and links forbidden; "
        "content hashes and byte sizes cover every sealed file except the "
        "index itself"
    ):
        raise DonorAbilityPolicyError("manifest output index policy changed")
    if manifest.get("policy_identity_sha256") != policy_payload()[
        "policy_identity_sha256"
    ]:
        raise DonorAbilityPolicyError("manifest policy identity changed")

    policy_source_sha = sha256_file(POLICY_SOURCE_PATH)
    if manifest.get("policy_source_sha256_start") != policy_source_sha:
        raise DonorAbilityPolicyError("policy source start hash changed")
    if manifest.get("policy_source_sha256_end") != policy_source_sha:
        raise DonorAbilityPolicyError("policy source end hash changed")
    if manifest.get("policy_source_start_equals_end") is not True:
        raise DonorAbilityPolicyError("policy source start/end gate is false")

    bound = validate_bound_evidence()
    if manifest.get("bound_evidence_start") != bound:
        raise DonorAbilityPolicyError("bound evidence start values changed")
    if manifest.get("bound_evidence_end") != bound:
        raise DonorAbilityPolicyError("bound evidence end values changed")
    if manifest.get("bound_evidence_start_equals_end") is not True:
        raise DonorAbilityPolicyError("bound evidence start/end gate is false")

    artifact_actual = _artifact_hashes(output_dir, BASE_ARTIFACT_FILES)
    if manifest.get("artifact_start") != artifact_actual:
        raise DonorAbilityPolicyError("artifact start hashes changed")
    if manifest.get("artifact_end") != artifact_actual:
        raise DonorAbilityPolicyError("artifact end hashes changed")
    if manifest.get("artifact_start_equals_end") is not True:
        raise DonorAbilityPolicyError("artifact start/end gate is false")

    input_frame = pd.read_csv(
        output_dir / "bridge_fixture_input.csv",
        dtype=CSV_STRING_COLUMNS,
    )
    evaluated_frame = pd.read_csv(
        output_dir / "bridge_fixture_evaluated.csv",
        dtype=CSV_STRING_COLUMNS,
    )
    observed_rows = int(
        evaluated_frame["donor_ability_value_kind"].eq(OBSERVED_VALUE_KIND).sum()
    )
    imputed_mask = evaluated_frame["donor_ability_value_kind"].eq(
        IMPUTED_VALUE_KIND
    )
    imputed_rows = int(imputed_mask.sum())
    imputed_dated = int(
        (imputed_mask & evaluated_frame["donor_skill_date"].notna()).sum()
    )
    imputed_undated = int(
        (imputed_mask & evaluated_frame["donor_skill_date"].isna()).sum()
    )
    expected_counts = {
        "fixture_rows": int(len(evaluated_frame)),
        "observed_rows": observed_rows,
        "imputed_rows": imputed_rows,
        "imputed_rows_with_preserved_skill_date": imputed_dated,
        "imputed_rows_without_skill_date": imputed_undated,
    }
    changed_counts = {
        key: (manifest.get(key), expected)
        for key, expected in expected_counts.items()
        if manifest.get(key) != expected
    }
    if changed_counts:
        raise DonorAbilityPolicyError(
            f"manifest fixture counts changed: {changed_counts}"
        )
    expected_frame = apply_frozen_policy(input_frame)
    validate_applied_frame(evaluated_frame)
    if _canonical_records(evaluated_frame) != _canonical_records(expected_frame):
        raise DonorAbilityPolicyError("sealed evaluated bridge differs from raw fixture")

    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    if "policy evidence only; not real-chain readiness" not in report:
        raise DonorAbilityPolicyError("research-only readiness label missing from report")

    return {
        "status": POLICY_STATUS,
        "rows": int(len(evaluated_frame)),
        "policy_identity_sha256": policy_payload()["policy_identity_sha256"],
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "output_index_sha256": sha256_file(output_dir / "OUTPUT_SHA256.json"),
        "artifact_start_equals_end": True,
        "bound_evidence_start_equals_end": True,
        "policy_source_start_equals_end": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.build:
        result = build_sealed_artifact(args.output_dir)
    else:
        result = validate_sealed_artifact(args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

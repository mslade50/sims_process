"""Fail-closed provenance checks for frozen base-skill downstream evaluators.

The compact finalist manifest is only the first link.  A valid handoff must
also bind the discovery selection manifest, the completion prediction panel,
the completion manifest, and both frozen candidate catalogs.  This module is
read-only and deliberately does not load or score sealed outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from archive import evaluate_complete_base_skill_market_candidates as discovery


FINALIST_CONTRACT = "base_skill_frozen_finalist_v1"
SELECTION_CONTRACT = "base_skill_discovery_selection_v1"
DISCOVERY_END = pd.Timestamp("2022-12-31")
STRICT_PT_CONTROL = "current_architecture_pt_v2_approx_control"
SELECTION_REQUIRED_INPUTS = frozenset(
    {
        "predictions",
        "candidate_catalog",
        "frozen_candidate_catalog",
        "candidate_manifest",
        "normalized_quotes",
        "feature_snapshots",
        "evaluation_cache_manifest",
        "coverage_by_candidate_year",
        "core_bridge",
    }
)
COMPLETION_REQUIRED_INPUTS = frozenset(
    {
        "snapshots",
        "all_tour_db",
        "clean_img",
        "round_panel",
        "weather_coefficients",
        "regression_envelope_gate_audit",
        "regression_envelope_gate_candidates",
        "regression_envelope_gate_folds",
        "core_bridge",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(parent: Path, value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else parent / path).resolve()


def _required_bool(payload: Mapping[str, Any], key: str, expected: bool) -> None:
    value = payload.get(key)
    if not isinstance(value, bool) or value is not expected:
        raise ValueError(f"manifest requires {key}={str(expected).lower()}; got {value!r}")


def _selection_cutoff(payload: Mapping[str, Any]) -> pd.Timestamp:
    for key in (
        "selection_window_end",
        "selection_period_end",
        "selection_cutoff",
        "discovery_end",
    ):
        parsed = pd.to_datetime(payload.get(key), errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    raise ValueError("manifest has no parseable discovery selection cutoff")


def _selection_name(selection: Mapping[str, Any]) -> str:
    finalists = selection.get("finalists")
    if isinstance(finalists, list) and finalists:
        value = finalists[0]
        if isinstance(value, Mapping):
            value = value.get("candidate") or value.get("name")
        if value:
            return str(value)
    value = selection.get("selected_candidate") or selection.get("finalist_name")
    if value:
        return str(value)
    value = selection.get("diagnostic_challenger")
    if isinstance(value, Mapping):
        value = value.get("candidate") or value.get("name")
    if value:
        return str(value)
    raise ValueError("selection manifest declares neither finalist nor diagnostic challenger")


def _compact_name(compact: Mapping[str, Any]) -> str:
    value = (
        compact.get("finalist_name")
        or compact.get("selected_model_name")
        or compact.get("diagnostic_challenger")
    )
    if not value or str(value) == "None":
        raise ValueError("compact handoff declares neither finalist nor diagnostic challenger")
    return str(value)


def _verified_file(parent: Path, entry: object, label: str) -> tuple[Path, str]:
    if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"selection manifest lacks hashed input {label!r}")
    path = resolve_path(parent, entry["path"])
    expected = str(entry["sha256"]).lower()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"selection input hash mismatch: {label}")
    return path, actual


def _manifest_output(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    filename: str,
    expected_path: Path,
) -> str:
    entry = manifest.get("outputs", {}).get(filename)
    if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"completion manifest lacks required hashed output {filename!r}")
    path = resolve_path(manifest_path.parent, entry["path"])
    if path != expected_path.resolve():
        raise ValueError(f"completion manifest output path mismatch for {filename}")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(entry["sha256"]).lower()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"completion manifest output hash mismatch for {filename}")
    return actual


def _verify_selection_artifacts(
    selection_path: Path, selection: Mapping[str, Any]
) -> dict[Path, Mapping[str, Any]]:
    verified: dict[Path, Mapping[str, Any]] = {}
    artifacts = selection.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("selection manifest artifacts must be a list")
    for entry in artifacts:
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("selection manifest contains a malformed artifact record")
        path = resolve_path(selection_path.parent, entry["path"])
        try:
            path.relative_to(selection_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"selection artifact escapes the selection directory: {path}"
            ) from exc
        if path in verified:
            raise ValueError(f"selection manifest duplicates artifact path: {path}")
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]).lower():
            raise ValueError(f"selection artifact hash mismatch: {path}")
        if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"selection artifact size mismatch: {path}")
        verified[path] = entry
    actual_files = {
        path.resolve()
        for path in selection_path.parent.rglob("*")
        if path.is_file() and path.resolve() != selection_path.resolve()
    }
    if set(verified) != actual_files:
        missing = sorted(str(path) for path in actual_files - set(verified))
        extras = sorted(str(path) for path in set(verified) - actual_files)
        raise ValueError(
            "selection artifact inventory is not exact; "
            f"unrecorded={missing}, nonexistent={extras}"
        )
    return verified


def _verify_recorded_files(
    parent: Path,
    records: Mapping[str, Any],
    label: str,
) -> dict[str, tuple[Path, str]]:
    """Rehash every named path/hash record in a provenance mapping."""

    if not isinstance(records, Mapping) or not records:
        raise ValueError(f"{label} must be a non-empty object")
    verified: dict[str, tuple[Path, str]] = {}
    for name, record in records.items():
        path, digest = _verified_file(parent, record, f"{label}.{name}")
        if record.get("bytes") is not None and path.stat().st_size != int(
            record["bytes"]
        ):
            raise ValueError(f"{label}.{name} byte-size mismatch")
        if record.get("size_bytes") is not None and path.stat().st_size != int(
            record["size_bytes"]
        ):
            raise ValueError(f"{label}.{name} byte-size mismatch")
        verified[str(name)] = (path, digest)
    return verified


def _verify_selection_execution_provenance(
    selection_path: Path,
    selection: Mapping[str, Any],
    artifacts: Mapping[Path, Mapping[str, Any]],
    expected_input_path_hashes: Mapping[str, str],
) -> dict[str, Any]:
    entry = selection.get("execution_provenance")
    if not isinstance(entry, Mapping) or not entry.get("path"):
        raise ValueError("selection manifest lacks execution provenance")
    _required_bool(entry, "verified_unchanged_before_complete_manifest", True)
    path = resolve_path(selection_path.parent, entry["path"])
    if path not in artifacts:
        raise ValueError("selection execution provenance is not a hashed artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("selection execution provenance is incomplete")
    _required_bool(payload, "verified_unchanged_before_complete_manifest", True)
    for field in ("start_utc", "end_utc", "pid"):
        if entry.get(field) != payload.get(field):
            raise ValueError(
                f"selection execution provenance {field} disagrees with manifest"
            )
    primary = resolve_path(selection_path.parent, payload.get("primary_source", ""))
    current_discovery = Path(discovery.__file__).resolve()
    if primary != current_discovery:
        raise ValueError("selection provenance does not bind the current discovery evaluator")
    sources = payload.get("sources")
    preserved_sources = payload.get("preserved_sources")
    inputs = payload.get("inputs")
    if not isinstance(sources, list) or not sources:
        raise ValueError("selection execution provenance has no source records")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("selection execution provenance has no input records")
    if not isinstance(preserved_sources, list) or len(preserved_sources) != len(sources):
        raise ValueError("selection provenance has no complete preserved-source records")

    expected_sources = tuple(
        dict.fromkeys(
            [
                current_discovery,
                *(Path(source).resolve() for source in discovery.PROVENANCE_CODE_PATHS),
            ]
        )
    )
    if len(sources) != len(expected_sources):
        raise ValueError(
            "selection provenance source set is incomplete or contains extras"
        )

    verified_sources: list[tuple[Path, str]] = []
    for index, record in enumerate(sources):
        if not isinstance(record, Mapping):
            raise ValueError("selection execution provenance has malformed source record")
        source, digest = _verified_file(
            selection_path.parent, record, f"selection provenance source[{index}]"
        )
        if record.get("bytes") is not None and source.stat().st_size != int(
            record["bytes"]
        ):
            raise ValueError("selection provenance source byte-size mismatch")
        preserved = path.parent / f"{index:02d}__{source.name}"
        if preserved not in artifacts or sha256_file(preserved) != digest:
            raise ValueError(
                f"selection provenance preserved source copy mismatch: {source.name}"
            )
        preserved_record = preserved_sources[index]
        if not isinstance(preserved_record, Mapping):
            raise ValueError("selection provenance has malformed preserved source record")
        recorded_copy, copy_digest = _verified_file(
            selection_path.parent,
            preserved_record,
            f"selection provenance preserved source[{index}]",
        )
        if recorded_copy != preserved or copy_digest != digest:
            raise ValueError(
                f"selection provenance preserved source record mismatch: {source.name}"
            )
        verified_sources.append((source, digest))
    observed_source_hashes = {source: digest for source, digest in verified_sources}
    expected_source_hashes = {
        source: sha256_file(source) for source in expected_sources
    }
    if (
        len(observed_source_hashes) != len(verified_sources)
        or observed_source_hashes != expected_source_hashes
    ):
        raise ValueError(
            "selection provenance does not bind the exact current evaluator source set"
        )

    verified_inputs: list[tuple[Path, str]] = []
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise ValueError("selection execution provenance has malformed input record")
        verified_inputs.append(
            _verified_file(
                selection_path.parent,
                record,
                f"selection provenance input[{index}]",
            )
        )
    observed_input_path_hashes = {
        str(input_path): digest for input_path, digest in verified_inputs
    }
    if len(observed_input_path_hashes) != len(verified_inputs):
        raise ValueError(
            "selection execution provenance contains duplicate resolved inputs"
        )
    if observed_input_path_hashes != dict(expected_input_path_hashes):
        raise ValueError(
            "selection execution provenance does not contain the exact "
            "discovery/cache/completed-study-authority input set"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sources_verified": len(verified_sources),
        "inputs_verified": len(verified_inputs),
        "verified_input_path_hashes": observed_input_path_hashes,
        "preserved_source_copies_verified": len(verified_sources),
        "verified_source_paths": [str(source) for source, _digest in verified_sources],
        "current_discovery_evaluator_sha256": sha256_file(current_discovery),
    }


def _verify_completion_execution_provenance(
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = source_manifest.get("execution_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("completion manifest lacks execution provenance")
    discovery_audit = discovery.validate_completion_manifest_provenance(
        source_manifest, source_manifest_path
    )
    for key in (
        "all_unchanged",
        "source_unchanged",
        "source_copy_unchanged",
        "source_copy_matches",
    ):
        _required_bool(provenance, key, True)

    source_before = provenance.get("source_before")
    source_after = provenance.get("source_after")
    copy_before = provenance.get("executed_source_copy_before")
    copy_after = provenance.get("executed_source_copy_after")
    expected_builder_hash = str(discovery.EXPECTED_BUILDER_SOURCE_SHA256).lower()
    for label, record in (
        ("completion source_before", source_before),
        ("completion source_after", source_after),
        ("completion executed source copy before", copy_before),
        ("completion executed source copy after", copy_after),
    ):
        _path, digest = _verified_file(source_manifest_path.parent, record, label)
        if digest != expected_builder_hash:
            raise ValueError(
                f"{label} is not the frozen completion builder source"
            )
    if source_before != source_after:
        raise ValueError("completion builder source before/after records disagree")
    if copy_before != copy_after:
        raise ValueError("completion preserved source before/after records disagree")
    if str(source_before.get("sha256", "")).lower() != str(
        copy_before.get("sha256", "")
    ).lower():
        raise ValueError("completion preserved source does not match builder source")

    before = provenance.get("inputs_before")
    after = provenance.get("inputs_after")
    before_verified = _verify_recorded_files(
        source_manifest_path.parent, before, "completion provenance inputs_before"
    )
    after_verified = _verify_recorded_files(
        source_manifest_path.parent, after, "completion provenance inputs_after"
    )
    if set(before_verified) != set(after_verified):
        raise ValueError("completion provenance input sets changed during execution")
    parsed_args = provenance.get("parsed_args")
    if not isinstance(parsed_args, Mapping) or "reuse_augmented" not in parsed_args:
        raise ValueError(
            "completion provenance lacks parsed_args.reuse_augmented applicability"
        )
    reuse_value = parsed_args.get("reuse_augmented")
    reuse_applicable = reuse_value is not None and str(reuse_value).strip() not in {
        "",
        "None",
        "null",
    }
    expected_inputs = set(COMPLETION_REQUIRED_INPUTS)
    if reuse_applicable:
        expected_inputs.add("reuse_augmented")
    if set(before_verified) != expected_inputs:
        missing = sorted(expected_inputs - set(before_verified))
        extras = sorted(set(before_verified) - expected_inputs)
        raise ValueError(
            "completion provenance input set differs from the frozen builder contract; "
            f"missing={missing}, extras={extras}"
        )
    unchanged = provenance.get("input_unchanged")
    if not isinstance(unchanged, Mapping) or set(unchanged) != set(before_verified):
        raise ValueError("completion provenance input_unchanged set is incomplete")
    for name in before_verified:
        if unchanged.get(name) is not True or before.get(name) != after.get(name):
            raise ValueError(f"completion provenance input changed: {name}")
    declared_inputs = source_manifest.get("inputs")
    if declared_inputs != after:
        raise ValueError("completion manifest inputs do not equal provenance inputs_after")
    return {
        "builder_source_path": str(resolve_path(source_manifest_path.parent, source_before["path"])),
        "builder_source_sha256": str(source_before["sha256"]).lower(),
        "executed_source_copy_path": str(resolve_path(source_manifest_path.parent, copy_before["path"])),
        "executed_source_copy_sha256": str(copy_before["sha256"]).lower(),
        "inputs_verified": len(before_verified),
        "reuse_augmented_applicable": reuse_applicable,
        "discovery_completion_validator": discovery_audit,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _stable_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip().casefold()
    if pd.notna(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value).strip().casefold()


def _verify_compact_core_values(
    compact_prediction_path: Path,
    source_prediction_path: Path,
    selected_name: str,
) -> dict[str, Any]:
    """Bind both compact cores to their exact rows in the hashed source panel."""

    compact = (
        pd.read_parquet(compact_prediction_path)
        if compact_prediction_path.suffix.casefold() in {".parquet", ".pq"}
        else pd.read_csv(compact_prediction_path)
    )
    compact_required = {
        "event_key",
        "dg_id",
        "candidate",
        "finalist_core",
        "control_core",
    }
    compact_missing = sorted(compact_required - set(compact.columns))
    if compact_missing:
        raise ValueError(f"compact prediction lacks core-binding columns: {compact_missing}")
    compact = compact.loc[compact["candidate"].astype(str).eq(selected_name)].copy()
    if compact.empty or compact.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("compact prediction has an empty or duplicate finalist menu")
    compact["event_key"] = compact["event_key"].astype(str).str.strip()
    compact["dg_id_key"] = compact["dg_id"].map(_stable_id)
    compact["finalist_core"] = pd.to_numeric(compact["finalist_core"], errors="coerce")
    compact["control_core"] = pd.to_numeric(compact["control_core"], errors="coerce")
    if (
        compact["event_key"].eq("").any()
        or compact["dg_id_key"].eq("").any()
        or not all(
            math.isfinite(float(value))
            for value in compact[["finalist_core", "control_core"]].to_numpy().ravel()
        )
    ):
        raise ValueError("compact prediction has invalid core keys or values")

    source_schema = set(pq.ParquetFile(source_prediction_path).schema.names)
    source_required = {
        "candidate",
        "canonical_event_key",
        "dg_id_key",
        "prediction",
    }
    source_missing = sorted(source_required - source_schema)
    if source_missing:
        raise ValueError(f"source prediction lacks core-binding columns: {source_missing}")
    source = pd.read_parquet(
        source_prediction_path,
        columns=list(source_required),
        filters=[("candidate", "in", [selected_name, STRICT_PT_CONTROL])],
    ).copy()
    source = source.loc[
        source["candidate"].astype(str).isin([selected_name, STRICT_PT_CONTROL])
    ].copy()
    source["event_key"] = source["canonical_event_key"].astype(str).str.strip()
    source["dg_id_key"] = source["dg_id_key"].map(_stable_id)
    source["prediction"] = pd.to_numeric(source["prediction"], errors="coerce")
    if source.duplicated(["candidate", "event_key", "dg_id_key"]).any():
        raise ValueError("source prediction core rows are ambiguous")
    expected = pd.concat(
        [
            compact[["event_key", "dg_id_key", "finalist_core"]]
            .rename(columns={"finalist_core": "prediction"})
            .assign(candidate=selected_name),
            compact[["event_key", "dg_id_key", "control_core"]]
            .rename(columns={"control_core": "prediction"})
            .assign(candidate=STRICT_PT_CONTROL),
        ],
        ignore_index=True,
    )
    comparison = expected.merge(
        source[["candidate", "event_key", "dg_id_key", "prediction"]],
        on=["candidate", "event_key", "dg_id_key"],
        how="outer",
        suffixes=("_compact", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("compact finalist/control menu differs from hashed source rows")
    error = (comparison["prediction_compact"] - comparison["prediction_source"]).abs()
    if error.isna().any() or error.max() > 1e-10:
        roles = comparison.loc[error.isna() | error.gt(1e-10), "candidate"].unique()
        raise ValueError(
            "compact finalist/control core differs from hashed source prediction: "
            + ", ".join(sorted(map(str, roles)))
        )
    control_error = error.loc[comparison["candidate"].eq(STRICT_PT_CONTROL)]
    finalist_error = error.loc[comparison["candidate"].eq(selected_name)]
    return {
        "compact_rows": int(len(compact)),
        "source_rows_verified": int(len(comparison)),
        "max_finalist_core_error": float(finalist_error.max()),
        "max_control_core_error": float(control_error.max()),
        "strict_pt_control_name": STRICT_PT_CONTROL,
    }


def selected_downstream_lambda(
    selection: Mapping[str, Any], selected_name: str
) -> float:
    """Return the discovery-selected exact-bridge sensitivity for one model."""

    payload = selection.get("downstream_preservation_sensitivity", {})
    if not isinstance(payload, Mapping):
        raise ValueError("selection manifest downstream sensitivity must be an object")
    if payload.get("lambda_one_is_isolation_default") is not True:
        raise ValueError("downstream sensitivity must preserve lambda=1 as isolation default")
    if payload.get("neutral_base_redefined") is not False:
        raise ValueError("downstream sensitivity may not redefine the neutral base")
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError("downstream sensitivity choices must be a list")
    matches = [
        choice
        for choice in choices
        if isinstance(choice, Mapping) and str(choice.get("candidate")) == selected_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selection manifest requires exactly one downstream-lambda choice for {selected_name!r}"
        )
    choice = matches[0]
    value = float(choice.get("selected_downstream_lambda"))
    if value not in {0.0, 0.5, 1.0}:
        raise ValueError("discovery-selected downstream lambda must be 0, 0.5, or 1")
    cutoff = _selection_cutoff(choice)
    if cutoff > DISCOVERY_END:
        raise ValueError("downstream lambda selection cutoff exceeds 2022-12-31")
    if choice.get("sealed_data_used_for_selection") is not False:
        raise ValueError("downstream lambda choice does not explicitly exclude sealed selection")
    return value


@dataclass(frozen=True)
class FrozenHandoff:
    compact: dict[str, Any]
    selection: dict[str, Any]
    selected_name: str
    selection_role: str
    promotion_gate_passed: bool
    compact_manifest_path: Path
    compact_prediction_path: Path
    selection_manifest_path: Path
    source_prediction_path: Path
    source_manifest_path: Path
    candidate_catalog_path: Path
    frozen_catalog_path: Path
    core_bridge_path: Path
    core_bridge_sha256: str
    verified_paths: tuple[Path, ...]
    downstream_lambda: float
    audit: dict[str, Any]


def validate_frozen_handoff_chain(manifest_path: Path) -> FrozenHandoff:
    """Verify the compact -> selection -> completion/catalog chain without scoring."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    compact = json.loads(manifest_path.read_text(encoding="utf-8"))
    if compact.get("contract") != FINALIST_CONTRACT:
        raise ValueError(f"unexpected compact handoff contract {compact.get('contract')!r}")
    for key, expected in (
        ("frozen", True),
        ("research_only", True),
        ("sealed_data_used_for_selection", False),
        ("frozen_before_sealed_scoring", True),
        ("uses_closing_odds_as_feature", False),
        ("control_core_point_in_time", True),
        ("control_core_uses_sealed_future", False),
    ):
        _required_bool(compact, key, expected)
    if str(compact.get("prediction_units", "")).casefold() != "sg_per_round":
        raise ValueError("compact finalist units must be sg_per_round")
    if str(compact.get("control_units", "")).casefold() != "sg_per_round":
        raise ValueError("compact control units must be sg_per_round")
    if str(compact.get("control_name")) != STRICT_PT_CONTROL:
        raise ValueError("compact control is not the strict point-in-time architecture control")
    if _selection_cutoff(compact) > DISCOVERY_END:
        raise ValueError("compact selection cutoff exceeds 2022-12-31")

    compact_prediction_path = resolve_path(
        manifest_path.parent, compact.get("prediction_file", "")
    )
    if not compact_prediction_path.is_file():
        raise FileNotFoundError(compact_prediction_path)
    compact_prediction_hash = sha256_file(compact_prediction_path)
    if compact_prediction_hash != str(compact.get("prediction_sha256", "")).lower():
        raise ValueError("compact prediction hash mismatch")

    selection_manifest_path = resolve_path(
        manifest_path.parent, compact.get("selection_manifest", "")
    )
    if not selection_manifest_path.is_file():
        raise FileNotFoundError(selection_manifest_path)
    selection_hash = sha256_file(selection_manifest_path)
    if selection_hash != str(compact.get("selection_manifest_sha256", "")).lower():
        raise ValueError("selection manifest hash mismatch")
    selection = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection.get("contract") != SELECTION_CONTRACT or selection.get("complete") is not True:
        raise ValueError("selection manifest contract is incomplete or unexpected")
    for key, expected in (
        ("research_only", True),
        ("selection_uses_sealed", False),
        ("sealed_data_used_for_selection", False),
        ("uses_closing_odds_as_feature", False),
    ):
        _required_bool(selection, key, expected)
    if _selection_cutoff(selection) > DISCOVERY_END:
        raise ValueError("selection cutoff exceeds 2022-12-31")
    frozen_parameters = selection.get("frozen_selection_parameter_contract")
    expected_parameters = discovery.validate_frozen_selection_cli(
        discovery.build_parser().parse_args([])
    )
    if frozen_parameters != expected_parameters:
        raise ValueError("selection parameter contract differs from frozen constants")

    selected_name = _compact_name(compact)
    if selected_name != _selection_name(selection):
        raise ValueError("compact selected name disagrees with selection manifest")
    if not isinstance(selection.get("promotion_gate_passed"), bool):
        raise ValueError("selection manifest requires boolean promotion_gate_passed")
    promotion_gate_passed = selection["promotion_gate_passed"]
    finalists = selection.get("finalists", [])
    if not isinstance(finalists, list):
        raise ValueError("selection manifest finalists must be a list")
    if promotion_gate_passed:
        if not finalists or not compact.get("finalist_name"):
            raise ValueError("passed promotion gate requires a declared promoted finalist")
    else:
        if finalists or compact.get("finalist_name") not in (None, ""):
            raise ValueError(
                "failed promotion gate must hand off only a diagnostic challenger"
            )
        if not selection.get("diagnostic_challenger"):
            raise ValueError("failed promotion gate lacks a diagnostic challenger")
    compact_gate = compact.get("promotion_gate_passed")
    if not isinstance(compact_gate, bool) or compact_gate is not promotion_gate_passed:
        raise ValueError("compact promotion-gate result disagrees with selection manifest")
    expected_role = "promoted_finalist" if promotion_gate_passed else "diagnostic_challenger"
    selection_role = str(compact.get("selection_role", expected_role))
    if selection_role != expected_role:
        raise ValueError("compact selection role disagrees with promotion-gate result")
    locked = selection.get("locked_diagnostic_ablation_set")
    if not isinstance(locked, Mapping):
        raise ValueError("selection manifest lacks locked diagnostic ablations")
    if locked.get("locked_before_sealed_scoring") is not True:
        raise ValueError("diagnostic ablations were not locked before sealed scoring")
    if locked.get("may_reselect_or_promote_from_sealed") is not False:
        raise ValueError("locked ablations do not prohibit sealed reselection")
    if str(locked.get("selected_candidate")) != selected_name:
        raise ValueError("locked ablation selected candidate disagrees with compact handoff")
    selection_artifacts = _verify_selection_artifacts(
        selection_manifest_path, selection
    )
    inputs = selection.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("selection manifest lacks hashed inputs")
    if set(inputs) != SELECTION_REQUIRED_INPUTS:
        missing = sorted(SELECTION_REQUIRED_INPUTS - set(inputs))
        extras = sorted(set(inputs) - SELECTION_REQUIRED_INPUTS)
        raise ValueError(
            "selection manifest input set differs from the frozen discovery contract; "
            f"missing={missing}, extras={extras}"
        )
    verified_selection_inputs = _verify_recorded_files(
        selection_manifest_path.parent, inputs, "selection inputs"
    )
    manifest_input_path_hashes = {
        str(path): digest for path, digest in verified_selection_inputs.values()
    }
    source_prediction_path, source_prediction_hash = _verified_file(
        selection_manifest_path.parent, inputs.get("predictions"), "predictions"
    )
    source_manifest_path, source_manifest_hash = _verified_file(
        selection_manifest_path.parent, inputs.get("candidate_manifest"), "candidate_manifest"
    )
    candidate_catalog_path, candidate_catalog_hash = _verified_file(
        selection_manifest_path.parent, inputs.get("candidate_catalog"), "candidate_catalog"
    )
    frozen_catalog_path, frozen_catalog_hash = _verified_file(
        selection_manifest_path.parent,
        inputs.get("frozen_candidate_catalog"),
        "frozen_candidate_catalog",
    )
    if "core_bridge" not in verified_selection_inputs:
        raise ValueError("selection manifest lacks directly hashed core_bridge input")
    core_bridge_path, core_bridge_hash = verified_selection_inputs["core_bridge"]
    evaluation_cache_manifest_path = verified_selection_inputs[
        "evaluation_cache_manifest"
    ][0]
    (
        _evaluation_cache_manifest,
        evaluation_cache_authority_audit,
        evaluation_cache_authority_bindings,
    ) = discovery.validate_evaluation_cache_authority(
        evaluation_cache_manifest_path.parent,
        core_bridge_path,
    )
    for direct_name, authority_name in (
        ("evaluation_cache_manifest", "evaluation_cache_manifest"),
        ("normalized_quotes", "normalized_quotes"),
        ("feature_snapshots", "feature_snapshots"),
        ("core_bridge", "cache_input_legacy_core_bridge"),
    ):
        if (
            verified_selection_inputs[direct_name][0]
            != evaluation_cache_authority_bindings[authority_name]
        ):
            raise ValueError(
                f"selection input is not evaluation-cache-authorized: {direct_name}"
            )
    (
        authority_source_manifest,
        completed_study_authority_audit,
        completed_study_authority_bindings,
    ) = discovery.validate_completed_study_authority(
        source_manifest_path,
        source_prediction_path,
        candidate_catalog_path,
    )
    for selection_name, authority_name in (
        ("candidate_manifest", "completion_manifest"),
        ("predictions", "predictions"),
        ("candidate_catalog", "candidate_catalog"),
        ("coverage_by_candidate_year", "coverage_by_candidate_year"),
        ("frozen_candidate_catalog", "frozen_candidate_catalog"),
    ):
        if (
            verified_selection_inputs[selection_name][0]
            != completed_study_authority_bindings[authority_name]
        ):
            raise ValueError(
                f"selection input is not completed-study-authorized: {selection_name}"
            )
    expected_execution_input_hashes = dict(manifest_input_path_hashes)
    for path in (
        *evaluation_cache_authority_bindings.values(),
        *completed_study_authority_bindings.values(),
    ):
        resolved = path.resolve()
        digest = sha256_file(resolved)
        recorded = expected_execution_input_hashes.setdefault(str(resolved), digest)
        if recorded != digest:
            raise ValueError("selection/cache/completed-study authority hashes disagree")
    if len(expected_execution_input_hashes) != 16:
        raise ValueError(
            "selection/cache/completed-study authority does not bind exactly 16 files"
        )
    selection_execution_audit = _verify_selection_execution_provenance(
        selection_manifest_path,
        selection,
        selection_artifacts,
        expected_execution_input_hashes,
    )
    compact_source_path = resolve_path(
        manifest_path.parent, compact.get("source_prediction_file", "")
    )
    if compact_source_path != source_prediction_path:
        raise ValueError("compact source prediction path differs from selection input")
    if str(compact.get("source_prediction_sha256", "")).lower() != source_prediction_hash:
        raise ValueError("compact source prediction hash differs from selection input")

    source_manifest = dict(authority_source_manifest)
    if source_manifest.get("complete") is not True or source_manifest.get("research_only") is not True:
        raise ValueError("completion manifest is not a complete research-only build")
    _required_bool(source_manifest, "production_files_changed", False)
    completion_execution_audit = _verify_completion_execution_provenance(
        source_manifest_path, source_manifest
    )
    completion_contract = source_manifest.get("contract")
    if not isinstance(completion_contract, Mapping):
        raise ValueError("completion manifest lacks its chronology/feature contract")
    for key in (
        "challengers_use_legacy_skill",
        "closing_prices_used_as_features",
        "selection_uses_sealed",
    ):
        _required_bool(completion_contract, key, False)
    _manifest_output(
        source_manifest,
        source_manifest_path,
        "player_event_predictions.parquet",
        source_prediction_path,
    )
    _manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog.csv",
        candidate_catalog_path,
    )
    _manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog_frozen.json",
        frozen_catalog_path,
    )
    compact_core_binding_audit = _verify_compact_core_values(
        compact_prediction_path,
        source_prediction_path,
        selected_name,
    )
    verified_completion_outputs = _verify_recorded_files(
        source_manifest_path.parent,
        source_manifest.get("outputs"),
        "completion outputs",
    )
    preserved_source_path = Path(
        completion_execution_audit["executed_source_copy_path"]
    ).resolve()
    if preserved_source_path not in {
        path for path, _digest in verified_completion_outputs.values()
    }:
        raise ValueError("completion preserved source copy is absent from hashed outputs")
    completion_inputs = source_manifest.get("inputs", {})
    verified_completion_inputs = _verify_recorded_files(
        source_manifest_path.parent,
        completion_inputs,
        "completion inputs",
    )
    source_core_entry = completion_inputs.get("core_bridge")
    source_core_path, source_core_hash = _verified_file(
        source_manifest_path.parent, source_core_entry, "completion inputs.core_bridge"
    )
    if source_core_path != core_bridge_path or source_core_hash != core_bridge_hash:
        raise ValueError(
            "selection core_bridge does not match the completion build's direct input"
        )
    declared_catalog_hashes = {
        str(selection.get("frozen_candidate_catalog_sha256", "")).lower(),
        str(source_manifest.get("frozen_candidate_catalog_sha256", "")).lower(),
        str(discovery.EXPECTED_FROZEN_CATALOG_SHA256).lower(),
    }
    if declared_catalog_hashes != {frozen_catalog_hash}:
        raise ValueError("selection/completion/code frozen-catalog hashes do not agree")
    catalog = pd.read_csv(candidate_catalog_path)
    catalog_audit = discovery.validate_frozen_catalog_contract(
        catalog, source_manifest, frozen_catalog_path
    )
    if "name" not in catalog:
        raise ValueError("candidate catalog lacks name")
    selected_catalog = catalog.loc[catalog["name"].astype(str).eq(selected_name)]
    if len(selected_catalog) != 1:
        raise ValueError("selected model is missing or duplicated in candidate catalog")
    for column, expected in (
        ("selection_eligible", True),
        ("sensitivity_only", False),
        ("control", False),
    ):
        if column not in selected_catalog or _as_bool(selected_catalog.iloc[0][column]) is not expected:
            raise ValueError(f"selected model violates candidate-catalog {column}={expected}")

    downstream_lambda = selected_downstream_lambda(selection, selected_name)
    audit = {
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "discovery_evaluator_source_sha256": sha256_file(
            Path(discovery.__file__).resolve()
        ),
        "compact_manifest_sha256": sha256_file(manifest_path),
        "compact_prediction_sha256": compact_prediction_hash,
        "selection_manifest_sha256": selection_hash,
        "source_prediction_sha256": source_prediction_hash,
        "source_manifest_sha256": source_manifest_hash,
        "candidate_catalog_sha256": candidate_catalog_hash,
        "frozen_catalog_sha256": frozen_catalog_hash,
        "selection_inputs_verified": len(verified_selection_inputs),
        "selection_artifacts_verified": len(selection_artifacts),
        "selection_execution_provenance": selection_execution_audit,
        "evaluation_cache_authority": evaluation_cache_authority_audit,
        "completed_study_authority": completed_study_authority_audit,
        "completion_execution_provenance": completion_execution_audit,
        "completion_outputs_verified": len(verified_completion_outputs),
        "completion_input_bindings": {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in verified_completion_inputs.items()
        },
        "compact_core_binding": compact_core_binding_audit,
        "core_bridge_path": str(core_bridge_path),
        "core_bridge_sha256": core_bridge_hash,
        "catalog_contract": catalog_audit,
        "selected_name": selected_name,
        "selection_role": selection_role,
        "promotion_gate_passed": promotion_gate_passed,
        "discovery_selected_downstream_lambda": downstream_lambda,
        "full_hash_chain_verified": True,
    }
    completion_input_paths = {
        path for path, _digest in verified_completion_inputs.values()
    }
    verified_paths = tuple(
        sorted(
            {
                manifest_path,
                compact_prediction_path,
                selection_manifest_path,
                source_manifest_path,
                *[path for path, _digest in verified_selection_inputs.values()],
                *evaluation_cache_authority_bindings.values(),
                *completed_study_authority_bindings.values(),
                *selection_artifacts.keys(),
                *[
                    Path(path).resolve()
                    for path in selection_execution_audit["verified_source_paths"]
                ],
                *completion_input_paths,
                Path(completion_execution_audit["builder_source_path"]).resolve(),
                *[
                    path
                    for path, _digest in verified_completion_outputs.values()
                ],
            },
            key=str,
        )
    )
    audit["verified_path_count"] = len(verified_paths)
    return FrozenHandoff(
        compact=compact,
        selection=selection,
        selected_name=selected_name,
        selection_role=selection_role,
        promotion_gate_passed=promotion_gate_passed,
        compact_manifest_path=manifest_path,
        compact_prediction_path=compact_prediction_path,
        selection_manifest_path=selection_manifest_path,
        source_prediction_path=source_prediction_path,
        source_manifest_path=source_manifest_path,
        candidate_catalog_path=candidate_catalog_path,
        frozen_catalog_path=frozen_catalog_path,
        core_bridge_path=core_bridge_path,
        core_bridge_sha256=core_bridge_hash,
        verified_paths=verified_paths,
        downstream_lambda=downstream_lambda,
        audit=audit,
    )

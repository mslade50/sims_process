"""Build the sealed 2023-selection -> 2024-26 runner handoff.

This research-only boundary validates the independent training, prediction-arm,
and market-selection authorities before it writes the one specification that
the all-round runner may consume.  Its only parquet reads are projected 2023
selection identifiers; no outcome, quote, or post-2023 label column is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


EXPECTED_CONTRACT_SHA256 = (
    "4091a0192a959640a9dc88ff6b348759cf3052ed7aed3a116b5989019111374c"
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ALL_ROUND_BASE_SKILL_REBUILD_CONTRACT_2026_08.md"
)
CONTROL_CANDIDATE = "baseline_current_prod"
LAMBDA_GRID = (0.0, 0.5, 1.0)
SELECTION_YEAR = 2023
TRAINING_CATALOG_HASH_COLUMNS = (
    "candidate",
    "target_arm",
    "weather_arm",
    "feature_family",
    "estimator",
    "alpha",
    "selectable",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_catalog_hash(catalog: pd.DataFrame) -> str:
    missing = sorted(set(TRAINING_CATALOG_HASH_COLUMNS).difference(catalog.columns))
    if missing:
        raise ValueError(f"training candidate catalog is missing hash columns: {missing}")
    payload = (
        catalog[list(TRAINING_CATALOG_HASH_COLUMNS)]
        .sort_values("candidate", kind="mergesort")
        .to_csv(index=False, lineterminator="\n")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _same_hash(observed: object, expected: object, label: str) -> None:
    if str(observed).casefold() != str(expected).casefold():
        raise ValueError(f"{label} hash mismatch: {observed!r} != {expected!r}")


def _require_fields(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields).difference(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _require_columns(frame: pd.DataFrame, fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} is not boolean: {value!r}")


def _float_lambda(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number not in LAMBDA_GRID:
        raise ValueError(f"{label} must be one of {LAMBDA_GRID}; found {value!r}")
    return number


def _recorded_path(owner: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def _validate_recorded_artifact(
    owner_path: Path,
    payload: Mapping[str, Any],
    *,
    prefix: str,
    actual_path: Path,
) -> None:
    path_field = f"{prefix}_path"
    hash_field = f"{prefix}_sha256"
    _require_fields(payload, (path_field, hash_field), owner_path.name)
    recorded = _recorded_path(owner_path, payload[path_field])
    if recorded != actual_path.resolve():
        raise ValueError(
            f"{owner_path.name} {path_field} does not resolve to the supplied artifact"
        )
    _same_hash(payload[hash_field], sha256_file(actual_path), f"{owner_path.name} {prefix}")


def _validate_optional_recorded_artifact(
    owner_path: Path,
    payload: Mapping[str, Any],
    *,
    prefix: str,
    actual_path: Path,
) -> bool:
    """Validate a recorded artifact, or report that this older authority omitted it."""

    fields = {f"{prefix}_path", f"{prefix}_sha256"}
    present = fields.intersection(payload)
    if present and present != fields:
        raise ValueError(f"{owner_path.name} only partially binds {prefix}")
    if not present:
        return False
    _validate_recorded_artifact(
        owner_path, payload, prefix=prefix, actual_path=actual_path
    )
    return True


def _validate_training_authority(
    catalog_path: Path,
    manifest_path: Path,
    contract_hash: str,
    selected: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    catalog = pd.read_csv(catalog_path)
    _require_columns(
        catalog,
        (*TRAINING_CATALOG_HASH_COLUMNS, "run_status"),
        "training candidate catalog",
    )
    catalog["candidate"] = catalog["candidate"].astype(str)
    if catalog.duplicated("candidate").any():
        raise ValueError("training candidate catalog contains duplicate candidates")
    canonical_hash = training_catalog_hash(catalog)
    manifest = _read_json(manifest_path, "training manifest")
    _require_fields(
        manifest,
        (
            "research_only",
            "production_files_written",
            "phase",
            "smoke",
            "smoke_outputs_selection_eligible",
            "contract_sha256",
            "candidate_catalog_sha256",
            "evaluation_outcomes_read",
        ),
        "training manifest",
    )
    if not _as_bool(manifest["research_only"], "training research_only"):
        raise ValueError("training manifest is not research-only")
    if _as_bool(manifest["production_files_written"], "training production flag"):
        raise ValueError("training manifest reports a production write")
    if str(manifest["phase"]) != "train":
        raise ValueError("training manifest phase is not train")
    if _as_bool(manifest["smoke"], "training smoke"):
        raise ValueError("a smoke training run cannot create an evaluation handoff")
    if not _as_bool(
        manifest["smoke_outputs_selection_eligible"], "training selection eligibility"
    ):
        raise ValueError("training outputs are marked selection-ineligible")
    if _as_bool(manifest["evaluation_outcomes_read"], "training outcome-read flag"):
        raise ValueError("training authority reports reading evaluation outcomes")
    _same_hash(manifest["contract_sha256"], contract_hash, "training contract")
    _same_hash(
        manifest["candidate_catalog_sha256"], canonical_hash, "training catalog canonical"
    )
    row = catalog.loc[catalog["candidate"].eq(selected)]
    if len(row) != 1:
        raise ValueError("selected market candidate has no unique training-catalog row")
    selected_row = row.iloc[0]
    if not _as_bool(selected_row["selectable"], "selected training selectable"):
        raise ValueError("selected candidate is a nonselectable runner diagnostic")
    if str(selected_row["run_status"]) != "ok":
        raise ValueError("selected candidate did not complete the training runner")
    if "diagnostic_only" in catalog and _as_bool(
        selected_row["diagnostic_only"], "selected diagnostic_only"
    ):
        raise ValueError("selected candidate is diagnostic-only")
    if "smoke_run" in catalog and _as_bool(selected_row["smoke_run"], "selected smoke_run"):
        raise ValueError("selected candidate came from a smoke run")
    return catalog, canonical_hash, manifest


def _validate_selection_prediction(
    path: Path,
    *,
    label: str,
    round_level: bool,
    expected_candidates: set[str],
) -> dict[str, Any]:
    schema = set(pq.ParquetFile(path).schema.names)
    required = {"candidate", "event_key", "dg_id_key", "prediction"}
    if round_level:
        required.add("round_num")
    missing = sorted(required - schema)
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    # Deliberately do not project prediction, quote, target, or outcome values.
    columns = ["candidate", "event_key", "dg_id_key"]
    if round_level:
        columns.append("round_num")
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError(f"{label} is empty")
    frame["candidate"] = frame["candidate"].astype(str)
    candidates = set(frame["candidate"])
    if candidates != expected_candidates:
        raise ValueError(
            f"{label} candidate closure differs from market authority: "
            f"{sorted(candidates)} != {sorted(expected_candidates)}"
        )
    event_text = frame["event_key"].astype(str)
    years = pd.to_numeric(
        event_text.str.extract(r"(?:^|\|)(20\d{2})(?:\||$)", expand=False),
        errors="coerce",
    )
    if years.isna().any() or not years.eq(SELECTION_YEAR).all():
        found = sorted(years.dropna().astype(int).unique().tolist())
        raise ValueError(f"{label} is not strictly 2023-only; found years {found}")
    keys = ["candidate", "event_key", "dg_id_key"]
    if round_level:
        frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
        rounds = set(frame["round_num"].dropna().astype(int))
        if rounds != {1, 2, 3, 4}:
            raise ValueError(f"{label} must contain R1-R4; found {sorted(rounds)}")
        keys.append("round_num")
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate prediction keys")
    return {
        "rows": int(len(frame)),
        "candidates": sorted(candidates),
        "events": int(frame["event_key"].nunique()),
        "projected_columns_read": columns,
        "outcome_quote_or_label_columns_read": [],
    }


def _validate_market_authorities(
    *,
    selected: str,
    baseline: str,
    contract_hash: str,
    frozen_lambdas_path: Path,
    market_authority_path: Path,
    arm_manifest_path: Path,
    arm_audit_path: Path,
    base_predictions_path: Path,
    round_predictions_path: Path,
) -> tuple[float, float, dict[str, Any]]:
    authority = pd.read_csv(market_authority_path)
    _require_columns(
        authority,
        ("candidate", "course_fit_lambda", "live_update_lambda"),
        "market candidate authority",
    )
    authority["candidate"] = authority["candidate"].astype(str)
    if authority.duplicated("candidate").any():
        raise ValueError("market candidate authority contains duplicates")
    expected_candidates = set(authority["candidate"])
    if selected not in expected_candidates or baseline not in expected_candidates:
        raise ValueError("market authority omits the selected candidate or control")
    selected_authority = authority.loc[authority["candidate"].eq(selected)].iloc[0]
    course_fit = _float_lambda(
        selected_authority["course_fit_lambda"], "selected authority course-fit lambda"
    )
    live = _float_lambda(
        selected_authority["live_update_lambda"], "selected authority live lambda"
    )

    frozen = _read_json(frozen_lambdas_path, "frozen lambdas")
    _same_hash(frozen.get("contract_sha256"), contract_hash, "frozen-lambda contract")
    if int(frozen.get("selection_year", -1)) != SELECTION_YEAR:
        raise ValueError("frozen lambdas were not selected in 2023")
    if tuple(float(x) for x in frozen.get("lambda_grid", ())) != LAMBDA_GRID:
        raise ValueError("frozen lambdas do not bind the locked lambda grid")
    _validate_recorded_artifact(
        frozen_lambdas_path,
        frozen,
        prefix="market_candidate_authority",
        actual_path=market_authority_path,
    )
    frozen_base_bound = _validate_optional_recorded_artifact(
        frozen_lambdas_path,
        frozen,
        prefix="selection_base_predictions",
        actual_path=base_predictions_path,
    )
    frozen_round_bound = _validate_optional_recorded_artifact(
        frozen_lambdas_path,
        frozen,
        prefix="selection_round_predictions",
        actual_path=round_predictions_path,
    )
    lambdas = pd.DataFrame(frozen.get("lambdas", []))
    _require_columns(
        lambdas,
        ("candidate", "downstream_lambda", "live_lambda"),
        "frozen lambda rows",
    )
    lambdas["candidate"] = lambdas["candidate"].astype(str)
    if lambdas.duplicated("candidate").any():
        raise ValueError("frozen lambdas contain duplicate candidate rows")
    challenger_closure = expected_candidates - {baseline}
    if set(lambdas["candidate"]) != challenger_closure:
        raise ValueError("frozen lambda rows do not close over the challenger authority")
    lambda_row = lambdas.loc[lambdas["candidate"].eq(selected)]
    if len(lambda_row) != 1:
        raise ValueError("frozen lambdas lack one selected-candidate row")
    downstream = _float_lambda(
        lambda_row.iloc[0]["downstream_lambda"], "selected frozen downstream lambda"
    )
    frozen_live = _float_lambda(
        lambda_row.iloc[0]["live_lambda"], "selected frozen live lambda"
    )
    if not math.isclose(course_fit, downstream, abs_tol=1e-12):
        raise ValueError("market course-fit lambda differs from frozen downstream lambda")
    if not math.isclose(live, frozen_live, abs_tol=1e-12):
        raise ValueError("market live lambda differs from frozen live lambda")

    base_audit = _validate_selection_prediction(
        base_predictions_path,
        label="2023 selection base predictions",
        round_level=False,
        expected_candidates=expected_candidates,
    )
    round_audit = _validate_selection_prediction(
        round_predictions_path,
        label="2023 selection round predictions",
        round_level=True,
        expected_candidates=expected_candidates,
    )

    arm_manifest = _read_json(arm_manifest_path, "selection-arm manifest")
    arm_audit = _read_json(arm_audit_path, "selection-arm audit")
    prediction_bindings: dict[str, dict[str, bool]] = {}
    for owner_path, payload in (
        (arm_manifest_path, arm_manifest),
        (arm_audit_path, arm_audit),
    ):
        _same_hash(payload.get("contract_sha256"), contract_hash, f"{owner_path.name} contract")
        _validate_recorded_artifact(
            owner_path,
            payload,
            prefix="market_candidate_authority",
            actual_path=market_authority_path,
        )
        base_bound = _validate_optional_recorded_artifact(
            owner_path,
            payload,
            prefix="selection_base_predictions",
            actual_path=base_predictions_path,
        )
        round_bound = _validate_optional_recorded_artifact(
            owner_path,
            payload,
            prefix="selection_round_predictions",
            actual_path=round_predictions_path,
        )
        prediction_bindings[owner_path.name] = {
            "base_hash_recorded": base_bound,
            "round_hash_recorded": round_bound,
        }
    if not _as_bool(arm_manifest.get("research_only"), "arm manifest research_only"):
        raise ValueError("selection-arm manifest is not research-only")
    if _as_bool(
        arm_manifest.get("production_files_written"), "arm manifest production flag"
    ):
        raise ValueError("selection-arm manifest reports a production write")
    _validate_recorded_artifact(
        arm_manifest_path,
        arm_manifest,
        prefix="frozen_lambdas",
        actual_path=frozen_lambdas_path,
    )
    manifest_closure = set(str(x) for x in arm_manifest.get("candidate_closure", ()))
    if manifest_closure != expected_candidates:
        raise ValueError("selection-arm manifest candidate closure mismatch")
    if int(arm_audit.get("selection_year", -1)) != SELECTION_YEAR:
        raise ValueError("selection-arm audit is not a 2023 selection artifact")
    if arm_audit.get("evaluation_target_columns_read") not in ([], None):
        raise ValueError("selection-arm audit reports reading evaluation targets")
    for payload, label in (
        (frozen, "frozen lambdas"),
        (arm_manifest, "selection-arm manifest"),
        (arm_audit, "selection-arm audit"),
    ):
        recorded_base_rows = payload.get(
            "selection_base_predictions_rows", payload.get("base_rows")
        )
        recorded_round_rows = payload.get(
            "selection_round_predictions_rows", payload.get("round_rows")
        )
        if recorded_base_rows is not None and int(recorded_base_rows) != base_audit["rows"]:
            raise ValueError(f"{label} base prediction row count mismatch")
        if recorded_round_rows is not None and int(recorded_round_rows) != round_audit["rows"]:
            raise ValueError(f"{label} round prediction row count mismatch")

    return downstream, frozen_live, {
        "authority": authority,
        "authority_sha256": sha256_file(market_authority_path),
        "frozen": frozen,
        "arm_manifest": arm_manifest,
        "arm_audit": arm_audit,
        "base_prediction_audit": base_audit,
        "round_prediction_audit": round_audit,
        "selection_prediction_hash_bindings": {
            "frozen_lambdas": {
                "base_hash_recorded": frozen_base_bound,
                "round_hash_recorded": frozen_round_bound,
            },
            **prediction_bindings,
            "handoff_is_physical_hash_authority": True,
        },
    }


def _validate_selection_freeze(
    freeze_path: Path,
    market_authority_path: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    sidecar_path = freeze_path.with_suffix(".sha256")
    if not sidecar_path.exists():
        raise ValueError("selection freeze lacks its SHA-256 sidecar")
    tokens = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError("selection freeze sidecar is empty")
    freeze_file_hash = sha256_file(freeze_path)
    _same_hash(tokens[0], freeze_file_hash, "selection freeze sidecar")
    freeze = _read_json(freeze_path, "selection freeze")
    recorded_payload_hash = str(freeze.get("freeze_payload_sha256", ""))
    hash_input = dict(freeze)
    hash_input.pop("freeze_payload_sha256", None)
    _same_hash(
        recorded_payload_hash,
        canonical_payload_hash(hash_input),
        "selection freeze payload self",
    )
    frozen_spec = freeze.get("frozen_spec")
    if not isinstance(frozen_spec, dict):
        raise ValueError("selection freeze lacks a frozen_spec object")
    _same_hash(
        freeze.get("frozen_spec_sha256"),
        canonical_payload_hash(frozen_spec),
        "selection frozen-spec self",
    )
    contract_hash = sha256_file(contract_path)
    _same_hash(contract_hash, EXPECTED_CONTRACT_SHA256, "locked contract")
    _same_hash(freeze.get("contract_sha256"), contract_hash, "selection freeze contract")
    if "contract_path" in freeze:
        recorded_contract = _recorded_path(freeze_path, freeze["contract_path"])
        if recorded_contract != contract_path.resolve():
            raise ValueError("selection freeze contract_path differs from supplied contract")
    _validate_recorded_artifact(
        freeze_path,
        freeze,
        prefix="candidate_catalog",
        actual_path=market_authority_path,
    )
    selected = str(freeze.get("selected_candidate", ""))
    baseline = str(freeze.get("baseline", ""))
    if not selected or not baseline or selected == baseline:
        raise ValueError("selection freeze does not name a distinct challenger and baseline")
    selected_spec = freeze.get("selected_spec")
    if not isinstance(selected_spec, dict) or str(selected_spec.get("candidate")) != selected:
        raise ValueError("selection freeze selected_spec candidate mismatch")
    if str(frozen_spec.get("candidate")) != selected:
        raise ValueError("selection freeze frozen_spec candidate mismatch")
    if str(frozen_spec.get("baseline")) != baseline:
        raise ValueError("selection freeze frozen_spec baseline mismatch")
    return freeze, selected, baseline, {
        "freeze_sidecar_path": sidecar_path,
        "freeze_sha256": freeze_file_hash,
        "freeze_sidecar_sha256": sha256_file(sidecar_path),
        "contract_sha256": contract_hash,
    }


def _selection_metadata(freeze: Mapping[str, Any]) -> dict[str, Any]:
    primary = freeze.get("primary_selection")
    primary_dict = dict(primary) if isinstance(primary, dict) else {}
    containers = (freeze, freeze.get("selected_spec", {}), freeze.get("frozen_spec", {}), primary_dict)
    flags: list[bool] = []
    for container in containers:
        if isinstance(container, Mapping) and "promotion_eligible" in container:
            flags.append(_as_bool(container["promotion_eligible"], "promotion_eligible"))
    if flags and any(flag != flags[0] for flag in flags[1:]):
        raise ValueError("selection freeze contains conflicting promotion_eligible flags")
    mode = next(
        (
            str(container[key])
            for container in containers
            if isinstance(container, Mapping)
            for key in ("selection_mode", "selection_authority_mode", "authority_mode")
            if container.get(key) not in (None, "")
        ),
        "primary_contract_selection",
    )
    explicit_fallback = any(
        _as_bool(container.get("diagnostic_fallback"), "diagnostic_fallback")
        for container in containers
        if isinstance(container, Mapping) and "diagnostic_fallback" in container
    )
    diagnostic_fallback = explicit_fallback or bool(
        re.search(r"diagnostic|fallback", mode, flags=re.IGNORECASE)
    )
    promotion_eligible = flags[0] if flags else not diagnostic_fallback
    if diagnostic_fallback and promotion_eligible:
        raise ValueError("a diagnostic-fallback freeze cannot be promotion-eligible")
    normalized_primary = primary_dict
    for key in (
        "primary_selection_outcome",
        "diagnostic_fallback_rule",
        "eligible_challenger_count",
        "selected_survived_dominance",
    ):
        if key in freeze:
            normalized_primary.setdefault(key, freeze[key])
    normalized_primary.setdefault("selection_mode", mode)
    normalized_primary.setdefault("diagnostic_fallback", diagnostic_fallback)
    normalized_primary.setdefault("promotion_eligible", promotion_eligible)
    # setdefault preserves the frozen metadata, while the agreement checks
    # above prevent this handoff from upgrading a false eligibility flag.
    return {
        "selection_mode": mode,
        "diagnostic_fallback": diagnostic_fallback,
        "promotion_eligible": promotion_eligible,
        "primary_selection": normalized_primary,
    }


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable handoff artifact already exists: {path}")
        return
    path.write_bytes(content)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n").encode(
        "utf-8"
    )


def build_selected_handoff(
    *,
    selection_freeze_path: str | Path,
    training_candidate_catalog_path: str | Path,
    training_manifest_path: str | Path,
    frozen_lambdas_path: str | Path,
    market_candidate_authority_path: str | Path,
    selection_arm_manifest_path: str | Path,
    selection_arm_audit_path: str | Path,
    selection_base_predictions_path: str | Path,
    selection_round_predictions_path: str | Path,
    output_dir: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate all 2023 authorities and write the immutable runner handoff."""

    paths = {
        "selection_freeze": Path(selection_freeze_path).resolve(),
        "training_candidate_catalog": Path(training_candidate_catalog_path).resolve(),
        "training_manifest": Path(training_manifest_path).resolve(),
        "frozen_lambdas": Path(frozen_lambdas_path).resolve(),
        "market_candidate_authority": Path(market_candidate_authority_path).resolve(),
        "selection_arm_manifest": Path(selection_arm_manifest_path).resolve(),
        "selection_arm_audit": Path(selection_arm_audit_path).resolve(),
        "selection_base_predictions": Path(selection_base_predictions_path).resolve(),
        "selection_round_predictions": Path(selection_round_predictions_path).resolve(),
        "contract": Path(contract_path).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"handoff input files are missing: {missing}")

    freeze, selected, baseline, freeze_audit = _validate_selection_freeze(
        paths["selection_freeze"],
        paths["market_candidate_authority"],
        paths["contract"],
    )
    catalog, catalog_hash, train_manifest = _validate_training_authority(
        paths["training_candidate_catalog"],
        paths["training_manifest"],
        freeze_audit["contract_sha256"],
        selected,
    )
    downstream, live, arm = _validate_market_authorities(
        selected=selected,
        baseline=baseline,
        contract_hash=freeze_audit["contract_sha256"],
        frozen_lambdas_path=paths["frozen_lambdas"],
        market_authority_path=paths["market_candidate_authority"],
        arm_manifest_path=paths["selection_arm_manifest"],
        arm_audit_path=paths["selection_arm_audit"],
        base_predictions_path=paths["selection_base_predictions"],
        round_predictions_path=paths["selection_round_predictions"],
    )
    selected_spec = freeze["selected_spec"]
    frozen_spec = freeze["frozen_spec"]
    for container, label in ((selected_spec, "selected_spec"), (frozen_spec, "frozen_spec")):
        if not math.isclose(
            _float_lambda(container.get("course_fit_lambda"), f"{label} course-fit"),
            downstream,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} course-fit lambda differs from frozen downstream lambda")
        if not math.isclose(
            _float_lambda(container.get("live_update_lambda"), f"{label} live"),
            live,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} live lambda differs from frozen live lambda")

    metadata = _selection_metadata(freeze)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "runner_selected_spec.json"
    spec_sidecar_path = output / "runner_selected_spec.sha256"
    handoff_path = output / "selected_handoff_manifest.json"
    handoff_sidecar_path = output / "selected_handoff_manifest.sha256"
    spec = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "candidate": selected,
        "selected_arm": selected,
        "downstream_lambda": downstream,
        "course_fit_lambda": downstream,
        "live_update_lambda": live,
        "catalog_sha256": catalog_hash,
        "contract_sha256": freeze_audit["contract_sha256"],
        "promotion_eligible": metadata["promotion_eligible"],
        "diagnostic_fallback": metadata["diagnostic_fallback"],
        "selection_mode": metadata["selection_mode"],
        "primary_selection": metadata["primary_selection"],
        "selection_freeze_sha256": freeze_audit["freeze_sha256"],
        "market_candidate_authority_sha256": arm["authority_sha256"],
        "selection_source": "2023-only frozen joint market selection",
        "evaluation_scope": "2024-2026 label-blind prediction application",
    }
    _write_immutable(spec_path, _json_bytes(spec))
    spec_hash = sha256_file(spec_path)
    _write_immutable(
        spec_sidecar_path,
        f"{spec_hash}  {spec_path.name}\n".encode("utf-8"),
    )

    inputs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    inputs["training_candidate_catalog"]["canonical_catalog_sha256"] = catalog_hash
    inputs["selection_freeze_sidecar"] = {
        "path": str(freeze_audit["freeze_sidecar_path"]),
        "sha256": freeze_audit["freeze_sidecar_sha256"],
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "phase": "2023_authoritative_selection_handoff",
        "contract_sha256": freeze_audit["contract_sha256"],
        "training_catalog_sha256": catalog_hash,
        "market_candidate_authority_sha256": arm["authority_sha256"],
        "selected_candidate": selected,
        "baseline": baseline,
        "downstream_lambda": downstream,
        "live_update_lambda": live,
        "promotion_eligible": metadata["promotion_eligible"],
        "diagnostic_fallback": metadata["diagnostic_fallback"],
        "selection_mode": metadata["selection_mode"],
        "primary_selection": metadata["primary_selection"],
        "selection_prediction_audit": {
            "base": arm["base_prediction_audit"],
            "round": arm["round_prediction_audit"],
        },
        "selection_prediction_hash_bindings": arm[
            "selection_prediction_hash_bindings"
        ],
        "evaluation_year_outcome_quote_or_label_files_read": [],
        "inputs": inputs,
        "outputs": {
            "runner_selected_spec": {"path": str(spec_path), "sha256": spec_hash},
            "runner_selected_spec_sidecar": {
                "path": str(spec_sidecar_path),
                "sha256": sha256_file(spec_sidecar_path),
            },
            "selected_handoff_manifest": {
                "path": str(handoff_path),
                "sidecar_path": str(handoff_sidecar_path),
                "integrity": "canonical self-hash plus external file-hash sidecar",
            },
        },
        "validations": [
            "selection freeze file sidecar and canonical payload self-hash",
            "selection frozen-spec canonical self-hash",
            "locked contract hash across every authority",
            "runner training catalog canonical hash and selectable successful candidate",
            "market authority physical hash across freeze, lambdas, manifest, and audit",
            "frozen downstream/course-fit and live lambda equality",
            "2023-only base and R1-R4 selection prediction closure and hashes",
            "diagnostic fallback cannot be upgraded to promotion eligible",
        ],
    }
    manifest["manifest_payload_sha256"] = canonical_payload_hash(manifest)
    _write_immutable(handoff_path, _json_bytes(manifest))
    _write_immutable(
        handoff_sidecar_path,
        f"{sha256_file(handoff_path)}  {handoff_path.name}\n".encode("utf-8"),
    )
    return manifest


def _validate_json_sidecar_and_self_hash(
    path: Path,
    *,
    self_hash_field: str,
    label: str,
) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"{label} lacks its SHA-256 sidecar")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError(f"{label} sidecar is empty")
    _same_hash(tokens[0], sha256_file(path), f"{label} sidecar")
    payload = _read_json(path, label)
    recorded = str(payload.get(self_hash_field, ""))
    hash_input = dict(payload)
    hash_input.pop(self_hash_field, None)
    _same_hash(recorded, canonical_payload_hash(hash_input), f"{label} self")
    return payload


def validate_selected_handoff_manifest(path: str | Path) -> dict[str, Any]:
    """Validate the handoff and every bound 2023 input without label reads."""

    source = Path(path).resolve()
    payload = _validate_json_sidecar_and_self_hash(
        source,
        self_hash_field="manifest_payload_sha256",
        label="selected handoff manifest",
    )
    if not _as_bool(payload.get("research_only"), "handoff research_only"):
        raise ValueError("handoff is not research-only")
    if _as_bool(payload.get("production_files_written"), "handoff production flag"):
        raise ValueError("handoff reports a production write")
    for name, record in payload.get("inputs", {}).items():
        if not isinstance(record, Mapping):
            raise ValueError(f"handoff input record {name} is malformed")
        bound_path = Path(str(record.get("path", ""))).resolve()
        if not bound_path.is_file():
            raise ValueError(f"handoff input {name} no longer exists")
        _same_hash(record.get("sha256"), sha256_file(bound_path), f"handoff input {name}")
    spec_record = payload.get("outputs", {}).get("runner_selected_spec", {})
    spec_path = Path(str(spec_record.get("path", ""))).resolve()
    _same_hash(
        spec_record.get("sha256"), sha256_file(spec_path), "handoff runner selected spec"
    )
    spec_sidecar_record = payload.get("outputs", {}).get(
        "runner_selected_spec_sidecar", {}
    )
    spec_sidecar = Path(str(spec_sidecar_record.get("path", ""))).resolve()
    _same_hash(
        spec_sidecar_record.get("sha256"),
        sha256_file(spec_sidecar),
        "handoff runner spec sidecar",
    )
    sidecar_token = spec_sidecar.read_text(encoding="utf-8").strip().split()[0]
    _same_hash(sidecar_token, sha256_file(spec_path), "runner selected spec sidecar")
    spec = _read_json(spec_path, "runner selected spec")
    for field in (
        "candidate",
        "selected_arm",
        "downstream_lambda",
        "live_update_lambda",
        "catalog_sha256",
        "contract_sha256",
        "promotion_eligible",
    ):
        if str(spec.get(field)) != str(
            payload.get(
                {
                    "candidate": "selected_candidate",
                    "selected_arm": "selected_candidate",
                    "catalog_sha256": "training_catalog_sha256",
                }.get(field, field)
            )
        ):
            raise ValueError(f"runner selected spec disagrees with handoff for {field}")
    return payload


def _resolve_output_record(
    owner_path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    _require_fields(record, ("path", "sha256"), label)
    path = _recorded_path(owner_path, record["path"])
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    _same_hash(record["sha256"], sha256_file(path), label)
    return path


def _year_from_event_key(values: pd.Series, label: str) -> pd.Series:
    years = pd.to_numeric(
        values.astype(str).str.extract(
            r"(?:^|\|)(20\d{2})(?:\||$)", expand=False
        ),
        errors="coerce",
    )
    if years.isna().any():
        raise ValueError(f"{label} contains event keys without a parseable year")
    return years.astype(int)


def _read_prediction_panel(
    path: Path,
    *,
    label: str,
    round_level: bool,
    allowed_years: set[int],
    expected_candidates: set[str] | None = None,
) -> pd.DataFrame:
    required = ["candidate", "event_key", "dg_id_key", "prediction"]
    if round_level:
        required.insert(3, "round_num")
    schema = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(required).difference(schema))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    frame = pd.read_parquet(path, columns=required)
    if frame.empty:
        raise ValueError(f"{label} is empty")
    frame["candidate"] = frame["candidate"].astype(str)
    years = _year_from_event_key(frame["event_key"], label)
    found_years = set(years.unique())
    if not found_years or not found_years.issubset(allowed_years):
        raise ValueError(
            f"{label} has years outside {sorted(allowed_years)}: {sorted(found_years)}"
        )
    if expected_candidates is not None and set(frame["candidate"]) != expected_candidates:
        raise ValueError(f"{label} candidate closure mismatch")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if not frame["prediction"].map(math.isfinite).all():
        raise ValueError(f"{label} contains a non-finite prediction")
    keys = ["candidate", "event_key", "dg_id_key"]
    if round_level:
        frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
        if frame["round_num"].isna().any():
            raise ValueError(f"{label} contains an invalid round")
        frame["round_num"] = frame["round_num"].astype(int)
        rounds = set(frame["round_num"])
        if rounds != {1, 2, 3, 4}:
            raise ValueError(f"{label} must include R1-R4; found {sorted(rounds)}")
        keys.append("round_num")
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate candidate keys")
    return frame


def _assert_exact_pair_closure(
    frame: pd.DataFrame,
    *,
    candidates: set[str],
    round_level: bool,
    label: str,
) -> None:
    if set(frame["candidate"]) != candidates:
        raise ValueError(f"{label} does not contain exactly selected + baseline")
    natural_key = ["event_key", "dg_id_key"]
    if round_level:
        natural_key.append("round_num")
    counts = frame.groupby(natural_key, observed=True)["candidate"].agg(
        lambda values: frozenset(values)
    )
    if not counts.map(lambda value: value == frozenset(candidates)).all():
        raise ValueError(f"{label} lacks exact selected/baseline closure at some keys")


def _filter_selection_mapper_window(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop Jan-Mar control warmup rows absent from chronological challenger OOF."""

    dates = pd.to_datetime(
        frame["event_key"].astype(str).str.extract(
            r"(20\d{2}-\d{2}-\d{2})(?!.*20\d{2}-\d{2}-\d{2})",
            expand=False,
        ),
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError(f"{label} contains an event key without an ISO event date")
    keep = dates.between(pd.Timestamp("2023-04-01"), pd.Timestamp("2023-12-31"))
    filtered = frame.loc[keep].copy()
    if filtered.empty:
        raise ValueError(f"{label} has no Apr-Dec 2023 mapper/selection rows")
    return filtered


def _validate_runner_evaluation_manifest(
    path: Path,
    *,
    handoff: Mapping[str, Any],
    runner_spec_path: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = _validate_json_sidecar_and_self_hash(
        path,
        self_hash_field="manifest_payload_sha256",
        label="runner evaluation manifest",
    )
    if not _as_bool(manifest.get("research_only"), "runner research_only"):
        raise ValueError("runner evaluation is not research-only")
    if _as_bool(manifest.get("production_files_written"), "runner production flag"):
        raise ValueError("runner evaluation reports a production write")
    if str(manifest.get("phase")) != "evaluate_prediction_only":
        raise ValueError("runner evaluation manifest has the wrong phase")
    selected = str(handoff["selected_candidate"])
    if str(manifest.get("selected_candidate")) != selected:
        raise ValueError("runner refit candidate differs from frozen handoff")
    if str(manifest.get("selected_arm")) != selected:
        raise ValueError("runner emitted an arm name different from the frozen candidate")
    _same_hash(manifest.get("contract_sha256"), handoff["contract_sha256"], "runner contract")
    _same_hash(
        manifest.get("candidate_catalog_sha256"),
        handoff["training_catalog_sha256"],
        "runner catalog",
    )
    for field in ("downstream_lambda", "live_update_lambda"):
        if not math.isclose(float(manifest.get(field)), float(handoff[field]), abs_tol=1e-12):
            raise ValueError(f"runner {field} differs from frozen handoff")
    if _as_bool(manifest.get("promotion_eligible"), "runner promotion eligibility") != _as_bool(
        handoff.get("promotion_eligible"), "handoff promotion eligibility"
    ):
        raise ValueError("runner promotion eligibility differs from frozen handoff")
    if _as_bool(manifest.get("diagnostic_fallback"), "runner diagnostic fallback") != _as_bool(
        handoff.get("diagnostic_fallback"), "handoff diagnostic fallback"
    ):
        raise ValueError("runner diagnostic-fallback flag differs from frozen handoff")
    if str(manifest.get("selection_mode")) != str(handoff.get("selection_mode")):
        raise ValueError("runner selection mode differs from frozen handoff")
    # The runner records the CLI argument verbatim.  A relative CLI path is
    # therefore relative to the runner's working directory (the repo root),
    # not to the newly-created evaluation output directory.
    recorded_spec = Path(str(manifest.get("selected_spec_path", ""))).resolve()
    if recorded_spec != runner_spec_path.resolve():
        raise ValueError("runner manifest selected-spec path mismatch")
    _same_hash(
        manifest.get("selected_spec_sha256"),
        sha256_file(runner_spec_path),
        "runner selected spec",
    )
    if manifest.get("post_2023_target_columns_projected") != []:
        raise ValueError("runner projected post-2023 target columns")
    if _as_bool(manifest.get("post_2023_outcomes_read"), "runner outcomes-read flag"):
        raise ValueError("runner reports reading post-2023 outcomes")
    if set(int(x) for x in manifest.get("prediction_years", ())) != {2024, 2025, 2026}:
        raise ValueError("runner prediction years differ from 2024-26")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("runner manifest lacks output hashes")
    resolved: dict[str, Path] = {}
    for name, record in outputs.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"runner output {name} is malformed")
        resolved[name] = _resolve_output_record(path, record, label=f"runner output {name}")
    if "raw_core_predictions" not in resolved:
        raise ValueError("runner manifest does not bind raw-core predictions")
    return manifest, resolved["raw_core_predictions"]


def _validate_adapter_apply_authority(
    manifest_path: Path,
    audit_path: Path,
    *,
    handoff: Mapping[str, Any],
    runner_raw_core_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    manifest = _validate_json_sidecar_and_self_hash(
        manifest_path,
        self_hash_field="manifest_payload_sha256",
        label="adapter apply manifest",
    )
    audit = _validate_json_sidecar_and_self_hash(
        audit_path,
        self_hash_field="audit_payload_sha256",
        label="adapter apply audit",
    )
    if not _as_bool(manifest.get("research_only"), "adapter research_only"):
        raise ValueError("adapter apply is not research-only")
    if _as_bool(manifest.get("production_files_written"), "adapter production flag"):
        raise ValueError("adapter apply reports a production write")
    if str(manifest.get("phase")) != "2024_2026_frozen_label_blind_apply":
        raise ValueError("adapter apply manifest has the wrong phase")
    selected = str(handoff["selected_candidate"])
    baseline = str(handoff["baseline"])
    for payload, label in ((manifest, "adapter manifest"), (audit, "adapter audit")):
        if str(payload.get("selected_candidate")) != selected:
            raise ValueError(f"{label} selected candidate mismatch")
        if str(payload.get("control_candidate")) != baseline:
            raise ValueError(f"{label} control candidate mismatch")
        _same_hash(payload.get("contract_sha256"), handoff["contract_sha256"], f"{label} contract")
        if not math.isclose(
            float(payload.get("downstream_lambda")),
            float(handoff["downstream_lambda"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} downstream lambda mismatch")
        if not math.isclose(
            float(payload.get("live_lambda")),
            float(handoff["live_update_lambda"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label} live lambda mismatch")
        if payload.get("target_columns_read") != []:
            raise ValueError(f"{label} reports reading evaluation targets")
        if str(payload.get("input_prediction_stage")) != (
            "raw_base_core_before_downstream_and_live"
        ):
            raise ValueError(f"{label} did not consume raw-core predictions")
        if _as_bool(
            payload.get("input_downstream_lambda_applied"),
            f"{label} input downstream flag",
        ):
            raise ValueError(f"{label} input already had downstream lambda applied")
        if _as_bool(
            payload.get("input_live_update_lambda_applied"),
            f"{label} input live flag",
        ):
            raise ValueError(f"{label} input already had live lambda applied")
    _validate_recorded_artifact(
        manifest_path,
        manifest,
        prefix="evaluation_arm_audit",
        actual_path=audit_path,
    )
    sources = manifest.get("source_artifacts", {})
    raw_record = sources.get("runner_raw_core_predictions", {})
    _require_fields(raw_record, ("path", "sha256"), "adapter raw-core source")
    raw_path = Path(str(raw_record["path"])).resolve()
    if raw_path != runner_raw_core_path.resolve():
        raise ValueError("adapter did not consume the runner-authorized raw core")
    _same_hash(raw_record["sha256"], sha256_file(raw_path), "adapter raw-core source")
    frozen_record = sources.get("frozen_lambdas", {})
    frozen_handoff = handoff["inputs"]["frozen_lambdas"]
    _same_hash(
        frozen_record.get("sha256"), frozen_handoff["sha256"], "adapter frozen lambdas"
    )
    outputs = manifest.get("outputs", {})
    base_path = _resolve_output_record(
        manifest_path,
        outputs.get("evaluation_base_predictions", {}),
        label="adapter evaluation base predictions",
    )
    round_path = _resolve_output_record(
        manifest_path,
        outputs.get("evaluation_round_predictions", {}),
        label="adapter evaluation round predictions",
    )
    for prefix, output_path in (
        ("evaluation_base_predictions", base_path),
        ("evaluation_round_predictions", round_path),
    ):
        _validate_recorded_artifact(
            audit_path, audit, prefix=prefix, actual_path=output_path
        )
    return manifest, base_path, round_path


def seal_prediction_panels(
    *,
    selected_handoff_manifest_path: str | Path,
    runner_selected_spec_path: str | Path,
    runner_evaluation_manifest_path: str | Path,
    adapter_apply_manifest_path: str | Path,
    adapter_apply_audit_path: str | Path,
    selection_base_predictions_path: str | Path,
    selection_round_predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal 2023 OOF + 2024-26 label-blind arms for the market evaluator."""

    handoff_path = Path(selected_handoff_manifest_path).resolve()
    spec_path = Path(runner_selected_spec_path).resolve()
    runner_manifest_path = Path(runner_evaluation_manifest_path).resolve()
    adapter_manifest_path = Path(adapter_apply_manifest_path).resolve()
    adapter_audit_path = Path(adapter_apply_audit_path).resolve()
    selection_base_path = Path(selection_base_predictions_path).resolve()
    selection_round_path = Path(selection_round_predictions_path).resolve()
    handoff = validate_selected_handoff_manifest(handoff_path)
    recorded_spec_path = Path(
        str(handoff["outputs"]["runner_selected_spec"]["path"])
    ).resolve()
    if recorded_spec_path != spec_path:
        raise ValueError("seal-predictions runner spec differs from handoff output")
    for key, supplied in (
        ("selection_base_predictions", selection_base_path),
        ("selection_round_predictions", selection_round_path),
    ):
        record = handoff["inputs"][key]
        if Path(str(record["path"])).resolve() != supplied:
            raise ValueError(f"seal-predictions {key} path differs from handoff")
        _same_hash(record["sha256"], sha256_file(supplied), f"handoff {key}")

    _, runner_raw_core = _validate_runner_evaluation_manifest(
        runner_manifest_path, handoff=handoff, runner_spec_path=spec_path
    )
    _, eval_base_path, eval_round_path = _validate_adapter_apply_authority(
        adapter_manifest_path,
        adapter_audit_path,
        handoff=handoff,
        runner_raw_core_path=runner_raw_core,
    )
    selected = str(handoff["selected_candidate"])
    baseline = str(handoff["baseline"])
    pair = {selected, baseline}
    selection_authority_candidates = set(
        handoff["selection_prediction_audit"]["base"]["candidates"]
    )
    selection_base_all = _read_prediction_panel(
        selection_base_path,
        label="selection base source",
        round_level=False,
        allowed_years={2023},
        expected_candidates=selection_authority_candidates,
    )
    selection_round_all = _read_prediction_panel(
        selection_round_path,
        label="selection round source",
        round_level=True,
        allowed_years={2023},
        expected_candidates=selection_authority_candidates,
    )
    selection_base = selection_base_all.loc[
        selection_base_all["candidate"].isin(pair)
    ].copy()
    selection_round = selection_round_all.loc[
        selection_round_all["candidate"].isin(pair)
    ].copy()
    selection_base_pair_rows_before_window = int(len(selection_base))
    selection_round_pair_rows_before_window = int(len(selection_round))
    # The exact cached control legitimately includes Jan-Mar warmup events,
    # while challenger chronology begins with the Apr-Jun mapper window.  The
    # frozen evaluator consumes Apr-Dec OOF rows, so seal that same window.
    selection_base = _filter_selection_mapper_window(
        selection_base, "filtered 2023 selection base"
    )
    selection_round = _filter_selection_mapper_window(
        selection_round, "filtered 2023 selection round"
    )
    selection_window_audit = {
        "window_start": "2023-04-01",
        "window_end": "2023-12-31",
        "reason": (
            "control has Jan-Mar warmup rows; challenger chronology and final mapper "
            "authority begin 2023-04-01"
        ),
        "base_source_rows_all_candidates": int(len(selection_base_all)),
        "round_source_rows_all_candidates": int(len(selection_round_all)),
        "base_selected_control_rows_before_window": selection_base_pair_rows_before_window,
        "round_selected_control_rows_before_window": selection_round_pair_rows_before_window,
        "base_warmup_rows_dropped": int(
            selection_base_pair_rows_before_window - len(selection_base)
        ),
        "round_warmup_rows_dropped": int(
            selection_round_pair_rows_before_window - len(selection_round)
        ),
        "base_selected_control_rows_sealed": int(len(selection_base)),
        "round_selected_control_rows_sealed": int(len(selection_round)),
    }
    _assert_exact_pair_closure(
        selection_base,
        candidates=pair,
        round_level=False,
        label="filtered 2023 selection base",
    )
    _assert_exact_pair_closure(
        selection_round,
        candidates=pair,
        round_level=True,
        label="filtered 2023 selection round",
    )
    evaluation_base = _read_prediction_panel(
        eval_base_path,
        label="adapter evaluation base",
        round_level=False,
        allowed_years={2024, 2025, 2026},
        expected_candidates=pair,
    )
    evaluation_round = _read_prediction_panel(
        eval_round_path,
        label="adapter evaluation round",
        round_level=True,
        allowed_years={2024, 2025, 2026},
        expected_candidates=pair,
    )
    if set(_year_from_event_key(evaluation_base["event_key"], "evaluation base")) != {
        2024,
        2025,
        2026,
    }:
        raise ValueError("evaluation base does not span every locked evaluation year")
    _assert_exact_pair_closure(
        evaluation_base,
        candidates=pair,
        round_level=False,
        label="2024-26 evaluation base",
    )
    _assert_exact_pair_closure(
        evaluation_round,
        candidates=pair,
        round_level=True,
        label="2024-26 evaluation round",
    )
    if set(selection_base["event_key"]).intersection(evaluation_base["event_key"]):
        raise ValueError("selection and evaluation base event keys overlap")
    if set(selection_round["event_key"]).intersection(evaluation_round["event_key"]):
        raise ValueError("selection and evaluation round event keys overlap")

    combined_base = pd.concat([selection_base, evaluation_base], ignore_index=True)
    combined_round = pd.concat([selection_round, evaluation_round], ignore_index=True)
    combined_base = combined_base.sort_values(
        ["candidate", "event_key", "dg_id_key"], kind="mergesort"
    ).reset_index(drop=True)
    combined_round = combined_round.sort_values(
        ["candidate", "event_key", "round_num", "dg_id_key"], kind="mergesort"
    ).reset_index(drop=True)
    if combined_base.duplicated(["candidate", "event_key", "dg_id_key"]).any():
        raise ValueError("combined base predictions contain overlapping keys")
    if combined_round.duplicated(
        ["candidate", "event_key", "dg_id_key", "round_num"]
    ).any():
        raise ValueError("combined round predictions contain overlapping keys")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_output = output / "combined_base_predictions.parquet"
    round_output = output / "combined_round_predictions.parquet"
    authority_path = output / "prediction_authority.json"
    authority_sidecar = output / "prediction_authority.sha256"
    for path in (base_output, round_output, authority_path, authority_sidecar):
        if path.exists():
            raise FileExistsError(f"immutable prediction authority output exists: {path}")
    combined_base.to_parquet(base_output, index=False)
    combined_round.to_parquet(round_output, index=False)
    sources = {
        "selected_handoff_manifest": handoff_path,
        "runner_selected_spec": spec_path,
        "runner_evaluation_manifest": runner_manifest_path,
        "runner_evaluation_manifest_sidecar": runner_manifest_path.with_suffix(".sha256"),
        "adapter_apply_manifest": adapter_manifest_path,
        "adapter_apply_manifest_sidecar": adapter_manifest_path.with_suffix(".sha256"),
        "adapter_apply_audit": adapter_audit_path,
        "adapter_apply_audit_sidecar": adapter_audit_path.with_suffix(".sha256"),
        "selection_base_predictions": selection_base_path,
        "selection_round_predictions": selection_round_path,
        "evaluation_base_predictions": eval_base_path,
        "evaluation_round_predictions": eval_round_path,
        "runner_raw_core_predictions": runner_raw_core,
    }
    authority: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "production_files_written": False,
        "phase": "sealed_joint_market_prediction_panels",
        "contract_sha256": handoff["contract_sha256"],
        "training_catalog_sha256": handoff["training_catalog_sha256"],
        "market_candidate_authority_sha256": handoff[
            "market_candidate_authority_sha256"
        ],
        "selection_freeze_sha256": handoff["inputs"]["selection_freeze"]["sha256"],
        "selected_handoff_manifest_sha256": sha256_file(handoff_path),
        "selected_candidate": selected,
        "baseline": baseline,
        "downstream_lambda": float(handoff["downstream_lambda"]),
        "live_update_lambda": float(handoff["live_update_lambda"]),
        "promotion_eligible": _as_bool(
            handoff["promotion_eligible"], "handoff promotion eligibility"
        ),
        "diagnostic_fallback": _as_bool(
            handoff["diagnostic_fallback"], "handoff diagnostic fallback"
        ),
        "selection_mode": handoff["selection_mode"],
        "candidate_closure": sorted(pair),
        "selection_years": [2023],
        "evaluation_years": [2024, 2025, 2026],
        "selection_mapper_window_audit": selection_window_audit,
        "lambda_application": {
            "runner_raw_core_downstream_applied": False,
            "runner_raw_core_live_applied": False,
            "adapter_downstream_application_count": 1,
            "adapter_live_application_count_r1": 0,
            "adapter_live_application_count_r2_r4": 1,
        },
        "evaluation_outcome_quote_or_label_columns_read": [],
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "outputs": {
            "combined_base_predictions": {
                "path": str(base_output),
                "sha256": sha256_file(base_output),
                "rows": int(len(combined_base)),
            },
            "combined_round_predictions": {
                "path": str(round_output),
                "sha256": sha256_file(round_output),
                "rows": int(len(combined_round)),
            },
        },
    }
    authority["authority_payload_sha256"] = canonical_payload_hash(authority)
    _write_immutable(authority_path, _json_bytes(authority))
    _write_immutable(
        authority_sidecar,
        f"{sha256_file(authority_path)}  {authority_path.name}\n".encode("utf-8"),
    )
    return authority


def validate_prediction_authority(
    path: str | Path,
    *,
    combined_base_predictions_path: str | Path | None = None,
    combined_round_predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate sealed prediction hashes before an evaluator reads test quotes."""

    source = Path(path).resolve()
    authority = _validate_json_sidecar_and_self_hash(
        source,
        self_hash_field="authority_payload_sha256",
        label="prediction authority",
    )
    if not _as_bool(authority.get("research_only"), "prediction authority research flag"):
        raise ValueError("prediction authority is not research-only")
    if _as_bool(
        authority.get("production_files_written"), "prediction authority production flag"
    ):
        raise ValueError("prediction authority reports a production write")
    for name, record in authority.get("sources", {}).items():
        bound = Path(str(record.get("path", ""))).resolve()
        _same_hash(record.get("sha256"), sha256_file(bound), f"prediction source {name}")
    handoff_record = authority.get("sources", {}).get("selected_handoff_manifest")
    if not isinstance(handoff_record, Mapping):
        raise ValueError("prediction authority does not bind the selected handoff")
    nested_handoff = validate_selected_handoff_manifest(handoff_record["path"])
    _same_hash(
        authority.get("contract_sha256"),
        nested_handoff.get("contract_sha256"),
        "prediction authority nested contract",
    )
    _same_hash(
        authority.get("training_catalog_sha256"),
        nested_handoff.get("training_catalog_sha256"),
        "prediction authority nested training catalog",
    )
    _same_hash(
        authority.get("market_candidate_authority_sha256"),
        nested_handoff.get("market_candidate_authority_sha256"),
        "prediction authority nested market authority",
    )
    _same_hash(
        authority.get("selection_freeze_sha256"),
        nested_handoff.get("inputs", {}).get("selection_freeze", {}).get("sha256"),
        "prediction authority nested selection freeze",
    )
    if str(authority.get("selected_candidate")) != str(
        nested_handoff.get("selected_candidate")
    ):
        raise ValueError("prediction authority selected candidate differs from handoff")
    supplied = {
        "combined_base_predictions": combined_base_predictions_path,
        "combined_round_predictions": combined_round_predictions_path,
    }
    for name, record in authority.get("outputs", {}).items():
        bound = Path(str(record.get("path", ""))).resolve()
        if supplied.get(name) is not None and bound != Path(str(supplied[name])).resolve():
            raise ValueError(f"supplied {name} differs from prediction authority")
        _same_hash(record.get("sha256"), sha256_file(bound), f"prediction output {name}")
    application = authority.get("lambda_application", {})
    expected_application = {
        "runner_raw_core_downstream_applied": False,
        "runner_raw_core_live_applied": False,
        "adapter_downstream_application_count": 1,
        "adapter_live_application_count_r1": 0,
        "adapter_live_application_count_r2_r4": 1,
    }
    if application != expected_application:
        raise ValueError("prediction authority does not prove exactly-once lambda application")
    return authority


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    build = sub.add_parser("build", help="write the runner-selected specification")
    build.add_argument("--selection-freeze", type=Path, required=True)
    build.add_argument("--training-candidate-catalog", type=Path, required=True)
    build.add_argument("--training-manifest", type=Path, required=True)
    build.add_argument("--frozen-lambdas", type=Path, required=True)
    build.add_argument("--market-candidate-authority", type=Path, required=True)
    build.add_argument("--selection-arm-manifest", type=Path, required=True)
    build.add_argument("--selection-arm-audit", type=Path, required=True)
    build.add_argument("--selection-base-predictions", type=Path, required=True)
    build.add_argument("--selection-round-predictions", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    seal = sub.add_parser(
        "seal-predictions", help="combine 2023 selection and 2024-26 frozen arms"
    )
    seal.add_argument("--selected-handoff-manifest", type=Path, required=True)
    seal.add_argument("--runner-selected-spec", type=Path, required=True)
    seal.add_argument("--runner-evaluation-manifest", type=Path, required=True)
    seal.add_argument("--adapter-apply-manifest", type=Path, required=True)
    seal.add_argument("--adapter-apply-audit", type=Path, required=True)
    seal.add_argument("--selection-base-predictions", type=Path, required=True)
    seal.add_argument("--selection-round-predictions", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.phase == "build":
        manifest = build_selected_handoff(
            selection_freeze_path=args.selection_freeze,
            training_candidate_catalog_path=args.training_candidate_catalog,
            training_manifest_path=args.training_manifest,
            frozen_lambdas_path=args.frozen_lambdas,
            market_candidate_authority_path=args.market_candidate_authority,
            selection_arm_manifest_path=args.selection_arm_manifest,
            selection_arm_audit_path=args.selection_arm_audit,
            selection_base_predictions_path=args.selection_base_predictions,
            selection_round_predictions_path=args.selection_round_predictions,
            output_dir=args.output_dir,
            contract_path=args.contract,
        )
        message = {
            "selected_candidate": manifest["selected_candidate"],
            "promotion_eligible": manifest["promotion_eligible"],
            "handoff_manifest": str(
                (Path(args.output_dir) / "selected_handoff_manifest.json").resolve()
            ),
        }
    else:
        authority = seal_prediction_panels(
            selected_handoff_manifest_path=args.selected_handoff_manifest,
            runner_selected_spec_path=args.runner_selected_spec,
            runner_evaluation_manifest_path=args.runner_evaluation_manifest,
            adapter_apply_manifest_path=args.adapter_apply_manifest,
            adapter_apply_audit_path=args.adapter_apply_audit,
            selection_base_predictions_path=args.selection_base_predictions,
            selection_round_predictions_path=args.selection_round_predictions,
            output_dir=args.output_dir,
        )
        message = {
            "selected_candidate": authority["selected_candidate"],
            "promotion_eligible": authority["promotion_eligible"],
            "prediction_authority": str(
                (Path(args.output_dir) / "prediction_authority.json").resolve()
            ),
        }
    print(
        json.dumps(message, indent=2)
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_CONTRACT_SHA256",
    "build_selected_handoff",
    "canonical_payload_hash",
    "seal_prediction_panels",
    "sha256_file",
    "training_catalog_hash",
    "validate_prediction_authority",
    "validate_selected_handoff_manifest",
]

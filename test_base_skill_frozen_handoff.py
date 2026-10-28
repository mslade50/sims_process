from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from archive import base_skill_frozen_handoff as handoff


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _build_chain(tmp_path: Path, monkeypatch) -> dict[str, Path | dict | str]:
    candidate = "complete__ridge__total__w0__annual__a1"
    source_dir = tmp_path / "completion"
    selection_dir = tmp_path / "selection"
    frozen_dir = tmp_path / "frozen"
    for directory in (source_dir, selection_dir, frozen_dir):
        directory.mkdir()

    source_prediction = source_dir / "player_event_predictions.parquet"
    pd.DataFrame(
        {
            "candidate": [
                candidate,
                "current_architecture_pt_v2_approx_control",
            ],
            "canonical_event_key": ["pga:R2025009", "pga:R2025009"],
            "dg_id_key": ["7", "7"],
            "prediction": [0.25, 0.10],
        }
    ).to_parquet(source_prediction, index=False)
    catalog = source_dir / "candidate_catalog.csv"
    pd.DataFrame(
        {
            "name": [candidate],
            "selection_eligible": [True],
            "sensitivity_only": [False],
            "control": [False],
        }
    ).to_csv(catalog, index=False)
    frozen_catalog = source_dir / "candidate_catalog_frozen.json"
    _write_json(frozen_catalog, {"contract": "synthetic_catalog"})
    frozen_catalog_hash = _sha(frozen_catalog)
    coverage_by_candidate_year = source_dir / "coverage_by_candidate_year.csv"
    coverage_by_candidate_year.write_text(
        "candidate,year\nsynthetic,2021\n", encoding="utf-8"
    )
    completed_study_audit_files = {
        "fit_log": source_dir / "fit_log.csv",
        "envelope_admission_by_candidate": (
            source_dir / "regression_envelope_admission_by_candidate.csv"
        ),
        "envelope_admission_by_fold": (
            source_dir / "regression_envelope_admission_by_fold.csv"
        ),
    }
    for name, path in completed_study_audit_files.items():
        path.write_text(f"synthetic {name}\n", encoding="utf-8")

    core_bridge = source_dir / "legacy_core_bridge.parquet"
    pd.DataFrame({"dg_id": [1], "downstream_preserved_total": [0.1]}).to_parquet(
        core_bridge, index=False
    )
    completion_input_files = {
        "snapshots": source_dir / "snapshots.parquet",
        "all_tour_db": source_dir / "all_tour.db",
        "clean_img": source_dir / "clean_img.parquet",
        "round_panel": source_dir / "round_panel.parquet",
        "weather_coefficients": source_dir / "weather_coefficients.csv",
        "regression_envelope_gate_audit": source_dir / "gate_audit.json",
        "regression_envelope_gate_candidates": source_dir / "gate_candidates.csv",
        "regression_envelope_gate_folds": source_dir / "gate_folds.csv",
        "core_bridge": core_bridge,
    }
    for name, path in completion_input_files.items():
        if name != "core_bridge":
            path.write_text(f"synthetic {name}\n", encoding="utf-8")
    builder_source = source_dir / "builder_source.py"
    builder_source.write_text("# synthetic exact builder\n", encoding="utf-8")
    builder_copy = source_dir / "executed_builder_source.py"
    builder_copy.write_bytes(builder_source.read_bytes())

    def completion_record(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    builder_record = completion_record(builder_source)
    builder_copy_record = completion_record(builder_copy)
    completion_inputs = {
        name: completion_record(path)
        for name, path in completion_input_files.items()
    }

    source_manifest = source_dir / "manifest.json"
    source_payload = {
        "complete": True,
        "research_only": True,
        "production_files_changed": False,
        "execution_provenance": {
            "contract": "base_skill_execution_provenance_v1",
            "parsed_args": {"reuse_augmented": None},
            "all_unchanged": True,
            "source_unchanged": True,
            "source_copy_unchanged": True,
            "source_copy_matches": True,
            "source_before": builder_record,
            "source_after": dict(builder_record),
            "executed_source_copy_before": builder_copy_record,
            "executed_source_copy_after": dict(builder_copy_record),
            "inputs_before": completion_inputs,
            "inputs_after": completion_inputs,
            "input_unchanged": {name: True for name in completion_inputs},
        },
        "contract": {
            "challengers_use_legacy_skill": False,
            "closing_prices_used_as_features": False,
            "selection_uses_sealed": False,
        },
        "frozen_candidate_catalog_sha256": frozen_catalog_hash,
        "inputs": completion_inputs,
        "outputs": {
            builder_copy.name: {
                "path": str(builder_copy.resolve()),
                "sha256": _sha(builder_copy),
            },
            source_prediction.name: {
                "path": str(source_prediction.resolve()),
                "sha256": _sha(source_prediction),
            },
            catalog.name: {
                "path": str(catalog.resolve()),
                "sha256": _sha(catalog),
            },
            frozen_catalog.name: {
                "path": str(frozen_catalog.resolve()),
                "sha256": frozen_catalog_hash,
            },
            coverage_by_candidate_year.name: {
                "path": str(coverage_by_candidate_year.resolve()),
                "sha256": _sha(coverage_by_candidate_year),
            },
            **{
                path.name: {
                    "path": str(path.resolve()),
                    "sha256": _sha(path),
                }
                for path in completed_study_audit_files.values()
            },
        },
    }
    _write_json(source_manifest, source_payload)

    artifact = selection_dir / "audit.json"
    _write_json(artifact, {"synthetic": True})
    evaluation_cache_dir = tmp_path / "evaluation_cache"
    evaluation_cache_dir.mkdir()
    normalized_quotes = evaluation_cache_dir / "normalized_matchup_quotes.parquet"
    pd.DataFrame({"synthetic": [1]}).to_parquet(normalized_quotes, index=False)
    feature_snapshots = evaluation_cache_dir / "feature_snapshots.parquet"
    pd.DataFrame({"synthetic": [1]}).to_parquet(feature_snapshots, index=False)
    evaluation_cache_manifest = evaluation_cache_dir / "manifest.json"
    _write_json(evaluation_cache_manifest, {"complete": True})
    evaluation_cache_authority_inputs = {
        "event_panel": evaluation_cache_dir / "player_event_outcomes.parquet",
        "legacy_core_bridge": core_bridge,
        "odds_cache_manifest": evaluation_cache_dir / "odds_cache_manifest.json",
        "odds_event_crosswalk": evaluation_cache_dir / "odds_event_crosswalk.parquet",
        "round_panel": evaluation_cache_dir / "player_round_panel.parquet",
    }
    for name, path in evaluation_cache_authority_inputs.items():
        if name != "legacy_core_bridge":
            path.write_text(f"synthetic cache authority {name}\n", encoding="utf-8")
    evaluation_cache_authority_bindings = {
        "evaluation_cache_manifest": evaluation_cache_manifest.resolve(),
        "normalized_quotes": normalized_quotes.resolve(),
        "feature_snapshots": feature_snapshots.resolve(),
        **{
            f"cache_input_{name}": path.resolve()
            for name, path in evaluation_cache_authority_inputs.items()
        },
    }
    completed_study_authority_bindings = {
        "completion_manifest": source_manifest.resolve(),
        "predictions": source_prediction.resolve(),
        "candidate_catalog": catalog.resolve(),
        "coverage_by_candidate_year": coverage_by_candidate_year.resolve(),
        "frozen_candidate_catalog": frozen_catalog.resolve(),
        **{
            name: path.resolve()
            for name, path in completed_study_audit_files.items()
        },
    }
    provenance_dir = selection_dir / "execution_provenance"
    provenance_dir.mkdir()
    discovery_source = Path(handoff.discovery.__file__).resolve()
    discovery_sources = list(
        dict.fromkeys(
            [
                discovery_source,
                *(Path(path).resolve() for path in handoff.discovery.PROVENANCE_CODE_PATHS),
            ]
        )
    )
    preserved_discovery_sources = []
    for index, source in enumerate(discovery_sources):
        preserved = provenance_dir / f"{index:02d}__{source.name}"
        preserved.write_bytes(source.read_bytes())
        preserved_discovery_sources.append(preserved)

    def selection_record(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    selection_provenance = provenance_dir / "execution_provenance.json"
    selection_provenance_payload = {
        "complete": True,
        "start_utc": "2026-08-23T00:00:00+00:00",
        "end_utc": "2026-08-23T00:01:00+00:00",
        "pid": 123,
        "cli": ["synthetic-discovery"],
        "primary_source": str(discovery_source),
        "sources": [selection_record(path) for path in discovery_sources],
        "preserved_sources": [
            selection_record(path) for path in preserved_discovery_sources
        ],
        "inputs": [
            selection_record(path)
            for path in (
                source_prediction,
                catalog,
                frozen_catalog,
                source_manifest,
                normalized_quotes,
                feature_snapshots,
                evaluation_cache_manifest,
                coverage_by_candidate_year,
                core_bridge,
                evaluation_cache_authority_inputs["event_panel"],
                evaluation_cache_authority_inputs["odds_cache_manifest"],
                evaluation_cache_authority_inputs["odds_event_crosswalk"],
                evaluation_cache_authority_inputs["round_panel"],
                *completed_study_audit_files.values(),
            )
        ],
        "verified_unchanged_before_complete_manifest": True,
    }
    _write_json(selection_provenance, selection_provenance_payload)
    selection_manifest = selection_dir / "selection_manifest.json"
    selection_payload = {
        "contract": "base_skill_discovery_selection_v1",
        "complete": True,
        "research_only": True,
        "selection_window_end": "2022-12-31",
        "selection_uses_sealed": False,
        "sealed_data_used_for_selection": False,
        "uses_closing_odds_as_feature": False,
        "promotion_gate_passed": False,
        "finalists": [],
        "diagnostic_challenger": candidate,
        "frozen_selection_parameter_contract": {"synthetic": "frozen"},
        "frozen_candidate_catalog_sha256": frozen_catalog_hash,
        "locked_diagnostic_ablation_set": {
            "selected_candidate": candidate,
            "locked_before_sealed_scoring": True,
            "may_reselect_or_promote_from_sealed": False,
        },
        "downstream_preservation_sensitivity": {
            "neutral_base_redefined": False,
            "lambda_one_is_isolation_default": True,
            "choices": [
                {
                    "candidate": candidate,
                    "selected_downstream_lambda": 0.5,
                    "selection_window_end": "2022-12-31",
                    "sealed_data_used_for_selection": False,
                }
            ],
        },
        "execution_provenance": {
            "path": str(selection_provenance.relative_to(selection_dir)),
            "start_utc": selection_provenance_payload["start_utc"],
            "end_utc": selection_provenance_payload["end_utc"],
            "pid": selection_provenance_payload["pid"],
            "verified_unchanged_before_complete_manifest": True,
        },
        "inputs": {
            "predictions": {
                "path": str(source_prediction.resolve()),
                "sha256": _sha(source_prediction),
            },
            "candidate_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": _sha(source_manifest),
            },
            "candidate_catalog": {
                "path": str(catalog.resolve()),
                "sha256": _sha(catalog),
            },
            "frozen_candidate_catalog": {
                "path": str(frozen_catalog.resolve()),
                "sha256": frozen_catalog_hash,
            },
            "normalized_quotes": {
                "path": str(normalized_quotes.resolve()),
                "sha256": _sha(normalized_quotes),
            },
            "feature_snapshots": {
                "path": str(feature_snapshots.resolve()),
                "sha256": _sha(feature_snapshots),
            },
            "evaluation_cache_manifest": {
                "path": str(evaluation_cache_manifest.resolve()),
                "sha256": _sha(evaluation_cache_manifest),
            },
            "coverage_by_candidate_year": {
                "path": str(coverage_by_candidate_year.resolve()),
                "sha256": _sha(coverage_by_candidate_year),
            },
            "core_bridge": {
                "path": str(core_bridge.resolve()),
                "sha256": _sha(core_bridge),
            },
        },
        "artifacts": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": _sha(artifact),
            },
            {
                "path": str(selection_provenance.relative_to(selection_dir)),
                "bytes": selection_provenance.stat().st_size,
                "sha256": _sha(selection_provenance),
            },
            *[
                {
                    "path": str(path.relative_to(selection_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in preserved_discovery_sources
            ],
        ],
    }
    _write_json(selection_manifest, selection_payload)

    compact_prediction = frozen_dir / "player_event_predictions.parquet"
    pd.DataFrame(
        {
            "event_key": ["pga:R2025009"],
            "dg_id": [7],
            "candidate": [candidate],
            "finalist_core": [0.25],
            "control_core": [0.10],
        }
    ).to_parquet(compact_prediction, index=False)
    compact_manifest = frozen_dir / "manifest.json"
    compact_payload = {
        "contract": "base_skill_frozen_finalist_v1",
        "frozen": True,
        "research_only": True,
        "finalist_name": None,
        "selected_model_name": candidate,
        "diagnostic_challenger": candidate,
        "selection_role": "diagnostic_challenger",
        "promotion_gate_passed": False,
        "prediction_file": compact_prediction.name,
        "prediction_sha256": _sha(compact_prediction),
        "prediction_units": "sg_per_round",
        "control_name": "current_architecture_pt_v2_approx_control",
        "control_units": "sg_per_round",
        "control_core_point_in_time": True,
        "control_core_uses_sealed_future": False,
        "selection_window_end": "2022-12-31",
        "sealed_data_used_for_selection": False,
        "frozen_before_sealed_scoring": True,
        "uses_closing_odds_as_feature": False,
        "selection_manifest": str(selection_manifest.resolve()),
        "selection_manifest_sha256": _sha(selection_manifest),
        "source_prediction_file": str(source_prediction.resolve()),
        "source_prediction_sha256": _sha(source_prediction),
    }
    _write_json(compact_manifest, compact_payload)

    monkeypatch.setattr(
        handoff.discovery,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda _: None),
    )
    monkeypatch.setattr(
        handoff.discovery,
        "validate_frozen_selection_cli",
        lambda _: {"synthetic": "frozen"},
    )
    monkeypatch.setattr(
        handoff.discovery,
        "EXPECTED_FROZEN_CATALOG_SHA256",
        frozen_catalog_hash,
    )
    monkeypatch.setattr(
        handoff.discovery,
        "EXPECTED_BUILDER_SOURCE_SHA256",
        _sha(builder_source),
    )
    monkeypatch.setattr(
        handoff.discovery,
        "validate_frozen_catalog_contract",
        lambda *_: {"synthetic": True},
    )
    monkeypatch.setattr(
        handoff.discovery,
        "validate_evaluation_cache_authority",
        lambda cache_dir, bridge_path: (
            {"complete": True},
            {"synthetic": True},
            evaluation_cache_authority_bindings,
        ),
    )
    monkeypatch.setattr(
        handoff.discovery,
        "validate_completed_study_authority",
        lambda manifest_path, predictions_path, catalog_path: (
            source_payload,
            {"synthetic": True},
            completed_study_authority_bindings,
        ),
    )
    return {
        "candidate": candidate,
        "compact_manifest": compact_manifest,
        "compact_payload": compact_payload,
        "selection_manifest": selection_manifest,
        "selection_payload": selection_payload,
        "source_manifest": source_manifest,
        "source_payload": source_payload,
        "catalog": catalog,
        "core_bridge": core_bridge,
        "selection_provenance": selection_provenance,
        "completion_inputs": completion_input_files,
        "selection_sources": discovery_sources,
        "preserved_selection_sources": preserved_discovery_sources,
        "builder_source": builder_source,
        "builder_copy": builder_copy,
        "evaluation_cache_authority_bindings": evaluation_cache_authority_bindings,
        "completed_study_authority_bindings": completed_study_authority_bindings,
    }


def _rebind_selection(chain: dict[str, Path | dict | str]) -> None:
    selection_manifest = chain["selection_manifest"]
    compact_manifest = chain["compact_manifest"]
    compact_payload = chain["compact_payload"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(compact_manifest, Path)
    assert isinstance(compact_payload, dict)
    compact_payload["selection_manifest_sha256"] = _sha(selection_manifest)
    _write_json(compact_manifest, compact_payload)


def _refresh_selection_provenance_artifact(
    chain: dict[str, Path | dict | str]
) -> None:
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    artifact = next(
        row
        for row in selection_payload["artifacts"]
        if row["path"] == str(provenance_path.relative_to(selection_manifest.parent))
    )
    artifact["sha256"] = _sha(provenance_path)
    artifact["bytes"] = provenance_path.stat().st_size
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)


def _rebind_source_manifest(chain: dict[str, Path | dict | str]) -> None:
    source_manifest = chain["source_manifest"]
    source_payload = chain["source_payload"]
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(source_manifest, Path)
    assert isinstance(source_payload, dict)
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    _write_json(source_manifest, source_payload)
    selection_payload["inputs"]["candidate_manifest"]["sha256"] = _sha(
        source_manifest
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for record in provenance["inputs"]:
        if Path(record["path"]).resolve() == source_manifest.resolve():
            stat = source_manifest.stat()
            record.update(
                {
                    "sha256": _sha(source_manifest),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    _write_json(provenance_path, provenance)
    _refresh_selection_provenance_artifact(chain)


def test_full_handoff_chain_accepts_diagnostic_and_freezes_lambda(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    result = handoff.validate_frozen_handoff_chain(chain["compact_manifest"])
    assert result.selected_name == chain["candidate"]
    assert result.selection_role == "diagnostic_challenger"
    assert result.promotion_gate_passed is False
    assert result.downstream_lambda == 0.5
    assert result.audit["full_hash_chain_verified"] is True
    assert result.audit["completed_study_authority"]["synthetic"] is True
    expected_original_sources = {
        *(Path(path).resolve() for path in chain["selection_sources"]),
        Path(chain["builder_source"]).resolve(),
    }
    assert expected_original_sources.issubset(set(result.verified_paths))
    assert set(chain["completed_study_authority_bindings"].values()).issubset(
        set(result.verified_paths)
    )


def test_handoff_rejects_catalog_tamper(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    catalog = chain["catalog"]
    assert isinstance(catalog, Path)
    catalog.write_text(catalog.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_catalog"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_requires_boolean_promotion_gate(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    selection_payload["promotion_gate_passed"] = 0
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="boolean promotion_gate_passed"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_changed_completion_source(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    source_manifest = chain["source_manifest"]
    source_payload = chain["source_payload"]
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(source_manifest, Path)
    assert isinstance(source_payload, dict)
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    source_payload["execution_provenance"]["source_unchanged"] = False
    _write_json(source_manifest, source_payload)
    selection_payload["inputs"]["candidate_manifest"]["sha256"] = _sha(
        source_manifest
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for record in provenance["inputs"]:
        if Path(record["path"]).resolve() == source_manifest.resolve():
            record.update(
                {
                    "sha256": _sha(source_manifest),
                    "bytes": source_manifest.stat().st_size,
                    "mtime_ns": source_manifest.stat().st_mtime_ns,
                }
            )
    _write_json(provenance_path, provenance)
    provenance_artifact = next(
        row
        for row in selection_payload["artifacts"]
        if row["path"] == str(provenance_path.relative_to(selection_manifest.parent))
    )
    provenance_artifact["sha256"] = _sha(provenance_path)
    provenance_artifact["bytes"] = provenance_path.stat().st_size
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="source_unchanged=true"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_unprovenanced_selection_input(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    extra = selection_manifest.parent.parent / "extra_input.csv"
    extra.write_text("x\n1\n", encoding="utf-8")
    selection_payload["inputs"]["extra_input"] = {
        "path": str(extra.resolve()),
        "sha256": _sha(extra),
    }
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="input set differs"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_self_consistent_omitted_selection_input(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    removed = selection_payload["inputs"].pop("coverage_by_candidate_year")
    removed_path = Path(removed["path"]).resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["inputs"] = [
        record
        for record in provenance["inputs"]
        if Path(record["path"]).resolve() != removed_path
    ]
    _write_json(provenance_path, provenance)
    _refresh_selection_provenance_artifact(chain)
    with pytest.raises(ValueError, match=r"missing=\['coverage_by_candidate_year'\]"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_self_consistent_omitted_selection_source(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["sources"].pop()
    provenance["preserved_sources"].pop()
    _write_json(provenance_path, provenance)
    _refresh_selection_provenance_artifact(chain)
    with pytest.raises(ValueError, match="source set is incomplete"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_omitted_completed_study_provenance_input(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    provenance_path = chain["selection_provenance"]
    authority = chain["completed_study_authority_bindings"]
    assert isinstance(provenance_path, Path)
    assert isinstance(authority, dict)
    fit_log = Path(authority["fit_log"]).resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["inputs"] = [
        row
        for row in provenance["inputs"]
        if Path(row["path"]).resolve() != fit_log
    ]
    _write_json(provenance_path, provenance)
    _refresh_selection_provenance_artifact(chain)
    with pytest.raises(ValueError, match="completed-study-authority input set"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_omitted_selection_artifact_inventory(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_payload = chain["selection_payload"]
    assert isinstance(selection_payload, dict)
    selection_payload["artifacts"] = [
        row for row in selection_payload["artifacts"] if row["path"] != "audit.json"
    ]
    selection_manifest = chain["selection_manifest"]
    assert isinstance(selection_manifest, Path)
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="artifact inventory is not exact"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_selection_artifact_path_escape(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_payload = chain["selection_payload"]
    selection_manifest = chain["selection_manifest"]
    assert isinstance(selection_payload, dict)
    assert isinstance(selection_manifest, Path)
    escaped = tmp_path / "escaped_decision_evidence.json"
    _write_json(escaped, {"forged": True})
    selection_payload["artifacts"].append(
        {
            "path": str(escaped.resolve()),
            "bytes": escaped.stat().st_size,
            "sha256": _sha(escaped),
        }
    )
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="escapes the selection directory"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_self_consistent_omitted_completion_input(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    source_payload = chain["source_payload"]
    assert isinstance(source_payload, dict)
    execution = source_payload["execution_provenance"]
    for records in (
        execution["inputs_before"],
        execution["inputs_after"],
        execution["input_unchanged"],
        source_payload["inputs"],
    ):
        records.pop("weather_coefficients", None)
    _rebind_source_manifest(chain)
    with pytest.raises(ValueError, match=r"missing=\['weather_coefficients'\]"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_self_consistent_forged_completion_builder(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    source_payload = chain["source_payload"]
    builder_source = chain["builder_source"]
    builder_copy = chain["builder_copy"]
    assert isinstance(source_payload, dict)
    assert isinstance(builder_source, Path)
    assert isinstance(builder_copy, Path)
    builder_source.write_text("# forged but self-consistent builder\n", encoding="utf-8")
    builder_copy.write_bytes(builder_source.read_bytes())

    def record(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    source_record = record(builder_source)
    copy_record = record(builder_copy)
    execution = source_payload["execution_provenance"]
    execution["source_before"] = source_record
    execution["source_after"] = dict(source_record)
    execution["executed_source_copy_before"] = copy_record
    execution["executed_source_copy_after"] = dict(copy_record)
    source_payload["outputs"][builder_copy.name]["sha256"] = _sha(builder_copy)
    _rebind_source_manifest(chain)
    with pytest.raises(ValueError, match="frozen builder source"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_selection_execution_provenance_tamper(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["complete"] = False
    _write_json(provenance_path, payload)
    artifact = next(
        row
        for row in selection_payload["artifacts"]
        if row["path"] == str(provenance_path.relative_to(selection_manifest.parent))
    )
    artifact["sha256"] = _sha(provenance_path)
    artifact["bytes"] = provenance_path.stat().st_size
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="provenance is incomplete"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_core_bridge_mismatch(tmp_path, monkeypatch) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    selection_manifest = chain["selection_manifest"]
    selection_payload = chain["selection_payload"]
    provenance_path = chain["selection_provenance"]
    assert isinstance(selection_manifest, Path)
    assert isinstance(selection_payload, dict)
    assert isinstance(provenance_path, Path)
    other = selection_manifest.parent.parent / "other_core_bridge.parquet"
    pd.DataFrame({"dg_id": [1], "downstream_preserved_total": [9.0]}).to_parquet(
        other, index=False
    )
    selection_payload["inputs"]["core_bridge"] = {
        "path": str(other.resolve()),
        "sha256": _sha(other),
    }
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    original = Path(str(chain["core_bridge"])).resolve()
    provenance["inputs"] = [
        {
            **record,
            "path": str(other.resolve()),
            "sha256": _sha(other),
            "bytes": other.stat().st_size,
            "mtime_ns": other.stat().st_mtime_ns,
        }
        if Path(record["path"]).resolve() == original
        else record
        for record in provenance["inputs"]
    ]
    _write_json(provenance_path, provenance)
    artifact = next(
        row
        for row in selection_payload["artifacts"]
        if row["path"] == str(provenance_path.relative_to(selection_manifest.parent))
    )
    artifact["sha256"] = _sha(provenance_path)
    artifact["bytes"] = provenance_path.stat().st_size
    _write_json(selection_manifest, selection_payload)
    _rebind_selection(chain)
    with pytest.raises(ValueError, match="core_bridge"):
        handoff.validate_frozen_handoff_chain(chain["compact_manifest"])


def test_handoff_rejects_compact_control_core_source_mismatch(
    tmp_path, monkeypatch
) -> None:
    chain = _build_chain(tmp_path, monkeypatch)
    compact_manifest = chain["compact_manifest"]
    compact_payload = chain["compact_payload"]
    assert isinstance(compact_manifest, Path)
    assert isinstance(compact_payload, dict)
    prediction = compact_manifest.parent / str(compact_payload["prediction_file"])
    frame = pd.read_parquet(prediction)
    frame["control_core"] = 0.11
    frame.to_parquet(prediction, index=False)
    compact_payload["prediction_sha256"] = _sha(prediction)
    _write_json(compact_manifest, compact_payload)
    with pytest.raises(ValueError, match="control core differs"):
        handoff.validate_frozen_handoff_chain(compact_manifest)

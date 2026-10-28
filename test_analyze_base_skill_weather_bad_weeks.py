from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import analyze_base_skill_weather_bad_weeks as audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def test_artifact_manifest_is_exact_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    root.mkdir()
    artifact = root / "audit.json"
    artifact.write_text('{"research_only": true}\n', encoding="utf-8")
    manifest = {
        "contract": audit.SEALED_CONTRACT,
        "complete": True,
        "research_only": True,
        "artifacts": [_artifact(artifact, root)],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded, paths = audit.validate_artifact_manifest(
        root, label="synthetic sealed", contract=audit.SEALED_CONTRACT
    )
    assert loaded["complete"] is True
    assert paths["audit.json"] == artifact

    artifact.write_text('{"research_only": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|byte count changed"):
        audit.validate_artifact_manifest(
            root, label="synthetic sealed", contract=audit.SEALED_CONTRACT
        )


def test_panel_authority_pins_manifest_and_relevant_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = {}
    pins = {}
    for name, payload in (
        ("player_round_panel", b"rounds"),
        ("player_event_outcomes", b"outcomes"),
        ("odds_event_crosswalk", b"crosswalk"),
    ):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(payload)
        pins[name] = {"sha256": _sha(path), "bytes": path.stat().st_size}
        outputs[name] = {
            "path": str(path),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "production_changes": False,
                "purpose": "research-only frozen synthetic panel",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "PANEL_MANIFEST_SHA256", _sha(manifest_path))
    monkeypatch.setattr(
        audit, "PANEL_MANIFEST_BYTES", manifest_path.stat().st_size
    )
    monkeypatch.setattr(audit, "PANEL_OUTPUT_PINS", pins)
    panel = Path(outputs["player_round_panel"]["path"])
    monkeypatch.setattr(audit, "PANEL_MANIFEST_AUTHORITY_PATH", manifest_path)
    monkeypatch.setattr(audit, "PANEL_ROUND_AUTHORITY_PATH", panel)
    _, result = audit.validate_panel_manifest(panel, manifest_path)
    assert result["panel_sha256"] == pins["player_round_panel"]["sha256"]

    Path(outputs["player_event_outcomes"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen panel output changed"):
        audit.validate_panel_manifest(panel, manifest_path)


def test_reviewed_downstream_source_pins_and_weather_closure_are_exact() -> None:
    expected_shared = {
        audit.ARCHIVE / "base_skill_frozen_handoff.py": (
            "b072b2fdf18769db99e65f633ea24c28c5491fa1c659b0da300b59d7c9fc3a4f"
        ),
        audit.ARCHIVE / "evaluate_complete_base_skill_market_candidates.py": (
            "1818f08cd03a8bde71058223686798c5eaf4aa68f714664671e7f42eb7442c49"
        ),
        audit.ARCHIVE / "evaluate_complete_base_skill_market.py": (
            "92d298661c5f189703c960fcab54e2c7cda321e8d8bbc8351f3d615467374f47"
        ),
        audit.ARCHIVE / "backtest_base_skill_market_models.py": (
            "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d"
        ),
        audit.ARCHIVE / "historical_odds_cache.py": (
            "58a7b774edc9a7e23aab772023846402c02f37396c1b8c7a0d3a4863feeb56c7"
        ),
        audit.ARCHIVE / "market_backtest_scoring.py": (
            "53681535e0d1da7bde9cccede5ac09d7943e7493160c1ac2a248a0c0bfae2608"
        ),
        audit.ARCHIVE / "research_execution_provenance.py": (
            "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034"
        ),
    }
    expected_sealed = {
        audit.ARCHIVE / "evaluate_frozen_base_skill_sealed_matchups.py": (
            "b9d5d74621568497b85ab73d641617ab17877f9107d2d7c3eaaf1097198c0500"
        ),
        **expected_shared,
    }
    expected_cascade = {
        audit.ARCHIVE / "evaluate_base_skill_round_cascade.py": (
            "8544ad2c0076c95b737e4dc214c1bd5d2e05a5c9068f6242a9403f8794938202"
        ),
        **expected_shared,
    }
    assert audit.SEALED_EVALUATOR_SOURCE_PINS == expected_sealed
    assert audit.CASCADE_EVALUATOR_SOURCE_PINS == expected_cascade
    assert set(audit.WEATHER_PROVENANCE_CODE_PATHS) == {
        *(path.resolve() for path in expected_sealed),
        *(path.resolve() for path in expected_cascade),
        audit.CASCADE_TERMINALIZER_PRIMARY_SOURCE.resolve(),
    }
    for path, expected_hash in {**expected_sealed, **expected_cascade}.items():
        assert path.is_file()
        assert _sha(path) == expected_hash
    assert audit.CASCADE_TERMINALIZER_PRIMARY_SOURCE.is_file()
    assert (
        _sha(audit.CASCADE_TERMINALIZER_PRIMARY_SOURCE)
        == audit.CASCADE_TERMINALIZER_SOURCE_SHA256
    )


def test_terminal_numeric_mapping_is_external_exact_and_fails_on_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "interrupted"
    source.mkdir()
    numeric = source / "audit.json"
    numeric.write_text('{"research_only": true}\n', encoding="utf-8")
    record = {
        "path": numeric.name,
        "absolute_path": str(numeric),
        "bytes": numeric.stat().st_size,
        "sha256": _sha(numeric),
    }
    manifest = {"numeric_artifacts": [record]}
    terminal_audit = {"numeric_artifacts": [record]}
    monkeypatch.setattr(
        audit, "CASCADE_TERMINAL_NUMERIC_ARTIFACTS", frozenset({"audit.json"})
    )
    mapping, paths = audit._terminal_numeric_artifact_mapping(
        manifest, terminal_audit, source
    )
    assert mapping == {"audit.json": numeric.resolve()}
    assert paths == (numeric.resolve(),)

    numeric.write_text('{"research_only": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|byte count mismatch"):
        audit._terminal_numeric_artifact_mapping(manifest, terminal_audit, source)


REAL_TERMINAL_CASCADE = (
    audit.RESEARCH_ROOT / "model_eval_round_cascade_v2_terminalized_20260823"
)


@pytest.mark.skipif(
    not (REAL_TERMINAL_CASCADE / "manifest.json").is_file(),
    reason="immutable terminal cascade fixture is unavailable",
)
def test_real_terminal_wrapper_resolves_without_completing_original() -> None:
    manifest, _, artifacts, context = audit.validate_cascade_provenance(
        REAL_TERMINAL_CASCADE, audit.PANEL_MANIFEST_AUTHORITY_PATH
    )
    interrupted = Path(manifest["interrupted_source_dir"]).resolve()
    assert set(artifacts) == set(audit.CASCADE_TERMINAL_NUMERIC_ARTIFACTS)
    assert all(path.parent == interrupted for path in artifacts.values())
    assert context["provenance_mode"] == "terminal_wrapper_external_numeric_v1"
    assert context["terminal_wrapper_complete"] is True
    assert context["original_evaluator_complete"] is False
    assert context["numeric_artifacts_external_read_only"] is True
    assert not (
        interrupted / "execution_provenance" / "execution_provenance.json"
    ).exists()


def _synthetic_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    panel_manifest = tmp_path / "panel_manifest.json"
    panel_manifest.write_text("{}\n", encoding="utf-8")
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(b"input")
    round_panel = tmp_path / "player_round_panel.parquet"
    round_panel.write_bytes(b"round-panel")
    weather_coefficients = tmp_path / "weather_coefficients.csv"
    weather_coefficients.write_text("coefficient\n0.1\n", encoding="utf-8")
    source = tmp_path / "cascade_source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(audit, "CASCADE_EVALUATOR_PRIMARY_SOURCE", source)
    monkeypatch.setattr(
        audit, "CASCADE_EVALUATOR_SOURCE_PINS", {source: _sha(source)}
    )
    compact_manifest = tmp_path / "frozen_manifest.json"
    compact_manifest.write_text("{}\n", encoding="utf-8")
    compact_prediction = tmp_path / "frozen_predictions.parquet"
    compact_prediction.write_bytes(b"prediction")
    core_bridge = tmp_path / "core_bridge.parquet"
    core_bridge.write_bytes(b"core")
    selected = "candidate"
    role = "diagnostic_challenger"
    chain_audit = {
        "compact_manifest_sha256": _sha(compact_manifest),
        "compact_prediction_sha256": _sha(compact_prediction),
        "completion_input_bindings": {
            "round_panel": {
                "path": str(round_panel),
                "sha256": _sha(round_panel),
            },
            "weather_coefficients": {
                "path": str(weather_coefficients),
                "sha256": _sha(weather_coefficients),
            },
            "core_bridge": {
                "path": str(core_bridge),
                "sha256": _sha(core_bridge),
            },
        },
    }
    handoff = SimpleNamespace(
        compact_manifest_path=compact_manifest,
        compact_prediction_path=compact_prediction,
        core_bridge_path=core_bridge,
        core_bridge_sha256=_sha(core_bridge),
        selected_name=selected,
        selection_role=role,
        promotion_gate_passed=False,
        verified_paths=(compact_manifest, compact_prediction, core_bridge),
        audit=chain_audit,
    )
    monkeypatch.setattr(
        audit.frozen_handoff,
        "validate_frozen_handoff_chain",
        lambda path: handoff,
    )
    root = tmp_path / "cascade"
    cascade_audit = {
        "research_only": True,
        "production_changed": False,
        "closing_price_used_as_skill_or_cascade_feature": False,
        "opening_price_used_as_skill_or_cascade_feature": False,
        "market_prices_used_for_cascade_probability_calibration": False,
        "sealed_results_used_for_selection": False,
        "target_or_upcoming_realized_weather_feature_used": False,
        "realized_weather_used_for_posthoc_attribution": True,
        "ungraded_bets": 0,
        "zero_gate_ungraded_bets": 0,
        "all_offered_ungraded_bets": 0,
        "nonzero_pnl_on_void_pushes": 0,
        "duplicate_player_prediction_keys": 0,
        "c3_training_boundary_violations": 0,
        "live_lambda_selection_after_discovery_rows": 0,
        "compatibility_quote_menu_drops": 0,
        "probability_sigma_upper_boundary_hits": 0,
        "probability_sum_max_abs_error": 0.0,
        "reverse_pair_symmetry_max_abs_error": 0.0,
        "pricing_probability_algebra": {
            "rows": 1,
            "ties_lose_rows": 0,
            "push_on_tie_rows": 1,
            "zero_push_rows": 0,
            "simplex_max_abs_error": 0.0,
            "pricing_identity_max_abs_error": 0.0,
            "zero_push_identity_max_abs_error": 0.0,
            "ties_lose_identity": (
                "pricing_p1_probability_equals_unconditional_p1_win"
            ),
            "push_on_tie_identity": (
                "pricing_p1_probability_equals_p1_win_divided_by_one_minus_push"
            ),
        },
        "selection_role": role,
        "promotion_gate_passed": False,
        "selected_candidates": [selected, audit.frozen_handoff.STRICT_PT_CONTROL],
        "frozen_finalist": {
            "manifest_path": str(compact_manifest),
            "manifest_sha256": _sha(compact_manifest),
            "prediction_path": str(compact_prediction),
            "prediction_sha256": _sha(compact_prediction),
            "finalist_name": selected,
            "selection_role": role,
            "promotion_gate_passed": False,
            "control_name": audit.frozen_handoff.STRICT_PT_CONTROL,
            "full_provenance_chain": chain_audit,
        },
    }
    provenance = audit.execution_provenance.begin(
        output_dir=root,
        primary_source=source,
        input_paths=[
            panel_manifest,
            input_file,
            compact_manifest,
            compact_prediction,
            core_bridge,
            round_panel,
            weather_coefficients,
        ],
        cli=["synthetic-cascade", "--seed", "7"],
    )
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(cascade_audit), encoding="utf-8")
    selection = pd.DataFrame(
        [
            {
                "candidate": selected,
                "candidate_type": "diagnostic_challenger",
                "selected": True,
                "finalist_role": role,
                "promotion_gate_passed": False,
            },
            {
                "candidate": audit.frozen_handoff.STRICT_PT_CONTROL,
                "candidate_type": "point_in_time_control",
                "selected": True,
                "finalist_role": "point_in_time_control",
                "promotion_gate_passed": False,
            },
        ]
    )
    selection.to_csv(root / "candidate_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "c3",
                "event_key": "event",
                "dg_id_key": "1",
                "round_num": 1,
                "variant": "C3_pt_continuous_partial_pool_w0",
                "event_start_date": "2023-05-01",
                "training_max_event_end": "2022-12-31",
                "prediction_year": 2023,
            }
        ]
    ).to_parquet(root / "cascade_player_round_predictions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "live_update_lambda": 0.5,
                "selection_max_event_date": "2022-12-31",
            }
        ]
    ).to_csv(root / "c3_discovery_hyperparameter_scores.csv", index=False)
    pd.DataFrame(
        [{"variant": "C1", "c0_quotes_dropped": 0}]
    ).to_csv(root / "model_quote_coverage.csv", index=False)
    pd.DataFrame(
        [{"sigma": 2.8, "discovery_max_event_date": "2022-12-31"}]
    ).to_csv(root / "discovery_probability_sigmas.csv", index=False)
    execution = audit.execution_provenance.finalize(provenance)
    manifest = {
        "complete": True,
        "research_only": True,
        "execution_provenance": execution,
        "inputs": {
            "panel_manifest": {
                "path": str(panel_manifest),
                "sha256": _sha(panel_manifest),
            },
            "other": {"path": str(input_file), "sha256": _sha(input_file)},
            "frozen_finalist_manifest": {
                "path": str(compact_manifest),
                "sha256": _sha(compact_manifest),
            },
            "frozen_finalist_predictions": {
                "path": str(compact_prediction),
                "sha256": _sha(compact_prediction),
            },
            "core_bridge": {
                "path": str(core_bridge),
                "sha256": _sha(core_bridge),
            },
            "player_round_panel": {
                "path": str(round_panel),
                "sha256": _sha(round_panel),
            },
            "weather_coefficients": {
                "path": str(weather_coefficients),
                "sha256": _sha(weather_coefficients),
            },
        },
        "completion_authority": chain_audit["completion_input_bindings"],
        "audit_sha256": _sha(audit_path),
    }
    manifest_path = root / "manifest.json"
    manifest["artifacts"] = [
        _artifact(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path, cascade_audit, manifest


def test_cascade_provenance_rejects_upcoming_realized_weather_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, cascade_audit, manifest = _synthetic_cascade(
        tmp_path, monkeypatch
    )
    panel_manifest = tmp_path / "panel_manifest.json"
    audit.validate_cascade_provenance(root, panel_manifest)

    cascade_audit["target_or_upcoming_realized_weather_feature_used"] = True
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(cascade_audit), encoding="utf-8")
    manifest["audit_sha256"] = _sha(audit_path)
    manifest["artifacts"] = [
        _artifact(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must be exactly False"):
        audit.validate_cascade_provenance(root, panel_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("future_c3_training", "c3_training_boundary_violations label"),
        ("missing_event_date", "event_start_date is missing"),
        ("wrong_prediction_year", "does not equal event_start_date year"),
        ("missing_prediction_year", "missing or nonintegral"),
        ("future_lambda_selection", "live_lambda_selection_after_discovery_rows label"),
        ("compatibility_drop", "compatibility_quote_menu_drops label"),
        ("sigma_boundary", "probability_sigma_upper_boundary_hits label"),
        ("future_sigma_fit", "discovery_max_event_date exceeds"),
    ],
)
def test_cascade_provenance_reproduces_critical_audit_before_probability_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    root, manifest_path, _, manifest = _synthetic_cascade(tmp_path, monkeypatch)
    if mutation == "future_c3_training":
        path = root / "cascade_player_round_predictions.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "training_max_event_end"] = "2023-01-01"
        frame.to_parquet(path, index=False)
    elif mutation == "missing_event_date":
        path = root / "cascade_player_round_predictions.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "event_start_date"] = None
        frame.to_parquet(path, index=False)
    elif mutation == "wrong_prediction_year":
        path = root / "cascade_player_round_predictions.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prediction_year"] = 2024
        frame.to_parquet(path, index=False)
    elif mutation == "missing_prediction_year":
        path = root / "cascade_player_round_predictions.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prediction_year"] = np.nan
        frame.to_parquet(path, index=False)
    elif mutation == "future_lambda_selection":
        path = root / "c3_discovery_hyperparameter_scores.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "selection_max_event_date"] = "2023-01-01"
        frame.to_csv(path, index=False)
    elif mutation == "compatibility_drop":
        path = root / "model_quote_coverage.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "c0_quotes_dropped"] = 1
        frame.to_csv(path, index=False)
    elif mutation == "sigma_boundary":
        path = root / "discovery_probability_sigmas.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "sigma"] = 12.0
        frame.to_csv(path, index=False)
    elif mutation == "future_sigma_fit":
        path = root / "discovery_probability_sigmas.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "discovery_max_event_date"] = "2023-01-01"
        frame.to_csv(path, index=False)
    manifest["artifacts"] = [
        _artifact(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        audit.validate_cascade_provenance(
            root, tmp_path / "panel_manifest.json"
        )


def test_cascade_c3_training_boundary_ignores_non_c3_control_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, _, manifest = _synthetic_cascade(tmp_path, monkeypatch)
    path = root / "cascade_player_round_predictions.parquet"
    frame = pd.read_parquet(path)
    control = frame.iloc[0].copy()
    control["model"] = audit.CASCADE_CONTROL
    control["dg_id_key"] = "2"
    control["variant"] = "C0"
    control["training_max_event_end"] = "2026-01-01"
    frame = pd.concat([frame, control.to_frame().T], ignore_index=True)
    frame.to_parquet(path, index=False)
    manifest["artifacts"] = [
        _artifact(artifact_path, root)
        for artifact_path in sorted(root.rglob("*"))
        if artifact_path.is_file() and artifact_path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, _, _, context = audit.validate_cascade_provenance(
        root, tmp_path / "panel_manifest.json"
    )
    assert context["critical_audit"]["c3_training_boundary_violations"] == 0


def test_cascade_probability_rows_must_reproduce_pricing_audit() -> None:
    producer_audit = {
        "probability_sum_max_abs_error": 0.0,
        "pricing_probability_algebra": {
            "rows": 1,
            "ties_lose_rows": 0,
            "push_on_tie_rows": 1,
            "zero_push_rows": 0,
            "simplex_max_abs_error": 0.0,
            "pricing_identity_max_abs_error": 0.0,
            "zero_push_identity_max_abs_error": 0.0,
            "ties_lose_identity": (
                "pricing_p1_probability_equals_unconditional_p1_win"
            ),
            "push_on_tie_identity": (
                "pricing_p1_probability_equals_p1_win_divided_by_one_minus_push"
            ),
        },
    }
    probabilities = pd.DataFrame(
        {
            "p1_probability": [0.4],
            "p2_probability": [0.4],
            "push_probability": [0.2],
            "pricing_p1_probability": [0.5],
            "tie_rule_ties_lose": [False],
        }
    )
    reproduced = audit.validate_cascade_probability_audit_consistency(
        probabilities, producer_audit
    )
    assert reproduced["producer_audit_exact"] is True

    probabilities.loc[0, "pricing_p1_probability"] = 0.99
    with pytest.raises(ValueError, match="pricing_identity_max_abs_error differs"):
        audit.validate_cascade_probability_audit_consistency(
            probabilities, producer_audit
        )


def test_cascade_critical_gate_rejects_forged_pricing_pass_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, producer_audit, manifest = _synthetic_cascade(
        tmp_path, monkeypatch
    )
    producer_audit["pricing_probability_algebra"][
        "pricing_identity_max_abs_error"
    ] = 0.01
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(producer_audit), encoding="utf-8")
    manifest["audit_sha256"] = _sha(audit_path)
    manifest["artifacts"] = [
        _artifact(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must be finite and <= 1e-12"):
        audit.validate_cascade_provenance(
            root, tmp_path / "panel_manifest.json"
        )


@pytest.mark.parametrize("record_group", ["sources", "inputs"])
def test_cascade_provenance_rehashes_declared_sources_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_group: str,
) -> None:
    root, _, _, manifest = _synthetic_cascade(tmp_path, monkeypatch)
    record = manifest["execution_provenance"][record_group][0]
    Path(record["path"]).write_text("changed after completion\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|mtime differs"):
        audit.validate_cascade_provenance(
            root, tmp_path / "panel_manifest.json"
        )


def test_execution_source_authority_rejects_primary_or_dependency_substitution(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "evaluator.py"
    dependency = tmp_path / "dependency.py"
    substitute = tmp_path / "substitute.py"
    primary.write_text("PRIMARY = True\n", encoding="utf-8")
    dependency.write_text("DEPENDENCY = True\n", encoding="utf-8")
    substitute.write_bytes(primary.read_bytes())
    expected = {primary: _sha(primary), dependency: _sha(dependency)}
    execution = {
        "primary_source": str(primary),
        "sources": [
            {"path": str(primary), "sha256": _sha(primary)},
            {"path": str(dependency), "sha256": _sha(dependency)},
        ],
    }
    result = audit.validate_execution_source_authority(
        execution,
        label="synthetic",
        expected_sources=expected,
        expected_primary=primary,
    )
    assert result["exact_source_set_verified"] is True

    replaced_primary = dict(execution)
    replaced_primary["primary_source"] = str(substitute)
    replaced_primary["sources"] = [
        {"path": str(substitute), "sha256": _sha(substitute)},
        execution["sources"][1],
    ]
    with pytest.raises(ValueError, match="primary source is not authoritative"):
        audit.validate_execution_source_authority(
            replaced_primary,
            label="synthetic",
            expected_sources=expected,
            expected_primary=primary,
        )

    missing_dependency = dict(execution)
    missing_dependency["sources"] = [execution["sources"][0]]
    with pytest.raises(ValueError, match="source authority differs"):
        audit.validate_execution_source_authority(
            missing_dependency,
            label="synthetic",
            expected_sources=expected,
            expected_primary=primary,
        )


def test_cascade_completion_authority_rejects_same_byte_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, _, manifest = _synthetic_cascade(tmp_path, monkeypatch)
    substitute = tmp_path / "weather_coefficients_copy.csv"
    authoritative = Path(
        manifest["completion_authority"]["weather_coefficients"]["path"]
    )
    substitute.write_bytes(authoritative.read_bytes())
    manifest["completion_authority"]["weather_coefficients"] = {
        "path": str(substitute),
        "sha256": _sha(substitute),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest/handoff completion authority differs"):
        audit.validate_cascade_provenance(
            root, tmp_path / "panel_manifest.json"
        )


def _weather_panel() -> pd.DataFrame:
    rows = []
    for event_index, event_key in enumerate(("pga:2023:a", "pga:2023:b")):
        for round_num in (1, 2):
            winds = np.arange(8, dtype=float) + 5.0 + event_index * 5 + round_num
            dews = np.arange(8, dtype=float) + 50.0
            temps = np.arange(8, dtype=float) + 65.0
            rain = np.arange(8, dtype=float) / 100.0
            for player in range(8):
                rows.append(
                    {
                        "event_key": event_key,
                        "event_name": f"Event {event_index}",
                        "event_start_date": "2023-06-01",
                        "year": 2023,
                        "round_num": round_num,
                        "round_date": f"2023-06-0{round_num}",
                        "course_num": 100 + event_index,
                        "dg_id": player,
                        "fin_text": "WD" if player == 7 else "T20",
                        "weather_source_teetime": (
                            f"07:{player:02d} AM" if player < 4 else f"01:{player:02d} PM"
                        ),
                        "weather_wind": winds[player],
                        "weather_dew": dews[player],
                        "weather_temp": temps[player],
                        "weather_rain": rain[player],
                        "weather_field_mean_wind": winds.mean(),
                        "weather_field_mean_dew": dews.mean(),
                        "weather_field_mean_temp": temps.mean(),
                        "weather_field_mean_rain": rain.mean(),
                        "weather_delta_wind": winds[player] - winds.mean(),
                        "weather_delta_dew": dews[player] - dews.mean(),
                        "weather_delta_temp": temps[player] - temps.mean(),
                        "weather_delta_rain": rain[player] - rain.mean(),
                        "dg_sg_total_field_adjusted": player / 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_weather_profiles_label_target_weather_oracle_and_completed_rounds(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "rounds.parquet"
    _weather_panel().to_parquet(panel_path, index=False)
    course_profiles, profiles, events, provenance = audit.build_weather_profiles(
        panel_path, required_event_keys={"pga:2023:a", "pga:2023:b"}
    )
    assert len(course_profiles) == 4
    assert len(profiles) == 4
    assert len(events) == 2
    assert profiles["tee_wave_available"].all()
    assert profiles["late_minus_early_wind"].gt(0).all()
    assert profiles["target_round_realized_weather_used_as_feature"].eq(False).all()
    assert provenance["target_weather_is_oracle_posthoc_attribution_only"] is True
    assert provenance["weather_value_domain_contract"]["wind_and_rain_rule"] == (
        "raw_and_physical_course_field_mean_finite_nonnegative_when_declared"
    )
    assert provenance["weather_value_domain_contract"][
        "dew_and_temperature_range_gate"
    ].startswith("none_")
    event_rows = pd.DataFrame(
        {
            "model": ["w2_model", "w2_model"],
            "analysis_pair": [
                "w2_model__vs__w0_model",
                "w2_model__vs__w0_model",
            ],
            "event_key": ["pga:2023:a", "pga:2023:a"],
            "event_start_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
            "round_num": [1, 2],
        }
    )
    lineage = pd.DataFrame(
        {
            "model": ["w2_model", "w2_model"],
            "event_key": ["pga:2023:a", "pga:2023:a"],
            "event_start_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
            "round_num": [1, 2],
            "lineage_completed_weather_observations": [0, 8],
            "lineage_players_with_completed_weather": [0, 8],
            "lineage_completed_weather_any": [False, True],
        }
    )
    attached = audit.attach_weather_to_cascade(event_rows, profiles, lineage)
    assert attached["completed_rounds_in_profile"].tolist() == [0, 1]
    assert attached["completed_weather_used_by_w2"].tolist() == [False, True]
    assert not attached["target_oracle_weather_used_by_w2"].any()


def test_weather_tee_waves_are_stratified_by_physical_course(tmp_path: Path) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    second = panel.copy()
    second["course_num"] = 999
    second["dg_id"] = second["dg_id"] + 100
    # A large course baseline difference must not be interpreted as a tee-wave
    # difference; the within-course centered deltas stay unchanged.
    second["weather_wind"] = second["weather_wind"] + 30.0
    second["weather_field_mean_wind"] = second["weather_field_mean_wind"] + 30.0
    combined = pd.concat([panel, second], ignore_index=True)
    path = tmp_path / "multi_course.parquet"
    combined.to_parquet(path, index=False)
    course, event_round, _, provenance = audit.build_weather_profiles(
        path, required_event_keys={"pga:2023:a"}
    )
    assert len(course) == 2
    assert event_round["physical_course_count"].tolist() == [2]
    assert event_round["multi_course_event_round"].tolist() == [True]
    assert event_round["physical_course_join_preserved"].all()
    assert event_round["late_minus_early_wind"].iloc[0] == pytest.approx(
        course["late_minus_early_wind"].mean()
    )
    assert provenance["physical_course_join_preserved_before_event_aggregation"] is True


def test_missing_weather_is_coverage_not_zero(tmp_path: Path) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    weather_columns = [
        column
        for column in panel.columns
        if column.startswith("weather_") and column != "weather_source_teetime"
    ]
    panel[weather_columns] = np.nan
    path = tmp_path / "missing_weather.parquet"
    panel.to_parquet(path, index=False)
    _, event_round, events, provenance = audit.build_weather_profiles(
        path, required_event_keys={"pga:2023:a"}
    )
    assert event_round["weather_coverage_rate"].tolist() == [0.0]
    assert event_round["weather_severity_percentile"].isna().all()
    base = pd.DataFrame(
        {
            "event_key": ["pga:2023:a"],
            "event_start_date": pd.to_datetime(["2023-06-01"]),
        }
    )
    attached = audit.attach_weather_to_base(base, events)
    assert attached["target_event_weather_missing"].tolist() == [True]
    assert pd.isna(attached["event_weather_severity_max"].iloc[0])
    assert provenance["missing_physical_course_rows"] == 0


def _candidate_metadata() -> pd.DataFrame:
    common = {
        "family": "ridge",
        "source": "total_cat",
        "refit_schedule": "annual",
        "alpha": 10.0,
        "short_game_penalty": 2.0,
        "candidate_role": "diagnostic",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "candidate": "w0",
                "weather_arm": "w0",
                "target_arm": "observed",
            },
            {
                **common,
                "candidate": "w1_feature",
                "weather_arm": "w1",
                "target_arm": "feature_only",
            },
            {
                **common,
                "candidate": "w1_neutral",
                "weather_arm": "w1",
                "target_arm": "neutral_target",
            },
        ]
    )


def test_w1_pairing_is_structural_and_does_not_guess_ambiguous_reference() -> None:
    paired = audit.build_w1_pairing_audit(_candidate_metadata())
    assert paired["pairing_status"].eq("paired").all()
    assert paired["w0_candidate"].eq("w0").all()

    ambiguous = pd.concat(
        [
            _candidate_metadata(),
            _candidate_metadata().iloc[[0]].assign(candidate="w0_duplicate"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="every W1 candidate requires exactly one"):
        audit.build_w1_pairing_audit(ambiguous)


def test_w1_pairing_reports_unavailable_when_frozen_menu_has_no_w1() -> None:
    metadata = _candidate_metadata()
    metadata = metadata.loc[
        ~metadata["weather_arm"].astype(str).str.startswith("w1")
    ].copy()
    pairing = audit.build_w1_pairing_audit(metadata)
    assert pairing.empty
    checked = audit.audit_w1_quote_menus(pairing, pd.DataFrame())
    assert checked.empty
    assert "quote_menu_status" in checked


def _base_logical() -> pd.DataFrame:
    rows = []
    for candidate, probability in (
        ("w1", 0.75),
        ("w0", 0.55),
        ("control", 0.50),
    ):
        for event_index, outcome in enumerate((1.0, 0.0)):
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_role": "synthetic_role",
                    "event_key": f"event-{event_index}",
                    "event_start_date": f"2023-01-0{event_index + 1}",
                    "market": "72h",
                    "pair_key": "1|2",
                    "canonical_outcome": outcome,
                    "canonical_probability": probability,
                    "canonical_push_probability": 0.0,
                    "probability_fit_end": "2022-12-31",
                }
            )
    return pd.DataFrame(rows)


def test_base_proper_delta_uses_exact_common_menu_and_correct_sign() -> None:
    logical = _base_logical()
    result = audit.pairwise_base_proper_deltas(
        logical,
        challenger="w1",
        reference="w0",
        analysis_pair="pair",
        comparison_kind="w1_vs_w0",
    )
    assert len(result) == 2
    assert result.loc[result["event_key"].eq("event-0"), "event_delta"].iloc[0] < 0
    mismatched = logical.loc[
        ~(
            logical["candidate"].eq("w0")
            & logical["event_key"].eq("event-1")
        )
    ]
    with pytest.raises(ValueError, match="menu mismatch"):
        audit.pairwise_base_proper_deltas(
            mismatched,
            challenger="w1",
            reference="w0",
            analysis_pair="pair",
            comparison_kind="w1_vs_w0",
        )


def _cascade_probabilities() -> pd.DataFrame:
    rows = []
    models = (
        ("w2_model", "candidate", audit.W2_W0_VARIANTS["w2"], 0.80),
        ("w0_model", "candidate", audit.W2_W0_VARIANTS["w0"], 0.60),
        (audit.CASCADE_CONTROL, "control", "C0", 0.50),
    )
    for model, candidate, variant, probability in models:
        for round_num in (1, 2):
            rows.append(
                {
                    "model": model,
                    "candidate": candidate,
                    "variant": variant,
                    "event_key": "event",
                    "event_start_date": "2023-05-01",
                    "round_num": round_num,
                    "pair_key": "1|2",
                    "quote_id": f"q-{round_num}",
                    "p1_id": "1",
                    "p2_id": "2",
                    "p1_outcome": 1.0,
                    "p2_outcome": 0.0,
                    "p1_probability": probability,
                    "p2_probability": 1.0 - probability,
                    "push_probability": 0.0,
                    "pricing_p1_probability": probability,
                    "tie_rule": "push on tie",
                    "tie_rule_ties_lose": False,
                    "derived_score_outcome": 1.0,
                    "score_model_eligible": True,
                    "market_devig_proper_score_eligible": True,
                    "market_devig_proper_score_exclusion_reason": (
                        "eligible_resolved_push_on_tie_two_way_benchmark"
                    ),
                    "binary_model_eligible": True,
                    "market_devig_probability_scope": (
                        "resolved_two_way_push_on_tie_market_benchmark"
                    ),
                    "book": "book_a",
                    "p1_open": -110.0,
                    "p2_open": -110.0,
                    "p1_close": -110.0,
                    "p2_close": -110.0,
                    "p1_open_devig_probability": 0.5,
                    "p1_close_devig_probability": 0.5,
                    "model_minus_devig_open_pp": 100.0 * (probability - 0.5),
                    "model_minus_devig_close_pp": 100.0 * (probability - 0.5),
                }
            )
    return pd.DataFrame(rows)


def _score_classified_cascade_probabilities() -> pd.DataFrame:
    frame = _cascade_probabilities()
    frame["p1_round_score"] = 70.0
    frame["p2_round_score"] = 71.0
    frame["both_target_round_scores_valid"] = True
    frame["p1_derived_score_outcome"] = 1.0
    frame["p2_derived_score_outcome"] = 0.0
    frame["settlement_score_outcome"] = 1.0
    frame["p1_settlement_score_outcome"] = 1.0
    frame["p2_settlement_score_outcome"] = 0.0
    frame["archived_outcome_pair_possible"] = True
    frame["p1_archive_outcome_matches_score"] = True
    frame["p2_archive_outcome_matches_score"] = True
    frame["archive_outcome_matches_score"] = True
    frame["actual_score_tie"] = False
    frame["score_tie_void"] = False
    frame["score_tie_loss"] = False
    frame["no_score_void"] = False
    frame["other_void_with_scores"] = False

    void = frame["round_num"].eq(2)
    frame.loc[void, ["p1_round_score", "p2_round_score"]] = np.nan
    frame.loc[void, ["p1_outcome", "p2_outcome"]] = 0.5
    frame.loc[
        void,
        [
            "derived_score_outcome",
            "p1_derived_score_outcome",
            "p2_derived_score_outcome",
            "settlement_score_outcome",
            "p1_settlement_score_outcome",
            "p2_settlement_score_outcome",
        ],
    ] = np.nan
    frame.loc[
        void,
        [
            "both_target_round_scores_valid",
            "p1_archive_outcome_matches_score",
            "p2_archive_outcome_matches_score",
            "archive_outcome_matches_score",
            "score_model_eligible",
            "market_devig_proper_score_eligible",
            "binary_model_eligible",
        ],
    ] = False
    frame.loc[void, "no_score_void"] = True
    frame.loc[void, "market_devig_proper_score_exclusion_reason"] = (
        "ungraded_or_score_settlement_mismatch"
    )
    frame.loc[
        void, ["model_minus_devig_open_pp", "model_minus_devig_close_pp"]
    ] = np.nan
    return frame


def _score_void_producer_audit() -> dict[str, object]:
    return {
        "score_void_classification": {
            "quotes": 2,
            "archived_voids": 1,
            "score_tie_voids": 0,
            "score_tie_losses": 0,
            "no_score_voids": 1,
            "other_voids_with_two_non_tied_scores": 0,
            "quotes_with_two_valid_round_scores": 1,
            "score_model_eligible_quotes": 1,
            "market_devig_proper_score_eligible_quotes": 1,
            "ties_lose_quotes_excluded_from_market_proper_score": 0,
            "archived_vs_score_outcome_mismatches_when_both_scores_valid": 0,
            "p1_archived_vs_score_mismatches": 0,
            "p2_archived_vs_score_mismatches": 0,
            "impossible_archived_outcome_pairs": 0,
            "rows_by_round": {
                "r1": {
                    "quotes": 1,
                    "score_tie_voids": 0,
                    "score_tie_losses": 0,
                    "no_score_voids": 0,
                    "score_model_eligible": 1,
                    "market_devig_proper_score_eligible": 1,
                    "ties_lose_market_benchmark_exclusions": 0,
                },
                "r2": {
                    "quotes": 1,
                    "score_tie_voids": 0,
                    "score_tie_losses": 0,
                    "no_score_voids": 1,
                    "score_model_eligible": 0,
                    "market_devig_proper_score_eligible": 0,
                    "ties_lose_market_benchmark_exclusions": 0,
                },
            },
        }
    }


def _score_void_player_prediction_authority(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "event_key": "event",
                "round_num": round_num,
                "dg_id_key": player,
                "prediction": 0.0,
            }
            for model in sorted(raw["model"].unique())
            for round_num in (1, 2)
            for player in ("1", "2")
        ]
    )


def _score_void_round_panel_authority() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_key": "event",
                "round_num": round_num,
                "dg_id": player,
                "score": score,
            }
            for round_num, scores in ((1, (70.0, 71.0)), (2, (np.nan, np.nan)))
            for player, score in zip(("1", "2"), scores)
        ]
    )


def test_cascade_w2_pairing_and_proper_delta_are_exact() -> None:
    probabilities = audit._canonical_cascade_probabilities(
        _cascade_probabilities()
    )
    predictions = pd.DataFrame(
        [
            {
                "model": "w2_model",
                "candidate": "candidate",
                "variant": audit.W2_W0_VARIANTS["w2"],
                "weather_arm": "w2",
            },
            {
                "model": "w0_model",
                "candidate": "candidate",
                "variant": audit.W2_W0_VARIANTS["w0"],
                "weather_arm": "w0",
            },
            {
                "model": audit.CASCADE_CONTROL,
                "candidate": "control",
                "variant": "C0",
                "weather_arm": "legacy",
            },
        ]
    )
    pairing = audit.build_w2_pairing_audit(predictions, probabilities)
    assert pairing["quote_menu_status"].tolist() == ["exact"]
    proper = audit.pairwise_cascade_proper_deltas(
        probabilities,
        challenger="w2_model",
        reference="w0_model",
        analysis_pair="pair",
        comparison_kind="w2_vs_w0",
    )
    three_way = proper.loc[proper["metric"].eq("three_way_logloss_delta")]
    assert len(three_way) == 2
    assert three_way["event_delta"].lt(0).all()
    assert set(three_way["weather_information_scope"]) == {
        "historical_training_only_no_current_completed_round",
        "historical_training_plus_completed_rounds",
    }


def test_cascade_valid_no_score_void_is_reported_and_excluded_from_proper() -> None:
    raw = _score_classified_cascade_probabilities()
    canonical = audit._canonical_cascade_probabilities(raw)
    void_rows = canonical["round_num"].eq(2)
    assert canonical.loc[void_rows, "derived_score_outcome"].isna().all()
    assert not canonical.loc[void_rows, "score_model_eligible"].any()
    reproduced, exclusions = audit.validate_cascade_score_void_classification(
        raw,
        _normalized_cascade_quotes(raw),
        _score_void_player_prediction_authority(raw),
        _score_void_round_panel_authority(),
        _score_void_producer_audit(),
    )
    assert reproduced["no_score_voids"] == 1
    assert exclusions["proper_exclusion_reason"].tolist() == [
        "no_target_round_scores_archived_void"
    ]
    proper = audit.pairwise_cascade_proper_deltas(
        canonical,
        challenger="w2_model",
        reference="w0_model",
        analysis_pair="pair",
        comparison_kind="w2_vs_w0",
    )
    assert set(proper["round_num"]) == {1}
    assert set(proper["metric"]) == {
        "three_way_logloss_delta",
        "three_way_brier_delta",
    }


def test_cascade_score_void_uses_prediction_covered_model_menu_and_full_audit() -> None:
    raw = _score_classified_cascade_probabilities()
    quotes = _normalized_cascade_quotes(raw)
    extra = quotes.loc[quotes["round_num"].eq(1)].iloc[[0]].copy()
    extra["quote_id"] = "q-extra"
    extra["pair_key"] = "3|4"
    extra["p1_id"] = "3"
    extra["p2_id"] = "4"
    quotes = pd.concat([quotes, extra], ignore_index=True)
    panel = pd.concat(
        [
            _score_void_round_panel_authority(),
            pd.DataFrame(
                [
                    {"event_key": "event", "round_num": 1, "dg_id": "3", "score": 69.0},
                    {"event_key": "event", "round_num": 1, "dg_id": "4", "score": 72.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    producer = _score_void_producer_audit()
    score_audit = producer["score_void_classification"]
    score_audit.update(
        {
            "quotes": 3,
            "quotes_with_two_valid_round_scores": 2,
            "score_model_eligible_quotes": 2,
            "market_devig_proper_score_eligible_quotes": 2,
        }
    )
    score_audit["rows_by_round"]["r1"].update(
        {
            "quotes": 2,
            "score_model_eligible": 2,
            "market_devig_proper_score_eligible": 2,
        }
    )

    reproduced, _ = audit.validate_cascade_score_void_classification(
        raw,
        quotes,
        _score_void_player_prediction_authority(raw),
        panel,
        producer,
    )

    assert reproduced["full_normalized_quote_panel_authority_rows"] == 3
    assert reproduced["c0_prediction_covered_quote_ids"] == 2
    assert reproduced["c0_normalized_quotes_outside_prediction_coverage"] == 1
    assert reproduced["all_model_prediction_covered_quote_menus_exact"] is True


def test_cascade_no_score_void_rejects_tampered_classification() -> None:
    raw = _score_classified_cascade_probabilities()
    raw.loc[raw["round_num"].eq(2), "no_score_void"] = False
    with pytest.raises(ValueError, match="classification is false at 'no_score_void'"):
        audit.validate_cascade_score_void_classification(
            raw,
            _normalized_cascade_quotes(raw),
            _score_void_player_prediction_authority(raw),
            _score_void_round_panel_authority(),
            _score_void_producer_audit(),
        )


def test_bet_attribution_flags_wd_newcomer_and_low_coverage() -> None:
    bets = pd.DataFrame(
        {
            "event_key": ["e", "e"],
            "event_start_date": ["2023-01-01", "2023-01-01"],
            "pnl": [-1.0, 1.0],
            "stake": [1.0, 1.0],
            "selected_history_rounds": [2, 20],
            "opponent_history_rounds": [10, 20],
            "selected_img_history_rounds": [2, 16],
            "opponent_img_history_rounds": [12, 16],
            "selected_img_history_coverage": [0.1, 0.8],
            "opponent_img_history_coverage": [0.8, 0.8],
            "selected_reliability_ok": [False, True],
            "opponent_reliability_ok": [True, True],
            "selected_cross_tour_fallback_reason": ["tour_prior", "weekly_pga_core"],
            "selected_event_withdrawal": [True, False],
            "opponent_event_withdrawal": [False, False],
            "selected_prior_event_withdrawal": [False, False],
            "opponent_prior_event_withdrawal": [False, False],
        }
    )
    normalized = audit._standardize_bet_attribution(bets, source="cascade")
    attributed = audit._attribute_event_bets(normalized, group_columns=["event_key"])
    assert attributed["target_withdrawal_bets"].iloc[0] == 1
    assert attributed["newcomer_bets"].iloc[0] == 1
    assert attributed["low_coverage_bets"].iloc[0] == 1
    assert attributed["cross_tour_fallback_bets"].iloc[0] == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("selected_event_withdrawal", "False", "exact booleans"),
        ("selected_reliability_ok", 2, "exact booleans"),
        ("selected_img_history_coverage", 999.0, r"\[0,1\]"),
        ("selected_history_rounds", -1, "nonnegative integer counts"),
        ("selected_history_rounds", 1.5, "nonnegative integer counts"),
        ("selected_img_history_rounds", np.inf, "nonnegative integer counts"),
    ],
)
def test_bet_attribution_rejects_malformed_boolean_count_and_coverage_domains(
    column: str, value: object, message: str
) -> None:
    bets = pd.DataFrame(
        {
            "event_key": ["e"],
            "event_start_date": ["2023-01-01"],
            "pnl": [-1.0],
            "stake": [1.0],
            "selected_history_rounds": [2],
            "opponent_history_rounds": [10],
            "selected_img_history_rounds": [2],
            "opponent_img_history_rounds": [10],
            "selected_img_history_coverage": [0.1],
            "opponent_img_history_coverage": [0.8],
            "selected_reliability_ok": [False],
            "opponent_reliability_ok": [True],
            "selected_cross_tour_fallback_reason": ["tour_prior"],
            "selected_event_withdrawal": [True],
            "opponent_event_withdrawal": [False],
            "selected_prior_event_withdrawal": [False],
            "opponent_prior_event_withdrawal": [False],
        }
    )
    bets[column] = bets[column].astype(object)
    bets.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        audit._standardize_bet_attribution(bets, source="cascade")


def test_cascade_legacy_nullable_binary_decoder_is_narrow_and_preserves_missing() -> None:
    decoded, repaired = audit._decode_cascade_legacy_nullable_binary(
        pd.Series([0.0, 1.0, np.nan]),
        column="selected_prior_event_withdrawal",
    )
    assert repaired is True
    assert decoded.tolist() == [False, True, pd.NA]

    for values in (
        ["0", "1", None],
        [0.0, 2.0, np.nan],
        [0.0, 0.5, np.nan],
        [0.0, np.inf, np.nan],
        [False, 1.0, np.nan],
    ):
        with pytest.raises(ValueError, match="pinned legacy numeric 0/1/NA"):
            audit._decode_cascade_legacy_nullable_binary(
                pd.Series(values, dtype=object),
                column="selected_prior_event_withdrawal",
            )


def test_cascade_legacy_nullable_binary_requires_all_four_serialized_columns() -> None:
    bets = pd.DataFrame(
        {
            "event_key": ["e"],
            "event_start_date": ["2023-01-01"],
            "pnl": [-1.0],
            "stake": [1.0],
            "selected_history_rounds": [10],
            "opponent_history_rounds": [10],
            "selected_img_history_rounds": [10],
            "opponent_img_history_rounds": [10],
            "selected_img_history_coverage": [0.8],
            "opponent_img_history_coverage": [0.8],
            "selected_reliability_ok": [1.0],
            "opponent_reliability_ok": [1.0],
            "selected_cross_tour_fallback_reason": ["weekly_pga_core"],
            "selected_event_withdrawal": [False],
            "opponent_event_withdrawal": [False],
            "selected_prior_event_withdrawal": [np.nan],
            "opponent_prior_event_withdrawal": [0.0],
        }
    )
    normalized = audit._standardize_bet_attribution(bets, source="cascade")
    assert normalized["missing_prior_withdrawal_bet"].tolist() == [True]
    assert normalized["prior_withdrawal_any"].tolist() == [False]
    assert normalized[
        "legacy_nullable_binary_serialization_repair_bet"
    ].tolist() == [True]

    partial = bets.copy()
    partial["selected_reliability_ok"] = pd.Series([True], dtype=object)
    with pytest.raises(ValueError, match="only partially present"):
        audit._standardize_bet_attribution(partial, source="cascade")


def test_base_event_attribution_reproduces_model_and_control_pnl() -> None:
    money = pd.DataFrame(
        {
            "analysis_pair": ["w1__vs__w0"],
            "model": ["w1"],
            "reference": ["w0"],
            "control": ["control"],
            "event_key": ["e"],
            "event_start_date": pd.to_datetime(["2023-01-01"]),
            "market": ["72h"],
            "opportunity_mode": ["pairwise_common_with_strict_pt_baseline"],
            "price_basis": ["close"],
            "line_policy": ["best_of_three"],
            "sizing": ["flat_1u"],
            "reliability_policy": ["all"],
            "model_pnl": [-1.0],
            "reference_pnl": [0.5],
            "control_pnl": [1.0],
        }
    )
    raw_bets = pd.DataFrame(
        {
            "candidate": ["w1", "w0", "control"],
            "comparison_candidate": ["w1", "w0", "w1"],
            "event_key": ["e", "e", "e"],
            "event_start_date": pd.to_datetime(["2023-01-01"] * 3),
            "market": ["72h", "72h", "72h"],
            "opportunity_mode": ["pairwise_common_with_strict_pt_baseline"] * 3,
            "price_basis": ["close", "close", "close"],
            "line_policy": ["best_of_three", "best_of_three", "best_of_three"],
            "sizing": ["flat_1u", "flat_1u", "flat_1u"],
            "reliability_policy": ["all", "all", "all"],
            "pnl": [-1.0, 0.5, 1.0],
            "stake": [1.0, 1.0, 1.0],
            "selected_history_rounds": [3, 20, 20],
            "opponent_history_rounds": [20, 20, 20],
            "selected_img_history_coverage": [0.2, 0.8, 0.8],
            "opponent_img_history_coverage": [0.8, 0.8, 0.8],
            "selected_reliability_ok": [False, True, True],
            "opponent_reliability_ok": [True, True, True],
            "selected_cross_tour_fallback_reason": [
                "tour_prior",
                "weekly_pga_core",
                "weekly_pga_core",
            ],
            "target_withdrawal_any": [True, False, False],
            "prior_withdrawal_any": [False, False, False],
        }
    )
    for column in (
        "newcomer_bet",
        "missing_history_bet",
        "low_coverage_bet",
        "missing_reliability_bet",
        "low_img_coverage_bet",
        "missing_img_coverage_bet",
        "cross_tour_fallback_bet",
    ):
        raw_bets[column] = [
            column
            in {
                "newcomer_bet",
                "low_coverage_bet",
                "low_img_coverage_bet",
                "cross_tour_fallback_bet",
            },
            False,
            False,
        ]
    result = audit.attach_base_bet_attribution(money, raw_bets)
    assert result["model_attribution_pnl"].tolist() == [-1.0]
    assert result["reference_attribution_pnl"].tolist() == [0.5]
    assert result["control_attribution_pnl"].tolist() == [1.0]
    assert result["model_newcomer_bets"].tolist() == [1]


def test_open_money_is_nonexecutable_when_model_state_is_late() -> None:
    frame = pd.DataFrame(
        {
            "price_point": ["close", "open", "open"],
            "analysis_block": [
                "cascade_w2_completed_round_weather_normalization"
            ] * 3,
            "round_num": [2, 2, 1],
            "model_state_time": [
                "2024-01-04T12:00:00Z",
                "2024-01-04T12:00:00Z",
                "2024-01-04T08:00:00Z",
            ],
            "quote_time": [
                "2024-01-04T09:00:00Z",
                "2024-01-04T09:00:00Z",
                "2024-01-04T09:00:00Z",
            ],
        }
    )
    annotated = audit.annotate_money_execution_eligibility(
        frame,
        price_column="price_point",
        model_state_timestamp_column="model_state_time",
        quote_timestamp_column="quote_time",
        timestamp_authority_verified=True,
    )
    assert annotated["execution_eligible"].tolist() == [True, False, True]
    assert annotated.loc[1, "execution_eligibility_reason"] == (
        "completed_round_state_timestamp_not_proven_before_open_quote"
    )
    assert annotated.loc[2, "execution_eligibility_reason"] == (
        "hash_bound_model_state_not_later_than_open_quote"
    )


def test_open_money_without_hash_bound_timestamps_is_sensitivity_only() -> None:
    frame = pd.DataFrame(
        {
            "price_basis": ["close", "open"],
            "analysis_block": ["base", "base"],
        }
    )
    annotated = audit.annotate_money_execution_eligibility(
        frame, price_column="price_basis"
    )
    assert annotated["execution_eligible"].tolist() == [True, False]
    assert annotated["execution_scope"].tolist() == [
        "executable_primary_close",
        "nonexecutable_open_sensitivity",
    ]


def test_cascade_money_builder_includes_zero_bet_event_universe() -> None:
    probabilities = _cascade_probabilities()
    # Add a second event with no selected wagers. It must remain in every arm.
    second = probabilities.copy()
    second["event_key"] = "event-zero"
    second["event_start_date"] = pd.Timestamp("2023-05-08")
    probabilities = pd.concat([probabilities, second], ignore_index=True)
    quote_keys = probabilities[
        ["model", "event_key", "event_start_date", "round_num", "pair_key"]
    ].drop_duplicates()
    quote_keys["price_point"] = "close"
    quote_keys["line_policy"] = "best_of_three"
    rows = []
    for model, pnl in (("w2_model", 1.0), ("w0_model", 0.0), (audit.CASCADE_CONTROL, -1.0)):
        rows.append(
            {
                "model": model,
                "event_key": "event",
                "event_start_date": pd.Timestamp("2023-05-01"),
                "round_num": 1,
                "pair_key": "1|2",
                "opportunity_mode": "ev_ge_3pct",
                "price_point": "close",
                "line_policy": "best_of_three",
                "sizing": "flat_1u",
                "pnl": pnl,
                "stake": 1.0,
            }
        )
    bets = pd.DataFrame(rows)
    pairing = pd.DataFrame(
        {
            "w2_model": ["w2_model"],
            "w0_model": ["w0_model"],
            "quote_menu_status": ["exact"],
        }
    )
    result = audit.build_cascade_money_deltas(
        bets,
        quote_keys,
        pairing,
        expected_price_points=("close",),
        expected_line_policies=("best_of_three",),
        expected_sizings=("flat_1u",),
    )
    zero = result.loc[
        result["event_key"].eq("event-zero") & result["round_num"].eq(1)
    ]
    assert len(zero) == 1
    assert zero[["model_pnl", "reference_pnl", "control_pnl"]].eq(0.0).all().all()


def test_cascade_common_control_menu_filters_portfolio_and_attribution_bets() -> None:
    event_date = pd.Timestamp("2023-05-01")
    quote_rows = []
    for model in ("w2_model", "w0_model", audit.CASCADE_CONTROL):
        quote_rows.append(
            {
                "model": model,
                "event_key": "event",
                "event_start_date": event_date,
                "round_num": 1,
                "pair_key": "1|2",
                "price_point": "close",
                "line_policy": "best_of_three",
            }
        )
    # W2/W0 have an exact larger menu; C0 lacks this prediction pair.
    for model in ("w2_model", "w0_model"):
        quote_rows.append(
            {
                "model": model,
                "event_key": "event",
                "event_start_date": event_date,
                "round_num": 1,
                "pair_key": "3|4",
                "price_point": "close",
                "line_policy": "best_of_three",
            }
        )
    quote_keys = pd.DataFrame(quote_rows)

    bet_rows = []
    for model, pair_key, pnl in (
        ("w2_model", "1|2", 1.0),
        ("w0_model", "1|2", 0.25),
        (audit.CASCADE_CONTROL, "1|2", -0.5),
        # These materially different W2/W0 results must be irrelevant to every
        # control-relative event total and its attribution.
        ("w2_model", "3|4", 50.0),
        ("w0_model", "3|4", -40.0),
    ):
        bet_rows.append(
            {
                "model": model,
                "event_key": "event",
                "event_start_date": event_date,
                "round_num": 1,
                "pair_key": pair_key,
                "opportunity_mode": "ev_ge_3pct",
                "price_point": "close",
                "line_policy": "best_of_three",
                "sizing": "flat_1u",
                "pnl": pnl,
                "stake": 1.0,
                "selected_history_rounds": 10,
                "opponent_history_rounds": 10,
                "selected_img_history_rounds": 10,
                "opponent_img_history_rounds": 10,
                "selected_img_history_coverage": 0.8,
                "opponent_img_history_coverage": 0.8,
                "selected_reliability_ok": 1.0,
                "opponent_reliability_ok": 1.0,
                "selected_cross_tour_fallback_reason": "weekly_pga_core",
                "selected_event_withdrawal": False,
                "opponent_event_withdrawal": False,
                "selected_prior_event_withdrawal": (
                    np.nan if model == "w2_model" and pair_key == "1|2" else 0.0
                ),
                "opponent_prior_event_withdrawal": 0.0,
            }
        )
    bets = pd.DataFrame(bet_rows)
    pairing = pd.DataFrame(
        {
            "w2_model": ["w2_model"],
            "w0_model": ["w0_model"],
            "quote_menu_status": ["exact"],
        }
    )
    money = audit.build_cascade_money_deltas(
        bets,
        quote_keys,
        pairing,
        expected_price_points=("close",),
        expected_line_policies=("best_of_three",),
        expected_sizings=("flat_1u",),
    )
    assert money[["model_pnl", "reference_pnl", "control_pnl"]].iloc[0].tolist() == [
        1.0,
        0.25,
        -0.5,
    ]
    assert money["three_arm_common_quote_keys"].tolist() == [1]
    assert money["w2_quote_keys_excluded_for_control_common"].tolist() == [1]

    attributed = audit.attach_cascade_bet_attribution(money, bets, quote_keys)
    assert attributed[
        [
            "model_attribution_pnl",
            "reference_attribution_pnl",
            "control_attribution_pnl",
        ]
    ].iloc[0].tolist() == [1.0, 0.25, -0.5]
    assert attributed["model_missing_prior_withdrawal_bets"].tolist() == [1]
    assert attributed[
        "model_legacy_nullable_binary_serialization_repair_bets"
    ].tolist() == [1]


def test_cascade_analysis_bet_scope_is_explicit_pairing_plus_c0() -> None:
    bets = pd.DataFrame(
        {
            "model": ["w2_model", "w0_model", audit.CASCADE_CONTROL, "c1_extra"],
            "stake": [1.0, 2.0, 3.0, 4.0],
            "pnl": [0.5, -1.0, 2.0, -4.0],
        }
    )
    pairing = pd.DataFrame(
        {
            "w2_model": ["w2_model"],
            "w0_model": ["w0_model"],
            "quote_menu_status": ["exact"],
        }
    )
    analysis, scope = audit.select_cascade_analysis_bets(bets, pairing)
    assert set(analysis["model"]) == {
        "w2_model",
        "w0_model",
        audit.CASCADE_CONTROL,
    }
    assert scope["analysis_model_menu_pairing_plus_c0_exact"] is True
    assert scope["excluded_nonanalysis_rows"] == 1
    assert scope["excluded_nonanalysis_rows_by_model"] == {"c1_extra": 1}
    assert scope["excluded_nonanalysis_pnl"] == -4.0

    missing = bets.loc[~bets["model"].eq("w0_model")].copy()
    with pytest.raises(ValueError, match="missing a required paired analysis model"):
        audit.select_cascade_analysis_bets(missing, pairing)


def _cascade_weather_profiles_and_lineage(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
    ].copy()
    panel["event_key"] = "event"
    panel["event_start_date"] = "2023-05-01"
    path = tmp_path / "cascade_round_weather.parquet"
    panel.to_parquet(path, index=False)
    _, profiles, _, _ = audit.build_weather_profiles(
        path, required_event_keys={"event"}
    )
    lineage = pd.DataFrame(
        {
            "model": ["w2_model", "w2_model"],
            "event_key": ["event", "event"],
            "event_start_date": pd.to_datetime(["2023-05-01", "2023-05-01"]),
            "round_num": [1, 2],
            "lineage_completed_weather_observations": [0, 8],
            "lineage_players_with_completed_weather": [0, 8],
            "lineage_completed_weather_any": [False, True],
        }
    )
    return profiles, lineage


def test_cascade_money_weather_lineage_full_integration(tmp_path: Path) -> None:
    probabilities = audit._canonical_cascade_probabilities(
        _cascade_probabilities()
    )
    pairing = audit.build_w2_pairing_audit(
        _cascade_predictions(), probabilities
    )
    quote_keys = probabilities[
        ["model", "event_key", "event_start_date", "round_num", "pair_key"]
    ].drop_duplicates()
    quote_keys["price_point"] = "close"
    quote_keys["line_policy"] = "best_of_three"
    bets = []
    for model, pnl in (
        ("w2_model", 1.0),
        ("w0_model", 0.25),
        (audit.CASCADE_CONTROL, -0.5),
    ):
        for round_num in (1, 2):
            bets.append(
                {
                    "model": model,
                    "event_key": "event",
                    "event_start_date": pd.Timestamp("2023-05-01"),
                    "round_num": round_num,
                    "pair_key": "1|2",
                    "opportunity_mode": "ev_ge_3pct",
                    "price_point": "close",
                    "line_policy": "best_of_three",
                    "sizing": "flat_1u",
                    "pnl": pnl,
                    "stake": 1.0,
                }
            )
    money = audit.build_cascade_money_deltas(
        pd.DataFrame(bets),
        quote_keys,
        pairing,
        expected_price_points=("close",),
        expected_line_policies=("best_of_three",),
        expected_sizings=("flat_1u",),
    )
    profiles, lineage = _cascade_weather_profiles_and_lineage(tmp_path)
    attached = audit.attach_weather_to_cascade(money, profiles, lineage)
    assert attached.sort_values("round_num")["completed_weather_used_by_w2"].tolist() == [
        False,
        True,
    ]


def test_cascade_proper_weather_lineage_full_integration(tmp_path: Path) -> None:
    probabilities = audit._canonical_cascade_probabilities(
        _cascade_probabilities()
    )
    pairing = audit.build_w2_pairing_audit(
        _cascade_predictions(), probabilities
    )
    proper = audit.build_cascade_proper_deltas(probabilities, pairing)
    assert set(proper["model"]) == {"w2_model"}
    w0_control = proper.loc[
        proper["comparison_kind"].eq("w0_vs_current_cascade_control")
    ]
    assert set(w0_control["challenger"]) == {"w0_model"}
    assert set(w0_control["model"]) == {"w2_model"}
    profiles, lineage = _cascade_weather_profiles_and_lineage(tmp_path)
    extra_lineage = lineage.iloc[[0]].assign(
        event_key="prediction-event-with-no-offered-proper-score"
    )
    lineage = pd.concat([lineage, extra_lineage], ignore_index=True)
    attached = audit.attach_weather_to_cascade(proper, profiles, lineage)
    by_round = attached.groupby("round_num")["completed_weather_used_by_w2"].all()
    assert by_round.to_dict() == {1: False, 2: True}


def test_bootstrap_is_whole_event_deterministic_and_keeps_zero_events() -> None:
    metrics = pd.DataFrame(
        {
            "analysis_block": ["base"] * 4,
            "analysis_pair": ["pair"] * 4,
            "comparison_kind": ["w1_vs_w0"] * 4,
            "challenger": ["w1"] * 4,
            "reference": ["w0"] * 4,
            "metric": ["pnl_delta"] * 4,
            "higher_is_better": [True] * 4,
            "event_key": ["a", "b", "c", "d"],
            "event_start_date": pd.date_range("2023-01-01", periods=4, freq="7D"),
            "event_delta": [1.0, 0.0, -0.25, 0.5],
        }
    )
    first = audit.bootstrap_event_metrics(
        metrics, draws=10_000, seed=7, block_sizes=(2, 4)
    )
    second = audit.bootstrap_event_metrics(
        metrics, draws=10_000, seed=7, block_sizes=(2, 4)
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["bootstrap_method"]) == {
        "whole_event_iid",
        "circular_moving_block",
    }
    assert first["zero_delta_events"].eq(1).all()
    assert first["events"].eq(4).all()
    with pytest.raises(ValueError, match="at least 10000"):
        audit.bootstrap_event_metrics(metrics, draws=9999, seed=7)


def test_output_must_be_fresh_below_research_root_and_disjoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen_research"
    root.mkdir()
    output = root / "run_v1"
    assert audit.validate_fresh_output_path(output, research_root=root) == output.resolve()
    with pytest.raises(ValueError, match="strict descendant"):
        audit.validate_fresh_output_path(root, research_root=root)
    with pytest.raises(ValueError, match="strict descendant"):
        audit.validate_fresh_output_path(tmp_path / "outside", research_root=root)
    existing = root / "existing_empty"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="fresh, nonexistent"):
        audit.validate_fresh_output_path(existing, research_root=root)
    output.mkdir()
    upstream_inside_output = output / "upstream.bin"
    upstream_inside_output.write_bytes(b"bad overlap")
    with pytest.raises(ValueError, match="overlaps"):
        audit.validate_output_disjoint_from_inputs(output, [upstream_inside_output])


def _provenance_state(
    tmp_path: Path, *, input_name: str = "input.bin"
) -> tuple[
    audit.execution_provenance.ExecutionProvenance,
    Path,
    Path,
]:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    input_path = tmp_path / input_name
    input_path.write_text("frozen\n", encoding="utf-8")
    state = audit.execution_provenance.begin(
        output_dir=tmp_path / "output",
        primary_source=source,
        input_paths=[input_path],
        cli=["audit.py", "--bootstrap-draws", "10000", "--seed", "17"],
    )
    return state, source, input_path


def test_execution_provenance_preserves_source_runtime_and_argv(
    tmp_path: Path,
) -> None:
    state, _, _ = _provenance_state(tmp_path)
    payload = audit.execution_provenance.finalize(state)
    assert payload["complete"] is True
    assert payload["cli"] == [
        "audit.py",
        "--bootstrap-draws",
        "10000",
        "--seed",
        "17",
    ]
    assert payload["runtime"]["python_executable"]
    assert payload["runtime"]["pandas"]
    assert len(payload["sources"]) == len(payload["preserved_sources"]) == 1
    assert payload["sources"][0]["sha256"] == payload["preserved_sources"][0][
        "sha256"
    ]


def test_execution_provenance_rejects_input_mutation(tmp_path: Path) -> None:
    state, _, input_path = _provenance_state(tmp_path)
    input_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluator input changed"):
        audit.execution_provenance.finalize(state)


def test_execution_provenance_rejects_config_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "execution_configuration.json"
    config.write_text('{"seed": 7, "bootstrap": 10000}\n', encoding="utf-8")
    state = audit.execution_provenance.begin(
        output_dir=tmp_path / "output",
        primary_source=source,
        input_paths=[config],
        cli=["audit.py"],
    )
    config.write_text('{"seed": 8, "bootstrap": 10000}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluator input changed"):
        audit.execution_provenance.finalize(state)


def test_execution_provenance_rejects_preserved_source_copy_mutation(
    tmp_path: Path,
) -> None:
    state, _, _ = _provenance_state(tmp_path)
    preserved = Path(state.source_copy_fingerprints[0]["path"])
    preserved.write_text("VALUE = 999\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preserved evaluator source changed"):
        audit.execution_provenance.finalize(state)


def test_post_finalize_snapshot_recheck_rejects_report_interval_mutation(
    tmp_path: Path,
) -> None:
    state, _, input_path = _provenance_state(tmp_path)
    payload = audit.execution_provenance.finalize(state)
    assert payload["complete"] is True
    # Simulate a mutation after shared finalize but before the report/artifact
    # inventory has been sealed by the complete manifest.
    input_path.write_text("mutated during report assembly\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed between provenance snapshot"):
        audit._write_complete_manifest_after_final_snapshot(
            tmp_path / "output",
            {"complete": True, "research_only": True},
            state,
        )
    assert not (tmp_path / "output" / "manifest.json").exists()


def test_execution_configuration_binds_parsed_score_settings(tmp_path: Path) -> None:
    root = tmp_path / "research"
    root.mkdir()
    output = root / "run"
    args = Namespace(
        sealed_dir=tmp_path / "sealed",
        cascade_dir=tmp_path / "cascade",
        round_panel=tmp_path / "rounds.parquet",
        panel_manifest=tmp_path / "panel.json",
        output_dir=output,
        bootstrap_draws=12_345,
        block_sizes=[2, 5],
        seed=19,
    )
    path, payload = audit.bind_execution_configuration(
        audit.validate_fresh_output_path(output, research_root=root),
        args,
        block_sizes=(2, 5),
        argv=("audit.py", "--seed", "19"),
    )
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["raw_argv"] == ["audit.py", "--seed", "19"]
    assert payload["parsed_args"]["bootstrap_draws"] == 12_345
    assert payload["score_affecting_settings_bound_before_scoring"] == {
        "bootstrap_draws": 12_345,
        "block_sizes_events": [2, 5],
        "seed": 19,
    }


def test_sealed_selection_cutoff_must_parse_and_precede_sealed_period() -> None:
    assert audit.validate_sealed_selection_cutoff(
        {"selection_window_end": "2022-12-31"}
    ).year == 2022
    for value in (None, "nonsense", "2023-01-01", "2026-01-01"):
        with pytest.raises(ValueError, match="must parse and precede"):
            audit.validate_sealed_selection_cutoff({"selection_window_end": value})


def test_weather_delta_algebra_is_fail_closed(tmp_path: Path) -> None:
    panel = _weather_panel()
    panel.loc[0, "weather_delta_wind"] += 1002.5
    path = tmp_path / "bad_delta.parquet"
    panel.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="wind delta algebra identity failed"):
        audit.build_weather_profiles(
            path, required_event_keys={"pga:2023:a", "pga:2023:b"}
        )


def test_weather_mean_only_unmatched_row_is_group_context_not_observation(
    tmp_path: Path,
) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    unmatched = panel.index[0]
    for field in ("wind", "dew", "temp", "rain"):
        raw_column = f"weather_{field}"
        mean_column = f"weather_field_mean_{field}"
        delta_column = f"weather_delta_{field}"
        panel.loc[unmatched, [raw_column, delta_column]] = np.nan
        recomputed_mean = float(panel[raw_column].mean())
        panel[mean_column] = recomputed_mean
        observed = panel[raw_column].notna()
        panel.loc[observed, delta_column] = (
            panel.loc[observed, raw_column] - recomputed_mean
        )
    path = tmp_path / "mean_only_unmatched.parquet"
    panel.to_parquet(path, index=False)

    course, _, _, provenance = audit.build_weather_profiles(
        path, required_event_keys={"pga:2023:a"}
    )
    assert course["weather_observed_rows"].tolist() == [7]
    assert course["weather_coverage_rate"].tolist() == [pytest.approx(7 / 8)]
    assert provenance["weather_mean_only_unmatched_rows_by_field"] == {
        "wind": 1,
        "dew": 1,
        "temp": 1,
        "rain": 1,
    }


@pytest.mark.parametrize(
    ("missing_column", "message"),
    [
        ("weather_delta_wind", "incomplete wind delta identity"),
        ("weather_wind", "incomplete wind delta identity"),
        ("weather_field_mean_wind", "incomplete wind delta identity"),
    ],
)
def test_weather_partial_player_observation_is_rejected(
    tmp_path: Path, missing_column: str, message: str
) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    panel.loc[panel.index[0], missing_column] = np.nan
    path = tmp_path / f"partial_{missing_column}.parquet"
    panel.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=message):
        audit.build_weather_profiles(path, required_event_keys={"pga:2023:a"})


def test_weather_declared_mean_must_equal_raw_physical_course_mean(
    tmp_path: Path,
) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    panel["weather_wind"] = 9.5
    panel["weather_field_mean_wind"] = 109.5
    panel["weather_delta_wind"] = -100.0
    path = tmp_path / "false_center.parquet"
    panel.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="declared physical-course mean failed"):
        audit.build_weather_profiles(path, required_event_keys={"pga:2023:a"})


@pytest.mark.parametrize(("field", "value"), [("wind", -1.0), ("rain", -0.1)])
def test_weather_nonnegative_domains_reject_self_consistent_negative_values(
    tmp_path: Path, field: str, value: float
) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    panel[f"weather_{field}"] = value
    panel[f"weather_field_mean_{field}"] = value
    panel[f"weather_delta_{field}"] = 0.0
    path = tmp_path / f"negative_{field}.parquet"
    panel.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="must be nonnegative"):
        audit.build_weather_profiles(path, required_event_keys={"pga:2023:a"})


def test_weather_declared_infinity_is_rejected(tmp_path: Path) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    panel[["weather_wind", "weather_field_mean_wind"]] = np.inf
    panel["weather_delta_wind"] = 0.0
    path = tmp_path / "infinite_wind.parquet"
    panel.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="declared non-finite"):
        audit.build_weather_profiles(path, required_event_keys={"pga:2023:a"})


def test_weather_coverage_requires_both_wind_and_dew(tmp_path: Path) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
        & frame["round_num"].eq(1)
    ].copy()
    panel[["weather_dew", "weather_field_mean_dew", "weather_delta_dew"]] = np.nan
    path = tmp_path / "no_dew.parquet"
    panel.to_parquet(path, index=False)
    course, event_round, _, provenance = audit.build_weather_profiles(
        path, required_event_keys={"pga:2023:a"}
    )
    assert course["weather_coverage_rate"].tolist() == [0.0]
    assert event_round["weather_coverage_rate"].tolist() == [0.0]
    assert provenance["weather_coverage_rate"] == 0.0
    assert provenance["weather_coverage_requires_wind_and_dew"] is True


def test_r2_completed_weather_claim_requires_lineage_and_valid_prior_weather(
    tmp_path: Path,
) -> None:
    panel = _weather_panel().loc[
        lambda frame: frame["event_key"].eq("pga:2023:a")
    ].copy()
    prior = panel["round_num"].eq(1)
    weather_columns = [
        column
        for column in panel.columns
        if column.startswith("weather_") and column != "weather_source_teetime"
    ]
    panel.loc[prior, weather_columns] = np.nan
    path = tmp_path / "r1_missing_weather.parquet"
    panel.to_parquet(path, index=False)
    _, profiles, _, _ = audit.build_weather_profiles(
        path, required_event_keys={"pga:2023:a"}
    )
    event_rows = pd.DataFrame(
        {
            "model": ["w2_model"],
            "analysis_pair": ["w2_model__vs__w0_model"],
            "event_key": ["pga:2023:a"],
            "event_start_date": pd.to_datetime(["2023-06-01"]),
            "round_num": [2],
        }
    )
    lineage = pd.DataFrame(
        {
            "model": ["w2_model"],
            "event_key": ["pga:2023:a"],
            "event_start_date": pd.to_datetime(["2023-06-01"]),
            "round_num": [2],
            "lineage_completed_weather_observations": [1],
            "lineage_players_with_completed_weather": [1],
            "lineage_completed_weather_any": [True],
        }
    )
    with pytest.raises(ValueError, match=r"without nonzero prior wind\+dew"):
        audit.attach_weather_to_cascade(event_rows, profiles, lineage)
    lineage.loc[0, [
        "lineage_completed_weather_observations",
        "lineage_players_with_completed_weather",
    ]] = 0
    lineage.loc[0, "lineage_completed_weather_any"] = False
    attached = audit.attach_weather_to_cascade(event_rows, profiles, lineage)
    assert attached["completed_weather_profile_valid"].tolist() == [False]
    assert attached["completed_weather_used_by_w2"].tolist() == [False]


def test_w1_menu_rejects_two_key_to_one_silent_shrink() -> None:
    logical = _base_logical()
    logical = logical.loc[
        ~(
            logical["candidate"].eq("w0")
            & logical["event_key"].eq("event-1")
        )
    ].copy()
    pairing = pd.DataFrame(
        {
            "w1_candidate": ["w1"],
            "w0_candidate": ["w0"],
            "pairing_status": ["paired"],
            "selected_model_in_pair": [False],
        }
    )
    with pytest.raises(ValueError, match="identical complete quote menu"):
        audit.audit_w1_quote_menus(pairing, logical)


def test_w1_pairing_rejects_any_unpaired_arm_even_if_another_pair_survives() -> None:
    metadata = _candidate_metadata()
    broken = metadata.copy()
    broken.loc[broken["candidate"].eq("w1_neutral"), "alpha"] = 99.0
    with pytest.raises(ValueError, match="every W1 candidate requires exactly one"):
        audit.build_w1_pairing_audit(broken)


def test_canonical_logical_rejects_nan_and_out_of_domain_values() -> None:
    logical = _base_logical().rename(columns={"event_key": "mapped_event_key"})
    for column, value in (
        ("canonical_probability", np.nan),
        ("canonical_push_probability", 1.2),
        ("canonical_outcome", 2.0),
    ):
        broken = logical.copy()
        broken.loc[0, column] = value
        with pytest.raises(ValueError, match="non-finite|out-of-domain"):
            audit._canonical_logical(broken, "synthetic logical")


def test_cascade_probabilities_reject_component_and_pricing_domain_errors() -> None:
    broken = _cascade_probabilities()
    broken.loc[0, ["p1_probability", "p2_probability", "pricing_p1_probability"]] = [
        -0.2,
        1.2,
        2.0,
    ]
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        audit._canonical_cascade_probabilities(broken)
    nan_outcome = _cascade_probabilities()
    nan_outcome.loc[0, "derived_score_outcome"] = np.nan
    with pytest.raises(ValueError, match="invalid for its score-model eligibility"):
        audit._canonical_cascade_probabilities(nan_outcome)


def test_cascade_pricing_probability_must_follow_tie_rule_and_push_algebra() -> None:
    push_on_tie = _cascade_probabilities()
    push_on_tie.loc[0, [
        "p1_probability",
        "p2_probability",
        "push_probability",
        "pricing_p1_probability",
    ]] = [0.72, 0.18, 0.10, 0.99]
    with pytest.raises(ValueError, match="tie-rule/push algebra"):
        audit._canonical_cascade_probabilities(push_on_tie)

    push_on_tie.loc[0, "pricing_p1_probability"] = 0.8
    assert len(audit._canonical_cascade_probabilities(push_on_tie)) == len(
        push_on_tie
    )

    ties_lose = push_on_tie.copy()
    ties_lose.loc[0, "tie_rule"] = "ties lose"
    ties_lose.loc[0, "tie_rule_ties_lose"] = True
    ties_lose.loc[0, "pricing_p1_probability"] = 0.72
    ties_lose.loc[0, [
        "market_devig_proper_score_eligible",
        "binary_model_eligible",
    ]] = False
    ties_lose.loc[0, "market_devig_proper_score_exclusion_reason"] = (
        "ties_lose_without_draw_quote_conditional_devig_not_comparable"
    )
    ties_lose.loc[0, "market_devig_probability_scope"] = (
        "conditional_on_either_player_winning_no_draw_quote_attribution_only"
    )
    ties_lose.loc[0, [
        "model_minus_devig_open_pp",
        "model_minus_devig_close_pp",
    ]] = np.nan
    assert len(audit._canonical_cascade_probabilities(ties_lose)) == len(ties_lose)

    false_flag = ties_lose.copy()
    false_flag.loc[0, "tie_rule_ties_lose"] = False
    with pytest.raises(ValueError, match="differs from tie_rule"):
        audit._canonical_cascade_probabilities(false_flag)


def test_cascade_mixed_book_tie_rules_remain_quote_scoped() -> None:
    push_book = _cascade_probabilities()
    ties_lose_book = push_book.copy()
    ties_lose_book["book"] = "book_b"
    ties_lose_book["quote_id"] = ties_lose_book["quote_id"].astype(str) + "-book-b"
    ties_lose_book["tie_rule"] = "ties lose"
    ties_lose_book["tie_rule_ties_lose"] = True
    ties_lose_book["pricing_p1_probability"] = ties_lose_book["p1_probability"]
    ties_lose_book["market_devig_proper_score_eligible"] = False
    ties_lose_book["binary_model_eligible"] = False
    ties_lose_book["market_devig_proper_score_exclusion_reason"] = (
        "ties_lose_without_draw_quote_conditional_devig_not_comparable"
    )
    ties_lose_book["market_devig_probability_scope"] = (
        "conditional_on_either_player_winning_no_draw_quote_attribution_only"
    )
    ties_lose_book[
        ["model_minus_devig_open_pp", "model_minus_devig_close_pp"]
    ] = np.nan
    raw = pd.concat([push_book, ties_lose_book], ignore_index=True)

    canonical = audit._canonical_cascade_probabilities(raw)
    assert len(canonical) == len(push_book)
    assert "tie_rule" not in canonical
    assert "pricing_p1_probability" not in canonical
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), canonical)
    proper = audit.pairwise_cascade_proper_deltas(
        canonical,
        challenger="w2_model",
        reference="w0_model",
        analysis_pair="pair",
        comparison_kind="w2_vs_w0",
    )
    assert set(proper["metric"]) == {
        "three_way_logloss_delta",
        "three_way_brier_delta",
    }
    quote_audit, _ = audit.audit_cascade_quote_slices(
        _normalized_cascade_quotes(raw),
        raw,
        pairing,
        expected_price_points=("close",),
        expected_books=("book_a", "book_b"),
    )
    assert quote_audit["normalized_quote_authority_exact"].all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("market_devig_proper_score_eligible", False, "eligibility contract"),
        ("binary_model_eligible", False, "eligibility contract"),
        (
            "market_devig_proper_score_exclusion_reason",
            "forged_eligible",
            "exclusion reason",
        ),
        ("model_minus_devig_close_pp", 999.0, "gap algebra"),
    ],
)
def test_cascade_market_benchmark_contract_rejects_tampering(
    column: str, value: object, message: str
) -> None:
    broken = _cascade_probabilities()
    broken.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        audit._canonical_cascade_probabilities(broken)


def _cascade_predictions(weather_arm: str = "w2") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "w2_model",
                "candidate": "candidate",
                "variant": audit.W2_W0_VARIANTS["w2"],
                "weather_arm": weather_arm,
            },
            {
                "model": "w0_model",
                "candidate": "candidate",
                "variant": audit.W2_W0_VARIANTS["w0"],
                "weather_arm": "w0",
            },
            {
                "model": audit.CASCADE_CONTROL,
                "candidate": "control",
                "variant": "C0",
                "weather_arm": "legacy",
            },
        ]
    )


def test_w2_pairing_rejects_partial_menu_and_bogus_weather_arm() -> None:
    probabilities = audit._canonical_cascade_probabilities(_cascade_probabilities())
    partial = probabilities.loc[
        ~(
            probabilities["model"].eq("w0_model")
            & probabilities["round_num"].eq(2)
        )
    ].copy()
    with pytest.raises(ValueError, match="identical complete quote menu"):
        audit.build_w2_pairing_audit(_cascade_predictions(), partial)
    with pytest.raises(ValueError, match="every W2 arm requires exactly one W0"):
        audit.build_w2_pairing_audit(
            _cascade_predictions(weather_arm="made_up_weather_arm"), probabilities
        )


def test_w2_pairing_requires_exact_declared_candidate_set() -> None:
    probabilities = audit._canonical_cascade_probabilities(_cascade_probabilities())
    with pytest.raises(ValueError, match="candidate set differs"):
        audit.build_w2_pairing_audit(
            _cascade_predictions(),
            probabilities,
            expected_candidates={"candidate", "missing_candidate"},
        )


def _c3_coefficients() -> pd.DataFrame:
    rows = []
    for model, arm in (("w2_model", "w2"), ("w0_model", "w0")):
        for round_num in (1, 2, 3, 4):
            features = ["base_event_hybrid", "__intercept_scaled__"]
            if round_num >= 2:
                features.append(f"prev_mean_resid_{arm}_pt")
            for feature in features:
                rows.append(
                    {
                        "model": model,
                        "candidate": "candidate",
                        "mode": "pt",
                        "weather_arm": arm,
                        "round_num": round_num,
                        "prediction_year": 2023,
                        "training_max_event_end": "2022-12-31",
                        "feature": feature,
                        "standardized_coefficient": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_c3_coefficient_lineage_rejects_current_weather_feature() -> None:
    pairing = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "w2_model": ["w2_model"],
            "w0_model": ["w0_model"],
        }
    )
    result = audit.validate_c3_coefficient_feature_lineage(
        _c3_coefficients(), pairing
    )
    assert result["completed_state_feature_arm_isolated"] is True
    broken = _c3_coefficients()
    target = (
        broken["model"].eq("w2_model")
        & broken["round_num"].eq(2)
        & broken["feature"].str.startswith("prev_mean_resid")
    )
    broken.loc[target, "feature"] = "weather_delta_wind"
    with pytest.raises(ValueError, match="current-weather features"):
        audit.validate_c3_coefficient_feature_lineage(broken, pairing)


def _c3_weather_lineage_fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    coefficients = tmp_path / "weather_coefficients.csv"
    coefficients.write_text("coefficient\n0.1\n", encoding="utf-8")
    coefficient_record = {
        "path": str(coefficients.resolve()),
        "sha256": _sha(coefficients),
    }
    rows = []
    for arm in ("w0", "w2"):
        for round_num in (1, 2, 3, 4):
            rows.append(
                {
                    "model": f"model_{arm}",
                    "candidate": "candidate",
                    "event_key": "event-a",
                    "event_start_date": "2023-06-01",
                    "round_num": round_num,
                    "variant": f"C3_pt_continuous_partial_pool_{arm}",
                    "weather_arm": arm,
                    "weather_state_target_column": f"target_total_{arm}",
                    "weather_state_source_columns": (
                        audit.C3_WEATHER_SOURCE_COLUMNS[arm]
                    ),
                    "completed_rounds_available": round_num - 1,
                    "completed_rounds_used": round_num - 1,
                    "completed_weather_observations_available": round_num - 1,
                    "completed_weather_state_rows_with_valid_coverage": (
                        round_num - 1
                    ),
                    "completed_weather_observations_used": round_num - 1,
                    "completed_weather_state_rows_used_by_w2": (
                        round_num - 1 if arm == "w2" else 0
                    ),
                    "completed_weather_used_by_w2": (
                        arm == "w2" and round_num > 1
                    ),
                    "completed_prior_weather_coverage_rate": (
                        0.0 if round_num == 1 else 1.0
                    ),
                    "completed_prior_round_boundary": audit.C3_WEATHER_BOUNDARY,
                    "completed_prior_rounds_only": True,
                    "weather_coefficient_fit_cutoff": "2022-12-31",
                    "weather_coefficient_source": "frozen-discovery-weather",
                    "weather_coefficients_path": str(coefficients.resolve()),
                    "weather_coefficients_sha256": _sha(coefficients),
                    "target_or_upcoming_realized_weather_excluded": True,
                    "strict_point_in_time": True,
                    "current_round_weather_columns_role": (
                        audit.C3_CURRENT_WEATHER_ROLE
                    ),
                }
            )
    predictions = pd.DataFrame(rows)
    producer = {
        "contract": audit.C3_WEATHER_LINEAGE_CONTRACT,
        "rows": len(predictions),
        "arms": ["w0", "w2"],
        "weather_coefficients_path": str(coefficients.resolve()),
        "weather_coefficients_sha256": _sha(coefficients),
        "completed_rounds_available_definition": "round_num_minus_one",
        "completed_rounds_used_definition": (
            "nonmissing_strictly_shifted_total_state_rows"
        ),
        "completed_weather_observations_available_definition": (
            "strictly_prior_rows_with_finite_wind_and_dew_deltas"
        ),
        "completed_weather_state_rows_with_valid_coverage_definition": (
            "strictly_prior_rows_with_finite_wind_dew_and_all_w2_state_components"
        ),
        "completed_weather_used_by_w2_definition": (
            "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
        ),
        "completed_prior_weather_coverage_rate_definition": (
            "valid_strictly_prior_w2_weather_state_rows_divided_by_completed_rounds_available"
        ),
        "completed_prior_round_boundary": audit.C3_WEATHER_BOUNDARY,
        "target_or_upcoming_realized_weather_excluded": True,
        "current_round_weather_columns_role": audit.C3_CURRENT_WEATHER_ROLE,
        "failures": {
            key: 0 for key in audit.C3_WEATHER_LINEAGE_FAILURE_KEYS
        },
    }
    state = {
        "weather_coefficient_boundary_violations": 0,
        "completed_prior_round_boundary_violations": 0,
        "weather_coefficients_path": str(coefficients.resolve()),
        "weather_coefficients_sha256": _sha(coefficients),
        "weather_coefficient_fit_cutoff_column": "fit_max_round_date",
        "weather_coefficient_source_column": "coefficient_source",
        "completed_rounds_available_column": "completed_rounds_available",
        "completed_rounds_used_columns": {
            "w0": "completed_rounds_used_w0",
            "w2": "completed_rounds_used_w2",
        },
        "completed_weather_observations_available_column": (
            "completed_weather_observations_available"
        ),
        "completed_weather_state_rows_with_valid_coverage_column": (
            "completed_weather_state_rows_with_valid_coverage"
        ),
        "completed_weather_used_by_w2_definition": (
            "weather_arm_is_w2_and_strictly_prior_valid_wind_dew_w2_state_count_gt_zero"
        ),
        "completed_prior_weather_coverage_rate_column": (
            "completed_prior_weather_coverage_rate"
        ),
        "completed_prior_round_boundary": audit.C3_WEATHER_BOUNDARY,
        "target_or_upcoming_realized_weather_excluded": True,
        "target_or_upcoming_weather_feature_used": False,
        "weather_state_source_columns": dict(audit.C3_WEATHER_SOURCE_COLUMNS),
    }
    manifest = {
        "inputs": {"weather_coefficients": dict(coefficient_record)},
        "completion_authority": {
            "weather_coefficients": dict(coefficient_record)
        },
    }
    cascade_audit = {"c3_weather_lineage": producer, "state": state}
    authority = {"weather_coefficients": coefficient_record}
    return predictions, manifest, cascade_audit, authority


def test_c3_weather_lineage_positive_and_independent_counts(tmp_path: Path) -> None:
    predictions, manifest, cascade_audit, authority = _c3_weather_lineage_fixture(
        tmp_path
    )
    result, summary = audit.validate_c3_weather_state_lineage(
        predictions, manifest, cascade_audit, authority
    )
    assert result["producer_lineage_audit_verified"] is True
    assert result["target_or_upcoming_realized_weather_excluded"] is True
    assert summary["lineage_completed_weather_any"].tolist() == [
        False,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_column", "missing columns"),
        ("count_crosses_boundary", "completed-weather state crosses"),
        ("target_weather_used", "boolean lineage is false"),
        ("r2_missing_weather_claim", "W2 completed-weather use count is false"),
        ("string_boolean", "must contain exact booleans"),
        ("string_count", "nonnegative integer counts"),
        ("coefficient_hash", "non-authoritative coefficient hash"),
        ("producer_failure", "audit has failures"),
    ],
)
def test_c3_weather_lineage_tamper_cases_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    predictions, manifest, cascade_audit, authority = _c3_weather_lineage_fixture(
        tmp_path
    )
    if mutation == "missing_column":
        predictions = predictions.drop(columns=["completed_prior_rounds_only"])
    elif mutation == "count_crosses_boundary":
        predictions.loc[
            predictions["round_num"].eq(2),
            "completed_weather_state_rows_with_valid_coverage",
        ] = 2
    elif mutation == "target_weather_used":
        predictions.loc[0, "target_or_upcoming_realized_weather_excluded"] = False
    elif mutation == "r2_missing_weather_claim":
        target = predictions["weather_arm"].eq("w2") & predictions[
            "round_num"
        ].eq(2)
        predictions.loc[
            target,
            [
                "completed_weather_observations_available",
                "completed_weather_state_rows_with_valid_coverage",
                "completed_weather_observations_used",
                "completed_prior_weather_coverage_rate",
            ],
        ] = 0
    elif mutation == "string_boolean":
        predictions["completed_prior_rounds_only"] = predictions[
            "completed_prior_rounds_only"
        ].astype(object)
        predictions.loc[0, "completed_prior_rounds_only"] = "True"
    elif mutation == "string_count":
        predictions["completed_rounds_used"] = predictions[
            "completed_rounds_used"
        ].astype(object)
        predictions.loc[0, "completed_rounds_used"] = "0"
    elif mutation == "coefficient_hash":
        predictions.loc[0, "weather_coefficients_sha256"] = "0" * 64
    elif mutation == "producer_failure":
        cascade_audit["c3_weather_lineage"]["failures"][
            "target_or_upcoming_exclusion_false"
        ] = 1
    with pytest.raises(ValueError, match=message):
        audit.validate_c3_weather_state_lineage(
            predictions, manifest, cascade_audit, authority
        )


def _quoted_cascade_probabilities() -> pd.DataFrame:
    frame = _cascade_probabilities()
    return frame


def _normalized_cascade_quotes(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[
        [
            "event_key",
            "event_start_date",
            "round_num",
            "pair_key",
            "book",
            "quote_id",
            "p1_id",
            "p2_id",
            "p1_outcome",
            "p2_outcome",
            "tie_rule",
            "p1_open",
            "p2_open",
            "p1_close",
            "p2_close",
        ]
    ].drop_duplicates()


def test_cascade_price_slice_rejects_missing_quote_in_one_arm() -> None:
    raw = _quoted_cascade_probabilities()
    probabilities = audit._canonical_cascade_probabilities(raw)
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), probabilities)
    raw.loc[
        raw["model"].eq("w0_model") & raw["round_num"].eq(2), "p1_close"
    ] = np.nan
    with pytest.raises(ValueError, match="differs from normalized executable"):
        audit.audit_cascade_quote_slices(
            _normalized_cascade_quotes(_quoted_cascade_probabilities()),
            raw,
            pairing,
            expected_price_points=("close",),
            expected_books=("book_a",),
        )


def test_cascade_quote_and_proper_control_use_verified_pairwise_common_subset() -> None:
    full = _quoted_cascade_probabilities()
    normalized = _normalized_cascade_quotes(full)
    subset = full.loc[
        ~(
            full["model"].eq(audit.CASCADE_CONTROL)
            & full["round_num"].eq(2)
        )
    ].copy()
    probabilities = audit._canonical_cascade_probabilities(subset)
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), probabilities)
    verified_menu = subset[["model", "quote_id"]].drop_duplicates()
    quote_audit, quote_keys = audit.audit_cascade_quote_slices(
        normalized,
        subset,
        pairing,
        expected_price_points=("close",),
        expected_books=("book_a",),
        verified_model_quote_menu=verified_menu,
    )
    assert quote_audit["w2_w0_complete_exact_menu"].all()
    assert not quote_audit["three_arm_complete_exact_menu"].any()
    assert quote_audit["w2_quote_keys_outside_control_common"].eq(1).all()
    assert set(
        quote_keys.loc[quote_keys["model"].eq(audit.CASCADE_CONTROL), "round_num"]
    ) == {1}

    proper = audit.build_cascade_proper_deltas(probabilities, pairing)
    w2_w0 = proper.loc[proper["comparison_kind"].eq("w2_vs_matched_w0")]
    control = proper.loc[
        proper["comparison_kind"].isin(
            ["w2_vs_current_cascade_control", "w0_vs_current_cascade_control"]
        )
    ]
    assert set(w2_w0["round_num"]) == {1, 2}
    assert set(w2_w0["proper_quote_menu_scope"]) == {"exact"}
    assert set(control["round_num"]) == {1}
    assert set(control["proper_quote_menu_scope"]) == {"pairwise_common"}


def test_cascade_verified_quote_menu_excludes_pre_window_rows() -> None:
    current = _quoted_cascade_probabilities()
    pre_window = current.iloc[[0]].copy()
    pre_window["event_start_date"] = "2022-12-31"
    pre_window["quote_id"] = "pre-window-quote"
    combined = pd.concat([current, pre_window], ignore_index=True)

    menu = audit._cascade_verified_model_quote_menu(combined)
    assert "pre-window-quote" not in set(menu["quote_id"])
    assert set(map(tuple, menu.to_numpy())) == set(
        map(tuple, current[["model", "quote_id"]].drop_duplicates().to_numpy())
    )


def test_report_tables_fall_back_when_optional_tabulate_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> str:
        raise ImportError("tabulate intentionally unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", unavailable)
    report_audit = {
        "weather": {"events": 2, "event_rounds": 8},
        "executable_primary_close_money_rows": 10,
        "nonexecutable_open_sensitivity_money_rows": 10,
        "base_w1_available": False,
        "base_w1_unavailable_reason": "not frozen",
        "sealed_bet_settlement_authority": {"rows": 100},
        "cascade_bet_settlement_authority": {"rows": 200},
        "cascade_bookless_proper_score_void_exclusion_ledger": {"rows": 3},
        "bootstrap_draws": 1,
        "block_sizes_events": [2, 3, 4],
    }
    associations = pd.DataFrame(
        {
            "analysis_block": ["cascade"],
            "analysis_pair": ["w2__vs__w0"],
            "events": [2],
            "pnl_delta_vs_reference": [0.125],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "analysis_block": ["cascade"],
            "comparison_kind": ["normalization_vs_matched_w0"],
            "metric": ["pnl_delta"],
            "events": [2],
            "estimate": [0.125],
        }
    )

    audit._write_report(tmp_path, report_audit, associations, bootstrap)

    rendered = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert rendered.count("```text") == 2
    assert "w2__vs__w0" in rendered
    assert "normalization_vs_matched_w0" in rendered
    assert "0.1250" in rendered


@pytest.mark.parametrize(
    ("column", "mutated_value"),
    [
        ("p1_close", -110.001),
        ("p1_outcome", 1.000001),
    ],
)
def test_cascade_normalized_quote_authority_rejects_small_numeric_mutation(
    column: str, mutated_value: float
) -> None:
    original = _quoted_cascade_probabilities()
    normalized = _normalized_cascade_quotes(original)
    probabilities = audit._canonical_cascade_probabilities(original)
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), probabilities)
    mutated = original.copy()
    mutated.loc[
        mutated["model"].eq("w0_model") & mutated["round_num"].eq(1),
        column,
    ] = mutated_value

    with pytest.raises(ValueError, match="differs from normalized executable"):
        audit.audit_cascade_quote_slices(
            normalized,
            mutated,
            pairing,
            expected_price_points=("close",),
            expected_books=("book_a",),
        )


def test_cascade_normalized_quote_authority_rejects_forged_string_as_missing() -> None:
    original = _quoted_cascade_probabilities()
    probabilities = audit._canonical_cascade_probabilities(original)
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), probabilities)
    normalized = _normalized_cascade_quotes(original)
    normalized.loc[normalized["round_num"].eq(1), "p1_outcome"] = np.nan
    mutated = original.copy()
    mutated["p1_outcome"] = mutated["p1_outcome"].astype(object)
    mutated.loc[mutated["round_num"].eq(1), "p1_outcome"] = np.nan
    mutated.loc[
        mutated["model"].eq("w0_model") & mutated["round_num"].eq(1),
        "p1_outcome",
    ] = "forged"

    with pytest.raises(ValueError, match="nonmissing non-numeric"):
        audit.audit_cascade_quote_slices(
            normalized,
            mutated,
            pairing,
            expected_price_points=("close",),
            expected_books=("book_a",),
        )


def test_sealed_price_slice_does_not_turn_missing_prices_into_zero_events() -> None:
    identity = pd.DataFrame(
        {
            "local_event_key": ["event-a", "event-b"],
            "canonical_event_key": ["canon-a", "canon-b"],
        }
    )
    quotes = pd.DataFrame(
        {
            "mapped_event_key": ["canon-a", "canon-b"],
            "event_start_date": ["2023-05-01", "2023-05-08"],
            "market": ["72h", "72h"],
            "book": ["book_a", "book_a"],
            "pair_key": ["1|2", "1|2"],
            "p1_close": [-110.0, np.nan],
            "p2_close": [-110.0, np.nan],
        }
    )
    universe = audit.build_sealed_price_quote_universe(
        quotes,
        identity,
        expected_price_bases=("close",),
        expected_line_policies=("book_a", "best_of_three"),
        expected_books=("book_a",),
    )
    assert set(universe["event_key"]) == {"event-a"}
    rows = []
    for opportunity in (
        "all_available",
        "pairwise_common_with_strict_pt_baseline",
    ):
        for policy in ("book_a", "best_of_three"):
            for event, date in (
                ("event-a", "2023-05-01"),
                ("event-b", "2023-05-08"),
            ):
                rows.append(
                        {
                            "candidate": "challenger",
                            "baseline": "baseline",
                        "opportunity_mode": opportunity,
                        "event_key": event,
                        "event_start_date": pd.Timestamp(date),
                        "market": "72h",
                        "price_basis": "close",
                        "line_policy": policy,
                        "sizing": "flat_1u",
                        "reliability_policy": "all",
                    }
                )
    with pytest.raises(ValueError, match="executable price-specific quote universe"):
        audit.validate_sealed_money_slice_coverage(
            pd.DataFrame(rows),
            universe,
            expected_candidates={"challenger", "baseline"},
            baseline_candidate="baseline",
            expected_price_bases=("close",),
            expected_line_policies=("book_a", "best_of_three"),
            expected_sizings=("flat_1u",),
            expected_reliability_policies=("all",),
            expected_markets=("72h",),
        )


def test_sealed_event_key_normalizer_validates_existing_producer_identity() -> None:
    identity = pd.DataFrame(
        {
            "local_event_key": ["event-a", "event-b"],
            "canonical_event_key": ["canon-a", "canon-b"],
        }
    )
    frame = pd.DataFrame(
        {
            "event_key": ["event-a", "canon-b"],
            "local_event_key": ["event-a", "event-b"],
            "canonical_event_key": ["canon-a", "canon-b"],
            "value": [1, 2],
        }
    )

    normalized = audit.normalize_sealed_event_keys(
        frame, identity, label="sealed producer rows"
    )

    assert normalized["event_key"].tolist() == ["event-a", "event-b"]
    assert normalized["source_sealed_event_key"].tolist() == ["event-a", "canon-b"]
    assert normalized["canonical_event_key"].tolist() == ["canon-a", "canon-b"]
    assert "local_event_key" not in normalized

    for column, value, message in (
        ("local_event_key", "event-b", "producer local_event_key disagrees"),
        ("canonical_event_key", "canon-b", "producer canonical_event_key disagrees"),
    ):
        invalid = frame.copy()
        invalid.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            audit.normalize_sealed_event_keys(
                invalid, identity, label="sealed producer rows"
            )


def test_sealed_money_price_filter_excludes_only_exact_zero_bet_padding() -> None:
    money = pd.DataFrame(
        {
            "event_key": ["event-a", "event-b"],
            "event_start_date": pd.to_datetime(["2023-05-01", "2023-05-08"]),
            "market": ["72h", "72h"],
            "price_basis": ["close", "close"],
            "line_policy": ["pinnacle", "pinnacle"],
            "candidate_pnl": [1.25, 0.0],
            "baseline_pnl": [-1.0, 0.0],
            "event_delta": [2.25, 0.0],
            "zero_bet_event": [False, True],
        }
    )
    universe = money.loc[
        money["event_key"].eq("event-a"),
        ["event_key", "event_start_date", "market", "price_basis", "line_policy"],
    ].copy()

    filtered, details = audit.filter_sealed_money_to_executable_price_universe(
        money, universe
    )

    assert filtered["event_key"].tolist() == ["event-a"]
    assert details == {
        "source_rows": 2,
        "retained_executable_price_rows": 1,
        "excluded_nonexecutable_zero_pad_rows": 1,
        "excluded_nonexecutable_event_price_policy_keys": 1,
        "excluded_nonexecutable_events": 1,
        "all_exclusions_exact_zero_bet_pads": True,
        "excluded_before_money_attribution_and_bootstrap": True,
    }

    for column, value in (
        ("candidate_pnl", 0.01),
        ("baseline_pnl", -0.01),
        ("event_delta", 0.01),
        ("zero_bet_event", False),
    ):
        invalid = money.copy()
        invalid.loc[invalid["event_key"].eq("event-b"), column] = value
        with pytest.raises(ValueError, match="not an exact zero-bet pad"):
            audit.filter_sealed_money_to_executable_price_universe(invalid, universe)


def test_sealed_money_rejects_relabelled_strict_pt_control() -> None:
    money = pd.DataFrame(
        {
            "candidate": ["challenger"],
            "baseline": ["relabeled_challenger"],
            "opportunity_mode": ["all_available"],
            "event_key": ["event"],
            "event_start_date": [pd.Timestamp("2023-05-01")],
            "market": ["72h"],
            "price_basis": ["close"],
            "line_policy": ["book_a"],
            "sizing": ["flat_1u"],
            "reliability_policy": ["all"],
        }
    )
    quote_universe = pd.DataFrame(
        {
            "event_key": ["event"],
            "event_start_date": [pd.Timestamp("2023-05-01")],
            "market": ["72h"],
            "price_basis": ["close"],
            "line_policy": ["book_a"],
        }
    )
    with pytest.raises(ValueError, match="exact frozen strict-PT control"):
        audit.validate_sealed_money_slice_coverage(
            money,
            quote_universe,
            expected_candidates={"challenger", audit.frozen_handoff.STRICT_PT_CONTROL},
            baseline_candidate=audit.frozen_handoff.STRICT_PT_CONTROL,
            expected_price_bases=("close",),
            expected_line_policies=("book_a",),
            expected_sizings=("flat_1u",),
            expected_reliability_policies=("all",),
            expected_markets=("72h",),
            expected_opportunity_modes=("all_available",),
        )


def _cascade_bets_for_validation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "w2_model",
                "candidate": "candidate",
                "variant": audit.W2_W0_VARIANTS["w2"],
                "event_key": "event",
                "event_start_date": "2023-05-01",
                "round_num": 1,
                "round_market": "r1",
                "pair_key": "1|2",
                "selection_id": "1",
                "opponent_id": "2",
                "price_point": "close",
                "line_policy": "best_of_three",
                "sizing": "flat_1u",
                "opportunity_mode": "ev_ge_3pct",
                "stake": 1.0,
                "pnl": -1.0,
                "is_graded": True,
            }
        ]
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("is_graded", "False", "exact booleans"),
        ("stake", -1.0, "negative stake"),
        ("pnl", np.inf, "non-finite"),
    ],
)
def test_cascade_bets_reject_boolish_and_invalid_money(
    column: str, value: object, message: str
) -> None:
    bets = _cascade_bets_for_validation()
    if column == "is_graded":
        bets[column] = bets[column].astype(object)
    bets.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        audit._validate_cascade_bets(bets)


def _sealed_settlement_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_id = "sealed|quote|1"
    quotes = pd.DataFrame(
        [
            {
                "quote_id": quote_id,
                "mapped_event_key": "event",
                "event_start_date": "2023-05-01",
                "market": "72h",
                "book": "betonline",
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_open": -105.0,
                "p2_open": -115.0,
                "p1_close": -110.0,
                "p2_close": 100.0,
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "p1_outcome_text": "win",
                "p2_outcome_text": "loss",
                "tie_rule": "push on tie",
            }
        ]
    )
    profit = 100.0 / 110.0
    bets = pd.DataFrame(
        [
            {
                "candidate": "w1",
                "selection_key": f"w1|{quote_id}|p1",
                "event_key": "event",
                "event_start_date": "2023-05-01",
                "market": "72h",
                "pair_key": "1|2",
                "selection_id": "1",
                "opponent_id": "2",
                "book": "betonline",
                "price_basis": "close",
                "line_policy": "betonline",
                "execution_selection_odds": -110.0,
                "execution_opponent_odds": 100.0,
                "same_book_close_selection_odds": -110.0,
                "same_book_close_opponent_odds": 100.0,
                "close_odds": -110.0,
                "outcome_numeric": 1.0,
                "outcome": "win",
                "tie_rule": "push on tie",
                "settlement_status": "win",
                "is_graded": True,
                "stake": 1.0,
                "pnl": profit,
                "return_amount": 1.0 + profit,
                "paid_fraction": 1.0,
                "sizing": "flat_1u",
            }
        ]
    )
    return bets, quotes


def test_sealed_settlement_is_rebuilt_from_exact_quote_book_side_and_price() -> None:
    bets, quotes = _sealed_settlement_fixture()
    result = audit.validate_sealed_bet_settlement_authority(bets, quotes)
    assert result["rows"] == 1
    assert result["pnl_max_abs_error"] == 0.0
    assert result["selection_key_quote_id_side_reconstructed"] is True


def test_sealed_settlement_accepts_only_numeric_tie_with_void_settlement_text() -> None:
    bets, quotes = _sealed_settlement_fixture()
    quotes.loc[0, "p1_outcome"] = 0.5
    quotes.loc[0, "p1_outcome_text"] = "void"
    quotes.loc[0, "tie_rule"] = "Bets are void in event of tie"
    bets.loc[0, "outcome_numeric"] = 0.5
    bets.loc[0, "outcome"] = "void"
    bets.loc[0, "tie_rule"] = "Bets are void in event of tie"
    bets.loc[0, "settlement_status"] = "void"
    bets.loc[0, "pnl"] = 0.0
    bets.loc[0, "return_amount"] = 1.0
    bets.loc[0, "paid_fraction"] = np.nan

    result = audit.validate_sealed_bet_settlement_authority(bets, quotes)

    assert result["voids"] == 1
    assert result["pushes"] == 0
    assert result["numeric_tie_text_void_rows"] == 1
    assert result["numeric_tie_text_void_is_producer_settlement_encoding"] is True

    quotes.loc[0, "p1_outcome_text"] = "loss"
    bets.loc[0, "outcome"] = "loss"
    with pytest.raises(ValueError, match="numeric/text outcomes disagree"):
        audit.validate_sealed_bet_settlement_authority(bets, quotes)


@pytest.mark.parametrize("authority_outcome", [np.nan, "forged"])
def test_sealed_settlement_rejects_forged_string_numeric_outcome(
    authority_outcome: object,
) -> None:
    bets, quotes = _sealed_settlement_fixture()
    quotes["p1_outcome"] = quotes["p1_outcome"].astype(object)
    bets["outcome_numeric"] = bets["outcome_numeric"].astype(object)
    quotes.loc[0, "p1_outcome"] = authority_outcome
    bets.loc[0, "outcome_numeric"] = "forged"

    with pytest.raises(ValueError, match="nonmissing non-numeric"):
        audit.validate_sealed_bet_settlement_authority(bets, quotes)


def test_sealed_settlement_rejects_forged_paid_fraction_as_missing() -> None:
    bets, quotes = _sealed_settlement_fixture()
    quotes.loc[0, "p1_outcome"] = np.nan
    quotes.loc[0, "p1_outcome_text"] = "void"
    bets.loc[0, "outcome_numeric"] = np.nan
    bets.loc[0, "outcome"] = "void"
    bets.loc[0, "settlement_status"] = "void"
    bets.loc[0, "pnl"] = 0.0
    bets.loc[0, "return_amount"] = 1.0
    bets["paid_fraction"] = bets["paid_fraction"].astype(object)
    bets.loc[0, "paid_fraction"] = "forged"

    with pytest.raises(ValueError, match="nonmissing non-numeric"):
        audit.validate_sealed_bet_settlement_authority(bets, quotes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("forged_quote_id", "absent from authority"),
        ("forged_selection_id", "selection_id differs"),
        ("execution_odds", "execution_selection_odds differs"),
        ("pnl", "P&L differs"),
        ("stake", "stake differs"),
        ("paid_fraction", "paid_fraction differs"),
    ],
)
def test_sealed_settlement_rejects_forged_identity_price_stake_and_pnl(
    mutation: str, message: str
) -> None:
    bets, quotes = _sealed_settlement_fixture()
    if mutation == "forged_quote_id":
        bets.loc[0, "selection_key"] = "w1|forged|quote|p1"
    elif mutation == "forged_selection_id":
        bets.loc[0, "selection_id"] = "999"
    elif mutation == "execution_odds":
        bets.loc[0, ["execution_selection_odds", "close_odds"]] = -110.001
    elif mutation == "pnl":
        bets.loc[0, "pnl"] += 0.000001
    elif mutation == "stake":
        bets.loc[0, "stake"] = 1.000001
    elif mutation == "paid_fraction":
        bets.loc[0, "paid_fraction"] = 0.999999
    with pytest.raises(ValueError, match=message):
        audit.validate_sealed_bet_settlement_authority(bets, quotes)


def _cascade_settlement_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quote_id = "event|r1|betonline|1|2"
    quotes = pd.DataFrame(
        [
            {
                "quote_id": quote_id,
                "event_key": "event",
                "event_start_date": "2023-05-01",
                "round_num": 1,
                "round_market": "r1",
                "book": "betonline",
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_open": -105.0,
                "p2_open": -115.0,
                "p1_close": -110.0,
                "p2_close": 100.0,
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "tie_rule": "push on tie",
            }
        ]
    )
    classification = pd.DataFrame(
        [
            {
                "model": "w2_model",
                "quote_id": quote_id,
                "no_score_void": False,
                "other_void_with_scores": False,
                "actual_score_tie": False,
            }
        ]
    )
    profit = 100.0 / 110.0
    bets = pd.DataFrame(
        [
            {
                "model": "w2_model",
                "event_key": "event",
                "event_start_date": "2023-05-01",
                "round_num": 1,
                "round_market": "r1",
                "pair_key": "1|2",
                "selection_id": "1",
                "opponent_id": "2",
                "book": "betonline",
                "price_point": "close",
                "line_policy": "betonline",
                "execution_odds": -110.0,
                "close_odds": -110.0,
                "archived_open_odds": -105.0,
                "archived_close_odds": -110.0,
                "outcome_numeric": 1.0,
                "outcome": "win",
                "tie_rule": "push on tie",
                "tie_rule_ties_lose": False,
                "settlement_status": "win",
                "is_graded": True,
                "stake": 1.0,
                "pnl": profit,
                "return_amount": 1.0 + profit,
                "paid_fraction": 1.0,
                "sizing": "flat_1u",
            }
        ]
    )
    return bets, quotes, classification


def test_cascade_settlement_is_rebuilt_and_no_score_void_is_not_a_push() -> None:
    bets, quotes, classification = _cascade_settlement_fixture()
    result = audit.validate_cascade_bet_settlement_authority(
        bets, quotes, classification
    )
    assert result["wins"] == 1
    assert result["pnl_max_abs_error"] == 0.0

    quotes.loc[0, ["p1_outcome", "p2_outcome"]] = 0.5
    classification.loc[0, "no_score_void"] = True
    bets.loc[0, "outcome_numeric"] = 0.5
    bets.loc[0, "outcome"] = "void"
    bets.loc[0, "settlement_status"] = "void"
    bets.loc[0, "pnl"] = 0.0
    bets.loc[0, "return_amount"] = 1.0
    bets.loc[0, "paid_fraction"] = np.nan
    result = audit.validate_cascade_bet_settlement_authority(
        bets, quotes, classification
    )
    assert result["voids"] == 1
    assert result["pushes"] == 0


def test_cascade_settlement_rejects_forged_string_when_outcome_authority_missing() -> None:
    bets, quotes, classification = _cascade_settlement_fixture()
    quotes.loc[0, "p1_outcome"] = np.nan
    classification.loc[0, "no_score_void"] = True
    bets["outcome_numeric"] = bets["outcome_numeric"].astype(object)
    bets.loc[0, "outcome_numeric"] = "forged"
    bets.loc[0, "outcome"] = "void"
    bets.loc[0, "settlement_status"] = "void"
    bets.loc[0, "pnl"] = 0.0
    bets.loc[0, "return_amount"] = 1.0
    bets.loc[0, "paid_fraction"] = np.nan

    with pytest.raises(ValueError, match="nonmissing non-numeric"):
        audit.validate_cascade_bet_settlement_authority(
            bets, quotes, classification
        )


def test_cascade_settlement_rejects_forged_archived_odds_as_missing() -> None:
    bets, quotes, classification = _cascade_settlement_fixture()
    quotes.loc[0, "p1_open"] = np.nan
    bets["archived_open_odds"] = bets["archived_open_odds"].astype(object)
    bets.loc[0, "archived_open_odds"] = "forged"

    with pytest.raises(ValueError, match="nonmissing non-numeric"):
        audit.validate_cascade_bet_settlement_authority(
            bets, quotes, classification
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("forged_pair", "absent from authority"),
        ("execution_odds", "execution_odds differs"),
        ("pnl", "P&L differs"),
        ("outcome", "outcome_numeric"),
        ("paid_fraction", "paid_fraction differs"),
    ],
)
def test_cascade_settlement_rejects_forged_identity_price_outcome_and_pnl(
    mutation: str, message: str
) -> None:
    bets, quotes, classification = _cascade_settlement_fixture()
    if mutation == "forged_pair":
        bets.loc[0, "pair_key"] = "1|999"
    elif mutation == "execution_odds":
        bets.loc[0, ["execution_odds", "close_odds"]] = -110.001
    elif mutation == "pnl":
        bets.loc[0, "pnl"] += 0.000001
    elif mutation == "outcome":
        bets.loc[0, "outcome_numeric"] = 0.0
    elif mutation == "paid_fraction":
        bets.loc[0, "paid_fraction"] = 0.999999
    with pytest.raises(ValueError, match=message):
        audit.validate_cascade_bet_settlement_authority(
            bets, quotes, classification
        )


def _sealed_candidate_authority_fixture(
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    dict[str, object],
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    selected = "selected_w1"
    included = [selected, "w0", "w1_sibling"]
    base_names = (
        selected,
        audit.frozen_handoff.STRICT_PT_CONTROL,
        audit.STRICT_LEGACY_CONTROL,
        audit.RETROSPECTIVE_CONTROL,
        "w0",
        "w1_sibling",
    )
    opener = f"{selected}{audit.FROZEN_OPENER_SUFFIX}"
    names = (*base_names, opener)
    catalog_rows = []
    for index, name in enumerate(base_names):
        weather_arm = "w0" if name == "w0" else (
            "w1" if name in {selected, "w1_sibling"} else "none"
        )
        target_arm = "observed" if name == "w0" else (
            "feature_only" if weather_arm == "w1" else "control"
        )
        catalog_rows.append(
            {
                "name": name,
                "family": "ridge",
                "source": "total_cat",
                "weather_arm": weather_arm,
                "target_arm": target_arm,
                "refit_schedule": "event_batch",
                "alpha": 10.0,
                "l1_ratio": np.nan,
                "huber_epsilon": np.nan,
                "dynamic_half_life": np.nan,
                "prior_rounds": np.nan,
                "process_variance_per_day": np.nan,
                "short_game_penalty": 2.0,
                "prediction_envelope_applied": False,
                "prediction_envelope_method": "none",
                "prediction_envelope_scale_factor": 1.0,
                "prediction_envelope_multiplier": 1.0,
                "raw_regression_component_selection_eligible": False,
                "control": index in {1, 2, 3},
                "sensitivity_only": name in {"w0", "w1_sibling"},
                "selection_eligible": name == selected,
            }
        )
    catalog = pd.DataFrame(catalog_rows)
    catalog_path = tmp_path / "candidate_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    handoff = SimpleNamespace(
        selected_name=selected,
        selection_role="diagnostic_challenger",
        selection={
            "locked_diagnostic_ablation_set": {
                "selected_candidate": selected,
                "included_candidates": included,
                "locked_before_sealed_scoring": True,
                "may_reselect_or_promote_from_sealed": False,
            }
        },
        candidate_catalog_path=catalog_path,
    )
    roles = {name: "locked_descriptive_ablation" for name in included}
    roles[selected] = "diagnostic_challenger"
    roles[audit.frozen_handoff.STRICT_PT_CONTROL] = (
        "strict_point_in_time_architecture_control"
    )
    roles[audit.STRICT_LEGACY_CONTROL] = (
        "strict_prior_legacy_input_current_coefficients_control"
    )
    roles[audit.RETROSPECTIVE_CONTROL] = "exact_current_code_retrospective_control"
    roles[opener] = "frozen_discovery_opener_blend_sensitivity"
    opener_payload = {
        "candidate": opener,
        "source_candidate": selected,
        "selection_window_end": "2022-12-31",
        "sealed_data_used_for_selection": False,
        "neutral_base_redefined": False,
    }
    manifest = {
        "selected_model": selected,
        "selected_model_role": "diagnostic_challenger",
        "evaluated_candidate_names": list(names),
        "locked_diagnostic_ablation_set": handoff.selection[
            "locked_diagnostic_ablation_set"
        ],
        "opener_blend_sensitivity": dict(opener_payload),
    }
    producer_audit = {
        "selected_model": selected,
        "selected_model_role": "diagnostic_challenger",
        "evaluated_candidate_names": list(names),
        "locked_diagnostic_ablation_set": handoff.selection[
            "locked_diagnostic_ablation_set"
        ],
        "opener_blend_sensitivity": dict(opener_payload),
        "probability_mapper_fit_end_max": "2022-12-31",
        "prediction_input": {
            "candidate_names": list(base_names),
            "candidate_roles": {name: roles[name] for name in base_names},
        },
    }
    metadata_rows = []
    by_name = catalog.set_index("name")
    for name in base_names:
        row = by_name.loc[name].to_dict()
        row.update(
            {
                "candidate": name,
                "candidate_role": roles[name],
                "sealed_selection_eligible": False,
                "sealed_result_can_reselect": False,
                "source_candidate": np.nan,
                "sealed_sensitivity_type": np.nan,
            }
        )
        metadata_rows.append(row)
    opener_row = by_name.loc[selected].to_dict()
    opener_row.update(
        {
            "candidate": opener,
            "candidate_role": roles[opener],
            "control": False,
            "sensitivity_only": True,
            "selection_eligible": False,
            "sealed_selection_eligible": False,
            "sealed_result_can_reselect": False,
            "source_candidate": selected,
            "sealed_sensitivity_type": "frozen_opener_market_blend",
        }
    )
    metadata_rows.append(opener_row)
    metadata = pd.DataFrame(metadata_rows)

    prediction_rows = []
    for name in base_names:
        for golfer in ("1", "2"):
            prediction_rows.append(
                {
                    "candidate": name,
                    "event_key": "canon-a",
                    "dg_id_key": golfer,
                    "event_start_date": "2023-05-01",
                    "prediction_asof": "2023-05-01",
                    "training_max_event_end": "2023-04-24",
                    "chronology_ok": True,
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    mapper = pd.DataFrame(
        [
            {
                "candidate": name,
                "market": market,
                "fit_end": "2022-12-31",
                "sigma": 1.0,
                "tie_threshold": 0.01,
                "frozen_before_sealed_scoring": True,
            }
            for name in base_names
            for market in audit.EXPECTED_SEALED_MARKETS
        ]
    )
    logical = pd.DataFrame(
        [
            {
                "candidate": name,
                "candidate_role": roles[name],
                "event_key": "canon-a",
                "event_start_date": pd.Timestamp("2023-05-01"),
                "market": market,
                "pair_key": "1|2",
                "canonical_outcome": 1.0,
                "canonical_probability": 0.6,
                "canonical_push_probability": 0.0,
                "probability_fit_end": pd.Timestamp("2022-12-31"),
            }
            for name in names
            for market in audit.EXPECTED_SEALED_MARKETS
        ]
    )
    return handoff, manifest, producer_audit, metadata, predictions, mapper, logical


def test_sealed_candidate_chronology_bundle_is_invoked_and_exact(
    tmp_path: Path,
) -> None:
    fixture = _sealed_candidate_authority_fixture(tmp_path)
    authoritative, names, roles, chronology = (
        audit.validate_sealed_candidate_chronology_bundle(*fixture)
    )
    assert tuple(authoritative["candidate"]) == names
    assert roles[audit.frozen_handoff.STRICT_PT_CONTROL] == (
        "strict_point_in_time_architecture_control"
    )
    assert chronology["prediction_training_strictly_pre_event"] is True


def test_sealed_chronology_allows_null_training_boundary_only_for_controls(
    tmp_path: Path,
) -> None:
    handoff, manifest, producer_audit, metadata, predictions, mapper, logical = (
        _sealed_candidate_authority_fixture(tmp_path)
    )
    controls = set(
        metadata.loc[metadata["control"].astype(bool), "candidate"].astype(str)
    )
    predictions.loc[predictions["candidate"].isin(controls), "training_max_event_end"] = (
        None
    )
    _, _, _, chronology = audit.validate_sealed_candidate_chronology_bundle(
        handoff,
        manifest,
        producer_audit,
        metadata,
        predictions,
        mapper,
        logical,
    )
    assert chronology["control_rows_without_fitted_training_boundary"] > 0
    assert chronology["direct_controls_may_omit_fitted_training_boundary"] is True

    predictions.loc[
        predictions["candidate"].eq(handoff.selected_name),
        "training_max_event_end",
    ] = None
    with pytest.raises(ValueError, match="strictly pre-event"):
        audit.validate_sealed_candidate_chronology_bundle(
            handoff,
            manifest,
            producer_audit,
            metadata,
            predictions,
            mapper,
            logical,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shrunken_menu", "candidate menu"),
        ("forged_metadata", "differs from frozen catalog"),
        ("relabeled_control", "role is forged"),
        ("future_training", "strictly pre-event"),
        ("future_mapper_label", "fit_end_max"),
    ],
)
def test_sealed_candidate_chronology_bundle_rejects_integrated_tampering(
    tmp_path: Path, mutation: str, message: str
) -> None:
    handoff, manifest, producer_audit, metadata, predictions, mapper, logical = (
        _sealed_candidate_authority_fixture(tmp_path)
    )
    if mutation == "shrunken_menu":
        dropped = "w1_sibling"
        manifest["evaluated_candidate_names"].remove(dropped)
        producer_audit["evaluated_candidate_names"].remove(dropped)
        metadata = metadata.loc[~metadata["candidate"].eq(dropped)].copy()
        logical = logical.loc[~logical["candidate"].eq(dropped)].copy()
    elif mutation == "forged_metadata":
        metadata.loc[metadata["candidate"].eq("w0"), "alpha"] = 99.0
    elif mutation == "relabeled_control":
        metadata.loc[
            metadata["candidate"].eq(audit.frozen_handoff.STRICT_PT_CONTROL),
            "candidate_role",
        ] = "diagnostic_challenger"
    elif mutation == "future_training":
        predictions.loc[0, "training_max_event_end"] = "2023-05-01"
    elif mutation == "future_mapper_label":
        producer_audit["probability_mapper_fit_end_max"] = "2023-01-02"
    with pytest.raises(ValueError, match=message):
        audit.validate_sealed_candidate_chronology_bundle(
            handoff,
            manifest,
            producer_audit,
            metadata,
            predictions,
            mapper,
            logical,
        )


def _sealed_logical_quote_authority_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    selected = "selected"
    candidates = (selected, "other", f"{selected}{audit.FROZEN_OPENER_SUFFIX}")
    quotes = pd.DataFrame(
        [
            {
                "quote_id": f"q-{event}",
                "mapped_event_key": f"canon-{event}",
                "event_start_date": f"2023-05-0{event}",
                "market": "72h",
                "book": "book_a",
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_open": -110.0,
                "p2_open": -110.0,
                "p1_close": -110.0,
                "p2_close": -110.0,
            }
            for event in (1, 2)
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "event_key": f"canon-{event}",
                "dg_id_key": golfer,
                "event_start_date": f"2023-05-0{event}",
            }
            for candidate in (selected, "other")
            for event in (1, 2)
            for golfer in ("1", "2")
        ]
    )
    exclusions = pd.DataFrame(
        columns=[
            "candidate",
            "quote_id",
            "mapped_event_key",
            "event_start_date",
            "market",
            "pair_key",
            "book",
            "p1_id",
            "p2_id",
            "missing_reason",
        ]
    )
    logical = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "event_key": f"canon-{event}",
                "event_start_date": pd.Timestamp(f"2023-05-0{event}"),
                "market": "72h",
                "pair_key": "1|2",
            }
            for candidate in candidates
            for event in (1, 2)
        ]
    )
    return quotes, exclusions, predictions, logical, candidates


def test_sealed_logical_authority_rejects_omission_from_every_arm() -> None:
    quotes, exclusions, predictions, logical, candidates = (
        _sealed_logical_quote_authority_fixture()
    )
    result = audit.validate_sealed_logical_quote_authority(
        quotes,
        exclusions,
        predictions,
        logical,
        expected_candidates=candidates,
        selected_name="selected",
        expected_books=("book_a",),
        expected_markets=("72h",),
    )
    assert result["offered_minus_excluded_exact"] is True
    omitted = logical.loc[~logical["event_key"].eq("canon-2")].copy()
    with pytest.raises(ValueError, match="offered-minus-excluded"):
        audit.validate_sealed_logical_quote_authority(
            quotes,
            exclusions,
            predictions,
            omitted,
            expected_candidates=candidates,
            selected_name="selected",
            expected_books=("book_a",),
            expected_markets=("72h",),
        )


def test_sealed_logical_authority_requires_exact_exclusion_ledger() -> None:
    quotes, _, predictions, logical, candidates = _sealed_logical_quote_authority_fixture()
    predictions = predictions.loc[
        ~(
            predictions["candidate"].eq("other")
            & predictions["event_key"].eq("canon-2")
            & predictions["dg_id_key"].eq("2")
        )
    ].copy()
    logical = logical.loc[
        ~(logical["candidate"].eq("other") & logical["event_key"].eq("canon-2"))
    ].copy()
    exclusion = pd.DataFrame(
        [
            {
                "candidate": "other",
                "quote_id": "q-2",
                "mapped_event_key": "canon-2",
                "event_start_date": "2023-05-02",
                "market": "72h",
                "pair_key": "1|2",
                "book": "book_a",
                "p1_id": "1",
                "p2_id": "2",
                "missing_reason": "p2_absent",
            }
        ]
    )
    audit.validate_sealed_logical_quote_authority(
        quotes,
        exclusion,
        predictions,
        logical,
        expected_candidates=candidates,
        selected_name="selected",
        expected_books=("book_a",),
        expected_markets=("72h",),
    )
    exclusion.loc[0, "missing_reason"] = "p1_absent"
    with pytest.raises(ValueError, match="false 'missing_reason'"):
        audit.validate_sealed_logical_quote_authority(
            quotes,
            exclusion,
            predictions,
            logical,
            expected_candidates=candidates,
            selected_name="selected",
            expected_books=("book_a",),
            expected_markets=("72h",),
        )


def test_sealed_logical_authority_binds_source_prediction_event_dates() -> None:
    quotes, exclusions, predictions, logical, candidates = (
        _sealed_logical_quote_authority_fixture()
    )
    predictions.loc[predictions["event_key"].eq("canon-2"), "event_start_date"] = (
        "2023-05-03"
    )
    with pytest.raises(ValueError, match="event dates differ"):
        audit.validate_sealed_logical_quote_authority(
            quotes,
            exclusions,
            predictions,
            logical,
            expected_candidates=candidates,
            selected_name="selected",
            expected_books=("book_a",),
            expected_markets=("72h",),
        )


def test_cascade_normalized_quote_authority_rejects_all_arm_omission() -> None:
    full = _quoted_cascade_probabilities()
    normalized = _normalized_cascade_quotes(full)
    omitted = full.loc[~full["round_num"].eq(2)].copy()
    probabilities = audit._canonical_cascade_probabilities(omitted)
    pairing = audit.build_w2_pairing_audit(_cascade_predictions(), probabilities)
    with pytest.raises(ValueError, match="normalized executable quote authority"):
        audit.audit_cascade_quote_slices(
            normalized,
            omitted,
            pairing,
            expected_price_points=("close",),
            expected_books=("book_a",),
        )


def test_code_frozen_menu_rejects_consistently_shrunken_producer_payloads() -> None:
    shrunken_audit = ["betonline", "betcris"]
    shrunken_artifact = pd.DataFrame({"book": shrunken_audit})
    assert set(shrunken_artifact["book"]) == set(shrunken_audit)
    with pytest.raises(ValueError, match="code-frozen menu"):
        audit._require_exact_string_menu(
            shrunken_audit, audit.EXPECTED_BOOKS, "synthetic producer books"
        )

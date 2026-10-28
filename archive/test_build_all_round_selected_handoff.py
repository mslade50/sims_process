import json
import os
from pathlib import Path

import pandas as pd
import pytest

from archive.build_all_round_market_prediction_arms import (
    FrozenArmBuild,
    write_apply_artifacts,
)
from archive.build_all_round_selected_handoff import (
    DEFAULT_CONTRACT_PATH,
    EXPECTED_CONTRACT_SHA256,
    build_selected_handoff,
    canonical_payload_hash,
    seal_prediction_panels,
    sha256_file,
    training_catalog_hash,
    validate_prediction_authority,
)
from archive.run_all_round_base_skill_rebuild import validate_selected_spec_sidecar


SELECTED = "allr__t2__w0__t_ball__ridge__a1000"
OTHER = "allr__t2__w0__t__ridge__a1000"
BASELINE = "baseline_current_prod"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_self_hashed(path: Path, payload: dict, field: str) -> None:
    payload = dict(payload)
    payload[field] = canonical_payload_hash(payload)
    _write_json(path, payload)
    path.with_suffix(".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def _prediction_frames(years=(2023,), candidates=(SELECTED, OTHER, BASELINE)):
    base_rows = []
    round_rows = []
    for year in years:
        event_key = f"pga|{year}|1|{year}-07-06"
        for player in (101, 202):
            for candidate_index, candidate in enumerate(candidates):
                prediction = 0.1 * candidate_index + 0.01 * (player - 100)
                base_rows.append(
                    {
                        "candidate": candidate,
                        "event_key": event_key,
                        "dg_id_key": str(player),
                        "prediction": prediction,
                    }
                )
                for round_num in (1, 2, 3, 4):
                    round_rows.append(
                        {
                            "candidate": candidate,
                            "event_key": event_key,
                            "dg_id_key": str(player),
                            "round_num": round_num,
                            "prediction": prediction + 0.02 * round_num,
                        }
                    )
    return pd.DataFrame(base_rows), pd.DataFrame(round_rows)


def _selection_fixture(
    root: Path,
    *,
    fallback: bool = False,
    frozen_downstream: float = 0.5,
    baseline_warmup: bool = False,
) -> dict[str, Path]:
    train_dir = root / "train"
    arm_dir = root / "arms"
    freeze_dir = root / "freeze"
    handoff_dir = root / "handoff"
    for path in (train_dir, arm_dir, freeze_dir):
        path.mkdir(parents=True)

    catalog_path = train_dir / "candidate_catalog.csv"
    catalog = pd.DataFrame(
        [
            {
                "candidate": SELECTED,
                "target_arm": "T2",
                "weather_arm": "W0",
                "feature_family": "T_BALL",
                "estimator": "ridge",
                "alpha": 1000.0,
                "selectable": True,
                "diagnostic_only": False,
                "run_status": "ok",
                "smoke_run": False,
            },
            {
                "candidate": OTHER,
                "target_arm": "T2",
                "weather_arm": "W0",
                "feature_family": "T",
                "estimator": "ridge",
                "alpha": 1000.0,
                "selectable": True,
                "diagnostic_only": False,
                "run_status": "ok",
                "smoke_run": False,
            },
        ]
    )
    catalog.to_csv(catalog_path, index=False, lineterminator="\n")
    canonical_catalog = training_catalog_hash(pd.read_csv(catalog_path))
    train_manifest_path = train_dir / "train_manifest.json"
    _write_json(
        train_manifest_path,
        {
            "research_only": True,
            "production_files_written": False,
            "phase": "train",
            "smoke": False,
            "smoke_outputs_selection_eligible": True,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "candidate_catalog_sha256": canonical_catalog,
            "evaluation_outcomes_read": False,
        },
    )

    authority_path = arm_dir / "market_candidate_authority.csv"
    authority = pd.DataFrame(
        [
            {
                "candidate": BASELINE,
                "course_fit_lambda": 1.0,
                "live_update_lambda": 1.0,
                "authority_role": "exact_cached_control",
            },
            {
                "candidate": SELECTED,
                "course_fit_lambda": 0.5,
                "live_update_lambda": 1.0,
                "authority_role": "challenger_2023_tuned",
            },
            {
                "candidate": OTHER,
                "course_fit_lambda": 0.0,
                "live_update_lambda": 0.5,
                "authority_role": "challenger_2023_tuned",
            },
        ]
    )
    authority.to_csv(authority_path, index=False, lineterminator="\n")
    authority_hash = sha256_file(authority_path)
    base, rounds = _prediction_frames()
    if baseline_warmup:
        warm_base, warm_round = _prediction_frames(
            years=(2023,), candidates=(BASELINE,)
        )
        warm_base["event_key"] = "pga|2023|99|2023-02-02"
        warm_round["event_key"] = "pga|2023|99|2023-02-02"
        base = pd.concat([warm_base, base], ignore_index=True)
        rounds = pd.concat([warm_round, rounds], ignore_index=True)
    base_path = arm_dir / "selection_base_predictions.parquet"
    round_path = arm_dir / "selection_round_predictions.parquet"
    base.to_parquet(base_path, index=False)
    rounds.to_parquet(round_path, index=False)
    common = {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "market_candidate_authority_path": authority_path.name,
        "market_candidate_authority_sha256": authority_hash,
        "selection_base_predictions_path": base_path.name,
        "selection_base_predictions_sha256": sha256_file(base_path),
        "selection_base_predictions_rows": len(base),
        "selection_round_predictions_path": round_path.name,
        "selection_round_predictions_sha256": sha256_file(round_path),
        "selection_round_predictions_rows": len(rounds),
    }
    frozen_path = arm_dir / "frozen_lambdas_2023.json"
    frozen = {
        **common,
        "selection_year": 2023,
        "lambda_grid": [0.0, 0.5, 1.0],
        "lambdas": [
            {
                "candidate": SELECTED,
                "downstream_lambda": frozen_downstream,
                "live_lambda": 1.0,
                "selection_year": 2023,
            },
            {
                "candidate": OTHER,
                "downstream_lambda": 0.0,
                "live_lambda": 0.5,
                "selection_year": 2023,
            },
        ],
    }
    _write_json(frozen_path, frozen)
    arm_manifest_path = arm_dir / "selection_arm_manifest.json"
    _write_json(
        arm_manifest_path,
        {
            **common,
            "schema_version": 1,
            "research_only": True,
            "production_files_written": False,
            "frozen_lambdas_path": frozen_path.name,
            "frozen_lambdas_sha256": sha256_file(frozen_path),
            "candidate_closure": sorted([SELECTED, OTHER, BASELINE]),
        },
    )
    arm_audit_path = arm_dir / "selection_arm_audit.json"
    _write_json(
        arm_audit_path,
        {
            **common,
            "phase": "2023_chronological_oof_selection",
            "selection_year": 2023,
            "evaluation_target_columns_read": [],
        },
    )

    freeze_path = freeze_dir / "selection_freeze.json"
    selected_spec = {
        "candidate": SELECTED,
        "course_fit_lambda": 0.5,
        "live_update_lambda": 1.0,
    }
    frozen_spec = {
        **selected_spec,
        "baseline": BASELINE,
        "minimum_ev": 0.03,
    }
    freeze = {
        "schema_version": 2,
        "research_only": True,
        "contract_path": str(DEFAULT_CONTRACT_PATH.resolve()),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "candidate_catalog_path": str(authority_path.resolve()),
        "candidate_catalog_sha256": authority_hash,
        "baseline": BASELINE,
        "selected_candidate": SELECTED,
        "selected_spec": selected_spec,
        "frozen_spec": frozen_spec,
        "frozen_spec_sha256": canonical_payload_hash(frozen_spec),
    }
    if fallback:
        freeze.update(
            {
                "promotion_eligible": False,
                "selection_mode": "diagnostic_fallback",
                "diagnostic_fallback": True,
                "primary_selection": {
                    "selection_mode": "diagnostic_fallback",
                    "promotion_eligible": False,
                    "diagnostic_fallback": True,
                    "reason": "no challenger survived primary coverage",
                },
            }
        )
    freeze["freeze_payload_sha256"] = canonical_payload_hash(freeze)
    _write_json(freeze_path, freeze)
    freeze_path.with_suffix(".sha256").write_text(
        f"{sha256_file(freeze_path)}  {freeze_path.name}\n", encoding="utf-8"
    )
    return {
        "catalog": catalog_path,
        "train_manifest": train_manifest_path,
        "authority": authority_path,
        "frozen": frozen_path,
        "arm_manifest": arm_manifest_path,
        "arm_audit": arm_audit_path,
        "base": base_path,
        "round": round_path,
        "freeze": freeze_path,
        "handoff_dir": handoff_dir,
    }


def _build(paths: dict[str, Path]):
    return build_selected_handoff(
        selection_freeze_path=paths["freeze"],
        training_candidate_catalog_path=paths["catalog"],
        training_manifest_path=paths["train_manifest"],
        frozen_lambdas_path=paths["frozen"],
        market_candidate_authority_path=paths["authority"],
        selection_arm_manifest_path=paths["arm_manifest"],
        selection_arm_audit_path=paths["arm_audit"],
        selection_base_predictions_path=paths["base"],
        selection_round_predictions_path=paths["round"],
        output_dir=paths["handoff_dir"],
    )


def test_builds_runner_spec_and_immutable_handoff(tmp_path):
    paths = _selection_fixture(tmp_path)
    manifest = _build(paths)
    spec_path = paths["handoff_dir"] / "runner_selected_spec.json"
    spec = json.loads(spec_path.read_text())
    assert spec["candidate"] == SELECTED
    assert spec["selected_arm"] == SELECTED
    assert spec["downstream_lambda"] == 0.5
    assert spec["live_update_lambda"] == 1.0
    assert spec["promotion_eligible"] is True
    assert manifest["evaluation_year_outcome_quote_or_label_files_read"] == []
    assert (paths["handoff_dir"] / "selected_handoff_manifest.sha256").is_file()
    assert validate_selected_spec_sidecar(spec_path) == sha256_file(spec_path)
    # Identical inputs are idempotent; immutable outputs cannot silently change.
    assert _build(paths)["manifest_payload_sha256"] == manifest["manifest_payload_sha256"]


def test_diagnostic_fallback_cannot_be_upgraded(tmp_path):
    paths = _selection_fixture(tmp_path, fallback=True)
    manifest = _build(paths)
    spec = json.loads(
        (paths["handoff_dir"] / "runner_selected_spec.json").read_text()
    )
    assert manifest["promotion_eligible"] is False
    assert spec["promotion_eligible"] is False
    assert spec["diagnostic_fallback"] is True
    assert spec["primary_selection"]["reason"] == "no challenger survived primary coverage"


def test_rejects_freeze_tamper_and_lambda_mismatch(tmp_path):
    tamper = _selection_fixture(tmp_path / "tamper")
    tamper["freeze"].write_text(tamper["freeze"].read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        _build(tamper)

    mismatch = _selection_fixture(tmp_path / "mismatch", frozen_downstream=1.0)
    with pytest.raises(ValueError, match="course-fit lambda differs"):
        _build(mismatch)


def _runner_and_adapter_artifacts(paths: dict[str, Path], root: Path) -> dict[str, Path]:
    runner_dir = root / "runner_eval"
    adapter_dir = root / "adapter_eval"
    runner_dir.mkdir()
    spec_path = paths["handoff_dir"] / "runner_selected_spec.json"
    raw_rows = []
    for year in (2024, 2025, 2026):
        for player in (101, 202):
            raw_rows.append(
                {
                    "candidate": SELECTED,
                    "event_key": f"pga|{year}|1|{year}-07-06",
                    "dg_id_key": str(player),
                    "prediction": 0.2,
                    "prediction_stage": "raw_base_core_before_downstream_and_live",
                    "downstream_lambda_applied": False,
                    "live_update_lambda_applied": False,
                }
            )
    raw = pd.DataFrame(raw_rows)
    raw_path = runner_dir / "winner_raw_core_predictions_2024_2026.parquet"
    raw.to_parquet(raw_path, index=False)
    composed_path = runner_dir / "winner_composed_pre_event_predictions_2024_2026.parquet"
    raw.to_parquet(composed_path, index=False)
    composed_round_path = runner_dir / "winner_composed_round_predictions_2024_2026.parquet"
    raw.assign(round_num=1).to_parquet(composed_round_path, index=False)
    context_path = runner_dir / "market_context_2024_2026.parquet"
    raw[["event_key", "dg_id_key"]].to_parquet(context_path, index=False)
    runner_manifest_path = runner_dir / "evaluation_prediction_manifest.json"
    outputs = {}
    for name, path, rows in (
        ("raw_core_predictions", raw_path, len(raw)),
        ("composed_pre_event_predictions", composed_path, len(raw)),
        ("composed_round_predictions", composed_round_path, len(raw)),
        ("market_context", context_path, len(raw)),
    ):
        outputs[name] = {"path": path.name, "sha256": sha256_file(path), "rows": rows}
    _write_self_hashed(
        runner_manifest_path,
        {
            "research_only": True,
            "production_files_written": False,
            "phase": "evaluate_prediction_only",
            "selected_candidate": SELECTED,
            "selected_arm": SELECTED,
            "downstream_lambda": 0.5,
            "live_update_lambda": 1.0,
            "promotion_eligible": True,
            "diagnostic_fallback": False,
            "selection_mode": "primary_contract_selection",
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "candidate_catalog_sha256": json.loads(
                spec_path.read_text()
            )["catalog_sha256"],
            # Runner manifests preserve the CLI spelling, which is commonly
            # repo-root-relative even though the handoff stores an absolute path.
            "selected_spec_path": os.path.relpath(spec_path.resolve(), Path.cwd()),
            "selected_spec_sha256": sha256_file(spec_path),
            "prediction_years": [2024, 2025, 2026],
            "post_2023_target_columns_projected": [],
            "post_2023_outcomes_read": False,
            "outputs": outputs,
        },
        "manifest_payload_sha256",
    )

    eval_base, eval_round = _prediction_frames(
        years=(2024, 2025, 2026), candidates=(SELECTED, BASELINE)
    )
    result = FrozenArmBuild(
        base_predictions=eval_base,
        round_predictions=eval_round,
        audit={
            "phase": "2024_2026_frozen_label_blind_apply",
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "selected_candidate": SELECTED,
            "control_candidate": BASELINE,
            "downstream_lambda": 0.5,
            "live_lambda": 1.0,
            "state_columns_read": ["event_key", "dg_id_key", "cascade_prediction_r1"],
            "input_prediction_stage": "raw_base_core_before_downstream_and_live",
            "input_downstream_lambda_applied": False,
            "input_live_update_lambda_applied": False,
            "target_columns_read": [],
            "base_rows": len(eval_base),
            "round_rows": len(eval_round),
        },
    )
    write_apply_artifacts(
        result,
        adapter_dir,
        source_paths={
            "runner_raw_core_predictions": raw_path,
            "frozen_lambdas": paths["frozen"],
        },
    )
    return {
        "spec": spec_path,
        "runner_manifest": runner_manifest_path,
        "adapter_manifest": adapter_dir / "evaluation_arm_manifest.json",
        "adapter_audit": adapter_dir / "evaluation_arm_audit.json",
    }


def test_seal_predictions_is_label_blind_and_tamper_evident(tmp_path, monkeypatch):
    # The exact control has Jan-Mar warmup rows that must not defeat Apr-Dec
    # selected/control closure in the sealed evaluator panel.
    paths = _selection_fixture(tmp_path, baseline_warmup=True)
    _build(paths)
    eval_paths = _runner_and_adapter_artifacts(paths, tmp_path)
    calls = []
    real_read = pd.read_parquet

    def read_spy(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return real_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_spy)
    sealed_dir = tmp_path / "sealed"
    authority = seal_prediction_panels(
        selected_handoff_manifest_path=paths["handoff_dir"]
        / "selected_handoff_manifest.json",
        runner_selected_spec_path=eval_paths["spec"],
        runner_evaluation_manifest_path=eval_paths["runner_manifest"],
        adapter_apply_manifest_path=eval_paths["adapter_manifest"],
        adapter_apply_audit_path=eval_paths["adapter_audit"],
        selection_base_predictions_path=paths["base"],
        selection_round_predictions_path=paths["round"],
        output_dir=sealed_dir,
    )
    assert authority["candidate_closure"] == sorted([SELECTED, BASELINE])
    assert authority["lambda_application"]["adapter_downstream_application_count"] == 1
    assert authority["selection_mapper_window_audit"]["base_warmup_rows_dropped"] > 0
    assert authority["selection_mapper_window_audit"]["round_warmup_rows_dropped"] > 0
    assert all(columns is not None for columns in calls)
    assert not any(
        any(token in column.casefold() for token in ("label", "outcome", "quote"))
        for columns in calls
        for column in columns
    )
    authority_path = sealed_dir / "prediction_authority.json"
    validate_prediction_authority(
        authority_path,
        combined_base_predictions_path=sealed_dir / "combined_base_predictions.parquet",
        combined_round_predictions_path=sealed_dir / "combined_round_predictions.parquet",
    )
    # A post-seal mutation cannot reach the evaluator under the authority.
    combined = pd.read_parquet(sealed_dir / "combined_base_predictions.parquet")
    combined.loc[0, "prediction"] += 0.25
    combined.to_parquet(sealed_dir / "combined_base_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="prediction output"):
        validate_prediction_authority(authority_path)

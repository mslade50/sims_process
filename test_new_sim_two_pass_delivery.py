"""Focused regressions for new_sim's two-pass delivery boundary."""

import ast
from datetime import datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from category_distribution_guard import require_complete_category_distributions
from portable_hash import (
    LF_NORMALIZED_HASH_MODE,
    lf_normalized_sha256,
    lf_normalized_size,
)


SOURCE_PATH = Path(__file__).with_name("new_sim.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(SOURCE_PATH))


def _load_definitions(*names):
    selected = [
        node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    namespace = {
        "datetime": datetime,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "pd": pd,
        "timezone": timezone,
        "LF_NORMALIZED_HASH_MODE": LF_NORMALIZED_HASH_MODE,
        "lf_normalized_sha256": lf_normalized_sha256,
        "lf_normalized_size": lf_normalized_size,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_calibration_uses_only_pre_course_fit_even_when_final_exists():
    select = _load_definitions("_select_prediction_input")["_select_prediction_input"]
    existing = {"final.csv", "pre_sim_summary.csv", "pre_course_fit.csv"}

    selected, run_pass = select(
        "final.csv",
        "pre_course_fit.csv",
        path_exists=existing.__contains__,
    )

    assert (selected, run_pass) == ("pre_course_fit.csv", "calibration")


def test_calibration_and_final_inputs_are_required_not_downgraded():
    select = _load_definitions("_select_prediction_input")["_select_prediction_input"]

    with pytest.raises(FileNotFoundError, match="Calibration pass requires"):
        select("final.csv", "pre_course_fit.csv", path_exists=lambda _path: False)

    with pytest.raises(FileNotFoundError, match="Final pass requires"):
        select(
            "final.csv",
            "pre_course_fit.csv",
            requested_pass="final",
            path_exists=lambda _path: False,
        )


def test_variance_inputs_cannot_silently_disappear(tmp_path):
    load_inputs = _load_definitions("_load_required_variance_inputs")[
        "_load_required_variance_inputs"
    ]

    class MissingLatent:
        pass

    config_path = tmp_path / "shot_dispersion_config.json"
    config_path.write_text('{"enabled": false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="must define WEEK_LATENT_SD"):
        load_inputs(MissingLatent(), str(config_path))

    class Inputs:
        WEEK_LATENT_SD = 0.0

    with pytest.raises(RuntimeError, match="required and must be readable"):
        load_inputs(Inputs(), str(tmp_path / "missing.json"))

    config_path.write_text('{"weights": {}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicit boolean enabled"):
        load_inputs(Inputs(), str(config_path))

    config_path.write_text('{"enabled": false}', encoding="utf-8")
    latent, config = load_inputs(Inputs(), str(config_path))
    assert latent == 0.0
    assert config == {"enabled": False}

    config_path.write_text(
        '{"enabled": true, "required_current_event": true}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Required weekly.*lacks"):
        load_inputs(Inputs(), str(config_path))


def _pretournament_category_dists(players):
    return pd.DataFrame([
        {
            "player_name": player,
            "category_clean": category,
            "mean": 0.1,
            "std": 1.0,
            "skew": 0.0,
            "n_eff": 25.0,
            "n": 50,
        }
        for player in players
        for category in ("sg_ott", "sg_app", "sg_arg", "sg_putt")
    ])


def test_pretournament_category_dists_require_complete_finite_active_field():
    with pytest.raises(ValueError, match="missing active-field category coverage"):
        require_complete_category_distributions(
            _pretournament_category_dists(["unrelated"]),
            ["alpha"],
            ["sg_ott", "sg_app", "sg_arg", "sg_putt"],
            source_label="this_week_dists_v2.csv",
            extra_numeric_columns=("n",),
        )

    invalid = _pretournament_category_dists(["alpha"])
    invalid.loc[invalid["category_clean"] == "sg_putt", "std"] = np.nan
    with pytest.raises(ValueError, match="non-finite active-field values"):
        require_complete_category_distributions(
            invalid,
            ["alpha"],
            ["sg_ott", "sg_app", "sg_arg", "sg_putt"],
            source_label="this_week_dists_v2.csv",
            extra_numeric_columns=("n",),
        )

    clean, active = require_complete_category_distributions(
        _pretournament_category_dists(["Alpha"]),
        ["ALPHA"],
        ["sg_ott", "sg_app", "sg_arg", "sg_putt"],
        source_label="this_week_dists_v2.csv",
        extra_numeric_columns=("n",),
    )
    assert active == ["alpha"]
    assert len(clean[clean["player_name"] == "alpha"]) == 4

    calls = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_complete_category_distributions"
    ]
    assert len(calls) == 1


def _tee_time_functions():
    namespace = _load_definitions("parse_time", "_load_required_tee_times")
    return namespace["_load_required_tee_times"]


def test_monday_tee_times_are_optional_on_the_eastern_calendar():
    namespace = _load_definitions(
        "_monday_tee_times_optional",
        "_tee_time_fallback_allowed",
    )
    monday_optional = namespace["_monday_tee_times_optional"]
    fallback_allowed = namespace["_tee_time_fallback_allowed"]

    monday_utc = datetime(2026, 8, 25, 3, 59, tzinfo=timezone.utc)
    tuesday_utc = datetime(2026, 8, 25, 4, 0, tzinfo=timezone.utc)
    assert monday_optional(monday_utc) is True
    assert monday_optional(tuesday_utc) is False
    assert fallback_allowed("final", now=monday_utc) is True
    assert fallback_allowed("final", now=tuesday_utc) is False
    assert fallback_allowed(
        "calibration", cli_override=True, now=tuesday_utc
    ) is True
    with pytest.raises(ValueError, match="Unknown tournament-sim pass"):
        fallback_allowed("auto", now=monday_utc)

    effective_policy = next(
        node
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_ALLOW_MISSING_TEE_TIMES"
            for target in node.targets
        )
    )
    policy_source = ast.unparse(effective_policy.value)
    assert "args.allow_missing_tee_times" in policy_source
    assert "_tee_time_fallback_allowed" in policy_source

    tee_time_call = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_required_tee_times"
    )
    allow_keyword = next(
        keyword for keyword in tee_time_call.keywords
        if keyword.arg == "allow_missing"
    )
    assert ast.unparse(allow_keyword.value) == "_ALLOW_MISSING_TEE_TIMES"


def test_fresh_r1_and_r2_tee_times_replace_retained_columns():
    load_tee_times = _tee_time_functions()
    predictions = pd.DataFrame({
        "player_name": ["alpha", "beta"],
        "r1_teetime": ["2000-01-01 06:00", "2000-01-01 06:00"],
        "r2_teetime": ["2000-01-02 06:00", "2000-01-02 06:00"],
    })
    payloads = {
        "r1_teetime": pd.DataFrame({
            "player_name": ["alpha", "beta"],
            "r1_teetime": ["2026-08-20 07:10", "2026-08-20 12:20"],
        }),
        "r2_teetime": pd.DataFrame({
            "player_name": ["alpha", "beta"],
            "r2_teetime": ["2026-08-21 12:20", "2026-08-21 07:10"],
        }),
    }

    def fetcher(_key, *, teetime_col, fill_missing_teetimes):
        assert fill_missing_teetimes is False
        return payloads[teetime_col]

    result, _fresh = load_tee_times(
        predictions,
        fetcher=fetcher,
        api_key="key",
        name_map={},
    )
    assert result["r1_teetime"].tolist() == [
        "2026-08-20 07:10",
        "2026-08-20 12:20",
    ]
    assert result["r2_teetime"].tolist() == [
        "2026-08-21 12:20",
        "2026-08-21 07:10",
    ]


def test_missing_or_low_coverage_tee_times_fail_closed():
    load_tee_times = _tee_time_functions()
    predictions = pd.DataFrame({"player_name": ["alpha", "beta", "gamma"]})
    full_r1 = pd.DataFrame({
        "player_name": ["alpha", "beta", "gamma"],
        "r1_teetime": [
            "2026-08-20 07:10",
            "2026-08-20 08:10",
            "2026-08-20 09:10",
        ],
    })

    def missing_r2(_key, *, teetime_col, fill_missing_teetimes):
        return full_r1 if teetime_col == "r1_teetime" else None

    with pytest.raises(RuntimeError, match="r2_teetime payload is missing"):
        load_tee_times(
            predictions,
            fetcher=missing_r2,
            api_key="key",
            name_map={},
        )

    def partial(_key, *, teetime_col, fill_missing_teetimes):
        return pd.DataFrame({
            "player_name": ["alpha", "beta"],
            teetime_col: ["2026-08-20 07:10", "2026-08-20 08:10"],
        })

    with pytest.raises(RuntimeError, match="coverage is 2/3"):
        load_tee_times(
            predictions,
            fetcher=partial,
            api_key="key",
            name_map={},
        )


def test_missing_tee_time_escape_is_calibration_only_and_clears_stale_values():
    load_tee_times = _tee_time_functions()
    predictions = pd.DataFrame({
        "player_name": ["alpha", "beta"],
        "r1_teetime": ["stale", "stale"],
        "r2_teetime": ["stale", "stale"],
    })
    result, _fresh = load_tee_times(
        predictions,
        fetcher=lambda *_args, **_kwargs: None,
        api_key="key",
        name_map={},
        allow_missing=True,
    )
    assert result["r1_teetime"].isna().all()
    assert result["r2_teetime"].isna().all()

    def partial(_key, *, teetime_col, fill_missing_teetimes):
        return pd.DataFrame({
            "player_name": ["alpha"],
            teetime_col: ["2026-08-20 07:10"],
        })

    partial_result, _fresh = load_tee_times(
        predictions,
        fetcher=partial,
        api_key="key",
        name_map={},
        allow_missing=True,
    )
    assert partial_result["r1_teetime"].isna().all()
    assert partial_result["r2_teetime"].isna().all()

    guard = next(
        node
        for node in TREE.body
        if isinstance(node, ast.If)
        and "args.allow_missing_tee_times" in ast.unparse(node.test)
    )
    assert "RUN_PASS != 'calibration'" in ast.unparse(guard.test)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "error"
        for node in ast.walk(guard)
    )


def test_missing_preferred_correlation_cannot_use_fallback(tmp_path):
    namespace = _load_definitions(
        "_sha256_file",
        "_read_stable_csv",
        "_file_contract",
        "load_corr_matrix",
    )
    preferred = tmp_path / "preferred.csv"
    fallback = tmp_path / "fallback.csv"
    fallback.write_text(
        ",sg_ott,sg_app\nsg_ott,1,0\nsg_app,0,1\n",
        encoding="utf-8",
    )
    namespace["CORR_PREFS"] = [str(preferred), str(fallback)]

    with pytest.raises(FileNotFoundError, match="refusing to substitute"):
        namespace["load_corr_matrix"](["sg_ott", "sg_app"])


def test_only_final_pass_enables_external_delivery():
    enabled = _load_definitions("_delivery_enabled_for_pass")[
        "_delivery_enabled_for_pass"
    ]

    assert enabled("calibration") is False
    assert enabled("final") is True
    with pytest.raises(ValueError, match="Unknown tournament-sim pass"):
        enabled("auto")


def test_cached_pricing_modes_require_explicit_final_pass():
    guard = next(
        node
        for node in TREE.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.price_only and RUN_PASS != 'final'"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_parser"
        and node.func.attr == "error"
        for node in ast.walk(guard)
    )


def test_primary_rust_draw_failure_is_fatal_without_explicit_python_mode():
    rust_try = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "run_pretournament"
            for child in ast.walk(node)
        )
    )
    assert rust_try.handlers
    for handler in rust_try.handlers:
        assert any(isinstance(child, ast.Raise) for child in ast.walk(handler))
        assert not any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "final_scores"
                for target in child.targets
            )
            for child in ast.walk(handler)
        )

    python_draw_assignment = next(
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_python_drew"
            for target in node.targets
        )
    )
    assert "args.use_python" in ast.unparse(python_draw_assignment.value)


def _cache_functions():
    namespace = _load_definitions(
        "_sha256_file",
        "_sim_cache_identity",
        "_write_sim_cache_manifest",
        "_validate_sim_cache_manifest",
    )
    return (
        namespace["_sim_cache_identity"],
        namespace["_write_sim_cache_manifest"],
        namespace["_validate_sim_cache_manifest"],
    )


CACHE_TIME = datetime(2026, 8, 20, 16, 0, tzinfo=timezone.utc)


def _prediction_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cache_identity(identity_for, predictions, **overrides):
    values = {
        "tourney_name": "event",
        "event_id": "28",
        "course_id": "101",
        "run_pass": "final",
        "prediction_path": str(predictions),
        "prediction_sha256": _prediction_sha256(predictions),
        "input_contract_sha256": "contract-a",
        "player_names": ["alpha", "beta"],
        "simulations": 100,
        "contract_time": CACHE_TIME,
    }
    values.update(overrides)
    return identity_for(**values)


def _cache_artifacts(tmp_path):
    paths = {}
    for name in (
        "final_scores",
        "player_names",
        "rank_probs",
        "top_finish_probs",
        "finish_equity",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"sealed-{name}".encode())
        paths[name] = str(path)
    paths["made_cut"] = None
    return paths


def test_calibration_cache_cannot_authorize_final_cached_pricing(tmp_path):
    identity_for, write_manifest, validate_manifest = _cache_functions()
    raw_predictions = tmp_path / "pre_course_fit.csv"
    final_predictions = tmp_path / "final_predictions.csv"
    raw_predictions.write_text("player_name,pred\na,1\n", encoding="utf-8")
    final_predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    artifacts = _cache_artifacts(tmp_path)
    manifest_path = tmp_path / "sim_cache_manifest.json"

    calibration_identity = _cache_identity(
        identity_for,
        raw_predictions,
        run_pass="calibration",
    )
    write_manifest(
        str(manifest_path),
        calibration_identity,
        artifacts,
        generated_at=CACHE_TIME,
    )

    final_identity = _cache_identity(
        identity_for,
        final_predictions,
    )
    with pytest.raises(RuntimeError, match="provenance does not match"):
        validate_manifest(
            str(manifest_path), final_identity, artifacts, now=CACHE_TIME
        )


def test_final_cache_requires_exact_input_field_and_artifact_hashes(tmp_path):
    identity_for, write_manifest, validate_manifest = _cache_functions()
    predictions = tmp_path / "final_predictions.csv"
    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    artifacts = _cache_artifacts(tmp_path)
    manifest_path = tmp_path / "sim_cache_manifest.json"

    identity = _cache_identity(
        identity_for,
        predictions,
    )
    write_manifest(
        str(manifest_path), identity, artifacts, generated_at=CACHE_TIME
    )
    assert validate_manifest(
        str(manifest_path), identity, artifacts, now=CACHE_TIME
    )["run_pass"] == "final"

    changed_field = _cache_identity(
        identity_for,
        predictions,
        player_names=["alpha", "gamma"],
    )
    with pytest.raises(RuntimeError, match="field_sha256"):
        validate_manifest(
            str(manifest_path), changed_field, artifacts, now=CACHE_TIME
        )

    Path(artifacts["final_scores"]).write_bytes(b"tampered scores")
    with pytest.raises(RuntimeError, match="failed hash validation"):
        validate_manifest(
            str(manifest_path), identity, artifacts, now=CACHE_TIME
        )


def test_prediction_reader_rejects_a_file_changed_during_read():
    namespace = _load_definitions("_sha256_file", "_read_stable_csv")
    read_stable = namespace["_read_stable_csv"]
    hashes = iter(["before", "after"])

    with pytest.raises(RuntimeError, match="changed while it was being read"):
        read_stable(
            "final_predictions.csv",
            read_csv=lambda _path: object(),
            hasher=lambda _path: next(hashes),
        )


def test_prediction_hash_is_captured_by_the_top_level_stable_read():
    assignment = next(
        node
        for node in TREE.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Tuple)
            and any(
                isinstance(item, ast.Name) and item.id == "_PREDICTION_SHA256"
                for item in target.elts
            )
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "_read_stable_csv"

    identity_node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sim_cache_identity"
    )
    assert "_sha256_file" not in {
        child.func.id
        for child in ast.walk(identity_node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


def test_canonical_input_contract_changes_for_model_weather_flags_and_files():
    namespace = _load_definitions(
        "_contract_json_value", "_canonical_contract_sha256"
    )
    contract_hash = namespace["_canonical_contract_sha256"]
    baseline = {
        "model": {"std_dev": 2.7, "cut_line": 65},
        "weather": {"wind": [4.0, 6.0], "dew": [60.0, 61.0]},
        "flags": {"no_week_latent": False, "use_python": False},
        "files": {"dists": {"sha256": "abc"}, "correlation": "def"},
    }
    baseline_hash = contract_hash(baseline)
    reordered = {
        "files": baseline["files"],
        "flags": baseline["flags"],
        "weather": baseline["weather"],
        "model": baseline["model"],
    }
    assert contract_hash(reordered) == baseline_hash

    mutations = [
        ("model", "std_dev", 2.8),
        ("weather", "wind", [4.0, 7.0]),
        ("flags", "no_week_latent", True),
        ("files", "dists", {"sha256": "changed"}),
    ]
    for family, key, value in mutations:
        changed = json.loads(json.dumps(baseline))
        changed[family][key] = value
        assert contract_hash(changed) != baseline_hash


def test_portable_text_file_contract_ignores_checkout_newlines(tmp_path):
    namespace = _load_definitions("_sha256_file", "_file_contract")
    file_contract = namespace["_file_contract"]
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(b"player,value\nalpha,1\n")
    crlf_path.write_bytes(b"player,value\r\nalpha,1\r\n")

    lf_contract = file_contract(lf_path, portable_text=True)
    crlf_contract = file_contract(crlf_path, portable_text=True)

    assert lf_contract["sha256"] == crlf_contract["sha256"]
    assert lf_contract["size"] == crlf_contract["size"]
    assert lf_contract["hash_mode"] == LF_NORMALIZED_HASH_MODE


def test_current_input_contract_names_every_requested_cache_dependency():
    contract_node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_current_sim_input_contract"
    )
    source = ast.unparse(contract_node)
    for required_name in (
        "STD_DEV",
        "CUT_LINE",
        "USE_10_SHOT_RULE",
        "wind_1",
        "wind_2",
        "dewpoint_1",
        "dewpoint_2",
        "COURSE_CAT_MULTS",
        "COURSE_CAT_SKEW",
        "WEEK_LATENT_SD",
        "args.use_python",
        "args.no_week_latent",
        "args.no_skew_cal",
        "DISTS_FILE",
        "_CORR_FILE_CONTRACT",
        "_simulation_source_contract",
        "model_ready_inputs",
    ):
        assert required_name in source

    source_contract_node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_simulation_source_contract"
    )
    source_contract = ast.unparse(source_contract_node)
    assert "category_distribution_guard.py" in source_contract
    assert "portable_hash.py" in source_contract
    assert "required=configured_for_current_event" in source_contract
    assert "portable_text=True" in source_contract


def test_cache_rejects_changed_input_contract(tmp_path):
    identity_for, write_manifest, validate_manifest = _cache_functions()
    predictions = tmp_path / "final_predictions.csv"
    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    artifacts = _cache_artifacts(tmp_path)
    manifest_path = tmp_path / "sim_cache_manifest.json"
    identity = _cache_identity(identity_for, predictions)
    write_manifest(
        str(manifest_path), identity, artifacts, generated_at=CACHE_TIME
    )

    changed = _cache_identity(
        identity_for,
        predictions,
        input_contract_sha256="contract-b",
    )
    with pytest.raises(RuntimeError, match="input_contract_sha256"):
        validate_manifest(
            str(manifest_path), changed, artifacts, now=CACHE_TIME
        )


def test_cache_expires_after_seven_days_and_across_iso_weeks(tmp_path):
    identity_for, write_manifest, validate_manifest = _cache_functions()
    predictions = tmp_path / "final_predictions.csv"
    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    artifacts = _cache_artifacts(tmp_path)
    manifest_path = tmp_path / "sim_cache_manifest.json"
    identity = _cache_identity(identity_for, predictions)
    payload = write_manifest(
        str(manifest_path), identity, artifacts, generated_at=CACHE_TIME
    )
    assert payload["generated_at_utc"] == "2026-08-20T16:00:00Z"

    assert validate_manifest(
        str(manifest_path),
        identity,
        artifacts,
        now=CACHE_TIME + timedelta(days=6),
    )["cache_iso_week"] == CACHE_TIME.isocalendar().week
    with pytest.raises(RuntimeError, match="older than 7 days"):
        validate_manifest(
            str(manifest_path),
            identity,
            artifacts,
            now=CACHE_TIME + timedelta(days=8),
        )

    next_week = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
    next_week_identity = _cache_identity(
        identity_for, predictions, contract_time=next_week
    )
    with pytest.raises(RuntimeError, match="cache_iso_week"):
        validate_manifest(
            str(manifest_path),
            next_week_identity,
            artifacts,
            now=next_week,
        )


def test_unsealed_stale_made_cut_artifact_is_rejected(tmp_path):
    identity_for, write_manifest, validate_manifest = _cache_functions()
    predictions = tmp_path / "final_predictions.csv"
    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    artifacts = _cache_artifacts(tmp_path)
    manifest_path = tmp_path / "sim_cache_manifest.json"
    identity = _cache_identity(identity_for, predictions)
    write_manifest(
        str(manifest_path), identity, artifacts, generated_at=CACHE_TIME
    )
    stale_path = tmp_path / "made_cut.npy"
    stale_path.write_bytes(b"stale")
    artifacts["made_cut"] = str(stale_path)

    with pytest.raises(RuntimeError, match="stale made-cut"):
        validate_manifest(
            str(manifest_path), identity, artifacts, now=CACHE_TIME
        )


def test_final_delivery_rechecks_prediction_and_full_contract(tmp_path):
    namespace = _load_definitions(
        "_sha256_file",
        "_contract_json_value",
        "_canonical_contract_sha256",
        "_verify_final_delivery_inputs",
    )
    verify = namespace["_verify_final_delivery_inputs"]
    hash_file = namespace["_sha256_file"]
    hash_contract = namespace["_canonical_contract_sha256"]
    predictions = tmp_path / "final_predictions.csv"
    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    contract = {"weather": [4.0, 6.0], "no_skew_cal": False}
    prediction_sha256 = hash_file(predictions)
    input_sha256 = hash_contract(contract)

    verify(
        prediction_path=str(predictions),
        prediction_sha256=prediction_sha256,
        input_contract_sha256=input_sha256,
        contract_builder=lambda: contract,
    )

    predictions.write_text("player_name,pred\na,3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction file changed"):
        verify(
            prediction_path=str(predictions),
            prediction_sha256=prediction_sha256,
            input_contract_sha256=input_sha256,
            contract_builder=lambda: contract,
        )

    predictions.write_text("player_name,pred\na,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Simulation inputs changed"):
        verify(
            prediction_path=str(predictions),
            prediction_sha256=prediction_sha256,
            input_contract_sha256=input_sha256,
            contract_builder=lambda: {**contract, "no_skew_cal": True},
        )


def _email_namespace(refused=None):
    namespace = _load_definitions("EmailDeliveryError", "send_tournament_email")

    class FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def login(self, _sender, _password):
            return None

        def sendmail(self, _sender, _recipients, _message):
            return refused or {}

    def build_html(*_args, **_kwargs):
        build_html._exchange_bets = []
        build_html._exchange_mu_replacements = []
        return "<p>report</p>"

    namespace.update({
        "EMAIL_PASSWORD": "app-password",
        "EMAIL_FROM": "sender@example.com",
        "EMAIL_TO": ["one@example.com", "two@example.com"],
        "EmailDeliveryError": namespace["EmailDeliveryError"],
        "MIMEApplication": MIMEApplication,
        "MIMEMultipart": MIMEMultipart,
        "MIMEText": MIMEText,
        "build_tournament_email_html": build_html,
        "os": os,
        "smtplib": type("SMTPModule", (), {"SMTP_SSL": FakeSMTP}),
        "tourney": "delivery_test",
    })
    return namespace


def test_required_main_email_needs_configuration_and_all_recipients():
    missing = _email_namespace()
    missing["EMAIL_PASSWORD"] = None
    with pytest.raises(missing["EmailDeliveryError"], match="EMAIL_PASSWORD"):
        missing["send_tournament_email"](None, None, {}, {}, required=True)

    refused = _email_namespace({"two@example.com": (550, b"rejected")})
    with pytest.raises(refused["EmailDeliveryError"], match="two@example.com"):
        refused["send_tournament_email"](None, None, {}, {}, required=True)


def test_required_main_email_returns_acceptance_receipt():
    namespace = _email_namespace()

    assert namespace["send_tournament_email"](
        None, None, {}, {}, required=True
    ) is True


def test_email_function_has_no_hidden_sheet_storage():
    email_node = next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "send_tournament_email"
    )
    called = {
        node.func.id
        for node in ast.walk(email_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "get_spreadsheet" not in called
    assert "store_finish_positions" not in called
    assert "store_tournament_matchups" not in called


@pytest.mark.parametrize("failing_family", ["matchup", "finish"])
def test_storage_helper_propagates_every_store_failure(failing_family):
    store_frames = _load_definitions("_store_tournament_bet_frames")[
        "_store_tournament_bet_frames"
    ]

    class Frame:
        empty = False

    def store_matchups(*_args, **_kwargs):
        if failing_family == "matchup":
            raise OSError("matchup ledger failed")

    def store_finishes(*_args, **_kwargs):
        if failing_family == "finish":
            raise OSError("finish ledger failed")

    with pytest.raises(OSError, match=f"{failing_family} ledger failed"):
        store_frames(
            matchup_frames=[Frame()],
            finish_frames=[Frame()],
            tourney_name="event",
            event_id="28",
            dg_id_lookup={},
            spreadsheet=object(),
            store_matchups=store_matchups,
            store_finishes=store_finishes,
        )


def test_required_publisher_sets_strict_mode_and_propagates(monkeypatch):
    publish_required = _load_definitions("_publish_sim_fairs_required")[
        "_publish_sim_fairs_required"
    ]
    monkeypatch.setenv("REQUIRE_SIM_FAIRS_PUBLISH", "previous")

    class AcceptedPublisher:
        @staticmethod
        def publish(*, push):
            assert push is True
            assert os.environ["REQUIRE_SIM_FAIRS_PUBLISH"] == "1"
            return {"generation": "accepted"}

    assert publish_required(AcceptedPublisher)["generation"] == "accepted"
    assert os.environ["REQUIRE_SIM_FAIRS_PUBLISH"] == "previous"

    class FailedPublisher:
        @staticmethod
        def publish(*, push):
            assert push is True
            assert os.environ["REQUIRE_SIM_FAIRS_PUBLISH"] == "1"
            raise RuntimeError("board dispatch failed")

    with pytest.raises(RuntimeError, match="board dispatch failed"):
        publish_required(FailedPublisher)
    assert os.environ["REQUIRE_SIM_FAIRS_PUBLISH"] == "previous"

    class NoReceiptPublisher:
        @staticmethod
        def publish(*, push):
            assert push is True
            return False

    with pytest.raises(RuntimeError, match="no publication receipt"):
        publish_required(NoReceiptPublisher)


def test_calibration_gate_precedes_all_top_level_delivery_calls():
    class TopLevelCalls(ast.NodeVisitor):
        def __init__(self):
            self.calls = []

        def visit_FunctionDef(self, _node):
            return

        def visit_AsyncFunctionDef(self, _node):
            return

        def visit_ClassDef(self, _node):
            return

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                self.calls.append((node.func.id, node.lineno, node))
            self.generic_visit(node)

    visitor = TopLevelCalls()
    visitor.visit(TREE)
    calls = visitor.calls
    gate = next(
        node
        for node in TREE.body
        if isinstance(node, ast.If)
        and "_delivery_enabled_for_pass" in ast.unparse(node.test)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "exit"
        for node in ast.walk(gate)
    )

    delivery_names = {
        "send_tournament_email",
        "send_exchange_email",
        "send_outrights_email",
        "send_sportsbook_priority_email",
        "get_spreadsheet",
        "_store_tournament_bet_frames",
        "copy_files",
        "git_push",
        "_publish_sim_fairs_required",
    }
    delivery_calls = [item for item in calls if item[0] in delivery_names]
    assert delivery_calls
    assert all(line > gate.lineno for _name, line, _call in delivery_calls)

    main_email = next(call for name, _line, call in calls if name == "send_tournament_email")
    required_kw = next(keyword for keyword in main_email.keywords if keyword.arg == "required")
    assert isinstance(required_kw.value, ast.Constant)
    assert required_kw.value.value is True

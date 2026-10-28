from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import backtest_shot_mean_closing_markets as v4bt
from archive import evaluate_base_skill_finish_markets as evaluator
from archive.evaluate_base_skill_finish_markets import (
    _one_matchup_direction,
    add_complete_zero_bet_event_clusters,
    attach_verified_source_attribution,
    annotate_line_replacement_paths,
    apply_core_delta_to_category_mu,
    attach_finalist,
    assert_common_market_menus,
    assert_matchup_probability_complements,
    bridge_candidate_prediction_with_lambda,
    cross_book_matchup_arbitrage_audit,
    cut_line_for_event,
    finish_event_format_exclusion,
    load_bridge,
    normalize_finish_market_prices,
    select_finish_line_sets,
    selected_downstream_lambdas,
)
from archive.market_backtest_scoring import settle_bets
from archive.market_backtest_scoring import event_cluster_bootstrap


def _write_active_runtime_lock(tmp_path: Path) -> Path:
    path = tmp_path / "finish_market_runtime_v1.json"
    path.write_text(
        json.dumps(
            evaluator.active_research_runtime_lock_payload(), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_finish_runtime_lock_default_is_explicit_research_authority() -> None:
    args = evaluator.parse_args([])
    assert args.runtime_lock == evaluator.DEFAULT_RUNTIME_LOCK
    audit = evaluator.validate_research_runtime_lock(args.runtime_lock)
    assert audit["active_runtime"]["research_only"] is True
    assert (
        audit["active_runtime"]["runtime_role"]
        == "presealed_finish_market_simulation_authority"
    )


def test_finish_runtime_lock_binds_actual_native_extension(tmp_path) -> None:
    lock = _write_active_runtime_lock(tmp_path)
    audit = evaluator.validate_research_runtime_lock(lock)
    package_path, native_path = evaluator.runtime_provenance_source_paths(audit)
    runtime = audit["active_runtime"]["sims_kernel"]
    native = runtime["native_extension"]

    assert package_path.name == "__init__.py"
    assert native_path != package_path
    assert native_path.suffix.casefold() == ".pyd"
    assert native["module"] == "sims_kernel.sims_kernel"
    assert native["source_label"] == "native_extension::sims_kernel.sims_kernel"
    assert native["sha256"] == hashlib.sha256(native_path.read_bytes()).hexdigest()
    assert native["bytes"] == native_path.stat().st_size
    assert runtime["api_distribution_version_match"] is False
    assert runtime["run_pretournament_bound_to_native"] is True


def test_finish_runtime_lock_rejects_tampered_native_hash(tmp_path) -> None:
    lock = _write_active_runtime_lock(tmp_path)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["sims_kernel"]["native_extension"]["sha256"] = "0" * 64
    lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match the active runtime"):
        evaluator.validate_research_runtime_lock(lock)


def test_finish_runtime_provenance_rejects_package_init_as_native(tmp_path) -> None:
    lock = _write_active_runtime_lock(tmp_path)
    audit = evaluator.validate_research_runtime_lock(lock)
    forged = json.loads(json.dumps(audit))
    package = forged["active_runtime"]["sims_kernel"]["package_init"]
    forged["active_runtime"]["sims_kernel"]["native_extension"]["path"] = package[
        "path"
    ]
    with pytest.raises(ValueError, match="not a compiled extension"):
        evaluator.runtime_provenance_source_paths(forged)


def test_finish_runtime_lock_rejects_noncompiled_native_path(tmp_path) -> None:
    source = tmp_path / "sims_kernel.py"
    source.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a compiled extension"):
        evaluator._compiled_extension_suffix(source)


def test_finish_runtime_lock_rejects_missing_native_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="native extension is missing"):
        evaluator._compiled_extension_suffix(tmp_path / "missing.pyd")


def test_finish_runtime_lock_detects_midrun_lock_mutation(tmp_path) -> None:
    lock = _write_active_runtime_lock(tmp_path)
    audit = evaluator.validate_research_runtime_lock(lock)
    lock.write_text(lock.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime lock changed"):
        evaluator.assert_research_runtime_current(audit)


def test_bridge_identity_at_zero_core_delta() -> None:
    baseline = np.array([0.75, -0.20, 1.10])
    legacy = np.array([0.60, -0.30, 0.90])
    full, delta = bridge_candidate_prediction_with_lambda(
        baseline, legacy, legacy, downstream_lambda=1.0
    )
    assert full == pytest.approx(baseline)
    assert delta == pytest.approx(np.zeros(3))


def test_downstream_lambda_formula_has_primary_isolation_identity() -> None:
    baseline = np.array([0.75])
    legacy = np.array([0.60])
    finalist = np.array([0.90])
    full_one, delta_one = bridge_candidate_prediction_with_lambda(
        baseline, legacy, finalist, downstream_lambda=1.0
    )
    full_zero, delta_zero = bridge_candidate_prediction_with_lambda(
        baseline, legacy, finalist, downstream_lambda=0.0
    )
    assert full_one == pytest.approx([1.05])
    assert delta_one == pytest.approx([0.30])
    assert full_zero == pytest.approx([0.90])
    assert delta_zero == pytest.approx([0.15])


def test_load_bridge_rejects_downstream_reconstruction_mismatch(
    tmp_path, monkeypatch
) -> None:
    bridge_path, manifest_path, crosswalk_path = _valid_bridge_files(
        tmp_path, preserved_total=0.11
    )
    _pin_synthetic_bridge_authority(monkeypatch, manifest_path)
    with pytest.raises(ValueError, match="downstream reconstruction mismatch"):
        load_bridge(bridge_path, manifest_path, crosswalk_path)


def _valid_bridge_files(
    tmp_path: Path, *, preserved_total: float = 0.10
) -> tuple[Path, Path, Path]:
    bridge_path = tmp_path / "legacy_core_bridge.parquet"
    pd.DataFrame(
        {
            "event_key": ["local"],
            "event_start": pd.to_datetime(["2025-01-02"]),
            "year": [2025],
            "event_id": [9],
            "dg_id": [7],
            "legacy_core_pred_step2": [0.20],
            "baseline_r1_prediction": [0.30],
            "downstream_preserved_total": [preserved_total],
            "reconstruction_error": [0.0],
        }
    ).to_parquet(bridge_path, index=False)
    crosswalk_path = tmp_path / "odds_event_crosswalk.parquet"
    pd.DataFrame(
        {
            "event_key": ["local"],
            "odds_match_found": [True],
            "crosswalk_ambiguous": [False],
            "odds_query_year": [2025],
            "odds_event_id": [9],
            "event_id_drift": [False],
            "crosswalk_score": [1.0],
        }
    ).to_parquet(crosswalk_path, index=False)
    local_identity = tmp_path / "local_event_identity.parquet"
    pd.DataFrame({"event_key": ["local"]}).to_parquet(local_identity, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_label": "CURRENT-CODE RETROSPECTIVE LEGACY DECOMPOSITION",
                "purpose": (
                    "research-only exact neutral-core bridge for the current-code "
                    "retrospective baseline"
                ),
                "max_abs_reconstruction_error": 0.0,
                "outputs": {
                    bridge_path.name: {
                        "sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest()
                    },
                    crosswalk_path.name: {
                        "sha256": hashlib.sha256(crosswalk_path.read_bytes()).hexdigest()
                    },
                    local_identity.name: {
                        "sha256": hashlib.sha256(local_identity.read_bytes()).hexdigest()
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return bridge_path, manifest_path, crosswalk_path


def _pin_synthetic_bridge_authority(monkeypatch, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_BRIDGE_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        evaluator, "EXPECTED_BRIDGE_MANIFEST_BYTES", manifest_path.stat().st_size
    )
    monkeypatch.setattr(
        evaluator,
        "EXPECTED_BRIDGE_OUTPUT_SHA256",
        {name: row["sha256"] for name, row in payload["outputs"].items()},
    )


def test_load_bridge_requires_manifest_bound_crosswalk_parent(
    tmp_path, monkeypatch
) -> None:
    bridge_path, manifest_path, crosswalk_path = _valid_bridge_files(tmp_path)
    _pin_synthetic_bridge_authority(monkeypatch, manifest_path)
    other = tmp_path / "other" / crosswalk_path.name
    other.parent.mkdir()
    other.write_bytes(crosswalk_path.read_bytes())
    with pytest.raises(ValueError, match="share one resolved parent"):
        load_bridge(bridge_path, manifest_path, other)


def test_load_bridge_rejects_tampered_manifest_bound_crosswalk(
    tmp_path, monkeypatch
) -> None:
    bridge_path, manifest_path, crosswalk_path = _valid_bridge_files(tmp_path)
    _pin_synthetic_bridge_authority(monkeypatch, manifest_path)
    crosswalk_path.write_bytes(crosswalk_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="crosswalk.parquet"):
        load_bridge(bridge_path, manifest_path, crosswalk_path)


def test_load_bridge_rejects_self_consistent_forged_authority(
    tmp_path, monkeypatch
) -> None:
    bridge_path, manifest_path, crosswalk_path = _valid_bridge_files(tmp_path)
    _pin_synthetic_bridge_authority(monkeypatch, manifest_path)
    crosswalk_path.write_bytes(crosswalk_path.read_bytes() + b"forged")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["outputs"][crosswalk_path.name]["sha256"] = hashlib.sha256(
        crosswalk_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest (byte size|hash)"):
        load_bridge(bridge_path, manifest_path, crosswalk_path)


def _write_odds_cache(tmp_path: Path) -> tuple[Path, list[Path], Path, Path]:
    cache = tmp_path / "odds"
    cache.mkdir()
    first = cache / "first.json"
    second = cache / "second.json"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = cache / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "active_shards": [
                    {
                        "data_path": first.name,
                        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return cache, evaluator._odds_cache_provenance_paths(cache), first, second


def _fake_provenance(paths: list[Path]) -> SimpleNamespace:
    return SimpleNamespace(
        input_fingerprints=tuple({"path": str(path.resolve())} for path in paths)
    )


def test_finish_post_begin_revalidation_rejects_odds_byte_swap(tmp_path) -> None:
    cache, expected, first, _second = _write_odds_cache(tmp_path)
    first.write_text("swapped", encoding="utf-8")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        evaluator.revalidate_finish_dynamic_inputs_after_begin(
            finalist_manifest_path=tmp_path / "unused.json",
            identity_preflight=True,
            expected_handoff_paths=[],
            odds_cache=cache,
            expected_odds_paths=expected,
            provenance_state=_fake_provenance(expected),
        )


def test_finish_post_begin_revalidation_rejects_valid_odds_path_switch(tmp_path) -> None:
    cache, expected, _first, second = _write_odds_cache(tmp_path)
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "active_shards": [
                    {
                        "data_path": second.name,
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tracked = [*expected, second]
    with pytest.raises(ValueError, match="path set changed"):
        evaluator.revalidate_finish_dynamic_inputs_after_begin(
            finalist_manifest_path=tmp_path / "unused.json",
            identity_preflight=True,
            expected_handoff_paths=[],
            odds_cache=cache,
            expected_odds_paths=expected,
            provenance_state=_fake_provenance(tracked),
        )


def test_finish_post_begin_revalidation_rejects_valid_handoff_path_switch(
    tmp_path, monkeypatch
) -> None:
    cache, odds_paths, _first, _second = _write_odds_cache(tmp_path)
    original = tmp_path / "original.json"
    switched = tmp_path / "switched.json"
    original.write_text("original", encoding="utf-8")
    switched.write_text("switched", encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "validate_finalist_manifest",
        lambda _path: ({"_verified_handoff_paths": [str(switched)]}, switched),
    )
    with pytest.raises(ValueError, match="handoff resolved path set changed"):
        evaluator.revalidate_finish_dynamic_inputs_after_begin(
            finalist_manifest_path=original,
            identity_preflight=False,
            expected_handoff_paths=[original],
            odds_cache=cache,
            expected_odds_paths=odds_paths,
            provenance_state=_fake_provenance([*odds_paths, original]),
        )


def _fake_v4_audit(paths: list[Path], *, digest: str = "a" * 64) -> dict:
    return {
        "manifest": {"path": str(paths[0]), "sha256": digest},
        "outputs": {
            "run_config.json": {"path": str(paths[1]), "sha256": digest},
            "event_audit.parquet": {"path": str(paths[2]), "sha256": digest},
        },
        "inputs": {
            f"input_{index}": {"path": str(path), "sha256": digest}
            for index, path in enumerate(paths[3:])
        },
        "contract": "shot_mean_closing_market_backtest_v2",
        "complete": True,
    }


def test_finish_post_begin_revalidation_rejects_v4_authority_path_switch(
    tmp_path, monkeypatch
) -> None:
    paths = [tmp_path / f"p{index}" for index in range(5)]
    switched = tmp_path / "switched"
    for path in [*paths, switched]:
        path.write_text(path.name, encoding="utf-8")
    expected = _fake_v4_audit(paths)
    current = _fake_v4_audit([paths[0], paths[1], paths[2], paths[3], switched])
    monkeypatch.setattr(
        evaluator, "validate_v4_authoritative_inputs", lambda **_kwargs: current
    )
    with pytest.raises(ValueError, match="resolved path set changed"):
        evaluator.revalidate_v4_authority_after_begin(
            expected_audit=expected,
            provenance_state=_fake_provenance([*paths, switched]),
            v4_dir=tmp_path,
            db=paths[3],
            predictions=paths[3],
            source_crosswalk=paths[3],
            correlation=paths[3],
            coefficients=paths[3],
            odds_cache=tmp_path,
            shot_export=tmp_path,
        )


def test_finish_post_begin_revalidation_rejects_v4_authority_hash_swap(
    tmp_path, monkeypatch
) -> None:
    paths = [tmp_path / f"p{index}" for index in range(5)]
    for path in paths:
        path.write_text(path.name, encoding="utf-8")
    expected = _fake_v4_audit(paths)
    current = _fake_v4_audit(paths, digest="b" * 64)
    monkeypatch.setattr(
        evaluator, "validate_v4_authoritative_inputs", lambda **_kwargs: current
    )
    with pytest.raises(ValueError, match="hashes changed"):
        evaluator.revalidate_v4_authority_after_begin(
            expected_audit=expected,
            provenance_state=_fake_provenance(paths),
            v4_dir=tmp_path,
            db=paths[3],
            predictions=paths[3],
            source_crosswalk=paths[3],
            correlation=paths[3],
            coefficients=paths[3],
            odds_cache=tmp_path,
            shot_export=tmp_path,
        )


def test_finish_execution_provenance_rejects_midrun_input_mutation(tmp_path) -> None:
    direct_input = tmp_path / "direct_input.csv"
    direct_input.write_text("before\n", encoding="utf-8")
    output = tmp_path / "finish_output"
    args = evaluator.parse_args(["--output-dir", str(output)])
    runtime_lock = _write_active_runtime_lock(tmp_path)
    runtime_audit = evaluator.validate_research_runtime_lock(runtime_lock)
    state = evaluator.begin_finish_execution_provenance(
        output_dir=output,
        input_paths=[direct_input],
        args=args,
        runtime_audit=runtime_audit,
    )
    source_records = {
        Path(record["path"]): record for record in state.source_fingerprints
    }
    _package_path, native_path = evaluator.runtime_provenance_source_paths(
        runtime_audit
    )
    assert native_path in source_records
    assert source_records[native_path]["sha256"] == runtime_audit["active_runtime"][
        "sims_kernel"
    ]["native_extension"]["sha256"]
    assert Path(runtime_audit["lock"]["path"]) in {
        Path(record["path"]) for record in state.input_fingerprints
    }
    direct_input.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed"):
        evaluator.finalize_finish_execution_provenance(state)


def test_finish_handoff_provenance_source_closure_is_exact() -> None:
    expected = tuple(
        dict.fromkeys(
            (
                evaluator.ARCHIVE / "base_skill_frozen_handoff.py",
                Path(evaluator.discovery.__file__).resolve(),
                *(
                    Path(path).resolve()
                    for path in evaluator.discovery.DISCOVERY_PROVENANCE_SOURCE_CLOSURE
                ),
            )
        )
    )
    assert evaluator.HANDOFF_PROVENANCE_SOURCE_CLOSURE == expected
    assert {path.name for path in expected} >= {
        "base_skill_frozen_handoff.py",
        "evaluate_complete_base_skill_market_candidates.py",
        "evaluate_complete_base_skill_market.py",
        "backtest_base_skill_market_models.py",
        "historical_odds_cache.py",
        "market_backtest_scoring.py",
        "research_execution_provenance.py",
    }
    assert all(path.is_file() for path in expected)


def test_finish_resume_rejects_late_untracked_checkpoint(tmp_path) -> None:
    run_config = tmp_path / "run_config.json"
    run_config.write_text("{}", encoding="utf-8")
    state = SimpleNamespace(
        input_fingerprints=({"path": str(run_config.resolve())},)
    )
    checkpoint = tmp_path / "event_checkpoint.parquet"
    checkpoint_meta = tmp_path / "event_checkpoint.meta.json"
    checkpoint.write_bytes(b"created after provenance begin")
    checkpoint_meta.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="was not tracked at provenance begin"):
        evaluator.require_tracked_resume_checkpoint(
            state,
            run_config_path=run_config,
            checkpoint_path=checkpoint,
            checkpoint_meta_path=checkpoint_meta,
        )


def test_finish_complete_manifest_rejects_post_finalize_input_mutation(
    tmp_path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"before")
    state = evaluator.execution_provenance.begin(
        output_dir=tmp_path / "output",
        primary_source=source,
        input_paths=[input_path],
    )
    evaluator.execution_provenance.finalize(state)
    input_path.write_bytes(b"after finalize")
    target = tmp_path / "output" / "manifest.json"
    with pytest.raises(RuntimeError, match="input changed"):
        evaluator.write_finish_complete_manifest(target, {"complete": True}, state)
    assert not target.exists()


def test_v4_authority_rejects_arbitrary_source_crosswalk(tmp_path) -> None:
    authoritative = tmp_path / "authoritative_crosswalk.parquet"
    arbitrary = tmp_path / "arbitrary_crosswalk.parquet"
    authoritative.write_text("frozen", encoding="utf-8")
    arbitrary.write_text("arbitrary", encoding="utf-8")
    locked = {
        "crosswalk": hashlib.sha256(authoritative.read_bytes()).hexdigest()
    }
    with pytest.raises(ValueError, match="V4-authoritative input hash mismatch"):
        evaluator._require_locked_hash(locked, "crosswalk", arbitrary)


def test_v4_authority_rejects_tampered_source_crosswalk(tmp_path) -> None:
    crosswalk = tmp_path / "crosswalk.parquet"
    crosswalk.write_text("frozen", encoding="utf-8")
    locked = {"crosswalk": hashlib.sha256(crosswalk.read_bytes()).hexdigest()}
    crosswalk.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="V4-authoritative input hash mismatch"):
        evaluator._require_locked_hash(locked, "crosswalk", crosswalk)


def test_finish_completion_authority_rejects_unrelated_round_panel(tmp_path) -> None:
    frozen_rounds = tmp_path / "frozen_rounds.parquet"
    supplied_rounds = tmp_path / "supplied_rounds.parquet"
    bridge = tmp_path / "legacy_core_bridge.parquet"
    frozen_rounds.write_text("frozen", encoding="utf-8")
    supplied_rounds.write_text("unrelated", encoding="utf-8")
    bridge.write_text("bridge", encoding="utf-8")
    manifest = {
        "_verified_completion_inputs": {
            "round_panel": {
                "path": str(frozen_rounds),
                "sha256": hashlib.sha256(frozen_rounds.read_bytes()).hexdigest(),
            },
            "core_bridge": {
                "path": str(bridge),
                "sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
            },
        }
    }
    with pytest.raises(ValueError, match="not the frozen completion input"):
        evaluator.validate_finish_completion_authorities(
            manifest, round_panel=supplied_rounds, bridge=bridge
        )


def test_category_shift_is_equal_and_sums_to_total_core_delta() -> None:
    original = np.array([[0.10, 0.20, -0.05, 0.25], [-0.30, 0.15, 0.05, -0.10]])
    delta = np.array([0.40, -0.20])
    shifted = apply_core_delta_to_category_mu({"mu": original}, delta)
    expected_shift = np.repeat(delta[:, None] / 4.0, 4, axis=1)
    assert shifted - original == pytest.approx(expected_shift)
    assert shifted.sum(axis=1) == pytest.approx(original.sum(axis=1) + delta)


def test_finish_probability_variants_keep_native_dead_heat_and_full_tie(
    monkeypatch,
) -> None:
    standard = np.array([[0.25, 0.50, 0.75], [0.10, 0.40, 0.60]])
    full_tie = np.array([[0.75, 0.75, 1.00], [0.50, 0.75, 0.90]])

    def aggregate(scores):
        return np.zeros((2, 2)), standard, full_tie

    monkeypatch.setitem(
        sys.modules,
        "sims_kernel",
        SimpleNamespace(aggregate_round=aggregate),
    )
    scores = np.array([[1, 2, 1, 1], [2, 2, 1, 3]], dtype=np.int64)
    variants = evaluator.finish_probability_variants(scores)
    assert variants["standard_dead_heat"]["top_5"] == pytest.approx([0.25, 0.10])
    assert variants["full_tie_qualification"]["top_5"] == pytest.approx(
        [0.75, 0.50]
    )
    # Win remains the existing split-first-place expectation under both labels.
    assert variants["standard_dead_heat"]["win"] == pytest.approx([0.75, 0.25])
    assert variants["full_tie_qualification"]["win"] == pytest.approx([0.75, 0.25])


def _synthetic_top_finish_line(book: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "pga:R2025009",
                "event_key": "pga:R2025009",
                "event_start": pd.Timestamp("2025-03-06"),
                "market": "top_5",
                "selection_id": "10",
                "opponent_id": "",
                "book": book,
                "close_odds": 100,
                "bet_outcome_numeric": 1.0,
            }
        ]
    )


def _synthetic_top_finish_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "pga:R2025009",
                "event_key": "pga:R2025009",
                "market": "top_5",
                "selection_id": "10",
                "opponent_id": "",
                "probability": 0.25,
                "standard_dead_heat_probability": 0.25,
                "full_tie_qualification_probability": 0.60,
                "push_probability": 0.0,
            }
        ]
    )


def test_executable_top_finish_probability_is_book_specific_and_changes_gate() -> None:
    probabilities = _synthetic_top_finish_probabilities()
    pinnacle_lines = _synthetic_top_finish_line("Pinnacle")
    pinnacle_all = evaluator.select_all_offered_family_bets(
        pinnacle_lines, probabilities, "baseline", "outright"
    )
    assert pinnacle_all.iloc[0]["probability"] == pytest.approx(0.60)
    assert pinnacle_all.iloc[0]["ev_per_unit"] == pytest.approx(0.20)
    assert (
        pinnacle_all.iloc[0]["model_probability_rule"]
        == "pinnacle_full_tie_qualification_top_finish"
    )
    assert len(
        evaluator.select_family_bets(
            pinnacle_lines, probabilities, "baseline", "outright", "flat_1u"
        )
    ) == 1

    for book in ("BetOnline", "Betcris"):
        lines = _synthetic_top_finish_line(book)
        standard_all = evaluator.select_all_offered_family_bets(
            lines, probabilities, "baseline", "outright"
        )
        assert standard_all.iloc[0]["probability"] == pytest.approx(0.25)
        assert standard_all.iloc[0]["ev_per_unit"] == pytest.approx(-0.50)
        assert (
            standard_all.iloc[0]["model_probability_rule"]
            == "standard_dead_heat_expected_paid_share_top_finish"
        )
        assert evaluator.select_family_bets(
            lines, probabilities, "baseline", "outright", "flat_1u"
        ).empty


def test_book_rule_resolution_preserves_win_and_matchup_semantics() -> None:
    lines = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "market": "win",
                "selection_id": "1",
                "opponent_id": "",
                "book": "pinnacle",
            },
            {
                "event_id": "e1",
                "market": "tournament_matchup",
                "selection_id": "1",
                "opponent_id": "2",
                "book": "pinnacle",
            },
        ]
    )
    probabilities = lines[evaluator.v4bt.KEY_COLS].copy()
    probabilities["probability"] = [0.08, 0.42]
    probabilities["push_probability"] = [0.0, 0.07]
    annotated, resolved = evaluator.resolve_book_rule_probabilities(lines, probabilities)
    assert resolved["probability"].tolist() == pytest.approx([0.08, 0.42])
    assert resolved["push_probability"].tolist() == pytest.approx([0.0, 0.07])
    assert annotated["model_probability_rule"].tolist() == [
        "win_split_first_place_dead_heat_share",
        "matchup_win_with_explicit_push_mass",
    ]


def test_top_finish_probability_resolution_fails_closed_without_native_variants() -> None:
    probabilities = _synthetic_top_finish_probabilities().drop(
        columns=["full_tie_qualification_probability"]
    )
    with pytest.raises(ValueError, match="lack native book-rule variants"):
        evaluator.resolve_book_rule_probabilities(
            _synthetic_top_finish_line("Pinnacle"), probabilities
        )


def _probability_menu() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "pga:R2025009",
                "market": "tournament_matchup",
                "selection_id": "1",
                "opponent_id": "2",
                "probability": 0.40,
                "push_probability": 0.10,
            },
            {
                "event_id": "pga:R2025009",
                "market": "tournament_matchup",
                "selection_id": "2",
                "opponent_id": "1",
                "probability": 0.50,
                "push_probability": 0.10,
            },
            {
                "event_id": "pga:R2025009",
                "market": "top_20",
                "selection_id": "1",
                "opponent_id": "",
                "probability": 0.30,
                "push_probability": 0.0,
            },
        ]
    )


def test_probability_complements_include_push_mass() -> None:
    probabilities = pd.concat(
        [
            _probability_menu().assign(arm="baseline"),
            _probability_menu().assign(arm="finalist_lambda_1p0"),
        ],
        ignore_index=True,
    )
    assert_matchup_probability_complements(probabilities)
    broken = probabilities.copy()
    broken.loc[
        broken["arm"].eq("baseline") & broken["selection_id"].eq("2"),
        "probability",
    ] = 0.49
    with pytest.raises(ValueError, match="do not complement"):
        assert_matchup_probability_complements(broken)


def test_market_menu_must_be_identical_across_arms() -> None:
    baseline = _probability_menu().assign(arm="baseline")
    finalist = _probability_menu().assign(arm="finalist_lambda_1p0")
    assert_common_market_menus(pd.concat([baseline, finalist], ignore_index=True))
    with pytest.raises(ValueError, match="market menu differs"):
        assert_common_market_menus(
            pd.concat([baseline, finalist.iloc[:-1]], ignore_index=True)
        )


def test_cut_rule_is_event_year_safe_and_major_parity() -> None:
    pga = v4bt.EventMeta(
        "pga:R2025033",
        2025,
        33,
        "PGA Championship",
        pd.Timestamp("2025-05-15"),
        ((4, 156),),
    )
    us_open = v4bt.EventMeta(
        "pga:R2025026",
        2025,
        26,
        "U.S. Open",
        pd.Timestamp("2025-06-12"),
        ((5, 156),),
    )
    assert cut_line_for_event(pga, 156) == (70, "pga_championship_top70")
    assert cut_line_for_event(us_open, 156) == (60, "us_open_top60")
    travelers_2023 = v4bt.EventMeta(
        "pga:R2023027",
        2023,
        27,
        "Travelers Championship",
        pd.Timestamp("2023-06-22"),
        ((6, 156),),
    )
    travelers_2024 = v4bt.EventMeta(
        "pga:R2024027",
        2024,
        27,
        "Travelers Championship",
        pd.Timestamp("2024-06-20"),
        ((6, 70),),
    )
    assert cut_line_for_event(travelers_2023, 156) == (65, "standard_top65")
    assert cut_line_for_event(travelers_2024, 70) == (70, "limited_field_no_cut")
    zurich = v4bt.EventMeta(
        "pga:R2025018",
        2025,
        18,
        "Zurich Classic of New Orleans",
        pd.Timestamp("2025-04-24"),
        ((7, 160),),
    )
    assert finish_event_format_exclusion(zurich) == "team_format_unmodeled"
    with pytest.raises(ValueError, match="team_format_unmodeled"):
        cut_line_for_event(zurich, 160)


def test_matchup_direction_guard_keeps_highest_ev_side() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["e", "e"],
            "selection_id": ["1", "2"],
            "opponent_id": ["2", "1"],
            "ev_per_unit": [0.04, 0.08],
        }
    )
    chosen = _one_matchup_direction(bets)
    assert len(chosen) == 1
    assert chosen.iloc[0]["selection_id"] == "2"


def test_settlement_preserves_matchup_push_and_outright_dead_heat() -> None:
    bets = pd.DataFrame(
        [
            {
                "event_id": "e",
                "market": "tournament_matchup",
                "selection_id": "1",
                "opponent_id": "2",
                "close_odds": 100,
                "stake": 2.0,
                "outcome_numeric": 0.5,
                "tie_rule": "void",
            },
            {
                "event_id": "e",
                "market": "top_5",
                "selection_id": "3",
                "opponent_id": "",
                "close_odds": 300,
                "stake": 2.0,
                "bet_outcome_numeric": 0.5,
            },
        ]
    )
    settled = settle_bets(bets)
    assert settled.loc[0, "settlement_status"] == "push"
    assert settled.loc[0, "return_amount"] == pytest.approx(2.0)
    assert settled.loc[0, "pnl"] == pytest.approx(0.0)
    assert settled.loc[1, "settlement_status"] == "dead_heat"
    assert settled.loc[1, "return_amount"] == pytest.approx(4.0)
    assert settled.loc[1, "pnl"] == pytest.approx(2.0)


def _synthetic_outright_source(
    book: str,
    fraction: float,
    record_id: str,
    *,
    finish: str = "T7",
    market: str = "top_10",
) -> dict[str, object]:
    return {
        "query_year": 2025,
        "event_id": 9,
        "event_name": "Synthetic",
        "market": market,
        "book": book,
        "dg_id": 19895,
        "player_name": "Schauffele, Xander",
        "open_odds": 100,
        "close_odds": 100,
        "open_time": "2025-01-01T12:00:00Z",
        "close_time": "2025-01-02T12:00:00Z",
        "outcome": finish,
        "bet_outcome_numeric": fraction,
        "bet_outcome_text": (
            "paid in full" if fraction == 1.0 else "dead-heat: synthetic"
        ),
        "record_id": record_id,
    }


def _synthetic_official_result(
    *, finish: str = "T7", tie_count: int = 2
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["pga:R2025009"],
            "selection_id": ["19895"],
            "official_finish": [finish],
            "official_tie_count": [tie_count],
            "local_year": [2025],
            "local_event_id": [9],
        }
    )


def test_pinnacle_t7_top10_full_payout_is_reconciled_and_propagated() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source(
                "pinnacle", 4.0 / 7.0, "pin-corrupt"
            ),
            _synthetic_outright_source("betonline", 1.0, "bo-corroborates"),
        ],
    )
    reconciled, corrections, audit = evaluator.reconcile_outright_settlements(
        lines, _synthetic_official_result()
    )
    pin = reconciled.loc[reconciled["book"].eq("pinnacle")].iloc[0]
    assert pin["archived_bet_outcome_numeric"] == pytest.approx(4.0 / 7.0)
    assert pin["standard_dead_heat_bet_outcome_numeric"] == pytest.approx(1.0)
    assert pin["book_rule_bet_outcome_numeric"] == pytest.approx(1.0)
    assert pin["bet_outcome_numeric"] == pytest.approx(1.0)
    assert bool(pin["settlement_reconstructed"])
    assert len(corrections) == 1
    assert audit["primary_override_count"] == 1
    assert audit["standard_discrepancy_count"] == 1
    assert audit["standard_discrepancies_explicitly_emitted"] is True
    assert (
        audit["nonconflicting_raw_payload_sha256"]
        == audit["nonconflicting_primary_payload_sha256"]
    )
    assert bool(corrections.iloc[0]["standard_dead_heat_differs_from_archive"])
    menus = select_finish_line_sets(reconciled)
    pin_menu = menus[("close", "pinnacle")].iloc[0]
    assert pin_menu["bet_outcome_numeric"] == pytest.approx(1.0)
    assert pin_menu["archived_bet_outcome_numeric"] == pytest.approx(4.0 / 7.0)
    settled = settle_bets(
        menus[("close", "pinnacle")].assign(
            stake=1.0,
            close_odds=menus[("close", "pinnacle")]["execution_odds"],
        )
    )
    assert settled.iloc[0]["settlement_status"] == "win"
    assert settled.iloc[0]["pnl"] == pytest.approx(1.0)


def test_boundary_tie_keeps_standard_sensitivity_but_pinnacle_primary_is_full() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source(
                "pinnacle", 0.75, "pin-boundary", finish="T8"
            ),
            _synthetic_outright_source(
                "betcris", 0.75, "bc-boundary", finish="T8"
            ),
        ],
    )
    reconciled, corrections, audit = evaluator.reconcile_outright_settlements(
        lines, _synthetic_official_result(finish="T8", tie_count=4)
    )
    pin = reconciled.loc[reconciled["book"].eq("pinnacle")].iloc[0]
    betcris = reconciled.loc[reconciled["book"].eq("betcris")].iloc[0]
    assert pin["standard_dead_heat_bet_outcome_numeric"] == pytest.approx(0.75)
    assert pin["book_rule_bet_outcome_numeric"] == pytest.approx(1.0)
    assert betcris["book_rule_bet_outcome_numeric"] == pytest.approx(0.75)
    assert len(corrections) == 1
    assert audit["primary_override_count"] == 1
    assert audit["standard_discrepancy_count"] == 0


def test_outright_reconciliation_rejects_non_pinnacle_source_error() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source("betonline", 0.5, "bo-bad"),
            _synthetic_outright_source("pinnacle", 1.0, "pin-good"),
        ],
    )
    with pytest.raises(ValueError, match="BetOnline/Betcris archive disagrees"):
        evaluator.reconcile_outright_settlements(
            lines, _synthetic_official_result()
        )


def test_outright_reconciliation_rejects_uncorroborated_override() -> None:
    lines = normalize_finish_market_prices(
        [], [_synthetic_outright_source("pinnacle", 0.5, "pin-alone")]
    )
    with pytest.raises(ValueError, match="cross-book qualification corroboration"):
        evaluator.reconcile_outright_settlements(
            lines, _synthetic_official_result()
        )


@pytest.mark.parametrize(
    ("finish", "tie_count", "message"),
    [("T7", 1, "ambiguous or impossible tie group"), ("n/a", 1, "unknown official")],
)
def test_outright_reconciliation_fails_closed_on_bad_official_result(
    finish: str, tie_count: int, message: str
) -> None:
    raw_fraction = 0.0 if finish == "n/a" else 1.0
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source(
                "betonline", raw_fraction, "bad-official", finish=finish
            )
        ],
    )
    with pytest.raises(ValueError, match=message):
        evaluator.reconcile_outright_settlements(
            lines,
            _synthetic_official_result(finish=finish, tie_count=tie_count),
        )


def test_unreconciled_cross_book_positive_fraction_difference_is_rejected() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source(
                "pinnacle", 0.5, "pin-unreconciled", finish="T8"
            ),
            _synthetic_outright_source(
                "betonline", 0.75, "bo-unreconciled", finish="T8"
            ),
        ],
    )
    with pytest.raises(ValueError, match="require locked official-result"):
        select_finish_line_sets(lines)


def test_frozen_settlement_inventory_count_is_fail_closed() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source("pinnacle", 0.5, "pin-count"),
            _synthetic_outright_source("betonline", 1.0, "bo-count"),
        ],
    )
    with pytest.raises(ValueError, match="override count differs"):
        evaluator.reconcile_outright_settlements(
            lines,
            _synthetic_official_result(),
            enforce_frozen_authority=True,
            expected_primary_override_count=2,
            expected_standard_discrepancy_count=1,
        )


def test_frozen_settlement_inventory_digest_is_fail_closed() -> None:
    lines = normalize_finish_market_prices(
        [],
        [
            _synthetic_outright_source("pinnacle", 0.5, "pin-digest"),
            _synthetic_outright_source("betonline", 1.0, "bo-digest"),
        ],
    )
    with pytest.raises(ValueError, match="record-id digest differs"):
        evaluator.reconcile_outright_settlements(
            lines,
            _synthetic_official_result(),
            enforce_frozen_authority=True,
            expected_primary_override_count=1,
            expected_primary_record_id_sha256="0" * 64,
            expected_standard_discrepancy_count=1,
        )


def test_outright_reconciliation_rejects_finish_disagreement() -> None:
    lines = normalize_finish_market_prices(
        [], [_synthetic_outright_source("betonline", 1.0, "finish-mismatch")]
    )
    with pytest.raises(ValueError, match="finish disagrees"):
        evaluator.reconcile_outright_settlements(
            lines, _synthetic_official_result(finish="T8", tie_count=2)
        )


def test_outright_reconciliation_rejects_missing_official_result() -> None:
    lines = normalize_finish_market_prices(
        [], [_synthetic_outright_source("betonline", 1.0, "missing-official")]
    )
    with pytest.raises(ValueError, match="official result is missing"):
        evaluator.reconcile_outright_settlements(
            lines, _synthetic_official_result().iloc[0:0]
        )


def test_alternate_downstream_lambda_requires_discovery_freeze() -> None:
    base = {"downstream_lambdas": [1.0, 0.5]}
    with pytest.raises(ValueError, match="cutoff"):
        selected_downstream_lambdas(base)
    frozen = {
        **base,
        "downstream_lambda_selection_window_end": "2022-12-31",
        "sealed_data_used_for_downstream_lambda_selection": False,
        "downstream_lambda_frozen_before_sealed_scoring": True,
    }
    assert selected_downstream_lambdas(frozen) == (1.0, 0.5)


def test_unbridged_player_uses_strict_pt_control_then_shared_prior_without_drop() -> None:
    field = pd.DataFrame(
        {
            "dg_id": [1, 2, 3],
            "player_name": ["one", "two", "three"],
            "skill_est": [0.8, 0.1, -0.5],
            "skill_source": ["r1_prediction", "dated_model_fallback", "field_floor_fallback"],
        }
    )
    bridge = pd.DataFrame(
        {
            "dg_id": [1],
            "legacy_core_pred_step2": [0.6],
            "baseline_r1_prediction": [0.8],
            "local_event_key": ["local"],
            "event_id_drift": [False],
        }
    )
    finalist = pd.DataFrame(
        {
            "dg_id": [1, 2],
            "finalist_core": [0.9, 0.4],
            "control_core": [0.7, 0.2],
        }
    )
    output, audit = attach_finalist(field, bridge, finalist, (1.0, 0.5, 0.0))
    assert output["replacement_path"].tolist() == [
        "exact_legacy_bridge",
        "strict_pt_v2_control_fallback",
        "shared_neutral_field_tour_prior",
    ]
    assert output["core_delta"].tolist() == pytest.approx([0.3, 0.2, 0.0])
    assert audit["exact_bridge_replacement_players"] == 1
    assert audit["strict_pt_control_fallback_players"] == 1
    assert audit["neutral_fallback_players"] == 1
    for arm in ("finalist_lambda_1p0", "finalist_lambda_0p5", "finalist_lambda_0p0"):
        assert output.loc[output["dg_id"].eq(3), f"{arm}_delta"].iloc[0] == 0.0
        assert output.loc[output["dg_id"].eq(3), f"{arm}_full_prediction"].iloc[0] == -0.5

    lines = pd.DataFrame(
        [
            {
                "event_key": "e",
                "market": "tournament_matchup",
                "selection_id": "1",
                "opponent_id": "2",
            },
            {
                "event_key": "e",
                "market": "top_20",
                "selection_id": "3",
                "opponent_id": "",
            },
        ]
    )
    annotated = annotate_line_replacement_paths(lines, {"e": output})
    assert len(annotated) == len(lines)
    assert annotated["replacement_path"].tolist() == [
        "strict_pt_v2_control_fallback",
        "shared_neutral_field_tour_prior",
    ]


def test_cross_book_arbitrage_is_reported_not_credited_as_model_alpha() -> None:
    lines = pd.DataFrame(
        [
            {
                "event_key": "e",
                "market": "tournament_matchup",
                "selection_id": "1",
                "opponent_id": "2",
                "selection_name": "one",
                "book": "pinnacle",
                "close_odds": 110,
                "tie_rule": "void on tie",
            },
            {
                "event_key": "e",
                "market": "tournament_matchup",
                "selection_id": "2",
                "opponent_id": "1",
                "selection_name": "two",
                "book": "betcris",
                "close_odds": 110,
                "tie_rule": "void on tie",
            },
        ]
    )
    audit = cross_book_matchup_arbitrage_audit(lines)
    assert len(audit) == 1
    assert bool(audit.iloc[0]["arbitrage"])
    assert bool(audit.iloc[0]["both_legs_push_on_tie"])
    assert audit.iloc[0]["raw_implied_sum"] == pytest.approx(2 / 2.1)


def test_finish_line_sets_are_true_per_book_open_close_and_same_book_devig() -> None:
    common = {
        "query_year": 2025,
        "event_id": 9,
        "event_name": "Synthetic",
        "bet_type": "Tournament Matchups",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p1_player_name": "one",
        "p2_player_name": "two",
        "open_time": "2025-01-01T12:00:00Z",
        "close_time": "2025-01-02T12:00:00Z",
        "p1_outcome": 1.0,
        "p2_outcome": 0.0,
        "p1_outcome_text": "win",
        "p2_outcome_text": "loss",
        "tie_rule": "void on tie",
    }
    rows = [
        {
            **common,
            "book": "betonline",
            "record_id": "bo",
            "p1_open": 100,
            "p2_open": -120,
            "p1_close": -110,
            "p2_close": 100,
        },
        {
            **common,
            "book": "pinnacle",
            "record_id": "pin",
            "p1_open": 120,
            "p2_open": -130,
            "p1_close": 105,
            "p2_close": -115,
        },
        {
            **common,
            "book": "betcris",
            "record_id": "bc",
            "p1_open": 110,
            "p2_open": 110,
            "p1_close": 115,
            "p2_close": 105,
        },
    ]
    normalized = normalize_finish_market_prices(rows, [])
    menus = select_finish_line_sets(normalized)
    assert set(menus[("open", "betonline")]["book"]) == {"betonline"}
    assert set(menus[("open", "pinnacle")]["book"]) == {"pinnacle"}
    best_open = menus[("open", "best_of_three")].set_index("selection_id")
    assert best_open.loc["1", "book"] == "pinnacle"
    assert best_open.loc["1", "execution_odds"] == 120
    assert best_open.loc["1", "same_book_opposite_odds"] == -130
    assert best_open.loc["2", "book"] == "betcris"
    assert best_open.loc["2", "execution_odds"] == 110
    assert best_open["cross_book_two_side_arbitrage"].all()
    assert best_open["cross_book_arbitrage_both_legs_push_on_tie"].all()
    assert not best_open["same_book_devig_proper_score_eligible"].any()
    best_close = menus[("close", "best_of_three")].set_index("selection_id")
    assert best_close.loc["1", "execution_odds"] == 115
    assert best_close.loc["1", "book"] == "betcris"
    assert best_close.loc["1", "archived_open_odds"] == 110


def test_ties_lose_cross_book_positive_hold_is_not_guaranteed_arbitrage() -> None:
    common = {
        "query_year": 2025,
        "event_id": 9,
        "event_name": "Synthetic",
        "bet_type": "Tournament Matchup 3 Way",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p1_player_name": "one",
        "p2_player_name": "two",
        "open_time": "2025-01-01T12:00:00Z",
        "close_time": "2025-01-02T12:00:00Z",
        "p1_outcome": 0.0,
        "p2_outcome": 0.0,
        "tie_rule": "3 way; ties lose",
    }
    normalized = normalize_finish_market_prices(
        [
            {
                **common,
                "book": "pinnacle",
                "record_id": "pin",
                "p1_open": 120,
                "p2_open": 100,
                "p1_close": 120,
                "p2_close": 100,
            },
            {
                **common,
                "book": "betcris",
                "record_id": "bc",
                "p1_open": 100,
                "p2_open": 120,
                "p1_close": 100,
                "p2_close": 120,
            },
        ],
        [],
    )
    best = select_finish_line_sets(normalized)[("close", "best_of_three")]
    assert set(best["execution_odds"]) == {120.0}
    assert not best["cross_book_two_side_arbitrage"].any()
    assert not best["cross_book_arbitrage_both_legs_push_on_tie"].any()
    assert best["opportunity_class"].eq("model_directional_opportunity").all()
    assert best["same_book_devig_probability_role"].str.contains(
        "attribution_only_no_draw_quote"
    ).all()
    assert not best["same_book_devig_proper_score_eligible"].any()
    settled = settle_bets(best.assign(stake=1.0, close_odds=best["execution_odds"]))
    assert settled["settlement_status"].eq("tie_loss").all()
    assert settled["pnl"].eq(-1.0).all()
    raw_audit = cross_book_matchup_arbitrage_audit(best)
    assert len(raw_audit) == 1
    assert raw_audit.iloc[0]["raw_implied_sum"] == pytest.approx(2 / 2.2)
    assert not bool(raw_audit.iloc[0]["both_legs_push_on_tie"])
    assert not bool(raw_audit.iloc[0]["arbitrage"])
    assert raw_audit.iloc[0]["opportunity_class"] == "model_directional_opportunity"


def test_finish_normalizer_rejects_tournament_three_ball_as_two_way() -> None:
    row = {
        "query_year": 2025,
        "event_id": 9,
        "bet_type": "72 Hole Three Ball Matchup",
        "book": "pinnacle",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p3_dg_id": 3,
        "p1_open": 120,
        "p2_open": 120,
        "p1_close": 110,
        "p2_close": 110,
    }
    with pytest.raises(ValueError, match="three-ball"):
        normalize_finish_market_prices([row], [])


def test_finish_normalizer_accepts_two_player_three_way_ties_lose_marker() -> None:
    row = {
        "query_year": 2025,
        "event_id": 9,
        "bet_type": "Tournament Matchup 3 Way",
        "book": "pinnacle",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p1_open": 120,
        "p2_open": 120,
        "p1_close": 110,
        "p2_close": 110,
        "p1_outcome": 0.0,
        "p2_outcome": 0.0,
        "tie_rule": "ties lose",
    }
    normalized = normalize_finish_market_prices([row], [])
    assert len(normalized) == 2


def test_finish_line_sets_reject_cross_book_settlement_conflict() -> None:
    common = {
        "query_year": 2025,
        "event_id": 9,
        "bet_type": "Tournament Matchup",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p1_open": 100,
        "p2_open": 100,
        "p1_close": 100,
        "p2_close": 100,
        "tie_rule": "void on tie",
    }
    lines = normalize_finish_market_prices(
        [
            {**common, "book": "pinnacle", "p1_outcome": 1.0, "p2_outcome": 0.0},
            {**common, "book": "betcris", "p1_outcome": 0.0, "p2_outcome": 1.0},
        ],
        [],
    )
    with pytest.raises(ValueError, match="cross-book physical settlement conflict"):
        select_finish_line_sets(lines)


@pytest.mark.parametrize("outcomes", [(1.0, 1.0), (0.5, 0.0)])
def test_finish_line_sets_reject_impossible_reverse_outcomes(outcomes) -> None:
    row = {
        "query_year": 2025,
        "event_id": 9,
        "bet_type": "Tournament Matchup",
        "book": "pinnacle",
        "p1_dg_id": 1,
        "p2_dg_id": 2,
        "p1_open": 100,
        "p2_open": 100,
        "p1_close": 100,
        "p2_close": 100,
        "p1_outcome": outcomes[0],
        "p2_outcome": outcomes[1],
        "tie_rule": "void on tie",
    }
    lines = normalize_finish_market_prices([row], [])
    with pytest.raises(ValueError, match="impossible reverse matchup settlement"):
        select_finish_line_sets(lines)


def test_zero_bet_event_clusters_keep_both_arms_and_paired_delta() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["e1"],
            "event_start": pd.to_datetime(["2025-01-01"]),
            "arm": ["baseline"],
            "stake": [1.0],
            "pnl": [1.0],
        }
    )
    lines = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "event_start": pd.to_datetime(["2025-01-01", "2025-01-08"]),
        }
    )
    clustered = add_complete_zero_bet_event_clusters(bets, lines)
    observed = set(zip(clustered["event_id"], clustered["arm"]))
    assert observed == {
        ("e1", "baseline"),
        ("e1", "challenger"),
        ("e2", "baseline"),
        ("e2", "challenger"),
    }
    boot = event_cluster_bootstrap(
        clustered,
        event_col="event_id",
        event_order_col="event_start",
        draws=100,
        seed=7,
    )
    assert set(boot["cluster_count"]) == {2}
    assert "challenger_minus_baseline" in set(boot["arm"])


def test_finish_attribution_comes_from_exact_source_prediction_rows(tmp_path) -> None:
    candidate = "complete__ridge__total__w0__annual__a1"
    control = "current_architecture_pt_v2_approx_control"
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "candidate": [candidate, control],
            "canonical_event_key": ["pga:R2025009", "pga:R2025009"],
            "dg_id_key": ["7", "7"],
            "prediction": [0.25, 0.10],
            "history_rounds": [42, 42],
            "days_since_last_round": [7, 7],
            "reliability_ok": [True, True],
            "cross_tour_fallback_reason": ["", ""],
            "img_history_rounds": [20, 20],
            "img_history_coverage": [0.8, 0.8],
            "prior_event_withdrawal": [False, False],
        }
    ).to_parquet(source, index=False)
    finalist = pd.DataFrame(
        {
            "event_key": ["pga:R2025009"],
            "dg_id": [7],
            "finalist_core": [0.25],
            "control_core": [0.10],
        }
    )
    attached, audit = attach_verified_source_attribution(
        finalist,
        {
            "_verified_source_prediction_path": str(source),
            "control_name": control,
        },
        candidate,
    )
    assert attached.loc[0, "history_rounds"] == 42
    assert attached.loc[0, "img_history_coverage"] == pytest.approx(0.8)
    assert audit["source_coverage_rate"] == 1.0

    bad = finalist.assign(finalist_core=0.30)
    with pytest.raises(ValueError, match="differs from verified source"):
        attach_verified_source_attribution(
            bad,
            {
                "_verified_source_prediction_path": str(source),
                "control_name": control,
            },
            candidate,
        )


def test_finish_attribution_rejects_tampered_compact_control_core(tmp_path) -> None:
    candidate = "complete__ridge__total__w0__annual__a1"
    control = "current_architecture_pt_v2_approx_control"
    source = tmp_path / "source.parquet"
    rows = []
    for name, prediction in ((candidate, 0.25), (control, 0.10)):
        rows.append(
            {
                "candidate": name,
                "canonical_event_key": "pga:R2025009",
                "dg_id_key": "7",
                "prediction": prediction,
                "history_rounds": 42,
                "days_since_last_round": 7,
                "reliability_ok": True,
                "cross_tour_fallback_reason": "",
                "img_history_rounds": 20,
                "img_history_coverage": 0.8,
                "prior_event_withdrawal": False,
            }
        )
    pd.DataFrame(rows).to_parquet(source, index=False)
    compact = pd.DataFrame(
        {
            "event_key": ["pga:R2025009"],
            "dg_id": [7],
            "finalist_core": [0.25],
            "control_core": [0.11],
        }
    )
    with pytest.raises(ValueError, match="compact control core differs"):
        attach_verified_source_attribution(
            compact,
            {
                "_verified_source_prediction_path": str(source),
                "control_name": control,
            },
            candidate,
        )

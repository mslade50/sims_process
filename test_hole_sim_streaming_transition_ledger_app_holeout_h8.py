from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_streaming_transition_ledger_app_holeout_h8 as holeout


def _result() -> holeout.streaming.StreamingHoleResult:
    size = len(holeout.PARTICLE_IDS)
    endpoint = np.column_stack(
        [np.full(size, 210.0, dtype=float), np.zeros(size, dtype=float)]
    )
    return holeout.streaming.StreamingHoleResult(
        particle_id=holeout.PARTICLE_IDS.copy(),
        hole_strokes=np.ones(size, dtype=np.int16),
        terminal=np.ones(size, dtype=bool),
        status_code=np.full(
            size, holeout.streaming.STATUS_TERMINAL, dtype=np.int8
        ),
        hazard_transitions=np.zeros(size, dtype=np.int16),
        penalty_strokes=np.zeros(size, dtype=np.int16),
        transition_count=np.zeros(size, dtype=np.int16),
        return_to_app_count=np.zeros(size, dtype=np.int16),
        tee_atom_id=np.full(size, "not_applicable_par3", dtype=object),
        app_atom_id=np.asarray(
            [f"spatial-{particle_id}" for particle_id in holeout.PARTICLE_IDS],
            dtype=object,
        ),
        initial_endpoint=endpoint.copy(),
        initial_lie=np.full(size, holeout.streaming.LIE_HOLE, dtype=np.int8),
        initial_remaining_yd=np.zeros(size, dtype=float),
        initial_strokes=np.ones(size, dtype=np.int16),
        initial_status_code=np.full(
            size, holeout.streaming.STATUS_TERMINAL, dtype=np.int8
        ),
        final_endpoint=endpoint.copy(),
        final_lie=np.full(size, holeout.streaming.LIE_HOLE, dtype=np.int8),
        final_remaining_yd=np.zeros(size, dtype=float),
    )


def _ledger() -> pd.DataFrame:
    uniforms = holeout.regenerate_initial_app_uniforms()
    size = len(holeout.PARTICLE_IDS)
    start_value = np.full(size, 2.35, dtype=float)
    return pd.DataFrame(
        {
            "particle_id": holeout.PARTICLE_IDS,
            "atom_sequence": np.zeros(size, dtype=np.int16),
            "arm": np.full(size, holeout.ARM, dtype=object),
            "category": np.full(size, "app", dtype=object),
            "support_key": np.full(
                size, holeout.EXPECTED_SUPPORT_KEY, dtype=object
            ),
            "atom_id": np.full(
                size, holeout.EXPECTED_TERMINAL_ATOM_ID, dtype=object
            ),
            "selected_spatial_proposal_atom_id": np.asarray(
                [f"spatial-{particle_id}" for particle_id in holeout.PARTICLE_IDS],
                dtype=object,
            ),
            "donor_event_key": np.full(
                size, holeout.EXPECTED_DONOR_EVENT_KEY, dtype=object
            ),
            "donor_player_id": np.full(
                size, holeout.EXPECTED_DONOR_PLAYER_ID, dtype=object
            ),
            "donor_player_name": np.full(
                size, holeout.EXPECTED_DONOR_PLAYER_NAME, dtype=object
            ),
            "donor_round_no": np.ones(size, dtype=np.int16),
            "donor_hole_no": np.full(size, 8, dtype=np.int16),
            "donor_stroke_no": np.ones(size, dtype=np.int16),
            "donor_ability_sg": np.full(
                size, holeout.EXPECTED_DONOR_ABILITY_SG, dtype=float
            ),
            "donor_ability_available": np.ones(size, dtype=bool),
            "start_lie": np.full(size, "tee", dtype=object),
            "end_lie": np.full(size, "hole", dtype=object),
            "causal_strokes": np.ones(size, dtype=np.int16),
            "penalty_strokes": np.zeros(size, dtype=np.int16),
            "stroke_cost": np.ones(size, dtype=np.int16),
            "terminal": np.ones(size, dtype=bool),
            "holed": np.ones(size, dtype=bool),
            "legacy_holeout_decision": np.ones(size, dtype=bool),
            "declared_app_base_holeout_probability": np.full(
                size, holeout.EXPECTED_HOLEOUT_PROBABILITY, dtype=float
            ),
            "rng_shot_transition_index": np.full(size, -1, dtype=np.int16),
            "rng_index_uniform": uniforms[
                "expected_rng_app_index_uniform"
            ].to_numpy(float),
            "rng_lie_uniform": uniforms[
                "expected_rng_app_lie_uniform"
            ].to_numpy(float),
            "rng_holeout_uniform": holeout.EXPECTED_HOLEOUT_UNIFORMS,
            "rng_holeout_residual_uniform": (
                holeout.EXPECTED_HOLEOUT_UNIFORMS
                / holeout.EXPECTED_HOLEOUT_PROBABILITY
            ),
            "start_reference_value": start_value,
            "end_reference_value": np.zeros(size, dtype=float),
            "fixed_reference_reward": start_value - 1.0,
        }
    )


def _execution() -> holeout.ledger_v6.TracedHoleExecution:
    result = _result()
    audit = pd.DataFrame(
        {
            "particle_id": holeout.PARTICLE_IDS,
            "atoms": np.ones(len(holeout.PARTICLE_IDS), dtype=np.int16),
            "hole_strokes": np.ones(len(holeout.PARTICLE_IDS), dtype=np.int16),
            "atom_stroke_cost": np.ones(
                len(holeout.PARTICLE_IDS), dtype=np.int16
            ),
            "stroke_identity_residual": np.zeros(
                len(holeout.PARTICLE_IDS), dtype=np.int16
            ),
            "reward_sum": np.full(len(holeout.PARTICLE_IDS), 1.35),
            "expected_reward": np.full(len(holeout.PARTICLE_IDS), 1.35),
            "reward_telescope_residual": np.zeros(len(holeout.PARTICLE_IDS)),
            "terminal_atoms": np.ones(
                len(holeout.PARTICLE_IDS), dtype=np.int16
            ),
        }
    )
    return holeout.ledger_v6.TracedHoleExecution(
        result=result,
        diagnostics=holeout.streaming.KernelDiagnostics(),
        ledger=_ledger(),
        telescope_audit=audit,
    )


def _proposal() -> tuple[pd.DataFrame, pd.DataFrame]:
    atoms = pd.DataFrame(
        {
            "arm": np.full(holeout.EXPECTED_SUPPORT_ATOMS, holeout.ARM),
            "support_key": np.full(
                holeout.EXPECTED_SUPPORT_ATOMS,
                holeout.EXPECTED_SUPPORT_KEY,
            ),
            "proposal_row_order": np.arange(
                holeout.EXPECTED_SUPPORT_ATOMS, dtype=np.int32
            ),
            "atom_id": np.asarray(
                [f"spatial-{index}" for index in range(903)]
                + [holeout.EXPECTED_TERMINAL_ATOM_ID],
                dtype=object,
            ),
            "source_shot_key": np.asarray(
                [f"spatial-{index}" for index in range(903)]
                + [holeout.EXPECTED_TERMINAL_ATOM_ID],
                dtype=object,
            ),
            "category": np.full(holeout.EXPECTED_SUPPORT_ATOMS, "app"),
            "atom_kind": np.asarray(
                ["app_spatial"] * 903 + ["app_holeout"], dtype=object
            ),
            "terminal": np.asarray([False] * 903 + [True]),
            "holed": np.asarray([False] * 903 + [True]),
            "donor_event_key": np.asarray(
                [""] * 903 + [holeout.EXPECTED_DONOR_EVENT_KEY], dtype=object
            ),
            "donor_player_id": np.asarray(
                [""] * 903 + [holeout.EXPECTED_DONOR_PLAYER_ID], dtype=object
            ),
            "donor_player_name": np.asarray(
                [""] * 903 + [holeout.EXPECTED_DONOR_PLAYER_NAME], dtype=object
            ),
            "donor_round_no": np.asarray([-1] * 903 + [1], dtype=np.int16),
            "donor_hole_no": np.asarray([-1] * 903 + [8], dtype=np.int16),
            "donor_stroke_no": np.asarray([-1] * 903 + [1], dtype=np.int16),
            "proposal_branch": np.asarray(
                ["spatial"] * 903 + ["terminal"], dtype=object
            ),
            "donor_ability_available": np.asarray([False] * 903 + [True]),
            "donor_ability_sg": np.asarray(
                [np.nan] * 903 + [holeout.EXPECTED_DONOR_ABILITY_SG]
            ),
            "base_hierarchy_mass": np.asarray(
                [0.998 / 903] * 903 + [0.002]
            ),
            "spatial_model_weight": np.ones(904),
            "quality_base_mass": np.asarray([0.998 / 903] * 903 + [0.002]),
            "proposal_probability": np.asarray([0.998 / 903] * 903 + [0.002]),
            "proposal_cdf_lower": np.asarray(
                [0.0] * 903 + [holeout.EXPECTED_PROPOSAL_CDF_LOWER]
            ),
            "proposal_cdf_upper": np.asarray([1.0] * 904),
            "branch_total_mass": np.asarray([0.998] * 903 + [0.002]),
            "branch_conditional_probability": np.asarray(
                [1.0 / 903] * 903 + [1.0]
            ),
            "branch_cdf_lower": np.asarray([0.0] * 904),
            "branch_cdf_upper": np.asarray([1.0] * 904),
            "declared_app_base_holeout_probability": np.full(904, 0.002),
            "proposal_support_measure_sha256": np.full(904, "measure"),
            "proposal_support_ability_sha256": np.full(904, "ability"),
        }
    )
    inventory = pd.DataFrame(
        [
            {
                "arm": holeout.ARM,
                "support_key": holeout.EXPECTED_SUPPORT_KEY,
                "category": "app",
                "atoms": 904,
                "spatial_atoms": 903,
                "terminal_atoms": 1,
            }
        ]
    )
    return atoms, inventory


def test_exact_counter_ids_and_holeout_uniforms_are_pinned() -> None:
    replay = holeout.regenerate_initial_app_uniforms()
    assert replay["particle_id"].tolist() == [345, 1915, 2461, 2573, 3382]
    assert np.array_equal(
        replay["expected_rng_app_holeout_uniform"].to_numpy(float),
        holeout.EXPECTED_HOLEOUT_UNIFORMS,
    )
    assert np.array_equal(
        replay["expected_rng_holeout_residual_uniform"].to_numpy(float),
        holeout.EXPECTED_HOLEOUT_UNIFORMS / 0.002,
    )
    assert (replay["expected_rng_app_holeout_uniform"] < 0.002).all()
    with pytest.raises(ValueError, match="requires exactly"):
        holeout.regenerate_initial_app_uniforms(holeout.PARTICLE_IDS[:-1])


def test_real_record_selection_is_unique_par3_bucket14() -> None:
    grid = pd.DataFrame(
        [
            {
                "event_key": holeout.TARGET_EVENT_KEY,
                "source_player_id": holeout.TARGET_PLAYER_ID,
                "round_no": 1,
                "hole_no": 8,
                "par": 3,
                "forecast_pin_long_yd": 210.0,
            }
        ]
    )
    record = holeout._select_exact_record(grid)
    assert int(record.par) == 3
    duplicate = pd.concat([grid, grid], ignore_index=True)
    with pytest.raises(ValueError, match="absent or duplicated"):
        holeout._select_exact_record(duplicate)
    wrong_bucket = grid.assign(forecast_pin_long_yd=180.0)
    with pytest.raises(AssertionError, match="bucket changed"):
        holeout._select_exact_record(wrong_bucket)


def test_traced_then_plain_pair_is_exact_and_support_closed() -> None:
    calls: list[str] = []
    repository = SimpleNamespace(
        supports={holeout.EXPECTED_SUPPORT_KEY: object()},
        explicit_app_supports={holeout.EXPECTED_SUPPORT_KEY: object()},
    )

    class Traced:
        def simulate_hole_with_ledger(self, **kwargs):
            calls.append("traced")
            assert np.array_equal(kwargs["particle_ids"], holeout.PARTICLE_IDS)
            assert kwargs["total_particles"] == 50_000
            assert kwargs["rng_mode"] == "counter"
            return _execution()

    class Plain:
        def simulate_hole(self, **kwargs):
            calls.append("plain_parent")
            assert calls == ["traced", "plain_parent"]
            return _result(), holeout.streaming.KernelDiagnostics()

    execution, plain, parity, discovery = holeout.run_traced_plain_pair(
        traced_chain=Traced(),
        plain_chain=Plain(),
        repository=repository,
        record=SimpleNamespace(),
    )
    assert execution.result.terminal.all() and plain.terminal.all()
    assert parity["exact"]
    assert discovery["exact_key_sets"]
    assert discovery["before"] == discovery["after"]
    assert calls == ["traced", "plain_parent"]

    class DiscoveringPlain:
        def simulate_hole(self, **kwargs):
            repository.supports["late"] = object()
            return _result(), holeout.streaming.KernelDiagnostics()

    with pytest.raises(AssertionError, match="discovered support"):
        holeout.run_traced_plain_pair(
            traced_chain=Traced(),
            plain_chain=DiscoveringPlain(),
            repository=repository,
            record=SimpleNamespace(),
        )


def test_terminal_ledger_and_counter_replay_fail_closed() -> None:
    ledger = _ledger()
    audit = holeout.validate_exact_terminal_ledger(ledger)
    assert audit["rows"] == 5
    assert audit["all_expected_terminal_atom"]
    replay = holeout.build_counter_replay(ledger)
    exact_columns = [name for name in replay if name.endswith("_exact")]
    residual_columns = [name for name in replay if name.endswith("_residual")]
    assert len(exact_columns) == 4
    assert len(residual_columns) == 4
    assert replay[exact_columns].all().all()
    assert replay[residual_columns].eq(0).all().all()

    broken = ledger.copy()
    broken.loc[broken.index[0], "atom_id"] = "wrong"
    with pytest.raises(AssertionError, match="atom_id"):
        holeout.validate_exact_terminal_ledger(broken)
    broken = ledger.copy()
    broken.loc[broken.index[0], "rng_holeout_residual_uniform"] += 1e-15
    with pytest.raises(AssertionError, match="residual uniforms"):
        holeout.validate_exact_terminal_ledger(broken)


def test_terminal_proposal_mass_cdf_identity_and_ability_are_exact() -> None:
    atoms, inventory = _proposal()
    terminal = holeout.validate_exact_terminal_proposal(atoms, inventory)
    assert terminal["atom_id"] == holeout.EXPECTED_TERMINAL_ATOM_ID
    assert terminal["quality_base_mass"] == 0.002
    assert terminal["proposal_cdf_lower"] == 0.9979999999999989
    assert terminal["branch_cdf_lower"] == 0.0
    assert terminal["branch_cdf_upper"] == 1.0
    assert terminal["donor_ability_sg"] == 0.65

    broken = atoms.copy()
    broken.loc[broken["terminal"], "quality_base_mass"] = 0.001
    with pytest.raises(AssertionError, match="mass/CDF"):
        holeout.validate_exact_terminal_proposal(broken, inventory)


def test_result_differential_and_selection_evidence_are_exact() -> None:
    result = _result()
    differential = holeout.build_result_differential(result, result)
    assert len(differential) == len(result.__dataclass_fields__)
    assert differential["exact"].all()
    assert differential["mismatch_count"].eq(0).all()

    atoms, inventory = _proposal()
    terminal = holeout.validate_exact_terminal_proposal(atoms, inventory)
    evidence = holeout.build_selection_evidence(_ledger(), terminal, result)
    assert len(evidence) == 5
    assert evidence["residual_inside_terminal_branch_cdf"].all()
    assert evidence["hole_strokes"].eq(1).all()
    assert evidence["atom_id"].eq(holeout.EXPECTED_TERMINAL_ATOM_ID).all()

    broken = _result()
    broken.hole_strokes[0] = 2
    with pytest.raises(AssertionError, match="score changed"):
        holeout.build_selection_evidence(_ledger(), terminal, broken)


def test_output_index_detects_extra_or_changed_payload(tmp_path: Path) -> None:
    for filename in holeout.OUTPUT_PAYLOAD_FILES:
        (tmp_path / filename).write_text(f"{filename}\n", encoding="utf-8")
    holeout.ledger_v6._output_index(tmp_path)
    audit = holeout._validate_output_index(tmp_path)
    assert audit == {
        "contract": "non_circular_output_sha256_v1",
        "files": len(holeout.OUTPUT_PAYLOAD_FILES),
        "all_exact": True,
    }
    (tmp_path / "extra.txt").write_text("late\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="file set changed"):
        holeout._validate_output_index(tmp_path)


def test_runner_is_frozen_bounded_and_prepares_authorities_once() -> None:
    assert holeout._sha256(Path(holeout.ledger_v6.__file__)) == (
        holeout.FROZEN_V6_SOURCE_SHA256
    )
    holeout._assert_frozen_v6_source()
    dependencies = holeout._dependency_paths()
    assert set(holeout.streaming.dependency_paths()).issubset(dependencies)
    assert {
        "frozen_v6_ledger_source",
        "streaming_source",
        "traced_population_source",
        "fixed_reference_source",
        "chain_adapter_source",
    }.issubset(dependencies)
    assert holeout.streaming.validate_frozen_dependencies()
    assert holeout.PARTICLE_IDS.tolist() == [345, 1915, 2461, 2573, 3382]
    assert holeout.ARM == "raw"
    assert holeout.ROUND_NO == 1 and holeout.HOLE_NO == 8

    source = inspect.getsource(holeout.build_real_h8_holeout_diagnostic)
    for call in (
        "streaming.prepare_streaming_event(",
        "build_strict_prior_course_trace_bundle(",
        "ReferenceValueModel.load(",
        "load_strict_bridge(",
        "materialize_proposal_support_contract(",
    ):
        assert source.count(call) == 1, call
    assert "np.arange" not in source
    assert "particle_search_performed\": False" in source
    assert source.index("run_traced_plain_pair(") < source.index(
        "materialize_proposal_support_contract("
    )


def test_output_manifest_contract_lists_only_bounded_diagnostic_files() -> None:
    source = inspect.getsource(holeout.build_real_h8_holeout_diagnostic)
    expected = {
        "explicit_holeout_transition_ledger.parquet",
        "particle_telescope_audit.parquet",
        "proposal_support_atoms.parquet",
        "proposal_support_inventory.parquet",
        "traced_plain_results.parquet",
        "traced_parent_differential.parquet",
        "counter_uniform_replay.parquet",
        "explicit_holeout_selection.parquet",
    }
    assert expected.issubset(set(holeout.OUTPUT_PAYLOAD_FILES))
    for filename in expected:
        assert json.dumps(filename) in source or f'"{filename}"' in source
    assert "anchor_enabled\": False" in source
    assert "production_authorized\": False" in source

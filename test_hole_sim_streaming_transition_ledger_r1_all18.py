from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_streaming_transition_ledger_r1_all18 as all18


def _records() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            event_key=all18.TARGET_EVENT_KEY,
            round_no=all18.ROUND_NO,
            hole_no=hole,
            source_player_id=all18.TARGET_PLAYER_ID,
            par=3 if hole in {3, 8, 11, 16} else 4,
        )
        for hole in all18.HOLES
    )


def _result(
    *,
    hole_no: int,
    hazard_particle: int | None = None,
) -> all18.streaming.StreamingHoleResult:
    size = all18.PARTICLES
    particle_id = all18.PARTICLE_IDS.copy()
    hazard = np.zeros(size, dtype=np.int16)
    penalty = np.zeros(size, dtype=np.int16)
    if hazard_particle is not None:
        hazard[int(hazard_particle)] = 1
        penalty[int(hazard_particle)] = 1
    endpoint = np.column_stack(
        [np.full(size, float(hole_no)), np.zeros(size, dtype=float)]
    )
    return all18.streaming.StreamingHoleResult(
        particle_id=particle_id,
        hole_strokes=np.full(size, 4, dtype=np.int16) + penalty,
        terminal=np.ones(size, dtype=bool),
        status_code=np.full(
            size, all18.streaming.STATUS_TERMINAL, dtype=np.int8
        ),
        hazard_transitions=hazard,
        penalty_strokes=penalty,
        transition_count=np.full(size, 2, dtype=np.int16),
        return_to_app_count=np.zeros(size, dtype=np.int16),
        tee_atom_id=np.full(size, f"tee-{hole_no}", dtype=object),
        app_atom_id=np.full(size, f"app-{hole_no}", dtype=object),
        initial_endpoint=endpoint.copy(),
        initial_lie=np.full(size, all18.streaming.LIE_GREEN, dtype=np.int8),
        initial_remaining_yd=np.full(size, 10.0, dtype=float),
        initial_strokes=np.full(size, 2, dtype=np.int16),
        initial_status_code=np.full(
            size, all18.streaming.STATUS_ACTIVE, dtype=np.int8
        ),
        final_endpoint=endpoint.copy(),
        final_lie=np.full(size, all18.streaming.LIE_HOLE, dtype=np.int8),
        final_remaining_yd=np.zeros(size, dtype=float),
    )


def _execution(
    *,
    hole_no: int,
    arm: str,
    include_rare_paths: bool = False,
) -> all18.ledger_v6.TracedHoleExecution:
    rare = include_rare_paths and hole_no == 3 and arm == "raw"
    result = _result(hole_no=hole_no, hazard_particle=0 if rare else None)
    size = all18.PARTICLES
    ledger = pd.DataFrame(
        {
            "particle_id": all18.PARTICLE_IDS,
            "atom_sequence": np.zeros(size, dtype=np.int16),
            "arm": np.full(size, arm, dtype=object),
            "category": np.full(
                size,
                "app" if hole_no in {3, 8, 11, 16} else "ott",
                dtype=object,
            ),
            "terminal": np.ones(size, dtype=bool),
            "penalty_strokes": np.zeros(size, dtype=np.int16),
            "stroke_cost": result.hole_strokes.copy(),
            "rng_shot_transition_index": np.full(size, -1, dtype=np.int16),
            "hazard_resolution_status": np.full(size, "none", dtype=object),
            "legacy_holeout_decision": np.zeros(size, dtype=bool),
        }
    )
    if rare:
        ledger.loc[0, "hazard_resolution_status"] = "resolved_complete_atom"
        ledger.loc[0, "penalty_strokes"] = 1
        ledger.loc[0, "legacy_holeout_decision"] = True
    audit = pd.DataFrame(
        {
            "particle_id": all18.PARTICLE_IDS,
            "atoms": np.ones(size, dtype=np.int16),
            "hole_strokes": result.hole_strokes,
            "atom_stroke_cost": result.hole_strokes,
            "stroke_identity_residual": np.zeros(size, dtype=np.int16),
            "reward_sum": np.zeros(size, dtype=float),
            "expected_reward": np.zeros(size, dtype=float),
            "reward_telescope_residual": np.zeros(size, dtype=float),
            "terminal_atoms": np.ones(size, dtype=np.int16),
        }
    )
    return all18.ledger_v6.TracedHoleExecution(
        result=result,
        diagnostics=all18.streaming.KernelDiagnostics(),
        ledger=ledger,
        telescope_audit=audit,
    )


def _traced_cases(*, include_rare_paths: bool = False):
    cases = []
    for record in _records():
        for arm in all18.ARMS:
            key = all18.CaseKey(all18.ROUND_NO, record.hole_no, arm)
            cases.append(
                all18.TracedCase(
                    key=key,
                    record=record,
                    execution=_execution(
                        hole_no=record.hole_no,
                        arm=arm,
                        include_rare_paths=include_rare_paths,
                    ),
                )
            )
    return tuple(cases)


def test_exact_grid_axis_rejects_missing_and_duplicate_holes() -> None:
    rows = [vars(record) for record in _records()]
    grid = pd.DataFrame(rows)
    selected = all18._records_for_exact_axis(grid)
    assert tuple(record.hole_no for record in selected) == all18.HOLES

    with pytest.raises(ValueError, match="exactly 18"):
        all18._records_for_exact_axis(grid.loc[grid["hole_no"].ne(18)])

    duplicate = pd.concat([grid, grid.loc[grid["hole_no"].eq(1)]])
    with pytest.raises(ValueError, match="exactly 18|duplicate"):
        all18._records_for_exact_axis(duplicate)


def test_all_traced_cases_precede_parent_and_parent_adds_no_supports() -> None:
    calls: list[tuple[str, int, str]] = []

    class TraceChain:
        def simulate_hole_with_ledger(self, *, record, arm, **kwargs):
            assert np.array_equal(kwargs["particle_ids"], all18.PARTICLE_IDS)
            assert kwargs["rng_mode"] == "counter"
            calls.append(("traced", int(record.hole_no), str(arm)))
            return _execution(hole_no=int(record.hole_no), arm=str(arm))

    class PlainChain:
        def simulate_hole(self, *, record, arm, **kwargs):
            assert np.array_equal(kwargs["particle_ids"], all18.PARTICLE_IDS)
            calls.append(("plain", int(record.hole_no), str(arm)))
            return _result(hole_no=int(record.hole_no)), all18.streaming.KernelDiagnostics()

    repositories = {
        arm: SimpleNamespace(
            supports={f"existing-{arm}": object()},
            explicit_app_supports={f"app-{arm}": object()},
        )
        for arm in all18.ARMS
    }
    traced = all18.run_traced_cases(TraceChain(), _records())
    plain, parity, discovery = all18.run_plain_parent_cases(
        PlainChain(), traced, repositories
    )
    assert len(traced) == len(plain) == len(parity) == 36
    assert [phase for phase, _, _ in calls[:36]] == ["traced"] * 36
    assert [phase for phase, _, _ in calls[36:]] == ["plain"] * 36
    assert discovery["exact_key_sets"]
    assert discovery["before"] == discovery["after"]
    assert all(row["exact"] for row in parity)


def test_parent_support_discovery_fails_closed() -> None:
    traced = _traced_cases()
    repositories = {
        arm: SimpleNamespace(supports={}, explicit_app_supports={})
        for arm in all18.ARMS
    }

    class DiscoveringPlainChain:
        def simulate_hole(self, *, record, arm, **kwargs):
            if int(record.hole_no) == 18 and str(arm) == "normalized":
                repositories[str(arm)].supports["late"] = object()
            return _result(hole_no=int(record.hole_no)), all18.streaming.KernelDiagnostics()

    with pytest.raises(AssertionError, match="discovered support"):
        all18.run_plain_parent_cases(
            DiscoveringPlainChain(), traced, repositories
        )


def test_assembled_frames_have_unfiltered_full_axes_and_route_coverage() -> None:
    traced = _traced_cases(include_rare_paths=True)
    plain = tuple(
        all18.PlainCase(key=case.key, result=case.execution.result)
        for case in traced
    )
    ledger, audit, results = all18.assemble_case_frames(
        traced,
        plain,
        event_key="pga:R2026034",
        event_start=pd.Timestamp("2026-06-25", tz="UTC"),
    )
    axis = all18.validate_complete_axes(ledger, audit, results)
    assert axis == {
        "expected_cases": 36,
        "traced_paths": 4608,
        "audit_rows": 4608,
        "saved_result_rows": 9216,
        "particle_ids_exact": True,
        "path_filtering": False,
        "all_terminal": True,
        "score_identity_exact": True,
    }
    coverage = all18.summarize_hole_coverage(traced, ledger)
    assert len(coverage) == 18
    assert coverage[0]["initializer_route"] == "par4_5_initial_tee_then_app"
    assert coverage[2]["initializer_route"] == "par3_initial_app"
    totals = all18._coverage_totals(coverage)
    assert totals["hazard_transitions"] == 1
    assert totals["penalty_strokes"] == 1
    assert totals["resolved_spatial_hazard_atoms"] == 1
    assert totals["explicit_app_holeouts_selected"] == 1
    empirical = all18.summarize_empirical_coverage(coverage)
    assert empirical["status"] == "GO"
    assert empirical["complete"]

    zero_rare = []
    for row in coverage:
        copied = {**row, "arms": {}}
        for arm, values in row["arms"].items():
            copied["arms"][arm] = {
                **values,
                "hazard_transitions": 0,
                "penalty_strokes": 0,
                "resolved_spatial_hazard_atoms": 0,
                "collapsed_population_penalty_atoms": 0,
                "explicit_app_holeouts_selected": 0,
            }
        zero_rare.append(copied)
    hold = all18.summarize_empirical_coverage(zero_rare)
    assert hold["status"] == "HOLD"
    assert not hold["complete"]
    assert hold["gates"]["par3_initializer_cases_present"]
    assert not hold["gates"]["resolved_spatial_hazard_atoms_present"]

    broken_penalty = ledger.copy()
    broken_penalty.loc[
        broken_penalty["hole_no"].eq(3)
        & broken_penalty["arm"].eq("raw")
        & broken_penalty["particle_id"].eq(0),
        "penalty_strokes",
    ] = 0
    with pytest.raises(AssertionError, match="Penalty-stroke accounting"):
        all18.summarize_hole_coverage(traced, broken_penalty)


def test_runner_is_pinned_to_the_audited_v6_seam_and_mechanics_axes() -> None:
    assert all18._sha256(Path(all18.ledger_v6.__file__)) == (
        all18.FROZEN_V6_SOURCE_SHA256
    )
    all18._assert_frozen_v6_source()
    dependencies = all18._dependency_paths()
    assert set(all18.streaming.dependency_paths()).issubset(dependencies)
    assert {
        "frozen_v6_ledger_source",
        "streaming_source",
        "traced_population_source",
        "fixed_reference_source",
        "chain_adapter_source",
        "production_target_source",
    }.issubset(dependencies)
    assert all18.streaming.validate_frozen_dependencies()
    assert all18.HOLES == tuple(range(1, 19))
    assert all18.ARMS == ("raw", "normalized")
    assert all18.PARTICLES == 128
    assert all18.TARGET_PLAYER_ID == "33204"
    assert all18.TARGET_EVENT_KEY == "pga:R2026034"
    assert all18.DEFAULT_OUTPUT_DIR.as_posix().endswith(
        "hole_sim_streaming_transition_ledger_lowry_r1_all18_128_v1"
    )


def test_real_builder_prepares_authorities_once_and_orders_execution_phases() -> None:
    source = inspect.getsource(all18.build_r1_all18)
    for call in (
        "streaming.prepare_streaming_event(",
        "build_strict_prior_course_trace_bundle(",
        "ReferenceValueModel.load(",
        "load_strict_bridge(",
        "load_validated_production_targets(",
        "materialize_proposal_support_contract(",
    ):
        assert source.count(call) == 1, call
    traced_at = source.index("traced_cases = run_traced_cases(")
    plain_constructor_at = source.index("plain_chain = streaming.StreamingShotChain(")
    plain_at = source.index("run_plain_parent_cases(")
    assert traced_at < plain_constructor_at < plain_at

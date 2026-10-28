from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_real_app_h8_trace_selection_contract as contract


SOURCE = ARCHIVE / "hole_sim_real_app_h8_trace_selection_contract.py"


def context(**changes: object) -> contract.H8TraceSelectionContext:
    values = {
        "event_key": contract.TARGET_EVENT_KEY,
        "player_id": contract.TARGET_PLAYER_ID,
        "arm": contract.TARGET_ARM,
        "round_no": contract.TARGET_ROUND_NO,
        "hole_no": contract.TARGET_HOLE_NO,
        "start_lie": contract.TARGET_START_LIE,
        "distance_bucket": contract.TARGET_DISTANCE_BUCKET,
        "transition_index": contract.TARGET_TRANSITION_INDEX,
        "particle_id": 17,
        "support_key": contract.TARGET_SUPPORT_KEY,
    }
    values.update(changes)
    return contract.H8TraceSelectionContext(**values)


def uniforms() -> contract.H8SemanticUniforms:
    return contract.H8SemanticUniforms(0.1, 0.2, 0.3, 0.4)


def parameters() -> contract.H8TraceWeightingParameters:
    return contract.H8TraceWeightingParameters(
        transfer_beta=(0.0, 0.4, 0.0, 0.0),
        target_ability_sg=(0.1, 0.2, 0.3, 0.4),
        lambda_correction=(0.0, -0.1, 0.0, 0.0),
    )


def request() -> contract.H8TraceSelectionRequest:
    return contract.H8TraceSelectionRequest(context(), parameters(), uniforms())


def physical(**changes: object) -> contract.CompletePhysicalAppSelection:
    values = {
        "contract": contract.PHYSICAL_SELECTION_CONTRACT,
        "support_key": contract.TARGET_SUPPORT_KEY,
        "atom_index": 3,
        "complete_row_order": 9,
        "atom_id": "a" * 64,
        "proposal_row_order": 2,
        "surface_class_index": 0,
        "hazard_donor_order": -1,
        "start_lie": "tee",
        "start_distance_yd": contract.TARGET_START_DISTANCE_YD,
        "start_long_yd": 0.0,
        "start_lat_yd": 0.0,
        "start_state_key": contract.TARGET_START_STATE_KEY,
        "end_lie": "green",
        "end_distance_yd": 10.0,
        "end_long_yd": 194.5,
        "end_lat_yd": 1.0,
        "causal_strokes": 1,
        "penalty_strokes": 0,
        "stroke_cost": 1,
        "hazard_resolution_status": "none",
        "hazard_donor_key": "not_applicable",
        "donor_event_key": "pga:R2025001",
        "donor_player_id": "100",
        "donor_player_name": "synthetic donor",
        "donor_round_no": 1,
        "donor_hole_no": 8,
        "donor_stroke_no": 1,
        "start_reference_value": 3.0,
        "end_reference_value": 1.5,
        "fixed_reference_reward": 0.5,
        "reward_reference_descriptor": "synthetic",
        "reference_model_sha256": contract.REWARD_REFERENCE_SHA256,
        "reward_reference_putting_sha256": "c" * 64,
        "terminal": False,
        "holed": False,
        "end_state_key": "green|10.000000yd",
        "next_support_key": "d" * 64,
    }
    values.update(changes)
    return contract.CompletePhysicalAppSelection(**values)


def test_public_status_and_capability_surface_remain_hold() -> None:
    assert contract.INTEGRATION_STATUS.startswith("HOLD_")
    assert contract.PINNED_CAPABILITIES["selection"] is True
    for name in ("callback", "chain", "anchor", "mean_solve", "simulation", "production"):
        assert contract.PINNED_CAPABILITIES[name] is False
    assert contract.PINNED_CAPABILITIES["research_only"] is True


def test_exact_raw_lowry_scope_is_immutable() -> None:
    value = context()
    assert value.support_key == "app|raw|R1|H8|tee|B14|33204"
    with pytest.raises(FrozenInstanceError):
        value.arm = "normalized"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("event_key", "pga:R2026035"),
        ("player_id", "other"),
        ("arm", "normalized"),
        ("round_no", 2),
        ("hole_no", 9),
        ("start_lie", "fairway"),
        ("distance_bucket", 13),
        ("transition_index", 0),
        ("support_key", "app|normalized|R1|H8|tee|B14|33204"),
        ("particle_id", 128),
    ],
)
def test_every_scope_escape_fails_closed(name: str, value: object) -> None:
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        context(**{name: value})


@pytest.mark.parametrize("value", [0, True, 0.0, "17"])
def test_particle_id_requires_exact_int(value: object) -> None:
    if value == 0 and type(value) is int:
        assert context(particle_id=value).particle_id == 0
    else:
        with pytest.raises(contract.H8TraceSelectionUnavailable):
            context(particle_id=value)


def test_four_uniform_order_and_counter_mapping_are_exact() -> None:
    bound = contract.bind_semantic_uniforms(uniforms())
    assert contract.SEMANTIC_UNIFORM_ORDER == (
        "holeout", "proposal", "surface", "hazard"
    )
    assert tuple(item.value for item in bound) == (0.1, 0.2, 0.3, 0.4)
    assert tuple((item.seed, item.transition_index, item.stream) for item in bound) == (
        (20260824, -1, "initial_app_holeout"),
        (20260824, -1, "initial_app_index"),
        (20260824, -1, "initial_app_lie"),
        (20261324, 0, "hazard"),
    )
    assert {item.counter_contract for item in bound} == {
        "blake2b_splitmix64_counter_v1"
    }


def test_weighting_request_matches_the_admitted_session_shape() -> None:
    value = parameters()
    assert value.reward_reference_sha256 == contract.REWARD_REFERENCE_SHA256
    assert value.max_abs_log_weight == 30.0
    with pytest.raises(FrozenInstanceError):
        value.lambda_correction = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        replace(value, transfer_beta=(0.0, -0.1, 0.0, 0.0))
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        replace(value, target_ability_sg=(0.0, 0.0, 0.0, 1))


@pytest.mark.parametrize("value", [-0.1, 1.0, float("nan"), float("inf")])
def test_uniform_bounds_are_half_open_and_finite(value: float) -> None:
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        contract.H8SemanticUniforms(value, 0.2, 0.3, 0.4)
    assert contract.H8SemanticUniforms(0.0, 0.0, 0.0, 0.0).holeout == 0.0


@pytest.mark.parametrize("value", [0, True, "0.2", None])
def test_uniforms_require_exact_float_type(value: object) -> None:
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        contract.H8SemanticUniforms(value, 0.2, 0.3, 0.4)


def test_every_pin_is_closed_and_hash_drift_rejects() -> None:
    assert tuple(contract.PINNED_HASHES) == tuple(
        field.name for field in __import__("dataclasses").fields(contract.H8TraceSelectionPins)
    )
    assert all(len(value) == 64 for value in contract.PINNED_HASHES.values())
    with pytest.raises(contract.H8TraceSelectionUnavailable, match="dependency changed"):
        replace(contract.EXPECTED_PINS, parent_trace_manifest_sha256="f" * 64)
    with pytest.raises(TypeError):
        contract.PINNED_HASHES["authority_id"] = "f" * 64


def test_physical_schema_contains_every_state_ledger_and_reward_field() -> None:
    required = {
        "support_key", "start_lie", "start_distance_yd", "start_state_key",
        "end_lie", "end_distance_yd", "end_long_yd", "end_lat_yd",
        "causal_strokes", "penalty_strokes", "stroke_cost",
        "fixed_reference_reward", "start_reference_value", "end_reference_value",
        "reference_model_sha256",
        "donor_event_key", "donor_player_id", "donor_round_no", "donor_hole_no",
        "donor_stroke_no", "terminal", "holed", "end_state_key", "next_support_key",
    }
    assert required.issubset(contract.REQUIRED_PHYSICAL_PROJECTION_FIELDS)
    assert contract.validate_complete_physical_projection(physical()).atom_id == "a" * 64


def test_missing_physical_fields_fail_closed() -> None:
    public_selection_shape = SimpleNamespace(
        atom_id="a" * 64,
        end_state_key="green|10.000000yd",
        next_support_key="d" * 64,
        stroke_cost=1,
        terminal=False,
    )
    with pytest.raises(
        contract.H8TraceSelectionUnavailable,
        match="cannot satisfy physical projection",
    ) as error:
        contract.validate_complete_physical_projection(public_selection_shape)
    assert "end_distance_yd" in str(error.value)
    assert "fixed_reference_reward" in str(error.value)


def test_physical_projection_is_exact_tee_b14_start_and_support() -> None:
    value = physical()
    assert value.start_lie == "tee"
    assert value.start_distance_yd == 204.54171527805522
    assert value.start_state_key == "tee|204.541715yd"
    assert value.support_key == "app|raw|R1|H8|tee|B14|33204"
    for changes in (
        {"start_lie": "fairway"},
        {"start_distance_yd": 200.0},
        {"start_state_key": "tee|204.541716yd"},
        {"support_key": "app|raw|R1|H8|tee|B13|33204"},
        {"support_key": "app|normalized|R1|H8|tee|B14|33204"},
    ):
        with pytest.raises(contract.H8TraceSelectionUnavailable):
            physical(**changes)


@pytest.mark.parametrize(
    "name",
    [
        "contract", "support_key", "atom_id", "start_lie", "start_state_key",
        "end_lie", "hazard_resolution_status", "hazard_donor_key",
        "donor_event_key", "donor_player_id", "donor_player_name",
        "reward_reference_descriptor", "reference_model_sha256",
        "reward_reference_putting_sha256", "end_state_key", "next_support_key",
    ],
)
def test_every_physical_identity_requires_exact_string_type(name: str) -> None:
    class StringSubclass(str):
        pass

    valid = getattr(physical(), name)
    with pytest.raises(contract.H8TraceSelectionUnavailable):
        physical(**{name: StringSubclass(valid)})


@pytest.mark.parametrize("name", ["atom_id", "next_support_key"])
@pytest.mark.parametrize("value", ["A" * 64, "a" * 63, "g" * 64])
def test_hashed_physical_identity_keys_are_lowercase_sha256(
    name: str, value: str
) -> None:
    with pytest.raises(contract.H8TraceSelectionUnavailable, match="lowercase SHA-256"):
        physical(**{name: value})


def test_reference_model_is_exactly_bound_to_frozen_reward_reference() -> None:
    assert physical().reference_model_sha256 == contract.REWARD_REFERENCE_SHA256
    with pytest.raises(contract.H8TraceSelectionUnavailable, match="reference model changed"):
        physical(reference_model_sha256="e" * 64)


def test_fixed_reference_reward_uses_upstream_fixed_absolute_tolerance() -> None:
    assert contract.ATOM_REWARD_TOLERANCE == 5e-6
    assert physical(fixed_reference_reward=0.5 + 0.5e-6).fixed_reference_reward > 0.5
    with pytest.raises(
        contract.H8TraceSelectionUnavailable,
        match="exact stroke/value accounting",
    ):
        physical(fixed_reference_reward=0.5 + 5.1e-6)


def test_physical_cost_identity_fails_closed() -> None:
    with pytest.raises(contract.H8TraceSelectionUnavailable, match="stroke-cost"):
        physical(stroke_cost=2)


def test_terminal_projection_requires_every_upstream_sentinel() -> None:
    terminal = {
        "terminal": True,
        "holed": True,
        "end_lie": "hole",
        "end_distance_yd": 0.0,
        "end_state_key": contract.TERMINAL_STATE_KEY,
        "end_reference_value": 0.0,
        "fixed_reference_reward": 2.0,
        "next_support_key": "",
    }
    assert physical(**terminal).terminal is True
    sentinel_drifts = (
        {"end_lie": "green"},
        {"end_distance_yd": 0.1},
        {"end_state_key": "hole|0.000000yd"},
        {"end_reference_value": 0.1, "fixed_reference_reward": 1.9},
        {"next_support_key": "d" * 64},
        {"penalty_strokes": 1, "stroke_cost": 2, "fixed_reference_reward": 1.0},
    )
    for drift in sentinel_drifts:
        with pytest.raises(contract.H8TraceSelectionUnavailable, match="terminal"):
            physical(**{**terminal, **drift})


def test_nonterminal_projection_rejects_terminal_and_child_sentinels() -> None:
    for changes in (
        {"end_lie": "hole", "end_state_key": "hole|10.000000yd"},
        {"end_state_key": contract.TERMINAL_STATE_KEY},
        {"end_state_key": "green|10yd"},
        {"next_support_key": ""},
    ):
        with pytest.raises(contract.H8TraceSelectionUnavailable, match="nonterminal"):
            physical(**changes)


def test_adapter_never_calls_session_and_has_no_fallback() -> None:
    class SpySession:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @property
        def readiness(self):
            self.calls.append("readiness")
            raise AssertionError("readiness must not be touched")

        def select(self, *_args: object):
            self.calls.append("select")
            raise AssertionError("select must not be touched")

    session = SpySession()
    with pytest.raises(contract.H8TraceSelectionUnavailable, match="session.select was not called"):
        contract.select_complete_physical_app(request(), session)
    assert session.calls == []


def test_source_imports_only_stdlib_contract_dependencies() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports == {"__future__", "dataclasses", "math", "types", "typing"}
    forbidden = {"runner", "solver", "simulation", "production"}
    assert not (forbidden & imports)


def test_public_namespace_is_fixed_and_does_not_offer_execution() -> None:
    assert type(contract.__all__) is tuple
    assert set(contract.__all__) == {
        name for name in vars(contract) if not name.startswith("_")
    }
    forbidden = {"run", "solve", "simulate", "produce", "callback"}
    assert not forbidden.intersection(contract.__all__)

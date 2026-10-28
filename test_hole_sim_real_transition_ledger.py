from __future__ import annotations

import json

import pandas as pd
import pytest

from archive import hole_sim_fixed_reference_value as reference
from archive import hole_sim_real_transition_ledger as ledger


def _validation_kwargs(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "expected_state_table_sha256": str(frame["materialized_state_table_sha256"].iloc[0]),
        "expected_source_shot_sg_sha256": str(frame["source_shot_sg_sha256"].iloc[0]),
        "expected_source_raw_shots_sha256": str(frame["source_raw_shots_sha256"].iloc[0]),
    }


def _reseal_physical_row(frame: pd.DataFrame, index: int) -> None:
    row = frame.loc[index].to_dict()
    digest = ledger._stable_sha(ledger._physical_payload(row))
    frame.loc[index, "physical_atom_sha256"] = digest
    frame.loc[index, "atom_id"] = digest


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, reference.ReferenceValueModel]:
    model = reference.ReferenceValueModel.load(ledger.DEFAULT_REFERENCE)
    sg, raw, sg_path, raw_path = ledger.load_fixture_sources(ledger.DEFAULT_EXPORT)
    frame, states, audit = ledger.build_ledger(
        sg,
        raw,
        model,
        sg_sha256=ledger._sha256(sg_path),
        raw_sha256=ledger._sha256(raw_path),
    )
    return frame, states, audit, model


def test_real_lowry_holes_telescope_and_preserve_categories(built) -> None:
    frame, states, audit, _ = built
    assert len(frame) == 8
    assert len(states) == 9
    assert audit["telescoping_residual"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert audit.set_index(["round_no", "hole_no"])["total_strokes"].to_dict() == {
        (1, 6): 4,
        (2, 13): 5,
    }
    for _, group in frame.groupby(["round_no", "hole_no"]):
        assert group.sort_values("transition_no")["category"].tolist() == [
            "ott",
            "app",
            "arg",
            "putt",
        ]


def test_real_rewards_and_linked_penalty_are_exact(built) -> None:
    frame, _, _, _ = built
    r1 = frame[(frame["round_no"] == 1) & (frame["hole_no"] == 6)]
    assert r1["fixed_reference_reward"].tolist() == pytest.approx(
        [
            0.06700000000000017,
            0.37244444199999993,
            0.3372823330642538,
            0.016273224935746233,
        ],
        abs=1e-15,
    )
    penalty = frame[
        (frame["round_no"] == 2)
        & (frame["hole_no"] == 13)
        & (frame["category"] == "app")
    ].iloc[0]
    assert penalty["penalty_strokes"] == 1
    assert penalty["stroke_cost"] == 2
    assert penalty["fixed_reference_reward"] == pytest.approx(-0.8133333260000004, abs=1e-15)
    assert len(json.loads(penalty["linked_penalty_row_keys"])) == 1
    assert len(json.loads(penalty["linked_drop_row_keys"])) == 1
    assert penalty["resolution_shot_key"] == (
        "5597bd8af5bcdcaf38aff032b07114e84e9fb3e1f5fad15bd6e8f5dff56207ba"
    )
    assert penalty["causal_observed_end_state_key"] == "recovery|11.777778yd"
    assert penalty["end_state_key"] == "fairway|10.833333yd"


def test_mutated_stroke_cost_and_reward_fail_closed(built) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    changed.loc[0, "stroke_cost"] = 2
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="causal plus penalties"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))
    changed = frame.copy()
    changed.loc[0, "fixed_reference_reward"] += 0.01
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="fixed-reference reward"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


def test_mutated_state_and_scope_fail_closed(built) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    changed.loc[1, "start_distance_yd"] += 1.0
    with pytest.raises(
        ledger.TransitionLedgerUnavailable,
        match="multiple physical states|Materialized state table hash",
    ):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))
    changed = frame.copy()
    changed.loc[0, "pre_event_donor_eligible"] = True
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="anchor-eligible"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


@pytest.mark.parametrize("column", ["source_shot_sg_sha256", "source_raw_shots_sha256"])
def test_forged_source_hashes_fail_closed(built, column: str) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    changed[column] = "0" * 64
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="expected source"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


def test_reference_version_and_state_table_hash_fail_closed(built) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    changed["reward_reference_version"] = "forged_reference_version"
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="version changed"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))

    changed = frame.copy()
    changed["materialized_state_table_sha256"] = "0" * 64
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="state table hash changed"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


def test_unmatched_drop_and_duplicate_source_rows_fail_closed(built) -> None:
    frame, _, _, model = built
    penalty_index = frame.index[frame["penalty_strokes"].gt(0)][0]
    changed = frame.copy()
    penalty_keys = json.loads(changed.loc[penalty_index, "linked_penalty_row_keys"])
    drop_keys = json.loads(changed.loc[penalty_index, "linked_drop_row_keys"])
    drop_keys.append("f" * 64)
    changed.loc[penalty_index, "linked_drop_row_keys"] = json.dumps(
        drop_keys, separators=(",", ":")
    )
    changed.loc[penalty_index, "source_row_keys"] = json.dumps(
        [changed.loc[penalty_index, "causal_shot_key"], *penalty_keys, *drop_keys],
        separators=(",", ":"),
    )
    _reseal_physical_row(changed, penalty_index)
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="linkage is incomplete"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))

    changed = frame.copy()
    first = changed.index[0]
    second = changed.index[
        changed["round_no"].eq(2) & changed["hole_no"].eq(13) & changed["transition_no"].eq(1)
    ][0]
    duplicate_key = str(changed.loc[first, "causal_shot_key"])
    changed.loc[second, "causal_shot_key"] = duplicate_key
    changed.loc[second, "source_row_keys"] = json.dumps([duplicate_key], separators=(",", ":"))
    _reseal_physical_row(changed, second)
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="not unique within the ledger"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("atom_kind", "terminal_holed", "Atom kind"),
        ("hazard_resolution_status", "linked_penalty_drop_collapsed", "Hazard status"),
        ("source_stroke_no", 99, "stroke sequence"),
    ],
)
def test_resealed_semantic_forgeries_fail_closed(
    built,
    column: str,
    value: object,
    message: str,
) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    changed.loc[0, column] = value
    _reseal_physical_row(changed, 0)
    with pytest.raises(ledger.TransitionLedgerUnavailable, match=message):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))


def test_same_rounded_key_flat_value_discontinuity_fails_closed(built) -> None:
    frame, _, _, model = built
    changed = frame.copy()
    approach_index = changed.index[
        changed["round_no"].eq(1)
        & changed["hole_no"].eq(6)
        & changed["transition_no"].eq(3)
    ][0]
    putt_index = changed.index[
        changed["round_no"].eq(1)
        & changed["hole_no"].eq(6)
        & changed["transition_no"].eq(4)
    ][0]

    changed.loc[approach_index, "causal_observed_end_distance_yd"] = 0.1
    changed.loc[approach_index, "causal_observed_end_state_key"] = model.state_key("green", 0.1)
    changed.loc[approach_index, "end_distance_yd"] = 0.1
    changed.loc[approach_index, "end_state_key"] = model.state_key("green", 0.1)
    changed.loc[approach_index, "end_reference_value"] = model.value("green", 0.1)
    changed.loc[approach_index, "fixed_reference_reward"] = (
        changed.loc[approach_index, "start_reference_value"]
        - changed.loc[approach_index, "stroke_cost"]
        - changed.loc[approach_index, "end_reference_value"]
    )
    _reseal_physical_row(changed, approach_index)

    changed.loc[putt_index, "start_distance_yd"] = 0.1000004
    changed.loc[putt_index, "start_state_key"] = model.state_key("green", 0.1000004)
    changed.loc[putt_index, "start_reference_value"] = model.value("green", 0.1000004)
    changed.loc[putt_index, "fixed_reference_reward"] = (
        changed.loc[putt_index, "start_reference_value"]
        - changed.loc[putt_index, "stroke_cost"]
        - changed.loc[putt_index, "end_reference_value"]
    )
    _reseal_physical_row(changed, putt_index)

    assert changed.loc[approach_index, "end_state_key"] == changed.loc[putt_index, "start_state_key"]
    assert (
        changed.loc[approach_index, "end_distance_yd"]
        != changed.loc[putt_index, "start_distance_yd"]
    )
    with pytest.raises(ledger.TransitionLedgerUnavailable, match="multiple physical states"):
        ledger.validate_complete_transition_ledger(changed, model, **_validation_kwargs(frame))

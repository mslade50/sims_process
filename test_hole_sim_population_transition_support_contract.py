from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from archive import hole_sim_fixed_reference_value as fixed_reference
from archive import hole_sim_population_transition_support_contract as contract
from archive.hole_sim_markov import MarkovHoleModel
from archive.hole_sim_traced_population_transition import (
    MARKOV_TRANSITION_COLUMNS,
    TracedBatchedPopulationTransitionSampler,
)


def _transition_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(70):
        holed = index % 19 == 0
        rows.append(
            {
                "transition_row_id": index,
                "event_key": f"pga:R202{4 + index % 2}034",
                "source_player_id": f"arg-player-{index:03d}",
                "player_name": f"ARG Donor {index:03d}",
                "round_no": 1,
                "hole_no": 6,
                "stroke_no": 2 + index % 4,
                "shot_key": f"arg-shot-{index:03d}",
                "category": "arg",
                "lie": "rough",
                "bucket": 0,
                "start_distance": float(index + 2),
                "next_lie": "hole" if holed else "green",
                "next_distance": 0.0 if holed else float(index % 20 + 1),
                "sg": float(index - 35) / 20.0,
                "cost": 4 if index == 37 else (2 if index % 23 == 0 else 1),
                "holed": holed,
                "sg_weight_valid": index % 13 != 0,
            }
        )
    for offset in range(20):
        index = 70 + offset
        holed = offset % 7 == 0
        rows.append(
            {
                "transition_row_id": index,
                "event_key": f"pga:R202{4 + offset % 2}034",
                "source_player_id": f"putt-player-{offset:03d}",
                "player_name": f"PUTT Donor {offset:03d}",
                "round_no": 1,
                "hole_no": 6,
                "stroke_no": 3 + offset % 3,
                "shot_key": f"putt-shot-{offset:03d}",
                "category": "putt",
                "lie": "green",
                "bucket": 0,
                "start_distance": float(offset + 1),
                "next_lie": "hole" if holed else "green",
                "next_distance": 0.0 if holed else float(max(offset - 2, 1)),
                "sg": float(10 - offset) / 25.0,
                "cost": 1,
                "holed": holed,
                "sg_weight_valid": offset % 6 != 0,
            }
        )
    return pd.DataFrame(rows)


def _reference_model() -> fixed_reference.ReferenceValueModel:
    rows: list[dict[str, object]] = []
    for lie in fixed_reference.NON_GREEN_LIES:
        rows.extend(
            [
                {
                    "lie": lie,
                    "distance_yd": 0.0,
                    "expected_strokes": 1.0,
                    "source_kind": "synthetic_contract_test",
                },
                {
                    "lie": lie,
                    "distance_yd": 600.0,
                    "expected_strokes": 7.0,
                    "source_kind": "synthetic_contract_test",
                },
            ]
        )
    rows.extend(
        [
            {
                "lie": "green",
                "distance_yd": 0.0,
                "expected_strokes": 1.0,
                "source_kind": "synthetic_contract_test",
            },
            {
                "lie": "green",
                "distance_yd": 100.0,
                "expected_strokes": 4.0,
                "source_kind": "synthetic_contract_test",
            },
            {
                "lie": "hole",
                "distance_yd": 0.0,
                "expected_strokes": 0.0,
                "source_kind": "synthetic_contract_test",
            },
        ]
    )
    table = pd.DataFrame(rows)
    putting_sha = "a" * 64
    return fixed_reference.ReferenceValueModel(
        table=table,
        putting_curve_sha256=putting_sha,
        model_sha256=fixed_reference.reference_model_sha256(table, putting_sha),
    )


def _bridge(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = transitions[["event_key", "source_player_id"]].drop_duplicates().copy()
    rows = rows.loc[rows["source_player_id"].ne("arg-player-035")].copy()
    starts = {
        "pga:R2024034": pd.Timestamp("2024-06-20", tz="UTC"),
        "pga:R2025034": pd.Timestamp("2025-06-19", tz="UTC"),
    }
    rows["event_start"] = rows["event_key"].map(starts)
    rows["skill_date"] = rows["event_start"] - pd.Timedelta(days=7)
    rows["ag_arg"] = np.linspace(-0.4, 0.4, len(rows), dtype=float)
    rows["ag_putt"] = np.linspace(0.3, -0.3, len(rows), dtype=float)
    return rows.reset_index(drop=True)


def _ledger_from_queries(
    sampler: TracedBatchedPopulationTransitionSampler,
    reference: fixed_reference.ReferenceValueModel,
) -> pd.DataFrame:
    queries = pd.DataFrame(
        {
            "arm": ["raw", "raw", "normalized", "raw"],
            "particle_id": [3, 8, 3, 5],
            "atom_sequence": [2, 2, 2, 4],
            "round_no": [1, 1, 1, 1],
            "hole_no": [6, 6, 6, 6],
            "category": ["arg", "arg", "arg", "putt"],
            "start_lie": ["rough", "rough", "rough", "green"],
            "start_distance_yd": [40.0, 40.0, 40.0, 8.0],
            "rng_population_uniform": [0.123, 0.731, 0.876, 0.55],
        }
    )
    sampled = sampler.sample_many(
        round_no=queries["round_no"].to_numpy(int),
        hole_no=queries["hole_no"].to_numpy(int),
        category=queries["category"].to_numpy(str),
        lie=queries["start_lie"].to_numpy(str),
        distance_yd=queries["start_distance_yd"].to_numpy(float),
        common_uniform=queries["rng_population_uniform"].to_numpy(float),
    )
    assert sampled.transition.available.all()
    rows: list[dict[str, object]] = []
    for index, query in queries.iterrows():
        normalized_end = str(
            contract.batched.normalize_population_lie(sampled.donor_next_lie[index])
        )
        terminal = bool(sampled.donor_holed[index] or normalized_end == "hole")
        end_lie = "hole" if terminal else normalized_end
        end_distance = (
            0.0
            if terminal
            else max(float(sampled.donor_next_distance[index]), 0.0)
        )
        cost = int(sampled.donor_cost[index])
        rows.append(
            {
                "target_event_key": "pga:R2026034",
                "target_event_start": pd.Timestamp("2026-06-25", tz="UTC"),
                "target_player_id": "33204",
                "arm": str(query.arm),
                "round_no": int(query.round_no),
                "hole_no": int(query.hole_no),
                "category": str(query.category),
                "start_lie": str(query.start_lie),
                "start_distance_yd": float(query.start_distance_yd),
                "start_state_key": reference.state_key(
                    query.start_lie, query.start_distance_yd
                ),
                "population_query_state_key": str(sampled.query_state_key[index]),
                "particle_id": int(query.particle_id),
                "atom_sequence": int(query.atom_sequence),
                "rng_population_uniform": float(query.rng_population_uniform),
                "population_pool_key": str(sampled.pool_key[index]),
                "population_pool_level": str(sampled.transition.level[index]),
                "population_pool_index_level": str(
                    sampled.pool_index_level[index]
                ),
                "population_source_scope": str(sampled.source_scope[index]),
                "population_donor_pool_row": int(sampled.donor_pool_row[index]),
                "population_donor_transition_row_id": int(
                    sampled.donor_transition_row_id[index]
                ),
                "population_normalized_knn_probability": float(
                    sampled.normalized_knn_probability[index]
                ),
                "population_cdf_lower": float(sampled.cdf_lower[index]),
                "population_cdf_upper": float(sampled.cdf_upper[index]),
                "population_donor_sg": float(sampled.donor_sg[index]),
                "population_donor_sg_weight_valid": bool(
                    sampled.donor_sg_weight_valid[index]
                ),
                "atom_id": str(sampled.source_shot_key[index]),
                "donor_event_key": str(sampled.source_event_key[index]),
                "donor_player_id": str(sampled.source_player_id[index]),
                "donor_player_name": str(sampled.source_player_name[index]),
                "donor_round_no": int(sampled.source_round_no[index]),
                "donor_hole_no": int(sampled.source_hole_no[index]),
                "donor_stroke_no": int(sampled.source_stroke_no[index]),
                "stroke_cost": cost,
                "terminal": terminal,
                "end_lie": end_lie,
                "end_distance_yd": end_distance,
                "fixed_reference_reward": reference.transition_reward(
                    start_lie=query.start_lie,
                    start_distance_yd=query.start_distance_yd,
                    end_lie=end_lie,
                    end_distance_yd=end_distance,
                    stroke_cost=cost,
                ),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def fixture() -> dict[str, object]:
    transitions = _transition_frame()
    canonical = transitions.loc[:, list(MARKOV_TRANSITION_COLUMNS)].copy()
    model = MarkovHoleModel(
        canonical,
        canonical.iloc[0:0].copy(),
        min_exact=4,
        knn=64,
    )
    sampler = TracedBatchedPopulationTransitionSampler(
        model,
        course_transitions=transitions,
        chunk_size=11,
    )
    reference = _reference_model()
    bridge = _bridge(transitions)
    ledger = _ledger_from_queries(sampler, reference)
    atoms, inventory, metadata = contract.materialize_population_support_contract(
        ledger, sampler, reference, bridge
    )
    return {
        "transitions": transitions,
        "sampler": sampler,
        "reference": reference,
        "bridge": bridge,
        "ledger": ledger,
        "atoms": atoms,
        "inventory": inventory,
        "metadata": metadata,
    }


def test_complete_support_matches_exact_sampler_measure_and_replays_ledger(
    fixture: dict[str, object],
) -> None:
    atoms = fixture["atoms"]
    inventory = fixture["inventory"]
    metadata = fixture["metadata"]
    audit = contract.validate_population_support_contract(
        atoms,
        inventory,
        fixture["ledger"],
        fixture["sampler"],
        fixture["reference"],
        fixture["bridge"],
        expected_metadata=metadata,
    )

    assert metadata["supports"] == 3
    assert metadata["atoms"] == 148
    assert metadata["selected_ledger_rows"] == 4
    assert metadata["selected_replay"]["selected_rows_replayed"] == 4
    assert audit["selected_rows_replayed"] == 4
    assert audit["inputs_reconstructed_exactly"]
    assert not metadata["anchor_ready"]
    assert metadata["anchor_status"] == "HOLD_missing_strict_prior_donor_ability"
    assert metadata["ability_audit"]["missing_rows"] > 0
    assert atoms["fixed_reference_reward"].min() < -2.0
    assert atoms["fixed_reference_reward"].max() > 0.0
    assert inventory["probability_mass"].to_numpy(float).tolist() == pytest.approx(
        [1.0, 1.0, 1.0], abs=2e-14, rel=0.0
    )
    assert inventory["cdf_upper"].to_numpy(float).tolist() == pytest.approx(
        [1.0, 1.0, 1.0], abs=2e-14, rel=0.0
    )
    for name in (
        "ledger_population_sha256",
        "transition_input_sha256",
        "bridge_input_sha256",
        "reference_function_sha256",
        "sampler_input_sha256",
    ):
        assert len(str(metadata["input_sha256"][name])) == 64

    raw_arg = atoms.loc[
        atoms["arm"].eq("raw")
        & atoms["category"].eq("arg")
        & atoms["query_start_distance_yd"].eq(40.0)
    ].sort_values("support_row_order")
    sampler = fixture["sampler"]
    query_key = (1, 6, "arg", "rough", 0)
    resolution = sampler.base_sampler._resolution[query_key]
    cache = sampler.base_sampler._pool_by_id[resolution.pool_id]
    metric = np.abs(np.log1p(40.0) - cache.start_log_distance)
    expected_rows = np.argpartition(metric, 63)[:64]
    expected_rows.sort()
    expected_weight = np.exp(-metric[expected_rows] / 0.18)
    np.maximum(expected_weight, 1e-12, out=expected_weight)
    expected_weight /= expected_weight.sum()
    assert np.array_equal(raw_arg["donor_pool_row"].to_numpy(int), expected_rows)
    assert np.array_equal(
        raw_arg["proposal_probability"].to_numpy(float), expected_weight
    )


def test_quality_group_excludes_player_and_arm_but_retains_query_state(
    fixture: dict[str, object],
) -> None:
    atoms = fixture["atoms"]
    row = atoms.iloc[0].to_dict()
    changed_player = dict(row)
    changed_player["donor_player_id"] = "different-player"
    changed_player["donor_player_name"] = "Different Name"
    changed_player["arm"] = "counterfactual-arm"
    assert contract.donor_quality_group_key(row) == contract.donor_quality_group_key(
        changed_player
    )
    changed_state = dict(row)
    changed_state["query_start_distance_hex"] = float(
        row["query_start_distance_yd"] + 0.5
    ).hex()
    assert contract.donor_quality_group_key(row) != contract.donor_quality_group_key(
        changed_state
    )
    same_event_support = atoms.loc[
        atoms["support_key"].eq(row["support_key"])
        & atoms["donor_event_key"].eq(row["donor_event_key"])
    ]
    assert same_event_support["donor_player_id"].nunique() > 1
    assert same_event_support["donor_quality_group_key"].nunique() == 1


def test_validator_rejects_measure_identity_and_ability_mutations(
    fixture: dict[str, object],
) -> None:
    atoms = fixture["atoms"]
    inventory = fixture["inventory"]
    args = (
        inventory,
        fixture["ledger"],
        fixture["sampler"],
        fixture["reference"],
        fixture["bridge"],
    )

    probability = atoms.copy()
    probability.loc[0, "proposal_probability"] += 1e-5
    with pytest.raises(contract.PopulationSupportContractError):
        contract.validate_population_support_contract(probability, *args)

    identity = atoms.copy()
    identity.loc[0, "donor_transition_row_id"] += 1
    with pytest.raises(contract.PopulationSupportContractError):
        contract.validate_population_support_contract(identity, *args)

    # Recomputing both per-support seals cannot legitimize a donor mutation:
    # the independent source reconstruction remains authoritative.
    rehashed = atoms.copy()
    rehashed_inventory = inventory.copy()
    support_key = str(rehashed.loc[0, "support_key"])
    selected = rehashed["support_key"].astype(str).eq(support_key)
    rehashed.loc[rehashed.index[selected][0], "donor_player_name"] = "forged name"
    group = rehashed.loc[selected].sort_values("support_row_order")
    measure_sha = contract.frame_sha256(group, contract.SUPPORT_MEASURE_COLUMNS)
    rehashed.loc[selected, "support_measure_sha256"] = measure_sha
    group = rehashed.loc[selected].sort_values("support_row_order")
    ability_sha = contract.frame_sha256(group, contract.SUPPORT_ABILITY_COLUMNS)
    rehashed.loc[selected, "support_ability_sha256"] = ability_sha
    inventory_selected = rehashed_inventory["support_key"].astype(str).eq(
        support_key
    )
    rehashed_inventory.loc[
        inventory_selected, "support_measure_sha256"
    ] = measure_sha
    rehashed_inventory.loc[
        inventory_selected, "support_ability_sha256"
    ] = ability_sha
    with pytest.raises(
        contract.PopulationSupportContractError,
        match="differ from reconstructed inputs",
    ):
        contract.validate_population_support_contract(
            rehashed,
            rehashed_inventory,
            fixture["ledger"],
            fixture["sampler"],
            fixture["reference"],
            fixture["bridge"],
        )

    ability = atoms.copy()
    available = ability.index[ability["donor_ability_available"]][0]
    ability.loc[available, "donor_ability_sg"] += 0.25
    with pytest.raises(contract.PopulationSupportContractError):
        contract.validate_population_support_contract(ability, *args)

    changed_inventory = inventory.copy()
    changed_inventory.loc[0, "selected_ledger_rows"] += 1
    with pytest.raises(
        contract.PopulationSupportContractError,
        match="inventory differs",
    ):
        contract.validate_population_support_contract(
            atoms,
            changed_inventory,
            fixture["ledger"],
            fixture["sampler"],
            fixture["reference"],
            fixture["bridge"],
        )


def test_validator_rejects_ledger_selection_and_uniform_mutations(
    fixture: dict[str, object],
) -> None:
    atoms = fixture["atoms"]
    inventory = fixture["inventory"]
    common = (
        fixture["sampler"],
        fixture["reference"],
        fixture["bridge"],
    )
    identity = fixture["ledger"].copy()
    identity.loc[0, "donor_player_id"] = "forged-player"
    with pytest.raises(contract.PopulationSupportContractError, match="replay changed"):
        contract.validate_population_support_contract(
            atoms, inventory, identity, *common
        )

    uniform = fixture["ledger"].copy()
    uniform.loc[0, "rng_population_uniform"] = np.nextafter(1.0, 0.0)
    with pytest.raises(contract.PopulationSupportContractError, match="replay changed"):
        contract.validate_population_support_contract(
            atoms, inventory, uniform, *common
        )


def test_validator_binds_transition_bridge_and_reference_inputs(
    fixture: dict[str, object],
) -> None:
    atoms = fixture["atoms"]
    inventory = fixture["inventory"]
    metadata = fixture["metadata"]
    ledger = fixture["ledger"]
    sampler = fixture["sampler"]
    reference = fixture["reference"]

    bridge = fixture["bridge"].copy()
    bridge.loc[0, "ag_arg"] += 0.01
    with pytest.raises(contract.PopulationSupportContractError):
        contract.validate_population_support_contract(
            atoms,
            inventory,
            ledger,
            sampler,
            reference,
            bridge,
            expected_metadata=metadata,
        )

    forged_metadata = copy.deepcopy(metadata)
    forged_metadata["input_sha256"]["bridge_input_sha256"] = "0" * 64
    with pytest.raises(
        contract.PopulationSupportContractError,
        match="metadata changed: input_sha256",
    ):
        contract.validate_population_support_contract(
            atoms,
            inventory,
            ledger,
            sampler,
            reference,
            fixture["bridge"],
            expected_metadata=forged_metadata,
        )

    changed_table = reference.table.copy()
    changed_table.loc[
        changed_table["lie"].eq("rough")
        & changed_table["distance_yd"].eq(600.0),
        "expected_strokes",
    ] += 0.01
    changed_reference = fixed_reference.ReferenceValueModel(
        table=changed_table,
        putting_curve_sha256=reference.putting_curve_sha256,
        model_sha256=fixed_reference.reference_model_sha256(
            changed_table, reference.putting_curve_sha256
        ),
    )
    with pytest.raises(contract.PopulationSupportContractError):
        contract.validate_population_support_contract(
            atoms,
            inventory,
            ledger,
            sampler,
            changed_reference,
            fixture["bridge"],
            expected_metadata=metadata,
        )

    trace = next(iter(sampler.registry._by_pool_id.values()))
    original = trace.rows.loc[0, "player_name"]
    trace.rows.loc[0, "player_name"] = "mutated donor identity"
    try:
        with pytest.raises((contract.PopulationSupportContractError, ValueError)):
            contract.validate_population_support_contract(
                atoms,
                inventory,
                ledger,
                sampler,
                reference,
                fixture["bridge"],
                expected_metadata=metadata,
            )
    finally:
        trace.rows.loc[0, "player_name"] = original


def test_strict_prior_ability_cutoffs_fail_closed(fixture: dict[str, object]) -> None:
    bridge = fixture["bridge"].copy()
    bridge["skill_date"] = bridge["event_start"]
    with pytest.raises(ValueError, match="strict cutoff"):
        contract.materialize_population_support_contract(
            fixture["ledger"],
            fixture["sampler"],
            fixture["reference"],
            bridge,
        )

    future = fixture["bridge"].copy()
    future["event_start"] = pd.Timestamp("2026-06-25", tz="UTC")
    future["skill_date"] = pd.Timestamp("2026-06-24", tz="UTC")
    with pytest.raises(ValueError, match="strict cutoff"):
        contract.materialize_population_support_contract(
            fixture["ledger"],
            fixture["sampler"],
            fixture["reference"],
            future,
        )

import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from archive.build_all_round_market_prediction_arms import (
    CONTROL_CANDIDATE,
    EXPECTED_CONTRACT_SHA256,
    apply_frozen_arms,
    apply_frozen_arms_from_paths,
    build_market_candidate_authority,
    build_2023_selection_arms,
    write_selection_artifacts,
)


def _inputs(year=2023, events=3, players=(101, 202), start_month_day="07-06"):
    base_rows = []
    bridge_rows = []
    state_rows = []
    for event_index in range(events):
        event_date = pd.Timestamp(f"{year}-{start_month_day}") + pd.Timedelta(days=7 * event_index)
        event_key = f"pga|{year}|{event_index + 1}|{event_date.date()}"
        for player_index, player_id in enumerate(players):
            core = -0.4 + 0.3 * player_index + 0.05 * event_index
            downstream = 0.25 if player_index == 0 else -0.15
            current_r1 = core + 0.7 * downstream + 0.1
            stages = (0.0, 0.20, -0.10, 0.35)
            base_rows.append(
                {
                    "candidate": "challenger",
                    "event_key": event_key,
                    "dg_id_key": str(player_id),
                    "event_start_date": event_date,
                    "prediction_asof": event_date,
                    "training_max_event_end": event_date - pd.Timedelta(days=1),
                    "chronology_ok": True,
                    "prediction": core,
                    "prediction_stage": "raw_base_core_before_downstream_and_live",
                    "downstream_lambda_applied": False,
                    "live_update_lambda_applied": False,
                }
            )
            bridge_rows.append(
                {
                    # Exercise the real bridge's different legacy event key.
                    "event_key": f"{year}__{event_index + 1}__{event_date:%Y%m%d}",
                    "event_start": event_date,
                    "dg_id": player_id,
                    "baseline_r1_prediction": current_r1,
                    "downstream_preserved_total": downstream,
                }
            )
            state = {
                "event_key": event_key,
                "event_start_date": event_date,
                "dg_id": player_id,
            }
            for round_num, stage in enumerate(stages, start=1):
                state[f"cascade_prediction_r{round_num}"] = current_r1 + stage
                # The synthetic truth is exactly lambda_downstream=lambda_live=1.
                state[f"label_dg_sg_adj_r{round_num}"] = core + downstream + stage
            state_rows.append(state)
    return pd.DataFrame(base_rows), pd.DataFrame(bridge_rows), pd.DataFrame(state_rows)


def test_selection_uses_all_four_rounds_and_tunes_one_common_live_lambda():
    base, bridge, outcomes = _inputs()
    result = build_2023_selection_arms(base, bridge, outcomes)

    chosen = result.frozen_lambdas.iloc[0]
    assert chosen["candidate"] == "challenger"
    assert chosen["downstream_lambda"] == 1.0
    assert chosen["live_lambda"] == 1.0
    assert chosen["observed_player_rounds"] == 3 * 2 * 4
    assert chosen["events"] == 3
    assert result.lambda_tuning.shape[0] == 9
    assert result.lambda_tuning["live_lambda_scope"].eq(
        "one_common_scalar_r2_r4"
    ).all()

    assert list(result.base_predictions) == [
        "candidate", "event_key", "dg_id_key", "prediction"
    ]
    assert list(result.round_predictions) == [
        "candidate", "event_key", "dg_id_key", "round_num", "prediction"
    ]
    candidate_rounds = result.round_predictions.loc[
        result.round_predictions["candidate"].eq("challenger")
    ]
    assert set(candidate_rounds["round_num"]) == {1, 2, 3, 4}
    assert result.audit["target_rounds"] == [1, 2, 3, 4]
    assert result.audit["evaluation_target_columns_read"] == []

    # The control is copied, not reconstructed or scalar-adjusted.
    expected = outcomes.assign(dg_id_key=outcomes["dg_id"].astype(str)).set_index(
        ["event_key", "dg_id_key"]
    )
    observed = result.round_predictions.loc[
        result.round_predictions["candidate"].eq(CONTROL_CANDIDATE)
    ]
    for row in observed.itertuples(index=False):
        assert np.isclose(
            row.prediction,
            expected.loc[(row.event_key, row.dg_id_key), f"cascade_prediction_r{row.round_num}"],
        )


def test_numerical_ties_prefer_both_lambdas_at_one():
    base, bridge, outcomes = _inputs(events=2)
    bridge["downstream_preserved_total"] = 0.0
    for round_num in range(2, 5):
        outcomes[f"cascade_prediction_r{round_num}"] = outcomes["cascade_prediction_r1"]
    base_truth = base.set_index(["event_key", "dg_id_key"])["prediction"]
    for round_num in range(1, 5):
        outcomes[f"label_dg_sg_adj_r{round_num}"] = [
            base_truth.loc[(event, str(player))]
            for event, player in zip(outcomes["event_key"], outcomes["dg_id"])
        ]
    chosen = build_2023_selection_arms(base, bridge, outcomes).frozen_lambdas.iloc[0]
    assert chosen["downstream_lambda"] == 1.0
    assert chosen["live_lambda"] == 1.0


def test_lambda_tuning_uses_july_december_not_mapper_calibration_months():
    july_base, july_bridge, july_outcomes = _inputs(events=2)
    april_base, april_bridge, april_outcomes = _inputs(
        events=2, start_month_day="04-06"
    )
    april_core = april_base.set_index(["event_key", "dg_id_key"])["prediction"]
    # April-June prefers (0, 0), while July-December prefers (1, 1).
    # The locked architecture-selection boundary must choose the latter.
    for round_num in range(1, 5):
        april_outcomes[f"label_dg_sg_adj_r{round_num}"] = [
            april_core.loc[(event, str(player))]
            for event, player in zip(april_outcomes["event_key"], april_outcomes["dg_id"])
        ]
    result = build_2023_selection_arms(
        pd.concat([april_base, july_base], ignore_index=True),
        pd.concat([april_bridge, july_bridge], ignore_index=True),
        pd.concat([april_outcomes, july_outcomes], ignore_index=True),
    )
    chosen = result.frozen_lambdas.iloc[0]
    assert chosen["downstream_lambda"] == 1.0
    assert chosen["live_lambda"] == 1.0
    assert chosen["events"] == 2


def test_selection_writes_deterministic_exact_closure_market_authority(tmp_path):
    base, bridge, outcomes = _inputs(events=2)
    result = build_2023_selection_arms(base, bridge, outcomes)
    authority = result.market_candidate_authority
    emitted = set(result.base_predictions["candidate"])

    assert set(authority["candidate"]) == emitted
    assert set(result.round_predictions["candidate"]) == emitted
    assert list(authority.columns[:3]) == [
        "candidate", "course_fit_lambda", "live_update_lambda"
    ]
    control = authority.loc[authority["candidate"].eq(CONTROL_CANDIDATE)].iloc[0]
    assert control["course_fit_lambda"] == 1.0
    assert control["live_update_lambda"] == 1.0
    assert bool(control["exact_cached_control"])
    assert control["authority_role"] == "exact_cached_control"
    challenger = authority.loc[authority["candidate"].eq("challenger")].iloc[0]
    assert challenger["course_fit_lambda"] == 1.0
    assert challenger["live_update_lambda"] == 1.0
    assert not bool(challenger["exact_cached_control"])

    with pytest.raises(ValueError, match="candidate closure differs"):
        build_market_candidate_authority(
            result.frozen_lambdas,
            result.base_predictions,
            result.round_predictions.loc[
                ~result.round_predictions["candidate"].eq("challenger")
            ],
        )

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_selection_artifacts(result, first)
    write_selection_artifacts(result, second)
    first_bytes = (first / "market_candidate_authority.csv").read_bytes()
    second_bytes = (second / "market_candidate_authority.csv").read_bytes()
    assert first_bytes == second_bytes
    authority_hash = hashlib.sha256(first_bytes).hexdigest()
    frozen = json.loads((first / "frozen_lambdas_2023.json").read_text())
    audit = json.loads((first / "selection_arm_audit.json").read_text())
    manifest = json.loads((first / "selection_arm_manifest.json").read_text())
    assert frozen["market_candidate_authority_sha256"] == authority_hash
    assert audit["market_candidate_authority_sha256"] == authority_hash
    assert manifest["market_candidate_authority_sha256"] == authority_hash
    assert frozen["candidate_catalog_sha256"] == authority_hash
    assert manifest["candidate_catalog_sha256"] == authority_hash
    assert manifest["candidate_closure"] == sorted(emitted)


def test_apply_is_label_blind_and_emits_only_control_plus_selected():
    base, bridge, state = _inputs(year=2024, events=2)
    base = pd.concat(
        [
            base,
            base.assign(candidate="not_selected", prediction=base["prediction"] + 100),
        ],
        ignore_index=True,
    )
    # Deliberately poison evaluation labels.  Apply projects them away before use.
    for round_num in range(1, 5):
        state[f"label_dg_sg_adj_r{round_num}"] = "DO_NOT_READ"
    frozen = pd.DataFrame(
        [
            {
                "candidate": "challenger",
                "downstream_lambda": 1.0,
                "live_lambda": 0.5,
                "selection_year": 2023,
            },
            {
                "candidate": "not_selected",
                "downstream_lambda": 0.0,
                "live_lambda": 0.0,
                "selection_year": 2023,
            },
        ]
    )
    result = apply_frozen_arms(
        base,
        frozen,
        bridge,
        state,
        selected_candidate="challenger",
    )

    expected_candidates = {CONTROL_CANDIDATE, "challenger"}
    assert set(result.base_predictions["candidate"]) == expected_candidates
    assert set(result.round_predictions["candidate"]) == expected_candidates
    assert result.audit["target_columns_read"] == []
    assert result.audit["evaluation_years"] == [2024, 2025, 2026]

    first_base = base.loc[base["candidate"].eq("challenger")].iloc[0]
    first_bridge = bridge.iloc[0]
    expected_pre = first_base["prediction"] + first_bridge["downstream_preserved_total"]
    first_event = first_base["event_key"]
    first_player = first_base["dg_id_key"]
    r4 = result.round_predictions.loc[
        result.round_predictions["candidate"].eq("challenger")
        & result.round_predictions["event_key"].eq(first_event)
        & result.round_predictions["dg_id_key"].eq(first_player)
        & result.round_predictions["round_num"].eq(4)
    ].iloc[0]
    current = state.iloc[0]
    expected_r4 = expected_pre + 0.5 * (
        current["cascade_prediction_r4"] - current["cascade_prediction_r1"]
    )
    assert np.isclose(r4["prediction"], expected_r4)


def test_path_apply_projects_no_evaluation_label_columns(tmp_path, monkeypatch):
    base, bridge, state = _inputs(year=2024, events=1)
    base_path = tmp_path / "base.parquet"
    bridge_path = tmp_path / "bridge.parquet"
    state_path = tmp_path / "state.parquet"
    frozen_path = tmp_path / "frozen.json"
    base.to_parquet(base_path, index=False)
    bridge.to_parquet(bridge_path, index=False)
    state.to_parquet(state_path, index=False)
    frozen_path.write_text(
        json.dumps(
            {
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "selection_year": 2023,
                "lambdas": [
                    {
                        "candidate": "challenger",
                        "downstream_lambda": 1.0,
                        "live_lambda": 1.0,
                        "selection_year": 2023,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    original = pd.read_parquet
    reads = []

    def recording_read(path, *args, **kwargs):
        reads.append((str(path), tuple(kwargs.get("columns", ()))))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    result = apply_frozen_arms_from_paths(
        base_path,
        frozen_path,
        bridge_path,
        state_path,
        selected_candidate="challenger",
    )
    state_columns = dict(reads)[str(state_path)]
    assert not any("label" in column or "target" in column for column in state_columns)
    assert state_columns == (
        "event_key",
        "event_start_date",
        "dg_id",
        "cascade_prediction_r1",
        "cascade_prediction_r2",
        "cascade_prediction_r3",
        "cascade_prediction_r4",
    )
    assert result.audit["target_columns_read"] == []


def test_apply_path_rejects_composed_pre_event_file_before_writing(tmp_path):
    base, bridge, state = _inputs(year=2024, events=1)
    base["prediction_stage"] = "composed_pre_event_after_downstream"
    base["downstream_lambda_applied"] = True
    base["live_update_lambda_applied"] = False
    base_path = tmp_path / "composed_base.parquet"
    bridge_path = tmp_path / "bridge.parquet"
    state_path = tmp_path / "state.parquet"
    frozen_path = tmp_path / "frozen.json"
    base.to_parquet(base_path, index=False)
    bridge.to_parquet(bridge_path, index=False)
    state.to_parquet(state_path, index=False)
    frozen_path.write_text(
        json.dumps(
            {
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "lambdas": [
                    {
                        "candidate": "challenger",
                        "downstream_lambda": 1.0,
                        "live_update_lambda": 1.0,
                        "selection_year": 2023,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_base_core_before_downstream_and_live"):
        apply_frozen_arms_from_paths(
            base_path,
            frozen_path,
            bridge_path,
            state_path,
            selected_candidate="challenger",
        )
    assert not (tmp_path / "evaluation_base_predictions.parquet").exists()
    assert not (tmp_path / "evaluation_round_predictions.parquet").exists()


def test_selection_fails_closed_if_predictions_reach_2024():
    base, bridge, outcomes = _inputs(year=2024, events=1)
    with pytest.raises(ValueError, match="2023-only"):
        build_2023_selection_arms(base, bridge, outcomes)

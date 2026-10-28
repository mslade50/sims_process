import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from archive import evaluate_base_skill_round_cascade as cascade
from archive.evaluate_base_skill_round_cascade import (
    _best_price_market_rows,
    _canonicalize_quote_sides,
    _feature_spec,
    _live_lambda_player_predictions,
    _populated_third_participant,
    _round_market,
    _unexpected_third_participant,
    _validate_round_quote_settlements,
    attach_core_and_candidate,
    classify_score_outcomes,
    discrete_round_probabilities,
    expand_betting_lines,
    load_frozen_finalist_predictions,
    select_and_settle,
)


def test_round_market_excludes_three_balls():
    assert _round_market("R1 Match-Up") == "r1"
    assert _round_market("R4 Matchup") == "r4"
    assert _round_market("R2 3-Ball") is None
    assert _round_market("R1 Three Ball Matchup") is None
    assert _round_market("R3 3ball Matchup") is None
    assert _round_market("R2 Matchup 3 Way") == "r2"
    assert _round_market("72-hole Match") is None


def test_round_parser_detects_nested_third_participant_authoritatively():
    assert _populated_third_participant(
        {"source_odds": {"participants": [{"id": 1}, {"id": 2}, {"id": 3}]}}
    )
    assert _populated_third_participant(
        {"source_odds": {"p3_dg_id": 3}}
    )
    assert not _populated_third_participant(
        {"source_odds": {"p1_dg_id": 1, "p2_dg_id": 2, "market": "3 way"}}
    )


def test_third_participant_guard_excludes_labelled_three_balls_only():
    frame = pd.DataFrame(
        [
            {
                "bet_type": "R1 3-Ball",
                "source_odds": {"p3_dg_id": 3},
            },
            {
                "bet_type": "R1 Match-Up",
                "source_odds": {"p1_dg_id": 1, "p2_dg_id": 2},
            },
            {
                "bet_type": "R2 Match-Up",
                "source_odds": {"p3_dg_id": 3},
            },
            {
                "bet_type": "R3 Matchup 3 Way",
                "source_odds": {"p3_dg_id": 3},
            },
        ]
    )
    assert _unexpected_third_participant(frame).tolist() == [False, False, True, True]


def test_discrete_round_probabilities_normalize_and_reverse():
    difference = np.array([-1.25, 0.0, 0.8])
    win, loss, push = discrete_round_probabilities(
        difference, sigma=2.7, tie_threshold=0.32
    )
    reverse_win, reverse_loss, reverse_push = discrete_round_probabilities(
        -difference, sigma=2.7, tie_threshold=0.32
    )
    np.testing.assert_allclose(win + loss + push, 1.0, atol=1e-12)
    np.testing.assert_allclose(win, reverse_loss, atol=1e-12)
    np.testing.assert_allclose(loss, reverse_win, atol=1e-12)
    np.testing.assert_allclose(push, reverse_push, atol=1e-12)
    assert np.all(push > 0)


def test_c3_features_use_only_completed_weather_state():
    frame = pd.DataFrame(
        {
            "base_core": [0.5],
            "base_event_hybrid": [0.5],
            "legacy_stage_adjustment": [0.0],
            "prev_mean_total_w2": [0.1],
            "prev_last_total_w2": [0.2],
            **{f"prev_mean_orth_{category}_w2": [0.0] for category in ("ott", "app", "arg", "putt")},
            **{f"prev_last_orth_{category}_w2": [0.0] for category in ("ott", "app", "arg", "putt")},
            **{f"prev_trend_orth_{category}_w2": [0.0] for category in ("ott", "app", "arg", "putt")},
            "prev_std_total_w2": [0.0],
            "prev_trend_total_w2": [0.0],
            "start_pos_scaled": [0.5],
            "prev_weather_observed_rate": [1.0],
            # These must never be selected as C3 features.
            "weather_delta_wind": [8.0],
            "weather_delta_dew": [10.0],
            "target_total_w2": [3.0],
        }
    )
    features = _feature_spec(frame, round_num=4, arm="w2", mode="pt")
    assert "weather_delta_wind" not in features
    assert "weather_delta_dew" not in features
    assert "target_total_w2" not in features
    assert "prev_weather_observed_rate" in features


def test_money_settlement_pushes_on_equal_round_score():
    lines = pd.DataFrame(
        [
            {
                "model": "m",
                "candidate": "c",
                "variant": "v",
                "strict_point_in_time": True,
                "retrospective_cascade_anchor": False,
                "event_key": "event",
                "event_id": "event",
                "event_name": "Event",
                "event_start_date": pd.Timestamp("2025-01-01"),
                "round_market": "r2",
                "round_num": 2,
                "pair_key": "1|2",
                "selection_id": "1",
                "selection_name": "One",
                "opponent_id": "2",
                "opponent_name": "Two",
                "book": "pinnacle",
                "close_odds": 120.0,
                "close_time": "2025-01-02 06:00:00",
                "tie_rule": "Bets are void in event of tie",
                "outcome_numeric": 0.5,
                "outcome": "void",
                "probability": 0.55,
                "loss_probability": 0.35,
                "push_probability": 0.10,
                "pricing_probability": 0.55 / 0.90,
                "implied_probability": 1 / 2.2,
                "close_devig_probability": 0.5,
                "model_minus_devig_close_pp": 11.111,
                "ev_per_unit": 0.55 * 1.2 - 0.35,
                "selected_skill": 0.5,
                "opponent_skill": 0.0,
                "selected_base_core": 0.5,
                "opponent_base_core": 0.0,
                "abs_realized_wind_difference": 0.0,
                "abs_realized_dew_difference": 0.0,
                "market": "tournament_matchup",
            }
        ]
    )
    settled = select_and_settle(
        lines, line_policy="pinnacle", sizing="flat_1u", min_ev=0.0
    )
    assert len(settled) == 1
    assert settled.iloc[0]["settlement_status"] == "void"
    assert settled.iloc[0]["pnl"] == 0.0


def test_score_ties_are_separate_from_no_score_voids():
    frame = pd.DataFrame(
        {
            "p1_round_score": [70.0, np.nan, 69.0],
            "p2_round_score": [70.0, 71.0, 71.0],
            "p1_outcome": [0.5, 0.5, 1.0],
            "p2_outcome": [0.5, 0.5, 0.0],
        }
    )
    classified = classify_score_outcomes(frame)
    assert classified["score_tie_void"].tolist() == [True, False, False]
    assert classified["no_score_void"].tolist() == [False, True, False]
    assert classified["score_model_eligible"].tolist() == [True, False, True]


def test_round_score_audit_rejects_bad_p2_even_when_p1_matches() -> None:
    classified = classify_score_outcomes(
        pd.DataFrame(
            {
                "p1_round_score": [69.0],
                "p2_round_score": [71.0],
                "p1_outcome": [1.0],
                "p2_outcome": [1.0],
                "tie_rule": ["void on tie"],
            }
        )
    )
    assert bool(classified.loc[0, "p1_archive_outcome_matches_score"])
    assert not bool(classified.loc[0, "p2_archive_outcome_matches_score"])
    assert not bool(classified.loc[0, "archive_outcome_matches_score"])


def test_round_quote_audit_rejects_bad_p2_across_captures() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["e", "e"],
            "round_market": ["r1", "r1"],
            "pair_key": ["1|2", "1|2"],
            "book": ["pinnacle", "pinnacle"],
            "p1_outcome": [1.0, 1.0],
            "p2_outcome": [0.0, 1.0],
            "tie_rule": ["void", "void"],
        }
    )
    with pytest.raises(ValueError, match="inconsistent p2 outcome"):
        _validate_round_quote_settlements(frame)


def test_round_quote_audit_rejects_impossible_pair() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["e"],
            "round_market": ["r1"],
            "pair_key": ["1|2"],
            "book": ["pinnacle"],
            "p1_outcome": [1.0],
            "p2_outcome": [1.0],
            "tie_rule": ["void"],
        }
    )
    with pytest.raises(ValueError, match="impossible archived round outcome pair"):
        _validate_round_quote_settlements(frame)


def test_round_quote_audit_accepts_two_player_three_way_ties_lose() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["e"],
            "round_market": ["r1"],
            "pair_key": ["1|2"],
            "book": ["pinnacle"],
            "p1_outcome": [0.0],
            "p2_outcome": [0.0],
            "tie_rule": ["3 way; ties lose"],
        }
    )
    audit = _validate_round_quote_settlements(frame)
    assert audit["physical_pairs_verified"] == 1


def test_frozen_pre_event_downstream_lambda_is_separate_from_live_update() -> None:
    rounds = pd.DataFrame(
        {
            "event_key": ["event"],
            "dg_id_key": ["7"],
            "event_start_date": pd.to_datetime(["2021-01-07"]),
            "cascade_prediction": [0.90],
        }
    )
    base = pd.DataFrame(
        {
            "candidate": ["candidate"],
            "event_key": ["event"],
            "dg_id_key": ["7"],
            "event_start_date": pd.to_datetime(["2021-01-07"]),
            "prediction": [0.50],
            "chronology_ok": [True],
        }
    )
    bridge = pd.DataFrame(
        {
            "event_start": pd.to_datetime(["2021-01-07"]),
            "dg_id": [7],
            "legacy_core_pred_step2": [0.50],
            "baseline_r1_prediction": [0.70],
            "downstream_preserved_total": [0.20],
            "reconstruction_error": [0.0],
        }
    )
    output = attach_core_and_candidate(
        rounds,
        base,
        bridge,
        "candidate",
        pre_event_downstream_lambda=0.5,
        pre_event_downstream_lambda_source="discovery_frozen_selection_manifest",
    )
    assert output.loc[0, "base_event_hybrid"] == pytest.approx(0.60)
    assert output.loc[0, "legacy_stage_adjustment"] == pytest.approx(0.20)
    assert output.loc[0, "c1_prediction"] == pytest.approx(0.80)
    assert output.loc[0, "pre_event_downstream_lambda"] == 0.5
    assert "pre_event" in output.loc[0, "pre_event_downstream_lambda_scope"]


def test_cascade_execution_provenance_rejects_midrun_input_mutation(tmp_path) -> None:
    direct_input = tmp_path / "round_panel.parquet"
    direct_input.write_bytes(b"before")
    output = tmp_path / "cascade_output"
    args = cascade.parse_args(["--output-dir", str(output)])
    state = cascade.begin_cascade_execution_provenance(
        output_dir=output,
        input_paths=[direct_input],
        args=args,
    )
    direct_input.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="input changed"):
        cascade.finalize_cascade_execution_provenance(state)


def test_cascade_provenance_source_closure_is_exact() -> None:
    expected = tuple(
        dict.fromkeys(
            (
                cascade.ARCHIVE / "base_skill_frozen_handoff.py",
                Path(cascade.discovery.__file__).resolve(),
                *(
                    Path(path).resolve()
                    for path in cascade.discovery.DISCOVERY_PROVENANCE_SOURCE_CLOSURE
                ),
            )
        )
    )
    assert cascade.PROVENANCE_CODE_PATHS == expected
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


def test_c3_weather_lineage_is_explicit_and_strict_prior(tmp_path) -> None:
    coefficients = tmp_path / "weather_coefficients.csv"
    coefficients.write_text("synthetic", encoding="utf-8")
    digest = hashlib.sha256(coefficients.read_bytes()).hexdigest()
    rows = []
    for arm in ("w0", "w2"):
        for round_num in (1, 2):
            rows.append(
                {
                    "variant": f"C3_pt_continuous_partial_pool_{arm}",
                    "event_start_date": pd.Timestamp("2023-01-05"),
                    "round_num": round_num,
                    "weather_arm": arm,
                    "weather_state_target_column": f"target_total_{arm}",
                    "weather_state_source_columns": (
                        f"target_total_{arm}|orth_ott_{arm}|orth_app_{arm}|"
                        f"orth_arg_{arm}|orth_putt_{arm}"
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
                    "completed_prior_round_boundary": (
                        "round_num_strictly_less_than_target_round"
                    ),
                    "completed_prior_rounds_only": True,
                    "weather_coefficient_fit_cutoff": pd.Timestamp("2022-12-31"),
                    "weather_coefficient_source": "strict_prior_synthetic",
                    "weather_coefficients_path": str(coefficients.resolve()),
                    "weather_coefficients_sha256": digest,
                    "target_or_upcoming_realized_weather_excluded": True,
                    "strict_point_in_time": True,
                    "current_round_weather_columns_role": (
                        "posthoc_outcome_attribution_only_not_model_features"
                    ),
                }
            )
    audit = cascade.audit_c3_weather_lineage(
        pd.DataFrame(rows),
        authoritative_coefficients_path=coefficients,
        authoritative_coefficients_sha256=digest,
    )
    assert audit["contract"] == "completed_prior_round_weather_state_v1"
    assert not any(audit["failures"].values())


def test_prepare_round_state_emits_completed_weather_lineage(tmp_path) -> None:
    rounds_path = tmp_path / "rounds.parquet"
    rows = pd.DataFrame(
        {
            "event_key": ["e", "e"],
            "event_start_date": pd.to_datetime(["2023-01-05"] * 2),
            "event_end_date": pd.to_datetime(["2023-01-08"] * 2),
            "event_name": ["Event"] * 2,
            "year": [2023] * 2,
            "event_id": [1] * 2,
            "dg_id": [7] * 2,
            "player_name": ["Player"] * 2,
            "round_num": [1, 2],
            "round_date": pd.to_datetime(["2023-01-05", "2023-01-06"]),
            "score": [70, 71],
            "field_size": [100] * 2,
            "start_pos": [1, 20],
            "dg_sg_total_field_adjusted": [1.0, 0.0],
            "weather_delta_wind": [2.0, 3.0],
            "weather_delta_dew": [1.0, 1.5],
            "cascade_actual": [1.0, 0.0],
            "cascade_prediction": [0.5, 0.4],
            "cascade_residual": [0.5, -0.4],
            "fin_text": ["T10"] * 2,
            **{
                f"dg_sg_{category}_field_adjusted": [0.25, 0.0]
                for category in ("ott", "app", "arg", "putt")
            },
        }
    )
    rows.to_parquet(rounds_path, index=False)
    weather_path = tmp_path / "weather.csv"
    weather = {
        "event_start_date": ["2023-01-05"],
        "fit_max_round_date": ["2022-12-31"],
        "coefficient_source": ["strict_prior_synthetic"],
        "total_beta_wind": [0.1],
        "total_beta_dew": [0.05],
    }
    for category in ("ott", "app", "arg", "putt"):
        weather[f"{category}_beta_wind"] = [0.025]
        weather[f"{category}_beta_dew"] = [0.0125]
    pd.DataFrame(weather).to_csv(weather_path, index=False)
    state, audit = cascade.prepare_round_state(rounds_path, weather_path)
    assert state["completed_rounds_available"].tolist() == [0, 1]
    assert state["completed_rounds_used_w2"].tolist() == [0, 1]
    assert state["completed_weather_observations_used"].tolist() == [0, 1]
    assert state["completed_weather_observations_available"].tolist() == [0, 1]
    assert state[
        "completed_weather_state_rows_with_valid_coverage"
    ].tolist() == [0, 1]
    assert state["completed_prior_weather_coverage_rate"].tolist() == [0.0, 1.0]
    assert state["target_or_upcoming_realized_weather_excluded"].eq(True).all()
    assert audit["completed_prior_round_boundary_violations"] == 0
    assert audit["weather_coefficients_sha256"] == hashlib.sha256(
        weather_path.read_bytes()
    ).hexdigest()


def test_w2_lineage_does_not_claim_missing_prior_weather_as_used(
    tmp_path: Path,
) -> None:
    rounds_path = tmp_path / "rounds_missing_weather.parquet"
    common = {
        "event_key": "e",
        "event_start_date": pd.Timestamp("2023-01-05"),
        "event_end_date": pd.Timestamp("2023-01-08"),
        "event_name": "Event",
        "year": 2023,
        "event_id": 1,
        "dg_id": 8,
        "player_name": "Missing Weather",
        "field_size": 100,
        "dg_sg_total_field_adjusted": 0.5,
        "cascade_actual": 0.5,
        "cascade_prediction": 0.4,
        "cascade_residual": 0.1,
        "fin_text": "T20",
        **{
            f"dg_sg_{category}_field_adjusted": 0.125
            for category in ("ott", "app", "arg", "putt")
        },
    }
    pd.DataFrame(
        [
            {
                **common,
                "round_num": 1,
                "round_date": pd.Timestamp("2023-01-05"),
                "score": 70,
                "start_pos": 1,
                "weather_delta_wind": np.nan,
                "weather_delta_dew": np.nan,
            },
            {
                **common,
                "round_num": 2,
                "round_date": pd.Timestamp("2023-01-06"),
                "score": 71,
                "start_pos": 20,
                "weather_delta_wind": 2.0,
                "weather_delta_dew": 1.0,
            },
        ]
    ).to_parquet(rounds_path, index=False)
    weather_path = tmp_path / "weather.csv"
    weather = {
        "event_start_date": ["2023-01-05"],
        "fit_max_round_date": ["2022-12-31"],
        "coefficient_source": ["strict_prior_synthetic"],
        "total_beta_wind": [0.1],
        "total_beta_dew": [0.05],
    }
    for category in ("ott", "app", "arg", "putt"):
        weather[f"{category}_beta_wind"] = [0.025]
        weather[f"{category}_beta_dew"] = [0.0125]
    pd.DataFrame(weather).to_csv(weather_path, index=False)
    state, _audit = cascade.prepare_round_state(rounds_path, weather_path)
    r2 = state.loc[state["round_num"].eq(2)].iloc[0]
    assert r2["completed_rounds_used_w2"] == 1
    assert r2["completed_weather_observations_available"] == 0
    assert r2["completed_weather_state_rows_with_valid_coverage"] == 0
    assert r2["completed_prior_weather_coverage_rate"] == 0.0

    digest = hashlib.sha256(weather_path.read_bytes()).hexdigest()
    lineage = pd.DataFrame(
        {
            "variant": ["C3_pt_continuous_partial_pool_w2"],
            "event_start_date": [pd.Timestamp("2023-01-05")],
            "round_num": [2],
            "weather_arm": ["w2"],
            "weather_state_target_column": ["target_total_w2"],
            "weather_state_source_columns": [
                "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2"
            ],
            "completed_rounds_available": [1],
            "completed_rounds_used": [1],
            "completed_weather_observations_available": [0],
            "completed_weather_state_rows_with_valid_coverage": [0],
            "completed_weather_observations_used": [0],
            "completed_weather_state_rows_used_by_w2": [0],
            "completed_weather_used_by_w2": [False],
            "completed_prior_weather_coverage_rate": [0.0],
            "completed_prior_round_boundary": [
                "round_num_strictly_less_than_target_round"
            ],
            "completed_prior_rounds_only": [True],
            "weather_coefficient_fit_cutoff": [pd.Timestamp("2022-12-31")],
            "weather_coefficient_source": ["strict_prior_synthetic"],
            "weather_coefficients_path": [str(weather_path.resolve())],
            "weather_coefficients_sha256": [digest],
            "target_or_upcoming_realized_weather_excluded": [True],
            "strict_point_in_time": [True],
            "current_round_weather_columns_role": [
                "posthoc_outcome_attribution_only_not_model_features"
            ],
        }
    )
    audit = cascade.audit_c3_weather_lineage(
        lineage,
        authoritative_coefficients_path=weather_path,
        authoritative_coefficients_sha256=digest,
    )
    assert not any(audit["failures"].values())
    lineage.loc[0, "completed_weather_used_by_w2"] = True
    with pytest.raises(ValueError, match="w2_weather_used_flag_mismatch"):
        cascade.audit_c3_weather_lineage(
            lineage,
            authoritative_coefficients_path=weather_path,
            authoritative_coefficients_sha256=digest,
        )


def test_c3_weather_lineage_rejects_target_weather_use_flag(tmp_path) -> None:
    coefficients = tmp_path / "weather_coefficients.csv"
    coefficients.write_text("synthetic", encoding="utf-8")
    digest = hashlib.sha256(coefficients.read_bytes()).hexdigest()
    frame = pd.DataFrame(
        {
            "variant": ["C3_pt_continuous_partial_pool_w2"],
            "event_start_date": [pd.Timestamp("2023-01-05")],
            "round_num": [1],
            "weather_arm": ["w2"],
            "weather_state_target_column": ["target_total_w2"],
            "weather_state_source_columns": [
                "target_total_w2|orth_ott_w2|orth_app_w2|orth_arg_w2|orth_putt_w2"
            ],
            "completed_rounds_available": [0],
            "completed_rounds_used": [0],
            "completed_weather_observations_available": [0],
            "completed_weather_state_rows_with_valid_coverage": [0],
            "completed_weather_observations_used": [0],
            "completed_weather_state_rows_used_by_w2": [0],
            "completed_weather_used_by_w2": [False],
            "completed_prior_weather_coverage_rate": [0.0],
            "completed_prior_round_boundary": [
                "round_num_strictly_less_than_target_round"
            ],
            "completed_prior_rounds_only": [True],
            "weather_coefficient_fit_cutoff": [pd.Timestamp("2022-12-31")],
            "weather_coefficient_source": ["strict_prior_synthetic"],
            "weather_coefficients_path": [str(coefficients.resolve())],
            "weather_coefficients_sha256": [digest],
            "target_or_upcoming_realized_weather_excluded": [False],
            "strict_point_in_time": [True],
            "current_round_weather_columns_role": [
                "posthoc_outcome_attribution_only_not_model_features"
            ],
        }
    )
    with pytest.raises(ValueError, match="target_or_upcoming_exclusion_false"):
        cascade.audit_c3_weather_lineage(
            frame,
            authoritative_coefficients_path=coefficients,
            authoritative_coefficients_sha256=digest,
        )


def test_cascade_weather_authority_rejects_unrelated_and_tampered_csv(
    tmp_path,
) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    round_panel = panel / "player_round_panel.parquet"
    weather = tmp_path / "weather.csv"
    unrelated = tmp_path / "unrelated.csv"
    bridge = tmp_path / "core.parquet"
    for path, value in (
        (round_panel, "rounds"),
        (weather, "weather"),
        (unrelated, "other"),
        (bridge, "bridge"),
    ):
        path.write_text(value, encoding="utf-8")

    def record(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "_verified_completion_inputs": {
            "round_panel": record(round_panel),
            "weather_coefficients": record(weather),
            "core_bridge": record(bridge),
        }
    }
    with pytest.raises(ValueError, match="weather_coefficients path"):
        cascade.validate_cascade_completion_authorities(
            manifest,
            panel_dir=panel,
            weather_coefficients=unrelated,
            core_bridge=bridge,
        )
    weather.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: weather_coefficients"):
        cascade.validate_cascade_completion_authorities(
            manifest,
            panel_dir=panel,
            weather_coefficients=weather,
            core_bridge=bridge,
        )


def _fake_cascade_provenance(paths: list[Path]) -> SimpleNamespace:
    return SimpleNamespace(
        input_fingerprints=tuple({"path": str(path.resolve())} for path in paths)
    )


def test_cascade_post_begin_revalidation_rejects_valid_path_switch(
    tmp_path, monkeypatch
) -> None:
    expected_panel = [tmp_path / "panel_manifest", tmp_path / "rounds"]
    expected_odds = [tmp_path / "odds_manifest", tmp_path / "shard"]
    expected_handoff = [tmp_path / "handoff"]
    switched = tmp_path / "switched"
    for path in [*expected_panel, *expected_odds, *expected_handoff, switched]:
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(
        cascade, "_verified_panel_provenance_paths", lambda _path: expected_panel
    )
    monkeypatch.setattr(
        cascade, "_odds_cache_provenance_paths", lambda _path: expected_odds
    )
    monkeypatch.setattr(
        cascade,
        "_validate_finalist_manifest",
        lambda _path: ({"_verified_handoff_paths": [str(switched)]}, switched),
    )
    with pytest.raises(ValueError, match="handoff resolved path set changed"):
        cascade.revalidate_cascade_dynamic_inputs_after_begin(
            panel_dir=tmp_path,
            expected_panel_paths=expected_panel,
            odds_cache=tmp_path,
            expected_odds_paths=expected_odds,
            finalist_dir=tmp_path,
            expected_handoff_paths=expected_handoff,
            provenance_state=_fake_cascade_provenance(
                [*expected_panel, *expected_odds, *expected_handoff, switched]
            ),
        )


def test_panel_authority_rejects_self_consistent_forged_output(
    tmp_path, monkeypatch
) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    output_keys = {
        "player_event_outcomes": "player_event_outcomes.parquet",
        "odds_event_crosswalk": "odds_event_crosswalk.parquet",
        "player_round_panel": "player_round_panel.parquet",
    }
    outputs = {}
    for key, name in output_keys.items():
        path = panel / name
        path.write_text(key, encoding="utf-8")
        outputs[key] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    odds_hash = "c" * 64
    manifest_path = panel / "manifest.json"
    payload = {
        "purpose": (
            "Research-only frozen outcome panel for walk-forward base-skill and "
            "market evaluation."
        ),
        "production_changes": False,
        "parameters": {
            "end": "2026-08-22",
            "market_start": "2020-01-01",
            "tour": "pga",
        },
        "inputs": {"odds_cache_manifest": {"sha256": odds_hash}},
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(cascade, "EXPECTED_PANEL_OUTPUT_KEYS", frozenset(output_keys))
    monkeypatch.setattr(
        cascade,
        "EXPECTED_PANEL_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        cascade, "EXPECTED_PANEL_MANIFEST_BYTES", manifest_path.stat().st_size
    )
    monkeypatch.setattr(
        cascade, "EXPECTED_PANEL_ODDS_CACHE_MANIFEST_SHA256", odds_hash
    )
    assert len(cascade._verified_panel_provenance_paths(panel)) == 4

    changed = panel / "player_round_panel.parquet"
    changed.write_text("forged rows", encoding="utf-8")
    payload["outputs"]["player_round_panel"]["sha256"] = hashlib.sha256(
        changed.read_bytes()
    ).hexdigest()
    payload["outputs"]["player_round_panel"]["size_bytes"] = changed.stat().st_size
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="panel manifest (byte size|hash)"):
        cascade._verified_panel_provenance_paths(panel)


def test_cascade_odds_authority_rejects_self_consistent_forged_shard(
    tmp_path, monkeypatch
) -> None:
    cache = tmp_path / "odds"
    cache.mkdir()
    shards = []
    for index in range(22):
        path = cache / f"shard_{index}.jsonl"
        path.write_text(str(index), encoding="utf-8")
        shards.append(
            {
                "data_path": path.name,
                "family": "event_list" if index == 0 else "matchups",
                "row_count": 348 if index == 0 else (146_351 if index == 1 else 0),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "complete": True,
        "tour": "pga",
        "mode": "full",
        "start_date": "2020-01-01",
        "end_date": "2026-08-22",
        "row_counts": {"event_list": 348, "matchups": 146_351},
        "active_shards": shards,
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        cascade,
        "EXPECTED_PANEL_ODDS_CACHE_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        cascade,
        "EXPECTED_PANEL_ODDS_CACHE_MANIFEST_BYTES",
        manifest_path.stat().st_size,
    )
    assert len(cascade._odds_cache_provenance_paths(cache)) == 23

    first = cache / "shard_0.jsonl"
    first.write_text("forged", encoding="utf-8")
    payload["active_shards"][0]["sha256"] = hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest (byte size|differs)"):
        cascade._odds_cache_provenance_paths(cache)


def test_frozen_diagnostic_contract_maps_canonical_event_and_keeps_label(
    tmp_path, monkeypatch
):
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    prediction_path = frozen_dir / "player_event_predictions.parquet"
    candidate = "complete__ridge__total__w0__annual__a1"
    pd.DataFrame(
        {
            "event_key": ["pga:R2023001"],
            "dg_id": [7],
            "event_start": pd.to_datetime(["2023-01-05"]),
            "candidate": [candidate],
            "finalist_core": [0.25],
            "control_core": [0.10],
            "split": ["sealed"],
            "prediction_asof": pd.to_datetime(["2023-01-05"]),
            "chronology_ok": [True],
        }
    ).to_parquet(prediction_path, index=False)
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    source_path = tmp_path / "all_candidate_predictions.parquet"
    pd.DataFrame(
        {
            "candidate": [
                candidate,
                "current_architecture_pt_v2_approx_control",
            ],
            "dg_id_key": ["7", "7"],
            "event_start_date": pd.to_datetime(["2023-01-05", "2023-01-05"]),
            "prediction": [0.25, 0.10],
            "history_rounds": [42, 42],
            "days_since_last_round": [7, 7],
            "reliability_ok": [True, True],
            "cross_tour_fallback_reason": ["", ""],
            "img_history_rounds": [20, 20],
            "img_history_coverage": [0.8, 0.8],
            "prior_event_withdrawal": [False, False],
        }
    ).to_parquet(source_path, index=False)
    manifest = {
        "contract": "base_skill_frozen_finalist_v1",
        "frozen": True,
        "research_only": True,
        "finalist_name": None,
        "selected_model_name": candidate,
        "selection_role": "diagnostic_challenger",
        "promotion_gate_passed": False,
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_hash,
        "prediction_units": "sg_per_round",
        "control_name": "current_architecture_pt_v2_approx_control",
        "control_units": "sg_per_round",
        "control_core_point_in_time": True,
        "control_core_uses_sealed_future": False,
        "selection_window_end": "2022-12-31",
        "sealed_data_used_for_selection": False,
        "frozen_before_sealed_scoring": True,
        "uses_closing_odds_as_feature": False,
        "source_prediction_file": str(source_path),
        "source_prediction_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    }
    (frozen_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        cascade,
        "validate_frozen_handoff_chain",
        lambda _: SimpleNamespace(
            compact=manifest,
            selected_name=candidate,
            selection_role="diagnostic_challenger",
            promotion_gate_passed=False,
            audit={"full_hash_chain_verified": True},
            compact_prediction_path=prediction_path,
            source_prediction_path=source_path,
            downstream_lambda=0.5,
            core_bridge_path=tmp_path / "core_bridge.parquet",
            core_bridge_sha256="synthetic",
            verified_paths=(prediction_path, source_path),
        ),
    )
    event_map = pd.DataFrame(
        {
            "event_key": ["pga|2023|1|2023-01-05"],
            "year": [2023],
            "event_id": [1],
            "event_start_date": pd.to_datetime(["2023-01-05"]),
            "dg_id_key": ["7"],
        }
    )
    predictions, selection, audit, detail = load_frozen_finalist_predictions(
        frozen_dir, event_map
    )
    assert set(predictions["candidate"]) == {
        candidate,
        "current_architecture_pt_v2_approx_control",
    }
    assert predictions["event_key"].eq("pga|2023|1|2023-01-05").all()
    selected = selection.loc[selection["candidate"].eq(candidate)].iloc[0]
    assert selected["candidate_type"] == "diagnostic_challenger"
    assert not bool(selected["promotion_gate_passed"])
    assert audit["selection_role"] == "diagnostic_challenger"
    assert audit["promotion_gate_passed"] is False
    assert detail["source_detail_available"] is True
    assert detail["detail_coverage_rate"] == 1.0


def test_live_lambda_zero_is_no_live_and_one_is_full_current_update():
    frame = pd.DataFrame(
        {
            "event_key": ["event"],
            "event_name": ["Event"],
            "event_start_date": pd.to_datetime(["2021-01-07"]),
            "event_end_date": pd.to_datetime(["2021-01-10"]),
            "dg_id_key": ["7"],
            "dg_id": [7],
            "player_name": ["Player"],
            "round_num": [2],
            "round_date": pd.to_datetime(["2021-01-08"]),
            "round_score": [70],
            "weather_delta_wind": [1.0],
            "weather_delta_dew": [2.0],
            "weather_observed": [True],
            "event_withdrawal": [False],
            "start_pos_scaled": [0.5],
            "base_core": [0.4],
            "base_delta": [0.1],
            "base_event_hybrid": [0.8],
            "legacy_stage_adjustment": [0.3],
        }
    )
    no_live = _live_lambda_player_predictions(
        frame,
        candidate="candidate",
        lambdas_by_round={2: 0.0},
        model="no_live",
        variant="test",
    )
    full_live = _live_lambda_player_predictions(
        frame,
        candidate="candidate",
        lambdas_by_round={2: 1.0},
        model="full_live",
        variant="test",
    )
    assert no_live.iloc[0]["prediction"] == 0.8
    assert full_live.iloc[0]["prediction"] == 1.1


def test_live_lambda_tuning_never_passes_sealed_rows_to_selector(monkeypatch):
    def fake_player(
        candidate_frame,
        *,
        candidate,
        lambdas_by_round,
        model,
        variant,
    ):
        return pd.DataFrame(
            {
                "model": [model],
                "candidate": [candidate],
                "variant": [variant],
                "round_num": [next(iter(lambdas_by_round))],
                "live_update_lambda": [next(iter(lambdas_by_round.values()))],
            }
        )

    def fake_pairs(player, quotes):
        value = float(player["live_update_lambda"].iloc[0])
        return pd.DataFrame(
            {
                "event_start_date": pd.to_datetime(
                    ["2021-06-01", "2022-06-01", "2025-06-01"]
                ),
                "_lambda": [value, value, value],
            }
        )

    def fake_score(discovery):
        assert discovery["event_start_date"].max() <= pd.Timestamp("2022-12-31")
        value = float(discovery["_lambda"].iloc[0])
        return 2.8, 0.3, (value - 0.25) ** 2, len(discovery), 2

    monkeypatch.setattr(cascade, "_live_lambda_player_predictions", fake_player)
    monkeypatch.setattr(cascade, "player_predictions_to_pairs", fake_pairs)
    monkeypatch.setattr(cascade, "fit_sigma_and_loss", fake_score)
    quotes = pd.DataFrame({"round_num": [1, 2, 3, 4]})
    output, tuning = cascade.tune_live_update_shrinkage(
        pd.DataFrame(), quotes, candidate="candidate"
    )
    selected = tuning.loc[tuning["selected"]]
    assert selected["live_update_lambda"].tolist() == [0.25] * 4
    assert selected["sealed_rows_used_for_selection"].sum() == 0
    assert set(output["live_update_lambda"]) == {0.25}


def test_open_and_close_are_distinct_execution_counterfactuals():
    scored = pd.DataFrame(
        [
            {
                "model": "m",
                "candidate": "c",
                "variant": "v",
                "strict_point_in_time": True,
                "retrospective_cascade_anchor": False,
                "event_key": "event",
                "event_name": "Event",
                "event_start_date": pd.Timestamp("2025-01-01"),
                "round_market": "r1",
                "round_num": 1,
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_player_name": "One",
                "p2_player_name": "Two",
                "book": "pinnacle",
                "p1_open": 120.0,
                "p2_open": -130.0,
                "p1_close": -110.0,
                "p2_close": 100.0,
                "open_time": "2025-01-01 12:00:00",
                "close_time": "2025-01-02 12:00:00",
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "p1_outcome_text": "win",
                "p2_outcome_text": "loss",
                "tie_rule": "void",
                "p1_probability": 0.55,
                "p2_probability": 0.40,
                "push_probability": 0.05,
                "pricing_p1_probability": 0.55 / 0.95,
                "p1_open_devig_probability": 0.48,
                "p1_close_devig_probability": 0.51,
                "p1_skill": 0.5,
                "p2_skill": 0.0,
                "p1_base_core": 0.5,
                "p2_base_core": 0.0,
                "abs_realized_wind_difference": 1.0,
                "abs_realized_dew_difference": 2.0,
                "pair_weather_observed": True,
            }
        ]
    )
    opening = expand_betting_lines(scored, "open")
    closing = expand_betting_lines(scored, "close")
    open_p1 = opening.loc[opening["selection_id"].eq("1")].iloc[0]
    close_p1 = closing.loc[closing["selection_id"].eq("1")].iloc[0]
    assert open_p1["price_point"] == "open"
    assert close_p1["price_point"] == "close"
    assert open_p1["execution_odds"] == 120.0
    assert close_p1["execution_odds"] == -110.0
    assert open_p1["archived_close_odds"] == -110.0
    assert close_p1["archived_open_odds"] == 120.0


def test_quote_capture_orientation_is_canonical_before_endpoint_collapse():
    frame = pd.DataFrame(
        {
            "p1_id": ["2"],
            "p2_id": ["1"],
            "p1_player_name": ["Two"],
            "p2_player_name": ["One"],
            "p1_open": [120.0],
            "p2_open": [-130.0],
            "p1_close": [110.0],
            "p2_close": [-120.0],
            "p1_outcome": [1.0],
            "p2_outcome": [0.0],
            "p1_outcome_text": ["win"],
            "p2_outcome_text": ["loss"],
        }
    )
    canonical, swaps = _canonicalize_quote_sides(frame)
    assert swaps == 1
    assert canonical.loc[0, "p1_id"] == "1"
    assert canonical.loc[0, "p1_player_name"] == "One"
    assert canonical.loc[0, "p1_open"] == -130.0
    assert canonical.loc[0, "p1_outcome"] == 0.0


def test_ties_lose_rule_moves_push_mass_to_loss_in_ev_and_settlement():
    scored = pd.DataFrame(
        [
            {
                "model": "m",
                "candidate": "c",
                "variant": "v",
                "strict_point_in_time": True,
                "retrospective_cascade_anchor": False,
                "event_key": "e",
                "event_name": "E",
                "event_start_date": pd.Timestamp("2025-01-01"),
                "round_market": "r2",
                "round_num": 2,
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_player_name": "One",
                "p2_player_name": "Two",
                "book": "pinnacle",
                "p1_open": 100.0,
                "p2_open": 100.0,
                "p1_close": 100.0,
                "p2_close": 100.0,
                "open_time": "2025-01-01",
                "close_time": "2025-01-02",
                "p1_outcome": 0.0,
                "p2_outcome": 0.0,
                "p1_outcome_text": "loss",
                "p2_outcome_text": "loss",
                "actual_score_tie": True,
                "no_score_void": False,
                "other_void_with_scores": False,
                "tie_rule": "separate; ties lose",
                "p1_probability": 0.45,
                "p2_probability": 0.45,
                "push_probability": 0.10,
                "p1_open_devig_probability": 0.5,
                "p1_close_devig_probability": 0.5,
                "p1_skill": 0.2,
                "p2_skill": 0.0,
                "p1_base_core": 0.2,
                "p2_base_core": 0.0,
            }
        ]
    )
    lines = expand_betting_lines(scored, "close")
    p1 = lines.loc[lines["selection_id"].eq("1")].iloc[0]
    assert p1["effective_push_probability"] == 0.0
    assert p1["loss_probability"] == pytest.approx(0.55)
    assert p1["ev_per_unit"] == pytest.approx(-0.10)
    assert not bool(p1["market_devig_price_comparison_eligible"])
    assert p1["market_devig_price_comparison_exclusion_reason"] == (
        "ties_lose_without_draw_quote"
    )
    assert np.isnan(p1["model_minus_devig_price_pp"])
    settled = select_and_settle(
        lines, line_policy="pinnacle", sizing="flat_1u", min_ev=-np.inf
    )
    assert len(settled) == 1
    assert settled.iloc[0]["settlement_status"] == "tie_loss"
    assert settled.iloc[0]["pnl"] == -1.0


def test_push_on_tie_price_comparison_remains_eligible() -> None:
    scored = pd.DataFrame(
        [
            {
                "model": "m",
                "candidate": "c",
                "variant": "v",
                "event_key": "e",
                "event_name": "E",
                "event_start_date": pd.Timestamp("2025-01-01"),
                "round_market": "r2",
                "round_num": 2,
                "pair_key": "1|2",
                "p1_id": "1",
                "p2_id": "2",
                "p1_player_name": "One",
                "p2_player_name": "Two",
                "book": "pinnacle",
                "p1_close": 100.0,
                "p2_close": 100.0,
                "close_time": "2025-01-02",
                "p1_outcome": 1.0,
                "p2_outcome": 0.0,
                "actual_score_tie": False,
                "no_score_void": False,
                "other_void_with_scores": False,
                "tie_rule": "void on tie",
                "p1_probability": 0.45,
                "p2_probability": 0.45,
                "push_probability": 0.10,
                "p1_close_devig_probability": 0.50,
                "p1_skill": 0.2,
                "p2_skill": 0.0,
                "p1_base_core": 0.2,
                "p2_base_core": 0.0,
            }
        ]
    )
    lines = expand_betting_lines(scored, "close")
    p1 = lines.loc[lines["selection_id"].eq("1")].iloc[0]
    assert bool(p1["market_devig_price_comparison_eligible"])
    assert p1["market_devig_price_comparison_exclusion_reason"] == ""
    assert p1["pricing_probability"] == pytest.approx(0.50)
    assert p1["model_minus_devig_price_pp"] == pytest.approx(0.0)


def test_ties_lose_three_way_price_is_excluded_from_market_proper_score() -> None:
    classified = classify_score_outcomes(
        pd.DataFrame(
            {
                "p1_round_score": [69.0],
                "p2_round_score": [71.0],
                "p1_outcome": [1.0],
                "p2_outcome": [0.0],
                "tie_rule": ["3 way; ties lose"],
            }
        )
    )
    row = classified.assign(
        event_key="e",
        quote_id="q",
        pair_key="1|2",
        p1_probability=0.45,
        p2_probability=0.45,
        push_probability=0.10,
        pricing_p1_probability=0.45,
        p1_close=110.0,
        p2_close=110.0,
        p1_close_devig_probability=0.50,
        model_minus_devig_close_pp=np.nan,
    )
    assert not bool(row.loc[0, "market_devig_proper_score_eligible"])
    assert "ties_lose_without_draw_quote" in row.loc[
        0, "market_devig_proper_score_exclusion_reason"
    ]
    algebra = cascade.audit_pricing_probability_algebra(row)
    assert algebra["pricing_identity_max_abs_error"] == 0.0
    proper = cascade._proper_group(row, "close")
    assert proper["ties_lose_rows_excluded_from_market_benchmark"] == 1
    assert proper["resolved_quotes"] == 0
    assert np.isnan(proper["event_equal_devig_price_logloss"])
    assert cascade._calibration_group(row) == {}


def test_pricing_probability_algebra_rejects_tampered_value() -> None:
    frame = pd.DataFrame(
        {
            "p1_probability": [0.45, 0.45, 0.60],
            "p2_probability": [0.45, 0.45, 0.40],
            "push_probability": [0.10, 0.10, 0.0],
            "tie_rule_ties_lose": [True, False, False],
            "pricing_p1_probability": [0.45, 0.50, 0.60],
        }
    )
    audit = cascade.audit_pricing_probability_algebra(frame)
    assert audit["pricing_identity_max_abs_error"] == pytest.approx(0.0)
    frame.loc[1, "pricing_p1_probability"] = 0.99
    with pytest.raises(AssertionError, match="pricing algebra"):
        cascade.audit_pricing_probability_algebra(frame)


def test_ungraded_selected_round_rows_are_preserved_for_completeness_gate():
    lines = pd.DataFrame(
        {
            "model": ["m"],
            "event_key": ["e"],
            "event_id": ["e"],
            "round_market": ["r1"],
            "pair_key": ["1|2"],
            "selection_id": ["1"],
            "opponent_id": ["2"],
            "book": ["pinnacle"],
            "close_time": ["2025-01-02"],
            "close_odds": [110.0],
            "ev_per_unit": [0.1],
            "market": ["tournament_matchup"],
            "outcome_numeric": [np.nan],
            "outcome": [np.nan],
            "tie_rule": ["void"],
        }
    )
    settled = select_and_settle(
        lines, line_policy="pinnacle", sizing="flat_1u", min_ev=0.0
    )
    assert len(settled) == 1
    assert not bool(settled.iloc[0]["is_graded"])
    assert settled.iloc[0]["settlement_status"] == "ungraded"


def test_best_price_proper_score_row_retains_same_book_opponent_quote():
    rows = pd.DataFrame(
        {
            "event_key": ["e", "e"],
            "round_market": ["r1", "r1"],
            "pair_key": ["1|2", "1|2"],
            "book": ["pinnacle", "betcris"],
            "p1_open": [120.0, 110.0],
            "p2_open": [-130.0, -115.0],
            "open_time": ["2025-01-01", "2025-01-01"],
        }
    )
    chosen = _best_price_market_rows(rows, "open")
    assert len(chosen) == 1
    assert chosen.iloc[0]["book"] == "pinnacle"
    assert chosen.iloc[0]["p1_open"] == 120.0
    assert chosen.iloc[0]["p2_open"] == -130.0


def test_cascade_complete_manifest_rejects_post_finalize_input_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cascade.py"
    direct_input = tmp_path / "cascade_input.parquet"
    source.write_text("# synthetic cascade\n", encoding="utf-8")
    direct_input.write_bytes(b"before")
    output = tmp_path / "cascade_output"
    state = cascade.execution_provenance.begin(
        output_dir=output,
        primary_source=source,
        input_paths=[direct_input],
    )
    cascade.execution_provenance.finalize(state)
    direct_input.write_bytes(b"mutated after finalize")
    manifest_path = output / "manifest.json"
    with pytest.raises(RuntimeError, match="input changed"):
        cascade.write_cascade_complete_manifest(
            manifest_path, {"complete": True}, state
        )
    assert not manifest_path.exists()

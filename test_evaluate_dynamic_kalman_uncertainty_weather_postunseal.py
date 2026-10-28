from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive import build_complete_base_skill_candidates as builder
from archive import evaluate_dynamic_kalman_uncertainty_weather_postunseal as module


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tour": ["pga", "pga", "euro"],
            "round_num": [1, 2, 1],
            "round_date": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-10"]
            ),
            "sg_total_adj": [1.0, -0.5, 0.75],
            "weather_exact_match": [True, True, False],
            "weather_available": [True, False, False],
            "weather_delta_wind": [2.0, 5.0, np.nan],
            "weather_delta_dew": [-1.0, 3.0, np.nan],
        }
    )


def test_w0_replay_preserves_legacy_mean_and_ages_variance_only() -> None:
    history = _history()
    offsets = {"pga": 0.0, "euro": 0.25}
    variances = {"pga": 4.0, "euro": 6.0}
    expected_mean = builder._kalman_state(
        history,
        offsets,
        variances,
        prior_mean=0.1,
        prior_rounds=20.0,
        process_variance_per_day=0.0005,
    )
    result = module.kalman_state_with_target_variance(
        history,
        offsets,
        variances,
        prior_mean=0.1,
        target=pd.Timestamp("2022-01-20"),
    )
    assert result["mean"] == pytest.approx(expected_mean, abs=1e-15)
    assert result["target_age_days"] == 10
    assert result["target_variance"] == pytest.approx(
        result["posterior_variance_last_observation"] + 10 * 0.0005
    )


def test_empty_history_has_target_asof_prior_variance_without_fake_age() -> None:
    history = _history().iloc[:0]
    result = module.kalman_state_with_target_variance(
        history,
        {"pga": 0.0},
        {"pga": 5.0},
        prior_mean=-0.2,
        target=pd.Timestamp("2022-01-20"),
    )
    assert result["mean"] == -0.2
    assert result["target_variance"] == pytest.approx(0.25)
    assert result["target_age_days"] == 0


def test_w1_adjusts_only_exact_available_pga_rows_with_correct_direction() -> None:
    history = _history()
    coefficient = {"total_beta_wind": 0.2, "total_beta_dew": -0.1}
    values = module.w1_values(history, coefficient)
    # Positive wind delta means worse conditions, so adding beta*delta
    # neutralizes the expected observed-SG penalty.
    assert values[0] == pytest.approx(1.0 + 0.2 * 2.0 + (-0.1) * -1.0)
    assert values[1] == pytest.approx(-0.5)  # incomplete weather remains W0
    assert values[2] == pytest.approx(0.75)  # non-PGA remains W0


def test_kalman_fails_closed_on_nonprior_observation() -> None:
    history = _history()
    with pytest.raises(ValueError, match="non-prior"):
        module.kalman_state_with_target_variance(
            history,
            {"pga": 0.0, "euro": 0.0},
            {"pga": 5.0, "euro": 5.0},
            prior_mean=0.0,
            target=pd.Timestamp("2022-01-10"),
        )


def test_exact_weather_join_leaves_unmatched_rows_w0(tmp_path: Path) -> None:
    rounds = pd.DataFrame(
        {
            "dg_id": [1, 2, 3],
            "year": [2022, 2022, 2022],
            "event_id": [10, 10, 20],
            "round_num": [1, 1, 1],
            "round_date": pd.to_datetime(["2022-01-01"] * 3),
            "course_num": [1, 1, 1],
            "tour": ["pga", "pga", "euro"],
            "sg_total_adj": [1.0, 2.0, 3.0],
        }
    )
    panel = pd.DataFrame(
        {
            "dg_id": [1],
            "year": [2022],
            "event_id": [10],
            "round_num": [1],
            "round_date": pd.to_datetime(["2022-01-01"]),
            "course_num": [1],
            "dg_sg_total_field_adjusted": [1.0],
            "weather_delta_wind": [2.0],
            "weather_delta_dew": [-1.0],
        }
    )
    path = tmp_path / "rounds.parquet"
    panel.to_parquet(path, index=False)
    joined, audit = module.attach_exact_pga_weather(
        rounds, path, target_dates=[pd.Timestamp("2022-01-02")]
    )
    keyed = joined.set_index("dg_id")
    assert bool(keyed.loc[1, "weather_available"])
    assert not bool(keyed.loc[2, "weather_available"])
    assert not bool(keyed.loc[3, "weather_available"])
    assert audit["unmatched_pga_rows_left_w0"] == 1
    assert audit["non_pga_rows_left_w0"] == 1
    assert audit["weather_panel_rows_loaded"] == 1
    assert audit["target_or_upcoming_weather_rows_loaded"] == 0
    assert not audit["target_or_upcoming_weather_loaded"]


def test_weather_audit_separates_loaded_future_rows_from_feature_use(
    tmp_path: Path,
) -> None:
    rounds = pd.DataFrame(
        {
            "dg_id": [1],
            "year": [2022],
            "event_id": [10],
            "round_num": [1],
            "round_date": pd.to_datetime(["2022-01-01"]),
            "course_num": [1],
            "tour": ["pga"],
            "sg_total_adj": [1.0],
        }
    )
    panel = pd.DataFrame(
        {
            "dg_id": [1, 2, 3],
            "year": [2022, 2022, 2022],
            "event_id": [10, 11, 12],
            "round_num": [1, 1, 1],
            "round_date": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03"]
            ),
            "course_num": [1, 1, 1],
            "dg_sg_total_field_adjusted": [1.0, 0.0, 0.0],
            "weather_delta_wind": [2.0, 3.0, 4.0],
            "weather_delta_dew": [-1.0, 0.0, 1.0],
        }
    )
    path = tmp_path / "rounds_with_future.parquet"
    panel.to_parquet(path, index=False)
    joined, audit = module.attach_exact_pga_weather(
        rounds, path, target_dates=[pd.Timestamp("2022-01-02")]
    )
    assert len(joined) == 1
    assert audit["weather_panel_rows_loaded"] == 3
    assert audit["target_or_upcoming_weather_rows_loaded"] == 2
    assert audit["target_or_upcoming_weather_loaded"]
    assert "target_or_upcoming_weather_read" not in audit


def test_settlement_summary_counts_void_ties_as_refunds() -> None:
    bets = pd.DataFrame(
        {
            "candidate": ["arm"] * 4,
            "round_market": ["72h"] * 4,
            "event_key": ["e1", "e1", "e2", "e2"],
            "event_start_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-01-08", "2022-01-08"]
            ),
            "stake": [1.0] * 4,
            "pnl": [1.0, -1.0, 0.0, 0.0],
            "ev_per_unit": [0.04] * 4,
            "open_to_close_clv_pp": [0.0] * 4,
            "outcome_numeric": [1.0, 0.0, 0.5, 0.5],
            "settlement_status": ["win", "loss", "void", "push"],
        }
    )
    summary = module._summarize_bets(
        bets,
        sample="sample",
        price_basis="close",
        line_policy="best_of_three",
        sizing="flat_1u",
    ).iloc[0]
    assert summary["pushes"] == 1
    assert summary["voids"] == 1
    assert summary["refunded_bets"] == 2
    assert summary["tie_refunds"] == 2


def test_effective_sigma_is_pair_specific_and_monotone_in_variance() -> None:
    frame = pd.DataFrame(
        {
            "skill_difference": [0.5, 0.5],
            "pair_target_variance": [0.1, 2.0],
        }
    )
    scored = module._score_with_parameters(
        frame,
        "arm",
        base_sigma=1.0,
        tie_threshold=0.1,
        alpha=2.0,
        fit_start=pd.Timestamp("2020-01-01"),
        fit_end=pd.Timestamp("2022-12-31"),
        mapper_type="test",
    )
    assert scored.loc[1, "sigma"] > scored.loc[0, "sigma"]
    assert scored.loc[1, "p1_probability"] < scored.loc[0, "p1_probability"]
    assert np.allclose(
        scored["p1_probability"]
        + scored["push_probability"]
        + norm_loss(frame["skill_difference"], scored["sigma"], 0.1),
        1.0,
    )


def norm_loss(
    difference: pd.Series, sigma: pd.Series, tie_threshold: float
) -> np.ndarray:
    from scipy.stats import norm

    return norm.cdf((-difference.to_numpy(float) - tie_threshold) / sigma.to_numpy(float))


def test_nested_mapper_loss_reduces_to_global_when_alpha_zero() -> None:
    difference = np.array([-1.0, 0.0, 1.0])
    variance = np.array([0.1, 1.0, 10.0])
    outcome = np.array([0.0, 0.5, 1.0])
    weights = np.ones(3)
    left = module._mapper_loss(
        difference, variance, outcome, weights, 2.0, 0.25, 0.0
    )
    right = module._mapper_loss(
        difference, np.zeros(3), outcome, weights, 2.0, 0.25, 99.0
    )
    assert left == pytest.approx(right, abs=1e-15)


def test_smoke_subset_is_deterministic_by_year() -> None:
    frame = pd.DataFrame(
        {
            "event_key": ["b", "a", "c", "d", "e", "f"],
            "event_start_date": pd.to_datetime(
                [
                    "2020-02-01",
                    "2020-01-01",
                    "2020-03-01",
                    "2021-01-01",
                    "2021-03-01",
                    "2021-02-01",
                ]
            ),
        }
    )
    subset = module._smoke_subset(frame, 2)
    assert set(subset["event_key"]) == {"a", "b", "d", "f"}

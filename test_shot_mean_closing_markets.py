from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from archive.backtest_shot_mean_closing_markets import (
    CAP_ABS_SG,
    EventMeta,
    add_zero_bet_event_clusters,
    arm_deltas,
    cut_line_for_event,
    event_format_exclusion,
    normalize_event_audit_rows,
    normalize_matchup_lines,
    normalize_outright_lines,
    probabilities_for_lines,
    recenter_factor_on_archived_roster,
    simulate_scores,
)


def test_odds_normalization_expands_matchup_sides_and_keeps_year_identity() -> None:
    matchup = normalize_matchup_lines(
        [
            {
                "season": 2025,
                "event_id": 9,
                "event_name": "Test",
                "book": "Betcris",
                "bet_type": "tournament_matchups",
                "close_time": "2025-03-06T12:00:00Z",
                "tie_rule": "void",
                "p1_dg_id": 1,
                "p1_player_name": "One",
                "p1_close": -110,
                "p1_outcome": 1,
                "p1_outcome_text": "win",
                "p2_dg_id": 2,
                "p2_player_name": "Two",
                "p2_close": 100,
                "p2_outcome": 0,
                "p2_outcome_text": "loss",
            }
        ]
    )
    assert len(matchup) == 2
    assert matchup["event_key"].unique().tolist() == ["pga:R2025009"]
    assert set(matchup["selection_id"]) == {"1", "2"}
    assert set(matchup["opponent_id"]) == {"1", "2"}

    outright = normalize_outright_lines(
        [
            {
                "season": 2024,
                "event_id": 9,
                "book": "BetOnline",
                "market": "top20",
                "dg_id": 1,
                "player_name": "One",
                "close_odds": 250,
                "bet_outcome_numeric": 0.5,
            }
        ]
    )
    assert outright.loc[0, "event_key"] == "pga:R2024009"
    assert outright.loc[0, "market"] == "top_20"
    assert outright.loc[0, "opponent_id"] == ""


def test_delta_arms_preserve_walk_forward_and_proportionally_cap() -> None:
    field = pd.DataFrame(
        {
            "factor_centered": [-2.0, 0.0, 2.0],
            "factor_delta_sg_per_round": [-0.30, 0.05, 0.25],
        }
    )
    output = arm_deltas(
        field,
        ["walk_forward_uncapped", "walk_forward_cap015", "fixed_beta_sensitivity"],
    )
    assert output["walk_forward_uncapped"].tolist() == pytest.approx([-0.30, 0.05, 0.25])
    assert np.max(np.abs(output["walk_forward_cap015"])) == pytest.approx(CAP_ABS_SG)
    assert output["walk_forward_cap015"].sum() == pytest.approx(0.0)
    assert output["fixed_beta_sensitivity"].sum() == pytest.approx(0.0)


def test_factor_is_recentered_on_exact_archived_roster_and_missing_is_neutral() -> None:
    field = pd.DataFrame(
        {
            "factor_raw": [8.0, 10.0, np.nan],
            "coefficient_sg_per_unit": [-0.1, -0.1, np.nan],
        }
    )
    output = recenter_factor_on_archived_roster(field)
    assert output["factor_available"].tolist() == [True, True, False]
    assert output["factor_centered"].tolist() == pytest.approx([-1.0, 1.0, 0.0])
    assert output["factor_delta_sg_per_round"].sum() == pytest.approx(0.0)
    assert output.loc[2, "factor_delta_sg_per_round"] == 0.0


def test_bootstrap_input_includes_zero_bet_eligible_event_for_each_arm() -> None:
    bets = pd.DataFrame(
        {
            "event_id": ["a", "a"],
            "event_start": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "arm": ["baseline", "challenger"],
            "stake": [1.0, 1.0],
            "pnl": [-1.0, 1.0],
        }
    )
    lines = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "event_start": pd.to_datetime(["2025-01-01", "2025-01-08"]),
        }
    )
    output = add_zero_bet_event_clusters(bets, lines)
    zero = output[output["event_id"].eq("b")]
    assert set(zero["arm"]) == {"baseline", "challenger"}
    assert zero["stake"].sum() == 0.0
    assert zero["pnl"].sum() == 0.0


def test_simulation_applies_identical_delta_to_app_and_all_total_means(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_kernel(mu, std, skew, chol, *common):
        captured["mu"] = np.asarray(mu)
        captured["std"] = np.asarray(std)
        captured["total_means"] = common[:4]
        n = len(mu)
        sims = int(common[-3])
        return (np.zeros((n, sims), dtype=np.int64),)

    monkeypatch.setitem(sys.modules, "sims_kernel", types.SimpleNamespace(run_pretournament=fake_kernel))
    params = {
        "mu": np.array([[0.1, 0.2, 0.3, 0.4], [-0.4, -0.3, -0.2, -0.1]]),
        "std_shot": np.ones((2, 4)),
        "skew": np.zeros((2, 4)),
    }
    field = pd.DataFrame({"skill_est": [1.0, -1.0]})
    delta = np.array([0.2, -0.1])
    coefficient_template = {
        "coefficients_r1_high": {"ott": 0, "putt": 0, "residual": 0, "residual2": 0},
        "coefficients_r1_midh": {"ott": 0, "putt": 0, "residual": 0, "residual2": 0},
        "coefficients_r1_midl": {"ott": 0, "putt": 0, "residual": 0, "residual2": 0},
        "coefficients_r1_low": {"ott": 0, "putt": 0, "residual": 0, "residual2": 0},
        "coefficients_r2": {
            "residual": 0,
            "residual2": 0,
            "residual3": 0,
            "avg_ott": 0,
            "avg_putt": 0,
            "avg_app": 0,
            "avg_arg": 0,
            "delta_app": 0,
        },
        "coefficients_r2_6_30": {
            "residual": 0,
            "residual2": 0,
            "residual3": 0,
            "avg_ott": 0,
            "avg_putt": 0,
            "avg_app": 0,
            "avg_arg": 0,
            "delta_app": 0,
        },
        "coefficients_r2_30_up": {
            "residual": 0,
            "residual2": 0,
            "residual3": 0,
            "avg_ott": 0,
            "avg_putt": 0,
            "avg_app": 0,
            "avg_arg": 0,
            "delta_app": 0,
        },
        "coefficients_r3": {"sg_ott_avg": 0, "sg_putt_avg": 0, "sg_app_avg": 0, "sg_arg_avg": 0},
        "coefficients_r3_mid": {"sg_ott_avg": 0, "sg_putt_avg": 0, "sg_app_avg": 0, "sg_arg_avg": 0},
        "coefficients_r3_high": {"sg_ott_avg": 0, "sg_putt_avg": 0, "sg_app_avg": 0, "sg_arg_avg": 0},
    }
    simulate_scores(
        params,
        field,
        np.eye(4),
        coefficient_template,
        delta,
        simulations=10,
        seed=7,
        cut_line=2,
        week_latent_sd=0.65,
    )
    expected_total = np.array([1.2, -1.1])
    for total_mean in captured["total_means"]:
        assert np.asarray(total_mean) == pytest.approx(expected_total)
    expected_mu = params["mu"].copy()
    expected_mu[:, 1] += delta
    assert np.asarray(captured["mu"]) == pytest.approx(expected_mu)
    assert np.asarray(captured["std"]) == pytest.approx(params["std_shot"])


def test_market_probabilities_use_full_tournament_scores_and_pushes(monkeypatch) -> None:
    fake_top = np.array([[0.8, 1.0, 1.0], [0.8, 1.0, 1.0]])

    def aggregate(scores):
        return np.zeros((2, 2)), fake_top, fake_top

    monkeypatch.setitem(sys.modules, "sims_kernel", types.SimpleNamespace(aggregate_round=aggregate))
    scores = np.array([[1, 2, 3, 1], [2, 2, 1, 1]])
    field = pd.DataFrame({"dg_id": [10, 20], "player_name": ["a", "b"]})
    lines = pd.DataFrame(
        [
            {
                "event_id": "pga:R2025009",
                "market": "tournament_matchup",
                "selection_id": "10",
                "opponent_id": "20",
            },
            {
                "event_id": "pga:R2025009",
                "market": "top_5",
                "selection_id": "10",
                "opponent_id": "",
            },
        ]
    )
    probabilities, audit = probabilities_for_lines(
        scores, field, lines, "pga:R2025009", pd.Timestamp("2025-03-06")
    )
    matchup = probabilities[probabilities["market"].eq("tournament_matchup")].iloc[0]
    assert matchup["probability"] == pytest.approx(0.25)
    assert matchup["push_probability"] == pytest.approx(0.50)
    assert audit["market_players_missing_from_roster"] == 0


def test_cut_rule_handles_signature_invitational_and_limited_no_cut() -> None:
    signature = EventMeta(
        "pga:R2025009", 2025, 9, "Arnold Palmer Invitational", pd.Timestamp("2025-03-06"), ((1, 72),)
    )
    limited = EventMeta(
        "pga:R2025001", 2025, 1, "The Sentry", pd.Timestamp("2025-01-02"), ((2, 59),)
    )
    assert cut_line_for_event(signature, 72) == (50, "signature_invitational_top50")
    assert cut_line_for_event(limited, 59) == (59, "curated_no_cut")

    heritage = EventMeta(
        "pga:R2026012", 2026, 12, "RBC Heritage", pd.Timestamp("2026-04-16"), ((3, 82),)
    )
    assert cut_line_for_event(heritage, 82) == (82, "curated_no_cut")


def test_cut_rule_handles_major_specific_field_sizes() -> None:
    pga = EventMeta(
        "pga:R2025033", 2025, 33, "PGA Championship", pd.Timestamp("2025-05-15"), ((4, 156),)
    )
    us_open = EventMeta(
        "pga:R2025026", 2025, 26, "U.S. Open", pd.Timestamp("2025-06-12"), ((5, 156),)
    )
    assert cut_line_for_event(pga, 156) == (70, "pga_championship_top70")
    assert cut_line_for_event(us_open, 156) == (60, "us_open_top60")


def test_event_audit_normalizes_fresh_and_resumed_start_values() -> None:
    audit = normalize_event_audit_rows(
        [
            {"event_key": "pga:R2025033", "start": pd.Timestamp("2025-05-15")},
            {"event_key": "pga:R2025026", "start": "2025-06-12T00:00:00"},
        ]
    )
    assert pd.api.types.is_datetime64_any_dtype(audit["start"])
    assert audit["start"].tolist() == [
        pd.Timestamp("2025-05-15"),
        pd.Timestamp("2025-06-12"),
    ]


def test_nonstandard_scoring_formats_are_excluded_before_simulation() -> None:
    tour_championship = EventMeta(
        "pga:R2025060", 2025, 60, "TOUR Championship", pd.Timestamp("2025-08-21"), ((1, 30),)
    )
    stableford = EventMeta(
        "pga:R2024472", 2024, 472, "Barracuda Championship", pd.Timestamp("2024-07-18"), ((2, 156),)
    )
    assert event_format_exclusion(tour_championship) == "staggered_start_unmodeled"
    assert event_format_exclusion(stableford) == "modified_stableford_unmodeled"

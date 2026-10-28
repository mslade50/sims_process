import numpy as np
import pandas as pd
import pytest

from archive.market_backtest_scoring import (
    GateConfig,
    LinePolicy,
    SizingConfig,
    american_decimal_odds,
    american_implied_probability,
    american_profit_multiple,
    compare_arm_bets,
    event_cluster_bootstrap,
    max_drawdown_from_pnl,
    run_two_arm_backtest,
    select_arm_bets,
    select_closing_lines,
    settle_bets,
)


KEYS = ["event_id", "market", "selection_id", "opponent_id"]


def test_american_odds_conversions_use_raw_prices():
    odds = np.array([-200.0, 100.0, 250.0])
    np.testing.assert_allclose(american_profit_multiple(odds), [0.5, 1.0, 2.5])
    np.testing.assert_allclose(american_decimal_odds(odds), [1.5, 2.0, 3.5])
    np.testing.assert_allclose(
        american_implied_probability(odds),
        [2 / 3, 1 / 2, 2 / 7],
    )
    with pytest.raises(ValueError, match="American odds"):
        american_profit_multiple(-50)


def test_matchup_settlement_handles_win_loss_push_and_ties_lose():
    rows = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 1],
            "market": ["72-hole matchup"] * 5,
            "close_odds": [-110, 150, -105, -105, 120],
            "stake": [1, 2, 1, 1, 1],
            "outcome": ["win", "loss", "tie", "tie", None],
            "outcome_numeric": [np.nan, np.nan, np.nan, np.nan, 1.0],
            "tie_rule": ["push", "push", "ties push", "separate bet offered", "push"],
        }
    )
    settled = settle_bets(rows)
    assert settled["settlement_status"].tolist() == [
        "win",
        "loss",
        "push",
        "tie_loss",
        "win",
    ]
    np.testing.assert_allclose(settled["pnl"], [100 / 110, -2, 0, -1, 1.2])
    np.testing.assert_allclose(settled["return_amount"], [1 + 100 / 110, 0, 1, 0, 2.2])


def test_outright_fractional_outcome_uses_dead_heat_stake_reduction():
    rows = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 1],
            "market": ["win", "top5", "top_10", "top20", "win"],
            "close_odds": [300, 100, 200, -125, 500],
            "stake": [1, 1, 2, 1, 1],
            "bet_outcome_numeric": [0.5, 0.5, 1.0, 0.0, np.nan],
            "outcome": [None, None, None, None, "void"],
        }
    )
    settled = settle_bets(rows)
    assert settled["settlement_status"].tolist() == [
        "dead_heat",
        "dead_heat",
        "win",
        "loss",
        "void",
    ]
    # +300 half dead heat returns 0.5 * 4.0 = 2.0, hence +1.0 net.
    np.testing.assert_allclose(settled["pnl"], [1.0, 0.0, 4.0, -1.0, 0.0])


def test_fixed_book_priority_and_best_of_sharp_choose_one_shared_line():
    rows = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 1, 2],
            "market": ["win"] * 5,
            "selection_id": [10, 10, 10, 11, 20],
            "book": ["betcris", "pinnacle", "betcris", "pinnacle", "pinnacle"],
            "close_odds": [120, 130, 125, -110, 200],
            "close_time": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T10:05:00Z",
                "2026-01-01T11:00:00Z",
                "2026-01-01T10:00:00Z",
                "2026-01-08T10:00:00Z",
            ],
        }
    )
    fixed = select_closing_lines(
        rows,
        LinePolicy("fixed_book", ("betcris", "pinnacle")),
    )
    selection_10 = fixed.loc[fixed["selection_id"] == 10].iloc[0]
    assert selection_10["book"] == "betcris"
    assert selection_10["close_odds"] == 125  # latest duplicate at the fixed book
    assert fixed.loc[fixed["selection_id"] == 11, "book"].item() == "pinnacle"

    best = select_closing_lines(
        rows,
        LinePolicy("best_of_sharp", ("betcris", "pinnacle")),
    )
    selection_10 = best.loc[best["selection_id"] == 10].iloc[0]
    assert selection_10["book"] == "pinnacle"
    assert selection_10["close_odds"] == 130


def test_default_identity_never_collapses_reused_event_id_across_seasons():
    rows = pd.DataFrame(
        {
            "season": [2024, 2025],
            "event_id": [7, 7],
            "market": ["win", "win"],
            "selection_id": [10, 10],
            "book": ["betcris", "betcris"],
            "close_odds": [150, 175],
        }
    )
    selected = select_closing_lines(rows, LinePolicy("fixed_book", ("betcris",)))
    assert len(selected) == 2
    assert set(selected["event_key"]) == {"2024:7", "2025:7"}


def test_event_key_is_stable_when_csv_infers_integer_ids_as_floats():
    rows = pd.DataFrame(
        {
            "season": [2024.0],
            "event_id": [7.0],
            "market": ["win"],
            "selection_id": [10],
            "book": ["betcris"],
            "close_odds": [150],
        }
    )
    selected = select_closing_lines(rows, LinePolicy("fixed_book", ("betcris",)))
    assert selected["event_key"].item() == "2024:7"


def _matchup_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "market": ["tournament_matchup"] * 3,
            "selection_id": ["a", "b", "c"],
            "opponent_id": ["x", "y", "z"],
            "book": ["betcris"] * 3,
            "close_odds": [100, 100, 100],
            "tie_rule": ["ties push"] * 3,
            "outcome": ["win", "loss", "win"],
            "close_time": ["2026-01-01"] * 3,
        }
    )


def test_identical_gate_and_capped_quarter_kelly_sizing():
    lines = _matchup_lines()
    probabilities = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "market": ["tournament_matchup"] * 3,
            "selection_id": ["a", "b", "c"],
            "opponent_id": ["x", "y", "z"],
            "probability": [0.60, 0.54, 0.58],
            "push_probability": [0.00, 0.00, 0.10],
        }
    )
    selected = select_arm_bets(
        lines,
        probabilities,
        "baseline",
        GateConfig(min_ev=0.10, min_probability_edge=0.05),
        SizingConfig(
            method="quarter_kelly",
            bankroll_units=100,
            kelly_fraction=0.25,
            max_bankroll_fraction=0.02,
        ),
        KEYS,
    )
    # b fails; a and c clear the same fixed gate.  Both hit the 2%-bankroll cap.
    assert selected["selection_id"].tolist() == ["a", "c"]
    np.testing.assert_allclose(selected["stake"], [2.0, 2.0])
    # c's 0.58 raw win probability is 0.6444 conditional on no push.
    assert selected.loc[selected["selection_id"] == "c", "pricing_probability"].item() == pytest.approx(
        0.58 / 0.90
    )


def test_two_arm_run_attributes_overlap_new_and_dropped():
    lines = _matchup_lines()
    baseline = pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "market": ["tournament_matchup"] * 3,
            "selection_id": ["a", "b", "c"],
            "opponent_id": ["x", "y", "z"],
            "probability": [0.60, 0.60, 0.50],
        }
    )
    challenger = baseline.copy()
    challenger["probability"] = [0.50, 0.65, 0.60]
    result = run_two_arm_backtest(
        lines,
        baseline,
        challenger,
        line_policy=LinePolicy("fixed_book", ("betcris",)),
        gate=GateConfig(min_ev=0.10),
        sizing=SizingConfig(method="flat", flat_stake=1),
        key_cols=KEYS,
        bootstrap_draws=100,
        bootstrap_seed=7,
    )
    labels = dict(zip(result.attribution["selection_id"], result.attribution["attribution"]))
    assert labels == {"a": "dropped", "b": "overlap", "c": "new"}
    summary = result.attribution_summary.set_index("attribution")
    assert summary.loc["new", "challenger_pnl"] == 1
    assert summary.loc["dropped", "baseline_pnl"] == 1
    assert summary.loc["overlap", "pnl_delta"] == 0


def test_event_cluster_bootstrap_is_paired_and_deterministic():
    bets = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 1, 1, 2],
            "arm": ["baseline"] * 3 + ["challenger"] * 3,
            "stake": [1.0] * 6,
            "pnl": [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            "close_time": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-08",
                "2026-01-01",
                "2026-01-01",
                "2026-01-08",
            ],
        }
    )
    first = event_cluster_bootstrap(bets, draws=500, seed=123)
    second = event_cluster_bootstrap(bets, draws=500, seed=123)
    pd.testing.assert_frame_equal(first, second)
    delta = first.loc[
        (first["arm"] == "challenger_minus_baseline") & (first["metric"] == "pnl")
    ].iloc[0]
    assert delta["estimate"] == 2.0
    assert delta["cluster_count"] == 2


def test_max_drawdown_uses_event_ordered_pnl():
    assert max_drawdown_from_pnl([2, -1, -3, 4, -2]) == 4


def test_compare_rejects_duplicate_probability_keys():
    lines = _matchup_lines().iloc[[0]]
    probabilities = pd.DataFrame(
        {
            "event_id": [1, 1],
            "market": ["tournament_matchup"] * 2,
            "selection_id": ["a", "a"],
            "opponent_id": ["x", "x"],
            "probability": [0.6, 0.7],
        }
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        select_arm_bets(lines, probabilities, "baseline", key_cols=KEYS)


def test_compare_arm_bets_counts_opposite_sides_as_new_and_dropped():
    baseline = pd.DataFrame(
        {
            "event_id": [1],
            "market": ["tournament_matchup"],
            "selection_id": ["a"],
            "opponent_id": ["b"],
            "probability": [0.6],
            "ev_per_unit": [0.2],
            "stake": [1.0],
            "pnl": [1.0],
        }
    )
    challenger = baseline.copy()
    challenger[["selection_id", "opponent_id"]] = challenger[["opponent_id", "selection_id"]]
    comparison = compare_arm_bets(baseline, challenger, KEYS)
    assert set(comparison["attribution"]) == {"new", "dropped"}


def test_two_arm_run_handles_no_bets_clearing_the_gate():
    lines = pd.DataFrame(
        {
            "season": [2025],
            "event_id": [1],
            "market": ["win"],
            "selection_id": [10],
            "book": ["betcris"],
            "close_odds": [100],
            "bet_outcome_numeric": [1.0],
        }
    )
    probabilities = pd.DataFrame(
        {
            "season": [2025],
            "event_id": [1],
            "market": ["win"],
            "selection_id": [10],
            "probability": [0.40],
        }
    )
    result = run_two_arm_backtest(
        lines,
        probabilities,
        probabilities,
        line_policy=LinePolicy(books=("betcris",)),
        gate=GateConfig(min_ev=0.10),
        bootstrap_draws=10,
    )
    assert result.bets.empty
    assert result.performance.empty
    assert result.bootstrap.empty


def test_zero_kelly_fraction_is_not_sent_to_settlement_as_zero_stake():
    lines = pd.DataFrame(
        {
            "event_id": [1],
            "market": ["win"],
            "selection_id": [10],
            "book": ["betcris"],
            "close_odds": [100],
            "bet_outcome_numeric": [1.0],
        }
    )
    probabilities = pd.DataFrame(
        {
            "event_id": [1],
            "market": ["win"],
            "selection_id": [10],
            "probability": [0.50],
        }
    )
    selected = select_arm_bets(
        lines,
        probabilities,
        "baseline",
        gate=GateConfig(),
        sizing=SizingConfig(method="quarter_kelly"),
    )
    assert selected.empty

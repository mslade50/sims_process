import pandas as pd

import grade_bets
from dashboard.data_layer import _drop_excluded_results


def test_dashboard_drops_audit_preserved_exclusions():
    bets = pd.DataFrame(
        {
            "result": ["win", "excluded_residual_bug", "EXCLUDED_OTHER_BUG", ""],
            "bet_on": ["valid", "bad residual", "other invalid", "ungraded"],
        }
    )

    filtered = _drop_excluded_results(bets)

    assert filtered["bet_on"].tolist() == ["valid", "ungraded"]


def test_regrade_does_not_restore_excluded_results(monkeypatch):
    sheet_rows = pd.DataFrame(
        {
            "result": ["win", "duplicate", "excluded_residual_bug", ""],
            "bet_on": ["valid", "duplicate", "invalid", "ungraded"],
        }
    )
    monkeypatch.setattr(grade_bets, "read_sheet_as_df", lambda *_: sheet_rows)

    selected = grade_bets.get_ungraded_bets(object(), "Round Matchups", regrade=True)

    assert selected["bet_on"].tolist() == ["valid", "ungraded"]


def test_performance_metrics_exclude_invalidated_bets():
    bets = [
        {
            "result": "win",
            "units_won": 1.0,
            "units_wagered": 1.0,
            "bet_type": "round_matchup",
        },
        {
            "result": "excluded_residual_bug",
            "units_won": 0.0,
            "units_wagered": 1.0,
            "bet_type": "round_matchup",
        },
    ]

    metrics = grade_bets.calculate_performance_metrics(bets, "3M Open", "525", 2026)

    assert metrics["total_bets"] == 1
    assert metrics["round_mu_bets"] == 1
    assert metrics["units_wagered"] == 1.0
    assert metrics["units_won"] == 1.0

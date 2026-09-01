import pandas as pd

import grade_bets
import sheets_storage
from dashboard import data_layer
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


def test_performance_loader_keeps_legacy_betcris_and_drops_only_explicit_exclusions(
    monkeypatch,
):
    rows = pd.DataFrame(
        [
            {
                "run_timestamp": "2026-08-27 20:00:00",
                "event_name": "tourchamp",
                "year": "2026",
                "event_id": "60",
                "round": "2",
                "player_1": "alpha",
                "player_2": "beta",
                "bet_on": "alpha",
                "bookmaker": "betcris",
                "p1_odds": "120",
                "p2_odds": "-150",
                "result": "excluded_bad_contract",
                "units_won": "",
            },
            {
                "run_timestamp": "2026-08-27 21:00:00",
                "event_name": "tourchamp",
                "year": "2026",
                "event_id": "60",
                "round": "2",
                "player_1": "alpha",
                "player_2": "beta",
                "bet_on": "alpha",
                "bookmaker": "betcris",
                "p1_odds": "124",
                "p2_odds": "-156",
                "result": "win",
                "units_won": "1.24",
            },
        ]
    )

    monkeypatch.setattr(sheets_storage, "get_spreadsheet", lambda: object())
    monkeypatch.setattr(
        data_layer,
        "_read_sheets_tab",
        lambda _spreadsheet, tab, _headers: (
            rows.copy() if tab == data_layer._TAB_ROUND_MU else pd.DataFrame()
        ),
    )
    monkeypatch.setitem(data_layer._SHEETS_CACHE, "data", None)
    monkeypatch.setitem(data_layer._SHEETS_CACHE, "timestamp", 0)

    loaded = data_layer._load_all_bets_from_sheets()

    assert loaded["bet_on"].tolist() == ["alpha"]
    assert loaded["result"].tolist() == ["win"]
    assert loaded["bookmaker"].tolist() == ["betcris"]


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

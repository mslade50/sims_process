import pandas as pd

import clv
import clv_alert
import sheets_storage


class _Worksheet:
    def __init__(self, records=None, values=None):
        self._records = records or []
        self._values = values or []

    def get_all_records(self):
        return self._records

    def get_all_values(self):
        return self._values


class _Spreadsheet:
    def __init__(self, tabs):
        self._tabs = tabs

    def worksheet(self, name):
        return self._tabs[name]


def test_morning_clv_excludes_invalidated_row_before_keep_first_dedup(monkeypatch):
    rows = [
        {
            "event_id": 28,
            "round": 4,
            "player_1": "Bhatia, Akshay",
            "player_2": "Thomas, Justin",
            "bet_on": "Bhatia, Akshay",
            "bookmaker": "BetCris",
            "p1_odds": 145,
            "p2_odds": -183,
            "result": "excluded_invalid_model",
        },
        {
            "event_id": 28,
            "round": 4,
            "player_1": "Bhatia, Akshay",
            "player_2": "Thomas, Justin",
            "bet_on": "Bhatia, Akshay",
            "bookmaker": "BetCris",
            "p1_odds": 140,
            "p2_odds": -175,
            "result": "",
        },
    ]
    spreadsheet = _Spreadsheet({"Round Matchups": _Worksheet(records=rows)})
    monkeypatch.setattr(sheets_storage, "get_spreadsheet", lambda: spreadsheet)

    bets = clv_alert.load_round_bets(4, 28)

    assert len(bets) == 1
    assert bets.iloc[0]["p1_odds"] == 140


def test_morning_clv_accepts_legacy_sheet_without_result(monkeypatch):
    rows = [{
        "event_id": 28,
        "round": 4,
        "player_1": "valid, player",
        "player_2": "other, player",
        "bet_on": "valid, player",
        "bookmaker": "Pinnacle",
        "p1_odds": 110,
        "p2_odds": -130,
    }]
    spreadsheet = _Spreadsheet({"Round Matchups": _Worksheet(records=rows)})
    monkeypatch.setattr(sheets_storage, "get_spreadsheet", lambda: spreadsheet)

    bets = clv_alert.load_round_bets(4, 28)

    assert len(bets) == 1


def test_event_clv_sheet_loader_excludes_matchup_and_finish_rows(monkeypatch):
    round_headers = [
        "event_id", "year", "round", "player_1", "player_2", "bookmaker",
        "p1_odds", "p2_odds", "dg_id_p1", "dg_id_p2", "bet_on", "result",
    ]
    finish_headers = [
        "event_id", "year", "market_type", "sportsbook", "american_odds",
        "dg_id", "player_name", "result",
    ]
    tournament_headers = [
        "event_id", "year", "player_1", "player_2", "bookmaker", "p1_odds",
        "p2_odds", "dg_id_p1", "dg_id_p2", "bet_on",
    ]
    spreadsheet = _Spreadsheet(
        {
            "Round Matchups": _Worksheet(values=[round_headers, [
                "28", "2026", "4", "rai, aaron", "straka, sepp", "betonline",
                "-120", "100", "18554", "17511", "rai, aaron",
                "EXCLUDED_INVALID_MODEL",
            ]]),
            # No result column: verifies compatibility with a legacy tab.
            "Tournament Matchups": _Worksheet(values=[tournament_headers, [
                "28", "2026", "valid, player", "other, player", "betcris",
                "110", "-130", "1", "2", "valid, player",
            ]]),
            "Finish Positions": _Worksheet(values=[
                finish_headers,
                ["28", "2026", "top_10", "betonline", "500", "3",
                 "excluded, player", "excluded_invalid_model"],
                ["28", "2026", "top_20", "betonline", "250", "4",
                 "valid, finisher", ""],
            ]),
        }
    )
    monkeypatch.setattr(sheets_storage, "get_spreadsheet", lambda: spreadsheet)
    monkeypatch.setattr(clv.time, "sleep", lambda *_: None)

    matchups, finishes = clv.load_bets_from_sheet(28)

    assert matchups["bet_on_name"].tolist() == ["valid, player"]
    assert finishes["player_name"].tolist() == ["valid, finisher"]


def _ledger_rows(include_result=True):
    rows = [
        {
            "event_id": "28", "year": 2026, "bet_type": "round_matchup",
            "round": 4, "bet_on": "rai, aaron", "opponent": "straka, sepp",
            "bookmaker": "betonline", "book_odds": -120,
            "dg_id_bet_on": 18554, "dg_id_opponent": 17511,
            "result": "excluded_invalid_model",
        },
        {
            "event_id": "28", "year": 2026, "bet_type": "round_matchup",
            "round": 4, "bet_on": "valid, player", "opponent": "other, player",
            "bookmaker": "betcris", "book_odds": 110,
            "dg_id_bet_on": 1, "dg_id_opponent": 2, "result": "",
        },
        {
            "event_id": "28", "year": 2026, "bet_type": "finish_position",
            "round": 0, "bet_on": "excluded, finisher", "opponent": "top_10",
            "bookmaker": "betonline", "book_odds": 500,
            "dg_id_bet_on": 3, "dg_id_opponent": "",
            "result": "excluded_invalid_model",
        },
        {
            "event_id": "28", "year": 2026, "bet_type": "finish_position",
            "round": 0, "bet_on": "valid, finisher", "opponent": "top_20",
            "bookmaker": "betonline", "book_odds": 250,
            "dg_id_bet_on": 4, "dg_id_opponent": "", "result": "",
        },
    ]
    frame = pd.DataFrame(rows)
    return frame if include_result else frame.drop(columns="result")


def test_event_clv_ledger_loader_excludes_invalidated_rows(monkeypatch):
    monkeypatch.setattr(clv.pd, "read_parquet", lambda *_: _ledger_rows())

    matchups, finishes = clv.load_bets_from_ledger(28)

    assert matchups["bet_on_name"].tolist() == ["valid, player"]
    assert finishes["player_name"].tolist() == ["valid, finisher"]


def test_event_clv_ledger_loader_accepts_legacy_schema_without_result(monkeypatch):
    monkeypatch.setattr(
        clv.pd, "read_parquet", lambda *_: _ledger_rows(include_result=False)
    )

    matchups, finishes = clv.load_bets_from_ledger(28)

    assert len(matchups) == 2
    assert len(finishes) == 2

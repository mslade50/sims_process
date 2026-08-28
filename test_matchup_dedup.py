import pandas as pd

import reprice_core
import sheets_storage
from sheets_storage import _append_rows_deduped, _canonical_matchup_row_key


class _Worksheet:
    def __init__(self, records=None, values=None):
        self._records = records or []
        self._values = values or [[]]
        self.appended = []

    def get_all_records(self):
        return self._records

    def get_all_values(self):
        return self._values

    def append_rows(self, rows, value_input_option=None):
        self.appended.extend(rows)


class _HeaderWorksheet:
    def __init__(self, headers, col_count=None):
        self.headers = list(headers)
        self.col_count = col_count if col_count is not None else len(headers)
        self.updated = []
        self.resize_calls = []

    def row_values(self, row):
        assert row == 1
        return list(self.headers)

    def resize(self, rows=None, cols=None):
        self.resize_calls.append((rows, cols))
        if cols is not None:
            self.col_count = cols

    def update_cells(self, cells, value_input_option=None):
        self.updated.extend(cells)
        for cell in cells:
            while len(self.headers) < cell.col:
                self.headers.append("")
            self.headers[cell.col - 1] = cell.value


class _Spreadsheet:
    def __init__(self, worksheet):
        self._worksheet = worksheet

    def worksheet(self, _name):
        return self._worksheet


def test_cache_free_reprice_dedup_is_order_insensitive(monkeypatch):
    existing = [{
        "event_id": "28",
        "round": "4",
        "player_1": "beta player",
        "player_2": "alpha player",
        "bookmaker": "pinnacle",
        "p1_odds": -120,
        "p2_odds": 100,
        "bet_on": "alpha player",
    }]
    ws = _Worksheet(records=existing)
    monkeypatch.setattr(
        "sheets_storage._get_or_create_tab", lambda *args, **kwargs: ws
    )
    combined = pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Bookmaker": "Pinnacle",
        "P1 Odds": 100.0,
        "P2 Odds": -120.0,
        "bet_on": "alpha player",
    }])

    fresh, seen_alerts = reprice_core.dedup_round_matchups(
        combined, object(), "28", 4
    )

    assert fresh.empty
    assert reprice_core.alerted_key(
        "alpha player", "beta player", "alpha player"
    ) in seen_alerts


def test_cache_free_reprice_stores_and_alerts_an_edge_flip(monkeypatch):
    existing = [{
        "event_id": "28",
        "round": "4",
        "player_1": "alpha player",
        "player_2": "beta player",
        "bookmaker": "pinnacle",
        "p1_odds": 100,
        "p2_odds": -120,
        "bet_on": "alpha player",
        "result": "",
    }]
    ws = _Worksheet(records=existing)
    monkeypatch.setattr(
        "sheets_storage._get_or_create_tab", lambda *args, **kwargs: ws
    )
    flipped = pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Bookmaker": "pinnacle",
        "P1 Odds": 100,
        "P2 Odds": -120,
        "bet_on": "beta player",
    }])

    fresh, seen_alerts = reprice_core.dedup_round_matchups(
        flipped, object(), "28", 4
    )
    alert_rows, _ = reprice_core.partition_matchup_alert_rows(fresh, seen_alerts)

    assert fresh["bet_on"].tolist() == ["beta player"]
    assert alert_rows["bet_on"].tolist() == ["beta player"]


def test_invalidated_sharp_row_suppresses_neither_storage_nor_alert(monkeypatch):
    existing = [{
        "event_id": "28",
        "round": "4",
        "player_1": "alpha player",
        "player_2": "beta player",
        "bookmaker": "pinnacle",
        "p1_odds": 100,
        "p2_odds": -120,
        "bet_on": "alpha player",
        "result": "excluded_invalid_model",
    }]
    ws = _Worksheet(records=existing)
    monkeypatch.setattr(
        "sheets_storage._get_or_create_tab", lambda *args, **kwargs: ws
    )
    corrected = pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Bookmaker": "pinnacle",
        "P1 Odds": 100,
        "P2 Odds": -120,
        "bet_on": "alpha player",
    }])

    fresh, seen_alerts = reprice_core.dedup_round_matchups(
        corrected, object(), "28", 4
    )

    assert len(fresh) == 1
    assert seen_alerts == set()


def _round_row(
    p1, p2, o1, o2, bet_on=None, result="", *,
    book="pinnacle", p1_line="", p2_line="", line_verified="",
):
    row = ["" for _ in range(len(sheets_storage.ROUND_MU_HEADERS))]
    row[3:7] = ["28", "4", p1, p2]
    row[9] = book
    row[11:13] = [o1, o2]
    row[sheets_storage.ROUND_MU_HEADERS.index("bet_on")] = bet_on or p1
    row[sheets_storage.ROUND_MU_HEADERS.index("result")] = result
    row[sheets_storage.ROUND_MU_HEADERS.index("p1_line")] = p1_line
    row[sheets_storage.ROUND_MU_HEADERS.index("p2_line")] = p2_line
    row[sheets_storage.ROUND_MU_HEADERS.index("line_verified")] = line_verified
    return row


def _round_key(row):
    return _canonical_matchup_row_key(
        row,
        event_index=3,
        round_index=4,
        player_indices=(5, 6),
        book_index=9,
        odds_indices=(11, 12),
        bet_on_index=17,
        line_indices=(
            sheets_storage.ROUND_MU_HEADERS.index("p1_line"),
            sheets_storage.ROUND_MU_HEADERS.index("p2_line"),
        ),
        line_verified_index=sheets_storage.ROUND_MU_HEADERS.index("line_verified"),
    )


def test_sheet_matchup_dedup_keeps_odds_attached_when_players_flip():
    existing = _round_row(
        "beta player", "alpha player", -120, 100, bet_on="alpha player"
    )
    incoming = _round_row(
        "alpha player", "beta player", 100, -120, bet_on="alpha player"
    )
    ws = _Worksheet(values=[[f"h{i}" for i in range(29)], existing])

    written, skipped = _append_rows_deduped(
        ws, [incoming], [3, 4, 5, 6, 9, 11, 12], key_fn=_round_key
    )

    assert (written, skipped) == (0, 1)
    assert ws.appended == []


def test_sheet_matchup_dedup_still_records_a_real_price_move():
    existing = _round_row(
        "beta player", "alpha player", -120, 100, bet_on="alpha player"
    )
    incoming = _round_row(
        "alpha player", "beta player", 105, -125, bet_on="alpha player"
    )
    ws = _Worksheet(values=[[f"h{i}" for i in range(29)], existing])

    written, skipped = _append_rows_deduped(
        ws, [incoming], [3, 4, 5, 6, 9, 11, 12], key_fn=_round_key
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_sheet_matchup_dedup_records_an_edge_flip_at_unchanged_prices():
    existing = _round_row(
        "alpha player", "beta player", 100, -120, bet_on="alpha player"
    )
    incoming = _round_row(
        "alpha player", "beta player", 100, -120, bet_on="beta player"
    )
    ws = _Worksheet(
        values=[sheets_storage.ROUND_MU_HEADERS, existing]
    )

    written, skipped = _append_rows_deduped(
        ws,
        [incoming],
        sheets_storage._DEDUP_KEYS["round_mu"],
        key_fn=_round_key,
        result_index=sheets_storage.ROUND_MU_HEADERS.index("result"),
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_sheet_matchup_dedup_keeps_straight_and_half_contracts_distinct():
    existing = _round_row(
        "alpha player", "beta player", 100, -120,
        p1_line="", p2_line="", line_verified=True,
    )
    incoming = _round_row(
        "alpha player", "beta player", 100, -120,
        p1_line=-0.5, p2_line=0.5, line_verified=True,
    )
    ws = _Worksheet(values=[sheets_storage.ROUND_MU_HEADERS, existing])

    written, skipped = _append_rows_deduped(
        ws, [incoming], sheets_storage._DEDUP_KEYS["round_mu"],
        key_fn=_round_key,
        result_index=sheets_storage.ROUND_MU_HEADERS.index("result"),
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_legacy_betcris_sheet_row_suppresses_neither_corrected_storage_nor_alert(
    monkeypatch,
):
    existing = [{
        "event_id": "28", "round": "4", "player_1": "alpha player",
        "player_2": "beta player", "bookmaker": "betcris",
        "p1_odds": 100, "p2_odds": -120, "bet_on": "alpha player",
        "line_verified": "",
    }]
    ws = _Worksheet(records=existing)
    monkeypatch.setattr(
        "sheets_storage._get_or_create_tab", lambda *args, **kwargs: ws
    )
    corrected = pd.DataFrame([{
        "Player 1": "alpha player", "Player 2": "beta player",
        "Bookmaker": "betcris", "P1 Odds": 100, "P2 Odds": -120,
        "P1 Line": None, "P2 Line": None, "line_verified": True,
        "bet_on": "alpha player",
    }])

    fresh, seen = reprice_core.dedup_round_matchups(
        corrected, object(), "28", 4
    )

    assert len(fresh) == 1
    assert seen == set()


def test_sheet_matchup_dedup_ignores_invalidated_existing_row():
    existing = _round_row(
        "alpha player", "beta player", 100, -120,
        bet_on="alpha player", result="INVALID_MANUAL_AUDIT",
    )
    incoming = _round_row(
        "alpha player", "beta player", 100, -120, bet_on="alpha player"
    )
    ws = _Worksheet(values=[sheets_storage.ROUND_MU_HEADERS, existing])

    written, skipped = _append_rows_deduped(
        ws,
        [incoming],
        sheets_storage._DEDUP_KEYS["round_mu"],
        key_fn=_round_key,
        result_index=sheets_storage.ROUND_MU_HEADERS.index("result"),
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_ledger_keeps_invalid_audit_row_but_allows_corrected_bet(tmp_path, monkeypatch):
    ledger_path = tmp_path / "bet_ledger.parquet"
    base = {
        "event_id": "28",
        "bet_type": "round_matchup",
        "round": 4,
        "bet_on": "alpha player",
        "opponent": "beta player",
        "bookmaker": "pinnacle",
    }
    pd.DataFrame([{**base, "bet_id": "bad", "result": "excluded_invalid_model"}]).to_parquet(
        ledger_path, index=False
    )
    monkeypatch.setattr(sheets_storage, "LEDGER_PATH", str(ledger_path))

    sheets_storage._append_to_ledger([
        {**base, "bet_id": "corrected", "result": ""},
        {**base, "bet_id": "same-batch-duplicate", "result": ""},
    ])

    stored = pd.read_parquet(ledger_path)
    assert stored["bet_id"].tolist() == ["bad", "corrected"]

    sheets_storage.update_ledger_grades([
        {**base, "result": "win", "units_wagered": 1, "units_won": 1}
    ])
    sheets_storage.update_ledger_clv([
        {
            **base,
            "open_odds": 100,
            "close_odds": -110,
            "tot_clv": 2.4,
            "clv": 2.4,
            "clv_book": "pinnacle",
        }
    ])

    stored = pd.read_parquet(ledger_path).set_index("bet_id")
    assert stored.loc["bad", "result"] == "excluded_invalid_model"
    assert stored.loc["corrected", "result"] == "win"
    assert pd.isna(stored.loc["bad", "close_odds"])
    assert stored.loc["corrected", "close_odds"] == -110


def test_legacy_betcris_ledger_row_cannot_tombstone_corrected_straight(
    tmp_path, monkeypatch
):
    ledger_path = tmp_path / "bet_ledger.parquet"
    base = {
        "event_id": "28", "bet_type": "round_matchup", "round": 2,
        "bet_on": "alpha player", "opponent": "beta player",
        "bookmaker": "betcris", "spread_line": 0.0, "result": "",
    }
    pd.DataFrame([{
        **base, "bet_id": "legacy-ambiguous", "line_verified": False,
    }]).to_parquet(ledger_path, index=False)
    monkeypatch.setattr(sheets_storage, "LEDGER_PATH", str(ledger_path))

    sheets_storage._append_to_ledger([
        {**base, "bet_id": "corrected", "line_verified": True},
        {**base, "bet_id": "corrected-duplicate", "line_verified": True},
    ])

    stored = pd.read_parquet(ledger_path)
    assert stored["bet_id"].tolist() == ["legacy-ambiguous", "corrected"]

    sheets_storage.update_ledger_grades([{
        **base, "result": "win", "units_wagered": 1, "units_won": 1,
    }])
    sheets_storage.update_ledger_clv([{
        **base, "open_odds": 100, "close_odds": -110,
        "tot_clv": 2.4, "clv": 2.4, "clv_book": "pinnacle",
    }])

    stored = pd.read_parquet(ledger_path).set_index("bet_id")
    assert stored.loc["legacy-ambiguous", "result"] == ""
    assert stored.loc["corrected", "result"] == "win"
    assert pd.isna(stored.loc["legacy-ambiguous", "close_odds"])
    assert stored.loc["corrected", "close_odds"] == -110


def _score_row(side, result=""):
    headers = sheets_storage.SCORE_EDGES_HEADERS
    values = {
        "event_id": "28",
        "round": "4",
        "player": "alpha player",
        "line": 68.5,
        "book": "fanduel",
        "best_side": side,
        "mkt_under": -110,
        "mkt_over": -110,
        "result": result,
    }
    row = ["" for _ in headers]
    for name, value in values.items():
        row[headers.index(name)] = value
    return row


def test_score_schema_has_grading_columns_at_the_storage_result_index():
    assert sheets_storage.SCORE_EDGES_HEADERS.index("result") == 16
    assert sheets_storage.SCORE_EDGES_HEADERS[16:] == [
        "result", "actual_score", "units_won"
    ]


def test_score_legacy_header_is_safely_extended_with_grading_columns():
    legacy = sheets_storage.SCORE_EDGES_HEADERS[:16]
    ws = _HeaderWorksheet(legacy, col_count=16)

    returned = sheets_storage._get_or_create_tab(
        _Spreadsheet(ws), "Score Edges", sheets_storage.SCORE_EDGES_HEADERS
    )

    assert returned is ws
    assert ws.headers == sheets_storage.SCORE_EDGES_HEADERS
    assert [(cell.col, cell.value) for cell in ws.updated] == [
        (17, "result"), (18, "actual_score"), (19, "units_won")
    ]
    assert ws.resize_calls == [(None, 19)]


def test_non_prefix_header_drift_fails_closed_before_update():
    drifted = list(sheets_storage.SCORE_EDGES_HEADERS)
    drifted[7], drifted[8] = drifted[8], drifted[7]
    ws = _HeaderWorksheet(drifted)

    try:
        sheets_storage._get_or_create_tab(
            _Spreadsheet(ws), "Score Edges", sheets_storage.SCORE_EDGES_HEADERS
        )
    except RuntimeError as exc:
        assert "not a safe trailing migration" in str(exc)
    else:
        raise AssertionError("non-prefix header drift should fail closed")

    assert ws.updated == []


def test_expected_header_prefix_allows_unrelated_trailing_columns():
    headers = sheets_storage.SCORE_EDGES_HEADERS + ["audit_note"]
    ws = _HeaderWorksheet(headers)

    returned = sheets_storage._get_or_create_tab(
        _Spreadsheet(ws), "Score Edges", sheets_storage.SCORE_EDGES_HEADERS
    )

    assert returned is ws
    assert ws.headers == headers
    assert ws.updated == []


def test_sheet_score_dedup_records_side_flip_at_unchanged_line_and_odds():
    existing = _score_row("Under")
    incoming = _score_row("Over")
    ws = _Worksheet(values=[sheets_storage.SCORE_EDGES_HEADERS, existing])

    written, skipped = _append_rows_deduped(
        ws,
        [incoming],
        sheets_storage._DEDUP_KEYS["score_edges"],
        result_index=sheets_storage.SCORE_EDGES_HEADERS.index("result"),
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_sheet_score_dedup_ignores_excluded_existing_row():
    existing = _score_row("Under", result="excluded_invalid_model")
    incoming = _score_row("Under")
    ws = _Worksheet(values=[sheets_storage.SCORE_EDGES_HEADERS, existing])

    written, skipped = _append_rows_deduped(
        ws,
        [incoming],
        sheets_storage._DEDUP_KEYS["score_edges"],
        result_index=sheets_storage.SCORE_EDGES_HEADERS.index("result"),
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]


def test_store_score_edges_writes_the_full_grading_schema(monkeypatch):
    ws = _Worksheet(values=[sheets_storage.SCORE_EDGES_HEADERS])
    monkeypatch.setattr(sheets_storage, "_get_or_create_tab", lambda *_args: ws)
    monkeypatch.setattr(sheets_storage, "_append_to_ledger", lambda *_args: None)
    edge = pd.DataFrame([{
        "Player": "alpha player",
        "Line": 68.5,
        "Book": "fanduel",
        "Best_Side": "Under",
        "Mkt_Under": -110,
        "Mkt_Over": -110,
        "Fair_Under": -125,
        "Fair_Over": 105,
        "Edge_Under": 7.0,
        "Edge_Over": -7.0,
        "Best_Edge": 7.0,
    }])

    sheets_storage.store_score_edges(
        edge, 4, "test_event", "28", spreadsheet=object()
    )

    assert len(ws.appended) == 1
    assert len(ws.appended[0]) == len(sheets_storage.SCORE_EDGES_HEADERS)
    assert ws.appended[0][sheets_storage.SCORE_EDGES_HEADERS.index("result")] == ""

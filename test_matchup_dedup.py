import pandas as pd

import reprice_core
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


def _round_row(p1, p2, o1, o2):
    row = ["" for _ in range(29)]
    row[3:7] = ["28", "4", p1, p2]
    row[9] = "pinnacle"
    row[11:13] = [o1, o2]
    return row


def _round_key(row):
    return _canonical_matchup_row_key(
        row,
        event_index=3,
        round_index=4,
        player_indices=(5, 6),
        book_index=9,
        odds_indices=(11, 12),
    )


def test_sheet_matchup_dedup_keeps_odds_attached_when_players_flip():
    existing = _round_row("beta player", "alpha player", -120, 100)
    incoming = _round_row("alpha player", "beta player", 100, -120)
    ws = _Worksheet(values=[[f"h{i}" for i in range(29)], existing])

    written, skipped = _append_rows_deduped(
        ws, [incoming], [3, 4, 5, 6, 9, 11, 12], key_fn=_round_key
    )

    assert (written, skipped) == (0, 1)
    assert ws.appended == []


def test_sheet_matchup_dedup_still_records_a_real_price_move():
    existing = _round_row("beta player", "alpha player", -120, 100)
    incoming = _round_row("alpha player", "beta player", 105, -125)
    ws = _Worksheet(values=[[f"h{i}" for i in range(29)], existing])

    written, skipped = _append_rows_deduped(
        ws, [incoming], [3, 4, 5, 6, 9, 11, 12], key_fn=_round_key
    )

    assert (written, skipped) == (1, 0)
    assert ws.appended == [incoming]

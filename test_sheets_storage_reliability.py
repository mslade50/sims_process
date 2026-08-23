"""Regression tests for fail-closed bet-ledger storage."""

from pathlib import Path

import pandas as pd
import pytest

import sheets_storage


def _tournament_matchup():
    return pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Bookmaker": "pinnacle",
        "bet_on": "alpha player",
    }])


def _finish_position():
    return pd.DataFrame([{
        "player_name": "alpha player",
        "market_type": "win",
        "bookmaker": "pinnacle",
        "decimal_odds": 3.0,
        "simulated_win_prob": 0.5,
        "stake": 2.0,
    }])


def _round_matchup():
    return pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Bookmaker": "pinnacle",
        "bet_on": "alpha player",
    }])


def _round_3ball():
    return pd.DataFrame([{
        "Player 1": "alpha player",
        "Player 2": "beta player",
        "Player 3": "gamma player",
        "Bookmaker": "pinnacle",
        "bet_on": "alpha player",
    }])


def _score_edge():
    return pd.DataFrame([{
        "Player": "alpha player",
        "Line": 70.5,
        "Book": "pinnacle",
        "Best_Side": "Under",
        "Best_Edge": 10.0,
    }])


@pytest.mark.parametrize(
    ("store_name", "frame_factory", "args", "store_kwargs"),
    [
        ("store_tournament_matchups", _tournament_matchup, ("event", "28"), {}),
        ("store_finish_positions", _finish_position, ("event", "28"), {}),
        (
            "store_finish_positions",
            _finish_position,
            ("event", "28"),
            {"tab_name": sheets_storage.TAB_LIVE},
        ),
        ("store_round_matchups", _round_matchup, (2, "event", "28"), {}),
        ("store_round_3balls", _round_3ball, (2, "event", "28"), {}),
        ("store_score_edges", _score_edge, (2, "event", "28"), {}),
    ],
)
def test_store_propagates_ledger_append_failure(
    monkeypatch, store_name, frame_factory, args, store_kwargs
):
    monkeypatch.setattr(sheets_storage, "_get_or_create_tab", lambda *a, **k: object())
    monkeypatch.setattr(sheets_storage, "_append_rows_deduped", lambda *a, **k: (1, 0))

    def fail_ledger_append(_records):
        raise OSError("ledger disk unavailable")

    monkeypatch.setattr(sheets_storage, "_append_to_ledger", fail_ledger_append)

    with pytest.raises(OSError, match="ledger disk unavailable"):
        getattr(sheets_storage, store_name)(
            frame_factory(), *args, spreadsheet=object(), **store_kwargs
        )


def test_unreadable_existing_ledger_is_never_replaced(monkeypatch, tmp_path):
    ledger_path = Path(tmp_path, "bet_ledger.parquet")
    original_bytes = b"existing ledger bytes that must survive"
    ledger_path.write_bytes(original_bytes)
    monkeypatch.setattr(sheets_storage, "LEDGER_PATH", str(ledger_path))

    def fail_read(path):
        assert path == str(ledger_path)
        raise OSError("ledger cannot be decoded")

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("an unreadable existing ledger must not trigger a replacement write")

    monkeypatch.setattr(sheets_storage.pd, "read_parquet", fail_read)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", unexpected_write)

    record = sheets_storage._empty_ledger_record()
    record.update({
        "event_id": "28",
        "bet_type": "round_matchup",
        "round": 2,
        "bet_on": "alpha player",
        "opponent": "beta player",
        "bookmaker": "pinnacle",
    })

    with pytest.raises(RuntimeError, match="refusing to overwrite it") as exc_info:
        sheets_storage._append_to_ledger([record])

    assert isinstance(exc_info.value.__cause__, OSError)
    assert ledger_path.read_bytes() == original_bytes

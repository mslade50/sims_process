import json

import pandas as pd
import pytest

import clv_alert


def _write_snapshot(tmp_path, *, event_id=28, round_key="4", row_round=4):
    snapshot = {
        "event_id": str(event_id),
        "rounds": {
            str(round_key): {
                "rmatch": {
                    "markets": {
                        "round_matchup": [
                            {
                                "player_a": "Rai, Aaron",
                                "player_b": "Straka, Sepp",
                                "round": row_round,
                                "books": {
                                    "pinnacle": {"a": -118, "b": 100},
                                    "betonline": {"a": -120, "b": 102},
                                },
                            }
                        ]
                    }
                }
            }
        },
    }
    path = tmp_path / "closing.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_fetch_board_closing_reads_valid_local_snapshot(tmp_path):
    path = _write_snapshot(tmp_path)

    result = clv_alert.fetch_board_closing(4, 28, snapshot_path=str(path))

    assert len(result) == 2
    assert set(result["Bookmaker"]) == {"pinnacle", "betonline"}
    assert result.iloc[0]["Player 1"] == "rai, aaron"
    assert result.iloc[0]["Player 2"] == "straka, sepp"


@pytest.mark.parametrize(
    ("event_id", "round_key", "row_round"),
    [(29, "4", 4), (28, "3", 3), (28, "4", 3)],
)
def test_fetch_board_closing_rejects_wrong_event_or_round(
    tmp_path, event_id, round_key, row_round
):
    path = _write_snapshot(
        tmp_path, event_id=event_id, round_key=round_key, row_round=row_round
    )

    result = clv_alert.fetch_board_closing(4, 28, snapshot_path=str(path))

    assert result.empty


def test_required_frozen_close_never_falls_back_to_live(monkeypatch):
    live_called = False

    def _live(_sim_round):
        nonlocal live_called
        live_called = True
        return pd.DataFrame([{"Bookmaker": "pinnacle"}])

    monkeypatch.setattr(clv_alert, "fetch_board_closing", lambda *_: pd.DataFrame())
    monkeypatch.setattr(clv_alert, "fetch_closing_odds", _live)

    with pytest.raises(clv_alert.FrozenClosingUnavailable):
        clv_alert.resolve_closing_odds(4, 28, require_frozen=True)

    assert not live_called


def test_interactive_close_keeps_live_fallback(monkeypatch):
    expected = pd.DataFrame([{"Bookmaker": "pinnacle"}])
    monkeypatch.setattr(clv_alert, "fetch_board_closing", lambda *_: pd.DataFrame())
    monkeypatch.setattr(clv_alert, "fetch_closing_odds", lambda *_: expected)

    result = clv_alert.resolve_closing_odds(4, 28, require_frozen=False)

    assert result.equals(expected)

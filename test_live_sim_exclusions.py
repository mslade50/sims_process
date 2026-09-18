import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from live_sim_exclusions import filter_live_sim_players
from r1_prediction_artifact import validate_r1_prediction_frame


def configure(monkeypatch, *, event_id=557, season=None):
    season = datetime.now().year if season is None else season
    monkeypatch.setitem(sys.modules, "sim_inputs", SimpleNamespace(
        event_ids=[event_id], name_replacements={"robert streb": "streb, robert"},
        live_sim_exclusions={f"{season}:557": ["streb, robert"]},
    ))


def test_exclusion_repairs_late_swap_without_allowing_other_missing_players(monkeypatch):
    configure(monkeypatch)
    players = [f"player {i}" for i in range(11)]
    predictions = pd.DataFrame({
        "player_name": players + ["hodges, lee"], "my_pred": 0.5,
        "wind_adj1": 0.1, "dew_adj1": 0.0,
    })
    live = pd.DataFrame({"player_name": players + ["streb, robert"], "sg_total": 1.2})
    with pytest.raises(ValueError, match="streb, robert"):
        validate_r1_prediction_frame(predictions, active_players=live.player_name)
    filtered = filter_live_sim_players(live)
    validate_r1_prediction_frame(predictions, active_players=filtered.player_name)
    assert len(live) == 12  # Raw source retained.
    assert filtered.sg_total.eq(1.2).all()
    with pytest.raises(ValueError, match="another replacement"):
        validate_r1_prediction_frame(
            predictions, active_players=list(filtered.player_name) + ["another replacement"]
        )


@pytest.mark.parametrize("event_id,season", [(558, None), (557, 2000)])
def test_exclusion_does_not_carry_to_other_events_or_seasons(monkeypatch, event_id, season):
    configure(monkeypatch, event_id=event_id, season=season)
    frame = pd.DataFrame({"player_name": ["streb, robert"]})
    pd.testing.assert_frame_equal(filter_live_sim_players(frame), frame)


def test_canonical_names_and_empty_inputs(monkeypatch):
    configure(monkeypatch)
    frame = pd.DataFrame({"player_name": [" ROBERT STREB ", "player 1"]})
    assert filter_live_sim_players(frame).player_name.tolist() == ["player 1"]
    assert filter_live_sim_players(None) is None
    assert filter_live_sim_players(frame.iloc[:0]).empty

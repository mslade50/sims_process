import pandas as pd
import pytest

import live_stats_engine as lse


def test_r1_multicourse_players_receive_their_exact_sheet_baseline():
    frame = pd.DataFrame({
        "player_name": ["a", "b", "c", "d"],
        "course": ["North", "South", "North", "South"],
    })
    result = lse._attach_authoritative_course_baselines(
        frame,
        "course",
        {"north": 68.7, "south": 70.1},
    )

    assert result["course_score_adj"].tolist() == [68.7, 70.1, 68.7, 70.1]


def test_r1_multicourse_mapping_fails_closed_on_unknown_course():
    frame = pd.DataFrame({
        "player_name": ["a", "b"],
        "course": ["North", "Unknown"],
    })
    with pytest.raises(RuntimeError, match="Unmapped course codes"):
        lse._attach_authoritative_course_baselines(
            frame,
            "course",
            {"north": 68.7, "south": 70.1},
        )


def test_automation_refuses_primary_only_multicourse_baseline_mutation(monkeypatch):
    monkeypatch.setattr(lse, "COURSE_SCORE_MAP", {"north": 68.7, "south": 70.1})
    with pytest.raises(RuntimeError, match="primary-only"):
        lse.update_expected_scores(1, sync_primary=True)

import pandas as pd

import live_stats_engine as engine


def test_r3_merge_recovers_missing_carried_skill_from_current_predictions(
    monkeypatch, tmp_path
):
    prior_path = tmp_path / "r2_live_model.csv"
    pred_path = tmp_path / "model_predictions_r3.csv"

    pd.DataFrame(
        {
            "player_name": ["present, player", "missing, player"],
            "updated_pred_r3": [1.25, float("nan")],
            "base_pred": [1.1, float("nan")],
        }
    ).to_csv(prior_path, index=False)
    pd.DataFrame(
        {
            "player_name": ["present, player", "missing, player"],
            "my_pred3": [9.99, 0.75],
            "wind_adj3": [0.1, 0.2],
            "dew_adj3": [0.0, 0.0],
            "r3_teetime": ["2026-08-22 08:00", "2026-08-22 08:10"],
        }
    ).to_csv(pred_path, index=False)

    paths = {
        "r2_live_model.csv": str(prior_path),
        "model_predictions_r3.csv": str(pred_path),
    }
    monkeypatch.setattr(engine, "_resolve_csv", lambda name: paths[name])

    live = pd.DataFrame(
        {
            "player_name": ["present, player", "missing, player"],
            "r3_teetime": ["stale", "stale"],
        }
    )
    merged = engine._merge_r3r4(live, 3)

    actual = merged.set_index("player_name")["updated_pred_r3"]
    assert actual["present, player"] == 1.25
    assert actual["missing, player"] == 0.75
    assert "_current_round_skill" not in merged.columns

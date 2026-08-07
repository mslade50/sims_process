import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from r1_prediction_artifact import (
    build_r1_prediction_manifest,
    load_matching_r1_predictions,
    manifest_path_for,
    validate_r1_prediction_frame,
)


def _predictions(players):
    return pd.DataFrame({
        "player_name": players,
        "my_pred": [0.1] * len(players),
        "wind_adj1": [0.2] * len(players),
        "dew_adj1": [0.0] * len(players),
    })


class R1PredictionArtifactTests(unittest.TestCase):
    def test_withdrawn_extra_is_allowed(self):
        players = [f"player {index}" for index in range(12)]
        details = validate_r1_prediction_frame(
            _predictions(players),
            active_players=pd.Series(players[:-1]),
        )

        self.assertEqual(details["extra_players"], ["player 11"])

    def test_same_size_field_swap_is_rejected_by_name(self):
        prediction_players = [f"player {index}" for index in range(12)]
        active_players = prediction_players[:-1] + ["late replacement"]

        with self.assertRaisesRegex(ValueError, "late replacement"):
            validate_r1_prediction_frame(
                _predictions(prediction_players),
                active_players=active_players,
            )

    def test_loader_uses_published_snapshot_when_root_is_stale(self):
        active = [f"player {index}" for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_predictions_r1.csv"
            published = Path(directory) / "dashboard_data" / root.name
            published.parent.mkdir()
            _predictions(active[:-1] + ["stale player"]).to_csv(root, index=False)
            _predictions(active + ["withdrawn player"]).to_csv(published, index=False)

            frame, selected, details = load_matching_r1_predictions(
                (root, published),
                active_players=active,
                expected_event_ids=[13],
                expected_tourney="wyndham",
            )

        self.assertEqual(selected, published)
        self.assertEqual(len(frame), 13)
        self.assertEqual(details["extra_players"], ["withdrawn player"])

    def test_manifest_rejects_wrong_event_before_snapshot_is_used(self):
        players = [f"player {index}" for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model_predictions_r1.csv"
            frame = _predictions(players)
            frame.to_csv(artifact, index=False)
            manifest = build_r1_prediction_manifest(
                frame,
                {
                    "event_id": 99,
                    "tourney": "wyndham",
                    "field": players,
                    "sim_run_at": "2026-08-07 00:00:00 UTC",
                },
            )
            manifest_path_for(artifact).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "manifest event"):
                load_matching_r1_predictions(
                    (artifact,),
                    active_players=players,
                    expected_event_ids=[13],
                    expected_tourney="wyndham",
                )


if __name__ == "__main__":
    unittest.main()

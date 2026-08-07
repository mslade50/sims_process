import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prediction_backfill import backfill_missing_predictions


class PredictionBackfillTests(unittest.TestCase):
    def _artifact(self, directory, **overrides):
        payload = {
            "event_id": "13",
            "tourney": "wyndham",
            "pred": {"merritt, troy": -0.66, "coody, pierceson": -0.04},
        }
        payload.update(overrides)
        path = Path(directory) / "sim_fairs.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_fills_missing_player_from_matching_last_sim(self):
        frame = pd.DataFrame({
            "player_name": ["merritt, troy", "coody, pierceson"],
            "pred": [None, 0.25],
        })
        with tempfile.TemporaryDirectory() as directory:
            result, filled = backfill_missing_predictions(
                frame,
                player_col="player_name",
                prediction_col="pred",
                artifact_path=self._artifact(directory),
                expected_event_ids=[13],
                expected_tourney="wyndham",
            )

        self.assertEqual(filled, ["merritt, troy"])
        self.assertEqual(result.loc[0, "pred"], -0.66)
        self.assertEqual(result.loc[1, "pred"], 0.25)

    def test_rejects_artifact_from_another_event(self):
        frame = pd.DataFrame({"player_name": ["merritt, troy"], "pred": [None]})
        with tempfile.TemporaryDirectory() as directory:
            result, filled = backfill_missing_predictions(
                frame,
                player_col="player_name",
                prediction_col="pred",
                artifact_path=self._artifact(directory, event_id="99"),
                expected_event_ids=[13],
                expected_tourney="wyndham",
            )

        self.assertEqual(filled, [])
        self.assertTrue(pd.isna(result.loc[0, "pred"]))

    def test_rejects_artifact_from_another_tournament(self):
        frame = pd.DataFrame({"player_name": ["merritt, troy"], "pred": [None]})
        with tempfile.TemporaryDirectory() as directory:
            result, filled = backfill_missing_predictions(
                frame,
                player_col="player_name",
                prediction_col="pred",
                artifact_path=self._artifact(directory, tourney="other"),
                expected_event_ids=[13],
                expected_tourney="wyndham",
            )

        self.assertEqual(filled, [])
        self.assertTrue(pd.isna(result.loc[0, "pred"]))

    def test_missing_artifact_fails_open(self):
        frame = pd.DataFrame({"player_name": ["merritt, troy"], "pred": [None]})
        result, filled = backfill_missing_predictions(
            frame,
            player_col="player_name",
            prediction_col="pred",
            artifact_path="does-not-exist.json",
            expected_event_ids=[13],
            expected_tourney="wyndham",
        )

        self.assertEqual(filled, [])
        self.assertTrue(pd.isna(result.loc[0, "pred"]))


if __name__ == "__main__":
    unittest.main()

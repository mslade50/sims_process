import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from generate_pin_high_r1 import build_event_key, validate_matching_predictions
from publish_sim_fairs import sync_r1_prediction_artifact


class GeneratePinHighR1Tests(unittest.TestCase):
    def test_builds_pga_archive_event_key(self):
        self.assertEqual(build_event_key("pga", 13, 2026), "pga:R2026013")

    def test_validates_matching_prediction_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "sim_fairs.json"
            predictions = Path(directory) / "model_predictions_r1.csv"
            artifact.write_text(
                json.dumps({
                    "event_id": "13",
                    "tourney": "wyndham",
                    "field": [f"player {index}" for index in range(12)],
                }),
                encoding="utf-8",
            )
            pd.DataFrame({
                "player_name": [f"player {index}" for index in range(12)],
                "my_pred": [index / 10 for index in range(12)],
                "wind_adj1": [0.1] * 12,
                "dew_adj1": [0.0] * 12,
            }).to_csv(predictions, index=False)
            count = validate_matching_predictions(
                artifact,
                predictions,
                expected_event_ids=[13],
                expected_tourney="wyndham",
            )

            self.assertEqual(count, 12)

    def test_rejects_stale_event_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "sim_fairs.json"
            artifact.write_text(
                json.dumps({
                    "event_id": "99",
                    "tourney": "wyndham",
                    "field": [f"player {index}" for index in range(12)],
                }),
                encoding="utf-8",
            )
            predictions = Path(directory) / "model_predictions_r1.csv"
            pd.DataFrame({
                "player_name": [f"player {index}" for index in range(12)],
                "my_pred": [0.0] * 12,
                "wind_adj1": [0.0] * 12,
                "dew_adj1": [0.0] * 12,
            }).to_csv(predictions, index=False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_matching_predictions(
                    artifact,
                    predictions,
                    expected_event_ids=[13],
                    expected_tourney="wyndham",
                )

    def test_syncs_only_complete_r1_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model_predictions_r1.csv"
            destination = Path(directory) / "dashboard" / source.name
            pd.DataFrame({
                "player_name": [f"player {index}" for index in range(12)],
                "my_pred": [0.0] * 12,
                "wind_adj1": [0.1] * 12,
                "dew_adj1": [0.0] * 12,
            }).to_csv(source, index=False)

            result = sync_r1_prediction_artifact(
                source,
                destination,
                payload={
                    "event_id": 13,
                    "tourney": "wyndham",
                    "field": [f"player {index}" for index in range(12)],
                    "sim_run_at": "2026-08-07 00:00:00 UTC",
                },
            )

            self.assertEqual(result, destination)
            self.assertEqual(len(pd.read_csv(destination)), 12)
            manifest = json.loads(destination.with_suffix(".meta.json").read_text())
            self.assertEqual(manifest["event_id"], "13")
            self.assertEqual(manifest["prediction_count"], 12)

    def test_rejects_one_for_one_late_field_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "sim_fairs.json"
            predictions = Path(directory) / "model_predictions_r1.csv"
            prediction_players = [f"player {index}" for index in range(12)]
            active_players = prediction_players[:-1] + ["late replacement"]
            artifact.write_text(
                json.dumps({
                    "event_id": "13",
                    "tourney": "wyndham",
                    "field": active_players,
                }),
                encoding="utf-8",
            )
            pd.DataFrame({
                "player_name": prediction_players,
                "my_pred": [0.0] * 12,
                "wind_adj1": [0.0] * 12,
                "dew_adj1": [0.0] * 12,
            }).to_csv(predictions, index=False)

            with self.assertRaisesRegex(ValueError, "late replacement"):
                validate_matching_predictions(
                    artifact,
                    predictions,
                    expected_event_ids=[13],
                    expected_tourney="wyndham",
                )


if __name__ == "__main__":
    unittest.main()

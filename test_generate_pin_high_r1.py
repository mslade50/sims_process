import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from generate_pin_high_r1 import (
    build_event_key,
    generate,
    validate_matching_predictions,
)
from publish_sim_fairs import sync_r1_prediction_artifact


class GeneratePinHighR1Tests(unittest.TestCase):
    def test_builds_pga_archive_event_key(self):
        self.assertEqual(build_event_key("pga", 13, 2026), "pga:R2026013")

    def test_passes_checkout_crosswalk_to_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector_root = root / "collector"
            module = collector_root / "shot_collector" / "pin_high_correction.py"
            module.parent.mkdir(parents=True)
            module.write_text("", encoding="utf-8")
            database = root / "shots.sqlite3"
            database.write_bytes(b"sqlite")
            crosswalk = root / "img_player_crosswalk.csv"
            crosswalk.write_text("source_player_id,datagolf_player_name\n", encoding="utf-8")
            args = SimpleNamespace(
                collector_root=collector_root,
                db=database,
                artifact=root / "sim_fairs.json",
                preds_source=root / "predictions.csv",
                crosswalk_csv=crosswalk,
                output=root / "pin_high_r1.csv",
                season=2026,
                allow_regression=False,
            )
            sim_inputs = SimpleNamespace(
                event_ids=[13], tour="pga", tourney="wyndham"
            )

            with (
                patch.dict("sys.modules", {"sim_inputs": sim_inputs}),
                patch(
                    "generate_pin_high_r1.validate_matching_predictions",
                    return_value=12,
                ),
                patch("generate_pin_high_r1.subprocess.run") as run_mock,
            ):
                generate(args)

            command = run_mock.call_args.args[0]
            self.assertEqual(
                command[command.index("--crosswalk-csv") + 1],
                str(crosswalk.resolve()),
            )

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

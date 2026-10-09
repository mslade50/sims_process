import json
import tempfile
import unittest
from unittest.mock import Mock
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
    def _write_locked(self, artifact, players, event_id=13):
        frame = _predictions(players)
        frame.to_csv(artifact, index=False)
        manifest_path_for(artifact).write_text(json.dumps(build_r1_prediction_manifest(
            frame, {"event_id": event_id, "tourney": "wyndham", "field": players},
        )), encoding="utf-8")
        return frame

    def test_late_replacement_appends_without_changing_locked_rows_or_file(self):
        players = [f"player {i}" for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r1.csv"
            locked = self._write_locked(path, players)
            before = path.read_bytes()
            builder = Mock(return_value=_predictions(["hardy, nick"]))
            frame, _, details = load_matching_r1_predictions(
                [path], active_players=players[:-1] + ["hardy, nick"],
                expected_event_ids=[13], expected_tourney="wyndham", late_player_builder=builder,
            )
            pd.testing.assert_frame_equal(frame.iloc[:12], locked)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(details["late_players"], ["hardy, nick"])
            builder.assert_called_once()

    def test_wrong_event_cannot_be_repaired(self):
        players = [f"player {i}" for i in range(12)]
        builder = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r1.csv"
            self._write_locked(path, players, event_id=99)
            with self.assertRaisesRegex(ValueError, "manifest event"):
                load_matching_r1_predictions([path], active_players=players[:-1]+["late"],
                    expected_event_ids=[13], expected_tourney="wyndham", late_player_builder=builder)
        builder.assert_not_called()

    def test_complete_snapshot_preferred_over_extending_earlier_candidate(self):
        players = [f"player {i}" for i in range(12)]
        builder = Mock()
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"first.csv", Path(directory)/"second.csv"
            self._write_locked(first, players)
            self._write_locked(second, players[:-1]+["late"])
            _, selected, _ = load_matching_r1_predictions([first, second],
                active_players=players[:-1]+["late"], expected_event_ids=[13],
                expected_tourney="wyndham", late_player_builder=builder)
            self.assertEqual(selected, second)
        builder.assert_not_called()

    def test_infinite_prediction_is_rejected(self):
        frame = _predictions([f"player {i}" for i in range(12)])
        frame.loc[0, "my_pred"] = float("inf")
        with self.assertRaisesRegex(ValueError, "missing prediction/weather"):
            validate_r1_prediction_frame(frame)

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

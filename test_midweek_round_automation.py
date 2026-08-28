import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import midweek_round_automation as automation

from midweek_round_automation import (
    NotReady,
    PipelineFailure,
    _load_course_coordinates,
    evaluate_odds_readiness,
    transition_action,
)
from sheet_config import _parse_course_lat_lon
from sheets_storage import (
    update_round_config_params,
    update_round_config_weather,
)


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def _row(event_id, round_num, books, p1="Player A", p2="Player B"):
    return {
        "p1_player_name": p1,
        "p2_player_name": p2,
        "round": round_num,
        "event_id": str(event_id),
        "odds": {
            book: {"p1": "-110", "p2": "-110"}
            for book in books
        },
    }


def _payload(rows, *, event_id=525, round_num=2, age_minutes=10):
    stamp = NOW - timedelta(minutes=age_minutes)
    return {
        "last_updated": stamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "event_id": event_id,
        "round": round_num,
        "match_list": rows,
    }


class OddsReadinessTests(unittest.TestCase):
    def test_requires_both_books_for_target_event_and_round(self):
        rows = []
        for idx in range(5):
            rows.append(
                _row(
                    525,
                    2,
                    ["betcris", "betonline"],
                    p1=f"Player {idx}",
                    p2=f"Opponent {idx}",
                )
            )
        rows.append(_row(999, 2, ["betcris", "betonline"]))
        rows.append(_row(525, 3, ["betcris", "betonline"]))

        result = evaluate_odds_readiness(
            _payload(rows),
            525,
            min_book_matchups=5,
            now=NOW,
        )

        self.assertEqual(result.target_round, 2)
        self.assertEqual(result.counts, {"betcris": 5, "betonline": 5})
        self.assertEqual(result.scoped_rows, 5)

    def test_rejects_partial_book_coverage(self):
        rows = [
            _row(
                525,
                2,
                ["betcris"],
                p1=f"Player {idx}",
                p2=f"Opponent {idx}",
            )
            for idx in range(5)
        ]
        rows.append(_row(525, 2, ["betonline"]))

        with self.assertRaisesRegex(NotReady, "betonline=1/5"):
            evaluate_odds_readiness(
                _payload(rows),
                525,
                min_book_matchups=5,
                now=NOW,
            )

    def test_manual_pinnacle_pair_accepts_without_betcris(self):
        rows = [
            _row(
                525,
                2,
                ["betonline", "pinnacle"],
                p1=f"Player {idx}",
                p2=f"Opponent {idx}",
            )
            for idx in range(5)
        ]

        result = evaluate_odds_readiness(
            _payload(rows),
            525,
            min_book_matchups=5,
            now=NOW,
            required_books=("betonline", "pinnacle"),
        )

        self.assertEqual(result.counts, {"betonline": 5, "pinnacle": 5})

    def test_manual_pinnacle_pair_rejects_partial_pinnacle_coverage(self):
        rows = [
            _row(
                525,
                2,
                ["betonline"] + (["pinnacle"] if idx < 4 else []),
                p1=f"Player {idx}",
                p2=f"Opponent {idx}",
            )
            for idx in range(5)
        ]

        with self.assertRaisesRegex(NotReady, "pinnacle=4/5"):
            evaluate_odds_readiness(
                _payload(rows),
                525,
                min_book_matchups=5,
                now=NOW,
                required_books=("betonline", "pinnacle"),
            )

    def test_rejects_stale_payload(self):
        rows = [_row(525, 2, ["betcris", "betonline"])]
        with self.assertRaisesRegex(NotReady, "stale"):
            evaluate_odds_readiness(
                _payload(rows, age_minutes=241),
                525,
                min_book_matchups=1,
                max_age_hours=3,
                now=NOW,
            )

    def test_rejects_odds_rows_without_event_scope(self):
        rows = [_row(525, 2, ["betcris", "betonline"])]
        rows[0].pop("event_id")
        payload = _payload(rows)
        payload.pop("event_id")

        with self.assertRaisesRegex(NotReady, "coverage"):
            evaluate_odds_readiness(
                payload,
                525,
                min_book_matchups=1,
                now=NOW,
            )


class TransitionActionTests(unittest.TestCase):
    def test_advances_one_stale_pointer(self):
        self.assertEqual(transition_action(0, 2, None, ""), "advance")

    def test_resumes_failed_transition(self):
        self.assertEqual(transition_action(1, 2, 2, "failed"), "resume")

    def test_skips_completed_transition(self):
        self.assertEqual(transition_action(1, 2, 2, "complete"), "complete")

    def test_force_rebuilds_completed_transition(self):
        self.assertEqual(
            transition_action(3, 4, 4, "complete", force=True),
            "resume",
        )

    def test_prior_event_completion_does_not_block_new_event(self):
        self.assertEqual(
            transition_action(0, 2, 2, "complete", 111, 222),
            "advance",
        )

    def test_rejects_multi_round_gap(self):
        with self.assertRaisesRegex(PipelineFailure, "more than one"):
            transition_action(0, 3, None, "")


class _FakeWorksheet:
    def __init__(self):
        self.rows = [
            ["Parameter", "Value", "Note"],
            ["round", "0", ""],
            ["wind", "", ""],
        ]
        self.updates = []

    def get(self, _range):
        return self.rows

    def update_cells(self, cells, value_input_option=None):
        self.updates.append((cells, value_input_option))


class _FakeSpreadsheet:
    def __init__(self):
        self.ws = _FakeWorksheet()

    def worksheet(self, name):
        if name != "round_config":
            raise AssertionError(name)
        return self.ws


class SheetUpdateTests(unittest.TestCase):
    def test_parameter_update_batches_existing_and_new_rows(self):
        spreadsheet = _FakeSpreadsheet()
        update_round_config_params(
            {"round": 1, "automation_status": "running"},
            notes={"automation_status": "managed"},
            spreadsheet=spreadsheet,
        )

        cells, mode = spreadsheet.ws.updates[0]
        values = {(cell.row, cell.col): cell.value for cell in cells}
        self.assertEqual(mode, "USER_ENTERED")
        self.assertEqual(values[(2, 2)], "1")
        self.assertEqual(values[(4, 1)], "automation_status")
        self.assertEqual(values[(4, 2)], "running")
        self.assertEqual(values[(4, 3)], "managed")

    def test_weather_update_uses_all_future_round_columns(self):
        spreadsheet = _FakeSpreadsheet()
        weather = {
            rnd: {"wind": [rnd] * 15, "dew": [40 + rnd] * 15}
            for rnd in range(2, 5)
        }
        update_round_config_weather(weather, 2, spreadsheet=spreadsheet)

        cells, _ = spreadsheet.ws.updates[0]
        values = {(cell.row, cell.col): cell.value for cell in cells}
        self.assertEqual(len(cells), 90)
        self.assertEqual(values[(3, 10)], 2)
        self.assertEqual(values[(17, 11)], 42)
        self.assertEqual(values[(3, 18)], 4)
        self.assertEqual(values[(17, 19)], 44)


class CourseCoordinateTests(unittest.TestCase):
    def test_parses_valid_sheet_coordinates(self):
        self.assertEqual(
            _parse_course_lat_lon("45.16, -93.235"),
            (45.16, -93.235),
        )

    def test_rejects_invalid_sheet_coordinates(self):
        self.assertEqual(_parse_course_lat_lon("91,-93"), (None, None))
        self.assertEqual(_parse_course_lat_lon("not coordinates"), (None, None))

    def test_sheet_coordinates_take_precedence(self):
        self.assertEqual(
            _load_course_coordinates(999999, 45.16, -93.235),
            (45.16, -93.235),
        )


class ShotCollectorReadinessTests(unittest.TestCase):
    @staticmethod
    def _field():
        return pd.DataFrame({
            "player_name": ["player one", "player two", "withdrawn player"],
            "r2_teetime": ["2026-08-14 07:00", "2026-08-14 07:10", None],
        })

    def test_warns_but_allows_an_incomplete_archived_player(self):
        archive = pd.DataFrame({
            "player_name": ["player one", "player two"],
            "complete": [True, False],
        })
        config = {"event_id": 27, "event_ids": [27], "tour": "pga"}
        with patch("api_utils.fetch_img_player_rounds", return_value=archive):
            with patch("builtins.print") as print_mock:
                active = automation._check_shot_collector_ready(
                    config, self._field(), completed_round=1, target_round=2
                )

        self.assertEqual(active, {"player one", "player two"})
        self.assertIn("player two", str(print_mock.call_args_list))

    def test_requires_every_active_next_round_player_to_be_archived(self):
        archive = pd.DataFrame({
            "player_name": ["player one"],
            "complete": [True],
        })
        config = {"event_id": 27, "event_ids": [27], "tour": "pga"}
        with patch("api_utils.fetch_img_player_rounds", return_value=archive):
            with self.assertRaisesRegex(NotReady, "1/2 archived.*player two"):
                automation._check_shot_collector_ready(
                    config, self._field(), completed_round=1, target_round=2
                )

    def test_ignores_players_without_next_round_tee_times(self):
        archive = pd.DataFrame({
            "player_name": ["player one", "player two"],
            "complete": [True, True],
        })
        config = {"event_id": 27, "event_ids": [27], "tour": "pga"}
        with patch("api_utils.fetch_img_player_rounds", return_value=archive):
            active = automation._check_shot_collector_ready(
                config, self._field(), completed_round=1, target_round=2
            )

        self.assertEqual(active, {"player one", "player two"})

    def test_pin_high_coverage_fails_closed_after_terminal_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "event_key": ["pga:R2026027", "pga:R2026027"],
                "player_name": ["player one", "player two"],
                "n_approaches": [8, 0],
                "coverage_status": ["full", "no_data"],
            }).to_csv(path, index=False)

            with self.assertRaisesRegex(PipelineFailure, "1/2.*player two"):
                automation._validate_pin_high_coverage(
                    path, {"player one", "player two"}, "pga:R2026027"
                )

    def test_pin_high_coverage_requires_six_usable_approaches_per_player(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "event_key": ["pga:R2026027", "pga:R2026027"],
                "player_name": ["player one", "player two"],
                "n_approaches": [8, 3],
                "coverage_status": ["full", "partial"],
            }).to_csv(path, index=False)

            with self.assertRaisesRegex(PipelineFailure, "below 6.*1/2.*player two"):
                automation._validate_pin_high_coverage(
                    path, {"player one", "player two"}, "pga:R2026027"
                )

    def test_pin_high_coverage_accepts_full_strength_active_players(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "event_key": ["pga:R2026027", "pga:R2026027"],
                "player_name": ["player one", "player two"],
                "n_approaches": [8, 6],
                "coverage_status": ["full", "full"],
            }).to_csv(path, index=False)

            automation._validate_pin_high_coverage(
                path, {"player one", "player two"}, "pga:R2026027"
            )

    def test_optional_pin_high_failure_disables_feature_and_returns_warning(self):
        config = {"event_id": 27, "event_ids": [27], "tour": "pga"}
        with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
            automation,
            "_check_shot_collector_ready",
            side_effect=NotReady("one harmless feed gap"),
        ):
            warning = automation._prepare_optional_pin_high(
                config,
                self._field(),
                completed_round=1,
                target_round=2,
            )

            self.assertEqual(warning, "one harmless feed gap")
            self.assertEqual(os.environ["LIVE_PIN_HIGH_ADJ"], "0")

    def test_optional_pin_high_success_keeps_feature_enabled(self):
        config = {"event_id": 27, "event_ids": [27], "tour": "pga"}
        active = {"player one", "player two"}
        with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
            automation, "_check_shot_collector_ready", return_value=active
        ), patch.object(automation, "_run") as run_mock, patch.object(
            automation, "_validate_pin_high_coverage"
        ) as validate_mock:
            warning = automation._prepare_optional_pin_high(
                config,
                self._field(),
                completed_round=1,
                target_round=2,
            )

            self.assertIsNone(warning)
            self.assertEqual(os.environ["LIVE_PIN_HIGH_ADJ"], "1")
            run_mock.assert_called_once()
            validate_mock.assert_called_once()


class PredictionVerificationTests(unittest.TestCase):
    @staticmethod
    def _prediction_frame(scores):
        return pd.DataFrame({
            "my_pred4": [0.4, 0.0],
            "scores_r4": scores,
            "weather_sg_r4": [0.0, 0.0],
            "wind_adj4": [1.0, 1.2],
            "dew_adj4": [-0.1, 0.1],
            "field_skill_mean": [0.2, 0.2],
            "centering_version": ["field_relative_v1"] * 2,
            "centering_group": ["field"] * 2,
        })

    def _run_verification(self, frame):
        prediction = MagicMock()
        prediction.exists.return_value = True
        prediction.stat.return_value.st_size = 100
        prediction.name = "model_predictions_r4.csv"
        root = MagicMock()
        root.__truediv__.return_value = prediction
        with patch.object(automation, "ROOT", root):
            with patch.object(automation.pd, "read_csv", return_value=frame):
                automation._verify_predictions(4)

    def test_accepts_centered_prediction_file(self):
        self._run_verification(self._prediction_frame([0.2, -0.2]))

    def test_rejects_uncentered_prediction_file(self):
        with self.assertRaisesRegex(PipelineFailure, "not field-centered"):
            self._run_verification(self._prediction_frame([0.2, 0.1]))


class WeatherForecastTests(unittest.TestCase):
    def test_ai_wind_overrides_best_match_and_builds_15_hour_arrays(self):
        os.environ["COEFFS_FROM_CACHE"] = "1"
        import api_utils

        round_dates = [
            datetime(2026, 7, 23) + timedelta(days=offset)
            for offset in range(4)
        ]
        timestamps = []
        dew = []
        best_wind = []
        for day in round_dates:
            for hour in range(6, 21):
                timestamps.append(day.replace(hour=hour).strftime("%Y-%m-%dT%H:%M"))
                dew.append(50.0 + hour / 10)
                best_wind.append(8.0)

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "timezone": "America/Chicago",
                    "hourly": {
                        "time": timestamps,
                        "dewpoint_2m": dew,
                        "wind_speed_10m": best_wind,
                    }
                }

        multimodel = api_utils.pd.DataFrame(
            {"time": timestamps, "wind_blend": [12.0] * len(timestamps)}
        )
        with patch.object(api_utils.requests, "get", return_value=_Response()):
            with patch.object(
                api_utils, "fetch_multimodel_wind", return_value=multimodel
            ):
                result = api_utils.fetch_event_weather_forecast(
                    44.0, -93.0, round_dates
                )

        self.assertEqual(result["ai_hours"], 60)
        self.assertEqual(result["timezone"], "America/Chicago")
        for rnd in range(1, 5):
            self.assertEqual(result["ai_hours_by_round"][rnd], 15)
            self.assertEqual(result["rounds"][rnd]["wind"], [12.0] * 15)
            self.assertEqual(len(result["rounds"][rnd]["dew"]), 15)


if __name__ == "__main__":
    unittest.main()

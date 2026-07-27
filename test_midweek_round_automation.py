import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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

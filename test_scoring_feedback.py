"""Regression coverage for no-weather actuals and event-bound feedback."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

import gspread
import pandas as pd

import scoring_feedback as feedback

CONFIG = {"event_id": 554, "tourney": "utah", "course_id": 930, "course_par": 71,
          "course_timezone": "America/Denver"}


class Worksheet:
    def __init__(self):
        self.rows = []
        self.ranges = {}
        self.fail_update = False

    def get_all_values(self):
        return copy.deepcopy(self.rows)

    def append_row(self, row, **kwargs):
        self.rows.append([str(v) for v in row])

    def update(self, *, range_name, values, **kwargs):
        if self.fail_update:
            raise RuntimeError("write failed")
        self.ranges[range_name] = [[str(v) for v in row] for row in values]
        if range_name == "A1:H1":
            self.rows = [[str(v) for v in row] for row in values] + self.rows[1:]

    def get(self, cell_range):
        if cell_range.startswith("AI"):
            row = cell_range[2:]
            return [[self.ranges[f"U{row}:AK{row}"][0][14]]]
        if cell_range == "U10:AK14":
            return [self.ranges.get("U10:AK10", [[]])[0]] + [
                self.ranges.get(f"U{r}:AK{r}", [[]])[0] for r in range(11, 15)]
        return self.ranges.get(cell_range, [])


class Spreadsheet:
    def __init__(self):
        self.tabs = {"round_config": Worksheet()}

    def worksheet(self, name):
        if name not in self.tabs:
            raise gspread.WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, *, title, **kwargs):
        self.tabs[title] = Worksheet()
        return self.tabs[title]


def scores(rnd=1, values=(-2, -2), **attrs):
    frame = pd.DataFrame({"player_name": ["a", "b"], "round": values,
                          "thru": [18, 18], "position": ["1", "2"]})
    frame.attrs.update(event_id="554", year=2026, course_id=930, round_num=rnd,
                       start_date="2026-10-01", tourney="utah", **attrs)
    return frame


def forecast(rnd=1, value=69, **kwargs):
    return feedback.forecast_record(
        feedback.context(CONFIG, rnd, 2026), value=value, players=kwargs.pop("players", ["a", "b"]),
        published_at="2026-09-30T20:00:00Z", round_start_at=f"2026-10-0{rnd}T13:30:00Z",
        source={"kind": "original_pre_round", "sha256": "evidence"}, **kwargs)


class ScoringFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.ss = Spreadsheet()

    def record(self, rnd=1, frame=None, fc=None, **kwargs):
        return feedback.record_actuals(self.ss, CONFIG, rnd, year=2026,
                                       frame=frame if frame is not None else scores(rnd),
                                       forecast=fc or forecast(rnd), **kwargs)

    def test_missing_weather_records_scores_and_zero_feedback(self):
        result = self.record(weather_loader=lambda metadata: {})
        self.assertEqual(result["actual"], 69)
        self.assertEqual(result["forecast_miss"], 0)
        records = feedback.read_receipts(self.ss)
        self.assertEqual([r["status"] for r in records if r["stage"] == "weather"], ["missing"])
        ingested = feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)
        self.assertEqual(ingested["status"], "applied")
        self.assertEqual(ingested["correction"], 0)
        self.assertEqual(ingested["weight"], .5)

    def test_live_actuals_entrypoint_does_not_refresh_forecasts_or_require_coordinates(self):
        os.environ.setdefault("COEFFS_FROM_CACHE", "1")
        import live_stats_engine
        feedback.save_receipt(self.ss, forecast())
        with (patch("sheets_storage.get_spreadsheet", return_value=self.ss),
              patch("sheet_config.load_config", return_value=CONFIG.copy()),
              patch("api_utils.fetch_scoring_round", return_value=scores()),
              patch("scoring_feedback.realized_diagnostics", return_value={}),
              patch("live_stats_engine.refresh_dew_forecasts") as refresh):
            actual = live_stats_engine.write_actuals_to_sheet(1)
        self.assertEqual(actual["status"], "applied")
        self.assertEqual(actual["forecast_miss"], 0)
        refresh.assert_not_called()

    def test_weather_failure_does_not_block_actuals(self):
        result = self.record(weather_loader=Mock(side_effect=RuntimeError("weather down")))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(feedback.read_receipts(self.ss)[-1]["status"], "failed")
        self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)["status"], "applied")

    def test_partial_active_round_is_not_due_and_does_not_write(self):
        frame = scores()
        frame.loc[1, "thru"] = 17
        result = self.record(frame=frame)
        self.assertEqual(result["status"], "not_due")
        self.assertEqual(self.ss.worksheet("round_config").ranges, {})

    def test_missing_round_scores_are_missing(self):
        frame = scores().iloc[:0].copy()
        self.assertEqual(self.record(frame=frame)["status"], "missing")

    def test_stale_event_round_course_and_season_fail_closed(self):
        for key, value in [("event_id", "527"), ("year", 2025), ("course_id", 100), ("round_num", 2)]:
            with self.subTest(key=key):
                frame = scores()
                frame.attrs[key] = value
                self.assertEqual(self.record(frame=frame)["status"], "failed")
        self.assertEqual(self.ss.worksheet("round_config").ranges, {})

    def test_duplicate_backfill_is_idempotent(self):
        first = self.record()
        length = len(feedback.read_receipts(self.ss))
        again = self.record()
        self.assertEqual(first["receipt_id"], again["receipt_id"])
        self.assertEqual(len(feedback.read_receipts(self.ss)), length)

    def test_uncertain_append_is_verified_before_retry(self):
        self.record()
        ledger = self.ss.worksheet(feedback.TAB)
        original = ledger.append_row
        class TransientError(Exception):
            code = 503
        def uncertain(row, **kwargs):
            original(row, **kwargs)
            raise TransientError()
        next_receipt = feedback.receipt(feedback.context(CONFIG, 2, 2026), "weather", "missing")
        with patch.object(ledger, "append_row", side_effect=uncertain) as append, patch("sheet_config.time.sleep"):
            feedback.save_receipt(self.ss, next_receipt)
        append.assert_called_once()
        self.assertEqual(sum(r["receipt_id"] == next_receipt["receipt_id"] for r in feedback.read_receipts(self.ss)), 1)

    def test_worksheet_handles_are_reused_but_receipts_are_read_fresh(self):
        self.record()
        with patch.object(self.ss, "worksheet", wraps=self.ss.worksheet) as lookup:
            self.record()
            feedback.read_receipts(self.ss)
        lookup.assert_not_called()

    def test_hindsight_forecast_rejected_but_actual_still_recorded(self):
        fc = forecast()
        fc["published_at"] = "2026-10-01T13:31:00Z"
        result = self.record(fc=fc)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["actual"], 69)
        self.assertEqual(result["feedback_status"], "failed")
        self.assertIsNone(result["forecast_miss"])
        self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)["status"], "failed")

    def test_wrong_date_forecast_cannot_be_used_as_original(self):
        fc = forecast()
        fc["round_start_at"] = "2026-10-08T13:30:00Z"
        result = self.record(fc=fc)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["feedback_status"], "failed")
        self.assertFalse(any(r["stage"] == "forecast" and r["status"] == "applied" for r in feedback.read_receipts(self.ss)))

    def test_missing_forecast_still_records_actuals(self):
        result = feedback.record_actuals(self.ss, CONFIG, 1, year=2026, frame=scores())
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["feedback_status"], "missing")
        self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)["status"], "missing")

    def test_explicit_player_coverage_exceptions_are_retained(self):
        fc = forecast(players=["a", "replacement_out"], allowed_missing=["replacement_out"], allowed_extra=["b"])
        result = self.record(fc=fc)
        self.assertEqual(result["feedback_status"], "applied")
        self.assertEqual(result["coverage"]["missing"], ["replacement_out"])
        self.assertEqual(result["coverage"]["extra"], ["b"])
        unapproved = forecast(players=["a", "replacement_out"])
        self.assertEqual(self.record(fc=unapproved)["feedback_status"], "failed")

    def test_completed_cut_players_are_included_unplayed_wd_excluded(self):
        frame = scores()
        frame.loc[0, "position"] = "CUT"
        frame.loc[2] = ["wd", None, None, "WD"]
        result = self.record(frame=frame, fc=forecast(players=["a", "b", "wd"]))
        self.assertEqual(result["completed_players"], 2)
        self.assertEqual(result["feedback_status"], "applied")
        self.assertEqual(result["coverage"]["inactive"], ["wd"])

    def test_duplicate_players_and_invalid_completed_scores_fail(self):
        for bad in ("duplicate", "nonfinite", "fraction"):
            frame = scores()
            frame["round"] = frame["round"].astype(float)
            if bad == "duplicate":
                frame.loc[1, "player_name"] = "a"
            else:
                frame.loc[1, "round"] = float("nan") if bad == "nonfinite" else -.5
            self.assertEqual(self.record(frame=frame)["status"], "failed")

    def test_sheet_write_failure_is_receipted_failed(self):
        self.ss.worksheet("round_config").fail_update = True
        self.assertEqual(self.record()["status"], "failed")
        self.assertFalse(any(r["stage"] == "actuals" and r["status"] == "applied" for r in feedback.read_receipts(self.ss)))

    def test_ingestion_rejects_stale_unreceipted_and_edited_sheet_rows(self):
        self.record(frame=scores(values=(-3, -3)))
        self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)["correction"], -.5)
        for index, value in [(11, "527"), (14, "unknown"), (8, "0"), (7, "72"), (6, "68")]:
            with self.subTest(index=index):
                self.record(frame=scores(values=(-3, -3)))
                self.ss.worksheet("round_config").ranges["U11:AK11"][0][index] = value
                self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 1, year=2026)["status"], "failed")

    def test_cancelling_misses_are_applied_zero_not_missing(self):
        self.record(frame=scores(values=(-3, -3)))
        self.record(2, frame=scores(2, values=(-1, -1)))
        result = feedback.ingest_feedback(self.ss, CONFIG, 2, year=2026)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["correction"], 0)
        self.assertEqual(result["weight"], .6)

    def test_missing_round_does_not_increase_evidence_weight(self):
        self.record(frame=scores(values=(-3, -3)))
        result = feedback.ingest_feedback(self.ss, CONFIG, 3, year=2026)
        self.assertEqual(result["missing_rounds"], [2, 3])
        self.assertEqual(result["correction"], -.5)
        self.assertEqual(result["weight"], .5)

    def test_finished_event_is_not_due(self):
        self.assertEqual(feedback.ingest_feedback(self.ss, CONFIG, 4, year=2026)["status"], "not_due")

    def test_github_log_timestamp_precision(self):
        self.assertEqual(feedback.stamp("2026-10-04T09:38:07.1053419Z").microsecond, 105341)

    def test_capture_rejects_hindsight_and_preserves_original(self):
        frame = pd.DataFrame({"player_name": ["a", "b"], "r1_teetime": ["2026-10-01 07:30", "2026-10-01 08:00"]})
        with patch("scoring_feedback.datetime") as clock:
            # Explicit publication time avoids depending on today's date.
            clock.fromisoformat = __import__("datetime").datetime.fromisoformat
            clock.now.return_value = __import__("datetime").datetime(2026, 9, 30)
            feedback.capture_forecast(self.ss, CONFIG, 1, 69, frame, published_at="2026-09-30T20:00:00Z")
            late = feedback.capture_forecast(self.ss, CONFIG, 1, 68, frame, published_at="2026-10-01T14:00:00Z")
        self.assertEqual(late["status"], "not_due")
        original = feedback.select_forecast(feedback.read_receipts(self.ss), feedback.context(CONFIG, 1, 2026))
        self.assertEqual(original["value"], 69)

    def test_corrupt_receipt_does_not_feed_model(self):
        self.record()
        self.ss.worksheet(feedback.TAB).rows[-1][7] = "{}"
        with self.assertRaises(ValueError):
            feedback.read_receipts(self.ss)


class OfficialScoringAPITests(unittest.TestCase):
    def test_schedule_binding_rejects_stale_provider_data(self):
        from api_utils import fetch_scoring_round
        schedule = {"season": 2026, "tour": "pga", "schedule": [{
            "event_id": "554", "course_key": "930", "event_name": "Utah",
            "course": "Black Desert", "start_date": "2026-10-01"}]}
        live = {"event_name": "Utah", "course_name": "Black Desert", "stat_round": "1",
                "last_updated": "2026-10-05 00:18:34 UTC", "live_stats": []}
        cases = [("event_name", "Next Tournament"), ("course_name", "Other Course"),
                 ("stat_round", "2"), ("last_updated", "2025-10-05 00:18:34 UTC")]
        for key, value in cases:
            response1, response2 = Mock(status_code=200), Mock(status_code=200)
            response1.json.return_value = schedule
            response2.json.return_value = {**live, key: value}
            with patch("api_utils.requests.get", side_effect=[response1, response2]):
                with self.assertRaises(ValueError):
                    fetch_scoring_round(554, 2026, 1, 930, "test-key")


if __name__ == "__main__":
    unittest.main()

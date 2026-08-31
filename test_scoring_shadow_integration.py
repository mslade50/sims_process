import os
import unittest
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("COEFFS_FROM_CACHE", "1")

import live_stats_engine
import scoring_baseline  # preload before the scoped os.path.exists mock


class _RoundConfig:
    def get(self, cell_range):
        return [
            [
                "Round", "Realized Wind", "Forecast Dew", "Realized Dew",
                "Wind Impact", "Dew Impact", "Published Forecast", "Actual",
                "Forecast Miss", "Structural Wx Baseline",
                "Structural Residual",
            ],
            ["R1", "3", "70", "70", ".5", "0", "68", "67", "-1"],
            ["R2", "3", "70", "70", ".5", "0", "68", "69", "1"],
            ["R3", "3", "70", "70", ".5", "0", "68", "67", "-1"],
        ]


class _Spreadsheet:
    def worksheet(self, name):
        if name != "round_config":
            raise AssertionError(f"unexpected tab {name}")
        return _RoundConfig()


class ScoringShadowIntegrationTests(unittest.TestCase):
    def _run(self, shadow_side_effect=None, append_side_effect=None):
        baselines = pd.DataFrame([
            {"year": "FINAL", "round_num": rnd, "baseline": value}
            for rnd, value in {
                1: 68.4, 2: 68.6, 3: 69.1, 4: 69.0
            }.items()
        ])
        active = {
            "path": "model_predictions_r4.csv",
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.2,
            "avg_wind": 5.0,
            "wind_effect": 1.0,
            "avg_dew": 68.5,
        }
        shadow_record = {
            "input_hash": "canary",
            "production_candidate": 68.5,
            "prior_score_average": 67.0,
            "transition_used": 0.0,
            "transition_source": "global",
            "weather_delta": 0.0,
            "weather_paired": -999.0,
            "robust_structural": -999.0,
            "shadow_unrounded": -999.0,
            "shadow_display": -999.0,
        }
        updates_seen = []

        def capture_updates(updates, **kwargs):
            updates_seen.append(dict(updates))

        build_kwargs = (
            {"side_effect": shadow_side_effect}
            if shadow_side_effect is not None
            else {"return_value": shadow_record}
        )
        with (
            patch("live_stats_engine.os.path.exists") as exists,
            patch("live_stats_engine.pd.read_csv", return_value=baselines),
            patch(
                "live_stats_engine._load_active_field_context",
                return_value=active,
            ),
            patch(
                "live_stats_engine._build_live_scoring_shadow",
                **build_kwargs,
            ),
            patch("live_stats_engine._print_scoring_shadow"),
            patch(
                "api_utils.compute_wind_factor", return_value=0.2
            ),
            patch(
                "sheets_storage.get_spreadsheet",
                return_value=_Spreadsheet(),
            ),
            patch(
                "sheets_storage.get_round_config_params",
                return_value={
                    "dewpoint_base": "68.5",
                    "expected_score_r4": "67.5",
                },
            ),
            patch(
                "sheets_storage.update_round_config_params",
                side_effect=capture_updates,
            ),
            patch(
                "sheets_storage.append_scoring_shadow",
                side_effect=append_side_effect,
            ) as append,
        ):
            exists.side_effect = lambda path: str(path).startswith(
                "scoring_baseline_"
            )
            before = dict(live_stats_engine.SCORE_ADJS)
            live_stats_engine.update_expected_scores(
                3, sync_primary=True
            )
            after = dict(live_stats_engine.SCORE_ADJS)

        return updates_seen, append, before, after

    def test_shadow_canary_never_reaches_authoritative_updates(self):
        updates, append, before, after = self._run()

        self.assertEqual(len(updates), 1)
        payload = updates[0]
        self.assertEqual(payload["expected_score_r4"], 68.6)
        self.assertEqual(payload["expected_score_1"], 68.6)
        self.assertNotIn(-999, payload.values())
        append.assert_called_once()
        self.assertEqual(before, after)

    def test_shadow_compute_failure_does_not_block_publication(self):
        updates, append, before, after = self._run(
            shadow_side_effect=RuntimeError("shadow failed")
        )

        self.assertEqual(updates[0]["expected_score_r4"], 68.6)
        append.assert_called_once()
        skipped = append.call_args.args[0]
        self.assertEqual(skipped["status"], "skipped")
        self.assertIn("shadow failed", skipped["reason"])
        self.assertEqual(before, after)

    def test_shadow_storage_failure_does_not_block_publication(self):
        updates, append, before, after = self._run(
            append_side_effect=RuntimeError("store failed")
        )

        self.assertEqual(updates[0]["expected_score_r4"], 68.6)
        append.assert_called_once()
        self.assertEqual(before, after)

    def test_builder_skips_multi_course_artifact_with_stale_empty_map(self):
        frame = pd.DataFrame({
            "player_name": ["a", "b"],
            "round": [-2, -1],
            "event_name": ["Test Event", "Test Event"],
            "course_name": ["Shared Event Label", "Shared Event Label"],
            "course_x": ["Course A", "Course B"],
        })
        active = {
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.0,
        }
        actual_grid = [
            ["Round", "Wind Impact", "Dew Impact", "Structural Residual"],
            ["R1", ".2", ".1", "-.4"],
        ]
        with (
            patch("live_stats_engine.COURSE_SCORE_MAP", {}),
            patch("live_stats_engine.os.path.exists", return_value=True),
            patch("live_stats_engine.pd.read_csv", return_value=frame),
            patch("scoring_shadow.compute_shadow_forecast") as compute,
        ):
            with self.assertRaisesRegex(Exception, "multiple courses"):
                live_stats_engine._build_live_scoring_shadow(
                    completed_round=1,
                    baselines={1: 69.0, 2: 69.0},
                    active_context=active,
                    target_weather_effect=0.4,
                    production_candidate=69.0,
                    sheet_before=69.0,
                    published_after=69.0,
                    actual_grid=actual_grid,
                    cut_line=65,
                )
        compute.assert_not_called()

    def test_builder_never_calls_external_api_for_missing_shadow_artifact(self):
        active = {
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.0,
        }
        actual_grid = [
            ["Round", "Wind Impact", "Dew Impact", "Structural Residual"],
            ["R1", ".2", ".1", "-.4"],
        ]
        with (
            patch("live_stats_engine.COURSE_SCORE_MAP", {}),
            patch("live_stats_engine.os.path.exists", return_value=False),
            patch("live_stats_engine.fetch_live_stats") as fetch,
        ):
            with self.assertRaisesRegex(Exception, "local cohort artifact"):
                live_stats_engine._build_live_scoring_shadow(
                    completed_round=1,
                    baselines={1: 69.0, 2: 69.0},
                    active_context=active,
                    target_weather_effect=0.4,
                    production_candidate=69.0,
                    sheet_before=69.0,
                    published_after=69.0,
                    actual_grid=actual_grid,
                    cut_line=65,
                )
        fetch.assert_not_called()

    def test_missing_setup_calibration_is_zero_impact_and_fail_soft(self):
        frame = pd.DataFrame({
            "player_name": ["a", "b"],
            "round": [-2, -1],
            "event_name": ["Test Event", "Test Event"],
            "course_name": ["Test Course", "Test Course"],
        })
        active = {
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.0,
        }
        actual_grid = [
            ["Round", "Wind Impact", "Dew Impact", "Structural Residual"],
            ["R1", ".2", ".1", "-.4"],
        ]
        with (
            patch("live_stats_engine.COURSE_SCORE_MAP", {}),
            patch("live_stats_engine.os.path.exists", return_value=True),
            patch("live_stats_engine.pd.read_csv", return_value=frame),
            patch(
                "scoring_shadow.load_setup_yardage_calibration",
                side_effect=FileNotFoundError("setup calibration missing"),
            ),
            patch("live_stats_engine.fetch_img_hole_geometry") as geometry,
            patch(
                "scoring_shadow.compute_shadow_forecast",
                return_value={"input_hash": "fail-soft"},
            ) as compute,
        ):
            result = live_stats_engine._build_live_scoring_shadow(
                completed_round=1,
                baselines={1: 69.0, 2: 69.0},
                active_context=active,
                target_weather_effect=0.4,
                production_candidate=69.0,
                sheet_before=69.0,
                published_after=69.0,
                actual_grid=actual_grid,
                cut_line=65,
            )

        geometry.assert_not_called()
        setup = compute.call_args.kwargs["setup_yardage"]
        self.assertEqual(setup["status"], "unavailable")
        self.assertEqual(setup["adjustment"], 0.0)
        self.assertEqual(setup["model_version"], "setup-yardage-unavailable")
        self.assertIn("setup calibration missing", setup["reason"])
        self.assertEqual(result["input_hash"], "fail-soft")

    def test_builder_passes_physical_datagolf_course_id_to_setup_model(self):
        frame = pd.DataFrame({
            "player_name": ["a", "b"],
            "round": [-2, -1],
            "event_name": ["Test Event", "Test Event"],
            "course_name": ["Test Course", "Test Course"],
        })
        active = {
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.0,
        }
        actual_grid = [
            ["Round", "Wind Impact", "Dew Impact", "Structural Residual"],
            ["R1", ".2", ".1", "-.4"],
        ]
        geometry = pd.DataFrame([{
            "round_no": 1,
            "hole_no": 1,
            "par": 4,
            "yardage": 400,
            "course_id": "layout:not-a-datagolf-id",
        }])
        geometry.attrs.update(source="local", event_key="pga:R2026060")
        setup_signal = {
            "status": "ok",
            "reason": "",
            "adjustment": 0.0,
        }
        with (
            patch("live_stats_engine.COURSE_SCORE_MAP", {}),
            patch("live_stats_engine.os.path.exists", return_value=True),
            patch("live_stats_engine.pd.read_csv", return_value=frame),
            patch(
                "scoring_shadow.load_setup_yardage_calibration",
                return_value={"calibration_version": "test"},
            ),
            patch(
                "live_stats_engine.fetch_img_hole_geometry",
                return_value=geometry,
            ),
            patch(
                "scoring_shadow.compute_setup_yardage_signal",
                return_value=setup_signal,
            ) as setup_compute,
            patch(
                "scoring_shadow.compute_shadow_forecast",
                return_value={"input_hash": "physical-course"},
            ),
        ):
            result = live_stats_engine._build_live_scoring_shadow(
                completed_round=1,
                baselines={1: 69.0, 2: 69.0},
                active_context=active,
                target_weather_effect=0.4,
                production_candidate=69.0,
                sheet_before=69.0,
                published_after=69.0,
                actual_grid=actual_grid,
                cut_line=65,
            )

        self.assertEqual(
            setup_compute.call_args.kwargs["course_id"],
            live_stats_engine.course_id,
        )
        self.assertEqual(
            setup_compute.call_args.kwargs["event_key"],
            "pga:R2026060",
        )
        self.assertEqual(result["input_hash"], "physical-course")

    def test_builder_prefers_fresh_timing_geometry_before_archive(self):
        hashed_setup = {}

        def forecast(_calibration, **kwargs):
            hashed_setup.update(kwargs["setup_yardage"])
            return {
                "input_hash": "timing-first",
                "setup_status": "ok",
                "setup_yardage": dict(kwargs["setup_yardage"]),
            }

        frame = pd.DataFrame({
            "player_name": ["a", "b"],
            "round": [-2, -1],
            "event_name": ["Test Event", "Test Event"],
            "course_name": ["Test Course", "Test Course"],
        })
        active = {
            "players": 2,
            "player_names": ["a", "b"],
            "field_mean_skill": 1.0,
        }
        actual_grid = [
            ["Round", "Wind Impact", "Dew Impact", "Structural Residual"],
            ["R1", ".2", ".1", "-.4"],
        ]
        geometry = pd.DataFrame([{
            "round_no": 1,
            "hole_no": 1,
            "par": 4,
            "yardage": 400,
            "course_id": "942",
        }])
        geometry.attrs.update(
            source="pga_course_stats_timing_v2",
            event_key="pga:R2026557",
            observed_at_utc="2026-09-17T12:00:00+00:00",
        )
        with (
            patch("live_stats_engine.COURSE_SCORE_MAP", {}),
            patch("live_stats_engine.os.path.exists", return_value=True),
            patch("live_stats_engine.pd.read_csv", return_value=frame),
            patch(
                "scoring_shadow.load_setup_yardage_calibration",
                return_value={"calibration_version": "test"},
            ),
            patch(
                "live_stats_engine._timing_geometry_for_shadow",
                return_value=geometry,
            ),
            patch("live_stats_engine.fetch_img_hole_geometry") as archive,
            patch(
                "scoring_shadow.compute_setup_yardage_signal",
                return_value={"status": "ok", "reason": "", "adjustment": 0.1},
            ) as setup_compute,
            patch(
                "scoring_shadow.compute_shadow_forecast",
                side_effect=forecast,
            ),
        ):
            result = live_stats_engine._build_live_scoring_shadow(
                completed_round=1,
                baselines={1: 69.0, 2: 69.0},
                active_context=active,
                target_weather_effect=0.4,
                production_candidate=69.0,
                sheet_before=69.0,
                published_after=69.0,
                actual_grid=actual_grid,
                cut_line=65,
            )

        archive.assert_not_called()
        self.assertEqual(
            setup_compute.call_args.kwargs["event_key"], "pga:R2026557"
        )
        self.assertNotIn("geometry_observed_at_utc", hashed_setup)
        self.assertEqual(
            result["setup_yardage"]["geometry_observed_at_utc"],
            "2026-09-17T12:00:00+00:00",
        )
        self.assertEqual(
            result["setup_yardage"]["data_source"],
            "pga_course_stats_timing_v2",
        )
        self.assertEqual(result["input_hash"], "timing-first")

    def test_pga_event_id_uses_year_and_three_digit_event_number(self):
        self.assertEqual(
            live_stats_engine._pga_tournament_id(28, 2026), "R2026028"
        )
        self.assertEqual(
            live_stats_engine._pga_tournament_id("R2026557", 2026),
            "R2026557",
        )


if __name__ == "__main__":
    unittest.main()

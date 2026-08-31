import json
import unittest
from unittest.mock import patch

import sheets_storage


LEGACY_SCORING_SHADOW_HEADERS = [
    "run_timestamp", "input_hash", "model_version", "tourney", "year",
    "event_id", "course_id", "target_round", "cut_format", "status",
    "active_players", "cohort_hash", "minimum_coverage", "round_scores",
    "round_coverages", "round_weather_effects",
    "round_structural_residuals", "prior_score_average",
    "transition_global", "transition_course_mean", "transition_course_n",
    "transition_used", "transition_source", "prior_weather_average",
    "target_weather_effect", "weather_delta", "raw_paired",
    "weather_paired", "structural_no_feedback",
    "median_structural_residual", "robust_residual_weight",
    "robust_structural", "paired_blend_weight", "production_candidate",
    "sheet_before", "published_after", "shadow_unrounded", "shadow_display",
    "shadow_minus_production", "calibration_hash", "model_input_hash",
    "weather_granularity", "target_baseline", "target_field_skill",
    "transition_pseudo_count", "reason", "setup_yardage", "setup_status",
    "setup_reason", "setup_calibration_hash", "setup_data_source",
    "setup_event_key", "setup_target_yardage",
    "setup_prior_yardage_average", "setup_yardage_delta",
    "setup_raw_expected_strokes_delta", "setup_historical_round_reference",
    "setup_centered_expected_strokes_delta", "setup_adjustment",
    "setup_was_capped", "shadow_before_setup", "setup_reference_source",
    "setup_reference_course_id", "setup_reference_course_n",
    "setup_reference_pseudocount", "setup_global_round_reference",
    "setup_course_round_mean",
]

SETUP_V3_TRAILING_HEADERS = [
    "setup_schema_version", "setup_geometry_observed_at_utc",
    "setup_training_event_overlap_checked",
    "setup_training_event_keys_sha256", "setup_selected_adjustment_mode",
    "setup_selected_adjustment", "setup_broadie_adjustment",
    "setup_empirical_global_adjustment",
    "setup_empirical_course_eb_adjustment",
    "setup_empirical_global_was_capped",
    "setup_empirical_course_was_capped", "setup_empirical_max_abs_adjustment",
    "setup_yardage_coefficient_model_version",
    "setup_yardage_coefficient_units", "setup_empirical_global_coefficient",
    "setup_empirical_global_cluster_se", "setup_empirical_course_coefficient",
    "setup_empirical_course_coefficient_source",
    "setup_empirical_course_coefficient_fallback_reason",
    "setup_empirical_course_n_informative_years",
    "setup_empirical_course_cluster_se", "setup_empirical_course_n_events",
    "setup_yardage_delta_reference_global",
    "setup_yardage_delta_reference_global_source",
    "setup_centered_yardage_delta_global",
    "setup_yardage_delta_reference_course",
    "setup_yardage_delta_reference_source",
    "setup_yardage_delta_reference_course_id",
    "setup_yardage_delta_reference_course_n",
    "setup_yardage_delta_reference_pseudocount",
    "setup_yardage_delta_reference_fallback_reason",
    "setup_centered_yardage_delta_course",
]


class _Worksheet:
    def __init__(self):
        self.rows = [list(sheets_storage.SCORING_SHADOW_HEADERS)]

    def get_all_values(self):
        return [list(row) for row in self.rows]

    def append_rows(self, rows, value_input_option=None):
        self.rows.extend([list(row) for row in rows])


class _LegacyWorksheet:
    def __init__(self):
        self.headers = list(LEGACY_SCORING_SHADOW_HEADERS)
        self.col_count = len(self.headers)
        self.update_calls = 0

    def row_values(self, row):
        self.assert_header_row(row)
        return list(self.headers)

    def resize(self, cols=None):
        self.col_count = cols

    def update_cells(self, cells, value_input_option=None):
        self.update_calls += 1
        for cell in cells:
            while len(self.headers) < cell.col:
                self.headers.append("")
            self.headers[cell.col - 1] = cell.value

    @staticmethod
    def assert_header_row(row):
        if row != 1:
            raise AssertionError(f"expected header row 1, got {row}")


class _Spreadsheet:
    def __init__(self, worksheet):
        self._worksheet = worksheet

    def worksheet(self, tab_name):
        if tab_name != sheets_storage.TAB_SCORING_SHADOW:
            raise AssertionError(f"unexpected tab {tab_name}")
        return self._worksheet


class ScoringShadowStorageTests(unittest.TestCase):
    def _record(self):
        return {
            "input_hash": "abc123",
            "model_version": "shadow-v1-test",
            "status": "ok",
            "year": 2026,
            "event_id": "60",
            "course_id": 688,
            "target_round": 4,
            "round_scores": {"1": 66.65, "2": 68.10, "3": 67.28},
            "round_coverages": {"1": 1.0, "2": 1.0, "3": 1.0},
            "round_weather_effects": {"1": 0.509},
            "round_structural_residuals": {"1": -1.08},
            "shadow_unrounded": 67.974,
            "shadow_display": 68.0,
        }

    def _v3_record(self):
        record = self._record()
        record.update({
            "setup_status": "ok",
            "setup_reason": "",
            "setup_calibration_hash": "setup-calibration-hash",
            "setup_adjustment": 0.121,
            "shadow_before_setup": 67.853,
        })
        record["setup_yardage"] = {
            "status": "ok",
            "schema_version": "live-scoring-setup-yardage/v3",
            "data_source": "timing_jsonl",
            "event_key": "pga:R2026060",
            "geometry_observed_at_utc": "2026-08-30T15:55:00Z",
            "training_event_overlap_checked": True,
            "training_event_keys_sha256": "training-event-hash",
            "selected_adjustment_mode": "broadie",
            "selected_adjustment": 0.121,
            "broadie_adjustment": 0.121,
            "empirical_global_adjustment": 0.28,
            "empirical_course_eb_adjustment": 0.208,
            "adjustment_arms": {
                "broadie": {
                    "adjustment": 0.121,
                    "was_capped": False,
                },
                "empirical_global": {
                    "adjustment": 0.28,
                    "was_capped": False,
                    "max_abs_adjustment": 0.35,
                    "coefficient": 0.028,
                    "coefficient_source": "global",
                    "coefficient_provenance": {
                        "model_version": "course-yardage-coefficient/v1",
                        "units": "strokes_per_10_yards",
                        "source": "global",
                        "cluster_se": 0.004,
                    },
                    "yardage_reference": 9.0,
                    "yardage_reference_source": "global",
                    "yardage_reference_provenance": {
                        "source": "global",
                    },
                    "centered_yardage_delta": 100.0,
                },
                "empirical_course_eb": {
                    "adjustment": 0.208,
                    "was_capped": False,
                    "max_abs_adjustment": 0.35,
                    "coefficient": 0.02,
                    "coefficient_source": "course_eb:688",
                    "coefficient_provenance": {
                        "model_version": "course-yardage-coefficient/v1",
                        "units": "strokes_per_10_yards",
                        "source": "course_eb:688",
                        "course_id": 688,
                        "n_informative_years": 8,
                        "cluster_se": 0.005,
                        "n_events": 8,
                        "fallback_reason": "",
                    },
                    "yardage_reference": 5.0,
                    "yardage_reference_source": "course_eb:688",
                    "yardage_reference_provenance": {
                        "source": "course_eb:688",
                        "course_id": 688,
                        "course_n": 8,
                        "pseudocount": 5.0,
                        "fallback_reason": "",
                    },
                    "centered_yardage_delta": 104.0,
                },
            },
        }
        return record

    def test_writes_only_isolated_tab_with_fixed_schema(self):
        worksheet = _Worksheet()
        with (
            patch(
                "sheets_storage._get_or_create_tab",
                return_value=worksheet,
            ) as get_tab,
            patch("sheets_storage._append_to_ledger") as ledger,
        ):
            written = sheets_storage.append_scoring_shadow(
                self._record(), spreadsheet=object()
            )

        self.assertEqual(written, 1)
        get_tab.assert_called_once()
        self.assertEqual(
            get_tab.call_args.args[1], sheets_storage.TAB_SCORING_SHADOW
        )
        self.assertEqual(
            get_tab.call_args.args[2],
            sheets_storage.SCORING_SHADOW_HEADERS,
        )
        ledger.assert_not_called()
        self.assertEqual(
            len(worksheet.rows[1]),
            len(sheets_storage.SCORING_SHADOW_HEADERS),
        )
        scores_idx = sheets_storage.SCORING_SHADOW_HEADERS.index(
            "round_scores"
        )
        self.assertEqual(
            json.loads(worksheet.rows[1][scores_idx])["1"], 66.65
        )

    def test_schema_v3_headers_and_nested_arm_mapping_are_exact(self):
        self.assertEqual(
            sheets_storage.SCORING_SHADOW_HEADERS[:len(LEGACY_SCORING_SHADOW_HEADERS)],
            LEGACY_SCORING_SHADOW_HEADERS,
        )
        self.assertEqual(
            sheets_storage.SCORING_SHADOW_HEADERS[len(LEGACY_SCORING_SHADOW_HEADERS):],
            SETUP_V3_TRAILING_HEADERS,
        )

        worksheet = _Worksheet()
        record = self._v3_record()
        with patch(
            "sheets_storage._get_or_create_tab",
            return_value=worksheet,
        ):
            written = sheets_storage.append_scoring_shadow(
                record, spreadsheet=object()
            )

        self.assertEqual(written, 1)
        mapped = dict(zip(sheets_storage.SCORING_SHADOW_HEADERS, worksheet.rows[1]))
        expected_trailing = {
            "setup_schema_version": "live-scoring-setup-yardage/v3",
            "setup_geometry_observed_at_utc": "2026-08-30T15:55:00Z",
            "setup_training_event_overlap_checked": True,
            "setup_training_event_keys_sha256": "training-event-hash",
            "setup_selected_adjustment_mode": "broadie",
            "setup_selected_adjustment": 0.121,
            "setup_broadie_adjustment": 0.121,
            "setup_empirical_global_adjustment": 0.28,
            "setup_empirical_course_eb_adjustment": 0.208,
            "setup_empirical_global_was_capped": False,
            "setup_empirical_course_was_capped": False,
            "setup_empirical_max_abs_adjustment": 0.35,
            "setup_yardage_coefficient_model_version": (
                "course-yardage-coefficient/v1"
            ),
            "setup_yardage_coefficient_units": "strokes_per_10_yards",
            "setup_empirical_global_coefficient": 0.028,
            "setup_empirical_global_cluster_se": 0.004,
            "setup_empirical_course_coefficient": 0.02,
            "setup_empirical_course_coefficient_source": "course_eb:688",
            "setup_empirical_course_coefficient_fallback_reason": "",
            "setup_empirical_course_n_informative_years": 8,
            "setup_empirical_course_cluster_se": 0.005,
            "setup_empirical_course_n_events": 8,
            "setup_yardage_delta_reference_global": 9.0,
            "setup_yardage_delta_reference_global_source": "global",
            "setup_centered_yardage_delta_global": 100.0,
            "setup_yardage_delta_reference_course": 5.0,
            "setup_yardage_delta_reference_source": "course_eb:688",
            "setup_yardage_delta_reference_course_id": 688,
            "setup_yardage_delta_reference_course_n": 8,
            "setup_yardage_delta_reference_pseudocount": 5.0,
            "setup_yardage_delta_reference_fallback_reason": "",
            "setup_centered_yardage_delta_course": 104.0,
        }
        self.assertEqual(
            {header: mapped[header] for header in SETUP_V3_TRAILING_HEADERS},
            expected_trailing,
        )
        self.assertEqual(mapped["setup_data_source"], "timing_jsonl")
        self.assertEqual(
            json.loads(mapped["setup_yardage"]),
            record["setup_yardage"],
        )

    def test_exact_legacy_header_prefix_migrates_once(self):
        worksheet = _LegacyWorksheet()
        spreadsheet = _Spreadsheet(worksheet)

        first = sheets_storage._get_or_create_tab(
            spreadsheet,
            sheets_storage.TAB_SCORING_SHADOW,
            sheets_storage.SCORING_SHADOW_HEADERS,
        )
        second = sheets_storage._get_or_create_tab(
            spreadsheet,
            sheets_storage.TAB_SCORING_SHADOW,
            sheets_storage.SCORING_SHADOW_HEADERS,
        )

        self.assertIs(first, worksheet)
        self.assertIs(second, worksheet)
        self.assertEqual(worksheet.update_calls, 1)
        self.assertEqual(worksheet.headers, sheets_storage.SCORING_SHADOW_HEADERS)
        self.assertEqual(worksheet.col_count, len(sheets_storage.SCORING_SHADOW_HEADERS))

    def test_same_input_hash_is_idempotent(self):
        worksheet = _Worksheet()
        with patch(
            "sheets_storage._get_or_create_tab",
            return_value=worksheet,
        ):
            first = sheets_storage.append_scoring_shadow(
                self._v3_record(), spreadsheet=object()
            )
            second = sheets_storage.append_scoring_shadow(
                self._v3_record(), spreadsheet=object()
            )

        self.assertEqual((first, second), (1, 0))
        self.assertEqual(len(worksheet.rows), 2)

    def test_legacy_record_leaves_new_trailing_columns_blank(self):
        worksheet = _Worksheet()
        with patch(
            "sheets_storage._get_or_create_tab",
            return_value=worksheet,
        ):
            written = sheets_storage.append_scoring_shadow(
                self._record(), spreadsheet=object()
            )

        self.assertEqual(written, 1)
        tail = worksheet.rows[1][len(LEGACY_SCORING_SHADOW_HEADERS):]
        self.assertEqual(tail, [""] * len(SETUP_V3_TRAILING_HEADERS))

    def test_same_hash_in_different_event_is_a_distinct_observation(self):
        worksheet = _Worksheet()
        first_record = self._record()
        second_record = self._record()
        second_record["event_id"] = "61"
        with patch(
            "sheets_storage._get_or_create_tab",
            return_value=worksheet,
        ):
            first = sheets_storage.append_scoring_shadow(
                first_record, spreadsheet=object()
            )
            second = sheets_storage.append_scoring_shadow(
                second_record, spreadsheet=object()
            )

        self.assertEqual((first, second), (1, 1))
        self.assertEqual(len(worksheet.rows), 3)

    def test_requires_input_hash(self):
        with self.assertRaisesRegex(ValueError, "input_hash"):
            sheets_storage.append_scoring_shadow(
                {}, spreadsheet=object()
            )


if __name__ == "__main__":
    unittest.main()

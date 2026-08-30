import json
import unittest
from unittest.mock import patch

import sheets_storage


class _Worksheet:
    def __init__(self):
        self.rows = [list(sheets_storage.SCORING_SHADOW_HEADERS)]

    def get_all_values(self):
        return [list(row) for row in self.rows]

    def append_rows(self, rows, value_input_option=None):
        self.rows.extend([list(row) for row in rows])


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

    def test_same_input_hash_is_idempotent(self):
        worksheet = _Worksheet()
        with patch(
            "sheets_storage._get_or_create_tab",
            return_value=worksheet,
        ):
            first = sheets_storage.append_scoring_shadow(
                self._record(), spreadsheet=object()
            )
            second = sheets_storage.append_scoring_shadow(
                self._record(), spreadsheet=object()
            )

        self.assertEqual((first, second), (1, 0))
        self.assertEqual(len(worksheet.rows), 2)

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

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import live_stats_engine as engine


class LiveSkillPinHighTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "player_name": ["merritt, troy", "skinns, david"],
            "event_name": ["wyndham", "wyndham"],
            "pred": [-0.66, 0.25],
            "total_adjustment": [-0.50, 0.10],
            "updated_pred": [-1.16, 0.35],
            "tot_resid_adj": [-0.40, 0.05],
            "ott_adj": [-0.10, 0.03],
            "putt_adj": [0.00, 0.02],
        })

    def test_pin_high_is_accounted_for_and_exposed_in_csv_and_email(self):
        frame = self._frame()
        with tempfile.TemporaryDirectory() as directory:
            adjustment_path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "event_key": ["pga:R2026013"] * 2,
                "player_name": frame["player_name"],
                "pin_high_adj": [0.06, -0.04],
                "generated_at": [pd.Timestamp.now(tz="UTC").isoformat()] * 2,
            }).to_csv(adjustment_path, index=False)

            with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
                engine,
                "_resolve_csv",
                return_value=str(adjustment_path),
            ), patch.object(engine, "event_ids", [13]):
                result = engine._apply_pin_high_adj(frame.copy())

            pd.testing.assert_series_equal(
                result["updated_pred"].reset_index(drop=True),
                (result["pred"] + result["total_adjustment"]).reset_index(drop=True),
                check_names=False,
            )
            self.assertAlmostEqual(result.loc[0, "total_adjustment"], -0.44)
            self.assertIn("Pin High Adj", engine.build_email_html(result, 1))

            previous = Path.cwd()
            try:
                os.chdir(directory)
                attribution_path = engine.build_attribution_csv(result, 1)
                attribution = pd.read_csv(attribution_path)
            finally:
                os.chdir(previous)

        self.assertIn("pin_high", attribution.columns)
        self.assertAlmostEqual(attribution.loc[0, "pin_high"], -0.04)
        merritt = attribution[attribution["player_name"] == "merritt, troy"].iloc[0]
        self.assertAlmostEqual(merritt["pin_high"], 0.06)

    def test_pin_high_note_reports_progressive_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            adjustment_path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "player_name": ["player, full", "player, partial", "player, none"],
                "pin_high_adj": [0.06, -0.02, 0.0],
                "pin_high_rate": [0.40, 0.75, float("nan")],
                "n_approaches": [8, 3, 0],
                "coverage_status": ["full", "partial", "no_data"],
                "field_mean": [0.50, 0.50, 0.50],
                "generated_at": [pd.Timestamp.now(tz="UTC").isoformat()] * 3,
            }).to_csv(adjustment_path, index=False)
            applied = pd.DataFrame({
                "player_name": ["player, full", "player, partial", "player, none"],
                "pin_high_adj": [0.06, -0.02, 0.0],
            })

            with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
                engine,
                "_resolve_csv",
                return_value=str(adjustment_path),
            ):
                note = engine.build_pin_high_note(applied)

        self.assertIn("1 full / 1 partial / 1 no data", note)
        self.assertIn("n=3", note)

    def test_pin_high_event_mismatch_fails_open_to_zero_adjustments(self):
        frame = self._frame()
        with tempfile.TemporaryDirectory() as directory:
            adjustment_path = Path(directory) / "pin_high_r1.csv"
            pd.DataFrame({
                "event_key": ["pga:R2026099"] * 2,
                "player_name": frame["player_name"],
                "pin_high_adj": [0.06, -0.04],
                "generated_at": [pd.Timestamp.now(tz="UTC").isoformat()] * 2,
            }).to_csv(adjustment_path, index=False)

            with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
                engine,
                "_resolve_csv",
                return_value=str(adjustment_path),
            ), patch.object(engine, "event_ids", [13]):
                result = engine._apply_pin_high_adj(frame.copy())

        self.assertTrue((result["pin_high_adj"] == 0).all())
        pd.testing.assert_series_equal(
            result["total_adjustment"],
            frame["total_adjustment"],
        )


if __name__ == "__main__":
    unittest.main()

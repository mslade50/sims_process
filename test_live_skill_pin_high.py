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
                "player_name": frame["player_name"],
                "pin_high_adj": [0.06, -0.04],
                "generated_at": [pd.Timestamp.now(tz="UTC").isoformat()] * 2,
            }).to_csv(adjustment_path, index=False)

            with patch.dict(os.environ, {"LIVE_PIN_HIGH_ADJ": "1"}), patch.object(
                engine,
                "_resolve_csv",
                return_value=str(adjustment_path),
            ):
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


if __name__ == "__main__":
    unittest.main()

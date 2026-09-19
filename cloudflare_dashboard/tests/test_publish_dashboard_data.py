from pathlib import Path
import sys
import unittest

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from publish_dashboard_data import cache_control, object_key, order_snapshot_files
from export_dashboard_data import _diagnostic_level_bias


class PublishDashboardDataTests(unittest.TestCase):
    def test_manifest_is_published_last(self):
        root = Path("snapshot")
        files = [
            root / "manifest.json",
            root / "performance.json",
            root / "history" / "28-bmw-pre.json",
        ]
        ordered = order_snapshot_files(files, root)
        self.assertEqual(ordered[-1], root / "manifest.json")
        self.assertEqual(len(ordered), 3)

    def test_r2_keys_and_cache_policy(self):
        root = Path("snapshot")
        history = root / "history" / "28-bmw-pre.json"
        manifest = root / "manifest.json"
        self.assertEqual(object_key(history, root), "data/history/28-bmw-pre.json")
        self.assertEqual(cache_control(manifest, root), "no-cache")
        self.assertIn("max-age=300", cache_control(history, root))

    def test_level_bias_uses_adjusted_full_field_rounds_only(self):
        rows = []
        for player, misses in [("cut", [-1.0, -2.0, -100.0]), ("made", [1.0, 2.0, 100.0])]:
            for round_num, miss in enumerate(misses, start=1):
                rows.append(
                    {
                        "event_id": 1,
                        "event_name": "test",
                        "year": 2026,
                        "player_name": player,
                        "round": round_num,
                        "category": "ott",
                        "predicted_sg": 0.0,
                        "actual_sg": miss,
                        "miss": miss,
                        "sg_type": "adjusted",
                    }
                )
        rows.append({**rows[0], "player_name": "raw", "miss": 999.0, "actual_sg": 999.0, "sg_type": "raw"})

        result = _diagnostic_level_bias(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rounds"], 4)
        self.assertAlmostEqual(result[0]["miss"], 0.0)


if __name__ == "__main__":
    unittest.main()

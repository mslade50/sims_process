from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from publish_dashboard_data import cache_control, object_key, order_snapshot_files


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


if __name__ == "__main__":
    unittest.main()

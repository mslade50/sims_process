import json
import tempfile
import unittest
from pathlib import Path

from push_odds_screen import _write_payload_files


class PushOddsScreenOutputTests(unittest.TestCase):
    def test_writes_publish_ready_json_atomically(self):
        payloads = {
            "round_matchups.json": {"round": 4, "matchups": [{"name": "a"}]},
            "meta.json": {"tourney": "test_event", "round": 4},
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "nested"
            written = _write_payload_files(payloads, output_dir)

            self.assertEqual(
                written,
                [output_dir / "round_matchups.json", output_dir / "meta.json"],
            )
            for name, expected in payloads.items():
                with (output_dir / name).open(encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), expected)
            self.assertEqual(list(output_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

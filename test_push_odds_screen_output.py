import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from push_odds_screen import (
    _atomic_upload_plan,
    _upload_atomic_payload_bundle,
    _write_atomic_payload_bundle,
    _write_payload_files,
)


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

    def test_atomic_bundle_uses_hashed_generation_and_pointer_last(self):
        payloads = {
            "round_matchups.json": {"round": 4, "matchups": [{"name": "a"}]},
            "meta.json": {
                "tourney": "test_event",
                "round": 4,
                "last_updated": "2026-08-23 12:34:56 UTC",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            written = _write_atomic_payload_bundle(
                payloads, output_dir, generation="test-generation"
            )
            self.assertEqual(written[-1], output_dir / "meta.json")
            pointer = json.loads((output_dir / "meta.json").read_text())
            self.assertEqual(pointer["generation"], "test-generation")
            self.assertEqual(pointer["schema_version"], "odds-screen-generation/v1")
            for name, binding in pointer["files"].items():
                path = output_dir / binding["key"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), binding["sha256"]
                )

    def test_r2_pointer_is_not_uploaded_when_a_generation_object_fails(self):
        class BrokenClient:
            def __init__(self):
                self.keys = []

            def put_object(self, **kwargs):
                self.keys.append(kwargs["Key"])
                if kwargs["Key"].endswith("round_matchups.json"):
                    raise RuntimeError("transport failed")

        client = BrokenClient()
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            _upload_atomic_payload_bundle(
                client,
                {
                    "round_matchups.json": {"round": 4},
                    "meta.json": {"round": 4, "last_updated": "2026-08-23 UTC"},
                },
                generation="failed-generation",
            )
        self.assertNotIn("odds_data/meta.json", client.keys)

    def test_upload_plan_covers_every_declared_file_and_pointer_is_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_atomic_payload_bundle(
                {
                    "round_matchups.json": {"round": 4},
                    "future_market.json": {"rows": [1]},
                    "meta.json": {"round": 4, "last_updated": "2026-08-23 UTC"},
                },
                output_dir,
                generation="future-generation",
            )
            plan = _atomic_upload_plan(output_dir)
            self.assertEqual(plan[-1][0], "meta.json")
            self.assertEqual(
                {key for key, _ in plan[:-1]},
                {
                    "generations/future-generation/round_matchups.json",
                    "generations/future-generation/future_market.json",
                    "generations/future-generation/meta.json",
                },
            )
            pointer_path = output_dir / "meta.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["files"]["omitted_market.json"] = {
                "key": "generations/future-generation/omitted_market.json",
                "sha256": "0" * 64,
                "size": 1,
            }
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing"):
                _atomic_upload_plan(output_dir)


if __name__ == "__main__":
    unittest.main()

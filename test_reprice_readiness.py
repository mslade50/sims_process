"""Offline readiness regressions for scrape-triggered repricing."""

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import reprice
from sim_health_gate import SimulationHealthError, utc_stamp


NOW = datetime(2026, 10, 2, 1, 0, tzinfo=timezone.utc)
CONFIG = {"tourney": "utah", "event_id": 554, "round_num": 1}


class RepriceReadinessTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def bundle(self, *, event_id="554", tourney="utah", round_num=2,
               generated=NOW, root_generated=None):
        identity = {"event_id": event_id, "tourney": tourney, "round": round_num}
        (self.root / "round_h2h_r2.parquet").write_bytes(b"tape")
        (self.root / "round_h2h_r2_meta.json").write_text(json.dumps(identity))
        health = {"simulation_manifest": {
            "event": identity,
            "source": {
                "generated_at": utc_stamp(generated),
                "root_generated_at": utc_stamp(root_generated or generated),
            },
        }}
        (self.root / "round_h2h_r2_health.json").write_text(json.dumps(health))

    def readiness(self):
        return reprice.reprice_not_ready_reason(
            self.root, tourney="utah", event_id=554, sim_round=2, now=NOW
        )

    def run_main(self, *args):
        rc = types.ModuleType("reprice_core")
        rc.send_telegram = Mock()
        cfg_module = types.ModuleType("sheet_config")
        cfg_module.load_config = Mock(return_value=CONFIG)
        inputs = types.ModuleType("sim_inputs")
        inputs.name_replacements = {}
        output = self.root / "github-output"
        with patch.dict(sys.modules, {
            "reprice_core": rc, "sheet_config": cfg_module, "sim_inputs": inputs,
        }), patch.object(sys, "argv", ["reprice.py", *args]), \
                patch.object(reprice, "_setup_env", return_value=str(self.root)), \
                patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
            result = reprice.main()
        rc.send_telegram.assert_not_called()
        return result, output.read_text()

    def test_no_sim_is_successful_skip_without_telegram(self):
        result, output = self.run_main()
        self.assertEqual(result, 0)
        self.assertEqual(output, "ready=false\n")

    def test_missing_bundle_is_not_ready(self):
        self.assertIn("not published", self.readiness())

    def test_expired_sim_is_successful_skip_without_telegram(self):
        self.bundle(generated=datetime.now(timezone.utc) - timedelta(hours=28.5))
        result, output = self.run_main()
        self.assertEqual(result, 0)
        self.assertEqual(output, "ready=false\n")

    def test_prior_event_and_round_are_not_ready(self):
        for identity in ({"event_id": "557"}, {"tourney": "biltmore"},
                         {"round_num": 1}):
            with self.subTest(identity=identity):
                self.bundle(**identity)
                self.assertIsNotNone(self.readiness())

    def test_current_fresh_bundle_is_ready(self):
        self.bundle()
        self.assertIsNone(self.readiness())

    def test_expired_sim_and_expired_root_are_not_ready(self):
        for source in ("generated", "root_generated"):
            with self.subTest(source=source):
                self.bundle(**{source: NOW - timedelta(hours=28.5)})
                self.assertIn("expired", self.readiness())

    def test_freshness_boundary_matches_health_gate(self):
        self.bundle(generated=NOW - timedelta(hours=18))
        self.assertIsNone(self.readiness())
        self.bundle(generated=NOW - timedelta(hours=18, seconds=1))
        self.assertIsNotNone(self.readiness())

    def test_current_partial_or_empty_bundle_is_error(self):
        for name in ("round_h2h_r2.parquet", "round_h2h_r2_meta.json",
                     "round_h2h_r2_health.json"):
            with self.subTest(missing=name):
                self.bundle()
                (self.root / name).unlink()
                with self.assertRaises(SimulationHealthError):
                    self.readiness()
            with self.subTest(empty=name):
                self.bundle()
                (self.root / name).write_bytes(b"")
                with self.assertRaises((SimulationHealthError, ValueError)):
                    self.readiness()

    def test_conflicting_current_bundle_is_error(self):
        self.bundle()
        path = self.root / "round_h2h_r2_health.json"
        health = json.loads(path.read_text())
        health["simulation_manifest"]["event"]["event_id"] = "557"
        path.write_text(json.dumps(health))
        with self.assertRaises(SimulationHealthError):
            self.readiness()

    def test_invalid_or_future_timestamp_is_error(self):
        for stamp in ("invalid", utc_stamp(NOW + timedelta(minutes=6))):
            with self.subTest(stamp=stamp):
                self.bundle()
                path = self.root / "round_h2h_r2_health.json"
                health = json.loads(path.read_text())
                health["simulation_manifest"]["source"]["generated_at"] = stamp
                path.write_text(json.dumps(health))
                with self.assertRaises(SimulationHealthError):
                    self.readiness()

    def test_preflight_does_not_start_pricing(self):
        self.bundle(generated=datetime.now(timezone.utc))
        result, output = self.run_main("--check-ready")
        self.assertEqual(result, 0)
        self.assertEqual(output, "ready=true\n")


if __name__ == "__main__":
    unittest.main()

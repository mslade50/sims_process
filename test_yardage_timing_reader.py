import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from yardage_timing_reader import (
    YardageTimingUnavailable,
    load_latest_round_geometry,
)


NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _setup(round_no, offset=0):
    holes = []
    for hole in range(1, 19):
        par = 3 if hole in (2, 7, 11, 14) else 5 if hole in (3, 8, 15, 18) else 4
        base = {3: 180, 4: 430, 5: 560}[par]
        holes.append({
            "hole": hole,
            "par": par,
            "yards": base + hole + offset,
            "live": False,
            "pin_published": False,
        })
    canonical = [
        {"hole": row["hole"], "par": row["par"], "yards": row["yards"]}
        for row in holes
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "round": round_no,
        "round_header": f"Round {round_no}",
        "complete": True,
        "hole_count": 18,
        "total_yards": sum(row["yards"] for row in holes),
        "setup_sha256": digest,
        "holes": holes,
    }


def _observation(received=NOW - timedelta(minutes=2)):
    return {
        "schema_version": "pga-yardage-availability/v2",
        "event_id": "R2026557",
        "sources": {
            "course_stats": {
                "http_status": 200,
                "received_at_utc": received.isoformat(),
                "data": {
                    "tournament_id": "R2026557",
                    "courses": [{
                        "course_id": "942",
                        "setups": [
                            _setup(1, 0),
                            _setup(2, 1),
                            _setup(3, -1),
                            _setup(4, 3),
                            {"round": None, "complete": True, "holes": []},
                        ],
                    }],
                },
            },
        },
        "summary": {
            "round_course_completeness": {
                str(rnd): {
                    "complete": True,
                    "used_course_ids": ["942"],
                    "complete_course_ids": ["942"],
                }
                for rnd in range(1, 5)
            },
        },
    }


class YardageTimingReaderTests(unittest.TestCase):
    def _write(self, rows, trailing=b""):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "timing.jsonl"
        with path.open("wb") as handle:
            for row in rows:
                handle.write(json.dumps(row).encode() + b"\n")
            handle.write(trailing)
        self.addCleanup(temp.cleanup)
        return path

    def test_reads_latest_complete_numbered_setups(self):
        old = _observation(NOW - timedelta(minutes=9))
        latest = _observation()
        latest["schema_version"] = "pga-yardage-availability/v2"
        path = self._write([
            {"schema_version": "pga-yardage-availability/v1"},
            old,
            latest,
        ], trailing=b'{"partial":')

        rows, provenance = load_latest_round_geometry(
            path,
            event_id="R2026557",
            target_round=4,
            expected_pga_course_id="942",
            now=NOW,
        )

        self.assertEqual(len(rows), 72)
        self.assertEqual({row["round_no"] for row in rows}, {1, 2, 3, 4})
        self.assertEqual({row["source"] for row in rows}, {"pga_course_stats_timing_v2"})
        self.assertEqual(provenance["event_key"], "pga:R2026557")
        self.assertAlmostEqual(provenance["age_minutes"], 2.0)

    def test_rejects_stale_observation(self):
        path = self._write([_observation(NOW - timedelta(minutes=16))])
        with self.assertRaisesRegex(YardageTimingUnavailable, "stale"):
            load_latest_round_geometry(
                path, event_id="R2026557", target_round=4, now=NOW
            )

    def test_rejects_nominal_only_or_incomplete_target(self):
        row = _observation()
        row["summary"]["round_course_completeness"]["4"]["complete"] = False
        path = self._write([row])
        with self.assertRaisesRegex(YardageTimingUnavailable, "not complete"):
            load_latest_round_geometry(
                path, event_id="R2026557", target_round=4, now=NOW
            )

    def test_rejects_setup_hash_mismatch(self):
        row = _observation()
        row["sources"]["course_stats"]["data"]["courses"][0]["setups"][3][
            "setup_sha256"
        ] = "bad"
        path = self._write([row])
        with self.assertRaisesRegex(YardageTimingUnavailable, "hash"):
            load_latest_round_geometry(
                path, event_id="R2026557", target_round=4, now=NOW
            )

    def test_rejects_malformed_interior_row(self):
        path = self._write([_observation()])
        payload = path.read_bytes()
        path.write_bytes(b"{bad}\n" + payload)
        with self.assertRaisesRegex(YardageTimingUnavailable, "interior"):
            load_latest_round_geometry(
                path, event_id="R2026557", target_round=4, now=NOW
            )

    def test_rejects_wrong_event_or_course(self):
        path = self._write([_observation()])
        with self.assertRaisesRegex(YardageTimingUnavailable, "matches"):
            load_latest_round_geometry(
                path, event_id="R2026060", target_round=4, now=NOW
            )
        with self.assertRaisesRegex(YardageTimingUnavailable, "does not match"):
            load_latest_round_geometry(
                path,
                event_id="R2026557",
                target_round=4,
                expected_pga_course_id="999",
                now=NOW,
            )


if __name__ == "__main__":
    unittest.main()

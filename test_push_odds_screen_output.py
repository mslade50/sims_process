import json
import hashlib
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from push_odds_screen import (
    OddsScreenContractError,
    _atomic_upload_plan,
    _build_outrights,
    _build_round_matchups,
    _build_score_lines,
    _build_tournament_matchups,
    _fetch_dg_outrights,
    _fetch_scraped_guarded,
    _load_committed_release_contract,
    _oriented_h2h_probability,
    _upload_atomic_payload_bundle,
    _validate_odds_screen_payloads,
    _validate_sim_fairs_semantics,
    _write_atomic_payload_bundle,
    _write_payload_files,
)
from sim_health_gate import file_sha256, seal_manifest


PROVENANCE = {
    "event_id": "99",
    "tourney": "test_event",
    "round": 4,
    "release_generation": "release-generation",
    "release_manifest_sha256": "a" * 64,
    "simulation_manifest_sha256": "b" * 64,
    "live_tournament_manifest_sha256": "c" * 64,
    "source_git_sha": "d" * 40,
}


def _write_indexed_parquet(path: Path, frame: pd.DataFrame, metadata: dict):
    table = pa.Table.from_pandas(frame, preserve_index=True)
    encoded = {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **encoded})
    pq.write_table(table, path)


def _complete_payloads():
    fair = {
        "p1": -120,
        "p2": 120,
        "p1_prob": 0.54545,
        "p2_prob": 0.45455,
    }
    outright_rows = [
        {
            "player": "a",
            "sim_prob": 0.6,
            "fair_odds": -150,
            "books": {"betcris": {"yes": 120}},
            "edge": {},
        }
    ]
    return {
        "round_matchups.json": {
            **PROVENANCE,
            "matchups": [
                {
                    "p1": "a",
                    "p2": "b",
                    "fair": fair,
                    "books": {"betcris": {"p1": -110, "p2": -110}},
                }
            ],
        },
        "tournament_matchups.json": {
            **PROVENANCE,
            "matchups": [
                {
                    "p1": "a",
                    "p2": "b",
                    "fair": fair,
                    "books": {"betcris": {"p1": -110, "p2": -110}},
                }
            ],
        },
        "score_lines.json": {
            **PROVENANCE,
            "lines": [
                {
                    "player": "a",
                    "pred": 70.1,
                    "line": 70.5,
                    "fair": {
                        "over": -105,
                        "under": -115,
                        "over_prob": 0.49,
                        "under_prob": 0.51,
                    },
                    "books": {"betcris": {"over": -110, "under": -110}},
                }
            ],
        },
        "outrights.json": {
            **PROVENANCE,
            "markets": {
                market: [dict(row) for row in outright_rows]
                for market in ("winner", "top_5", "top_10", "top_20")
            },
        },
        "meta.json": {**PROVENANCE, "last_updated": "2026-08-23 12:34:56 UTC"},
    }


def _write_strict_release(
    root: Path,
    *,
    release_generated_at: str | None = None,
    simulation_generated_at: str | None = None,
    root_generated_at: str | None = None,
):
    current = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    generated_at = release_generated_at or current
    simulation_generated_at = simulation_generated_at or current
    root_generated_at = root_generated_at or simulation_generated_at
    simulation_manifest = seal_manifest(
        {
            "kind": "round_simulation",
            "event": {"event_id": "99", "tourney": "test_event", "round": 4},
            "source": {
                "generated_at": simulation_generated_at,
                "root_generated_at": root_generated_at,
            },
            "approval": {"status": "approved"},
            "checks": {"passed": True},
        }
    )
    simulation_id = simulation_manifest["manifest_sha256"]
    live_health = seal_manifest(
        {
            "kind": "live_tournament_tape",
            "generated_at": generated_at,
            "simulation_manifest": simulation_manifest,
            "files": {
                label: {"path": name, "sha256": "e" * 64}
                for label, name in {
                    "final_scores": "final_scores_live_test_event.npy",
                    "player_names": "player_names_live_test_event.json",
                    "made_cut": "made_cut_live_test_event.npy",
                    "finish_probs": "simulated_probs_live.csv",
                    "finish_probs_event": "top_finish_probs_live_test_event.csv",
                }.items()
            },
        }
    )
    live_id = live_health["manifest_sha256"]
    generation = f"event-99-r4-{simulation_id[:12]}-{live_id[:12]}"
    field = ["a", "b"]
    fairs = {
        "event_id": "99",
        "event_name": "Test Event",
        "tourney": "test_event",
        "round": 4,
        "generated_at": generated_at,
        "release_generation": generation,
        "simulation_manifest_sha256": simulation_id,
        "live_tournament_manifest_sha256": live_id,
        "outrights_source": "live",
        "matchups_source": "final_scores_live",
        "field": field,
        "outrights": {
            market: {"a": 0.6, "b": 0.4}
            for market in ("winner", "top_5", "top_10", "top_20", "make_cut")
        },
        "outrights_nodh": {
            market: {"a": 0.6, "b": 0.4}
            for market in ("top_5", "top_10", "top_20")
        },
        "matchups": [["a", "b", 0.6]],
        "round_scores": {
            "a": {"70": 0.6, "71": 0.4},
            "b": {"70": 0.4, "71": 0.6},
        },
    }
    (root / "sim_fairs.json").write_text(json.dumps(fairs), encoding="utf-8")
    pd.DataFrame(
        [{"player_a": "a", "player_b": "b", "p_a_lt_b": 0.55, "p_tie": 0.1}]
    ).to_parquet(root / "round_h2h_r4.parquet", index=False)
    h2h_meta = {
        "event_id": "99",
        "tourney": "test_event",
        "round": 4,
        "source_manifest_sha256": simulation_id,
        "num_players": 2,
    }
    (root / "round_h2h_r4_meta.json").write_text(
        json.dumps(h2h_meta), encoding="utf-8"
    )
    h2h_health = seal_manifest(
        {
            "kind": "published_round_h2h",
            "generated_at": generated_at,
            "simulation_manifest": simulation_manifest,
            "files": {
                "h2h_parquet": {
                    "path": "round_h2h_r4.parquet",
                    "sha256": file_sha256(root / "round_h2h_r4.parquet"),
                },
                "h2h_meta": {
                    "path": "round_h2h_r4_meta.json",
                    "sha256": file_sha256(root / "round_h2h_r4_meta.json"),
                },
            },
        }
    )
    (root / "round_h2h_r4_health.json").write_text(
        json.dumps(h2h_health), encoding="utf-8"
    )

    _write_indexed_parquet(
        root / "round_samples.parquet",
        pd.DataFrame([[70, 71], [71, 70]], index=field, columns=["0", "1"]),
        {
            "event_id": "99",
            "tourney": "test_event",
            "round": 4,
            "sim_run_at": generated_at,
        },
    )
    tournament_metadata = {
        "event_id": "99",
        "tourney": "test_event",
        "source": "final_scores_live",
        "simulation_manifest_sha256": simulation_id,
        "live_tournament_manifest_sha256": live_id,
        "sim_run_at": generated_at,
    }
    _write_indexed_parquet(
        root / "tournament_samples.parquet",
        pd.DataFrame([[280, 281], [282, 279]], index=field, columns=["0", "1"]),
        tournament_metadata,
    )
    _write_indexed_parquet(
        root / "tournament_made_cut.parquet",
        pd.DataFrame([[1, 1], [1, 0]], index=field, columns=["0", "1"]),
        {**tournament_metadata, "source": "made_cut_live"},
    )
    (root / "tournament_live_test_event_health.json").write_text(
        json.dumps(live_health), encoding="utf-8"
    )

    pd.DataFrame(
        columns=["player_a", "player_b", "player_c", "p_a", "p_b", "p_c"]
    ).to_parquet(root / "round_3ball_r4.parquet", index=False)
    threeball_meta = {
        "event_id": "99",
        "tourney": "test_event",
        "round": 4,
        "status": "no_groups_offered",
        "num_groups": 0,
    }
    (root / "round_3ball_r4_meta.json").write_text(
        json.dumps(threeball_meta), encoding="utf-8"
    )
    threeball_contract = seal_manifest(
        {
            "kind": "published_round_3ball",
            "generated_at": generated_at,
            "simulation_manifest": simulation_manifest,
            "files": {
                "threeball_parquet": {
                    "path": "round_3ball_r4.parquet",
                    "sha256": file_sha256(root / "round_3ball_r4.parquet"),
                },
                "threeball_meta": {
                    "path": "round_3ball_r4_meta.json",
                    "sha256": file_sha256(root / "round_3ball_r4_meta.json"),
                },
            },
            "extra": {
                "status": "no_groups_offered",
                "event_id": "99",
                "round": 4,
                "num_groups": 0,
                "tee_group_source": {
                    "round": 4,
                    "requested_event_id": "99",
                    "event_identity_verified": True,
                    "simulation_field_overlap": 1.0,
                    "simulation_tee_time_coverage": 1.0,
                },
            },
        }
    )
    (root / "round_3ball_r4_contract.json").write_text(
        json.dumps(threeball_contract), encoding="utf-8"
    )
    declared = [
        "sim_fairs.json",
        "round_samples.parquet",
        "round_h2h_r4.parquet",
        "round_h2h_r4_meta.json",
        "round_h2h_r4_health.json",
        "round_3ball_r4.parquet",
        "round_3ball_r4_meta.json",
        "round_3ball_r4_contract.json",
        "tournament_samples.parquet",
        "tournament_made_cut.parquet",
        "tournament_live_test_event_health.json",
    ]
    manifest = seal_manifest(
        {
            "schema_version": "complete-live-package/v1",
            "generation": generation,
            "event_id": "99",
            "tourney": "test_event",
            "round": 4,
            "generated_at": generated_at,
            "simulation_manifest_sha256": simulation_id,
            "live_tournament_manifest_sha256": live_id,
            "release_assets": {
                label: {
                    "name": (
                        f"{label}.{generation}.{str(index) * 16}.parquet"
                    ),
                    "sha256": str(index) * 64,
                    "size": index,
                }
                for index, label in enumerate(
                    (
                        "tournament_samples_full",
                        "tournament_made_cut_full",
                        "matchup_scores_live",
                    ),
                    start=1,
                )
            },
            "git_files": {
                relative: {
                    "sha256": file_sha256(root / relative),
                    "size": (root / relative).stat().st_size,
                }
                for relative in declared
            },
        }
    )
    (root / "sim_release_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


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
        payloads = _complete_payloads()
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
                _complete_payloads(),
                generation="failed-generation",
            )
        self.assertNotIn("odds_data/meta.json", client.keys)

    def test_upload_plan_covers_every_declared_file_and_pointer_is_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_atomic_payload_bundle(
                {**_complete_payloads(), "future_market.json": {**PROVENANCE, "rows": [1]}},
                output_dir,
                generation="future-generation",
            )
            plan = _atomic_upload_plan(output_dir)
            self.assertEqual(plan[-1][0], "meta.json")
            self.assertEqual(
                {key for key, _ in plan[:-1]},
                {
                    f"generations/future-generation/{name}"
                    for name in (*_complete_payloads(), "future_market.json")
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

    def test_semantic_validation_rejects_books_only_generation(self):
        payloads = _complete_payloads()
        payloads["round_matchups.json"]["matchups"][0]["fair"] = {}
        with self.assertRaisesRegex(OddsScreenContractError, "no committed model fair"):
            _validate_odds_screen_payloads(payloads)

    def test_semantic_validation_rejects_model_only_generation(self):
        payloads = _complete_payloads()
        payloads["score_lines.json"]["lines"][0]["books"] = {}
        with self.assertRaisesRegex(OddsScreenContractError, "no usable current book quote"):
            _validate_odds_screen_payloads(payloads)

    def test_semantic_validation_rejects_nonfinite_score_line(self):
        payloads = _complete_payloads()
        payloads["score_lines.json"]["lines"][0]["line"] = float("nan")
        with self.assertRaisesRegex(OddsScreenContractError, "finite market line"):
            _validate_odds_screen_payloads(payloads)

    def test_semantic_validation_rejects_integer_score_line(self):
        payloads = _complete_payloads()
        payloads["score_lines.json"]["lines"][0]["line"] = 70
        with self.assertRaisesRegex(OddsScreenContractError, "half-strokes"):
            _validate_odds_screen_payloads(payloads)

    def test_atomic_bundle_rejects_nonstandard_json_numbers(self):
        payloads = {
            **_complete_payloads(),
            "future_market.json": {**PROVENANCE, "value": float("inf")},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                _write_atomic_payload_bundle(
                    payloads, Path(tmp), generation="nonfinite-generation"
                )

    def test_upload_plan_rechecks_semantics_before_pointer_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_atomic_payload_bundle(
                _complete_payloads(), output_dir, generation="semantic-failure"
            )
            pointer_path = output_dir / "meta.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            binding = pointer["files"]["round_matchups.json"]
            market_path = output_dir / binding["key"]
            market = json.loads(market_path.read_text(encoding="utf-8"))
            market["matchups"][0]["fair"] = {}
            _write_payload_files({"round_matchups.json": market}, market_path.parent)
            binding["size"] = market_path.stat().st_size
            binding["sha256"] = hashlib.sha256(market_path.read_bytes()).hexdigest()
            digest_input = b"".join(
                name.encode("utf-8")
                + b"\0"
                + (output_dir / file_binding["key"]).read_bytes()
                for name, file_binding in sorted(pointer["files"].items())
            )
            pointer["content_sha256"] = hashlib.sha256(digest_input).hexdigest()
            _write_payload_files({"meta.json": pointer}, output_dir)

            with self.assertRaisesRegex(OddsScreenContractError, "no committed model fair"):
                _atomic_upload_plan(output_dir)

    def test_each_outright_market_requires_a_quote(self):
        payloads = _complete_payloads()
        payloads["outrights.json"]["markets"]["top_20"][0]["books"] = {}
        with self.assertRaisesRegex(OddsScreenContractError, "top_20"):
            _validate_odds_screen_payloads(payloads)


class CommittedReleaseContractTests(unittest.TestCase):
    def test_producer_approved_h2h_rounding_overshoot_is_normalized(self):
        probabilities = _oriented_h2h_probability(
            {("a", "b"): (0.02813, 0.97188)}, "a", "b"
        )
        p1_push, p2_push, p1_raw, p2_raw, p_tie = probabilities
        self.assertAlmostEqual(p1_push + p2_push, 1.0)
        self.assertAlmostEqual(p1_raw + p2_raw + p_tie, 1.0)
        self.assertGreaterEqual(min(probabilities), 0.0)

    def test_stale_release_generation_is_rejected(self):
        stale = (datetime.now(timezone.utc) - timedelta(hours=19)).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root, release_generated_at=stale)
            with self.assertRaisesRegex(OddsScreenContractError, "stale"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_future_simulation_generation_is_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root, simulation_generated_at=future)
            with self.assertRaisesRegex(OddsScreenContractError, "future"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_stale_root_simulation_generation_is_rejected(self):
        stale = (datetime.now(timezone.utc) - timedelta(hours=19)).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root, root_generated_at=stale)
            with self.assertRaisesRegex(OddsScreenContractError, "root simulation is stale"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_generation_must_bind_event_round_and_manifest_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            manifest_path = root / "sim_release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generation"] = "self-consistent-but-wrong"
            manifest_path.write_text(
                json.dumps(seal_manifest(manifest)), encoding="utf-8"
            )
            with self.assertRaisesRegex(OddsScreenContractError, "does not bind"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_release_asset_set_is_required_even_when_manifest_is_resealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            manifest_path = root / "sim_release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["release_assets"] = {}
            manifest = seal_manifest(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(OddsScreenContractError, "asset set"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_self_consistent_wrong_event_threeball_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            contract_path = root / "round_3ball_r4_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["extra"]["event_id"] = "100"
            contract = seal_manifest(contract)
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            manifest_path = root / "sim_release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git_files"][contract_path.name] = {
                "sha256": file_sha256(contract_path),
                "size": contract_path.stat().st_size,
            }
            manifest_path.write_text(
                json.dumps(seal_manifest(manifest)), encoding="utf-8"
            )
            with self.assertRaisesRegex(OddsScreenContractError, "3-ball contract identity"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_clean_hosted_style_checkout_loads_only_committed_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Odds Screen Test",
                    "-c",
                    "user.email=odds-screen@example.invalid",
                    "commit",
                    "-qm",
                    "strict release",
                ],
                cwd=root,
                check=True,
            )

            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
            )
            self.assertEqual(release["source_git_sha"], subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip())

            with (root / "sim_fairs.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(OddsScreenContractError, "differs"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                )

    def test_loads_exact_manifest_bound_event_and_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                source_git_sha="d" * 40,
                verify_git=False,
            )
            self.assertEqual(release["manifest"]["manifest_sha256"], manifest["manifest_sha256"])
            self.assertEqual(len(release["round_h2h"]), 1)

    def test_changed_declared_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            (root / "round_h2h_r4_meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OddsScreenContractError, "changed after sealing"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=99,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_event_or_round_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            with self.assertRaisesRegex(OddsScreenContractError, "event_id"):
                _load_committed_release_contract(
                    expected_tourney="test_event",
                    expected_event_id=100,
                    expected_round=4,
                    project_root=root,
                    verify_git=False,
                )

    def test_round_builder_never_falls_back_to_books_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            offered = {
                "match_list": [
                    {
                        "p1_player_name": "a",
                        "p2_player_name": "b",
                        "ties": "void",
                        "odds": {"betcris": {"p1": -110, "p2": -110}},
                    }
                ]
            }
            with patch("push_odds_screen._fetch_scraped_guarded", return_value=offered):
                rows = _build_round_matchups(
                    "test_event", 4, {}, event_id=99, release=release
                )
            self.assertEqual(rows[0]["fair"]["p1_prob"], 0.61111)
            self.assertIn("betcris", rows[0]["books"])

            with patch("push_odds_screen._fetch_scraped_guarded", return_value=None):
                with self.assertRaisesRegex(OddsScreenContractError, "retaining prior"):
                    _build_round_matchups(
                        "test_event", 4, {}, event_id=99, release=release
                    )

    def test_round_builder_uses_tie_loss_probability_when_tie_is_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            offered = {
                "match_list": [
                    {
                        "p1_player_name": "a",
                        "p2_player_name": "b",
                        "ties": "separate bet offered",
                        "odds": {"betcris": {"p1": -110, "p2": -110}},
                    }
                ]
            }
            with patch("push_odds_screen._fetch_scraped_guarded", return_value=offered):
                rows = _build_round_matchups(
                    "test_event", 4, {}, event_id=99, release=release
                )
            self.assertEqual(rows[0]["fair"]["p1_prob"], 0.55)
            self.assertEqual(rows[0]["fair"]["p2_prob"], 0.35)
            self.assertEqual(rows[0]["fair"]["tie_prob"], 0.1)

    def test_invalid_book_quote_cannot_ride_beside_a_model_fair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            offered = {
                "match_list": [
                    {
                        "p1_player_name": "a",
                        "p2_player_name": "b",
                        "ties": "void",
                        "odds": {"broken": {"p1": 0, "p2": -110}},
                    }
                ]
            }
            with patch("push_odds_screen._fetch_scraped_guarded", return_value=offered):
                with self.assertRaisesRegex(OddsScreenContractError, "no usable book quote"):
                    _build_round_matchups(
                        "test_event", 4, {}, event_id=99, release=release
                    )

    def test_tournament_builder_uses_raw_wins_when_ties_lose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            release["fairs"]["matchups"] = [["a", "b", 2 / 3]]
            release["tournament_samples"] = pd.DataFrame(
                [[280, 281, 282, 283], [281, 281, 281, 284]],
                index=["a", "b"],
            )
            offered = {
                "match_list": [
                    {
                        "p1_player_name": "a",
                        "p2_player_name": "b",
                        "ties": "separate bet offered",
                        "odds": {"betcris": {"p1": -110, "p2": -110}},
                    }
                ]
            }
            with patch("push_odds_screen._fetch_scraped_guarded", return_value=offered):
                rows = _build_tournament_matchups(
                    "test_event", {}, event_id=99, release=release
                )
            self.assertEqual(rows[0]["fair"]["p1_prob"], 0.5)
            self.assertEqual(rows[0]["fair"]["p2_prob"], 0.25)
            self.assertEqual(rows[0]["fair"]["tie_prob"], 0.25)

            payloads = _complete_payloads()
            payloads["tournament_matchups.json"]["matchups"] = rows
            _validate_odds_screen_payloads(payloads)

    def test_score_builder_rejects_off_model_and_integer_offers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            quoted = {"betcris": {"over": -110, "under": -110}}
            off_model = {
                "lines": [{"player_name": "c", "line": 70.5, "odds": quoted}]
            }
            with patch(
                "push_odds_screen._fetch_scraped_guarded", return_value=off_model
            ):
                with self.assertRaisesRegex(OddsScreenContractError, "lack a sealed"):
                    _build_score_lines(
                        "test_event", 4, {}, event_id=99, release=release
                    )

            integer_line = {
                "lines": [{"player_name": "a", "line": 70, "odds": quoted}]
            }
            with patch(
                "push_odds_screen._fetch_scraped_guarded", return_value=integer_line
            ):
                with self.assertRaisesRegex(OddsScreenContractError, "half-stroke"):
                    _build_score_lines(
                        "test_event", 4, {}, event_id=99, release=release
                    )

    def test_outright_builder_rejects_attributable_off_model_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_strict_release(root)
            release = _load_committed_release_contract(
                expected_tourney="test_event",
                expected_event_id=99,
                expected_round=4,
                project_root=root,
                verify_git=False,
            )
            with patch(
                "push_odds_screen._fetch_dg_outrights",
                return_value={"c": {"betcris": 150}},
            ), patch("push_odds_screen._fetch_scraped_guarded", return_value=None):
                with self.assertRaisesRegex(OddsScreenContractError, "lack a sealed"):
                    _build_outrights(
                        "test_event", {}, event_id=99, release=release
                    )

            betcris = {
                "lines": [
                    {
                        "market_type": "winner",
                        "player": "c",
                        "odds": 150,
                    }
                ]
            }
            with patch(
                "push_odds_screen._fetch_dg_outrights", return_value={}
            ), patch(
                "push_odds_screen._fetch_scraped_guarded", return_value=betcris
            ):
                with self.assertRaisesRegex(OddsScreenContractError, "lack a sealed"):
                    _build_outrights(
                        "test_event", {}, event_id=99, release=release
                    )


class StrictQuoteProvenanceTests(unittest.TestCase):
    def test_datagolf_outrights_require_exact_response_event(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        payload = {
            "odds": [{"player_name": "a", "betcris": 2.5}],
        }
        with patch.dict("os.environ", {"DATAGOLF_API_KEY": "test"}), patch(
            "push_odds_screen.requests.get", return_value=Response(payload)
        ):
            self.assertEqual(_fetch_dg_outrights("win", {}, event_id=99), {})

        with patch.dict("os.environ", {"DATAGOLF_API_KEY": "test"}), patch(
            "push_odds_screen.requests.get",
            return_value=Response({**payload, "event_id": "99"}),
        ):
            self.assertEqual(
                _fetch_dg_outrights("win", {}, event_id=99),
                {"a": {"betcris": 150}},
            )

    def test_requires_parseable_timestamp_and_exact_event_evidence(self):
        fresh = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        base = {
            "last_updated": fresh,
            "round": 4,
            "match_list": [
                {
                    "event_id": "99",
                    "round": 4,
                    "p1_player_name": "a",
                    "p2_player_name": "b",
                }
            ],
        }
        with patch("odds_loader._fetch_scraped_json", return_value=base):
            result = _fetch_scraped_guarded("round_matchups", round=4, event_id=99)
        self.assertEqual(len(result["match_list"]), 1)

        with patch(
            "odds_loader._fetch_scraped_json",
            return_value={**base, "last_updated": "unknown"},
        ):
            self.assertIsNone(
                _fetch_scraped_guarded("round_matchups", round=4, event_id=99)
            )

        untagged = {
            **base,
            "match_list": [{key: value for key, value in base["match_list"][0].items()
                            if key != "event_id"}],
        }
        with patch("odds_loader._fetch_scraped_json", return_value=untagged):
            self.assertIsNone(
                _fetch_scraped_guarded("round_matchups", round=4, event_id=99)
            )

    def test_tournament_model_requires_every_canonical_pair(self):
        fairs = {
            "field": ["a", "b", "c"],
            "outrights": {
                market: {"a": 0.4, "b": 0.35, "c": 0.25}
                for market in ("winner", "top_5", "top_10", "top_20", "make_cut")
            },
            "outrights_nodh": {
                market: {"a": 0.4, "b": 0.35, "c": 0.25}
                for market in ("top_5", "top_10", "top_20")
            },
            "matchups": [["a", "b", 0.55]],
            "round_scores": {
                player: {"70": 1.0} for player in ("a", "b", "c")
            },
        }
        with self.assertRaisesRegex(OddsScreenContractError, "do not cover"):
            _validate_sim_fairs_semantics(fairs)


if __name__ == "__main__":
    unittest.main()

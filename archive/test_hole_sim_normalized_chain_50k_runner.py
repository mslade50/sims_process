"""Small, non-simulation tests for the gated 50k engineering runner."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_normalized_chain_50k_runner as runner


class FiftyKRunnerContracts(unittest.TestCase):
    def _gate(self) -> dict[str, object]:
        exact = lambda label, rows: {  # noqa: E731
            "label": label, "rows": rows, "mismatches": {"value": 0}
        }
        streams = {
            f"{family}:{name}": "1" * 64
            for family, name in runner.EXPECTED_STREAMS
        }
        return {
            "legacy_terminal": exact("terminal", 2304),
            "legacy_initializer_atom_ids": exact("atoms", 2304),
            "legacy_initializer_physical_state": exact("state", 2304),
            "counter_chunk_invariance": {
                arm: exact(f"chunk-{arm}", 9216)
                for arm in runner.EXPECTED_ARMS
            },
            "counter_50k_prefix_of_100k": {
                "exact": True,
                "transition_indices": list(
                    runner.EXPECTED_TRANSITION_INDICES
                ),
                "max_transition_index": (
                    runner.tail_policy.MAX_CONTINUATION_TRANSITIONS - 1
                ),
                "streams": streams,
            },
            "counter_support_inventory": {
                arm: {
                    "second_replay_new_calls": 0,
                    "second_replay_new_misses": 0,
                }
                for arm in runner.EXPECTED_ARMS
            },
            "unsupported_state_rows": 0,
            "survivor_filtering": False,
            "tail_policy": runner.tail_policy.manifest_policy(),
            "known_tail_regression": {
                "exact": True,
                "particle_ids": list(
                    sorted(runner.stream.KNOWN_TAIL_REGRESSION)
                ),
                "observed_hole_strokes": [
                    runner.stream.KNOWN_TAIL_REGRESSION[particle_id]
                    for particle_id in sorted(
                        runner.stream.KNOWN_TAIL_REGRESSION
                    )
                ],
                "singleton_chunk_exact": True,
                "legacy_score_cap_status_emitted": False,
                "tail_policy": runner.tail_policy.manifest_policy(),
            },
        }

    @staticmethod
    def _diagnostics(particles: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "arm": arm,
                    "round_no": round_no,
                    "hole_no": hole_no,
                    "particles": particles,
                    "stroke_sum": 4 * particles,
                    "stroke_sum_sq": 16 * particles,
                    "hazard_particles": 0,
                    "hazard_transitions": 0,
                    "penalty_strokes": 0,
                    "high_score_diagnostic_particles": 0,
                    "max_transition_count": 4,
                    "unsupported_state_rows": 0,
                    "mean_hole_strokes": 4.0,
                    "variance_hole_strokes": 0.0,
                    "hazard_hole_rate": 0.0,
                }
                for arm in runner.EXPECTED_ARMS
                for round_no in runner.EXPECTED_ROUNDS
                for hole_no in runner.EXPECTED_HOLES
            ]
        )

    def _write_support_cache(self, root: Path, arm: str) -> dict[str, object]:
        arm_root = root / "support_cache" / arm
        arm_root.mkdir(parents=True)
        support_path = arm_root / "support.npz"
        np.savez_compressed(
            support_path,
            atom_ids=np.asarray([0], dtype=np.int32),
            points_yd=np.zeros((1, 2), dtype=np.float32),
            probability=np.ones(1),
            cdf=np.ones(1),
            holeout_probability=np.zeros(1),
            source_cache_sha256=np.asarray(["2" * 64]),
            payload_sha256=np.asarray(["3" * 64]),
        )
        inventory = pd.DataFrame(
            [
                {
                    "key": f"fixture|{arm}",
                    "file": support_path.name,
                    "atoms": 1,
                    "source_cache_sha256": "2" * 64,
                    "payload_sha256": "3" * 64,
                    "file_sha256": runner.stream.sha256_file(support_path),
                    "file_bytes": support_path.stat().st_size,
                }
            ]
        )
        inventory_path = arm_root / "inventory.csv"
        inventory.to_csv(inventory_path, index=False)
        metadata = {
            "arm": arm,
            "supports": 1,
            "atoms": 1,
            "serialized_bytes": int(support_path.stat().st_size),
            "inventory_sha256": runner.stream.sha256_file(inventory_path),
            "logical_support_sha256": "4" * 64,
            "seconds": 0.01,
        }
        (arm_root / "manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata

    def _artifact_and_lock(self, parent: Path) -> tuple[Path, Path]:
        root = parent / "authoritative_5k"
        root.mkdir()
        dependency_hashes = {
            name: runner.stream.sha256_file(path)
            for name, path in runner.stream.dependency_paths().items()
        }
        streaming_hash = runner.stream.sha256_file(Path(runner.stream.__file__))
        cache = {
            arm: self._write_support_cache(root, arm)
            for arm in runner.EXPECTED_ARMS
        }
        gate = self._gate()
        hazard = {"scalar_batch_exact_rows": 56}
        benchmark = {
            "particles": 5_000,
            "chunk_size": 1_000,
            "arms_order": list(runner.EXPECTED_ARMS),
            "rng_mode": "counter",
            "tail_policy": runner.tail_policy.manifest_policy(),
            "timed_materialization_calls": {"raw": 0, "normalized": 0},
            "timed_support_misses": {"raw": 0, "normalized": 0},
            "arms": {
                arm: {"unsupported_state_rows": 0}
                for arm in runner.EXPECTED_ARMS
            },
            "preparation_peak_rss_bytes": 100,
            "prewarm_peak_rss_bytes": 100,
            "timed_kernel_peak_rss_bytes": 100,
            "global_peak_rss_bytes": 100,
            "safe_rss_cap_bytes": runner.RSS_CAP_BYTES,
            "numeric_support_cache": cache,
        }
        (root / "raw_128_gate.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )
        (root / "hazard_scalar_oracle.json").write_text(
            json.dumps(hazard, indent=2), encoding="utf-8"
        )
        (root / "benchmark.json").write_text(
            json.dumps(benchmark, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "case_id": np.repeat(np.arange(18), 128),
                "particle_id": np.tile(np.arange(128), 18),
            }
        ).to_parquet(
            root / "raw_128_parity_particles.parquet", index=False
        )
        totals = np.full((2, 4, 5_000), 70, dtype=np.int16)
        np.savez_compressed(
            root / "round_totals_int16.npz",
            arm_names=np.asarray(runner.EXPECTED_ARMS),
            round_totals=totals,
            round_numbers=np.asarray(runner.EXPECTED_ROUNDS, dtype=np.int8),
            particle_ids=np.arange(5_000, dtype=np.int64),
            player_id=np.asarray(["player"]),
            event_key=np.asarray(["event"]),
        )
        self._diagnostics(5_000).to_csv(root / "hole_diagnostics.csv", index=False)
        pd.DataFrame(
            [
                {"arm": arm, "round_no": round_no, "particles": 5_000}
                for arm in runner.EXPECTED_ARMS
                for round_no in runner.EXPECTED_ROUNDS
            ]
        ).to_csv(root / "round_summary.csv", index=False)
        (root / "REPORT.md").write_text("# fixture\n", encoding="utf-8")
        input_hashes = {"fixture": "5" * 64}
        input_logical_hashes = {"fixture_logical": "6" * 64}
        manifest = {
            "artifact": runner.stream.STREAMING_VERSION,
            "research_only": True,
            "production_touched": False,
            "event": "Travelers Championship",
            "event_key": "event",
            "player_id": "player",
            "production_mean_anchor_status": "not_calibrated_mechanics_only",
            "production_skill_embedded": False,
            "posthoc_recentering_applied": False,
            "tail_policy": runner.tail_policy.manifest_policy(),
            "reference_precompute_provenance": (
                runner.stream.reference_precompute_provenance(
                    dependency_hashes["precompute_source"]
                )
            ),
            "arms": list(runner.EXPECTED_ARMS),
            "normalization_mode": runner.evaluator.NORMALIZATION_MODE,
            "ability_log_weight_per_sg2": 0.0,
            "round_totals_sha256": runner.stream.sha256_file(
                root / "round_totals_int16.npz"
            ),
            "benchmark_rounds": list(runner.EXPECTED_ROUNDS),
            "holes_per_round": 18,
            "rng_modes": {
                "benchmark": "counter", "arm_in_key": False,
                "chunk_in_key": False,
                "counter_version": runner.stream.hook.RNG_VERSION,
            },
            "streaming_contract": {
                "survivor_filtering": False,
                "unsupported_states_fail_closed": True,
                "max_continuation_transitions": (
                    runner.tail_policy.MAX_CONTINUATION_TRANSITIONS
                ),
                "physical_hole_score_cap": None,
                "terminal_by_score_cap": False,
                "hole_score_clipping": False,
                "high_score_diagnostic_strokes": (
                    runner.tail_policy.HIGH_SCORE_DIAGNOSTIC_STROKES
                ),
                "high_score_diagnostic_only": True,
                "legacy_score_cap_status_emitted": False,
            },
            "streaming_source_start_sha256": streaming_hash,
            "streaming_source_end_sha256": streaming_hash,
            "dependency_start_hashes_sha256": dependency_hashes,
            "dependency_end_hashes_sha256": dependency_hashes,
            "input_hashes_sha256": input_hashes,
            "input_end_hashes_sha256": input_hashes,
            "input_logical_hashes_sha256": input_logical_hashes,
            "transform_order": list(runner.evaluator.REQUIRED_TRANSFORM_ORDER),
            "alignment_count": 1,
            "mass_contract": "base_hierarchy_mass_times_mean_one_model_weight",
            "raw_128_gate": gate,
            "hazard_scalar_oracle": hazard,
            "benchmark": benchmark,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        runner.stream.write_complete_output_hashes(root)
        lock = {
            "schema_version": runner.LOCK_SCHEMA_VERSION,
            "authoritative_5k_manifest_sha256": runner.stream.sha256_file(manifest_path),
            "authoritative_5k_output_index_sha256": runner.stream.sha256_file(
                root / "OUTPUT_SHA256.json"
            ),
            "runner_source_sha256": runner.stream.sha256_file(
                Path(runner.__file__)
            ),
            "evaluator_source_sha256": runner.stream.sha256_file(
                Path(runner.evaluator.__file__)
            ),
            "streaming_source_sha256": streaming_hash,
            "dependency_sha256": dependency_hashes,
            "tail_policy": runner.tail_policy.manifest_policy(),
        }
        lock_path = parent / "approval_lock.json"
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        return root, lock_path

    @staticmethod
    def _args(root: Path, lock: Path, output: Path, **overrides) -> argparse.Namespace:
        values = {
            "authoritative_5k_dir": root,
            "approval_lock": lock,
            "output_dir": output,
            "particles": runner.PARTICLES,
            "chunk_size": runner.DEFAULT_CHUNK_SIZE,
            "dry_run": True,
            "export_root": runner.stream.precompute.DEFAULT_EXPORT,
            "history_db": runner.stream.precompute.asof.DEFAULT_HISTORY_DB,
            "profile_dir": runner.stream.precompute.DEFAULT_PROFILES,
            "baseline_cases": runner.stream.precompute.DEFAULT_BASELINES,
            "reference_dir": runner.stream.DEFAULT_REFERENCE,
            "event_name": runner.stream.precompute.DEFAULT_EVENT,
            "holdout_year": runner.stream.precompute.DEFAULT_HOLDOUT,
            "course_start_year": 2020,
            "roster_seed": runner.stream.DEFAULT_SEED,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _write_emitted_50k_fixture(
        self, parent: Path, *, mutate_evaluator_at_finalization: bool = False
    ) -> tuple[Path, dict[str, object]]:
        fivek_root, lock_path = self._artifact_and_lock(parent)
        approval = runner.load_approval_lock(lock_path)
        admission = runner.validate_authoritative_5k(fivek_root, approval)
        prepared = SimpleNamespace(
            atoms=SimpleNamespace(
                event_name="Travelers Championship",
                target_event_key="event",
                holdout_year=2026,
            ),
            grid=pd.DataFrame([{"source_player_id": "player"}]),
            input_hashes=admission["input_hashes_sha256"],
            logical_hashes=admission["input_logical_hashes_sha256"],
        )
        totals = {
            arm: np.full((4, runner.PARTICLES), 70, dtype=np.int16)
            for arm in runner.EXPECTED_ARMS
        }
        gate = self._gate()
        prefix = {
            "exact": True,
            "narrow_particles": runner.PARTICLES,
            "wide_particles": runner.PREFIX_WIDE_PARTICLES,
            "transition_indices": list(runner.EXPECTED_TRANSITION_INDICES),
            "max_transition_index": (
                runner.tail_policy.MAX_CONTINUATION_TRANSITIONS - 1
            ),
            "streams": dict(gate["counter_50k_prefix_of_100k"]["streams"]),
        }
        runner.validate_independent_prefix_matches_gate(prefix, gate)
        output = parent / "emitted_50k"
        args = self._args(
            fivek_root, lock_path, output, dry_run=False, chunk_size=128
        )
        source_hash = runner.stream.sha256_file(Path(runner.__file__))
        dependencies = runner.validate_current_frozen_graph(approval)
        def emit() -> None:
            runner.write_outputs(
                output,
                args=args,
                approval=approval,
                admission=admission,
                prepared=prepared,
                gate=gate,
                hazard_oracle={"scalar_batch_exact_rows": 56},
                prefix=prefix,
                parity_rows=pd.read_parquet(
                    fivek_root / "raw_128_parity_particles.parquet"
                ),
                totals=totals,
                diagnostics=self._diagnostics(runner.PARTICLES),
                execution={
                    "production_mean_anchor_status": (
                        runner.evaluator.MECHANICS_ANCHOR_STATUS
                    ),
                    "predictive_scoring_permitted": False,
                },
                source_start_sha256=source_hash,
                streaming_source_start_sha256=approval.streaming_source_sha256,
                dependency_start_sha256=dependencies,
                input_end_sha256=prepared.input_hashes,
                monitor_snapshot_before_final_hash={
                    "lifetime_peak_rss_bytes": 100,
                    "cap_exceeded": False,
                    "phases": {},
                },
            )

        if mutate_evaluator_at_finalization:
            original_sha = runner.stream.sha256_file
            evaluator_path = Path(runner.evaluator.__file__).resolve()

            def changed_sha(path: Path) -> str:
                if Path(path).resolve() == evaluator_path:
                    return "f" * 64
                return original_sha(path)

            with mock.patch.object(
                runner.stream, "sha256_file", side_effect=changed_sha
            ):
                emit()
        else:
            emit()
        schema = runner.validate_output_evaluator_compatibility(output)
        return output, schema

    def test_cli_contract_accepts_only_50k_and_bounded_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock = self._artifact_and_lock(Path(temporary))
            args = self._args(root, lock, Path(temporary) / "output")
            runner.validate_cli_contract(args)
            for particles in (128, 5_000, 49_999, 50_001, 100_000):
                with self.subTest(particles=particles):
                    with self.assertRaisesRegex(ValueError, "exactly 50,000"):
                        runner.validate_cli_contract(
                            self._args(root, lock, Path(temporary) / "out", particles=particles)
                        )
            for chunk in (1, runner.MIN_CHUNK_SIZE - 1, runner.MAX_CHUNK_SIZE + 1):
                with self.subTest(chunk=chunk):
                    with self.assertRaisesRegex(ValueError, "chunk_size"):
                        runner.validate_cli_contract(
                            self._args(root, lock, Path(temporary) / "out", chunk_size=chunk)
                        )

    def test_complete_5k_fixture_passes_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock_path = self._artifact_and_lock(Path(temporary))
            lock = runner.load_approval_lock(lock_path)
            result = runner.validate_authoritative_5k(root, lock)
            self.assertEqual(result["event_key"], "event")
            self.assertGreaterEqual(result["files_verified"], 15)

    def test_checksum_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock_path = self._artifact_and_lock(Path(temporary))
            lock = runner.load_approval_lock(lock_path)
            (root / "REPORT.md").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "(hash|byte-size) mismatch"):
                runner.validate_authoritative_5k(root, lock)

    def test_approval_lock_requires_exact_dependency_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            payload["dependency_sha256"].pop(next(iter(payload["dependency_sha256"])))
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dependency names"):
                runner.load_approval_lock(lock_path)

    def test_approval_lock_binds_current_precompute_not_legacy_producer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            approval = runner.load_approval_lock(lock_path)
            current = runner.stream.sha256_file(
                Path(runner.stream.precompute.__file__)
            )
            self.assertEqual(
                approval.dependency_sha256["precompute_source"], current
            )
            self.assertNotEqual(
                current,
                runner.stream.FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256,
            )

    def test_approval_lock_requires_exact_tail_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            payload["tail_policy"]["physical_hole_score_cap"] = 15
            lock_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tail_policy"):
                runner.load_approval_lock(lock_path)

    def test_approval_lock_requires_executable_sources_in_closed_schema(self) -> None:
        for missing in ("runner_source_sha256", "evaluator_source_sha256"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                _, lock_path = self._artifact_and_lock(Path(temporary))
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                payload.pop(missing)
                lock_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "closed approval-lock schema"):
                    runner.load_approval_lock(lock_path)

    def test_wrong_approved_runner_source_fails_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            approval = runner.load_approval_lock(lock_path)
            wrong = replace(approval, runner_source_sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "runner source"):
                runner.validate_current_frozen_graph(wrong)

    def test_mutated_evaluator_source_fails_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            approval = runner.load_approval_lock(lock_path)
            mutated = replace(approval, evaluator_source_sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "evaluator source"):
                runner.validate_current_frozen_graph(mutated)

    def test_evaluator_source_mutation_fails_at_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(AssertionError, "Evaluator source"):
                self._write_emitted_50k_fixture(
                    Path(temporary), mutate_evaluator_at_finalization=True
                )

    def test_approval_lock_mutation_is_detected_at_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, lock_path = self._artifact_and_lock(Path(temporary))
            approval = runner.load_approval_lock(lock_path)
            lock_path.write_text(
                lock_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                runner.validate_approval_lock_unchanged(approval)

    def test_5k_producer_contract_fields_fail_closed_individually(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._artifact_and_lock(Path(temporary))
            valid = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            runner._validate_producer_contract(valid)
            mutations = {
                "normalization": lambda value: value.__setitem__(
                    "normalization_mode", "wrong"
                ),
                "beta": lambda value: value.__setitem__(
                    "ability_log_weight_per_sg2", 0.01
                ),
                "streaming": lambda value: value["streaming_contract"].__setitem__(
                    "survivor_filtering", True
                ),
                "rng": lambda value: value["rng_modes"].__setitem__(
                    "benchmark", "legacy128"
                ),
                "transform": lambda value: value.__setitem__(
                    "transform_order", list(reversed(value["transform_order"]))
                ),
                "alignment": lambda value: value.__setitem__(
                    "alignment_count", 2
                ),
                "mass": lambda value: value.__setitem__(
                    "mass_contract", "wrong"
                ),
                "tail_policy": lambda value: value["tail_policy"].__setitem__(
                    "physical_hole_score_cap", 15
                ),
                "tail_streaming": lambda value: value[
                    "streaming_contract"
                ].__setitem__("hole_score_clipping", True),
                "reference_provenance": lambda value: value[
                    "reference_precompute_provenance"
                ].__setitem__("legacy_reference_tail_policy_attested", True),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = copy.deepcopy(valid)
                    mutate(candidate)
                    with self.assertRaises(ValueError):
                        runner._validate_producer_contract(candidate)

    def test_prepared_logical_hashes_are_bound_to_5k_admission(self) -> None:
        prepared = SimpleNamespace(
            atoms=SimpleNamespace(target_event_key="event"),
            grid=pd.DataFrame([{"source_player_id": "player"}]),
            input_hashes={"input": "a" * 64},
            logical_hashes={"logical": "b" * 64},
        )
        admission = {
            "event_key": "event",
            "player_id": "player",
            "input_hashes_sha256": {"input": "a" * 64},
            "input_logical_hashes_sha256": {"logical": "b" * 64},
        }
        runner.validate_prepared_against_admission(prepared, admission)
        prepared.logical_hashes = {"logical": "c" * 64}
        with self.assertRaisesRegex(ValueError, "logical simulation inputs"):
            runner.validate_prepared_against_admission(prepared, admission)

    def test_5k_npz_identity_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = self._artifact_and_lock(Path(temporary))
            cases = (
                ("event", "different-player", "player_id disagrees"),
                ("different-event", "player", "event_key disagrees"),
            )
            for event_key, player_id, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        runner._validate_5k_totals(
                            root / "round_totals_int16.npz",
                            expected_event_key=event_key,
                            expected_player_id=player_id,
                        )

    def test_round_summary_requires_exactly_eight_unique_rows(self) -> None:
        valid = pd.DataFrame(
            [
                {"arm": arm, "round_no": round_no, "particles": 5_000}
                for arm in runner.EXPECTED_ARMS
                for round_no in runner.EXPECTED_ROUNDS
            ]
        )
        runner._validate_round_summary(valid, particles=5_000, label="fixture")
        duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            runner._validate_round_summary(
                duplicate, particles=5_000, label="fixture"
            )

    def test_dry_run_validates_but_never_prepares_or_simulates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock = self._artifact_and_lock(Path(temporary))
            args = self._args(root, lock, Path(temporary) / "output")
            with mock.patch.object(
                runner.stream, "prepare_streaming_event",
                side_effect=AssertionError("dry-run reached preparation"),
            ):
                result = runner.run_50k(args)
            self.assertTrue(result["admission_valid"])
            self.assertFalse(result["simulation_launched"])
            self.assertEqual(result["particles"], 50_000)

    def test_cli_dry_run_emits_validated_schema_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock = self._artifact_and_lock(Path(temporary))
            output = Path(temporary) / "output"
            stdout = io.StringIO()
            with mock.patch.object(
                runner.stream, "prepare_streaming_event",
                side_effect=AssertionError("CLI dry-run reached preparation"),
            ), redirect_stdout(stdout):
                runner.main(
                    [
                        "--authoritative-5k-dir", str(root),
                        "--approval-lock", str(lock),
                        "--output-dir", str(output),
                        "--particles", "50000",
                        "--chunk-size", "128",
                        "--dry-run",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["admission_valid"])
            self.assertFalse(payload["simulation_launched"])
            self.assertEqual(payload["chunk_size"], 128)

    def test_prefix_attestation_covers_all_semantic_streams(self) -> None:
        record = SimpleNamespace(
            event_key="event", source_player_id="player",
            round_no=1, hole_no=1,
        )
        result = runner.attest_counter_prefix(record)
        self.assertTrue(result["exact"])
        self.assertEqual(
            set(result["streams"]),
            {f"{family}:{name}" for family, name in runner.EXPECTED_STREAMS},
        )
        self.assertEqual(
            result["max_transition_index"],
            runner.tail_policy.MAX_CONTINUATION_TRANSITIONS - 1,
        )
        self.assertEqual(
            tuple(result["transition_indices"]),
            runner.EXPECTED_TRANSITION_INDICES,
        )

    def test_independent_prefix_hashes_must_equal_fresh_gate_hashes(self) -> None:
        gate = self._gate()
        independent = {
            "exact": True,
            "transition_indices": list(runner.EXPECTED_TRANSITION_INDICES),
            "max_transition_index": (
                runner.tail_policy.MAX_CONTINUATION_TRANSITIONS - 1
            ),
            "streams": dict(gate["counter_50k_prefix_of_100k"]["streams"]),
        }
        runner.validate_independent_prefix_matches_gate(independent, gate)
        first = next(iter(independent["streams"]))
        independent["streams"][first] = "f" * 64
        with self.assertRaisesRegex(ValueError, "differ from fresh parity-gate"):
            runner.validate_independent_prefix_matches_gate(independent, gate)

    def test_emitted_50k_fixture_passes_current_evaluator_as_mechanics_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, schema = self._write_emitted_50k_fixture(Path(temporary))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(schema["schema_valid"])
            self.assertEqual(schema["particles_per_player_round"], 50_000)
            self.assertFalse(schema["predictive_scoring_permitted"])
            self.assertEqual(
                manifest["normalization_mode"],
                runner.evaluator.NORMALIZATION_MODE,
            )
            self.assertEqual(manifest["ability_log_weight_per_sg2"], 0.0)
            self.assertEqual(
                tuple(manifest["transform_order"]),
                runner.evaluator.REQUIRED_TRANSFORM_ORDER,
            )
            self.assertEqual(manifest["alignment_count"], 1)
            self.assertFalse(manifest["checkpoint_resume_supported"])
            self.assertEqual(
                manifest["input_hashes_sha256"],
                manifest["authoritative_5k_admission"]["input_hashes_sha256"],
            )

    def test_simulation_contract_rejects_incomplete_axis(self) -> None:
        totals = {
            arm: np.full((4, runner.PARTICLES), 70, dtype=np.int16)
            for arm in runner.EXPECTED_ARMS
        }
        details = {
            "arms": {
                arm: {"unsupported_state_rows": 0}
                for arm in runner.EXPECTED_ARMS
            }
        }
        frame = self._diagnostics(runner.PARTICLES).iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "4x18 axes"):
            runner.validate_simulation_result(totals, details, frame)

    def test_round_total_persistence_never_wraps_or_clips(self) -> None:
        prepared = SimpleNamespace(
            grid=pd.DataFrame([{"source_player_id": "player"}]),
            atoms=SimpleNamespace(target_event_key="event"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            overflow = {
                arm: np.full(
                    (4, runner.PARTICLES), 40_000, dtype=np.int64
                )
                for arm in runner.EXPECTED_ARMS
            }
            with self.assertRaises(OverflowError):
                runner._write_round_totals(
                    Path(temporary), overflow, prepared
                )
            outside_lattice = {
                arm: np.full(
                    (4, runner.PARTICLES), 141, dtype=np.int16
                )
                for arm in runner.EXPECTED_ARMS
            }
            with self.assertRaisesRegex(ValueError, "physical round lattice"):
                runner._write_round_totals(
                    Path(temporary), outside_lattice, prepared
                )

    def test_lifetime_monitor_records_distinct_phases(self) -> None:
        monitor = runner.LifetimeRssMonitor(interval_seconds=0.005)
        monitor.start()
        try:
            with monitor.phase("setup"):
                np.arange(128, dtype=np.int64).sum()
            with monitor.phase("timed"):
                np.arange(128, dtype=np.int64).sum()
        finally:
            monitor.stop()
        snapshot = monitor.snapshot()
        self.assertFalse(snapshot["cap_exceeded"])
        self.assertEqual(set(snapshot["phases"]), {"setup", "timed"})
        self.assertGreater(snapshot["lifetime_peak_rss_bytes"], 0)
        for phase in snapshot["phases"].values():
            self.assertIsNotNone(phase["elapsed_seconds"])

    def test_rss_seal_is_in_final_checksum_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "payload.txt").write_text("payload", encoding="utf-8")
            runner.stream.write_complete_output_hashes(root)
            snapshot = {
                "lifetime_peak_rss_bytes": 123,
                "cap_exceeded": False,
                "phases": {"finalization": {"peak_rss_bytes": 123}},
            }
            runner.seal_finalization_rss(root, snapshot)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rss_monitor_sealed_snapshot"], snapshot)
            index = json.loads((root / "OUTPUT_SHA256.json").read_text(encoding="utf-8"))
            self.assertIn("RSS_AUDIT.json", index)
            self.assertEqual(
                index["RSS_AUDIT.json"]["sha256"],
                runner.stream.sha256_file(root / "RSS_AUDIT.json"),
            )

    def test_postseal_evaluator_audit_binds_stable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"stable": True}, indent=2), encoding="utf-8"
            )
            (root / "payload.txt").write_text("payload", encoding="utf-8")
            runner.stream.write_complete_output_hashes(root)

            def final_manifest_schema(output_dir: Path) -> dict[str, object]:
                final_hash = runner.stream.sha256_file(
                    Path(output_dir) / "manifest.json"
                )
                return {
                    "schema_valid": True,
                    "bundle_audits": [{"manifest_sha256": final_hash}],
                    "input_hashes_sha256": {
                        str((Path(output_dir) / "manifest.json").resolve()): final_hash
                    },
                }

            with mock.patch.object(
                runner,
                "validate_output_evaluator_compatibility",
                side_effect=final_manifest_schema,
            ):
                final_hash = runner.stream.sha256_file(manifest_path)
                schema = runner.write_postseal_evaluator_audit(root)

            self.assertTrue(schema["schema_valid"])
            self.assertEqual(runner.stream.sha256_file(manifest_path), final_hash)
            audit = json.loads(
                (root / "evaluator_schema_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                audit["bundle_audits"][0]["manifest_sha256"], final_hash
            )
            index = json.loads(
                (root / "OUTPUT_SHA256.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                index["evaluator_schema_audit.json"]["sha256"],
                runner.stream.sha256_file(root / "evaluator_schema_audit.json"),
            )

    def test_output_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock = self._artifact_and_lock(Path(temporary))
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "existing.txt").write_text("no overwrite", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                runner.validate_cli_contract(self._args(root, lock, output))

    def test_output_cannot_mutate_authoritative_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, lock = self._artifact_and_lock(Path(temporary))
            with self.assertRaisesRegex(ValueError, "authoritative 5k"):
                runner.validate_cli_contract(
                    self._args(root, lock, root / "new_50k_output")
                )


if __name__ == "__main__":
    unittest.main()

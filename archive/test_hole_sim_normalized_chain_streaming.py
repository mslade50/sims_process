"""Focused contracts for the archive-only streaming shot chain."""

from __future__ import annotations

import unittest
import zlib
from pathlib import Path
import sys
import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_full_chain as full
import hole_sim_batched_population_transition as batched
import hole_sim_hazard_resolution as hazard
import hole_sim_normalized_atom_precompute as precompute
import hole_sim_normalized_chain_streaming as stream
import hole_sim_tail_policy as tail_policy


class StreamingChainContracts(unittest.TestCase):
    @staticmethod
    def _record(*, hole_no: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            case_id=hole_no - 1,
            event_key="event",
            source_player_id="player",
            round_no=1,
            hole_no=hole_no,
            par=4,
            forecast_tee_x=0.0,
            forecast_tee_y=0.0,
            forecast_pin_x=1200.0,
            forecast_pin_y=0.0,
            forecast_pin_long_yd=400.0,
        )

    @staticmethod
    def _green_engine(
        *, start_strokes: int, transition_cost: int, holed: bool
    ) -> stream.StreamingShotChain:
        class Sampler:
            def sample_many(self, **kwargs):
                size = len(kwargs["lie"])
                return batched.BatchTransitionResult(
                    holed=np.full(size, holed, dtype=bool),
                    next_lie=np.full(size, "hole" if holed else "green"),
                    next_distance=np.zeros(size) if holed else np.full(size, 5.0),
                    cost=np.full(size, transition_cost, dtype=np.int16),
                    level=np.full(size, "fixture"),
                    available=np.ones(size, dtype=bool),
                    error=np.full(size, ""),
                )

        engine = stream.StreamingShotChain.__new__(stream.StreamingShotChain)
        engine.repositories = {"raw": object()}
        engine.state_sampler = Sampler()
        engine.max_transitions = tail_policy.MAX_CONTINUATION_TRANSITIONS
        engine.max_hole_strokes = None

        def initialize_green(**kwargs):
            state = kwargs["state"]
            positions = np.flatnonzero(kwargs["active"])
            state.endpoint[positions] = np.asarray([395.0, 0.0])
            state.lie[positions] = stream.LIE_GREEN
            state.remaining_yd[positions] = 5.0
            state.strokes[positions] = start_strokes
            state.status[positions] = stream.STATUS_ACTIVE

        engine._sample_tee = initialize_green
        engine._sample_app = lambda **kwargs: None
        engine._resolve_hazards = lambda **kwargs: None
        return engine

    @staticmethod
    def _hazard_donors(count: int) -> pd.DataFrame:
        rows = []
        for index in range(count):
            endpoint_x = float(index)
            resolution_x = float(3 + index)
            rows.append(
                {
                    "donor_key": f"d{index:04d}",
                    "event_key": "prior",
                    "season": 2024,
                    "round_no": 1,
                    "hole_no": 1,
                    "category": "ott",
                    "causal_endpoint_x": endpoint_x,
                    "causal_endpoint_y": 0.0,
                    "penalty_count": 1,
                    "penalty_stroke_cost": 1,
                    "resolution_shot_key": f"r{index:04d}",
                    "resolution_event_type": "STROKE",
                    "resolution_x": resolution_x,
                    "resolution_y": 0.0,
                    "resolution_lie": "rough",
                    "resolution_lie_raw": "rough",
                    "donor_geometry_resolution_distance_yd": resolution_x / 3.0,
                    "donor_sg_resolution_distance_yd": resolution_x / 3.0,
                    "donor_sg_resolution_lie": "rough",
                    "donor_pin_x": 0.0,
                    "donor_pin_y": 0.0,
                    "physical_course_id": "course",
                    "foundation_version": hazard.FOUNDATION_VERSION,
                    "coordinate_reference_year": 2025,
                    "coordinate_reference_event_key": "prior",
                    "coordinate_reference_mode": "latest_prior",
                    "data_role": "predictor_donor",
                }
            )
        return pd.DataFrame(rows)

    def test_numeric_support_inverse_cdf_uses_right_boundary(self) -> None:
        support = stream.NumericSupport(
            key="fixture",
            atom_ids=np.asarray(["a", "b", "c"]),
            points=np.zeros((3, 2)),
            probability=np.asarray([0.2, 0.3, 0.5]),
            cdf=np.asarray([0.2, 0.5, 1.0]),
            holeout_probability=0.0,
            cache_sha256="0" * 64,
        )
        index, atom = support.select(np.asarray([0.0, 0.2, 0.499, 0.5, 0.999]))
        np.testing.assert_array_equal(index, [0, 1, 1, 2, 2])
        np.testing.assert_array_equal(atom, ["a", "b", "b", "c", "c"])

    def test_counter_uniforms_are_chunk_and_50k_prefix_invariant(self) -> None:
        common = dict(
            mode="counter",
            seed=stream.DEFAULT_SEED,
            continuation_seed=stream.DEFAULT_CONTINUATION_SEED,
            event_key="pga:R2026034",
            player_id="33204",
            round_no=1,
            hole_no=1,
        )
        narrow = stream.UniformFactory(total_particles=50_000, **common)
        wide = stream.UniformFactory(total_particles=100_000, **common)
        ids = np.arange(50_000, dtype=np.int64)
        for transition_index in range(
            tail_policy.MAX_CONTINUATION_TRANSITIONS
        ):
            expected = narrow.transition(
                transition_index, "app_index", ids
            )
            np.testing.assert_array_equal(
                expected,
                wide.transition(
                    transition_index, "app_index", np.arange(100_000)
                )[:50_000],
            )
            chunks = np.concatenate(
                [
                    narrow.transition(
                        transition_index,
                        "app_index",
                        ids[start : start + 997],
                    )
                    for start in range(0, len(ids), 997)
                ]
            )
            np.testing.assert_array_equal(expected, chunks)

    def test_green_at_twelve_can_hole_in_thirteen_or_fourteen(self) -> None:
        for cost, expected in ((1, 13), (2, 14)):
            with self.subTest(cost=cost):
                engine = self._green_engine(
                    start_strokes=12, transition_cost=cost, holed=True
                )
                result, _ = engine.simulate_hole(
                    record=self._record(),
                    arm="raw",
                    particle_ids=np.asarray([0], dtype=np.int64),
                    total_particles=1,
                    rng_mode="counter",
                )
                self.assertTrue(bool(result.terminal[0]))
                self.assertEqual(int(result.hole_strokes[0]), expected)
                self.assertEqual(result.hole_strokes.dtype, np.int16)
                self.assertEqual(
                    str(stream.STATUS_NAMES[result.status_code[0]]), "terminal"
                )

    def test_never_holing_fails_at_exactly_eighteen_transitions(self) -> None:
        engine = self._green_engine(
            start_strokes=0, transition_cost=1, holed=False
        )
        result, _ = engine.simulate_hole(
            record=self._record(),
            arm="raw",
            particle_ids=np.asarray([0], dtype=np.int64),
            total_particles=1,
            rng_mode="counter",
        )
        self.assertFalse(bool(result.terminal[0]))
        self.assertEqual(
            int(result.transition_count[0]),
            tail_policy.MAX_CONTINUATION_TRANSITIONS,
        )
        self.assertEqual(
            str(stream.STATUS_NAMES[result.status_code[0]]),
            tail_policy.TRANSITION_LOOP_STATUS,
        )
        self.assertNotEqual(
            str(stream.STATUS_NAMES[result.status_code[0]]),
            tail_policy.LEGACY_SCORE_CAP_STATUS,
        )
        grid = pd.DataFrame([vars(self._record())])
        with self.assertRaisesRegex(
            full.ChainIncomplete, "Survivor filtering forbidden"
        ):
            stream._simulate_grid(
                engine,
                grid,
                arms=("raw",),
                particles=1,
                chunk_size=1,
                rng_mode="counter",
                collect_holes=False,
            )

    def test_spatial_and_collapsed_penalties_preserve_exact_strokes(self) -> None:
        self.assertEqual(full._charge_spatial_hazard(12, 2), 14)
        engine = self._green_engine(
            start_strokes=12, transition_cost=3, holed=True
        )
        state = stream.HoleState.empty(1)
        state.endpoint[0] = [395.0, 0.0]
        state.lie[0] = stream.LIE_GREEN
        state.remaining_yd[0] = 5.0
        state.strokes[0] = 12
        engine._population_transition(
            state=state,
            active=np.asarray([True]),
            category="putt",
            record=self._record(),
            common_uniform=np.asarray([0.5]),
            diagnostics=stream.KernelDiagnostics(),
        )
        self.assertTrue(bool(state.terminal[0]))
        self.assertEqual(int(state.strokes[0]), 15)
        self.assertEqual(int(state.penalty_strokes[0]), 2)
        self.assertEqual(int(state.hazard_count[0]), 1)

    def test_score_is_never_clipped_and_int16_overflow_fails_closed(self) -> None:
        engine = self._green_engine(
            start_strokes=12, transition_cost=20, holed=True
        )
        result, _ = engine.simulate_hole(
            record=self._record(), arm="raw",
            particle_ids=np.asarray([0]), total_particles=1,
            rng_mode="counter",
        )
        self.assertEqual(int(result.hole_strokes[0]), 32)
        with self.assertRaises(OverflowError):
            tail_policy.int16_exact(
                np.asarray([40_000], dtype=np.int64), label="overflow fixture"
            )
        with self.assertRaisesRegex(ValueError, "physical round lattice"):
            tail_policy.round_totals_int16(
                np.asarray([39, 141], dtype=np.int64),
                label="lattice fixture",
            )

    def test_legacy_initializer_stream_order_matches_reference_rng(self) -> None:
        factory = stream.UniformFactory(
            mode="legacy128",
            seed=stream.DEFAULT_SEED,
            continuation_seed=stream.DEFAULT_CONTINUATION_SEED,
            event_key="pga:R2026034",
            player_id="33204",
            round_no=1,
            hole_no=1,
            total_particles=128,
        )
        seed = np.random.SeedSequence(
            [stream.DEFAULT_SEED, zlib.crc32(b"33204|1|1")]
        )
        rng = np.random.default_rng(seed)
        expected = {
            name: rng.random(128)
            for name in (
                "tee_index", "tee_lie", "app_index", "app_holeout", "app_lie"
            )
        }
        ids = np.arange(128)
        for name, values in expected.items():
            np.testing.assert_array_equal(factory.initial(name, ids), values)

    def test_state_routing_matches_full_chain_scalar(self) -> None:
        lies = np.asarray(
            [
                stream.LIE_HOLE,
                stream.LIE_GREEN,
                stream.LIE_TEE,
                stream.LIE_FAIRWAY,
                stream.LIE_ROUGH,
                stream.LIE_PENALTY,
                stream.LIE_UNKNOWN,
            ],
            dtype=np.int8,
        )
        distance = np.asarray([0.0, 10.0, 400.0, 49.9, 80.0, 30.0, np.nan])
        categories = stream._state_categories(
            lies, distance, np.ones(len(lies), dtype=bool), par=4
        )
        expected_names = [
            full._state_category(str(stream.LIE_NAMES[lie]), dist, 4)
            for lie, dist in zip(lies, distance)
        ]
        mapping = {
            "terminal": stream.CATEGORY_TERMINAL,
            "putt": stream.CATEGORY_PUTT,
            "ott": stream.CATEGORY_OTT,
            "app": stream.CATEGORY_APP,
            "arg": stream.CATEGORY_ARG,
            "invalid": stream.CATEGORY_INVALID,
        }
        np.testing.assert_array_equal(
            categories, np.asarray([mapping[name] for name in expected_names])
        )

    def test_legacy_initializer_hazard_other_maps_to_penalty_sentinel(self) -> None:
        frame = pd.DataFrame(
            {
                "case_id": [0],
                "particle_id": [0],
                "tee_end_long_yd": [300.0],
                "tee_end_lat_yd": [12.0],
                "app_end_long_yd": [np.nan],
                "app_end_lat_yd": [np.nan],
                "tee_end_lie": ["other"],
                "app_end_lie": ["not_attempted"],
                "tee_remaining_yd": [145.0],
                "app_remaining_yd": [np.nan],
                "app_geometric_remaining_yd": [np.nan],
                "shots_taken": [1],
                "chain_status": ["tee_hazard_unresolved"],
            }
        )
        projected = stream._legacy_initializer_state(frame)
        self.assertEqual(projected.loc[0, "initial_end_lie"], "penalty")

    def test_legacy_initializer_ordinary_app_other_stays_recovery(self) -> None:
        frame = pd.DataFrame(
            {
                "case_id": [0],
                "particle_id": [0],
                "tee_end_long_yd": [280.0],
                "tee_end_lat_yd": [4.0],
                "app_end_long_yd": [430.0],
                "app_end_lat_yd": [2.0],
                "tee_end_lie": ["fairway"],
                "app_end_lie": ["other"],
                "tee_remaining_yd": [160.0],
                "app_remaining_yd": [35.0],
                "app_geometric_remaining_yd": [35.0],
                "shots_taken": [2],
                "chain_status": ["app_ready_arg"],
            }
        )
        projected = stream._legacy_initializer_state(frame)
        self.assertEqual(projected.loc[0, "initial_end_lie"], "recovery")

    def test_batched_hazard_matches_min_support_and_right_cdf(self) -> None:
        sparse = hazard.EndpointLinkedHazardResolver(
            self._hazard_donors(3), minimum_category_support=12,
            same_hole_fallback_max_distance_yd=150.0,
        )
        batch_sparse = stream.BatchedEndpointHazardResolver(sparse)
        with self.assertRaises(hazard.HazardResolutionUnavailable):
            sparse.resolve(
                endpoint_x=0.0, endpoint_y=0.0, round_no=1, hole_no=1,
                category="ott", common_uniform=0.5,
                forecast_pin_x=0.0, forecast_pin_y=0.0,
            )
        sparse_result = batch_sparse.resolve_many(
            endpoint_x=np.asarray([0.0]), endpoint_y=np.asarray([0.0]),
            round_no=1, hole_no=1, category="ott",
            common_uniform=np.asarray([0.5]),
            forecast_pin_x=0.0, forecast_pin_y=0.0,
        )
        self.assertFalse(bool(sparse_result.available[0]))

        scalar = hazard.EndpointLinkedHazardResolver(
            self._hazard_donors(17), minimum_category_support=12,
            same_hole_fallback_max_distance_yd=150.0,
        )
        batch = stream.BatchedEndpointHazardResolver(scalar)
        candidates, weights, *_ = scalar._distribution(0.0, 0.0, 1, 1, "ott")
        order = np.argsort(candidates["donor_key"].astype(str).to_numpy())
        boundary = float(np.cumsum(weights[order])[0])
        uniforms = np.asarray([0.0, boundary, np.nextafter(1.0, 0.0)])
        result = batch.resolve_many(
            endpoint_x=np.zeros(3), endpoint_y=np.zeros(3), round_no=1,
            hole_no=1, category="ott", common_uniform=uniforms,
            forecast_pin_x=0.0, forecast_pin_y=0.0,
        )
        expected = [
            scalar.resolve(
                endpoint_x=0.0, endpoint_y=0.0, round_no=1, hole_no=1,
                category="ott", common_uniform=float(value),
                forecast_pin_x=0.0, forecast_pin_y=0.0,
            ).donor_key
            for value in uniforms
        ]
        np.testing.assert_array_equal(result.donor_key, expected)

    def test_batched_hazard_primary_fallback_unavailable_and_700yd_gate(self) -> None:
        scalar = hazard.EndpointLinkedHazardResolver(
            self._hazard_donors(17), minimum_category_support=12,
            same_hole_fallback_max_distance_yd=150.0,
        )
        batch = stream.BatchedEndpointHazardResolver(scalar)
        # Donors span only 0-16 feet: 0 is primary, 406 feet is 130 yd
        # from the last donor and therefore same-hole fallback, 616 feet is
        # 200 yd away and unavailable.
        x = np.asarray([0.0, 406.0, 616.0])
        result = batch.resolve_many(
            endpoint_x=x, endpoint_y=np.zeros(3), round_no=1, hole_no=1,
            category="ott", common_uniform=np.full(3, 0.5),
            forecast_pin_x=0.0, forecast_pin_y=0.0,
        )
        np.testing.assert_array_equal(result.available, [True, True, False])
        np.testing.assert_array_equal(result.fallback_used, [False, True, False])
        too_far = batch.resolve_many(
            endpoint_x=np.asarray([0.0]), endpoint_y=np.asarray([0.0]),
            round_no=1, hole_no=1, category="ott",
            common_uniform=np.asarray([0.5]),
            forecast_pin_x=-3000.0, forecast_pin_y=0.0,
        )
        self.assertFalse(bool(too_far.available[0]))
        with self.assertRaises(hazard.HazardResolutionUnavailable):
            scalar.resolve(
                endpoint_x=0.0, endpoint_y=0.0, round_no=1, hole_no=1,
                category="ott", common_uniform=0.5,
                forecast_pin_x=-3000.0, forecast_pin_y=0.0,
            )

    def test_substituted_atom_ids_receive_a_new_payload_hash(self) -> None:
        support = precompute.SupportArrays.create(
            key="tee|raw|fixture",
            atom_ids=np.asarray(["real-a", "real-b"]),
            points=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            weights=np.asarray([0.4, 0.6]),
            source="fixture", n_setup=2, n_hole=2, n_broad=2,
        )
        kept = stream.StreamingSupportRepository(
            arm="raw", keep_atom_ids=True
        )._convert(support)
        replaced = stream.StreamingSupportRepository(
            arm="raw", keep_atom_ids=False
        )._convert(support)
        self.assertEqual(kept.cache_sha256, replaced.cache_sha256)
        self.assertNotEqual(kept.payload_sha256, replaced.payload_sha256)
        np.testing.assert_array_equal(replaced.atom_ids, [0, 1])

    def test_reference_hash_gate_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                "neutral_tee_atoms.parquet": b"tee",
                "neutral_app_atoms.parquet": b"app",
                "legacy_terminal_scores.parquet": b"terminal",
                "selected_atom_ids.parquet": b"ids",
                "raw_support_arrays.npz": b"raw",
                "normalized_support_arrays.npz": b"normalized",
            }
            for name, value in payloads.items():
                (root / name).write_bytes(value)
            file_hash = {
                name: hashlib.sha256(value).hexdigest()
                for name, value in payloads.items()
            }
            manifest = {
                "artifact": stream.FROZEN_REFERENCE_PRECOMPUTE_VERSION,
                "event": precompute.DEFAULT_EVENT,
                "holdout_year": precompute.DEFAULT_HOLDOUT,
                "particles": 128,
                "alignment_count": 1,
                "normalization_mode": precompute.DEFAULT_MODE,
                "code_sha256": (
                    stream.FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256
                ),
                "raw_support_npz": {"sha256": file_hash["raw_support_arrays.npz"]},
                "normalized_support_npz": {
                    "sha256": file_hash["normalized_support_arrays.npz"]
                },
            }
            manifest_bytes = json.dumps(manifest).encode()
            (root / "manifest.json").write_bytes(manifest_bytes)
            expected = {
                name: file_hash[name]
                for name in (
                    "neutral_tee_atoms.parquet", "neutral_app_atoms.parquet",
                    "legacy_terminal_scores.parquet", "selected_atom_ids.parquet",
                )
            }
            expected["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
            with mock.patch.object(stream, "FROZEN_REFERENCE_SHA256", expected):
                stream._load_reference_manifest(root)
                bad_manifest = {
                    **manifest,
                    "code_sha256": stream.FROZEN_DEPENDENCY_SHA256[
                        "precompute_source"
                    ],
                }
                bad_manifest_bytes = json.dumps(bad_manifest).encode()
                (root / "manifest.json").write_bytes(bad_manifest_bytes)
                expected["manifest.json"] = hashlib.sha256(
                    bad_manifest_bytes
                ).hexdigest()
                with self.assertRaisesRegex(ValueError, "producer source"):
                    stream._load_reference_manifest(root)
                (root / "manifest.json").write_bytes(manifest_bytes)
                expected["manifest.json"] = hashlib.sha256(
                    manifest_bytes
                ).hexdigest()
                (root / "neutral_app_atoms.parquet").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    stream._load_reference_manifest(root)

    def test_reference_producer_is_distinct_from_current_precompute(self) -> None:
        current = stream.FROZEN_DEPENDENCY_SHA256["precompute_source"]
        self.assertNotEqual(
            current, stream.FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256
        )
        provenance = stream.reference_precompute_provenance(current)
        self.assertEqual(
            provenance["historical_producer_sha256"],
            stream.FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256,
        )
        self.assertEqual(provenance["current_precompute_sha256"], current)
        self.assertFalse(provenance["legacy_reference_tail_policy_attested"])
        self.assertEqual(
            provenance["authoritative_runtime_tail_policy"],
            tail_policy.manifest_policy(),
        )

    def test_kernel_diagnostics_merges_peak_by_max(self) -> None:
        left = stream.KernelDiagnostics(
            tee_draws=3, max_live_chunk_bytes=100
        )
        right = stream.KernelDiagnostics(
            tee_draws=4, max_live_chunk_bytes=80
        )
        left.merge(right)
        self.assertEqual(left.tee_draws, 7)
        self.assertEqual(left.max_live_chunk_bytes, 100)

    def test_report_table_does_not_require_tabulate(self) -> None:
        frame = pd.DataFrame(
            {
                "round_no": [1],
                "mean_strokes": [70.25],
                "note": ["raw|control"],
            }
        )
        with mock.patch.object(
            pd.DataFrame,
            "to_markdown",
            side_effect=ImportError("tabulate intentionally unavailable"),
        ):
            rendered = stream._deterministic_markdown_table(frame)
        self.assertEqual(
            rendered,
            "| round_no | mean_strokes | note |\n"
            "| --- | --- | --- |\n"
            "| 1 | 70.25 | raw\\|control |",
        )

    def test_incomplete_particles_fail_without_survivor_filtering(self) -> None:
        class IncompleteEngine:
            def simulate_hole(self, **kwargs):
                ids = np.asarray(kwargs["particle_ids"], dtype=np.int64)
                size = len(ids)
                return stream.StreamingHoleResult(
                    particle_id=ids,
                    hole_strokes=np.full(size, -1, dtype=np.int16),
                    terminal=np.zeros(size, dtype=bool),
                    status_code=np.full(
                        size, stream.STATUS_NO_SPATIAL_SUPPORT, dtype=np.int8
                    ),
                    hazard_transitions=np.zeros(size, dtype=np.int16),
                    penalty_strokes=np.zeros(size, dtype=np.int16),
                    transition_count=np.zeros(size, dtype=np.int16),
                    return_to_app_count=np.zeros(size, dtype=np.int16),
                    tee_atom_id=np.full(size, "x", dtype=object),
                    app_atom_id=np.full(size, "x", dtype=object),
                    initial_endpoint=np.zeros((size, 2), dtype=float),
                    initial_lie=np.full(size, stream.LIE_TEE, dtype=np.int8),
                    initial_remaining_yd=np.ones(size, dtype=float),
                    initial_strokes=np.zeros(size, dtype=np.int16),
                    initial_status_code=np.full(
                        size, stream.STATUS_ACTIVE, dtype=np.int8
                    ),
                    final_endpoint=np.zeros((size, 2), dtype=float),
                    final_lie=np.full(size, stream.LIE_UNKNOWN, dtype=np.int8),
                    final_remaining_yd=np.full(size, np.nan),
                ), stream.KernelDiagnostics(unsupported_state_rows=size)

        grid = pd.DataFrame(
            [
                {
                    "round_no": 1,
                    "hole_no": 1,
                    "case_id": 0,
                    "event_key": "event",
                    "source_player_id": "player",
                    "par": 4,
                    "forecast_tee_x": 0.0,
                    "forecast_tee_y": 0.0,
                    "forecast_pin_x": 1.0,
                    "forecast_pin_y": 0.0,
                    "forecast_pin_long_yd": 400.0,
                }
            ]
        )
        with self.assertRaisesRegex(full.ChainIncomplete, "Survivor filtering forbidden"):
            stream._simulate_grid(
                IncompleteEngine(),
                grid,
                arms=("raw",),
                particles=4,
                chunk_size=2,
                rng_mode="counter",
                collect_holes=False,
            )

    def test_thirteen_stroke_hole_round_aggregation_is_chunk_invariant(self) -> None:
        class TerminalEngine:
            def simulate_hole(self, **kwargs):
                ids = np.asarray(kwargs["particle_ids"], dtype=np.int64)
                size = len(ids)
                score = 13 if int(kwargs["record"].hole_no) == 1 else 3
                zeros_i16 = np.zeros(size, dtype=np.int16)
                zeros_xy = np.zeros((size, 2), dtype=float)
                return stream.StreamingHoleResult(
                    particle_id=ids,
                    hole_strokes=np.full(size, score, dtype=np.int16),
                    terminal=np.ones(size, dtype=bool),
                    status_code=np.full(
                        size, stream.STATUS_TERMINAL, dtype=np.int8
                    ),
                    hazard_transitions=zeros_i16.copy(),
                    penalty_strokes=zeros_i16.copy(),
                    transition_count=np.ones(size, dtype=np.int16),
                    return_to_app_count=zeros_i16.copy(),
                    tee_atom_id=np.full(size, "fixture", dtype=object),
                    app_atom_id=np.full(size, "fixture", dtype=object),
                    initial_endpoint=zeros_xy.copy(),
                    initial_lie=np.full(
                        size, stream.LIE_GREEN, dtype=np.int8
                    ),
                    initial_remaining_yd=np.ones(size),
                    initial_strokes=zeros_i16.copy(),
                    initial_status_code=np.full(
                        size, stream.STATUS_ACTIVE, dtype=np.int8
                    ),
                    final_endpoint=zeros_xy.copy(),
                    final_lie=np.full(size, stream.LIE_HOLE, dtype=np.int8),
                    final_remaining_yd=np.zeros(size),
                ), stream.KernelDiagnostics()

        grid = pd.DataFrame(
            [vars(self._record(hole_no=hole_no)) for hole_no in range(1, 19)]
        )
        one, _, diagnostics = stream._simulate_grid(
            TerminalEngine(), grid, arms=("raw",), particles=7,
            chunk_size=1, rng_mode="counter", collect_holes=False,
        )
        many, _, _ = stream._simulate_grid(
            TerminalEngine(), grid, arms=("raw",), particles=7,
            chunk_size=4, rng_mode="counter", collect_holes=False,
        )
        np.testing.assert_array_equal(one["raw"], many["raw"])
        np.testing.assert_array_equal(
            one["raw"][0], np.full(7, 64, dtype=np.int16)
        )
        self.assertEqual(one["raw"].dtype, np.int16)
        hole_summary = pd.DataFrame(diagnostics["hole_summary"])
        self.assertEqual(
            int(
                hole_summary.loc[
                    hole_summary["hole_no"].eq(1), "stroke_sum"
                ].iloc[0]
            ),
            13 * 7,
        )

    def test_known_50k_tail_regression_contract_is_exact_and_bounded(self) -> None:
        class RegressionEngine:
            def __init__(self, *, mutate: bool = False):
                self.mutate = mutate

            def simulate_hole(self, **kwargs):
                ids = np.asarray(kwargs["particle_ids"], dtype=np.int64)
                scores = np.asarray(
                    [stream.KNOWN_TAIL_REGRESSION[int(value)] for value in ids],
                    dtype=np.int16,
                )
                if self.mutate and len(scores) and int(ids[0]) == 27481:
                    scores[0] += 1
                size = len(ids)
                zeros_i16 = np.zeros(size, dtype=np.int16)
                zeros_xy = np.zeros((size, 2), dtype=float)
                return stream.StreamingHoleResult(
                    particle_id=ids,
                    hole_strokes=scores,
                    terminal=np.ones(size, dtype=bool),
                    status_code=np.full(
                        size, stream.STATUS_TERMINAL, dtype=np.int8
                    ),
                    hazard_transitions=np.asarray(
                        [{27481: 5, 35295: 3, 48797: 5}[int(value)] for value in ids],
                        dtype=np.int16,
                    ),
                    penalty_strokes=np.asarray(
                        [{27481: 5, 35295: 3, 48797: 5}[int(value)] for value in ids],
                        dtype=np.int16,
                    ),
                    transition_count=np.asarray(
                        [{27481: 8, 35295: 8, 48797: 7}[int(value)] for value in ids],
                        dtype=np.int16,
                    ),
                    return_to_app_count=zeros_i16.copy(),
                    tee_atom_id=np.full(size, "fixture", dtype=object),
                    app_atom_id=np.full(size, "fixture", dtype=object),
                    initial_endpoint=zeros_xy.copy(),
                    initial_lie=np.full(size, stream.LIE_GREEN, dtype=np.int8),
                    initial_remaining_yd=np.ones(size),
                    initial_strokes=zeros_i16.copy(),
                    initial_status_code=np.full(
                        size, stream.STATUS_ACTIVE, dtype=np.int8
                    ),
                    final_endpoint=zeros_xy.copy(),
                    final_lie=np.full(size, stream.LIE_HOLE, dtype=np.int8),
                    final_remaining_yd=np.zeros(size),
                ), stream.KernelDiagnostics()

        prepared = SimpleNamespace(
            grid=pd.DataFrame(
                [vars(self._record(hole_no=stream.KNOWN_TAIL_REGRESSION_HOLE))]
            ).assign(round_no=stream.KNOWN_TAIL_REGRESSION_ROUND)
        )
        audit = stream.run_known_tail_regression(
            RegressionEngine(), prepared
        )
        self.assertTrue(audit["exact"])
        self.assertEqual(audit["observed_hole_strokes"], [14, 13, 14])
        self.assertTrue(audit["singleton_chunk_exact"])
        with self.assertRaisesRegex(AssertionError, "scores changed"):
            stream.run_known_tail_regression(
                RegressionEngine(mutate=True), prepared
            )


if __name__ == "__main__":
    unittest.main()

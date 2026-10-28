"""Archive-only streaming normalized shot-chain research kernel.

This module is deliberately separate from the authoritative normalized atom
precompute, the validated full-hole chain and production.  It consumes their
frozen contracts, processes one round/hole/particle chunk at a time and keeps
only int16 round totals plus compact diagnostics.

Two execution modes are explicit:

* ``legacy128`` reproduces the authoritative Travelers raw 128-particle
  artifact with its historical RNG streams.
* ``counter`` uses the arm-independent counter RNG required for chunk and
  50k/100k-prefix invariance.

APP state grouping uses vectorized lie/bucket encoding. ARG/PUTT uses
``BatchedPopulationTransitionSampler.sample_many``. Endpoint-linked hazards
use a vectorized resolver that reproduces the scalar hierarchy without pandas
or per-endpoint neighbour searches in the timed kernel.  Target support
materialization is timed separately and cached once per state/setup.

The default command runs only the 128 parity gate and a 5,000-particle
benchmark.  It cannot launch 50k/100k; those sizes are rejected here and the
separate 50k harness remains approval-lock gated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Iterable, Mapping
import zlib

import numpy as np
import pandas as pd
import psutil


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_app_transition as app  # noqa: E402
import hole_sim_batched_population_transition as batched_population  # noqa: E402
import hole_sim_full_chain as full  # noqa: E402
import hole_sim_hazard_resolution as hazard  # noqa: E402
import hole_sim_normalized_atom_precompute as precompute  # noqa: E402
import hole_sim_normalized_chain_hook as hook  # noqa: E402
import hole_sim_tail_policy as tail_policy  # noqa: E402
import hole_sim_tee_app_integrated_v2 as spatial  # noqa: E402
import hole_sim_tee_transition as tee  # noqa: E402


STREAMING_VERSION = "normalized_chain_streaming_v2_unbounded_score_tail"
DEFAULT_REFERENCE = (
    REPO / "permanent_data" / "hole_sim_normalized_atom_precompute_travelers_smoke"
)
DEFAULT_OUTPUT = (
    REPO / "permanent_data" / "hole_sim_normalized_chain_streaming_travelers_5k"
)
DEFAULT_BENCHMARK_PARTICLES = 5_000
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_SEED = 20260824
DEFAULT_CONTINUATION_SEED = DEFAULT_SEED + 500
MAX_APPROVED_PARTICLES = 5_000
SAFE_RSS_CAP_BYTES = 4 * 2**30
KNOWN_TAIL_REGRESSION = {
    27481: 14,
    35295: 13,
    48797: 14,
}
KNOWN_TAIL_REGRESSION_ROUND = 3
KNOWN_TAIL_REGRESSION_HOLE = 13

FROZEN_REFERENCE_SHA256 = {
    "manifest.json": "6bfacca2cb4a0714e4ae3a31ee2e09d8ee851cfba85357b548c023b66207830c",
    "neutral_tee_atoms.parquet": "366eaafea86d1cc046011169a8edc820e327bec047cfd9ce71c5a4f8186be82d",
    "neutral_app_atoms.parquet": "29d413698c50fbc90c7c14d4574d8f9a61e035744273793d3642513272144b20",
    "legacy_terminal_scores.parquet": "46a560ce3e49fc6684c30cfebb23527c13b699465779b561f379fca0ea6befde",
    "selected_atom_ids.parquet": "9d0061daf7fb803ca185e70151f43ff0bf7889dddf3a0557b4f1847c1a4e0681",
}
# The immutable 128-particle reference was produced before the shared tail
# policy existed.  Its historical producer identity is verified independently
# from the current precompute implementation consumed by this executable.
FROZEN_REFERENCE_PRECOMPUTE_VERSION = precompute.REFERENCE_PRECOMPUTE_VERSION
FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256 = (
    "daf6f7aba48a92767f64f6e758771445046dd1ac9dfdb9e6fa65748c74842509"
)
FROZEN_DEPENDENCY_SHA256 = {
    "precompute_source": "3bd580b135f343cf2dbb0f9347b8678dd9c6eace66218826852f411e02d6e773",
    "full_chain_source": "10d8971b45843bc74d4dbd7b5c53d94cdb3f35e0e8621bcba357acce77a20c89",
    "batched_population_source": "0da3a934b671c3ed74a24c8c754a8404e3a329eb4f54f3b429152e45cd7fca23",
    "hazard_source": "8a3d8e019c39156299ebde6e7e8a344ba2edf5b55dc8602981ab75e4466163e5",
    "spatial_v2_source": "dc85b38b5601ba5f6afbe2ce4d900e109243f60cb6a9dd7a03426b436443ea97",
    "tee_source": "006f1de02d98be47dfd621a5e46d14672466ba6ee673cf7e0f56bc94daf56b43",
    "app_source": "6acceff91e9130e3584d02d4d113daecc5e84831dfe3fff6f7190d40f5450e1f",
    "asof_source": "9463aa4b0c2fd67adec11ccd109126d0bbf144088dca8095ad5965216482df3c",
    "normalizer_source": "57beb668622f2635dd47bc60b05de323ee05cae63247f0afdafb71e6e4b75e17",
    "hook_source": "311523564b178227ade614ffa598ed9787142b1a23136da9ea71f5c8b37e03a7",
    "markov_source": "6740b3c2ce6119084bbb5371a3ea84ab159f5eeeb1db746188fe25d5ab67a4cf",
    "state_foundations_source": "3f499f4a4746f896f3cf8ae39465cc6be4ab5cd5d7688151e99886bd4a381eaf",
    "tail_policy_source": "1f01e4719febe527cf064c3e5e0ff7281958c1d8adcd7027352d83dd9d95f4de",
}

LIE_HOLE = np.int8(0)
LIE_TEE = np.int8(1)
LIE_FAIRWAY = np.int8(2)
LIE_ROUGH = np.int8(3)
LIE_SAND = np.int8(4)
LIE_RECOVERY = np.int8(5)
LIE_GREEN = np.int8(6)
LIE_PENALTY = np.int8(7)
LIE_UNKNOWN = np.int8(8)
LIE_NAMES = np.asarray(
    ["hole", "tee", "fairway", "rough", "sand", "recovery", "green", "penalty", "unknown"],
    dtype=object,
)
APP_START_NAMES = np.asarray(
    ["unknown", "tee", "fairway", "rough", "sand", "recovery", "unknown", "unknown", "unknown"],
    dtype=object,
)
TEE_SURFACE_TO_LIE = np.asarray(
    [LIE_FAIRWAY, LIE_ROUGH, LIE_SAND, LIE_RECOVERY, LIE_PENALTY, LIE_GREEN, LIE_PENALTY],
    dtype=np.int8,
)
APP_SURFACE_TO_LIE = np.asarray(
    [LIE_GREEN, LIE_FAIRWAY, LIE_ROUGH, LIE_SAND, LIE_RECOVERY, LIE_TEE,
     LIE_HOLE, LIE_PENALTY, LIE_RECOVERY],
    dtype=np.int8,
)

CATEGORY_INACTIVE = np.int8(-1)
CATEGORY_TERMINAL = np.int8(0)
CATEGORY_OTT = np.int8(1)
CATEGORY_APP = np.int8(2)
CATEGORY_ARG = np.int8(3)
CATEGORY_PUTT = np.int8(4)
CATEGORY_INVALID = np.int8(5)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_logical_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(map(str, frame.columns))
    work = frame.reindex(columns=columns).copy()
    for name in columns:
        if work[name].dtype == object:
            work[name] = work[name].map(
                lambda value: (
                    json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (list, tuple, dict, set, np.ndarray))
                    else str(value)
                )
            )
    hashes = pd.util.hash_pandas_object(
        work, index=False, categorize=True
    ).to_numpy(np.uint64)
    hashes.sort()
    digest = sha256(json.dumps(
        [(name, str(work[name].dtype)) for name in columns],
        separators=(",", ":"),
    ).encode("utf-8"))
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def dependency_paths() -> dict[str, Path]:
    return {
        "precompute_source": Path(precompute.__file__),
        "full_chain_source": Path(full.__file__),
        "batched_population_source": Path(batched_population.__file__),
        "hazard_source": Path(hazard.__file__),
        "spatial_v2_source": Path(spatial.__file__),
        "tee_source": Path(tee.__file__),
        "app_source": Path(app.__file__),
        "asof_source": Path(precompute.asof.__file__),
        "normalizer_source": Path(precompute.norm.__file__),
        "hook_source": Path(hook.__file__),
        "markov_source": Path(full.markov.__file__),
        "state_foundations_source": Path(precompute.foundations.__file__),
        "tail_policy_source": Path(tail_policy.__file__),
    }


def validate_frozen_dependencies() -> dict[str, str]:
    observed = {name: sha256_file(path) for name, path in dependency_paths().items()}
    mismatches = {
        name: {"expected": FROZEN_DEPENDENCY_SHA256[name], "observed": value}
        for name, value in observed.items()
        if value != FROZEN_DEPENDENCY_SHA256.get(name)
    }
    if mismatches:
        raise ValueError(f"Frozen dependency compatibility failure: {mismatches}")
    return observed


def reference_precompute_provenance(
    current_precompute_sha256: str,
) -> dict[str, object]:
    """Describe the frozen v1 parity producer and current v2 consumer."""

    return {
        "reference_artifact": FROZEN_REFERENCE_PRECOMPUTE_VERSION,
        "historical_producer_sha256": (
            FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256
        ),
        "current_precompute_artifact": precompute.PRECOMPUTE_VERSION,
        "current_precompute_sha256": str(current_precompute_sha256),
        "legacy_reference_tail_policy_attested": False,
        "legacy_reference_role": "bounded_128_particle_parity_only",
        "authoritative_runtime_tail_policy": tail_policy.manifest_policy(),
    }


def array_sha256(*arrays: np.ndarray) -> str:
    digest = sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        if array.dtype.kind in {"O", "U", "S"}:
            for item in array.reshape(-1):
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def support_payload_sha256(
    key: str,
    atom_ids: np.ndarray,
    points: np.ndarray,
    probability: np.ndarray,
    cdf: np.ndarray,
    holeout_probability: float,
) -> str:
    """Hash the arrays actually held by the streaming cache."""

    digest = sha256(str(key).encode("utf-8"))
    digest.update(
        array_sha256(
            np.asarray(atom_ids),
            np.asarray(points),
            np.asarray(probability),
            np.asarray(cdf),
            np.asarray([holeout_probability], dtype=float),
        ).encode("ascii")
    )
    return digest.hexdigest()


class RssPeakMonitor:
    """Low-overhead process RSS sampler for phase-separated memory audits."""

    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.peak_bytes = int(self.process.memory_info().rss)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS monitor already started")

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_bytes = max(
                    self.peak_bytes, int(self.process.memory_info().rss)
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_bytes = max(
            self.peak_bytes, int(self.process.memory_info().rss)
        )
        return int(self.peak_bytes)

    def assert_below(self, cap_bytes: int, phase: str) -> None:
        current = int(self.process.memory_info().rss)
        self.peak_bytes = max(self.peak_bytes, current)
        if self.peak_bytes > int(cap_bytes):
            raise MemoryError(
                f"{phase} RSS exceeded cap: {self.peak_bytes / 2**30:.2f} GiB "
                f"> {int(cap_bytes) / 2**30:.2f} GiB"
            )


@dataclass(frozen=True)
class NumericSupport:
    key: str
    atom_ids: np.ndarray
    points: np.ndarray
    probability: np.ndarray
    cdf: np.ndarray
    holeout_probability: float
    cache_sha256: str
    payload_sha256: str = ""

    def select(self, uniforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        index = np.searchsorted(self.cdf, np.asarray(uniforms, float), side="right")
        np.minimum(index, len(self.atom_ids) - 1, out=index)
        return index, self.atom_ids[index]


class StreamingSupportRepository:
    """Numeric cache with optional out-of-kernel provider materialization."""

    def __init__(
        self,
        *,
        arm: str,
        artifact_npz: Path | None = None,
        provider: precompute.AtomSupportProvider | None = None,
        fail_on_miss: bool = False,
        keep_atom_ids: bool = True,
    ) -> None:
        self.arm = str(arm)
        self.provider = provider
        self.fail_on_miss = bool(fail_on_miss)
        self.keep_atom_ids = bool(keep_atom_ids)
        self.supports: dict[str, NumericSupport] = {}
        self.materialization_seconds = 0.0
        self.materialization_calls = 0
        self.materialized_atoms = 0
        self.hit_count = 0
        self.miss_count = 0
        if artifact_npz is not None:
            self._load_npz(Path(artifact_npz))

    def _load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            keys = data["support_keys"].astype(str)
            offsets = data["offsets"].astype(np.int64)
            all_points = data["points_yd"]
            all_probability = data["probability"]
            all_cdf = data["cdf"]
            all_atom_ids = data["atom_ids"]
            all_holeout = data["holeout_probability"]
            all_cache_sha = data["cache_sha256"].astype(str)
            for index, key in enumerate(keys):
                if not (
                    str(key).startswith(f"tee|{self.arm}|")
                    or str(key).startswith(f"app|{self.arm}|")
                ):
                    raise ValueError(
                        f"Persisted support arm/key mismatch: {self.arm} vs {key}"
                    )
                start, stop = int(offsets[index]), int(offsets[index + 1])
                points = np.ascontiguousarray(all_points[start:stop])
                probability = np.ascontiguousarray(all_probability[start:stop])
                cdf = np.ascontiguousarray(all_cdf[start:stop])
                atom_ids = (
                    np.ascontiguousarray(all_atom_ids[start:stop].astype(str))
                    if self.keep_atom_ids
                    else np.arange(stop - start, dtype=np.int32)
                )
                if not len(atom_ids) or cdf[-1] != 1.0:
                    raise ValueError(f"Invalid persisted support {key}")
                if len(str(all_cache_sha[index])) != 64:
                    raise ValueError(f"Invalid source cache hash for {key}")
                holeout = float(all_holeout[index])
                self.supports[str(key)] = NumericSupport(
                    key=str(key),
                    atom_ids=atom_ids,
                    points=points,
                    probability=probability,
                    cdf=cdf,
                    holeout_probability=holeout,
                    cache_sha256=str(all_cache_sha[index]),
                    payload_sha256=support_payload_sha256(
                        str(key), atom_ids, points, probability, cdf, holeout
                    ),
                )

    @staticmethod
    def tee_key(arm: str, round_no: int, hole_no: int, par: int, player_id: str) -> str:
        return f"tee|{arm}|R{int(round_no)}|H{int(hole_no)}|P{int(par)}|{player_id}"

    @staticmethod
    def app_key(
        arm: str,
        round_no: int,
        hole_no: int,
        start_lie: str,
        bucket: int,
        player_id: str,
    ) -> str:
        return (
            f"app|{arm}|R{int(round_no)}|H{int(hole_no)}|{start_lie}|"
            f"B{int(bucket)}|{player_id}"
        )

    def _convert(self, support: precompute.SupportArrays) -> NumericSupport:
        atom_ids = (
            np.ascontiguousarray(support.atom_ids)
            if self.keep_atom_ids
            else np.arange(len(support.atom_ids), dtype=np.int32)
        )
        points = np.ascontiguousarray(support.points)
        probability = np.ascontiguousarray(support.weights)
        cdf = np.ascontiguousarray(support.cdf)
        return NumericSupport(
            key=support.key,
            atom_ids=atom_ids,
            points=points,
            probability=probability,
            cdf=cdf,
            holeout_probability=float(support.holeout_probability),
            cache_sha256=str(support.cache_sha256),
            payload_sha256=support_payload_sha256(
                support.key,
                atom_ids,
                points,
                probability,
                cdf,
                float(support.holeout_probability),
            ),
        )

    def _materialize(self, key: str, builder) -> NumericSupport:
        existing = self.supports.get(key)
        if existing is not None:
            self.hit_count += 1
            return existing
        self.miss_count += 1
        if self.fail_on_miss or self.provider is None:
            raise KeyError(f"Required support is absent from frozen cache: {key}")
        started = time.perf_counter()
        support = self._convert(builder())
        if support.key != key:
            raise AssertionError(
                f"Support builder key drift: requested {key}, built {support.key}"
            )
        expected_payload = support_payload_sha256(
            support.key,
            support.atom_ids,
            support.points,
            support.probability,
            support.cdf,
            support.holeout_probability,
        )
        if support.payload_sha256 != expected_payload:
            raise AssertionError("Support builder payload hash is inconsistent")
        if len(support.cache_sha256) != 64:
            raise AssertionError("Support builder source cache hash is invalid")
        self.materialization_seconds += time.perf_counter() - started
        self.materialization_calls += 1
        self.materialized_atoms += len(support.atom_ids)
        self.supports[key] = support
        # The authoritative provider retains audit DataFrames. Streaming owns
        # only numeric supports, so release those large transient rows now.
        self.provider.materialized_rows.clear()
        self.provider.tee_cache.clear()
        self.provider.app_cache.clear()
        return support

    def tee(
        self, round_no: int, hole_no: int, par: int, player_id: str
    ) -> NumericSupport:
        key = self.tee_key(self.arm, round_no, hole_no, par, player_id)
        return self._materialize(
            key,
            lambda: self.provider.tee_support(
                round_no=round_no,
                hole_no=hole_no,
                par=par,
                player_id=player_id,
            ),
        )

    def app(
        self,
        round_no: int,
        hole_no: int,
        start_lie: str,
        bucket: int,
        player_id: str,
    ) -> NumericSupport:
        key = self.app_key(
            self.arm, round_no, hole_no, start_lie, bucket, player_id
        )
        return self._materialize(
            key,
            lambda: self.provider.app_support(
                round_no=round_no,
                hole_no=hole_no,
                start_lie=start_lie,
                distance_bucket=bucket,
                player_id=player_id,
            ),
        )

    @property
    def cache_nbytes(self) -> int:
        return int(
            sum(
                support.atom_ids.nbytes
                + support.points.nbytes
                + support.probability.nbytes
                + support.cdf.nbytes
                for support in self.supports.values()
            )
        )


class UniformFactory:
    """Legacy-compatibility or scalable counter-based semantic streams."""

    def __init__(
        self,
        *,
        mode: str,
        seed: int,
        continuation_seed: int,
        event_key: str,
        player_id: str,
        round_no: int,
        hole_no: int,
        total_particles: int,
    ) -> None:
        if mode not in {"legacy128", "counter"}:
            raise ValueError("RNG mode must be legacy128 or counter")
        self.mode = mode
        self.seed = int(seed)
        self.continuation_seed = int(continuation_seed)
        self.event_key = str(event_key)
        self.player_id = str(player_id)
        self.round_no = int(round_no)
        self.hole_no = int(hole_no)
        self.total_particles = int(total_particles)
        self._legacy_initial: dict[str, np.ndarray] = {}
        if mode == "legacy128":
            if self.total_particles != 128:
                raise ValueError("legacy128 RNG is defined only for 128 particles")
            case_seed = np.random.SeedSequence(
                [
                    self.seed,
                    zlib.crc32(
                        f"{self.player_id}|{self.round_no}|{self.hole_no}".encode()
                    ),
                ]
            )
            rng = np.random.default_rng(case_seed)
            self._legacy_initial = {
                name: rng.random(self.total_particles)
                for name in (
                    "tee_index", "tee_lie", "app_index", "app_holeout", "app_lie"
                )
            }

    def initial(self, stream: str, particle_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(particle_ids, dtype=np.int64)
        if self.mode == "legacy128":
            return self._legacy_initial[str(stream)][ids]
        return hook.counter_uniforms(
            seed=self.seed,
            event_key=self.event_key,
            player_id=self.player_id,
            round_no=self.round_no,
            hole_no=self.hole_no,
            transition_index=-1,
            stream=f"initial_{stream}",
            particle_ids=ids,
        )

    def transition(
        self, transition_index: int, stream: str, particle_ids: np.ndarray
    ) -> np.ndarray:
        ids = np.asarray(particle_ids, dtype=np.int64)
        if self.mode == "legacy128":
            return full._common_uniforms(
                seed=self.continuation_seed,
                event_key=self.event_key,
                player_id=self.player_id,
                round_no=self.round_no,
                hole_no=self.hole_no,
                particle_ids=ids,
                transition_index=int(transition_index),
                stream=str(stream),
            )
        return hook.counter_uniforms(
            seed=self.continuation_seed,
            event_key=self.event_key,
            player_id=self.player_id,
            round_no=self.round_no,
            hole_no=self.hole_no,
            transition_index=int(transition_index),
            stream=str(stream),
            particle_ids=ids,
        )


@dataclass(frozen=True)
class HazardBatchResult:
    available: np.ndarray
    donor_key: np.ndarray
    penalty_count: np.ndarray
    penalty_stroke_cost: np.ndarray
    resolution_x: np.ndarray
    resolution_y: np.ndarray
    resolution_lie: np.ndarray
    resolution_distance_yd: np.ndarray
    fallback_used: np.ndarray


@dataclass(frozen=True)
class _HazardCategoryArrays:
    donor_key: np.ndarray
    endpoint_x: np.ndarray
    endpoint_y: np.ndarray
    round_no: np.ndarray
    hole_no: np.ndarray
    penalty_count: np.ndarray
    penalty_stroke_cost: np.ndarray
    resolution_x: np.ndarray
    resolution_y: np.ndarray
    resolution_lie: np.ndarray


class BatchedEndpointHazardResolver:
    """Vectorized equivalent of EndpointLinkedHazardResolver.resolve."""

    def __init__(
        self,
        scalar: hazard.EndpointLinkedHazardResolver,
        *,
        query_chunk_size: int = 512,
    ) -> None:
        self.scalar = scalar
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.categories: dict[str, _HazardCategoryArrays] = {}
        for category, frame in scalar.donors.groupby("category", observed=True):
            frame = frame.sort_values("donor_key").reset_index(drop=True)
            self.categories[str(category)] = _HazardCategoryArrays(
                donor_key=frame["donor_key"].astype(str).to_numpy(),
                endpoint_x=frame["causal_endpoint_x"].to_numpy(float),
                endpoint_y=frame["causal_endpoint_y"].to_numpy(float),
                round_no=frame["round_no"].to_numpy(np.int16),
                hole_no=frame["hole_no"].to_numpy(np.int16),
                penalty_count=frame["penalty_count"].to_numpy(np.int16),
                penalty_stroke_cost=frame["penalty_stroke_cost"].to_numpy(np.int16),
                resolution_x=frame["resolution_x"].to_numpy(float),
                resolution_y=frame["resolution_y"].to_numpy(float),
                resolution_lie=frame["resolution_lie"].astype(str).to_numpy(),
            )
        self.query_rows = 0
        self.max_working_bytes = 0
        self.kernel_seconds = 0.0

    def _resolve_chunk(
        self,
        arrays: _HazardCategoryArrays,
        endpoint_x: np.ndarray,
        endpoint_y: np.ndarray,
        uniforms: np.ndarray,
        *,
        round_no: int,
        hole_no: int,
        forecast_pin_x: float,
        forecast_pin_y: float,
    ) -> HazardBatchResult:
        q = len(endpoint_x)
        dx = arrays.endpoint_x[None, :] - endpoint_x[:, None]
        dy = arrays.endpoint_y[None, :] - endpoint_y[:, None]
        distance = np.hypot(dx, dy) / hazard.FEET_PER_YARD
        nearest_category = distance.min(axis=1)
        same_hole = arrays.hole_no == int(hole_no)
        nearest_hole = (
            distance[:, same_hole].min(axis=1)
            if same_hole.any() else np.full(q, np.inf)
        )
        enough_category_support = (
            len(arrays.donor_key) >= self.scalar.minimum_category_support
        )
        primary = (
            enough_category_support
            & (nearest_category <= self.scalar.max_nearest_distance_yd)
        )
        fallback = np.zeros(q, dtype=bool)
        if self.scalar.same_hole_fallback_max_distance_yd is not None:
            fallback = (
                enough_category_support
                & ~primary
                & (nearest_hole <= self.scalar.same_hole_fallback_max_distance_yd)
            )
        available = primary | fallback
        within = distance <= self.scalar.max_nearest_distance_yd
        if self.scalar.same_hole_fallback_max_distance_yd is not None and fallback.any():
            within[fallback] = (
                distance[fallback] <= self.scalar.same_hole_fallback_max_distance_yd
            ) & same_hole[None, :]
        mass = np.zeros_like(distance)
        remaining_mass = np.ones(q, dtype=float)
        static_pools = (
            (
                (arrays.round_no == int(round_no)) & same_hole,
                self.scalar.setup_prior,
            ),
            (same_hole, self.scalar.hole_prior),
            (np.ones(len(arrays.donor_key), dtype=bool), math.inf),
        )
        for static_mask, prior in static_pools:
            pool = within & static_mask[None, :]
            counts = pool.sum(axis=1)
            present = counts > 0
            if not present.any():
                continue
            masked = np.where(pool, distance, np.nan)
            quantile = np.zeros(q, dtype=float)
            enough = counts > 2
            if enough.any():
                quantile[enough] = np.nanquantile(
                    masked[enough], 0.40, axis=1
                )
            bandwidth = np.maximum(self.scalar.spatial_bandwidth_yd, quantile)
            kernel = np.exp(-0.5 * (distance / bandwidth[:, None]) ** 2)
            kernel = np.where(pool, np.maximum(kernel, 1e-12), 0.0)
            denominator = kernel.sum(axis=1)
            kernel[present] /= denominator[present, None]
            fraction = np.ones(q, dtype=float)
            if np.isfinite(prior):
                fraction = counts / (counts + float(prior))
            component = remaining_mass * fraction
            mass += component[:, None] * kernel
            remaining_mass -= component
        totals = mass.sum(axis=1)
        valid = available & np.isclose(totals, 1.0, atol=1e-8)
        mass[valid] /= totals[valid, None]
        cumulative = np.cumsum(mass, axis=1)
        # Scalar resolver uses searchsorted(..., side="right") and clamps an
        # all-false tail to the final donor. Counting CDF values <= u is the
        # exact row-wise vector equivalent, including exact boundaries and
        # nextafter(1, 0).
        choices = np.count_nonzero(cumulative <= uniforms[:, None], axis=1)
        np.minimum(choices, len(arrays.donor_key) - 1, out=choices)
        choices[~valid] = 0
        resolution_distance = np.hypot(
            arrays.resolution_x[choices] - float(forecast_pin_x),
            arrays.resolution_y[choices] - float(forecast_pin_y),
        ) / hazard.FEET_PER_YARD
        valid &= resolution_distance <= 700.0
        empty_text = np.full(q, "", dtype=object)
        result = HazardBatchResult(
            available=valid,
            donor_key=np.where(valid, arrays.donor_key[choices], empty_text),
            penalty_count=np.where(valid, arrays.penalty_count[choices], 0).astype(np.int16),
            penalty_stroke_cost=np.where(
                valid, arrays.penalty_stroke_cost[choices], 0
            ).astype(np.int16),
            resolution_x=np.where(valid, arrays.resolution_x[choices], np.nan),
            resolution_y=np.where(valid, arrays.resolution_y[choices], np.nan),
            resolution_lie=np.where(valid, arrays.resolution_lie[choices], empty_text),
            resolution_distance_yd=np.where(valid, resolution_distance, np.nan),
            fallback_used=fallback & valid,
        )
        working = (
            dx.nbytes + dy.nbytes + distance.nbytes + within.nbytes
            + mass.nbytes + cumulative.nbytes
        )
        self.max_working_bytes = max(self.max_working_bytes, int(working))
        return result

    def resolve_many(
        self,
        *,
        endpoint_x: np.ndarray,
        endpoint_y: np.ndarray,
        round_no: int,
        hole_no: int,
        category: str,
        common_uniform: np.ndarray,
        forecast_pin_x: float,
        forecast_pin_y: float,
    ) -> HazardBatchResult:
        started = time.perf_counter()
        x, y, uniforms = np.broadcast_arrays(
            np.asarray(endpoint_x, float),
            np.asarray(endpoint_y, float),
            np.asarray(common_uniform, float),
        )
        x, y, uniforms = x.reshape(-1), y.reshape(-1), uniforms.reshape(-1)
        if not (
            np.isfinite(x).all()
            and np.isfinite(y).all()
            and np.isfinite(uniforms).all()
        ):
            raise ValueError("Hazard endpoints and uniforms must be finite")
        if ((uniforms < 0.0) | (uniforms >= 1.0)).any():
            raise ValueError("Hazard uniforms must satisfy 0 <= u < 1")
        arrays = self.categories.get(str(category))
        if arrays is None:
            raise ValueError(f"Unknown hazard category {category}")
        parts: list[HazardBatchResult] = []
        for start in range(0, len(x), self.query_chunk_size):
            stop = min(start + self.query_chunk_size, len(x))
            parts.append(
                self._resolve_chunk(
                    arrays,
                    x[start:stop],
                    y[start:stop],
                    uniforms[start:stop],
                    round_no=round_no,
                    hole_no=hole_no,
                    forecast_pin_x=forecast_pin_x,
                    forecast_pin_y=forecast_pin_y,
                )
            )
        self.query_rows += len(x)
        self.kernel_seconds += time.perf_counter() - started
        if not parts:
            empty = np.asarray([], dtype=float)
            return HazardBatchResult(
                available=np.asarray([], dtype=bool), donor_key=np.asarray([], dtype=object),
                penalty_count=np.asarray([], dtype=np.int16),
                penalty_stroke_cost=np.asarray([], dtype=np.int16),
                resolution_x=empty, resolution_y=empty,
                resolution_lie=np.asarray([], dtype=object),
                resolution_distance_yd=empty,
                fallback_used=np.asarray([], dtype=bool),
            )
        return HazardBatchResult(
            **{
                name: np.concatenate([getattr(part, name) for part in parts])
                for name in HazardBatchResult.__dataclass_fields__
            }
        )


# Compact status codes stay inside one hole/chunk.  STATUS_MAX_STROKES is a
# read/compare compatibility label for frozen artifacts only; the current
# no-score-cap policy never emits it.
STATUS_ACTIVE = np.int8(0)
STATUS_TEE_HAZARD = np.int8(1)
STATUS_APP_HAZARD = np.int8(2)
STATUS_TERMINAL = np.int8(3)
STATUS_NO_SPATIAL_SUPPORT = np.int8(4)
STATUS_NO_SURFACE_SUPPORT = np.int8(5)
STATUS_HAZARD_UNAVAILABLE = np.int8(6)
STATUS_HAZARD_UNKNOWN_LIE = np.int8(7)
STATUS_MAX_STROKES = np.int8(8)
STATUS_INVALID_STATE = np.int8(9)
STATUS_POPULATION_UNAVAILABLE = np.int8(10)
STATUS_POPULATION_INVALID = np.int8(11)
STATUS_TRANSITION_LOOP = np.int8(12)
STATUS_NAMES = np.asarray(
    [
        "active",
        "tee_hazard_unresolved",
        "app_hazard_unresolved",
        "terminal",
        "app_no_spatial_support",
        "no_surface_support",
        "hazard_unavailable",
        "hazard_resolution_unknown_lie",
        "max_hole_strokes_unresolved",
        "invalid_state_category",
        "population_transition_unavailable",
        "population_invalid_next_state",
        "transition_loop_unresolved",
    ],
    dtype=object,
)


def _encode_population_lies(values: np.ndarray) -> np.ndarray:
    """Encode lies after applying full-chain's exact normalisation policy."""

    normalized = batched_population.normalize_population_lies(np.asarray(values))
    result = np.full(normalized.shape, LIE_UNKNOWN, dtype=np.int8)
    for code, label in (
        (LIE_HOLE, "hole"),
        (LIE_TEE, "tee"),
        (LIE_FAIRWAY, "fairway"),
        (LIE_ROUGH, "rough"),
        (LIE_SAND, "sand"),
        (LIE_RECOVERY, "recovery"),
        (LIE_GREEN, "green"),
    ):
        result[normalized == label] = code
    return result


def _state_categories(
    lie: np.ndarray,
    remaining_yd: np.ndarray,
    active: np.ndarray,
    par: int,
) -> np.ndarray:
    """Vectorised equivalent of ``full._state_category``."""

    categories = np.full(len(lie), CATEGORY_INACTIVE, dtype=np.int8)
    positions = np.asarray(active, bool)
    categories[positions & (lie == LIE_HOLE)] = CATEGORY_TERMINAL
    categories[positions & (lie == LIE_GREEN)] = CATEGORY_PUTT
    tee_state = positions & (lie == LIE_TEE)
    categories[tee_state] = CATEGORY_APP if int(par) == 3 else CATEGORY_OTT
    unresolved = positions & (categories == CATEGORY_INACTIVE)
    invalid = unresolved & (
        (lie == LIE_UNKNOWN)
        | (lie == LIE_PENALTY)
        | ~np.isfinite(remaining_yd)
    )
    categories[invalid] = CATEGORY_INVALID
    playable = unresolved & ~invalid
    categories[playable & (remaining_yd <= 50.0)] = CATEGORY_ARG
    categories[playable & (remaining_yd > 50.0)] = CATEGORY_APP
    return categories


@dataclass
class HoleState:
    endpoint: np.ndarray
    lie: np.ndarray
    remaining_yd: np.ndarray
    strokes: np.ndarray
    terminal: np.ndarray
    invalid: np.ndarray
    status: np.ndarray
    hazard_count: np.ndarray
    penalty_strokes: np.ndarray
    transition_count: np.ndarray
    return_to_app_count: np.ndarray

    @classmethod
    def empty(cls, size: int) -> "HoleState":
        return cls(
            endpoint=np.full((int(size), 2), np.nan, dtype=float),
            lie=np.full(int(size), LIE_UNKNOWN, dtype=np.int8),
            remaining_yd=np.full(int(size), np.nan, dtype=float),
            # Wide internal counters prevent arithmetic wrap.  Exact int16 is
            # enforced only when a coherent terminal result is emitted.
            strokes=np.zeros(int(size), dtype=np.int32),
            terminal=np.zeros(int(size), dtype=bool),
            invalid=np.zeros(int(size), dtype=bool),
            status=np.full(int(size), STATUS_ACTIVE, dtype=np.int8),
            hazard_count=np.zeros(int(size), dtype=np.int32),
            penalty_strokes=np.zeros(int(size), dtype=np.int32),
            transition_count=np.zeros(int(size), dtype=np.int32),
            return_to_app_count=np.zeros(int(size), dtype=np.int32),
        )


@dataclass(frozen=True)
class StreamingHoleResult:
    particle_id: np.ndarray
    hole_strokes: np.ndarray
    terminal: np.ndarray
    status_code: np.ndarray
    hazard_transitions: np.ndarray
    penalty_strokes: np.ndarray
    transition_count: np.ndarray
    return_to_app_count: np.ndarray
    tee_atom_id: np.ndarray
    app_atom_id: np.ndarray
    initial_endpoint: np.ndarray
    initial_lie: np.ndarray
    initial_remaining_yd: np.ndarray
    initial_strokes: np.ndarray
    initial_status_code: np.ndarray
    final_endpoint: np.ndarray
    final_lie: np.ndarray
    final_remaining_yd: np.ndarray


@dataclass
class KernelDiagnostics:
    kernel_seconds: float = 0.0
    surface_seconds: float = 0.0
    support_select_seconds: float = 0.0
    population_seconds: float = 0.0
    app_state_groups: int = 0
    tee_draws: int = 0
    app_draws: int = 0
    arg_draws: int = 0
    putt_draws: int = 0
    spatial_hazard_draws: int = 0
    population_penalty_draws: int = 0
    unsupported_state_rows: int = 0
    max_live_chunk_bytes: int = 0

    def merge(self, other: "KernelDiagnostics") -> None:
        for name in self.__dataclass_fields__:
            if name == "max_live_chunk_bytes":
                setattr(self, name, max(getattr(self, name), getattr(other, name)))
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))


class StreamingShotChain:
    """One-hole numeric kernel over frozen spatial supports."""

    def __init__(
        self,
        *,
        repositories: Mapping[str, StreamingSupportRepository],
        course_drives: pd.DataFrame,
        course_app: pd.DataFrame,
        state_sampler: batched_population.BatchedPopulationTransitionSampler,
        hazard_resolver: BatchedEndpointHazardResolver,
        max_transitions: int | None = None,
        max_hole_strokes: int | None = None,
    ) -> None:
        if set(repositories).difference({"raw", "normalized"}):
            raise ValueError("Streaming arms must be raw and/or normalized")
        self.repositories = dict(repositories)
        self.tee_surfaces = {
            int(hole): tee.SurfaceMap(history)
            for hole, history in course_drives.groupby("hole_no", observed=True)
        }
        self.app_surfaces = spatial._surface_in_common_frame(course_app)
        self.state_sampler = state_sampler
        self.hazard_resolver = hazard_resolver
        if (
            max_transitions is not None
            and int(max_transitions) != tail_policy.MAX_CONTINUATION_TRANSITIONS
        ):
            raise ValueError(
                "max_transitions is fixed by the shared tail policy at "
                f"{tail_policy.MAX_CONTINUATION_TRANSITIONS}"
            )
        if max_hole_strokes is not None:
            raise ValueError(
                "Physical hole-score caps are forbidden; "
                "HIGH_SCORE_DIAGNOSTIC_STROKES is diagnostic only"
            )
        self.max_transitions = tail_policy.MAX_CONTINUATION_TRANSITIONS
        self.max_hole_strokes = None

    @staticmethod
    def _frame(record: object) -> full.CourseFrame:
        return full.CourseFrame(
            float(record.forecast_tee_x),
            float(record.forecast_tee_y),
            float(record.forecast_pin_x),
            float(record.forecast_pin_y),
        )

    def _hazard_support(
        self,
        endpoints: np.ndarray,
        *,
        record: object,
        category: str,
    ) -> np.ndarray:
        absolute = self._frame(record).local_yards_to_absolute_feet_many(endpoints)
        return np.asarray(
            self.hazard_resolver.scalar.has_support_many(
                absolute[:, 0],
                absolute[:, 1],
                hole_no=int(record.hole_no),
                round_no=int(record.round_no),
                category=str(category),
            ),
            bool,
        )

    def _sample_tee(
        self,
        *,
        state: HoleState,
        active: np.ndarray,
        record: object,
        repo: StreamingSupportRepository,
        index_uniform: np.ndarray,
        lie_uniform: np.ndarray,
        atom_id: np.ndarray,
        diagnostics: KernelDiagnostics,
    ) -> None:
        positions = np.flatnonzero(np.asarray(active, bool))
        if not len(positions):
            return
        surface = self.tee_surfaces.get(int(record.hole_no))
        if surface is None:
            state.invalid[positions] = True
            state.status[positions] = STATUS_NO_SURFACE_SUPPORT
            diagnostics.unsupported_state_rows += len(positions)
            return
        support = repo.tee(
            int(record.round_no), int(record.hole_no), int(record.par),
            str(record.source_player_id),
        )
        selected_at = time.perf_counter()
        selected, ids = support.select(np.asarray(index_uniform, float)[positions])
        endpoints = support.points[selected]
        diagnostics.support_select_seconds += time.perf_counter() - selected_at
        surface_at = time.perf_counter()
        probabilities = spatial._supported_surface_probabilities(
            surface.probabilities(endpoints),
            surface.codes,
            class_count=len(tee.LIE_CLASSES),
            enforce_empirical_class_support=True,
            endpoints=endpoints,
            hazard_class_indices=tuple(
                tee.LIE_CLASSES.index(name) for name in ("water", "other")
            ),
            hazard_support_checker=lambda points: self._hazard_support(
                points, record=record, category="ott"
            ),
        )
        lie_code = spatial._categorical_from_uniform(
            probabilities, np.asarray(lie_uniform, float)[positions]
        )
        diagnostics.surface_seconds += time.perf_counter() - surface_at
        mapped_lie = TEE_SURFACE_TO_LIE[lie_code]
        pin = np.asarray([float(record.forecast_pin_long_yd), 0.0])
        state.endpoint[positions] = endpoints
        state.remaining_yd[positions] = np.linalg.norm(
            pin[None, :] - endpoints, axis=1
        )
        state.lie[positions] = mapped_lie
        state.status[positions] = STATUS_ACTIVE
        state.status[positions[mapped_lie == LIE_PENALTY]] = STATUS_TEE_HAZARD
        state.strokes[positions] += 1
        atom_id[positions] = ids
        diagnostics.tee_draws += len(positions)

    def _sample_app(
        self,
        *,
        state: HoleState,
        active: np.ndarray,
        record: object,
        repo: StreamingSupportRepository,
        uniforms: Mapping[str, np.ndarray],
        atom_id: np.ndarray,
        diagnostics: KernelDiagnostics,
    ) -> None:
        active = np.asarray(active, bool).copy()
        active &= np.isfinite(state.endpoint).all(axis=1)
        active &= np.isfinite(state.remaining_yd)
        active &= np.isin(
            state.lie,
            np.asarray([LIE_TEE, LIE_FAIRWAY, LIE_ROUGH, LIE_SAND, LIE_RECOVERY]),
        )
        positions = np.flatnonzero(active)
        if not len(positions):
            return
        surface = self.app_surfaces.get(int(record.hole_no))
        if surface is None:
            state.invalid[positions] = True
            state.status[positions] = STATUS_NO_SURFACE_SUPPORT
            diagnostics.unsupported_state_rows += len(positions)
            return

        # State grouping is vectorised; Python work is bounded by the number
        # of unique lie x 15-yard buckets, never by particle count.
        buckets = np.maximum(
            1, np.rint(state.remaining_yd[positions] / 15.0).astype(np.int32)
        )
        encoded = state.lie[positions].astype(np.int64) * 100_000 + buckets
        unique, inverse = np.unique(encoded, return_inverse=True)
        diagnostics.app_state_groups += len(unique)
        pin = np.asarray([float(record.forecast_pin_long_yd), 0.0])
        for group_index, code in enumerate(unique):
            group_positions = positions[inverse == group_index]
            lie_code = int(code // 100_000)
            bucket = int(code % 100_000)
            start_lie = str(APP_START_NAMES[lie_code])
            try:
                support = repo.app(
                    int(record.round_no), int(record.hole_no), start_lie, bucket,
                    str(record.source_player_id),
                )
            except (KeyError, ValueError):
                state.invalid[group_positions] = True
                state.status[group_positions] = STATUS_NO_SPATIAL_SUPPORT
                diagnostics.unsupported_state_rows += len(group_positions)
                continue
            selected_at = time.perf_counter()
            selected, ids = support.select(
                np.asarray(uniforms["index"], float)[group_positions]
            )
            misses = support.points[selected]
            vector = pin[None, :] - state.endpoint[group_positions]
            length = np.maximum(np.linalg.norm(vector, axis=1), 1e-8)
            unit = vector / length[:, None]
            perpendicular = np.column_stack([-unit[:, 1], unit[:, 0]])
            endpoints = (
                pin[None, :]
                + unit * misses[:, 0, None]
                + perpendicular * misses[:, 1, None]
            )
            holed = (
                np.asarray(uniforms["holeout"], float)[group_positions]
                < support.holeout_probability
            )
            endpoints[holed] = pin
            diagnostics.support_select_seconds += time.perf_counter() - selected_at
            surface_at = time.perf_counter()
            probabilities = spatial._supported_surface_probabilities(
                surface.probabilities(endpoints),
                surface.codes,
                class_count=len(app.END_LIES),
                enforce_empirical_class_support=True,
                endpoints=endpoints,
                hazard_class_indices=(app.END_LIES.index("penalty"),),
                hazard_support_checker=lambda points: self._hazard_support(
                    points, record=record, category="app"
                ),
            )
            if holed.any():
                probabilities[holed] = 0.0
                probabilities[holed, app.END_LIES.index("hole")] = 1.0
            end_code = spatial._categorical_from_uniform(
                probabilities, np.asarray(uniforms["lie"], float)[group_positions]
            )
            diagnostics.surface_seconds += time.perf_counter() - surface_at
            mapped_lie = APP_SURFACE_TO_LIE[end_code]
            mapped_lie[holed] = LIE_HOLE
            geometric = np.linalg.norm(endpoints - pin[None, :], axis=1)
            state.endpoint[group_positions] = endpoints
            state.remaining_yd[group_positions] = geometric
            state.remaining_yd[group_positions[holed]] = 0.0
            state.lie[group_positions] = mapped_lie
            state.status[group_positions] = STATUS_ACTIVE
            penalty = mapped_lie == LIE_PENALTY
            state.status[group_positions[penalty]] = STATUS_APP_HAZARD
            state.terminal[group_positions[holed]] = True
            state.status[group_positions[holed]] = STATUS_TERMINAL
            state.strokes[group_positions] += 1
            atom_id[group_positions] = ids
            diagnostics.app_draws += len(group_positions)

    def _resolve_hazards(
        self,
        *,
        state: HoleState,
        active: np.ndarray,
        record: object,
        uniform: np.ndarray,
        diagnostics: KernelDiagnostics,
    ) -> None:
        frame = self._frame(record)
        for status_code, category in (
            (STATUS_TEE_HAZARD, "ott"),
            (STATUS_APP_HAZARD, "app"),
        ):
            positions = np.flatnonzero(active & (state.status == status_code))
            if not len(positions):
                continue
            absolute = frame.local_yards_to_absolute_feet_many(
                state.endpoint[positions]
            )
            resolved = self.hazard_resolver.resolve_many(
                endpoint_x=absolute[:, 0],
                endpoint_y=absolute[:, 1],
                round_no=int(record.round_no),
                hole_no=int(record.hole_no),
                category=category,
                common_uniform=np.asarray(uniform, float)[positions],
                forecast_pin_x=float(record.forecast_pin_x),
                forecast_pin_y=float(record.forecast_pin_y),
            )
            unavailable = ~resolved.available
            if unavailable.any():
                bad = positions[unavailable]
                state.invalid[bad] = True
                state.status[bad] = STATUS_HAZARD_UNAVAILABLE
                diagnostics.unsupported_state_rows += len(bad)
            good = positions[resolved.available]
            if not len(good):
                continue
            good_result = np.flatnonzero(resolved.available)
            normalized_lie = _encode_population_lies(
                resolved.resolution_lie[good_result]
            )
            unknown = normalized_lie == LIE_UNKNOWN
            if unknown.any():
                bad = good[unknown]
                state.invalid[bad] = True
                state.status[bad] = STATUS_HAZARD_UNKNOWN_LIE
                diagnostics.unsupported_state_rows += len(bad)
            accepted = ~unknown
            target = good[accepted]
            if not len(target):
                continue
            source = good_result[accepted]
            absolute_resolution = np.column_stack(
                [resolved.resolution_x[source], resolved.resolution_y[source]]
            )
            delta = absolute_resolution - np.asarray([frame.tee_x, frame.tee_y])
            local = np.column_stack(
                [delta @ frame.unit / 3.0, delta @ frame.perpendicular / 3.0]
            )
            geometric = np.hypot(
                absolute_resolution[:, 0] - float(record.forecast_pin_x),
                absolute_resolution[:, 1] - float(record.forecast_pin_y),
            ) / hazard.FEET_PER_YARD
            if not np.allclose(
                geometric, resolved.resolution_distance_yd[source],
                atol=1e-8, rtol=0.0,
            ):
                raise AssertionError("Batched hazard recursive distance is not geometric")
            cost = resolved.penalty_stroke_cost[source]
            if (cost < 1).any():
                raise AssertionError("Spatial hazard stroke cost must be positive")
            before = state.strokes[target].astype(np.int32)
            state.strokes[target] += cost
            if not np.array_equal(
                state.strokes[target].astype(np.int32) - before,
                cost.astype(np.int32),
            ):
                raise AssertionError("Batched hazard stroke-cost telescoping failed")
            state.penalty_strokes[target] += cost
            state.hazard_count[target] += 1
            state.endpoint[target] = local
            state.remaining_yd[target] = geometric
            state.lie[target] = normalized_lie[accepted]
            state.status[target] = STATUS_ACTIVE
            diagnostics.spatial_hazard_draws += len(target)

    def _population_transition(
        self,
        *,
        state: HoleState,
        active: np.ndarray,
        category: str,
        record: object,
        common_uniform: np.ndarray,
        diagnostics: KernelDiagnostics,
    ) -> None:
        positions = np.flatnonzero(active)
        if not len(positions):
            return
        sampled_at = time.perf_counter()
        result = self.state_sampler.sample_many(
            round_no=int(record.round_no),
            hole_no=int(record.hole_no),
            category=category,
            lie=LIE_NAMES[state.lie[positions]],
            distance_yd=state.remaining_yd[positions],
            common_uniform=np.asarray(common_uniform, float)[positions],
        )
        diagnostics.population_seconds += time.perf_counter() - sampled_at
        unavailable = ~result.available
        if unavailable.any():
            bad = positions[unavailable]
            state.invalid[bad] = True
            state.status[bad] = STATUS_POPULATION_UNAVAILABLE
            diagnostics.unsupported_state_rows += len(bad)
        good = positions[result.available]
        if not len(good):
            return
        selected = np.flatnonzero(result.available)
        cost = result.cost[selected]
        if (cost < 1).any():
            raise AssertionError("Population transition cost must include the shot")
        state.strokes[good] += cost
        penalty = cost > 1
        if penalty.any():
            penalty_positions = good[penalty]
            state.hazard_count[penalty_positions] += 1
            state.penalty_strokes[penalty_positions] += cost[penalty] - 1
            diagnostics.population_penalty_draws += int(penalty.sum())
        normalized = _encode_population_lies(result.next_lie[selected])
        holed = result.holed[selected] | (normalized == LIE_HOLE)
        if holed.any():
            target = good[holed]
            state.terminal[target] = True
            state.lie[target] = LIE_HOLE
            state.remaining_yd[target] = 0.0
            state.endpoint[target] = np.asarray(
                [float(record.forecast_pin_long_yd), 0.0]
            )
            state.status[target] = STATUS_TERMINAL
        continuing = ~holed
        target = good[continuing]
        if len(target):
            source = selected[continuing]
            next_distance = result.next_distance[source]
            next_lie = normalized[continuing]
            invalid = (next_lie == LIE_UNKNOWN) | ~np.isfinite(next_distance)
            if invalid.any():
                bad = target[invalid]
                state.invalid[bad] = True
                state.status[bad] = STATUS_POPULATION_INVALID
                diagnostics.unsupported_state_rows += len(bad)
            accepted = ~invalid
            accepted_target = target[accepted]
            if len(accepted_target):
                accepted_distance = np.maximum(next_distance[accepted], 0.0)
                pin = np.asarray([float(record.forecast_pin_long_yd), 0.0])
                vector = state.endpoint[accepted_target] - pin[None, :]
                length = np.linalg.norm(vector, axis=1)
                unit = np.zeros_like(vector)
                nonzero = np.isfinite(length) & (length > 1e-9)
                unit[nonzero] = vector[nonzero] / length[nonzero, None]
                unit[~nonzero, 0] = 1.0
                state.endpoint[accepted_target] = (
                    pin[None, :] + unit * accepted_distance[:, None]
                )
                state.remaining_yd[accepted_target] = accepted_distance
                state.lie[accepted_target] = next_lie[accepted]
                returned_app = _state_categories(
                    state.lie,
                    state.remaining_yd,
                    np.isin(np.arange(len(state.lie)), accepted_target),
                    int(record.par),
                ) == CATEGORY_APP
                state.return_to_app_count[returned_app] += 1
                state.status[accepted_target] = STATUS_ACTIVE
        if category == "arg":
            diagnostics.arg_draws += len(positions)
        else:
            diagnostics.putt_draws += len(positions)

    def simulate_hole(
        self,
        *,
        record: object,
        arm: str,
        particle_ids: np.ndarray,
        total_particles: int,
        rng_mode: str,
        seed: int = DEFAULT_SEED,
        continuation_seed: int = DEFAULT_CONTINUATION_SEED,
    ) -> tuple[StreamingHoleResult, KernelDiagnostics]:
        """Simulate one player/round/hole chunk without survivor filtering."""

        if arm not in self.repositories:
            raise KeyError(f"No support repository for arm {arm}")
        particle_ids = np.asarray(particle_ids, dtype=np.int64)
        size = len(particle_ids)
        state = HoleState.empty(size)
        tee_atom_id = np.full(size, "not_applicable_par3", dtype=object)
        app_atom_id = np.full(size, "not_attempted", dtype=object)
        diagnostics = KernelDiagnostics()
        repo = self.repositories[arm]
        factory = UniformFactory(
            mode=rng_mode,
            seed=int(seed),
            continuation_seed=int(continuation_seed),
            event_key=str(record.event_key),
            player_id=str(record.source_player_id),
            round_no=int(record.round_no),
            hole_no=int(record.hole_no),
            total_particles=int(total_particles),
        )
        started = time.perf_counter()
        if int(record.par) == 3:
            state.endpoint[:] = 0.0
            state.lie[:] = LIE_TEE
            state.remaining_yd[:] = float(record.forecast_pin_long_yd)
            self._sample_app(
                state=state,
                active=np.ones(size, dtype=bool),
                record=record,
                repo=repo,
                uniforms={
                    "index": factory.initial("app_index", particle_ids),
                    "holeout": factory.initial("app_holeout", particle_ids),
                    "lie": factory.initial("app_lie", particle_ids),
                },
                atom_id=app_atom_id,
                diagnostics=diagnostics,
            )
        else:
            self._sample_tee(
                state=state,
                active=np.ones(size, dtype=bool),
                record=record,
                repo=repo,
                index_uniform=factory.initial("tee_index", particle_ids),
                lie_uniform=factory.initial("tee_lie", particle_ids),
                atom_id=tee_atom_id,
                diagnostics=diagnostics,
            )
            playable = np.isin(
                state.lie,
                np.asarray([LIE_FAIRWAY, LIE_ROUGH, LIE_SAND, LIE_RECOVERY]),
            ) & np.isfinite(state.remaining_yd) & ~state.invalid
            self._sample_app(
                state=state,
                active=playable,
                record=record,
                repo=repo,
                uniforms={
                    "index": factory.initial("app_index", particle_ids),
                    "holeout": factory.initial("app_holeout", particle_ids),
                    "lie": factory.initial("app_lie", particle_ids),
                },
                atom_id=app_atom_id,
                diagnostics=diagnostics,
            )

        # Promotion parity audits the initializer selections. Recursive OTT/
        # APP draws may reuse these working arrays, so preserve the initial
        # atom identities before continuation starts.
        initial_tee_atom_id = tee_atom_id.copy()
        initial_app_atom_id = app_atom_id.copy()
        initial_endpoint = state.endpoint.copy()
        initial_lie = state.lie.copy()
        initial_remaining_yd = state.remaining_yd.copy()
        initial_strokes = state.strokes.copy()
        initial_status_code = state.status.copy()

        for transition_index in range(self.max_transitions):
            active = ~(state.terminal | state.invalid)
            if not active.any():
                break
            state.transition_count[active] += 1
            self._resolve_hazards(
                state=state,
                active=active,
                record=record,
                uniform=factory.transition(
                    transition_index, "hazard", particle_ids
                ),
                diagnostics=diagnostics,
            )
            active = ~(state.terminal | state.invalid)
            categories = _state_categories(
                state.lie, state.remaining_yd, active, int(record.par)
            )
            newly_terminal = active & (categories == CATEGORY_TERMINAL)
            state.terminal[newly_terminal] = True
            state.status[newly_terminal] = STATUS_TERMINAL
            invalid_state = active & (categories == CATEGORY_INVALID)
            state.invalid[invalid_state] = True
            state.status[invalid_state] = STATUS_INVALID_STATE

            ott = ~(state.terminal | state.invalid) & (categories == CATEGORY_OTT)
            self._sample_tee(
                state=state,
                active=ott,
                record=record,
                repo=repo,
                index_uniform=factory.transition(
                    transition_index, "tee_index", particle_ids
                ),
                lie_uniform=factory.transition(
                    transition_index, "tee_lie", particle_ids
                ),
                atom_id=tee_atom_id,
                diagnostics=diagnostics,
            )
            app_active = (
                ~(state.terminal | state.invalid) & (categories == CATEGORY_APP)
            )
            self._sample_app(
                state=state,
                active=app_active,
                record=record,
                repo=repo,
                uniforms={
                    "index": factory.transition(
                        transition_index, "app_index", particle_ids
                    ),
                    "holeout": factory.transition(
                        transition_index, "app_holeout", particle_ids
                    ),
                    "lie": factory.transition(
                        transition_index, "app_lie", particle_ids
                    ),
                },
                atom_id=app_atom_id,
                diagnostics=diagnostics,
            )
            for category_name, category_code in (
                ("arg", CATEGORY_ARG), ("putt", CATEGORY_PUTT)
            ):
                category_active = (
                    ~(state.terminal | state.invalid)
                    & (categories == category_code)
                )
                self._population_transition(
                    state=state,
                    active=category_active,
                    category=category_name,
                    record=record,
                    common_uniform=factory.transition(
                        transition_index, f"state_{category_name}", particle_ids
                    ),
                    diagnostics=diagnostics,
                )

        unresolved = ~(state.terminal | state.invalid)
        state.invalid[unresolved] = True
        state.status[unresolved] = STATUS_TRANSITION_LOOP
        if (state.status == STATUS_MAX_STROKES).any():
            raise AssertionError(
                "Current no-score-cap tail policy emitted a legacy cap status"
            )
        terminal = state.terminal & ~state.invalid
        hole_strokes = tail_policy.int16_exact(
            np.where(terminal, state.strokes, -1),
            label="streaming terminal hole strokes",
        )
        diagnostics.kernel_seconds += time.perf_counter() - started
        diagnostics.max_live_chunk_bytes = int(
            state.endpoint.nbytes
            + state.lie.nbytes
            + state.remaining_yd.nbytes
            + state.strokes.nbytes
            + state.terminal.nbytes
            + state.invalid.nbytes
            + state.status.nbytes
            + state.hazard_count.nbytes
            + state.penalty_strokes.nbytes
            + state.transition_count.nbytes
            + state.return_to_app_count.nbytes
        )
        return StreamingHoleResult(
            particle_id=particle_ids,
            hole_strokes=hole_strokes,
            terminal=terminal,
            status_code=state.status.copy(),
            hazard_transitions=tail_policy.int16_exact(
                state.hazard_count, label="streaming hazard counts"
            ),
            penalty_strokes=tail_policy.int16_exact(
                state.penalty_strokes, label="streaming penalty strokes"
            ),
            transition_count=tail_policy.int16_exact(
                state.transition_count, label="streaming transition counts"
            ),
            return_to_app_count=tail_policy.int16_exact(
                state.return_to_app_count,
                label="streaming return-to-APP counts",
            ),
            tee_atom_id=initial_tee_atom_id,
            app_atom_id=initial_app_atom_id,
            initial_endpoint=initial_endpoint,
            initial_lie=initial_lie,
            initial_remaining_yd=initial_remaining_yd,
            initial_strokes=tail_policy.int16_exact(
                initial_strokes, label="streaming initializer strokes"
            ),
            initial_status_code=initial_status_code,
            final_endpoint=state.endpoint.copy(),
            final_lie=state.lie.copy(),
            final_remaining_yd=state.remaining_yd.copy(),
        ), diagnostics


@dataclass
class PreparedStreamingEvent:
    atoms: precompute.NormalizedEventAtoms
    grid: pd.DataFrame
    roster_metadata: dict[str, object]
    scalar_hazard: hazard.EndpointLinkedHazardResolver
    batched_hazard: BatchedEndpointHazardResolver
    state_model: object
    state_rows: pd.DataFrame
    preparation_seconds: float
    preparation_rss_bytes: int
    preparation_peak_rss_bytes: int
    input_hashes: dict[str, str]
    logical_hashes: dict[str, str]
    input_paths: dict[str, Path]


def _load_reference_manifest(reference_dir: Path) -> dict[str, object]:
    reference_dir = Path(reference_dir)
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    for filename, expected_hash in FROZEN_REFERENCE_SHA256.items():
        path = reference_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Frozen reference {filename} hash mismatch: "
                f"{observed_hash} != {expected_hash}"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "artifact": FROZEN_REFERENCE_PRECOMPUTE_VERSION,
        "event": precompute.DEFAULT_EVENT,
        "holdout_year": precompute.DEFAULT_HOLDOUT,
        "particles": 128,
        "alignment_count": 1,
        "normalization_mode": precompute.DEFAULT_MODE,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"Reference manifest {name}={manifest.get(name)!r}; expected {value!r}"
            )
    validate_frozen_dependencies()
    if (
        manifest.get("code_sha256")
        != FROZEN_REFERENCE_PRECOMPUTE_PRODUCER_SHA256
    ):
        raise ValueError(
            "Frozen reference producer source no longer matches its manifest"
        )
    for filename, metadata_name in (
        ("raw_support_arrays.npz", "raw_support_npz"),
        ("normalized_support_arrays.npz", "normalized_support_npz"),
    ):
        observed = sha256_file(reference_dir / filename)
        expected_hash = str(manifest[metadata_name]["sha256"])
        if observed != expected_hash:
            raise ValueError(f"Frozen {filename} hash mismatch")
    return manifest


def prepare_streaming_event(
    *,
    export_root: Path,
    history_db: Path,
    profile_dir: Path,
    baseline_cases: Path,
    reference_dir: Path,
    event_name: str,
    holdout_year: int,
    course_start_year: int,
    roster_seed: int,
) -> PreparedStreamingEvent:
    """Build and verify all strict-prior inputs exactly once."""

    started = time.perf_counter()
    reference = _load_reference_manifest(reference_dir)
    # Rebuilding donor-native normalization peaks near 5-8 GiB and is not
    # necessary for streaming integration: the frozen promotion artifact
    # already persists the complete exactly-once-aligned neutral donor panels.
    # Reconstruct the provider contract from those compact panels plus current
    # verified raw course surfaces and the hash-validated profile snapshot.
    prepared_inputs = spatial.prepare_event_foundations(
        Path(export_root), str(event_name), int(holdout_year),
        int(course_start_year), include_scoring_targets=False,
    )
    full._validate_provenance(
        prepared_inputs["provenance"], int(holdout_year)
    )
    event_keys = dict(prepared_inputs["event_keys"])
    target_event_key = str(event_keys[int(holdout_year)])
    calendar = precompute._event_dates(
        Path(history_db), Path(export_root), event_keys
    )
    target_date = calendar[calendar["event_key"].astype(str).eq(target_event_key)]
    if len(target_date) != 1:
        raise ValueError("Target event date is absent or duplicated")
    profile_manifest_path = Path(profile_dir) / "manifest.json"
    observed_profile_manifest_hash = sha256_file(profile_manifest_path)
    if observed_profile_manifest_hash != str(reference["profile_manifest_sha256"]):
        raise ValueError(
            "Current strict-prior profile registry is not the frozen atom "
            f"registry: {observed_profile_manifest_hash} != "
            f"{reference['profile_manifest_sha256']}"
        )
    tee_profiles, app_profiles, profile_manifest = precompute._load_profile_bundle(
        Path(profile_dir), event_keys
    )
    neutral_tee_path = Path(reference_dir) / "neutral_tee_atoms.parquet"
    neutral_app_path = Path(reference_dir) / "neutral_app_atoms.parquet"
    neutral_tee = pd.read_parquet(neutral_tee_path)
    neutral_app = pd.read_parquet(neutral_app_path)
    event_end = calendar[["event_key", "event_end"]].copy()
    for label, panel in (("tee", neutral_tee), ("app", neutral_app)):
        if panel["shot_key"].duplicated().any():
            raise ValueError(f"Frozen neutral {label} atom IDs are not unique")
        if not panel["alignment_count"].eq(1).all():
            raise ValueError(f"Frozen neutral {label} atoms are not aligned once")
        if set(panel["event_key"].astype(str)) - set(event_keys.values()):
            raise ValueError(f"Frozen neutral {label} panel contains another event")
    neutral_tee = neutral_tee.merge(
        event_end, on="event_key", how="left", validate="many_to_one"
    )
    neutral_app = neutral_app.merge(
        event_end, on="event_key", how="left", validate="many_to_one"
    )
    if neutral_tee["event_end"].isna().any() or neutral_app["event_end"].isna().any():
        raise ValueError("Frozen neutral atoms lack strict-prior completion dates")
    expected_raw = reference["raw_alignment_equivalence"]
    if len(neutral_tee) != int(expected_raw["tee_rows"]):
        raise ValueError("Frozen neutral TEE row count differs from parity manifest")
    if len(neutral_app) != int(expected_raw["app_spatial_rows"]):
        raise ValueError("Frozen neutral APP row count differs from parity manifest")
    physical_course_id = precompute._event_course_id(
        Path(export_root), event_keys,
        int(prepared_inputs["provenance"]["setup_reference_year"]),
    )
    if physical_course_id != str(reference["physical_course_id"]):
        raise ValueError("Frozen neutral panels target another physical course")
    atoms = precompute.NormalizedEventAtoms(
        prepared=prepared_inputs,
        calendar=calendar,
        tee_profiles=tee_profiles,
        app_profiles=app_profiles,
        native_tee=pd.DataFrame(),
        native_app=pd.DataFrame(),
        neutral_tee=neutral_tee,
        neutral_app=neutral_app,
        normalization_metadata={
            "raw_alignment_equivalence": expected_raw,
            "app_normalization_exclusions": reference[
                "app_normalization_exclusions"
            ],
        },
        profile_manifest=profile_manifest,
        event_name=str(event_name),
        holdout_year=int(holdout_year),
        target_event_key=target_event_key,
        target_event_start=pd.Timestamp(target_date.iloc[0]["event_start"]),
        physical_course_id=physical_course_id,
        timings={
            "frozen_neutral_panel_reconstruction_seconds": float(
                time.perf_counter() - started
            )
        },
    )
    roster, roster_meta = full._load_pre_event_roster(
        Path(baseline_cases), str(event_name), 1, int(roster_seed)
    )
    grid = full._build_complete_grid(
        str(event_name),
        atoms.target_event_key,
        atoms.prepared["target_raw"],
        atoms.forecast,
        roster,
    )
    if len(grid) != 72 or grid["round_no"].nunique() != 4:
        raise ValueError("Streaming grid is not one player x four rounds x 18 holes")
    selected_players = grid["source_player_id"].astype(str).unique().tolist()
    if selected_players != [str(reference["player_id"])]:
        raise ValueError(
            f"Streaming roster {selected_players} != frozen player {reference['player_id']}"
        )
    if atoms.target_event_key != str(reference["event_key"]):
        raise ValueError("Current strict-prior target event differs from reference")
    scalar_hazard, _ = precompute._hazard_resolver(
        export_root=Path(export_root),
        event_name=str(event_name),
        holdout_year=int(holdout_year),
        course_start_year=int(course_start_year),
    )
    provenance = atoms.prepared["provenance"]
    if int(scalar_hazard.coordinate_reference_year) != int(
        provenance["setup_reference_year"]
    ):
        raise ValueError("Hazard and spatial reference years disagree")
    state_model, state_rows = full._course_population_model(
        Path(export_root), atoms.prepared["event_keys"], int(holdout_year)
    )
    input_paths = {
        "streaming_source": Path(__file__),
        **dependency_paths(),
        "normalization_profile_manifest": Path(profile_dir) / "manifest.json",
        "normalization_profile_tee": (
            Path(profile_dir) / "tee_player_spatial_profiles_asof.parquet"
        ),
        "normalization_profile_app": (
            Path(profile_dir) / "app_player_spatial_profiles_asof.parquet"
        ),
        "history_db": Path(history_db),
        "baseline_cases": Path(baseline_cases),
        "legacy_tee_profiles": Path(full.DEFAULT_TEE_PROFILES),
        "legacy_app_profiles": Path(full.DEFAULT_APP_PROFILES),
        "export_events": Path(export_root) / "events.parquet",
        "reference_manifest": Path(reference_dir) / "manifest.json",
        "reference_raw_support": Path(reference_dir) / "raw_support_arrays.npz",
        "reference_normalized_support": (
            Path(reference_dir) / "normalized_support_arrays.npz"
        ),
        "reference_neutral_tee": Path(reference_dir) / "neutral_tee_atoms.parquet",
        "reference_neutral_app": Path(reference_dir) / "neutral_app_atoms.parquet",
        "reference_legacy_terminal": (
            Path(reference_dir) / "legacy_terminal_scores.parquet"
        ),
        "reference_selected_atom_ids": (
            Path(reference_dir) / "selected_atom_ids.parquet"
        ),
    }
    for year in range(int(course_start_year), int(holdout_year) + 1):
        for stem in ("shots", "shot_sg", "hole_geometry"):
            input_paths[f"export_pga_{year}_{stem}"] = (
                Path(export_root) / f"pga_{year}_{stem}.parquet"
            )
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Streaming inputs are missing: {missing_inputs}")
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    logical_hashes = {
        "grid": frame_logical_sha256(grid),
        "forecast": frame_logical_sha256(atoms.forecast),
        "raw_course_drives": frame_logical_sha256(atoms.raw_course_drives),
        "raw_course_app": frame_logical_sha256(atoms.raw_course_app),
        "neutral_tee": frame_logical_sha256(atoms.neutral_tee),
        "neutral_app": frame_logical_sha256(atoms.neutral_app),
        "target_tee_profiles": frame_logical_sha256(
            tee_profiles[tee_profiles["event_key"].astype(str).eq(target_event_key)]
        ),
        "target_app_profiles": frame_logical_sha256(
            app_profiles[app_profiles["event_key"].astype(str).eq(target_event_key)]
        ),
        "hazard_donors": frame_logical_sha256(scalar_hazard.donors),
        "population_state_rows": frame_logical_sha256(state_rows),
        "event_keys": sha256(
            json.dumps(event_keys, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "provenance": sha256(
            json.dumps(
                prepared_inputs["provenance"], sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest(),
    }
    return PreparedStreamingEvent(
        atoms=atoms,
        grid=grid,
        roster_metadata=roster_meta,
        scalar_hazard=scalar_hazard,
        batched_hazard=BatchedEndpointHazardResolver(scalar_hazard),
        state_model=state_model,
        state_rows=state_rows,
        preparation_seconds=float(time.perf_counter() - started),
        preparation_rss_bytes=int(psutil.Process().memory_info().rss),
        preparation_peak_rss_bytes=int(psutil.Process().memory_info().rss),
        input_hashes=hashes,
        logical_hashes=logical_hashes,
        input_paths=input_paths,
    )


def audit_batched_hazard_oracle(
    prepared: PreparedStreamingEvent,
    *,
    rows_per_category: int = 24,
) -> dict[str, object]:
    """Compare vector hazard results with the scalar audited resolver."""

    scalar = prepared.scalar_hazard
    batch = prepared.batched_hazard
    comparisons = 0
    fallback_rows = 0
    for category in ("ott", "app"):
        category_rows = scalar.donors[
            scalar.donors["category"].astype(str).eq(category)
        ].sort_values("donor_key")
        if category_rows.empty:
            raise ValueError(f"Hazard oracle has no {category} rows")
        selected = category_rows.iloc[
            np.linspace(
                0, len(category_rows) - 1,
                min(int(rows_per_category), len(category_rows)), dtype=int,
            )
        ].copy()
        for (round_no, hole_no), group in selected.groupby(
            ["round_no", "hole_no"], observed=True, sort=True
        ):
            grid_row = prepared.grid[
                prepared.grid["round_no"].eq(int(round_no))
                & prepared.grid["hole_no"].eq(int(hole_no))
            ]
            if grid_row.empty:
                raise ValueError("Hazard donor setup is absent from full grid")
            record = grid_row.iloc[0]
            uniforms = (
                np.arange(len(group), dtype=float) + 0.5
            ) / max(len(group), 1)
            result = batch.resolve_many(
                endpoint_x=group["causal_endpoint_x"].to_numpy(float),
                endpoint_y=group["causal_endpoint_y"].to_numpy(float),
                round_no=int(round_no),
                hole_no=int(hole_no),
                category=category,
                common_uniform=uniforms,
                forecast_pin_x=float(record["forecast_pin_x"]),
                forecast_pin_y=float(record["forecast_pin_y"]),
            )
            if not result.available.all():
                raise AssertionError("Batched hazard oracle rejected a donor endpoint")
            for index, (_, donor) in enumerate(group.iterrows()):
                expected = scalar.resolve(
                    endpoint_x=float(donor["causal_endpoint_x"]),
                    endpoint_y=float(donor["causal_endpoint_y"]),
                    hole_no=int(hole_no),
                    round_no=int(round_no),
                    category=category,
                    common_uniform=float(uniforms[index]),
                    forecast_pin_x=float(record["forecast_pin_x"]),
                    forecast_pin_y=float(record["forecast_pin_y"]),
                    physical_course_id=scalar.physical_course_id,
                )
                exact_pairs = (
                    (result.donor_key[index], expected.donor_key),
                    (int(result.penalty_count[index]), expected.penalty_count),
                    (
                        int(result.penalty_stroke_cost[index]),
                        expected.penalty_stroke_cost,
                    ),
                    (result.resolution_lie[index], expected.resolution_lie),
                    (bool(result.fallback_used[index]), expected.fallback_used),
                )
                if any(left != right for left, right in exact_pairs):
                    raise AssertionError(
                        f"Batched hazard donor mismatch for {category} R{round_no} "
                        f"H{hole_no}: {exact_pairs}"
                    )
                if not np.array_equal(
                    np.asarray(
                        [
                            result.resolution_x[index], result.resolution_y[index],
                            result.resolution_distance_yd[index],
                        ]
                    ),
                    np.asarray(
                        [
                            expected.resolution_x, expected.resolution_y,
                            expected.resolution_distance_yd,
                        ]
                    ),
                ):
                    raise AssertionError("Batched hazard numeric state differs from scalar")
                comparisons += 1
                fallback_rows += int(expected.fallback_used)
        # Explicit right-CDF boundaries, including the scalar last-atom
        # clamp that production counter uniforms can hit.
        donor = category_rows.iloc[0]
        round_no, hole_no = int(donor["round_no"]), int(donor["hole_no"])
        record = prepared.grid[
            prepared.grid["round_no"].eq(round_no)
            & prepared.grid["hole_no"].eq(hole_no)
        ].iloc[0]
        candidates, weights, *_ = scalar._distribution(
            float(donor["causal_endpoint_x"]),
            float(donor["causal_endpoint_y"]),
            hole_no, round_no, category,
        )
        order = np.argsort(candidates["donor_key"].astype(str).to_numpy())
        cdf = np.cumsum(weights[order])
        boundary_uniforms = np.asarray(
            [0.0, float(cdf[0]), np.nextafter(1.0, 0.0)]
        )
        boundary_batch = batch.resolve_many(
            endpoint_x=np.full(3, float(donor["causal_endpoint_x"])),
            endpoint_y=np.full(3, float(donor["causal_endpoint_y"])),
            round_no=round_no, hole_no=hole_no, category=category,
            common_uniform=boundary_uniforms,
            forecast_pin_x=float(record["forecast_pin_x"]),
            forecast_pin_y=float(record["forecast_pin_y"]),
        )
        boundary_scalar = [
            scalar.resolve(
                endpoint_x=float(donor["causal_endpoint_x"]),
                endpoint_y=float(donor["causal_endpoint_y"]),
                round_no=round_no, hole_no=hole_no, category=category,
                common_uniform=float(uniform),
                forecast_pin_x=float(record["forecast_pin_x"]),
                forecast_pin_y=float(record["forecast_pin_y"]),
            ).donor_key
            for uniform in boundary_uniforms
        ]
        if not np.array_equal(boundary_batch.donor_key, boundary_scalar):
            raise AssertionError(f"Hazard right-CDF boundary mismatch: {category}")
        comparisons += 3

        # Find and require a genuine 120-150 yd same-hole fallback query.
        fallback_query: tuple[float, float, int, int] | None = None
        donor_sample = category_rows.iloc[
            np.linspace(0, len(category_rows) - 1, min(40, len(category_rows)), dtype=int)
        ]
        for candidate in donor_sample.itertuples(index=False):
            for radius in (121.0, 135.0, 149.0):
                for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                    query_x = float(candidate.causal_endpoint_x) + 3.0 * radius * math.cos(angle)
                    query_y = float(candidate.causal_endpoint_y) + 3.0 * radius * math.sin(angle)
                    support = scalar.support_diagnostics_many(
                        query_x, query_y, hole_no=int(candidate.hole_no),
                        round_no=int(candidate.round_no), category=category,
                    )
                    if bool(np.asarray(support.fallback_used).item()):
                        fallback_query = (
                            query_x, query_y,
                            int(candidate.round_no), int(candidate.hole_no),
                        )
                        break
                if fallback_query is not None:
                    break
            if fallback_query is not None:
                break
        if fallback_query is None:
            raise AssertionError(f"No real fallback hazard query found for {category}")
        query_x, query_y, round_no, hole_no = fallback_query
        record = prepared.grid[
            prepared.grid["round_no"].eq(round_no)
            & prepared.grid["hole_no"].eq(hole_no)
        ].iloc[0]
        expected_fallback = scalar.resolve(
            endpoint_x=query_x, endpoint_y=query_y, round_no=round_no,
            hole_no=hole_no, category=category, common_uniform=0.5,
            forecast_pin_x=float(record["forecast_pin_x"]),
            forecast_pin_y=float(record["forecast_pin_y"]),
        )
        observed_fallback = batch.resolve_many(
            endpoint_x=np.asarray([query_x]), endpoint_y=np.asarray([query_y]),
            round_no=round_no, hole_no=hole_no, category=category,
            common_uniform=np.asarray([0.5]),
            forecast_pin_x=float(record["forecast_pin_x"]),
            forecast_pin_y=float(record["forecast_pin_y"]),
        )
        if (
            not bool(observed_fallback.available[0])
            or not bool(observed_fallback.fallback_used[0])
            or observed_fallback.donor_key[0] != expected_fallback.donor_key
        ):
            raise AssertionError(f"Hazard fallback mismatch for {category}")
        comparisons += 1
        fallback_rows += 1

        # Both implementations must fail outside 150 yd and when the selected
        # resolution is over 700 yd from the supplied forecast pin.
        far_x = float(category_rows["causal_endpoint_x"].max()) + 10_000.0
        far_y = float(category_rows["causal_endpoint_y"].max()) + 10_000.0
        unavailable = batch.resolve_many(
            endpoint_x=np.asarray([far_x]), endpoint_y=np.asarray([far_y]),
            round_no=1, hole_no=1, category=category,
            common_uniform=np.asarray([0.5]), forecast_pin_x=0.0,
            forecast_pin_y=0.0,
        )
        if bool(unavailable.available[0]):
            raise AssertionError(f"Hazard >150yd query was available for {category}")
        with np.testing.assert_raises(hazard.HazardResolutionUnavailable):
            scalar.resolve(
                endpoint_x=far_x, endpoint_y=far_y, round_no=1, hole_no=1,
                category=category, common_uniform=0.5,
                forecast_pin_x=0.0, forecast_pin_y=0.0,
            )
        over_700 = batch.resolve_many(
            endpoint_x=np.asarray([float(donor["causal_endpoint_x"])]),
            endpoint_y=np.asarray([float(donor["causal_endpoint_y"])]),
            round_no=int(donor["round_no"]), hole_no=int(donor["hole_no"]),
            category=category, common_uniform=np.asarray([0.5]),
            forecast_pin_x=float(record["forecast_pin_x"]) + 10_000.0,
            forecast_pin_y=float(record["forecast_pin_y"]) + 10_000.0,
        )
        if bool(over_700.available[0]):
            raise AssertionError(f"Hazard >700yd resolution survived for {category}")
    return {
        "scalar_batch_exact_rows": int(comparisons),
        "fallback_rows": int(fallback_rows),
        "right_cdf_boundary_rows": 6,
        "unavailable_radius_rows": 2,
        "resolution_over_700_rows": 2,
        "categories": ["ott", "app"],
    }


def _result_frame(
    result: StreamingHoleResult,
    *,
    record: object,
    arm: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": int(record.case_id),
            "round_num": int(record.round_no),
            "hole_no": int(record.hole_no),
            "arm": str(arm),
            "particle_id": result.particle_id,
            "hole_strokes": result.hole_strokes,
            "high_score_diagnostic": tail_policy.high_score_diagnostic_mask(
                result.hole_strokes
            ),
            "terminal": result.terminal,
            "chain_status": STATUS_NAMES[result.status_code],
            "hazard_involved": result.hazard_transitions > 0,
            "hazard_transitions": result.hazard_transitions,
            "penalty_strokes": result.penalty_strokes,
            "transition_count": result.transition_count,
            "return_to_app_radial_projections": result.return_to_app_count,
            "tee_atom_id": result.tee_atom_id,
            "app_atom_id": result.app_atom_id,
            "initial_end_long_yd": result.initial_endpoint[:, 0],
            "initial_end_lat_yd": result.initial_endpoint[:, 1],
            "initial_end_lie": LIE_NAMES[result.initial_lie],
            "initial_remaining_yd": result.initial_remaining_yd,
            "initial_strokes": result.initial_strokes,
            "initial_status": STATUS_NAMES[result.initial_status_code],
            "final_end_long_yd": result.final_endpoint[:, 0],
            "final_end_lat_yd": result.final_endpoint[:, 1],
            "final_lie": LIE_NAMES[result.final_lie],
            "final_remaining_yd": result.final_remaining_yd,
        }
    )


def _simulate_grid(
    engine: StreamingShotChain,
    grid: pd.DataFrame,
    *,
    arms: Iterable[str],
    particles: int,
    chunk_size: int,
    rng_mode: str,
    collect_holes: bool,
    rss_monitor: RssPeakMonitor | None = None,
    rss_cap_bytes: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    if int(particles) <= 0 or int(chunk_size) <= 0:
        raise ValueError("Particle and chunk counts must be positive")
    arms = tuple(map(str, arms))
    round_totals = {
        arm: np.zeros((4, int(particles)), dtype=np.int32) for arm in arms
    }
    aggregate = {arm: KernelDiagnostics() for arm in arms}
    hole_parts: list[pd.DataFrame] = []
    hole_rows: list[dict[str, object]] = []
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    ordered = grid.sort_values(["round_no", "hole_no"])
    for record in ordered.itertuples(index=False):
        for arm in arms:
            summary = {
                "arm": arm,
                "round_no": int(record.round_no),
                "hole_no": int(record.hole_no),
                "particles": 0,
                "stroke_sum": 0,
                "stroke_sum_sq": 0,
                "hazard_particles": 0,
                "hazard_transitions": 0,
                "penalty_strokes": 0,
                "high_score_diagnostic_particles": 0,
                "max_transition_count": 0,
                "unsupported_state_rows": 0,
            }
            for start in range(0, int(particles), int(chunk_size)):
                stop = min(start + int(chunk_size), int(particles))
                ids = np.arange(start, stop, dtype=np.int64)
                result, diagnostics = engine.simulate_hole(
                    record=record,
                    arm=arm,
                    particle_ids=ids,
                    total_particles=int(particles),
                    rng_mode=str(rng_mode),
                )
                aggregate[arm].merge(diagnostics)
                if not result.terminal.all():
                    counts = pd.Series(
                        STATUS_NAMES[result.status_code[~result.terminal]]
                    ).value_counts().to_dict()
                    aggregate[arm].unsupported_state_rows += int(
                        (~result.terminal).sum()
                    )
                    raise full.ChainIncomplete(
                        f"Survivor filtering forbidden: {arm} R{record.round_no} "
                        f"H{record.hole_no} incomplete={counts}"
                    )
                round_index = int(record.round_no) - 1
                round_totals[arm][round_index, ids] = (
                    tail_policy.add_hole_strokes_to_round(
                        round_totals[arm][round_index, ids],
                        result.hole_strokes,
                        label=(
                            f"{arm} R{record.round_no} H{record.hole_no} "
                            "round accumulation"
                        ),
                    )
                )
                summary["particles"] += len(ids)
                summary["stroke_sum"] += int(result.hole_strokes.sum())
                summary["stroke_sum_sq"] += int(
                    np.square(result.hole_strokes.astype(np.int32)).sum()
                )
                summary["hazard_particles"] += int(
                    (result.hazard_transitions > 0).sum()
                )
                summary["hazard_transitions"] += int(
                    result.hazard_transitions.sum()
                )
                summary["penalty_strokes"] += int(result.penalty_strokes.sum())
                summary["high_score_diagnostic_particles"] += int(
                    tail_policy.high_score_diagnostic_mask(
                        result.hole_strokes
                    ).sum()
                )
                summary["max_transition_count"] = max(
                    int(summary["max_transition_count"]),
                    int(result.transition_count.max(initial=0)),
                )
                summary["unsupported_state_rows"] += int(
                    diagnostics.unsupported_state_rows
                )
                if collect_holes:
                    hole_parts.append(
                        _result_frame(result, record=record, arm=arm)
                    )
                peak_rss = max(peak_rss, int(process.memory_info().rss))
                if rss_monitor is not None and rss_cap_bytes is not None:
                    rss_monitor.assert_below(
                        int(rss_cap_bytes), "streaming particle chunk"
                    )
            summary["mean_hole_strokes"] = (
                float(summary["stroke_sum"]) / int(summary["particles"])
            )
            summary["variance_hole_strokes"] = max(
                0.0,
                float(summary["stroke_sum_sq"]) / int(summary["particles"])
                - float(summary["mean_hole_strokes"]) ** 2,
            )
            summary["hazard_hole_rate"] = (
                float(summary["hazard_particles"]) / int(summary["particles"])
            )
            hole_rows.append(summary)
    holes_per_round = (
        grid[["round_no", "hole_no"]]
        .drop_duplicates()
        .groupby("round_no", observed=True)["hole_no"]
        .nunique()
    )
    if not holes_per_round.eq(18).all():
        raise full.ChainIncomplete(
            "Round total emission requires exactly 18 distinct holes per "
            f"represented round: {holes_per_round.to_dict()}"
        )
    represented_rounds = set(holes_per_round.index.astype(int))
    for arm in arms:
        for round_no in represented_rounds:
            round_totals[arm][round_no - 1] = tail_policy.round_totals_int16(
                round_totals[arm][round_no - 1],
                label=f"{arm} R{round_no} round totals",
            )
        round_totals[arm] = tail_policy.int16_exact(
            round_totals[arm], label=f"{arm} persisted round totals"
        )
    wall = float(time.perf_counter() - started)
    holes = (
        pd.concat(hole_parts, ignore_index=True)
        if hole_parts else pd.DataFrame()
    )
    if rss_monitor is not None:
        peak_rss = max(peak_rss, int(rss_monitor.peak_bytes))
    diagnostics = {
        "wall_seconds": wall,
        "peak_rss_bytes": int(peak_rss),
        "arms": {
            arm: {
                name: getattr(values, name)
                for name in KernelDiagnostics.__dataclass_fields__
            }
            for arm, values in aggregate.items()
        },
        "hole_summary": hole_rows,
    }
    return round_totals, holes, diagnostics


def run_known_tail_regression(
    engine: StreamingShotChain,
    prepared: PreparedStreamingEvent,
) -> dict[str, object]:
    """Replay the three bounded Travelers tails exposed by the failed 50k."""

    selected = prepared.grid[
        prepared.grid["round_no"].astype(int).eq(KNOWN_TAIL_REGRESSION_ROUND)
        & prepared.grid["hole_no"].astype(int).eq(KNOWN_TAIL_REGRESSION_HOLE)
    ]
    if len(selected) != 1:
        raise AssertionError(
            "Known tail regression requires one Travelers R3 H13 record"
        )
    record = next(selected.itertuples(index=False))
    particle_ids = np.asarray(
        sorted(KNOWN_TAIL_REGRESSION), dtype=np.int64
    )
    together, _ = engine.simulate_hole(
        record=record,
        arm="raw",
        particle_ids=particle_ids,
        total_particles=50_000,
        rng_mode="counter",
    )
    singleton_parts: list[StreamingHoleResult] = []
    for particle_id in particle_ids:
        result, _ = engine.simulate_hole(
            record=record,
            arm="raw",
            particle_ids=np.asarray([particle_id], dtype=np.int64),
            total_particles=50_000,
            rng_mode="counter",
        )
        singleton_parts.append(result)
    expected = np.asarray(
        [KNOWN_TAIL_REGRESSION[int(value)] for value in particle_ids],
        dtype=np.int16,
    )
    singleton_scores = np.concatenate(
        [result.hole_strokes for result in singleton_parts]
    )
    singleton_status = np.concatenate(
        [result.status_code for result in singleton_parts]
    )
    singleton_transitions = np.concatenate(
        [result.transition_count for result in singleton_parts]
    )
    if not together.terminal.all() or not all(
        result.terminal.all() for result in singleton_parts
    ):
        raise AssertionError("Known 50k tail particles did not all hole")
    if not np.array_equal(together.hole_strokes, expected):
        raise AssertionError(
            "Known 50k tail scores changed: "
            f"{dict(zip(particle_ids.tolist(), together.hole_strokes.tolist()))}"
        )
    if not np.array_equal(singleton_scores, together.hole_strokes):
        raise AssertionError("Known tail scores changed across chunk boundaries")
    if not np.array_equal(singleton_status, together.status_code):
        raise AssertionError("Known tail statuses changed across chunk boundaries")
    if not np.array_equal(singleton_transitions, together.transition_count):
        raise AssertionError(
            "Known tail transition counts changed across chunk boundaries"
        )
    if (together.status_code == STATUS_MAX_STROKES).any():
        raise AssertionError("Known tails emitted the legacy score-cap status")
    if int(together.transition_count.max()) > int(
        tail_policy.MAX_CONTINUATION_TRANSITIONS
    ):
        raise AssertionError("Known tails exceeded the transition guard")
    return {
        "exact": True,
        "scope": "raw Travelers 2026 R3 H13 bounded three-particle replay",
        "total_particle_lattice": 50_000,
        "particle_ids": particle_ids.tolist(),
        "expected_hole_strokes": expected.tolist(),
        "observed_hole_strokes": together.hole_strokes.tolist(),
        "transition_counts": together.transition_count.tolist(),
        "hazard_transitions": together.hazard_transitions.tolist(),
        "penalty_strokes": together.penalty_strokes.tolist(),
        "singleton_chunk_exact": True,
        "legacy_score_cap_status_emitted": False,
        "tail_policy": tail_policy.manifest_policy(),
    }


def _assert_frame_columns_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> dict[str, object]:
    keys = ["case_id", "particle_id"]
    joined = left[keys + list(columns)].merge(
        right[keys + list(columns)],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise AssertionError(f"{label} particle roster differs")
    mismatches: dict[str, int] = {}
    for name in columns:
        candidate = joined[f"{name}_candidate"].to_numpy()
        reference = joined[f"{name}_reference"].to_numpy()
        if candidate.dtype.kind in "fc" or reference.dtype.kind in "fc":
            equal = np.isclose(candidate, reference, atol=0.0, rtol=0.0, equal_nan=True)
        else:
            equal = candidate == reference
        mismatches[name] = int((~equal).sum())
    if any(mismatches.values()):
        raise AssertionError(f"{label} mismatches: {mismatches}")
    return {"label": label, "rows": int(len(joined)), "mismatches": mismatches}


def _legacy_initializer_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the authoritative long initializer into recursion state."""

    out = frame.copy()
    tee_endpoint = out[["tee_end_long_yd", "tee_end_lat_yd"]].to_numpy(float)
    app_endpoint = out[["app_end_long_yd", "app_end_lat_yd"]].to_numpy(float)
    app_attempted = np.isfinite(app_endpoint).all(axis=1)
    endpoint = tee_endpoint.copy()
    endpoint[app_attempted] = app_endpoint[app_attempted]
    tee_lie = out["tee_end_lie"].astype(str).to_numpy(object)
    app_lie = out["app_end_lie"].astype(str).to_numpy(object)
    lie = tee_lie.copy()
    lie[app_attempted] = app_lie[app_attempted]
    normalized = np.full(len(out), "unknown", dtype=object)
    text = np.char.lower(np.char.strip(lie.astype(str)))
    for direct in ("hole", "tee", "fairway", "rough", "sand", "recovery", "green"):
        normalized[text == direct] = direct
    normalized[text == "other"] = "recovery"
    normalized[np.isin(text, ["water", "penalty", "hazard", "native area"])] = "penalty"
    # The legacy initializer stores the raw surface label and the hazard state
    # separately.  In particular, a causal OTT ``other`` draw is recorded as
    # ``tee_hazard_unresolved`` even though ordinary ``other`` lies normalize
    # to recovery.  Streaming uses a penalty sentinel to route that same
    # particle through endpoint-linked resolution, so preserve the status
    # meaning in this parity-only canonical projection.
    chain_status = out["chain_status"].astype(str).str.strip().str.casefold()
    unresolved_hazard = chain_status.isin(
        ["tee_hazard_unresolved", "app_hazard_unresolved"]
    ).to_numpy(bool)
    normalized[unresolved_hazard] = "penalty"
    remaining = pd.to_numeric(out["tee_remaining_yd"], errors="coerce").to_numpy(float)
    app_remaining = pd.to_numeric(
        out["app_remaining_yd"], errors="coerce"
    ).to_numpy(float)
    app_geometric = pd.to_numeric(
        out["app_geometric_remaining_yd"], errors="coerce"
    ).to_numpy(float)
    app_state = np.where(np.isfinite(app_remaining), app_remaining, app_geometric)
    remaining[app_attempted] = app_state[app_attempted]
    return pd.DataFrame(
        {
            "case_id": out["case_id"].to_numpy(int),
            "particle_id": out["particle_id"].to_numpy(int),
            "initial_end_long_yd": endpoint[:, 0],
            "initial_end_lat_yd": endpoint[:, 1],
            "initial_end_lie": normalized,
            "initial_remaining_yd": remaining,
            "initial_strokes": pd.to_numeric(
                out["shots_taken"], errors="raise"
            ).to_numpy(np.int16),
        }
    )


def run_raw_128_gate(
    prepared: PreparedStreamingEvent,
    *,
    reference_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Require exact cached-ID and terminal parity with the frozen artifact."""

    repo = StreamingSupportRepository(
        arm="raw",
        artifact_npz=Path(reference_dir) / "raw_support_arrays.npz",
        fail_on_miss=True,
        keep_atom_ids=True,
    )
    state_sampler = batched_population.BatchedPopulationTransitionSampler(
        prepared.state_model, chunk_size=128
    )
    engine = StreamingShotChain(
        repositories={"raw": repo},
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        state_sampler=state_sampler,
        hazard_resolver=prepared.batched_hazard,
    )
    r1_grid = prepared.grid[prepared.grid["round_no"].eq(1)].copy().reset_index(drop=True)
    r1_grid["case_id"] = np.arange(len(r1_grid), dtype=int)
    _, candidate, legacy_diagnostics = _simulate_grid(
        engine,
        r1_grid,
        arms=("raw",),
        particles=128,
        chunk_size=128,
        rng_mode="legacy128",
        collect_holes=True,
    )
    legacy_tee_profiles = pd.read_csv(
        full.DEFAULT_TEE_PROFILES, dtype={"source_player_id": str}
    )
    legacy_app_profiles = pd.read_csv(
        full.DEFAULT_APP_PROFILES, dtype={"source_player_id": str}
    )
    legacy_initializer = spatial.simulate_forecast_grid(
        r1_grid,
        prepared.atoms.raw_course_drives,
        prepared.atoms.raw_course_app,
        legacy_tee_profiles,
        legacy_app_profiles,
        128,
        DEFAULT_SEED,
        enforce_empirical_class_support=True,
        hazard_support_factory=full._forecast_hazard_support_factory(
            prepared.scalar_hazard
        ),
    )
    legacy_initializer = legacy_initializer[
        legacy_initializer["arm"].eq("population")
    ].copy().reset_index(drop=True)
    initializer_state = _assert_frame_columns_equal(
        candidate,
        _legacy_initializer_state(legacy_initializer),
        [
            "initial_end_long_yd", "initial_end_lat_yd", "initial_end_lie",
            "initial_remaining_yd", "initial_strokes",
        ],
        label="legacy128_r1_initializer_physical_state_exact",
    )
    reference_terminal = pd.read_parquet(
        Path(reference_dir) / "legacy_terminal_scores.parquet"
    )
    reference_terminal = reference_terminal[
        reference_terminal["arm"].eq("population")
    ].copy()
    terminal_columns = [
        "hole_strokes", "terminal", "chain_status", "hazard_involved",
        "hazard_transitions", "penalty_strokes", "transition_count",
        "return_to_app_radial_projections",
        "final_lie", "final_remaining_yd",
    ]
    terminal = _assert_frame_columns_equal(
        candidate, reference_terminal, terminal_columns,
        label="legacy128_terminal_exact",
    )
    reference_ids = pd.read_parquet(
        Path(reference_dir) / "selected_atom_ids.parquet"
    ).rename(
        columns={
            "tee_atom_id_cached": "tee_atom_id",
            "app_atom_id_cached": "app_atom_id",
        }
    )
    atom_ids = _assert_frame_columns_equal(
        candidate,
        reference_ids,
        ["tee_atom_id", "app_atom_id"],
        label="legacy128_initializer_atom_ids_exact",
    )

    # Counter RNG reaches a different state inventory than legacy128. Build
    # those extra supports from the same strict-prior atoms during the first
    # audit replay, then require the second chunking replay to add no supports.
    # The frozen fail-on-miss repository above remains exclusive to legacy
    # promotion parity.
    counter_repositories: dict[str, StreamingSupportRepository] = {}
    for counter_arm, artifact_name in (
        ("raw", "raw_support_arrays.npz"),
        ("normalized", "normalized_support_arrays.npz"),
    ):
        counter_repositories[counter_arm] = StreamingSupportRepository(
            arm=counter_arm,
            artifact_npz=Path(reference_dir) / artifact_name,
            provider=precompute.AtomSupportProvider(
                prepared.atoms, arm=counter_arm, cache_enabled=True
            ),
            fail_on_miss=False,
            keep_atom_ids=True,
        )
    counter_engine = StreamingShotChain(
        repositories=counter_repositories,
        course_drives=prepared.atoms.raw_course_drives,
        course_app=prepared.atoms.raw_course_app,
        state_sampler=batched_population.BatchedPopulationTransitionSampler(
            prepared.state_model, chunk_size=128
        ),
        hazard_resolver=prepared.batched_hazard,
    )
    # Counter mode must be exactly invariant across both arms, all four
    # rounds, all 18 holes, and alternate chunk boundaries.
    _, counter_one, _ = _simulate_grid(
        counter_engine, prepared.grid, arms=("raw", "normalized"),
        particles=128, chunk_size=128, rng_mode="counter", collect_holes=True,
    )
    tail_regression = run_known_tail_regression(counter_engine, prepared)
    calls_after_discovery = {
        arm: repository.materialization_calls
        for arm, repository in counter_repositories.items()
    }
    misses_after_discovery = {
        arm: repository.miss_count
        for arm, repository in counter_repositories.items()
    }
    _, counter_many, _ = _simulate_grid(
        counter_engine, prepared.grid, arms=("raw", "normalized"),
        particles=128, chunk_size=17, rng_mode="counter", collect_holes=True,
    )
    replay_new_calls = {
        arm: repository.materialization_calls - calls_after_discovery[arm]
        for arm, repository in counter_repositories.items()
    }
    replay_new_misses = {
        arm: repository.miss_count - misses_after_discovery[arm]
        for arm, repository in counter_repositories.items()
    }
    if any(replay_new_calls.values()) or any(replay_new_misses.values()):
        raise AssertionError(
            "Counter chunk replay changed strict-prior support inventory: "
            f"calls={replay_new_calls}, misses={replay_new_misses}"
        )
    chunk_columns = terminal_columns + [
        "tee_atom_id", "app_atom_id", "initial_end_long_yd",
        "initial_end_lat_yd", "initial_end_lie", "initial_remaining_yd",
        "initial_strokes", "final_end_long_yd", "final_end_lat_yd",
    ]
    chunk = {
        arm: _assert_frame_columns_equal(
            counter_one[counter_one["arm"].eq(arm)],
            counter_many[counter_many["arm"].eq(arm)],
            chunk_columns,
            label=f"counter_{arm}_r1_r4_chunk_128_vs_17_exact",
        )
        for arm in ("raw", "normalized")
    }

    # Exercise the stated 50k-prefix-of-100k contract for every semantic RNG
    # family used by the kernel.  This is cheap and exact; terminal chunk
    # invariance above proves state routing is particle-independent.
    sample_record = r1_grid.iloc[0]
    prefix_factory = UniformFactory(
        mode="counter", seed=DEFAULT_SEED,
        continuation_seed=DEFAULT_CONTINUATION_SEED,
        event_key=str(sample_record["event_key"]),
        player_id=str(sample_record["source_player_id"]),
        round_no=int(sample_record["round_no"]),
        hole_no=int(sample_record["hole_no"]), total_particles=50_000,
    )
    wide_factory = UniformFactory(
        mode="counter", seed=DEFAULT_SEED,
        continuation_seed=DEFAULT_CONTINUATION_SEED,
        event_key=str(sample_record["event_key"]),
        player_id=str(sample_record["source_player_id"]),
        round_no=int(sample_record["round_no"]),
        hole_no=int(sample_record["hole_no"]), total_particles=100_000,
    )
    ids_50 = np.arange(50_000, dtype=np.int64)
    ids_100 = np.arange(100_000, dtype=np.int64)
    rng_streams = [
        ("initial", "tee_index"), ("initial", "tee_lie"),
        ("initial", "app_index"), ("initial", "app_holeout"),
        ("initial", "app_lie"), ("transition", "hazard"),
        ("transition", "tee_index"), ("transition", "tee_lie"),
        ("transition", "app_index"), ("transition", "app_holeout"),
        ("transition", "app_lie"), ("transition", "state_arg"),
        ("transition", "state_putt"),
    ]
    prefix_hashes: dict[str, str] = {}
    for family, stream in rng_streams:
        if family == "initial":
            narrow = prefix_factory.initial(stream, ids_50)
            wide = wide_factory.initial(stream, ids_100)[:50_000]
            if not np.array_equal(narrow, wide):
                raise AssertionError(
                    f"Counter RNG prefix failed for {family}:{stream}"
                )
        else:
            transition_values: list[np.ndarray] = []
            for transition_index in range(
                tail_policy.MAX_CONTINUATION_TRANSITIONS
            ):
                narrow_at = prefix_factory.transition(
                    transition_index, stream, ids_50
                )
                wide_at = wide_factory.transition(
                    transition_index, stream, ids_100
                )[:50_000]
                if not np.array_equal(narrow_at, wide_at):
                    raise AssertionError(
                        "Counter RNG prefix failed for "
                        f"{family}:{stream} at transition {transition_index}"
                    )
                transition_values.append(narrow_at)
            narrow = np.stack(transition_values)
        prefix_hashes[f"{family}:{stream}"] = array_sha256(narrow)
    audit = {
        "legacy_terminal": terminal,
        "legacy_initializer_atom_ids": atom_ids,
        "legacy_initializer_physical_state": initializer_state,
        "counter_chunk_invariance": chunk,
        "counter_50k_prefix_of_100k": {
            "exact": True,
            "transition_indices": list(
                range(tail_policy.MAX_CONTINUATION_TRANSITIONS)
            ),
            "max_transition_index": (
                tail_policy.MAX_CONTINUATION_TRANSITIONS - 1
            ),
            "streams": prefix_hashes,
        },
        "legacy_diagnostics": legacy_diagnostics,
        "raw_support_cache_bytes": repo.cache_nbytes,
        "counter_support_inventory": {
            arm: {
                "supports": int(len(repository.supports)),
                "provider_materialization_calls": int(
                    repository.materialization_calls
                ),
                "provider_materialized_atoms": int(repository.materialized_atoms),
                "provider_materialization_seconds": float(
                    repository.materialization_seconds
                ),
                "second_replay_new_calls": int(replay_new_calls[arm]),
                "second_replay_new_misses": int(replay_new_misses[arm]),
                "inventory_sha256": array_sha256(
                    np.asarray(sorted(repository.supports)),
                    np.asarray(
                        [
                            repository.supports[key].cache_sha256
                            for key in sorted(repository.supports)
                        ]
                    ),
                    np.asarray(
                        [
                            repository.supports[key].payload_sha256
                            for key in sorted(repository.supports)
                        ]
                    ),
                ),
            }
            for arm, repository in counter_repositories.items()
        },
        "unsupported_state_rows": 0,
        "survivor_filtering": False,
        "tail_policy": tail_policy.manifest_policy(),
        "known_tail_regression": tail_regression,
    }
    return audit, candidate


def _new_benchmark_repositories(
    prepared: PreparedStreamingEvent,
    reference_dir: Path,
) -> dict[str, StreamingSupportRepository]:
    raw_provider = precompute.AtomSupportProvider(
        prepared.atoms, arm="raw", cache_enabled=True
    )
    normalized_provider = precompute.AtomSupportProvider(
        prepared.atoms, arm="normalized", cache_enabled=True
    )
    return {
        "raw": StreamingSupportRepository(
            arm="raw",
            artifact_npz=Path(reference_dir) / "raw_support_arrays.npz",
            provider=raw_provider,
            keep_atom_ids=False,
        ),
        "normalized": StreamingSupportRepository(
            arm="normalized",
            artifact_npz=Path(reference_dir) / "normalized_support_arrays.npz",
            provider=normalized_provider,
            keep_atom_ids=False,
        ),
    }


def persist_numeric_support_repository(
    repository: StreamingSupportRepository,
    output_dir: Path,
    *,
    rss_monitor: RssPeakMonitor | None = None,
    rss_cap_bytes: int | None = None,
) -> dict[str, object]:
    """Persist one support at a time to bound serialization memory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for key in sorted(repository.supports):
        support = repository.supports[key]
        file_name = f"{sha256(key.encode()).hexdigest()}.npz"
        path = output_dir / file_name
        np.savez_compressed(
            path,
            atom_ids=support.atom_ids,
            points_yd=support.points,
            probability=support.probability,
            cdf=support.cdf,
            holeout_probability=np.asarray([support.holeout_probability]),
            source_cache_sha256=np.asarray([support.cache_sha256]),
            payload_sha256=np.asarray([support.payload_sha256]),
        )
        rows.append(
            {
                "key": key,
                "file": file_name,
                "atoms": int(len(support.atom_ids)),
                "source_cache_sha256": support.cache_sha256,
                "payload_sha256": support.payload_sha256,
                "file_sha256": sha256_file(path),
                "file_bytes": int(path.stat().st_size),
            }
        )
        if rss_monitor is not None and rss_cap_bytes is not None:
            rss_monitor.assert_below(
                int(rss_cap_bytes), "numeric support serialization"
            )
    inventory = pd.DataFrame(rows)
    inventory_path = output_dir / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    logical_hash = array_sha256(
        inventory["key"].to_numpy(),
        inventory["source_cache_sha256"].to_numpy(),
        inventory["payload_sha256"].to_numpy(),
        inventory["file_sha256"].to_numpy(),
    )
    payload = {
        "arm": repository.arm,
        "supports": int(len(inventory)),
        "atoms": int(inventory["atoms"].sum()),
        "serialized_bytes": int(inventory["file_bytes"].sum()),
        "inventory_sha256": sha256_file(inventory_path),
        "logical_support_sha256": logical_hash,
        "seconds": float(time.perf_counter() - started),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def release_heavy_precompute_payload(
    prepared: PreparedStreamingEvent,
    repositories: Mapping[str, StreamingSupportRepository],
) -> int:
    """Sever pandas provider frames after exact support discovery."""

    for repository in repositories.values():
        repository.provider = None
    atoms = prepared.atoms
    atoms.tee_profiles = pd.DataFrame()
    atoms.app_profiles = pd.DataFrame()
    atoms.neutral_tee = pd.DataFrame()
    atoms.neutral_app = pd.DataFrame()
    atoms.calendar = pd.DataFrame()
    atoms.prepared["course_drives"] = pd.DataFrame()
    atoms.prepared["course_app"] = pd.DataFrame()
    atoms.prepared["target_raw"] = pd.DataFrame()
    prepared.state_rows = pd.DataFrame()
    import gc

    gc.collect()
    return int(psutil.Process().memory_info().rss)


def run_5k_benchmark(
    prepared: PreparedStreamingEvent,
    *,
    reference_dir: Path,
    particles: int,
    chunk_size: int,
    cache_output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    if int(particles) > MAX_APPROVED_PARTICLES:
        raise ValueError(
            f"{particles} exceeds the approved benchmark ceiling "
            f"{MAX_APPROVED_PARTICLES}; 50k is intentionally disabled"
        )
    if prepared.preparation_peak_rss_bytes > SAFE_RSS_CAP_BYTES:
        raise MemoryError("Preparation exceeded the safe 4 GiB streaming cap")
    prewarm_monitor = RssPeakMonitor()
    prewarm_monitor.start()
    try:
        repositories = _new_benchmark_repositories(prepared, Path(reference_dir))
        prewarm_monitor.assert_below(SAFE_RSS_CAP_BYTES, "support repository load")
        state_sampler = batched_population.BatchedPopulationTransitionSampler(
            prepared.state_model, chunk_size=int(chunk_size)
        )
        engine = StreamingShotChain(
            repositories=repositories,
            course_drives=prepared.atoms.raw_course_drives,
            course_app=prepared.atoms.raw_course_app,
            state_sampler=state_sampler,
            hazard_resolver=prepared.batched_hazard,
        )
        # Discovery/prewarm is explicitly outside the timed benchmark. It
        # follows the exact deterministic paths once, materializing every
        # reached APP state. The timed pass must have zero provider misses.
        warm_started = time.perf_counter()
        _, _, warm_diagnostics = _simulate_grid(
            engine, prepared.grid, arms=("raw", "normalized"),
            particles=int(particles), chunk_size=int(chunk_size),
            rng_mode="counter", collect_holes=False,
            rss_monitor=prewarm_monitor, rss_cap_bytes=SAFE_RSS_CAP_BYTES,
        )
        warm_seconds = float(time.perf_counter() - warm_started)
        cache_manifests = {
            arm: persist_numeric_support_repository(
                repo, Path(cache_output_dir) / arm,
                rss_monitor=prewarm_monitor,
                rss_cap_bytes=SAFE_RSS_CAP_BYTES,
            )
            for arm, repo in repositories.items()
        }
        post_release_rss = release_heavy_precompute_payload(
            prepared, repositories
        )
        prewarm_monitor.assert_below(
            SAFE_RSS_CAP_BYTES, "post-materialization release"
        )
    finally:
        prewarm_global_peak = prewarm_monitor.stop()
    if prewarm_global_peak > SAFE_RSS_CAP_BYTES:
        raise MemoryError("Prewarm/serialization exceeded the safe 4 GiB cap")
    before_calls = {
        arm: repo.materialization_calls for arm, repo in repositories.items()
    }
    before_misses = {arm: repo.miss_count for arm, repo in repositories.items()}
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    timed_monitor = RssPeakMonitor()
    timed_monitor.start()
    try:
        totals, _, benchmark = _simulate_grid(
            engine, prepared.grid, arms=("raw", "normalized"),
            particles=int(particles), chunk_size=int(chunk_size),
            rng_mode="counter", collect_holes=False,
            rss_monitor=timed_monitor, rss_cap_bytes=SAFE_RSS_CAP_BYTES,
        )
    finally:
        timed_global_peak = timed_monitor.stop()
    if timed_global_peak > SAFE_RSS_CAP_BYTES:
        raise MemoryError("Timed kernel exceeded the safe 4 GiB cap")
    rss_after = int(process.memory_info().rss)
    timed_new_calls = {
        arm: repo.materialization_calls - before_calls[arm]
        for arm, repo in repositories.items()
    }
    timed_new_misses = {
        arm: repo.miss_count - before_misses[arm]
        for arm, repo in repositories.items()
    }
    if any(timed_new_calls.values()) or any(timed_new_misses.values()):
        raise AssertionError(
            "Timed benchmark entered pandas support materialization: "
            f"calls={timed_new_calls}, misses={timed_new_misses}"
        )
    if any(
        int(details["unsupported_state_rows"]) != 0
        for details in benchmark["arms"].values()
    ):
        raise AssertionError("Timed benchmark contains unsupported states")
    diagnostics = pd.DataFrame(benchmark.pop("hole_summary"))
    wall = float(benchmark["wall_seconds"])
    fixed_seconds = float(prepared.preparation_seconds + warm_seconds)
    projection_scale = 50_000 / int(particles)
    benchmark.update(
        {
            "particles": int(particles),
            "chunk_size": int(chunk_size),
            "arms_order": ["raw", "normalized"],
            "rng_mode": "counter",
            "tail_policy": tail_policy.manifest_policy(),
            "prewarm_seconds": warm_seconds,
            "preparation_end_rss_bytes": int(prepared.preparation_rss_bytes),
            "preparation_peak_rss_bytes": int(
                prepared.preparation_peak_rss_bytes
            ),
            "prewarm_peak_rss_bytes": int(prewarm_global_peak),
            "post_release_rss_bytes": int(post_release_rss),
            "timed_kernel_peak_rss_bytes": int(timed_global_peak),
            "global_peak_rss_bytes": int(
                max(
                    prepared.preparation_peak_rss_bytes,
                    prewarm_global_peak,
                    timed_global_peak,
                )
            ),
            "safe_rss_cap_bytes": int(SAFE_RSS_CAP_BYTES),
            "numeric_support_cache": cache_manifests,
            "prewarm_role": "support_discovery_outside_timed_kernel",
            "timed_materialization_calls": timed_new_calls,
            "timed_support_misses": timed_new_misses,
            "support_repository": {
                arm: {
                    "supports": int(len(repo.supports)),
                    "cache_bytes": int(repo.cache_nbytes),
                    "materialization_calls": int(repo.materialization_calls),
                    "materialized_atoms": int(repo.materialized_atoms),
                    "materialization_seconds": float(repo.materialization_seconds),
                }
                for arm, repo in repositories.items()
            },
            "rss_before_timed_bytes": rss_before,
            "rss_after_timed_bytes": rss_after,
            "projected_50k_lower_bound_only": {
                "timed_kernel_seconds_linear_lower_bound": wall * projection_scale,
                "preparation_plus_5k_prewarm_plus_kernel_lower_bound_seconds": (
                    fixed_seconds + wall * projection_scale
                ),
                "method": (
                    "5k measured timed kernel x10 with the 5k support inventory; "
                    "lower bound only because 50k may discover rarer APP states, "
                    "increase cache size, and scale nonlinearly"
                ),
                "runtime_commitment": False,
            },
        }
    )
    return totals, diagnostics, benchmark


def run_memory_probe(
    prepared: PreparedStreamingEvent,
    *,
    reference_dir: Path,
    particles: int,
    chunk_size: int,
    cache_output_dir: Path,
) -> dict[str, object]:
    """Bounded prewarm-only proof; no timed benchmark or promotion claim."""

    if int(particles) > 256:
        raise ValueError("Memory probe is intentionally capped at 256 particles")
    if prepared.preparation_peak_rss_bytes > SAFE_RSS_CAP_BYTES:
        raise MemoryError("Preparation exceeded the safe 4 GiB streaming cap")
    started = time.perf_counter()
    phase_monitor = RssPeakMonitor()
    phase_monitor.start()
    try:
        repositories = _new_benchmark_repositories(prepared, Path(reference_dir))
        phase_monitor.assert_below(SAFE_RSS_CAP_BYTES, "support repository load")
        engine = StreamingShotChain(
            repositories=repositories,
            course_drives=prepared.atoms.raw_course_drives,
            course_app=prepared.atoms.raw_course_app,
            state_sampler=batched_population.BatchedPopulationTransitionSampler(
                prepared.state_model, chunk_size=int(chunk_size)
            ),
            hazard_resolver=prepared.batched_hazard,
        )
        _, _, warm = _simulate_grid(
            engine, prepared.grid, arms=("raw", "normalized"),
            particles=int(particles), chunk_size=int(chunk_size),
            rng_mode="counter", collect_holes=False,
            rss_monitor=phase_monitor, rss_cap_bytes=SAFE_RSS_CAP_BYTES,
        )
        caches = {
            arm: persist_numeric_support_repository(
                repo, Path(cache_output_dir) / arm,
                rss_monitor=phase_monitor,
                rss_cap_bytes=SAFE_RSS_CAP_BYTES,
            )
            for arm, repo in repositories.items()
        }
        post_release = release_heavy_precompute_payload(prepared, repositories)
        phase_monitor.assert_below(
            SAFE_RSS_CAP_BYTES, "post-materialization release"
        )
    finally:
        phase_peak = phase_monitor.stop()
    memory = {
        "particles": int(particles),
        "chunk_size": int(chunk_size),
        "prewarm_only": True,
        "tail_policy": tail_policy.manifest_policy(),
        "preparation_peak_rss_bytes": int(prepared.preparation_peak_rss_bytes),
        "preparation_end_rss_bytes": int(prepared.preparation_rss_bytes),
        "prewarm_peak_rss_bytes": int(phase_peak),
        "post_release_rss_bytes": int(post_release),
        "safe_rss_cap_bytes": int(SAFE_RSS_CAP_BYTES),
        "global_peak_rss_bytes": int(
            max(prepared.preparation_peak_rss_bytes, phase_peak)
        ),
        "support_cache": caches,
        "elapsed_seconds": float(time.perf_counter() - started),
        "unsupported_state_rows": {
            arm: int(values["unsupported_state_rows"])
            for arm, values in warm["arms"].items()
        },
    }
    if any(
        int(memory[name]) > SAFE_RSS_CAP_BYTES
        for name in (
            "preparation_peak_rss_bytes", "prewarm_peak_rss_bytes",
            "post_release_rss_bytes", "global_peak_rss_bytes",
        )
    ):
        raise MemoryError(
            "Bounded prewarm memory probe exceeded safe 4 GiB cap: "
            f"{_jsonable(memory)}"
        )
    return memory


def write_memory_probe(
    output_dir: Path,
    *,
    source_start_sha256: str,
    prepared: PreparedStreamingEvent,
    memory: Mapping[str, object],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_end = sha256_file(Path(__file__))
    if source_end != source_start_sha256:
        raise AssertionError("Streaming source changed during memory probe")
    input_end_hashes = verify_input_hashes_unchanged(prepared)
    payload = {
        "artifact": f"{STREAMING_VERSION}_memory_probe",
        "research_only": True,
        "production_touched": False,
        "authoritative_model_evidence": False,
        "event": prepared.atoms.event_name,
        "event_key": prepared.atoms.target_event_key,
        "player_id": str(prepared.grid.iloc[0]["source_player_id"]),
        "production_mean_anchor_status": "not_calibrated_mechanics_only",
        "tail_policy": tail_policy.manifest_policy(),
        "reference_precompute_provenance": reference_precompute_provenance(
            prepared.input_hashes["precompute_source"]
        ),
        "source_start_sha256": source_start_sha256,
        "source_end_sha256": source_end,
        "input_hashes_sha256": prepared.input_hashes,
        "input_end_hashes_sha256": input_end_hashes,
        "input_logical_hashes_sha256": prepared.logical_hashes,
        "dependency_start_hashes_sha256": {
            name: prepared.input_hashes[name] for name in dependency_paths()
        },
        "dependency_end_hashes_sha256": {
            name: input_end_hashes[name] for name in dependency_paths()
        },
        "memory": dict(memory),
    }
    (output_dir / "memory_probe.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Streaming support prewarm memory probe",
                "",
                "Bounded prewarm only; this is not a terminal-parity or model result.",
                "",
                f"- Preparation peak: {int(memory['preparation_peak_rss_bytes']) / 2**30:.2f} GiB",
                f"- Prewarm peak: {int(memory['prewarm_peak_rss_bytes']) / 2**30:.2f} GiB",
                f"- Post-release RSS: {int(memory['post_release_rss_bytes']) / 2**30:.2f} GiB",
                f"- Safe cap: {int(memory['safe_rss_cap_bytes']) / 2**30:.2f} GiB",
                "- Production touched: no",
            ]
        ),
        encoding="utf-8",
    )
    write_complete_output_hashes(output_dir)
    return output_dir


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _deterministic_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without pandas' optional ``tabulate`` extra."""

    if frame.empty:
        return "_No rows._"

    def cell(value: object) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (
            isinstance(value, float) and not np.isfinite(value)
        ):
            text = ""
        elif isinstance(value, float):
            text = format(value, ".12g")
        else:
            text = str(value)
        return (
            text.replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", " ")
            .replace("\n", " ")
        )

    headers = [cell(name) for name in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _round_summary(totals: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, values in totals.items():
        array = tail_policy.round_totals_int16(
            values, label=f"{arm} round-summary totals"
        )
        if array.ndim != 2 or array.shape[0] != 4:
            raise ValueError("Round totals must have shape [4, particles]")
        for round_index in range(4):
            round_values = array[round_index].astype(float)
            rows.append(
                {
                    "arm": arm,
                    "round_no": round_index + 1,
                    "particles": len(round_values),
                    "mean_round_strokes": float(round_values.mean()),
                    "sd_round_strokes": float(round_values.std(ddof=0)),
                    "p01": float(np.quantile(round_values, 0.01)),
                    "p50": float(np.quantile(round_values, 0.50)),
                    "p99": float(np.quantile(round_values, 0.99)),
                    "round_total_sha256": array_sha256(array[round_index]),
                }
            )
    result = pd.DataFrame(rows)
    if {"raw", "normalized"}.issubset(totals):
        paired = (
            result.pivot(index="round_no", columns="arm", values="mean_round_strokes")
            .reset_index()
        )
        paired["normalized_minus_raw_mean_strokes"] = (
            paired["normalized"] - paired["raw"]
        )
        result = result.merge(
            paired[["round_no", "normalized_minus_raw_mean_strokes"]],
            on="round_no", how="left", validate="many_to_one",
        )
    return result


def verify_input_hashes_unchanged(
    prepared: PreparedStreamingEvent,
) -> dict[str, str]:
    observed = {
        name: sha256_file(path) for name, path in prepared.input_paths.items()
    }
    mismatches = {
        name: {"start": prepared.input_hashes.get(name), "end": value}
        for name, value in observed.items()
        if value != prepared.input_hashes.get(name)
    }
    if mismatches:
        raise AssertionError(f"Streaming input changed during run: {mismatches}")
    return observed


def write_complete_output_hashes(output_dir: Path) -> Path:
    """Write a non-circular checksum index for every completed artifact."""

    output_dir = Path(output_dir)
    checksum_path = output_dir / "OUTPUT_SHA256.json"
    rows: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(output_dir).as_posix()
        rows[relative] = {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    checksum_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checksum_path


def write_outputs(
    output_dir: Path,
    *,
    source_start_sha256: str,
    prepared: PreparedStreamingEvent,
    gate: Mapping[str, object],
    hazard_oracle: Mapping[str, object],
    parity_rows: pd.DataFrame,
    totals: Mapping[str, np.ndarray] | None,
    hole_diagnostics: pd.DataFrame | None,
    benchmark: Mapping[str, object] | None,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        unexpected = [
            path for path in output_dir.iterdir() if path.name != "support_cache"
        ]
        if unexpected:
            raise FileExistsError(
                f"Output has unexpected pre-existing files; versioned runs never "
                f"overwrite: {unexpected[:5]}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_path = output_dir / "raw_128_parity_particles.parquet"
    parity_rows.to_parquet(parity_path, index=False)
    (output_dir / "raw_128_gate.json").write_text(
        json.dumps(_jsonable(dict(gate)), indent=2), encoding="utf-8"
    )
    (output_dir / "hazard_scalar_oracle.json").write_text(
        json.dumps(_jsonable(dict(hazard_oracle)), indent=2), encoding="utf-8"
    )
    round_summary = pd.DataFrame()
    if totals is not None:
        arm_names = list(totals)
        stacked = np.stack(
            [
                tail_policy.round_totals_int16(
                    totals[name], label=f"{name} persisted round totals"
                )
                for name in arm_names
            ]
        )
        if stacked.dtype != np.int16:
            raise AssertionError("Persisted round totals must be int16")
        np.savez_compressed(
            output_dir / "round_totals_int16.npz",
            arm_names=np.asarray(arm_names),
            round_totals=stacked,
            round_numbers=np.asarray([1, 2, 3, 4], dtype=np.int8),
            particle_ids=np.arange(stacked.shape[2], dtype=np.int64),
            player_id=np.asarray([str(prepared.grid.iloc[0]["source_player_id"])]),
            event_key=np.asarray([prepared.atoms.target_event_key]),
        )
        round_summary = _round_summary(totals)
        round_summary.to_csv(output_dir / "round_summary.csv", index=False)
    if hole_diagnostics is not None:
        hole_diagnostics.to_csv(output_dir / "hole_diagnostics.csv", index=False)
    if benchmark is not None:
        (output_dir / "benchmark.json").write_text(
            json.dumps(_jsonable(dict(benchmark)), indent=2), encoding="utf-8"
        )
    source_end_sha256 = sha256_file(Path(__file__))
    if source_end_sha256 != source_start_sha256:
        raise AssertionError("Streaming source changed during authoritative run")
    input_end_hashes = verify_input_hashes_unchanged(prepared)
    dependency_end_hashes = {
        name: input_end_hashes[name] for name in dependency_paths()
    }
    dependency_start_hashes = {
        name: prepared.input_hashes[name] for name in dependency_paths()
    }
    payload_output_hashes = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, object] = {
        "artifact": STREAMING_VERSION,
        "research_only": True,
        "production_touched": False,
        "event": prepared.atoms.event_name,
        "event_key": prepared.atoms.target_event_key,
        "holdout_year": prepared.atoms.holdout_year,
        "player_id": str(prepared.grid.iloc[0]["source_player_id"]),
        "legacy_artifact_parity_scope": "raw_R1_18_holes_128_particles",
        "counter_chunk_parity_scope": (
            "raw_and_normalized_R1_R4_72_holes_128_particles"
        ),
        "benchmark_rounds": [] if totals is None else [1, 2, 3, 4],
        "holes_per_round": 18,
        "arms": [] if totals is None else list(totals),
        "normalization_mode": precompute.DEFAULT_MODE,
        "ability_log_weight_per_sg2": 0.0,
        "production_mean_anchor_status": "not_calibrated_mechanics_only",
        "production_skill_embedded": False,
        "posthoc_recentering_applied": False,
        "tail_policy": tail_policy.manifest_policy(),
        "reference_precompute_provenance": reference_precompute_provenance(
            dependency_start_hashes["precompute_source"]
        ),
        "generative_case_moments": (
            "raw_and_normalized_per-round-hole means/variances in hole_diagnostics.csv"
        ),
        "mass_contract": "base_hierarchy_mass_times_mean_one_model_weight",
        "rng_modes": {
            "parity": "legacy128",
            "benchmark": "counter",
            "counter_version": hook.RNG_VERSION,
            "arm_in_key": False,
            "chunk_in_key": False,
        },
        "streaming_contract": {
            "one_hole_chunk_at_a_time": True,
            "persisted_particle_payload": "int16_round_totals_only",
            "app_grouping": "vectorized_lie_x_15yd_bucket",
            "app_physical_classification": "deferred_after_particle_start_rotation",
            "arg_putt_sampler": "BatchedPopulationTransitionSampler.sample_many",
            "hazard_sampler": "batched_endpoint_linked_with_scalar_oracle",
            "pandas_in_timed_kernel": False,
            "per_particle_knn_in_timed_kernel": False,
            "survivor_filtering": False,
            "unsupported_states_fail_closed": True,
            "max_continuation_transitions": (
                tail_policy.MAX_CONTINUATION_TRANSITIONS
            ),
            "physical_hole_score_cap": None,
            "terminal_by_score_cap": False,
            "hole_score_clipping": False,
            "high_score_diagnostic_strokes": (
                tail_policy.HIGH_SCORE_DIAGNOSTIC_STROKES
            ),
            "high_score_diagnostic_only": True,
            "legacy_score_cap_status_emitted": False,
        },
        "transform_order": [
            "donor_native_asof_normalize",
            "rigid_align_neutral_endpoint_exactly_once",
            "target_profile_reapply_beta_zero",
            "runtime_physical_surface_reclassification",
        ],
        "coordinate_frame_version": prepared.atoms.prepared["provenance"].get(
            "coordinate_frame_version"
        ),
        "coordinate_reference_year": int(
            prepared.atoms.prepared["provenance"]["setup_reference_year"]
        ),
        "alignment_count": 1,
        "input_hashes_sha256": prepared.input_hashes,
        "input_end_hashes_sha256": input_end_hashes,
        "input_logical_hashes_sha256": prepared.logical_hashes,
        "dependency_start_hashes_sha256": dependency_start_hashes,
        "dependency_end_hashes_sha256": dependency_end_hashes,
        "payload_output_hashes_sha256": payload_output_hashes,
        "streaming_source_start_sha256": source_start_sha256,
        "streaming_source_end_sha256": source_end_sha256,
        "parity_artifact_sha256": sha256_file(parity_path),
        "raw_128_gate": dict(gate),
        "hazard_scalar_oracle": dict(hazard_oracle),
        "benchmark": None if benchmark is None else dict(benchmark),
        "cli": vars(args),
    }
    if totals is not None:
        manifest["round_totals_sha256"] = sha256_file(
            output_dir / "round_totals_int16.npz"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Streaming normalized shot-chain gate",
        "",
        (
            "Archive-only Travelers 2026 research run for the frozen pre-event "
            f"player {manifest['player_id']}. No production module or deployment was touched."
        ),
        "",
        "## Promotion gates",
        "",
        "- Legacy raw R1 (18 holes x 128) initializer physical states and atom IDs: exact.",
        "- Legacy raw R1 terminal/final states, hazards, penalty costs, and transitions: exact.",
        "- Counter raw+normalized R1-R4 paths under chunk sizes 128 and 17: exact.",
        "- Counter RNG particle 0:49,999 prefix of 100k: exact for all semantic streams.",
        f"- Batched endpoint hazard vs scalar oracle: {hazard_oracle['scalar_batch_exact_rows']} exact rows.",
        "- Unsupported states and survivor filtering: zero / forbidden.",
        "",
    ]
    if benchmark is not None:
        report_lines.extend(
            [
                "## 5k benchmark",
                "",
                f"Timed two-arm four-round wall: {float(benchmark['wall_seconds']):.2f} seconds.",
                f"Measured timed-kernel peak RSS: {int(benchmark['timed_kernel_peak_rss_bytes']) / 2**30:.2f} GiB.",
                (
                    "50k timed-kernel lower bound: "
                    f"{float(benchmark['projected_50k_lower_bound_only']['timed_kernel_seconds_linear_lower_bound']):.1f} seconds "
                    "by a 10x linear extrapolation. This is not a runtime commitment: "
                    "50k can discover rarer APP supports. No 50k run was launched."
                ),
                "Timed support materialization calls/misses: zero / zero.",
                "",
            "The normalized arm is beta-zero and remains research evidence only.",
                (
                    "Production mean anchor: not calibrated (mechanics only); "
                    "no post-hoc recentering or production skill embedding was applied."
                ),
                "",
            ]
        )
    if not round_summary.empty:
        report_lines.extend(
            [
                "## Round summary", "",
                _deterministic_markdown_table(round_summary), "",
            ]
        )
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    write_complete_output_hashes(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=precompute.DEFAULT_EXPORT)
    parser.add_argument("--history-db", type=Path, default=precompute.asof.DEFAULT_HISTORY_DB)
    parser.add_argument("--profile-dir", type=Path, default=precompute.DEFAULT_PROFILES)
    parser.add_argument("--baseline-cases", type=Path, default=precompute.DEFAULT_BASELINES)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-name", default=precompute.DEFAULT_EVENT)
    parser.add_argument("--holdout-year", type=int, default=precompute.DEFAULT_HOLDOUT)
    parser.add_argument("--course-start-year", type=int, default=2020)
    parser.add_argument("--roster-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stage", choices=("memory", "gate", "all"), default="all",
        help=(
            "memory runs bounded prewarm only; gate runs exact 128; all "
            "additionally runs the approved <=5k benchmark"
        ),
    )
    parser.add_argument("--memory-probe-particles", type=int, default=128)
    parser.add_argument(
        "--benchmark-particles", type=int, default=DEFAULT_BENCHMARK_PARTICLES
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.benchmark_particles) > MAX_APPROVED_PARTICLES:
        raise SystemExit(
            f"Refusing {args.benchmark_particles:,} particles: this version is capped "
            f"at {MAX_APPROVED_PARTICLES:,}; 50k/100k require explicit approval"
        )
    output_path = Path(args.output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise SystemExit(
            f"Refusing non-empty versioned output path: {output_path}"
        )
    source_start = sha256_file(Path(__file__))
    print("Preparing Travelers strict-prior normalized atoms once...")
    prep_monitor = RssPeakMonitor()
    prep_monitor.start()
    try:
        prepared = prepare_streaming_event(
            export_root=Path(args.export_root),
            history_db=Path(args.history_db),
            profile_dir=Path(args.profile_dir),
            baseline_cases=Path(args.baseline_cases),
            reference_dir=Path(args.reference_dir),
            event_name=str(args.event_name),
            holdout_year=int(args.holdout_year),
            course_start_year=int(args.course_start_year),
            roster_seed=int(args.roster_seed),
        )
    finally:
        preparation_peak_rss = prep_monitor.stop()
    prepared.preparation_peak_rss_bytes = int(preparation_peak_rss)
    if args.stage == "memory":
        print("Running bounded support-prewarm memory probe only...")
        memory = run_memory_probe(
            prepared,
            reference_dir=Path(args.reference_dir),
            particles=int(args.memory_probe_particles),
            chunk_size=min(int(args.chunk_size), int(args.memory_probe_particles)),
            cache_output_dir=Path(args.output_dir) / "support_cache",
        )
        output = write_memory_probe(
            Path(args.output_dir),
            source_start_sha256=source_start,
            prepared=prepared,
            memory=memory,
        )
        print(f"Wrote {output}")
        return
    print("Auditing batched endpoint hazards against scalar resolver...")
    hazard_oracle = audit_batched_hazard_oracle(prepared)
    print("Running exact raw 18-hole x 128 parity and counter chunk gate...")
    gate, parity_rows = run_raw_128_gate(
        prepared, reference_dir=Path(args.reference_dir)
    )
    totals: dict[str, np.ndarray] | None = None
    diagnostics: pd.DataFrame | None = None
    benchmark: dict[str, object] | None = None
    if args.stage == "all":
        print(
            f"Prewarming and benchmarking two arms x 4 rounds x "
            f"{int(args.benchmark_particles):,} particles..."
        )
        totals, diagnostics, benchmark = run_5k_benchmark(
            prepared,
            reference_dir=Path(args.reference_dir),
            particles=int(args.benchmark_particles),
            chunk_size=int(args.chunk_size),
            cache_output_dir=Path(args.output_dir) / "support_cache",
        )
    output = write_outputs(
        Path(args.output_dir),
        source_start_sha256=source_start,
        prepared=prepared,
        gate=gate,
        hazard_oracle=hazard_oracle,
        parity_rows=parity_rows,
        totals=totals,
        hole_diagnostics=diagnostics,
        benchmark=benchmark,
        args=args,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

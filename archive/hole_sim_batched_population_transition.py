"""Batched exact-course ARG/PUTT transition cache for research shot chains.

The scalar reference is ``PopulationTransitionSampler.sample`` in
``hole_sim_full_chain.py``.  This module preserves that sampler's state-pool
hierarchy, log-distance nearest-64 donor selection, donor-index ordering,
distance weights, uniform inverse-CDF rule, transition tuple, exceptions, and
backoff counts.  It moves pool resolution and array extraction outside the
particle loop and performs nearest-neighbour work in bounded NumPy batches.

This file is archive-only.  Nothing in the production simulation imports it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_markov as markov  # noqa: E402


CATEGORIES = ("arg", "putt")
NORMALIZED_LIES = (
    "hole",
    "tee",
    "fairway",
    "rough",
    "sand",
    "recovery",
    "green",
    "unknown",
)
DEFAULT_ROUNDS = (1, 2, 3, 4)
DEFAULT_HOLES = tuple(range(1, 19))
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_PROVENANCE = (
    REPO
    / "permanent_data"
    / "hole_sim_full_chain_authoritative_support_v2"
    / "wyndham_run"
    / "wyndham_championship"
    / "provenance.json"
)
DEFAULT_OUTPUT = (
    REPO / "permanent_data" / "hole_sim_batched_population_transition_benchmark"
)


def normalize_population_lie(value: object) -> str:
    """Match ``hole_sim_full_chain._normalize_lie`` exactly for one value."""

    text = str(value).strip().casefold()
    if text in {"hole", "holed", "cup"}:
        return "hole"
    if text in {"tee", "fairway", "rough", "sand", "recovery", "green"}:
        return text
    if text == "other":
        return "recovery"
    if "bunker" in text or "sand" in text:
        return "sand"
    return "unknown"


def normalize_population_lies(values: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of :func:`normalize_population_lie`."""

    raw = np.asarray(values).astype(str)
    text = np.char.lower(np.char.strip(raw))
    output = np.full(text.shape, "unknown", dtype="<U8")
    output[np.isin(text, np.asarray(["hole", "holed", "cup"]))] = "hole"
    for direct in ("tee", "fairway", "rough", "sand", "recovery", "green"):
        output[text == direct] = direct
    output[text == "other"] = "recovery"
    unresolved = output == "unknown"
    sand_like = (np.char.find(text, "bunker") >= 0) | (
        np.char.find(text, "sand") >= 0
    )
    output[unresolved & sand_like] = "sand"
    return output


def distance_bucket_many(distance: np.ndarray) -> np.ndarray:
    """Vectorised exact equivalent of ``markov.distance_bucket``."""

    values = np.asarray(distance, dtype=float)
    output = np.zeros(values.shape, dtype=np.int8)
    finite = np.isfinite(values)
    output[finite] = np.clip(
        np.searchsorted(markov.DISTANCE_EDGES, values[finite], side="right") - 1,
        0,
        3,
    ).astype(np.int8)
    return output


@dataclass(frozen=True)
class _PoolCache:
    pool: markov.TransitionPool
    start_log_distance: np.ndarray

    @classmethod
    def from_pool(cls, pool: markov.TransitionPool) -> "_PoolCache":
        start = np.asarray(pool.start_distance, dtype=float)
        return cls(
            pool=pool,
            start_log_distance=np.log1p(np.maximum(start, 0.0)),
        )


@dataclass(frozen=True)
class _Resolution:
    pool_id: int | None
    level: str | None
    error: str | None


@dataclass(frozen=True)
class BatchTransitionResult:
    """Columnar transition result aligned to the input row order."""

    holed: np.ndarray
    next_lie: np.ndarray
    next_distance: np.ndarray
    cost: np.ndarray
    level: np.ndarray
    available: np.ndarray
    error: np.ndarray

    def __len__(self) -> int:
        return len(self.available)

    def scalar_tuple(self, index: int) -> tuple[bool, str, float, int, str]:
        if not bool(self.available[index]):
            raise RuntimeError(str(self.error[index]))
        return (
            bool(self.holed[index]),
            str(self.next_lie[index]),
            float(self.next_distance[index]),
            int(self.cost[index]),
            str(self.level[index]),
        )

    def prefix(self, size: int) -> "BatchTransitionResult":
        stop = int(size)
        return BatchTransitionResult(
            holed=self.holed[:stop].copy(),
            next_lie=self.next_lie[:stop].copy(),
            next_distance=self.next_distance[:stop].copy(),
            cost=self.cost[:stop].copy(),
            level=self.level[:stop].copy(),
            available=self.available[:stop].copy(),
            error=self.error[:stop].copy(),
        )


def _broadcast_flat(*values: object) -> list[np.ndarray]:
    arrays = [np.asarray(value) for value in values]
    try:
        broadcast = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        raise ValueError("Transition query arrays are not broadcast-compatible") from exc
    return [np.asarray(array).reshape(-1) for array in broadcast]


class BatchedPopulationTransitionSampler:
    """Preindexed, uniform-driven batch equivalent of the scalar sampler."""

    def __init__(
        self,
        model: markov.MarkovHoleModel,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        round_values: Sequence[int] = DEFAULT_ROUNDS,
        hole_values: Sequence[int] = DEFAULT_HOLES,
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.model = model
        self.chunk_size = int(chunk_size)
        self.backoff_counts: Counter[str] = Counter()
        self._pool_by_id: dict[int, _PoolCache] = {}
        self._resolution: dict[tuple[int, int, str, str, int], _Resolution] = {}
        self._representative_by_level: dict[
            str, tuple[int, int, str, str, int]
        ] = {}
        self.resolution_cache_misses = 0
        self.last_sample_stats: dict[str, int | float] = {}
        started = time.perf_counter()
        self._preindex(round_values, hole_values)
        self.preindex_seconds = float(time.perf_counter() - started)

    def _cache_resolution(
        self,
        key: tuple[int, int, str, str, int],
    ) -> _Resolution:
        existing = self._resolution.get(key)
        if existing is not None:
            return existing
        round_no, hole_no, category, lie, bucket = key
        try:
            pool, level = self.model._pool(
                int(round_no),
                int(hole_no),
                str(category),
                str(lie),
                int(bucket),
            )
        except RuntimeError as exc:
            resolution = _Resolution(None, None, str(exc))
        else:
            pool_id = id(pool)
            if pool_id not in self._pool_by_id:
                self._pool_by_id[pool_id] = _PoolCache.from_pool(pool)
            resolution = _Resolution(pool_id, str(level), None)
            self._representative_by_level.setdefault(str(level), key)
        self._resolution[key] = resolution
        return resolution

    def _preindex(
        self,
        round_values: Sequence[int],
        hole_values: Sequence[int],
    ) -> None:
        rounds = sorted({int(value) for value in round_values})
        holes = sorted({int(value) for value in hole_values})
        if not rounds or not holes:
            raise ValueError("round_values and hole_values cannot be empty")
        for round_no in rounds:
            for hole_no in holes:
                for category in CATEGORIES:
                    for lie in NORMALIZED_LIES:
                        for bucket in range(4):
                            self._cache_resolution(
                                (round_no, hole_no, category, lie, bucket)
                            )

    @property
    def representative_state_by_level(
        self,
    ) -> Mapping[str, tuple[int, int, str, str, int]]:
        return dict(self._representative_by_level)

    @property
    def available_levels(self) -> tuple[str, ...]:
        return tuple(sorted(self._representative_by_level))

    @property
    def cache_nbytes(self) -> int:
        return int(
            sum(cache.start_log_distance.nbytes for cache in self._pool_by_id.values())
        )

    @property
    def preindexed_states(self) -> int:
        return len(self._resolution)

    @property
    def preindexed_available_states(self) -> int:
        return sum(value.pool_id is not None for value in self._resolution.values())

    def reset_counts(self) -> None:
        self.backoff_counts.clear()

    def sample(
        self,
        *,
        round_no: int,
        hole_no: int,
        category: str,
        lie: str,
        distance_yd: float,
        common_uniform: float,
    ) -> tuple[bool, str, float, int, str]:
        """Scalar compatibility wrapper with the reference exceptions."""

        if category not in CATEGORIES:
            raise ValueError(f"Population state sampler cannot draw {category}")
        result = self.sample_many(
            round_no=np.asarray([round_no]),
            hole_no=np.asarray([hole_no]),
            category=np.asarray([category], object),
            lie=np.asarray([lie], object),
            distance_yd=np.asarray([distance_yd]),
            common_uniform=np.asarray([common_uniform]),
            chunk_size=1,
        )
        return result.scalar_tuple(0)

    def sample_many(
        self,
        *,
        round_no: np.ndarray | int,
        hole_no: np.ndarray | int,
        category: np.ndarray | str,
        lie: np.ndarray | str,
        distance_yd: np.ndarray | float,
        common_uniform: np.ndarray | float,
        chunk_size: int | None = None,
    ) -> BatchTransitionResult:
        """Sample aligned state rows without pandas or per-row nearest-neighbour work."""

        raw_round, raw_hole, raw_category, raw_lie, raw_distance, raw_uniform = (
            _broadcast_flat(
                round_no,
                hole_no,
                category,
                lie,
                distance_yd,
                common_uniform,
            )
        )
        size = len(raw_round)
        chunks = self.chunk_size if chunk_size is None else int(chunk_size)
        if chunks <= 0:
            raise ValueError("chunk_size must be positive")
        rounds = raw_round.astype(np.int64)
        holes = raw_hole.astype(np.int64)
        categories = raw_category.astype(str)
        invalid_category = ~np.isin(categories, np.asarray(CATEGORIES))
        if invalid_category.any():
            first = int(np.flatnonzero(invalid_category)[0])
            raise ValueError(
                f"Population state sampler cannot draw {categories[first]}"
            )
        lies = normalize_population_lies(raw_lie)
        distances = raw_distance.astype(float)
        uniforms = raw_uniform.astype(float)
        buckets = distance_bucket_many(distances)

        holed = np.zeros(size, dtype=bool)
        next_lie = np.full(size, "", dtype=object)
        next_distance = np.full(size, np.nan, dtype=float)
        cost = np.zeros(size, dtype=np.int16)
        level = np.full(size, "", dtype=object)
        available = np.zeros(size, dtype=bool)
        error = np.full(size, "", dtype=object)
        if size == 0:
            return BatchTransitionResult(
                holed, next_lie, next_distance, cost, level, available, error
            )

        category_code = (categories == "putt").astype(np.int8)
        lie_code = np.zeros(size, dtype=np.int8)
        for code, label in enumerate(NORMALIZED_LIES):
            lie_code[lies == label] = code
        state = np.empty(
            size,
            dtype=[
                ("round_no", "<i8"),
                ("hole_no", "<i8"),
                ("category", "i1"),
                ("lie", "i1"),
                ("bucket", "i1"),
            ],
        )
        state["round_no"] = rounds
        state["hole_no"] = holes
        state["category"] = category_code
        state["lie"] = lie_code
        state["bucket"] = buckets
        unique_state, inverse = np.unique(state, return_inverse=True)

        max_working_bytes = 0
        max_pool_rows = 0
        processed_chunks = 0
        for group_index, encoded in enumerate(unique_state):
            positions = np.flatnonzero(inverse == group_index)
            query_key = (
                int(encoded["round_no"]),
                int(encoded["hole_no"]),
                CATEGORIES[int(encoded["category"])],
                NORMALIZED_LIES[int(encoded["lie"])],
                int(encoded["bucket"]),
            )
            resolution = self._resolution.get(query_key)
            if resolution is None:
                self.resolution_cache_misses += 1
                resolution = self._cache_resolution(query_key)
            if resolution.pool_id is None:
                error[positions] = str(resolution.error)
                continue
            pool_cache = self._pool_by_id[int(resolution.pool_id)]
            pool = pool_cache.pool
            group_level = str(resolution.level)
            available[positions] = True
            level[positions] = group_level
            self.backoff_counts[group_level] += int(len(positions))
            pool_rows = len(pool)
            max_pool_rows = max(max_pool_rows, pool_rows)
            k = min(int(self.model.knn), pool_rows)
            for start in range(0, len(positions), chunks):
                selected_positions = positions[start : start + chunks]
                query_log = np.log1p(
                    np.maximum(distances[selected_positions], 0.0)
                )
                metric = np.abs(
                    query_log[:, None]
                    - pool_cache.start_log_distance[None, :]
                )
                if k < pool_rows:
                    neighbours = np.argpartition(
                        metric, k - 1, axis=1
                    )[:, :k]
                else:
                    neighbours = np.broadcast_to(
                        np.arange(pool_rows, dtype=np.int64),
                        (len(selected_positions), pool_rows),
                    ).copy()
                neighbours.sort(axis=1)
                selected_metric = np.take_along_axis(metric, neighbours, axis=1)
                weights = np.exp(-selected_metric / 0.18)
                np.maximum(weights, 1e-12, out=weights)
                weights /= weights.sum(axis=1, keepdims=True)
                cumulative = np.cumsum(weights, axis=1)
                selected = np.count_nonzero(
                    cumulative <= uniforms[selected_positions, None], axis=1
                )
                selected[np.isnan(uniforms[selected_positions])] = k
                np.minimum(selected, k - 1, out=selected)
                choices = neighbours[
                    np.arange(len(selected_positions), dtype=np.int64), selected
                ]
                holed[selected_positions] = pool.holed[choices]
                next_lie[selected_positions] = pool.next_lie[choices]
                next_distance[selected_positions] = pool.next_distance[choices]
                cost[selected_positions] = pool.cost[choices]
                working_bytes = int(
                    metric.nbytes
                    + neighbours.nbytes
                    + selected_metric.nbytes
                    + weights.nbytes
                    + cumulative.nbytes
                    + choices.nbytes
                )
                max_working_bytes = max(max_working_bytes, working_bytes)
                processed_chunks += 1

        self.last_sample_stats = {
            "rows": int(size),
            "unique_state_groups": int(len(unique_state)),
            "processed_chunks": int(processed_chunks),
            "max_pool_rows": int(max_pool_rows),
            "max_numpy_working_bytes": int(max_working_bytes),
            "chunk_size": int(chunks),
            "resolution_cache_misses": int(self.resolution_cache_misses),
            "unavailable_rows": int((~available).sum()),
        }
        return BatchTransitionResult(
            holed=holed,
            next_lie=next_lie,
            next_distance=next_distance,
            cost=cost,
            level=level,
            available=available,
            error=error,
        )


def _hash_update_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    if array.dtype.kind in {"O", "U", "S"}:
        for item in array.reshape(-1):
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(contiguous.tobytes())


def query_sha256(query: Mapping[str, np.ndarray], size: int | None = None) -> str:
    digest = hashlib.sha256()
    stop = None if size is None else int(size)
    for name in (
        "round_no",
        "hole_no",
        "category",
        "lie",
        "distance_yd",
        "common_uniform",
    ):
        _hash_update_array(digest, name, np.asarray(query[name])[:stop])
    return digest.hexdigest()


def result_sha256(result: BatchTransitionResult) -> str:
    digest = hashlib.sha256()
    for name in (
        "holed",
        "next_lie",
        "next_distance",
        "cost",
        "level",
        "available",
        "error",
    ):
        _hash_update_array(digest, name, getattr(result, name))
    return digest.hexdigest()


def results_equal(left: BatchTransitionResult, right: BatchTransitionResult) -> bool:
    if len(left) != len(right):
        return False
    for name in ("holed", "next_lie", "cost", "level", "available", "error"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return False
    return bool(
        np.array_equal(
            left.next_distance,
            right.next_distance,
            equal_nan=True,
        )
    )


def _representative_queries(
    transitions: object,
    size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build a deterministic 70% PUTT / 30% ARG real-course workload."""

    frame = transitions.reset_index(drop=True)
    arg_rows = np.flatnonzero(frame["category"].astype(str).to_numpy() == "arg")
    putt_rows = np.flatnonzero(frame["category"].astype(str).to_numpy() == "putt")
    if not len(arg_rows) or not len(putt_rows):
        raise ValueError("Benchmark transitions must contain ARG and PUTT rows")
    rng = np.random.default_rng(int(seed))
    use_putt = rng.random(int(size)) < 0.70
    chosen = np.empty(int(size), dtype=np.int64)
    chosen[use_putt] = rng.choice(putt_rows, size=int(use_putt.sum()), replace=True)
    chosen[~use_putt] = rng.choice(
        arg_rows, size=int((~use_putt).sum()), replace=True
    )
    selected = frame.iloc[chosen]
    return {
        "round_no": selected["round_no"].to_numpy(np.int16),
        "hole_no": selected["hole_no"].to_numpy(np.int16),
        "category": selected["category"].astype(str).to_numpy(),
        "lie": selected["lie"].astype(str).to_numpy(),
        "distance_yd": selected["start_distance"].to_numpy(float),
        "common_uniform": rng.random(int(size)),
    }


def _measure_sample(
    sampler: BatchedPopulationTransitionSampler,
    query: Mapping[str, np.ndarray],
    size: int,
    chunk_size: int,
) -> tuple[BatchTransitionResult, dict[str, object]]:
    sampler.reset_counts()
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = sampler.sample_many(
        **{name: np.asarray(value)[: int(size)] for name, value in query.items()},
        chunk_size=int(chunk_size),
    )
    seconds = float(time.perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diagnostics: dict[str, object] = {
        "rows": int(size),
        "sample_many_seconds": seconds,
        "states_per_second": float(size / seconds),
        "tracemalloc_peak_bytes": int(peak),
        "output_array_bytes": int(
            result.holed.nbytes
            + result.next_distance.nbytes
            + result.cost.nbytes
            + result.available.nbytes
        ),
        "backoff_counts": dict(sorted(sampler.backoff_counts.items())),
        "input_sha256": query_sha256(query, int(size)),
        "output_sha256": result_sha256(result),
        **sampler.last_sample_stats,
    }
    return result, diagnostics


def _distance_for_bucket(pool: markov.TransitionPool, bucket: int) -> float:
    values = np.asarray(pool.start_distance, float)
    candidates = values[distance_bucket_many(values) == int(bucket)]
    if len(candidates):
        return float(np.median(candidates))
    return (25.0, 125.0, 180.0, 250.0)[int(bucket)]


def _real_level_equality_rows(
    batch: BatchedPopulationTransitionSampler,
    reference: object,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level_name, key in sorted(batch.representative_state_by_level.items()):
        resolution = batch._resolution[key]
        if resolution.pool_id is None:
            continue
        pool = batch._pool_by_id[int(resolution.pool_id)].pool
        round_no, hole_no, category, lie, bucket = key
        distance = _distance_for_bucket(pool, bucket)
        for uniform in (0.0, 0.5, np.nextafter(1.0, 0.0)):
            expected = reference.sample(
                round_no=round_no,
                hole_no=hole_no,
                category=category,
                lie=lie,
                distance_yd=distance,
                common_uniform=float(uniform),
            )
            actual = batch.sample(
                round_no=round_no,
                hole_no=hole_no,
                category=category,
                lie=lie,
                distance_yd=distance,
                common_uniform=float(uniform),
            )
            equal = actual == expected
            rows.append(
                {
                    "level": level_name,
                    "round_no": round_no,
                    "hole_no": hole_no,
                    "category": category,
                    "lie": lie,
                    "bucket": bucket,
                    "distance_yd": distance,
                    "uniform": float(uniform),
                    "exact_equal": bool(equal),
                    "expected": repr(expected),
                    "actual": repr(actual),
                }
            )
            if not equal:
                raise AssertionError(
                    f"Real-course equality failed for {level_name}/{key}/{uniform}"
                )
    return rows


def _real_workload_equality(
    batch: BatchedPopulationTransitionSampler,
    reference: object,
    query: Mapping[str, np.ndarray],
    *,
    rows: int = 2048,
) -> dict[str, object]:
    """Compare a deterministic, workload-wide real-course subset row by row."""

    total = len(np.asarray(query["round_no"]))
    count = min(int(rows), total)
    indices = np.linspace(0, total - 1, count, dtype=np.int64)
    subset = {name: np.asarray(value)[indices] for name, value in query.items()}
    batch.reset_counts()
    batch_started = time.perf_counter()
    actual = batch.sample_many(**subset)
    batch_seconds = float(time.perf_counter() - batch_started)
    reference.backoff_counts.clear()
    scalar_started = time.perf_counter()
    exact = True
    unavailable = 0
    for index in range(count):
        try:
            expected = reference.sample(
                round_no=int(subset["round_no"][index]),
                hole_no=int(subset["hole_no"][index]),
                category=str(subset["category"][index]),
                lie=str(subset["lie"][index]),
                distance_yd=float(subset["distance_yd"][index]),
                common_uniform=float(subset["common_uniform"][index]),
            )
        except RuntimeError as exc:
            unavailable += 1
            row_equal = (
                not bool(actual.available[index])
                and str(actual.error[index]) == str(exc)
            )
        else:
            row_equal = bool(actual.available[index]) and (
                actual.scalar_tuple(index) == expected
            )
        exact &= bool(row_equal)
    scalar_seconds = float(time.perf_counter() - scalar_started)
    counter_equal = batch.backoff_counts == reference.backoff_counts
    exact &= counter_equal
    if not exact:
        raise AssertionError("Real-course workload scalar/batch equality failed")
    return {
        "rows": int(count),
        "exact": bool(exact),
        "unavailable_rows": int(unavailable),
        "counter_exact": bool(counter_equal),
        "batch_seconds": batch_seconds,
        "scalar_seconds": scalar_seconds,
        "speedup": float(scalar_seconds / max(batch_seconds, 1e-12)),
        "input_sha256": query_sha256(subset),
        "batch_output_sha256": result_sha256(actual),
        "backoff_counts": dict(sorted(batch.backoff_counts.items())),
    }


def run_benchmark(
    provenance_path: Path,
    output: Path,
    *,
    export_root: Path,
    holdout_year: int = 2026,
    seed: int = 20260828,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, object]:
    """Build a strict-prior real-course model and benchmark 50k/100k states."""

    # Lazy import avoids a circular dependency if the batch sampler is later
    # wired into the archive-only full-chain harness.
    import hole_sim_full_chain as full_chain

    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    event_keys = {
        int(year): str(key)
        for year, key in provenance["verified_event_keys"].items()
    }
    build_started = time.perf_counter()
    model, transitions = full_chain._course_population_model(
        Path(export_root), event_keys, int(holdout_year)
    )
    model_build_seconds = float(time.perf_counter() - build_started)
    sampler = BatchedPopulationTransitionSampler(
        model,
        chunk_size=int(chunk_size),
    )
    query = _representative_queries(transitions, 100_000, int(seed))
    result_50, bench_50 = _measure_sample(
        sampler, query, 50_000, int(chunk_size)
    )
    result_100, bench_100 = _measure_sample(
        sampler, query, 100_000, int(chunk_size)
    )
    prefix_equal = results_equal(result_50, result_100.prefix(50_000))
    if not prefix_equal:
        raise AssertionError("50k result is not an exact prefix of the 100k result")

    alternate_rows = 5_000
    alternate_chunk = max(1, min(257, int(chunk_size) - 1))
    if alternate_chunk == int(chunk_size):
        alternate_chunk = int(chunk_size) + 1
    alternate = sampler.sample_many(
        **{
            name: np.asarray(value)[:alternate_rows]
            for name, value in query.items()
        },
        chunk_size=alternate_chunk,
    )
    canonical = result_100.prefix(alternate_rows)
    chunk_equal = results_equal(alternate, canonical)
    if not chunk_equal:
        raise AssertionError("Chunk size changed batched transition output")

    reference = full_chain.PopulationTransitionSampler(model)
    equality_rows = _real_level_equality_rows(sampler, reference)
    if not equality_rows:
        raise AssertionError("No real-course hierarchy equality rows were produced")
    workload_equality = _real_workload_equality(
        sampler, reference, query, rows=2048
    )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "real_course_hierarchy_equality.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(equality_rows[0]))
        writer.writeheader()
        writer.writerows(equality_rows)
    with (output / "benchmark_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        flat_rows: list[dict[str, object]] = []
        for benchmark in (bench_50, bench_100):
            row = benchmark.copy()
            row["backoff_counts"] = json.dumps(row["backoff_counts"], sort_keys=True)
            flat_rows.append(row)
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    manifest: dict[str, object] = {
        "status": "complete",
        "research_only": True,
        "reference": "hole_sim_full_chain.PopulationTransitionSampler.sample",
        "model_build_seconds": model_build_seconds,
        "sampler_preindex_seconds": sampler.preindex_seconds,
        "sampler_cache_nbytes": sampler.cache_nbytes,
        "preindexed_states": sampler.preindexed_states,
        "preindexed_available_states": sampler.preindexed_available_states,
        "unique_cached_pools": len(sampler._pool_by_id),
        "knn": int(model.knn),
        "min_exact": int(model.min_exact),
        "chunk_size": int(chunk_size),
        "seed": int(seed),
        "transition_rows": int(len(transitions)),
        "event_keys": event_keys,
        "available_hierarchy_levels": list(sampler.available_levels),
        "real_course_equality_rows": len(equality_rows),
        "all_real_course_equality_rows_exact": bool(
            all(row["exact_equal"] for row in equality_rows)
        ),
        "real_course_workload_equality": workload_equality,
        "prefix_50k_of_100k_exact": prefix_equal,
        "chunk_invariance_rows": alternate_rows,
        "alternate_chunk_size": alternate_chunk,
        "chunk_invariance_exact": chunk_equal,
        "benchmark_50k": bench_50,
        "benchmark_100k": bench_100,
        "provenance_path": str(Path(provenance_path).resolve()),
        "provenance_sha256": hashlib.sha256(
            Path(provenance_path).read_bytes()
        ).hexdigest(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_full_chain_sha256": hashlib.sha256(
            (ARCHIVE / "hole_sim_full_chain.py").read_bytes()
        ).hexdigest(),
        "markov_sha256": hashlib.sha256(
            (ARCHIVE / "hole_sim_markov.py").read_bytes()
        ).hexdigest(),
        "production_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = [
        "# Batched exact-course ARG/PUTT transition benchmark",
        "",
        "The batch sampler exactly reproduces the archive full-chain scalar "
        "transition tuple for every real-course hierarchy level available in "
        "the strict-prior model. Pool resolution and pool-array preparation are "
        "outside the timed loop; sampling uses bounded NumPy chunks with no "
        "pandas or per-particle nearest-neighbour call.",
        "",
        f"- Model build: {model_build_seconds:.3f}s",
        f"- Sampler preindex: {sampler.preindex_seconds:.3f}s",
        f"- Cached log-distance arrays: {sampler.cache_nbytes:,} bytes",
        f"- 50k sampling: {float(bench_50['sample_many_seconds']):.3f}s "
        f"({float(bench_50['states_per_second']):,.0f} states/s)",
        f"- 100k sampling: {float(bench_100['sample_many_seconds']):.3f}s "
        f"({float(bench_100['states_per_second']):,.0f} states/s)",
        f"- 100k tracemalloc peak: {int(bench_100['tracemalloc_peak_bytes']):,} bytes",
        f"- 100k maximum NumPy working set inside one chunk: "
        f"{int(bench_100['max_numpy_working_bytes']):,} bytes",
        f"- 50k exact prefix of 100k: {prefix_equal}",
        f"- Chunk-size invariance on {alternate_rows:,} states: {chunk_equal}",
        f"- Real-course equality rows: {len(equality_rows)}; all exact: "
        f"{all(row['exact_equal'] for row in equality_rows)}",
        f"- Real-workload scalar parity: {int(workload_equality['rows']):,} "
        f"rows exact, including Counter diagnostics; measured speedup "
        f"{float(workload_equality['speedup']):.1f}x",
        "",
        "Input/output SHA-256 values, hierarchy counts, timing, and memory are "
        "recorded in `manifest.json` and `benchmark_runs.csv`.",
        "",
        "This artifact is research-only and does not modify the production "
        "simulation or the existing full-chain/Markov modules.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export-root", type=Path, default=markov.DEFAULT_EXPORT)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    manifest = run_benchmark(
        args.provenance,
        args.output,
        export_root=args.export_root,
        holdout_year=args.holdout_year,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

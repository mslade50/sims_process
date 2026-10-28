from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from archive.hole_sim_batched_population_transition import (
    BatchedPopulationTransitionSampler,
    distance_bucket_many,
    normalize_population_lie,
    normalize_population_lies,
    result_sha256,
    results_equal,
)
from archive.hole_sim_full_chain import (
    PopulationTransitionSampler,
    _normalize_lie,
)
from archive.hole_sim_markov import MarkovHoleModel, distance_bucket


TRANSITION_COLUMNS = [
    "round_no",
    "hole_no",
    "category",
    "lie",
    "bucket",
    "start_distance",
    "next_distance",
    "next_lie",
    "sg",
    "cost",
    "holed",
    "sg_weight_valid",
]


def _rows(
    *,
    round_no: int,
    hole_no: int,
    category: str,
    lie: str,
    bucket: int,
    count: int,
    tag: int,
) -> list[dict[str, object]]:
    base_distance = (20.0, 125.0, 180.0, 250.0)[bucket]
    return [
        {
            "round_no": round_no,
            "hole_no": hole_no,
            "category": category,
            "lie": lie,
            "bucket": bucket,
            "start_distance": base_distance + (index % 3) * 0.25,
            "next_distance": tag * 1000.0 + index + 0.125,
            "next_lie": "hole" if index % 5 == 0 else "green",
            "sg": 0.0,
            "cost": 1 + (index % 4 == 0),
            "holed": index % 5 == 0,
            "sg_weight_valid": True,
        }
        for index in range(count)
    ]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRANSITION_COLUMNS)
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def _hierarchy_model() -> MarkovHoleModel:
    course: list[dict[str, object]] = []
    tour: list[dict[str, object]] = []
    # exact_hole
    course += _rows(
        round_no=1, hole_no=1, category="arg", lie="fairway",
        bucket=0, count=5, tag=1,
    )
    # exact_hole_lie: queried bucket=1 has 2, but lie total has 5.
    course += _rows(
        round_no=1, hole_no=2, category="arg", lie="rough",
        bucket=1, count=2, tag=2,
    )
    course += _rows(
        round_no=1, hole_no=2, category="arg", lie="rough",
        bucket=0, count=3, tag=3,
    )
    # course: neither exact group reaches four, course bucket total does.
    course += _rows(
        round_no=1, hole_no=3, category="arg", lie="sand",
        bucket=0, count=3, tag=4,
    )
    course += _rows(
        round_no=1, hole_no=4, category="arg", lie="sand",
        bucket=0, count=2, tag=5,
    )
    # course_relaxed: queried bucket=2 has one, all buckets total four.
    course += _rows(
        round_no=1, hole_no=5, category="arg", lie="recovery",
        bucket=2, count=1, tag=6,
    )
    course += _rows(
        round_no=1, hole_no=6, category="arg", lie="recovery",
        bucket=0, count=3, tag=7,
    )
    # tour and tour_relaxed.
    tour += _rows(
        round_no=1, hole_no=7, category="putt", lie="green",
        bucket=0, count=5, tag=8,
    )
    tour += _rows(
        round_no=1, hole_no=8, category="putt", lie="sand",
        bucket=1, count=2, tag=9,
    )
    tour += _rows(
        round_no=1, hole_no=9, category="putt", lie="sand",
        bucket=0, count=3, tag=10,
    )
    return MarkovHoleModel(
        _frame(course), _frame(tour), min_exact=4, knn=64
    )


def _result_rows(result) -> list[tuple[bool, str, float, int, str]]:
    return [result.scalar_tuple(index) for index in range(len(result))]


def test_lie_normalisation_and_distance_buckets_match_reference() -> None:
    lies = np.asarray(
        [
            "hole", " Holed ", "CUP", "tee", "FairWay", "rough",
            "sand", "recovery", "green", "other", "greenside bunker",
            "waste sand area", "native", None,
        ],
        object,
    )
    expected_lies = np.asarray([_normalize_lie(value) for value in lies])
    assert np.array_equal(normalize_population_lies(lies), expected_lies)
    assert [normalize_population_lie(value) for value in lies] == expected_lies.tolist()

    distances = np.asarray(
        [-1.0, 0.0, 99.999, 100.0, 149.999, 150.0, 214.999, 215.0, 1e6, np.nan, np.inf]
    )
    expected_buckets = np.asarray([distance_bucket(value) for value in distances])
    assert np.array_equal(distance_bucket_many(distances), expected_buckets)


def test_all_hierarchy_levels_exactly_match_scalar_reference_and_counters() -> None:
    model = _hierarchy_model()
    reference = PopulationTransitionSampler(model)
    batch = BatchedPopulationTransitionSampler(model, chunk_size=3)
    queries = {
        "round_no": np.asarray([1, 1, 2, 2, 3, 3, 4]),
        "hole_no": np.asarray([1, 2, 3, 4, 5, 6, 7]),
        "category": np.asarray(["arg", "arg", "arg", "arg", "putt", "putt", "putt"]),
        "lie": np.asarray(["fairway", "rough", "bunker", "other", "green", "sand", "native"]),
        "distance_yd": np.asarray([20.1, 125.1, 20.2, 180.0, 20.0, 125.0, 20.0]),
        "common_uniform": np.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, np.nextafter(1.0, 0.0)]),
    }
    expected = [
        reference.sample(
            round_no=int(queries["round_no"][index]),
            hole_no=int(queries["hole_no"][index]),
            category=str(queries["category"][index]),
            lie=str(queries["lie"][index]),
            distance_yd=float(queries["distance_yd"][index]),
            common_uniform=float(queries["common_uniform"][index]),
        )
        for index in range(len(queries["round_no"]))
    ]
    actual = batch.sample_many(**queries)

    assert _result_rows(actual) == expected
    assert list(actual.level) == [
        "exact_hole",
        "exact_hole_lie",
        "course",
        "course_relaxed",
        "tour",
        "tour_relaxed",
        "tour_last_resort",
    ]
    assert batch.backoff_counts == reference.backoff_counts
    assert batch.backoff_counts == Counter({level: 1 for level in actual.level})


def _tie_model() -> MarkovHoleModel:
    rows: list[dict[str, object]] = []
    distances = np.repeat(np.asarray([8.0, 10.0, 12.0]), 32)
    for index, distance in enumerate(distances):
        rows.append(
            {
                "round_no": 1,
                "hole_no": 1,
                "category": "arg",
                "lie": "fairway",
                "bucket": 0,
                "start_distance": float(distance),
                "next_distance": float(index) + 0.125,
                "next_lie": "hole" if index % 11 == 0 else "green",
                "sg": 0.0,
                "cost": 1 + (index % 7 == 0),
                "holed": index % 11 == 0,
                "sg_weight_valid": True,
            }
        )
    return MarkovHoleModel(_frame(rows), _frame([]), min_exact=4, knn=64)


def test_tie_heavy_nearest64_chunk_and_prefix_outputs_are_exact() -> None:
    model = _tie_model()
    rng = np.random.default_rng(20260828)
    size = 1000
    queries = {
        "round_no": np.ones(size, dtype=int),
        "hole_no": np.ones(size, dtype=int),
        "category": np.full(size, "arg"),
        "lie": np.full(size, "fairway"),
        "distance_yd": rng.choice(np.asarray([9.0, 10.0, 11.0]), size=size),
        "common_uniform": rng.random(size),
    }
    queries["common_uniform"][:5] = [
        0.0,
        np.nextafter(0.0, 1.0),
        0.5,
        np.nextafter(1.0, 0.0),
        1.0,
    ]
    reference = PopulationTransitionSampler(model)
    expected = [
        reference.sample(
            round_no=1,
            hole_no=1,
            category="arg",
            lie="fairway",
            distance_yd=float(queries["distance_yd"][index]),
            common_uniform=float(queries["common_uniform"][index]),
        )
        for index in range(size)
    ]
    sampler = BatchedPopulationTransitionSampler(model, chunk_size=257)
    result_257 = sampler.sample_many(**queries)
    result_7 = sampler.sample_many(**queries, chunk_size=7)
    result_1 = sampler.sample_many(**queries, chunk_size=1)
    prefix = sampler.sample_many(
        **{name: value[:333] for name, value in queries.items()},
        chunk_size=19,
    )

    assert _result_rows(result_257) == expected
    assert results_equal(result_257, result_7)
    assert results_equal(result_257, result_1)
    assert results_equal(result_257.prefix(333), prefix)
    assert result_sha256(result_257) == result_sha256(result_7)
    assert result_sha256(result_257.prefix(333)) == result_sha256(prefix)


def test_preindexed_batch_never_calls_pool_or_pandas_inside_sampling(monkeypatch) -> None:
    model = _tie_model()
    sampler = BatchedPopulationTransitionSampler(model, chunk_size=8)

    def fail_pool(*_args, **_kwargs):
        raise AssertionError("model._pool was called after preindex")

    monkeypatch.setattr(model, "_pool", fail_pool)
    result = sampler.sample_many(
        round_no=np.ones(50, dtype=int),
        hole_no=np.ones(50, dtype=int),
        category=np.full(50, "arg"),
        lie=np.full(50, "fairway"),
        distance_yd=np.linspace(8.0, 12.0, 50),
        common_uniform=np.linspace(0.0, 0.99, 50),
    )

    assert result.available.all()
    assert sampler.resolution_cache_misses == 0
    source = inspect.getsource(BatchedPopulationTransitionSampler.sample_many)
    assert ".iterrows(" not in source
    assert ".iloc[" not in source
    assert "pd." not in source
    assert "DataFrame" not in source


def test_unavailable_and_invalid_category_exception_semantics_match_reference() -> None:
    course = _frame(
        _rows(
            round_no=1, hole_no=1, category="arg", lie="fairway",
            bucket=0, count=5, tag=1,
        )
    )
    model = MarkovHoleModel(course, _frame([]), min_exact=4, knn=64)
    reference = PopulationTransitionSampler(model)
    batch = BatchedPopulationTransitionSampler(model)

    with pytest.raises(RuntimeError) as expected_error:
        reference.sample(
            round_no=1, hole_no=1, category="putt", lie="green",
            distance_yd=10.0, common_uniform=0.5,
        )
    result = batch.sample_many(
        round_no=np.asarray([1]),
        hole_no=np.asarray([1]),
        category=np.asarray(["putt"]),
        lie=np.asarray(["green"]),
        distance_yd=np.asarray([10.0]),
        common_uniform=np.asarray([0.5]),
    )
    assert not result.available[0]
    assert result.error[0] == str(expected_error.value)
    with pytest.raises(RuntimeError, match="No transition donor"):
        batch.sample(
            round_no=1, hole_no=1, category="putt", lie="green",
            distance_yd=10.0, common_uniform=0.5,
        )
    assert not batch.backoff_counts

    with pytest.raises(ValueError) as reference_category_error:
        reference.sample(
            round_no=1, hole_no=1, category="app", lie="fairway",
            distance_yd=100.0, common_uniform=0.5,
        )
    with pytest.raises(ValueError) as batch_category_error:
        batch.sample(
            round_no=1, hole_no=1, category="app", lie="fairway",
            distance_yd=100.0, common_uniform=0.5,
        )
    assert str(batch_category_error.value) == str(reference_category_error.value)

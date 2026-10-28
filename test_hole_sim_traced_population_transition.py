from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archive.hole_sim_batched_population_transition import (
    BatchedPopulationTransitionSampler,
    results_equal,
)
from archive.hole_sim_markov import MarkovHoleModel
from archive.hole_sim_traced_population_transition import (
    MARKOV_TRANSITION_COLUMNS,
    TracedBatchedPopulationTransitionSampler,
    TransitionPoolTraceRegistry,
    TransitionTraceParityError,
    assert_markov_frame_parity,
    build_strict_prior_course_trace_bundle,
)


REAL_EXPORT = Path(r"C:\Users\McKinley Slade\dev\golf_scraping\output\export")


def _synthetic_traced_frame(count: int = 96) -> pd.DataFrame:
    distances = np.resize(np.asarray([8.0, 10.0, 12.0]), count)
    rows: list[dict[str, object]] = []
    for index, distance in enumerate(distances):
        rows.append(
            {
                "transition_row_id": index,
                "event_key": "pga:R2025001",
                "source_player_id": 1000 + (index % 7),
                "player_name": f"donor {index % 7}",
                "round_no": 1,
                "hole_no": 1,
                "stroke_no": 2 + (index % 3),
                "shot_key": f"synthetic-shot-{index:03d}",
                "source_season": 2025,
                "source_raw_row": index,
                "category": "arg",
                "lie": "fairway",
                "bucket": 0,
                "start_distance": float(distance),
                "next_distance": float(index) + 0.125,
                "next_lie": "hole" if index % 11 == 0 else "green",
                "sg": (index - 48) / 100.0,
                "cost": 2 if index % 13 == 0 else 1,
                "holed": index % 11 == 0,
                "sg_weight_valid": index % 17 != 0,
            }
        )
    return pd.DataFrame(rows)


def test_synthetic_trace_preserves_existing_batch_result_and_exact_donor_mass() -> None:
    frame = _synthetic_traced_frame()
    canonical = frame.loc[:, list(MARKOV_TRANSITION_COLUMNS)].copy()
    model = MarkovHoleModel(
        canonical,
        canonical.iloc[0:0].copy(),
        min_exact=4,
        knn=64,
    )
    rng = np.random.default_rng(20260821)
    size = 137
    query = {
        "round_no": np.ones(size, dtype=int),
        "hole_no": np.ones(size, dtype=int),
        "category": np.full(size, "arg"),
        "lie": np.full(size, "fairway"),
        "distance_yd": rng.choice(np.asarray([9.0, 10.0, 11.0]), size=size),
        "common_uniform": rng.random(size),
    }
    query["common_uniform"][:4] = [
        0.0,
        np.nextafter(0.0, 1.0),
        0.5,
        np.nextafter(1.0, 0.0),
    ]
    reference = BatchedPopulationTransitionSampler(model, chunk_size=17)
    expected = reference.sample_many(**query)
    traced_sampler = TracedBatchedPopulationTransitionSampler(
        model,
        course_transitions=frame,
        chunk_size=17,
    )
    actual = traced_sampler.sample_many(**query)

    assert results_equal(actual.transition, expected)
    assert np.array_equal(actual.base_hierarchy_mass, actual.normalized_knn_probability)
    assert np.all(actual.normalized_knn_probability > 0.0)
    assert np.all(actual.normalized_knn_probability <= 1.0)
    assert np.allclose(
        actual.cdf_upper - actual.cdf_lower,
        actual.normalized_knn_probability,
        rtol=0.0,
        atol=2e-16,
    )
    assert np.all(actual.cdf_lower <= query["common_uniform"])
    assert np.all(query["common_uniform"] < actual.cdf_upper)
    assert set(actual.pool_index_level) == {"exact"}
    assert set(actual.pool_key) == {'[1,1,"arg","fairway",0]'}
    assert set(actual.query_state_key) == {'[1,1,"arg","fairway",0]'}
    assert set(actual.source_scope) == {"course"}

    by_id = frame.set_index("transition_row_id")
    chosen = by_id.loc[actual.donor_transition_row_id].reset_index()
    assert np.array_equal(actual.source_shot_key, chosen["shot_key"].to_numpy())
    assert np.array_equal(actual.source_event_key, chosen["event_key"].to_numpy())
    assert np.array_equal(actual.source_player_id, chosen["source_player_id"].to_numpy())
    assert np.array_equal(actual.source_stroke_no, chosen["stroke_no"].to_numpy(np.int16))
    assert np.array_equal(actual.donor_next_lie, actual.transition.next_lie)
    assert np.array_equal(actual.donor_next_distance, actual.transition.next_distance)
    assert np.array_equal(actual.donor_cost, actual.transition.cost)
    assert np.array_equal(actual.donor_holed, actual.transition.holed)
    assert len(actual.donor_frame()) == size


def test_pool_registry_and_frame_comparison_fail_closed_on_reordering_or_mutation() -> None:
    frame = _synthetic_traced_frame(12)
    canonical = frame.loc[:, list(MARKOV_TRANSITION_COLUMNS)].copy()
    model = MarkovHoleModel(
        canonical,
        canonical.iloc[0:0].copy(),
        min_exact=4,
        knn=8,
    )
    reordered = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(TransitionTraceParityError, match="Pool row order/value mismatch"):
        TransitionPoolTraceRegistry(model, course_transitions=reordered)

    mutated = frame.copy()
    mutated.loc[0, "next_distance"] += 1.0
    with pytest.raises(TransitionTraceParityError, match="reconstruction diverged"):
        assert_markov_frame_parity(canonical, mutated)


def test_real_travelers_single_prior_edition_trace_is_bounded_and_exact() -> None:
    required = [
        REAL_EXPORT / "pga_2025_shots.parquet",
        REAL_EXPORT / "pga_2025_shot_sg.parquet",
        REAL_EXPORT / "pga_2026_shots.parquet",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Local frozen ShotLink export is unavailable")

    prior = pd.DataFrame(
        [{"year": 2025, "event_key": "pga:R2025034"}]
    )
    bundle = build_strict_prior_course_trace_bundle(
        REAL_EXPORT,
        prior,
        "pga:R2026034",
        categories=("arg", "putt"),
        min_exact=12,
        knn=64,
    )
    assert set(bundle.transitions["event_key"]) == {"pga:R2025034"}
    assert set(bundle.transitions["category"]) == {"arg", "putt"}
    assert bundle.transitions["shot_key"].notna().all()
    assert bundle.transitions["shot_key"].is_unique
    assert set(bundle.target_raw["event_key"]) == {"pga:R2026034"}

    # Select only a tiny real fixture from states with enough same-lie course
    # support to make this an exact-course trace test, not a broad simulation.
    counts = bundle.transitions.groupby(
        ["category", "lie"], observed=True
    )["transition_row_id"].transform("size")
    fixture = (
        bundle.transitions.loc[counts.ge(4)]
        .groupby("category", observed=True, sort=False)
        .head(6)
        .reset_index(drop=True)
    )
    assert 2 <= len(fixture) <= 12
    uniforms = np.linspace(0.05, 0.95, len(fixture), dtype=float)
    query = {
        "round_no": fixture["round_no"].to_numpy(int),
        "hole_no": fixture["hole_no"].to_numpy(int),
        "category": fixture["category"].astype(str).to_numpy(),
        "lie": fixture["lie"].astype(str).to_numpy(),
        "distance_yd": fixture["start_distance"].to_numpy(float),
        "common_uniform": uniforms,
    }
    reference = BatchedPopulationTransitionSampler(bundle.model, chunk_size=5)
    expected = reference.sample_many(**query)
    actual = bundle.traced_sampler(chunk_size=5).sample_many(**query)

    assert results_equal(actual.transition, expected)
    assert actual.transition.available.all()
    assert set(actual.source_event_key) == {"pga:R2025034"}
    assert np.all(actual.source_shot_key != "")
    assert np.all(actual.donor_transition_row_id >= 0)
    assert np.all(actual.donor_pool_row >= 0)
    assert np.all(actual.normalized_knn_probability > 0.0)
    assert np.all(actual.cdf_lower <= uniforms)
    assert np.all(uniforms < actual.cdf_upper)
    assert np.array_equal(actual.donor_next_lie, actual.transition.next_lie)
    assert np.array_equal(actual.donor_next_distance, actual.transition.next_distance)
    assert np.array_equal(actual.donor_cost, actual.transition.cost)
    assert np.array_equal(actual.donor_holed, actual.transition.holed)


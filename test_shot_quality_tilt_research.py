import argparse

import numpy as np
import pandas as pd
import pytest

from archive.evaluate_shot_quality_tilt import (
    PriorPuttingCurve,
    TiltAccumulator,
    iter_event_date_batches,
    score_event,
)


def _reward_frame(rewards, abilities):
    return pd.DataFrame(
        {
            "sg_category": "app",
            "exact_state": "app|fairway|100-150",
            "distance_backoff": "app|100-150",
            "reward_sg": rewards,
            "ability_sg_round": abilities,
            "reward_curve_sha256": "test_curve_hash",
        }
    )


def _score_args():
    return argparse.Namespace(
        state_prior=0.0,
        distance_prior=0.0,
        minimum_global_atoms=1,
        variance_floor_ratio=0.0,
        variance_floor_absolute=0.0,
        uniform_mass=0.0,
    )


def test_prior_putting_curve_is_monotone_and_strictly_prior():
    curve = PriorPuttingCurve()
    observations = pd.DataFrame(
        {
            "green_start_distance_yd": [1.0, 2.0, 3.0, 4.0],
            "green_remaining_strokes": [1.2, 2.1, 1.8, 2.6],
        }
    )
    curve.update(observations, event_start="2020-01-02")
    snapshot = curve.snapshot(1, as_of="2020-01-09")
    assert snapshot is not None
    assert np.all(np.diff(snapshot.distances_ft) >= 0)
    assert np.all(np.diff(snapshot.expected) >= -1e-12)
    with pytest.raises(AssertionError, match="same-date"):
        curve.snapshot(1, as_of="2020-01-02")
    with pytest.raises(AssertionError, match="chronologically"):
        curve.update(observations, event_start="2020-01-01")


def test_beta_zero_is_identical_across_donor_normalizations():
    betas = np.array([-0.5, 0.0, 0.5])
    edges = np.array([-1.5, -0.5, 0.5, 1.5])
    centers = np.array([-1.0, 0.0, 1.0])
    accumulator = TiltAccumulator(betas, centers, edges)
    accumulator.add_event(_reward_frame([-1.0, 0.0, 1.0], [0.7, -0.2, 1.0]))
    distributions = []
    for mode in ("event_normalized", "globally_normalized"):
        result = accumulator.base_distribution(
            "app", "app|fairway|100-150", "app|100-150", mode,
            0.0, 0.0, 1, 0.0, 0.0, 0.0,
        )
        assert result is not None
        distributions.append(result[0][1])
    np.testing.assert_allclose(distributions[0], distributions[1], atol=1e-12)
    np.testing.assert_allclose(distributions[0], np.repeat(1.0 / 3.0, 3), atol=1e-12)

    scored = score_event(
        _reward_frame([-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        accumulator,
        "event_normalized",
        _score_args(),
    )
    beta0 = scored.loc[np.isclose(scored["beta"], 0.0)].iloc[0]
    assert np.isfinite(beta0["logloss"])
    assert np.isfinite(beta0["crps"])


def test_donor_and_target_tilts_cancel_at_equal_ability():
    betas = np.array([0.0, 1.0])
    edges = np.array([-1.5, 0.0, 1.5])
    centers = np.array([-0.75, 0.75])
    accumulator = TiltAccumulator(betas, centers, edges)
    donors = _reward_frame([-0.75, 0.75], [1.0, 1.0])
    accumulator.add_event(donors)
    base = accumulator.base_distribution(
        "app", "app|fairway|100-150", "app|100-150", "event_normalized",
        0.0, 0.0, 1, 0.0, 0.0, 0.0,
    )
    assert base is not None
    base_probability = base[0][1]
    target_weight = np.exp(centers)
    recovered = base_probability * target_weight
    recovered /= recovered.sum()
    np.testing.assert_allclose(recovered, [0.5, 0.5], atol=1e-12)


def test_same_date_events_are_one_batch_and_curve_gate_rejects_early_update():
    calendar = pd.DataFrame(
        {
            "event_key": ["A", "B", "C"],
            "event_start": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-09"]),
        }
    )
    batches = list(iter_event_date_batches(calendar))
    assert [set(group["event_key"]) for _, group in batches] == [{"A", "B"}, {"C"}]

    curve = PriorPuttingCurve()
    observations = pd.DataFrame(
        {"green_start_distance_yd": [2.0], "green_remaining_strokes": [2.0]}
    )
    curve.update(observations, event_start="2020-01-02")
    with pytest.raises(AssertionError, match="same-date"):
        curve.snapshot(1, as_of="2020-01-02")
    assert curve.snapshot(1, as_of="2020-01-09") is not None

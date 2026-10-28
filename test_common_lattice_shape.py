from __future__ import annotations

import numpy as np
import pandas as pd

from archive.evaluate_common_lattice_shape import (
    REFERENCE_RELATIVE,
    SYMMETRIC_PHYSICAL,
    calibrate_common_pmf,
    calibrate_scenario_pmfs,
    finite_particle_latent_grid,
    frozen_market_like_pairs,
)


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    probability = np.asarray([0.10, 0.20, 0.40, 0.20, 0.10])
    common = {
        "event_key": "synthetic",
        "event": "Synthetic Event",
        "source_player_id": "1",
        "player_name": "alpha",
        "round_num": 1,
        "round_par": 70,
        "actual_strokes": 69.0,
        "actual_relative": 0.25,
        "field_scoring_center": 68.75,
        "target_sg": 0.50,
    }
    candidate = pd.DataFrame(
        [
            {
                **common,
                "arm": "spatial:population",
                "score_value": score,
                "probability": mass,
            }
            for score, mass in zip(range(68, 73), probability)
        ]
    )
    baseline = pd.DataFrame(
        [
            {
                **common,
                "arm": "category_shot",
                "score_value": score,
                "probability": mass,
            }
            for score, mass in zip(range(-2, 3), probability)
        ]
    )
    return candidate, baseline


def test_128_path_latent_grid_matches_adapter_sparsity_clamp() -> None:
    nodes, weights = finite_particle_latent_grid(128, 31)

    assert len(nodes) == 5
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights @ nodes, 0.0, atol=1e-12)
    assert np.isclose(weights @ (nodes**2), 1.0, atol=1e-12)
    assert sorted((weights * 128).astype(int).tolist()) == [25, 25, 26, 26, 26]


def test_common_calibrator_translates_before_exact_moment_solve() -> None:
    nodes, weights = finite_particle_latent_grid(128, 31)
    support = np.asarray([68, 69, 70, 71, 72], float)
    probability = np.asarray([0.05, 0.20, 0.45, 0.20, 0.10])
    raw_mean = float(probability @ support)
    raw_variance = float(probability @ ((support - raw_mean) ** 2))

    calibrated_support, calibrated_probability, audit = calibrate_common_pmf(
        support,
        probability,
        target_mean=66.25,
        week_latent_sd=0.65,
        latent_nodes=nodes,
        node_weights=weights,
    )

    shift = int(np.rint(66.25 - raw_mean))
    assert shift == -4
    assert np.array_equal(calibrated_support, support + shift)
    assert np.isclose(calibrated_probability.sum(), 1.0)
    mean = float(calibrated_probability @ calibrated_support)
    variance = float(
        calibrated_probability @ ((calibrated_support - mean) ** 2)
    )
    assert np.isclose(mean, 66.25, atol=1e-8)
    assert np.isclose(variance, raw_variance, atol=1e-6)
    assert abs(audit["post_shift_residual"]) <= 0.5


def test_symmetric_physical_puts_identical_raw_shapes_on_identical_support() -> None:
    candidate, baseline = _raw_frames()
    nodes, weights = finite_particle_latent_grid(128, 31)

    pmfs, diagnostics = calibrate_scenario_pmfs(
        candidate,
        baseline,
        scenario=SYMMETRIC_PHYSICAL,
        week_latent_sd=0.65,
        latent_nodes=nodes,
        node_weights=weights,
    )
    left = pmfs[pmfs["arm"].eq("category_shot")].sort_values("score_value")
    right = pmfs[pmfs["arm"].eq("spatial:population")].sort_values("score_value")

    assert np.array_equal(
        left["score_value"].to_numpy(), right["score_value"].to_numpy()
    )
    assert np.allclose(
        left["probability"].to_numpy(), right["probability"].to_numpy()
    )
    assert set(diagnostics["adapter_policy"]) == {
        "common_discrete_location_week_latent"
    }


def test_reference_relative_leaves_frozen_baseline_pmf_unchanged() -> None:
    candidate, baseline = _raw_frames()
    nodes, weights = finite_particle_latent_grid(128, 31)

    pmfs, diagnostics = calibrate_scenario_pmfs(
        candidate,
        baseline,
        scenario=REFERENCE_RELATIVE,
        week_latent_sd=0.65,
        latent_nodes=nodes,
        node_weights=weights,
    )
    frozen = pmfs[pmfs["arm"].eq("category_shot")].sort_values("score_value")
    original = baseline.sort_values("score_value")

    assert np.array_equal(
        frozen["score_value"].to_numpy(), original["score_value"].to_numpy()
    )
    assert np.array_equal(
        frozen["probability"].to_numpy(), original["probability"].to_numpy()
    )
    baseline_diagnostic = diagnostics[diagnostics["arm"].eq("category_shot")]
    candidate_diagnostic = diagnostics[
        diagnostics["arm"].eq("spatial:population")
    ]
    assert baseline_diagnostic["adapter_policy"].iat[0] == "frozen_reference_unchanged"
    assert (
        candidate_diagnostic["adapter_policy"].iat[0]
        == "common_discrete_location_week_latent"
    )


def test_market_like_pair_selector_is_frozen_from_raw_category_current() -> None:
    rows: list[dict[str, object]] = []
    player_pmfs = {
        "alpha": ([-1, 0, 1], [0.25, 0.50, 0.25], 69.0),
        "bravo": ([-1, 0, 1], [0.25, 0.50, 0.25], 70.0),
        "charlie": ([-10, -9], [0.50, 0.50], 71.0),
    }
    for player, (support, probability, actual) in player_pmfs.items():
        for score, mass in zip(support, probability):
            rows.append(
                {
                    "event": "Synthetic Event",
                    "round_num": 1,
                    "player_name": player,
                    "arm": "category_current",
                    "score_value": score,
                    "probability": mass,
                    "actual_strokes": actual,
                }
            )

    pairs = frozen_market_like_pairs(pd.DataFrame(rows))

    assert len(pairs) == 1
    assert pairs.iloc[0]["left"] == "alpha"
    assert pairs.iloc[0]["right"] == "bravo"
    assert np.isclose(pairs.iloc[0]["selector_probability"], 0.5)

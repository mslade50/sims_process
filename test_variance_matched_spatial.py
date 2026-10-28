from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archive.evaluate_common_lattice_shape import finite_particle_latent_grid
from archive.evaluate_variance_matched_spatial import (
    LATENT_BINS,
    calibrate_variance_matched_pmfs,
    category_shot_variance_lookup,
)


def _case_metadata() -> dict[str, object]:
    return {
        "event_key": "synthetic",
        "event": "Synthetic Event",
        "source_player_id": "1",
        "player_name": "alpha",
        "round_num": 1,
        "round_par": 70,
        "actual_strokes": 69.0,
        "field_scoring_center": 68.75,
        "target_sg": 0.50,
    }


def _pmf_rows(
    arm: str,
    support: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    common = _case_metadata()
    return pd.DataFrame(
        [
            {
                **common,
                "arm": arm,
                "score_value": float(score),
                "probability": float(mass),
            }
            for score, mass in zip(support, probability)
        ]
    )


def _moments(frame: pd.DataFrame) -> tuple[float, float]:
    support = frame["score_value"].to_numpy(float)
    probability = frame["probability"].to_numpy(float)
    mean = float(probability @ support)
    variance = float(probability @ ((support - mean) ** 2))
    return mean, variance


def test_spatial_arms_match_exact_case_reference_variance_on_translated_support() -> None:
    baseline_support = np.arange(-5, 6, dtype=float)
    baseline_probability = np.exp(-(baseline_support**2) / 10.0)
    baseline_probability /= baseline_probability.sum()
    baseline = _pmf_rows(
        "category_shot", baseline_support, baseline_probability
    )

    narrow_support = np.arange(66, 75, dtype=float)
    narrow_probability = np.asarray(
        [0.01, 0.03, 0.08, 0.18, 0.40, 0.18, 0.08, 0.03, 0.01]
    )
    wide_support = np.arange(63, 78, dtype=float)
    wide_probability = np.exp(-((wide_support - 70.0) ** 2) / 18.0)
    wide_probability /= wide_probability.sum()
    candidate = pd.concat(
        [
            _pmf_rows("spatial:narrow", narrow_support, narrow_probability),
            _pmf_rows("spatial:wide", wide_support, wide_probability),
        ],
        ignore_index=True,
    )
    nodes, weights = finite_particle_latent_grid(128, LATENT_BINS)

    pmfs, audits, reference_cases = calibrate_variance_matched_pmfs(
        candidate,
        baseline,
        week_latent_sd=0.65,
        latent_nodes=nodes,
        node_weights=weights,
    )

    assert len(reference_cases) == 1
    reference_variance = float(
        reference_cases["category_shot_reference_variance"].iat[0]
    )
    target_mean = 68.25
    spatial_audits = audits[audits["arm"].str.startswith("spatial:")]
    assert set(spatial_audits["variance_policy"]) == {
        "exact_case_category_shot"
    }
    assert set(spatial_audits["power_at_search_boundary"]) == {False}
    assert np.allclose(spatial_audits["expected_variance"], reference_variance)
    assert np.allclose(spatial_audits["expected_mean"], target_mean)

    raw_by_arm = {
        "spatial:narrow": narrow_support,
        "spatial:wide": wide_support,
    }
    for arm, raw_support in raw_by_arm.items():
        calibrated = pmfs[pmfs["arm"].eq(arm)].sort_values("score_value")
        mean, variance = _moments(calibrated)
        audit = spatial_audits[spatial_audits["arm"].eq(arm)].iloc[0]
        shift = int(audit["integer_support_shift"])
        assert np.array_equal(
            calibrated["score_value"].to_numpy(float), raw_support + shift
        )
        assert np.isclose(mean, target_mean, atol=1e-8)
        assert np.isclose(variance, reference_variance, atol=1e-6)

    narrow_raw_variance = float(
        narrow_probability
        @ ((narrow_support - narrow_probability @ narrow_support) ** 2)
    )
    wide_raw_variance = float(
        wide_probability @ ((wide_support - wide_probability @ wide_support) ** 2)
    )
    assert narrow_raw_variance < reference_variance < wide_raw_variance


def test_reference_variance_uses_physical_support_but_translation_does_not_change_it() -> None:
    relative_support = np.asarray([-3.0, -1.0, 0.0, 2.0, 4.0])
    probability = np.asarray([0.05, 0.20, 0.40, 0.25, 0.10])
    baseline = _pmf_rows("category_shot", relative_support, probability)

    lookup, cases = category_shot_variance_lookup(baseline)

    relative_mean = float(probability @ relative_support)
    expected_variance = float(
        probability @ ((relative_support - relative_mean) ** 2)
    )
    assert len(lookup) == 1
    assert np.isclose(next(iter(lookup.values())), expected_variance)
    assert np.isclose(
        cases["category_shot_raw_mean_physical"].iat[0],
        relative_mean + 70.0,
    )


def test_unattainable_spatial_variance_fails_instead_of_accepting_boundary_fit() -> None:
    baseline = _pmf_rows(
        "category_shot",
        np.asarray([-5.0, 5.0]),
        np.asarray([0.5, 0.5]),
    )
    candidate = _pmf_rows(
        "spatial:too_narrow",
        np.asarray([68.0, 69.0]),
        np.asarray([0.5, 0.5]),
    )
    nodes, weights = finite_particle_latent_grid(128, LATENT_BINS)

    with pytest.raises(AssertionError, match="variance error"):
        calibrate_variance_matched_pmfs(
            candidate,
            baseline,
            week_latent_sd=0.65,
            latent_nodes=nodes,
            node_weights=weights,
        )


def test_requested_128_particle_quadrature_is_exactly_five_nodes() -> None:
    nodes, weights = finite_particle_latent_grid(128, LATENT_BINS)

    assert len(nodes) == 5
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights @ nodes, 0.0, atol=1e-12)
    assert np.isclose(weights @ (nodes**2), 1.0, atol=1e-12)
    assert sorted((weights * 128).astype(int).tolist()) == [25, 25, 26, 26, 26]

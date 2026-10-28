from __future__ import annotations

import numpy as np

from archive.hole_sim_calibration_adapter import (
    anchor_discrete_pmf_with_week_latent,
    anchor_discrete_rounds_with_week_latent,
    empirical_integer_pmf,
    variance_ratio,
)


def _off_diagonal_covariance(values: np.ndarray) -> float:
    output: list[float] = []
    for player in range(values.shape[0]):
        covariance = np.cov(values[player], bias=True)
        output.extend(covariance[np.triu_indices(values.shape[1], k=1)].tolist())
    return float(np.mean(output))


def test_twenty_thousand_particle_sparse_tails_do_not_underflow() -> None:
    rng = np.random.default_rng(20260820)
    raw = np.rint(
        70.0 + 2.8 * rng.standard_normal((2, 4, 20_000))
    ).astype(np.int16)
    anchored, diagnostics = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=70.0,
        target_sg=np.zeros((2, 4)),
        week_latent_sd=0.65,
        seed=91,
    )

    assert np.issubdtype(anchored.dtype, np.integer)
    assert float(diagnostics["max_abs_expected_mean_error"]) < 1e-6
    assert float(diagnostics["max_abs_expected_variance_error"]) < 1e-5
    for player in range(2):
        for round_index in range(4):
            shift = int(diagnostics["integer_support_shift"][player, round_index])
            assert set(np.unique(anchored[player, round_index])).issubset(
                set(np.unique(raw[player, round_index] + shift))
            )
            assert abs(
                float(
                    diagnostics["post_shift_residual_mean_adjustment"][
                        player, round_index
                    ]
                )
            ) <= 0.5 + 1e-12


def test_adapter_preserves_strong_raw_cross_round_copula() -> None:
    rng = np.random.default_rng(7)
    one_round = np.rint(
        70.0 + 2.8 * rng.standard_normal((2, 1, 20_000))
    ).astype(np.int16)
    raw = np.repeat(one_round, 4, axis=1)
    anchored, _ = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=70.0,
        target_sg=np.zeros((2, 4)),
        week_latent_sd=0.65,
        seed=11,
    )

    raw_covariance = _off_diagonal_covariance(raw.astype(float))
    output_covariance = _off_diagonal_covariance(anchored.astype(float))
    assert output_covariance / raw_covariance > 0.95
    assert float(variance_ratio(anchored).mean()) > 3.8


def test_independent_raw_rounds_gain_only_shared_week_covariance() -> None:
    rng = np.random.default_rng(17)
    raw = np.rint(
        70.0 + 2.8 * rng.standard_normal((3, 4, 20_000))
    ).astype(np.int16)
    anchored, _ = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=70.0,
        target_sg=np.zeros((3, 4)),
        week_latent_sd=0.65,
        seed=23,
    )

    covariance = _off_diagonal_covariance(anchored.astype(float))
    assert abs(covariance - 0.65**2) < 0.08


def test_clipped_week_nodes_still_preserve_exact_marginal_mean() -> None:
    rng = np.random.default_rng(5)
    raw = np.rint(
        70.0 + 1.5 * rng.standard_normal((1, 4, 128))
    ).astype(np.int16)
    _, diagnostics = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=np.full((1, 4), 68.0),
        target_sg=np.full((1, 4), 0.7),
        week_latent_sd=4.0,
        seed=3,
    )

    assert float(np.max(diagnostics["conditional_mean_clipped_mass"])) > 0.0
    assert float(diagnostics["max_abs_expected_mean_error"]) < 1e-8


def test_large_skill_shift_translates_support_before_shape_calibration() -> None:
    raw = np.asarray(
        [[[
            68, 69, 69, 70, 70, 70, 71, 71,
            68, 69, 69, 70, 70, 70, 71, 72,
        ]]],
        dtype=np.int16,
    )
    anchored, diagnostics = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=np.asarray([[66.25]]),
        target_sg=np.asarray([[0.0]]),
        week_latent_sd=0.0,
        seed=13,
    )

    shift = int(diagnostics["integer_support_shift"][0, 0])
    assert shift == -4
    assert set(np.unique(anchored[0, 0])).issubset(
        set(np.unique(raw[0, 0] + shift))
    )
    assert abs(
        float(diagnostics["post_shift_residual_mean_adjustment"][0, 0])
    ) <= 0.5
    assert float(diagnostics["max_abs_expected_mean_error"]) < 1e-8
    assert float(diagnostics["max_abs_expected_variance_error"]) < 1e-5


def test_marginal_helper_matches_128_particle_latent_quadrature() -> None:
    rng = np.random.default_rng(88)
    raw = np.rint(70.0 + 2.5 * rng.standard_normal((1, 1, 128))).astype(
        np.int16
    )
    _, joint_diagnostics = anchor_discrete_rounds_with_week_latent(
        raw,
        round_pars=np.asarray([[68.4]]),
        target_sg=np.asarray([[0.3]]),
        week_latent_sd=0.65,
        seed=29,
        latent_bins=31,
    )
    support, probability = empirical_integer_pmf(raw[0, 0])
    marginal, marginal_diagnostics = anchor_discrete_pmf_with_week_latent(
        support,
        probability,
        target_strokes=68.1,
        week_latent_sd=0.65,
        latent_bins=31,
        latent_particles=128,
    )

    assert np.array_equal(
        np.asarray(marginal_diagnostics["calibrated_support"], int),
        np.asarray(joint_diagnostics["marginal_support"][0, 0], int),
    )
    assert np.allclose(
        marginal,
        np.asarray(joint_diagnostics["marginal_probability"][0, 0], float),
        atol=1e-12,
        rtol=0.0,
    )

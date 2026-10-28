"""Research property test for live single-round category-profile invariance.

The production single-round kernel re-centers every four-category mean vector
to the separately supplied total-skill estimate.  Consequently, changing only
the zero-sum category profile must not change integer round scores when the
kernel is rerun with the same random seed (outside the category clipping edge
case).

This test intentionally defines the two research transformations locally so
it does not change or couple production code:

* blanket: retain 65% of every category's deviation from the player mean;
* targeted: apply the same 35% move only to ARG/PUTT and compensate their
  combined move equally in OTT/APP.

Both use the proposed common-vector 0.15 SG adjustment cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

import sims_kernel


CATEGORIES = ("ott", "app", "arg", "putt")
PROFILE_FACTOR = 0.65
ADJUSTMENT_CAP = 0.15
N_PLAYERS = 32
N_SIMS = 8_192
FIELD_SEED = 20_260_825
KERNEL_SEED = 789


@dataclass(frozen=True)
class SyntheticField:
    mu: np.ndarray
    std_course: np.ndarray
    eff_skew: np.ndarray
    l_corr: np.ndarray
    skill: np.ndarray
    wx_delta: np.ndarray
    player_avg: np.ndarray


def _sum4(values: np.ndarray) -> np.ndarray:
    """Use the same left-to-right four-value summation order as the kernel."""

    return ((values[:, 0] + values[:, 1]) + values[:, 2]) + values[:, 3]


def _ulp_neighbors(value: np.float64, radius: int = 16):
    """Yield ``value`` followed by its nearest binary64 neighbors."""

    yield value
    lower = value
    upper = value
    for _ in range(radius):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        yield lower
        yield upper


def _lock_sum_exactly(values: np.ndarray, target_sum: np.ndarray) -> np.ndarray:
    """Normalize sub-ULP arithmetic residue so each row sum is bit-identical.

    The profile rules are exactly zero-sum over real numbers. Binary64 addition
    can nevertheless leave a 1-ULP residue after applying an adjustment. This
    searches only adjacent representable ARG/PUTT values and removes that
    numerical residue (the largest nudge is many orders below model precision).
    """

    locked = np.array(values, dtype=np.float64, copy=True)
    for row in np.flatnonzero(_sum4(locked) != target_sum):
        original_arg = locked[row, 2]
        found = False
        for candidate_arg in _ulp_neighbors(original_arg):
            first_three = (
                (locked[row, 0] + locked[row, 1]) + candidate_arg
            )
            putt_center = np.float64(target_sum[row] - first_three)
            for candidate_putt in _ulp_neighbors(putt_center):
                if np.float64(first_three + candidate_putt) == target_sum[row]:
                    locked[row, 2] = candidate_arg
                    locked[row, 3] = candidate_putt
                    found = True
                    break
            if found:
                break
        if not found:
            raise AssertionError("could not normalize zero-sum floating residue")
    np.testing.assert_array_equal(_sum4(locked), target_sum)
    return locked


def _apply_common_vector_cap(raw_adjustment: np.ndarray) -> tuple[np.ndarray, int]:
    row_max = np.max(np.abs(raw_adjustment), axis=1)
    scale = np.ones_like(row_max)
    cap_bound = row_max > ADJUSTMENT_CAP
    scale[cap_bound] = ADJUSTMENT_CAP / row_max[cap_bound]
    return raw_adjustment * scale[:, None], int(cap_bound.sum())


def _blanket_centered_profile(mu: np.ndarray) -> tuple[np.ndarray, int]:
    target_sum = _sum4(mu)
    player_mean = target_sum[:, None] / 4.0
    raw_adjustment = (PROFILE_FACTOR - 1.0) * (mu - player_mean)
    adjustment, cap_bound_players = _apply_common_vector_cap(raw_adjustment)
    return _lock_sum_exactly(mu + adjustment, target_sum), cap_bound_players


def _targeted_arg_putt_profile(mu: np.ndarray) -> tuple[np.ndarray, int]:
    target_sum = _sum4(mu)
    player_mean = target_sum[:, None] / 4.0
    raw_adjustment = np.zeros_like(mu)
    raw_adjustment[:, 2:] = (PROFILE_FACTOR - 1.0) * (
        mu[:, 2:] - player_mean
    )
    # Compensate the combined ARG/PUTT move equally into OTT and APP.
    compensation = -raw_adjustment[:, 2:].sum(axis=1) / 2.0
    raw_adjustment[:, 0] = compensation
    raw_adjustment[:, 1] = compensation
    adjustment, cap_bound_players = _apply_common_vector_cap(raw_adjustment)
    return _lock_sum_exactly(mu + adjustment, target_sum), cap_bound_players


TRANSFORMS: tuple[
    tuple[str, Callable[[np.ndarray], tuple[np.ndarray, int]]], ...
] = (
    ("blanket_centered_0.65", _blanket_centered_profile),
    ("targeted_arg_putt_0.65", _targeted_arg_putt_profile),
)


def _make_production_like_field() -> SyntheticField:
    rng = np.random.default_rng(FIELD_SEED)

    # Span weak alternates through elite players and give them heterogeneous
    # category profiles, including several cap-binding specialists.
    skill = np.linspace(-1.65, 2.35, N_PLAYERS)
    skill += 0.12 * np.sin(np.linspace(0.0, 3.0 * np.pi, N_PLAYERS))
    profile = rng.normal(0.0, 0.46, size=(N_PLAYERS, 4))
    profile[:4] += np.array(
        [
            [1.20, 0.25, -0.55, -0.90],
            [-0.75, 1.15, 0.20, -0.60],
            [-0.50, -0.25, 1.05, -0.30],
            [-0.85, -0.10, 0.10, 0.85],
        ]
    )
    profile -= profile.mean(axis=1, keepdims=True)
    mu = skill[:, None] / 4.0 + profile

    player_scale = rng.uniform(0.88, 1.12, size=(N_PLAYERS, 1))
    category_sd = np.array([0.86, 1.02, 0.72, 0.91])
    course_multiplier = np.array([1.18, 1.04, 0.94, 1.02])
    std_course = player_scale * category_sd * course_multiplier

    base_skew = np.array([-0.88, -0.24, -0.16, -0.05])
    eff_skew = base_skew + rng.normal(0.0, 0.055, size=(N_PLAYERS, 4))

    corr = np.array(
        [
            [1.00, 0.18, 0.08, -0.04],
            [0.18, 1.00, 0.14, -0.06],
            [0.08, 0.14, 1.00, -0.02],
            [-0.04, -0.06, -0.02, 1.00],
        ]
    )
    l_corr = np.linalg.cholesky(corr)
    wx_delta = np.linspace(-0.42, 0.46, N_PLAYERS)
    wx_delta += 0.04 * np.cos(np.linspace(0.0, 2.0 * np.pi, N_PLAYERS))
    player_avg = np.linspace(69.8, 72.4, N_PLAYERS)
    player_avg += rng.normal(0.0, 0.18, size=N_PLAYERS)

    return SyntheticField(
        mu=np.ascontiguousarray(mu),
        std_course=np.ascontiguousarray(std_course),
        eff_skew=np.ascontiguousarray(eff_skew),
        l_corr=np.ascontiguousarray(l_corr),
        skill=np.ascontiguousarray(skill),
        wx_delta=np.ascontiguousarray(wx_delta),
        player_avg=np.ascontiguousarray(player_avg),
    )


def _run_single_round(
    field: SyntheticField, mu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores, cat_mu = sims_kernel.run_single_round(
        np.ascontiguousarray(mu),
        field.std_course,
        field.eff_skew,
        field.l_corr,
        field.skill,
        field.wx_delta,
        field.player_avg,
        N_SIMS,
        KERNEL_SEED,
    )
    return np.asarray(scores), np.asarray(cat_mu)


@pytest.fixture(scope="module")
def synthetic_field() -> SyntheticField:
    return _make_production_like_field()


@pytest.fixture(scope="module")
def baseline_kernel_result(
    synthetic_field: SyntheticField,
) -> tuple[np.ndarray, np.ndarray]:
    return _run_single_round(synthetic_field, synthetic_field.mu)


@pytest.mark.parametrize(("name", "transform"), TRANSFORMS)
def test_zero_sum_profile_transform_preserves_total_exactly(
    synthetic_field: SyntheticField,
    name: str,
    transform: Callable[[np.ndarray], tuple[np.ndarray, int]],
) -> None:
    transformed, cap_bound_players = transform(synthetic_field.mu)
    baseline_sum = _sum4(synthetic_field.mu)
    transformed_sum = _sum4(transformed)
    adjustment = transformed - synthetic_field.mu
    max_sum_residue = float(np.max(np.abs(transformed_sum - baseline_sum)))

    np.testing.assert_array_equal(transformed_sum, baseline_sum)
    assert np.count_nonzero(adjustment) > 0
    assert np.max(np.abs(adjustment)) <= ADJUSTMENT_CAP + 64 * np.finfo(float).eps

    print(
        f"[{name}] exact_zero_sum=True max_abs_sum_residue={max_sum_residue:.3e} "
        f"mean_abs_category_move={np.mean(np.abs(adjustment)):.6f} "
        f"max_abs_category_move={np.max(np.abs(adjustment)):.6f} "
        f"cap_bound_players={cap_bound_players}/{N_PLAYERS}"
    )


@pytest.mark.parametrize(("name", "transform"), TRANSFORMS)
def test_run_single_round_scores_are_identical_with_common_rng(
    synthetic_field: SyntheticField,
    baseline_kernel_result: tuple[np.ndarray, np.ndarray],
    name: str,
    transform: Callable[[np.ndarray], tuple[np.ndarray, int]],
) -> None:
    baseline_scores, baseline_cat_mu = baseline_kernel_result
    transformed, _ = transform(synthetic_field.mu)
    candidate_scores, candidate_cat_mu = _run_single_round(
        synthetic_field, transformed
    )

    category_mean_difference = candidate_cat_mu - baseline_cat_mu
    differing_scores = int(np.count_nonzero(candidate_scores != baseline_scores))
    mean_abs_by_category = np.mean(np.abs(category_mean_difference), axis=0)

    # The profiles genuinely moved even though the total-score path did not.
    assert np.max(np.abs(category_mean_difference)) > 0.01
    np.testing.assert_allclose(
        _sum4(candidate_cat_mu),
        _sum4(baseline_cat_mu),
        rtol=0.0,
        atol=4 * np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(candidate_scores, baseline_scores)

    print(
        f"[{name}] score_arrays_identical=True "
        f"differing_scores={differing_scores}/{candidate_scores.size} "
        f"mean_abs_cat_mu_diff={np.mean(np.abs(category_mean_difference)):.6f} "
        f"max_abs_cat_mu_diff={np.max(np.abs(category_mean_difference)):.6f} "
        "mean_abs_by_category="
        + ",".join(
            f"{category}:{difference:.6f}"
            for category, difference in zip(CATEGORIES, mean_abs_by_category)
        )
    )

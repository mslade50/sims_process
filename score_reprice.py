"""Deterministic cached-score repricing for decimal scoring-average moves.

The round cache stores integer settlement draws.  Their pre-rounding latent
values are no longer available, so a fractional scoring-average shift cannot be
recovered exactly.  We use the same approximation as the odds-board UI:
probability is uniform inside each integer score's one-stroke rounding bin.
A fractional shift therefore moves a proportional amount of mass to the
adjacent settlement score.  Whole-stroke shifts remain exact.

This module is deliberately pure and dependency-light so every backend path can
share one implementation instead of subtly different rounding rules.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


FRACTIONAL_SCORE_REPRICE_METHOD = "uniform_rounding_bin_v1"
_INTEGER_TOLERANCE = 1e-9


def fractional_settlement_pmf(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return integer settlement scores and probability mass for shifted draws.

    ``values`` may be the original integer cache or that cache plus any uniform
    decimal shift.  A value ``69.3`` contributes 70% to 69 and 30% to 70, which
    is the exact overlap of its inferred latent rounding bin with those integer
    settlement bins.  The output is deterministic, sums to one, and preserves
    the input mean (apart from floating-point noise).
    """
    draws = np.asarray(values, dtype=np.float64).reshape(-1)
    if draws.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    if not np.isfinite(draws).all():
        raise ValueError("score draws contain non-finite values")

    # Floating-point addition can turn a mathematically integral shift into
    # 68.99999999999999. Snap those values before splitting mass so every
    # whole-stroke reprice has exact parity with the historical point-mass path.
    nearest = np.rint(draws)
    draws = np.where(np.abs(draws - nearest) <= _INTEGER_TOLERANCE, nearest, draws)

    lower = np.floor(draws).astype(np.int64)
    upper_weight = draws - lower
    lower_weight = 1.0 - upper_weight

    scores = np.concatenate((lower, lower + 1))
    weights = np.concatenate((lower_weight, upper_weight))
    keep = weights > 0.0
    scores = scores[keep]
    weights = weights[keep]

    unique_scores, inverse = np.unique(scores, return_inverse=True)
    mass = np.zeros(unique_scores.size, dtype=np.float64)
    np.add.at(mass, inverse, weights)
    mass /= float(draws.size)

    # Remove microscopic arithmetic drift without changing relative mass.
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("score probability mass is invalid")
    mass /= total
    return unique_scores, mass


def strict_under_probability(values: Iterable[Any], line: float) -> float:
    """Price a strict score under from the fractional settlement PMF.

    Sportsbook score props use half-stroke lines, so ``under 69.5`` settles on
    integer scores below 69.5. Fractional repricing is intentionally scoped to
    those no-push half-stroke markets.
    """
    line = float(line)
    if not np.isfinite(line):
        raise ValueError("score line must be finite")
    scores, mass = fractional_settlement_pmf(values)
    if scores.size == 0:
        return float("nan")
    return float(mass[scores < line].sum())


def uniformly_shift_score_tape(
    sim_dict: Mapping[str, Iterable[Any]], delta: float
) -> dict[str, np.ndarray]:
    """Shift every joint draw by one constant without changing its ordering.

    Matchups and 3-balls must retain the exact shared-draw copula under a field-
    wide scoring-baseline update. Keeping the values continuous here avoids any
    player-specific re-rounding; only marginal score-prop publishing applies the
    deterministic settlement-mass approximation above.
    """
    delta = float(delta)
    if not np.isfinite(delta):
        raise ValueError("score shift must be finite")
    return {
        str(player): np.asarray(values, dtype=np.float64) + delta
        for player, values in sim_dict.items()
    }


def score_est_requires_live_refresh(
    *, price_only: bool, score_shift_delta: float, completed_round: int
) -> bool:
    """Whether a cached decimal reprice must rebuild its live tournament tape."""
    return bool(
        price_only
        and abs(float(score_shift_delta)) > _INTEGER_TOLERANCE
        and int(completed_round) >= 1
    )

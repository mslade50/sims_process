"""Prototype fused proper-score kernel for the shot-quality evaluator.

Research helper only.  It leaves the chronological evaluator and its running
artifacts untouched.  The kernel fuses the four forecast arms, shares the
target exponential tilt, and computes ``sum(abs(y - center))`` from prefix
counts/sums instead of allocating one shot-by-reward-bin matrix per arm.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np


def target_sufficient_statistics(
    observed: np.ndarray,
    ability_codes: np.ndarray,
    centers: np.ndarray,
    edges: np.ndarray,
    physical_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scenario x ability x reward-bin counts and absolute sums."""
    observed = np.asarray(observed, dtype=float)
    ability_codes = np.asarray(ability_codes, dtype=np.intp)
    physical_mask = np.asarray(physical_mask, dtype=bool)
    levels = int(ability_codes.max(initial=-1)) + 1
    bins = len(centers)
    counts = np.zeros((2, levels, bins), dtype=float)
    absolute_sum = np.zeros_like(counts)
    group_sizes = np.zeros((2, levels), dtype=float)
    observed_bins = np.searchsorted(edges, observed, side="right") - 1
    observed_bins = np.clip(observed_bins, 0, bins - 1)
    insertion = np.searchsorted(centers, observed, side="left")

    for scenario, mask in enumerate((np.ones(len(observed), dtype=bool), physical_mask)):
        codes = ability_codes[mask]
        values = observed[mask]
        np.add.at(counts[scenario], (codes, observed_bins[mask]), 1.0)
        group_sizes[scenario] = np.bincount(codes, minlength=levels)

        side_count = np.zeros((levels, bins + 1), dtype=float)
        side_sum = np.zeros((levels, bins + 1), dtype=float)
        np.add.at(side_count, (codes, insertion[mask]), 1.0)
        np.add.at(side_sum, (codes, insertion[mask]), values)
        left_count = np.cumsum(side_count[:, :bins], axis=1)
        left_sum = np.cumsum(side_sum[:, :bins], axis=1)
        total_sum = side_sum.sum(axis=1, keepdims=True)
        right_count = group_sizes[scenario, :, None] - left_count
        right_sum = total_sum - left_sum
        absolute_sum[scenario] = (
            centers[None, :] * left_count
            - left_sum
            + right_sum
            - centers[None, :] * right_count
        )
    return counts, absolute_sum, group_sizes


def _reference_one_arm(
    base_probability: np.ndarray,
    abilities: np.ndarray,
    betas: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    absolute_sum: np.ndarray,
    group_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_log = np.clip(
        abilities[:, None, None]
        * betas[None, :, None]
        * centers[None, None, :],
        -30.0,
        30.0,
    )
    tilted = base_probability[None, :, :] * np.exp(target_log)
    tilted /= tilted.sum(axis=2, keepdims=True)
    log_sum = -np.einsum(
        "lbk,lk->b",
        np.log(np.maximum(tilted, 1e-300)),
        counts,
    )
    cumulative_before = np.cumsum(tilted, axis=2) - tilted
    cumulative_moment_before = (
        np.cumsum(tilted * centers[None, None, :], axis=2)
        - tilted * centers[None, None, :]
    )
    pair_distance = 2.0 * np.sum(
        tilted
        * (
            centers[None, None, :] * cumulative_before
            - cumulative_moment_before
        ),
        axis=2,
    )
    first_sum = np.einsum("lbk,lk->lb", tilted, absolute_sum)
    crps_sum = np.sum(
        first_sum - 0.5 * pair_distance * group_sizes[:, None], axis=0
    )
    return log_sum, crps_sum


def reference_four_arm_scores(
    base_probability: np.ndarray,
    abilities: np.ndarray,
    betas: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    absolute_sum: np.ndarray,
    group_sizes: np.ndarray,
    arm_scenario: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal arm-at-a-time equivalent of the current evaluator kernel."""
    arms, beta_count, _ = base_probability.shape
    log_sum = np.zeros((arms, beta_count), dtype=float)
    crps_sum = np.zeros_like(log_sum)
    for arm in range(arms):
        scenario = int(arm_scenario[arm])
        log_sum[arm], crps_sum[arm] = _reference_one_arm(
            base_probability[arm], abilities, betas, centers,
            counts[scenario], absolute_sum[scenario], group_sizes[scenario],
        )
    return log_sum, crps_sum


def threaded_four_arm_scores(
    base_probability: np.ndarray,
    abilities: np.ndarray,
    betas: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    absolute_sum: np.ndarray,
    group_sizes: np.ndarray,
    arm_scenario: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run independent, read-only arms concurrently without changing math."""
    arms = len(base_probability)
    with ThreadPoolExecutor(max_workers=arms) as pool:
        futures = []
        for arm in range(arms):
            scenario = int(arm_scenario[arm])
            futures.append(
                pool.submit(
                    _reference_one_arm,
                    base_probability[arm],
                    abilities,
                    betas,
                    centers,
                    counts[scenario],
                    absolute_sum[scenario],
                    group_sizes[scenario],
                )
            )
        results = [future.result() for future in futures]
    return np.stack([item[0] for item in results]), np.stack(
        [item[1] for item in results]
    )


def fused_four_arm_scores(
    base_probability: np.ndarray,
    abilities: np.ndarray,
    betas: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    absolute_sum: np.ndarray,
    group_sizes: np.ndarray,
    arm_scenario: np.ndarray,
    beta_chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Score all arms together while sharing each target exponential tilt."""
    arms, beta_count, _ = base_probability.shape
    log_sum = np.zeros((arms, beta_count), dtype=float)
    crps_sum = np.zeros_like(log_sum)
    arm_counts = counts[arm_scenario]
    arm_absolute = absolute_sum[arm_scenario]
    arm_sizes = group_sizes[arm_scenario]
    for start in range(0, beta_count, beta_chunk):
        stop = min(start + beta_chunk, beta_count)
        target_log = np.clip(
            abilities[:, None, None]
            * betas[None, start:stop, None]
            * centers[None, None, :],
            -30.0,
            30.0,
        )
        factor = np.exp(target_log)
        tilted = (
            base_probability[:, None, start:stop, :]
            * factor[None, :, :, :]
        )
        tilted /= tilted.sum(axis=3, keepdims=True)
        log_sum[:, start:stop] = -np.einsum(
            "alck,alk->ac",
            np.log(np.maximum(tilted, 1e-300)),
            arm_counts,
        )
        cumulative_before = np.cumsum(tilted, axis=3) - tilted
        cumulative_moment_before = (
            np.cumsum(tilted * centers[None, None, None, :], axis=3)
            - tilted * centers[None, None, None, :]
        )
        pair_distance = 2.0 * np.sum(
            tilted
            * (
                centers[None, None, None, :] * cumulative_before
                - cumulative_moment_before
            ),
            axis=3,
        )
        first_sum = np.einsum(
            "alck,alk->alc", tilted, arm_absolute
        )
        crps_sum[:, start:stop] = np.sum(
            first_sum - 0.5 * pair_distance * arm_sizes[:, :, None],
            axis=1,
        )
    return log_sum, crps_sum


def _direct_absolute_sum(
    observed: np.ndarray,
    ability_codes: np.ndarray,
    centers: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    levels = int(ability_codes.max(initial=-1)) + 1
    result = np.zeros((levels, len(centers)), dtype=float)
    np.add.at(
        result,
        ability_codes[mask],
        np.abs(observed[mask, None] - centers[None, :]),
    )
    return result


def benchmark(seed: int, repeats: int) -> None:
    rng = np.random.default_rng(seed)
    shots, levels, arms, beta_count, bins = 2400, 145, 4, 31, 240
    centers = np.linspace(-2.9875, 2.9875, bins)
    edges = np.linspace(-3.0, 3.0, bins + 1)
    betas = np.linspace(-0.5, 2.5, beta_count)
    observed = np.clip(rng.normal(0.0, 0.65, shots), -3.0, 3.0)
    ability_codes = rng.integers(0, levels, shots)
    physical = rng.random(shots) > 0.025
    abilities = rng.normal(0.0, 0.45, levels)
    base = rng.gamma(0.8, 1.0, size=(arms, beta_count, bins)) + 1e-5 / bins
    base /= base.sum(axis=2, keepdims=True)
    arm_scenario = np.array([0, 0, 1, 1], dtype=np.intp)

    started = time.perf_counter()
    counts, absolute_sum, group_sizes = target_sufficient_statistics(
        observed, ability_codes, centers, edges, physical
    )
    stats_seconds = time.perf_counter() - started
    for scenario, mask in enumerate((np.ones(shots, dtype=bool), physical)):
        direct = _direct_absolute_sum(observed, ability_codes, centers, mask)
        if not np.allclose(absolute_sum[scenario], direct, rtol=0.0, atol=2e-11):
            raise AssertionError("prefix absolute-sum statistics differ")

    reference_four_arm_scores(
        base, abilities, betas, centers, counts, absolute_sum, group_sizes,
        arm_scenario,
    )
    fused_four_arm_scores(
        base, abilities, betas, centers, counts, absolute_sum, group_sizes,
        arm_scenario,
    )
    threaded_four_arm_scores(
        base, abilities, betas, centers, counts, absolute_sum, group_sizes,
        arm_scenario,
    )
    started = time.perf_counter()
    for _ in range(repeats):
        reference = reference_four_arm_scores(
            base, abilities, betas, centers, counts, absolute_sum, group_sizes,
            arm_scenario,
        )
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        fused = fused_four_arm_scores(
            base, abilities, betas, centers, counts, absolute_sum, group_sizes,
            arm_scenario,
        )
    fused_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        threaded = threaded_four_arm_scores(
            base, abilities, betas, centers, counts, absolute_sum, group_sizes,
            arm_scenario,
        )
    threaded_seconds = time.perf_counter() - started
    for expected, actual in zip(reference, fused):
        if not np.allclose(expected, actual, rtol=2e-13, atol=2e-9):
            raise AssertionError(
                f"fused score differs: max={np.max(np.abs(expected - actual))}"
            )
    for expected, actual in zip(reference, threaded):
        if not np.array_equal(expected, actual):
            raise AssertionError("threaded per-arm score is not bit-identical")
    max_log = float(np.max(np.abs(reference[0] - fused[0])))
    max_crps = float(np.max(np.abs(reference[1] - fused[1])))
    print(f"target-statistics seconds: {stats_seconds:.6f}")
    print(f"reference seconds ({repeats}): {reference_seconds:.6f}")
    print(f"fused seconds ({repeats}): {fused_seconds:.6f}")
    print(f"threaded seconds ({repeats}): {threaded_seconds:.6f}")
    print(f"kernel speedup: {reference_seconds / fused_seconds:.3f}x")
    print(f"thread speedup: {reference_seconds / threaded_seconds:.3f}x")
    print(f"max log-score sum delta: {max_log:.3e}")
    print(f"max CRPS sum delta: {max_crps:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    benchmark(arguments.seed, arguments.repeats)

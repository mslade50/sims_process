"""Research-only mean and shared-latent adapter for spatial round samples.

This module is deliberately independent of ``new_sim.py`` and ``round_sim.py``.
It turns complete hole/round stroke particles into a shape-only distribution,
anchors every player/round to a frozen production SG mean, and adds the same
kind of player-week latent used by the production tournament simulator while
preserving each one-round variance.

It also reads a frozen production fixture and, when requested, reconstructs
the BMW full-strength shot-dispersion standard deviations from the frozen
feature-input file.  Nothing here prices, publishes, or writes a production
cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtri


CAT_ORDER = ("sg_ott", "sg_app", "sg_arg", "sg_putt")
SHORT_CATS = ("ott", "app", "arg", "putt")
DEFAULT_WEEK_LATENT_SD = 0.65
MIN_IDIOSYNCRATIC_VARIANCE_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_player(value: object) -> str:
    return str(value).casefold().strip()


def _as_player_round(value: np.ndarray | float, players: int, rounds: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full((players, rounds), float(array))
    if array.shape == (rounds,):
        return np.broadcast_to(array[None, :], (players, rounds)).copy()
    if array.shape == (players,):
        return np.broadcast_to(array[:, None], (players, rounds)).copy()
    if array.shape != (players, rounds):
        raise ValueError(
            f"Expected scalar, ({rounds},), ({players},), or ({players}, {rounds}); "
            f"got {array.shape}"
        )
    return array.copy()


@dataclass(frozen=True)
class FrozenProductionSpec:
    """Immutable shape/mean inputs needed by the research evaluator."""

    player_names: tuple[str, ...]
    category_mu: np.ndarray  # [player, category], sums to my_pred_base
    std_course: np.ndarray  # [player, category], optionally full dispersion
    effective_skew: np.ndarray  # [player, category]
    l_corr: np.ndarray  # [category, category]
    target_sg: np.ndarray  # [player, round], nominal pre-event SG means
    round_pars: np.ndarray  # [player, round], physical scoring par
    week_latent_sd: float
    metadata: Mapping[str, object]

    @property
    def n_players(self) -> int:
        return len(self.player_names)

    @property
    def n_rounds(self) -> int:
        return int(self.target_sg.shape[1])

    @property
    def correlation(self) -> np.ndarray:
        return self.l_corr @ self.l_corr.T

    @property
    def category_round_variance(self) -> np.ndarray:
        return np.einsum(
            "ij,jk,ik->i", self.std_course, self.correlation, self.std_course
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as source:
        return {key: source[key].copy() for key in source.files}


def apply_frozen_dispersion_variance(
    std_course: np.ndarray,
    players: tuple[str, ...],
    feature_inputs_path: Path,
    *,
    mode: str = "oos",
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a frozen field-mean-preserving variance overlay to course stds.

    ``bmw_feature_inputs.csv`` stores both the pre-overlay category variance and
    the already field-scaled candidate variance.  Multiplying ``std_course`` by
    ``sqrt(new/base)`` preserves the course multiplier already in the fixture.
    """

    if mode not in {"oos", "50"}:
        raise ValueError("dispersion mode must be 'oos' or '50'")
    frame = pd.read_csv(feature_inputs_path)
    if "player_name" not in frame:
        raise ValueError(f"{feature_inputs_path} lacks player_name")
    frame["player_name"] = frame["player_name"].map(normalise_player)
    if frame["player_name"].duplicated().any():
        raise ValueError(f"{feature_inputs_path} has duplicate player names")
    frame = frame.set_index("player_name")
    missing = sorted(set(players) - set(frame.index))
    if missing:
        raise ValueError(
            f"Frozen dispersion input misses {len(missing)} fixture players: {missing}"
        )

    output = np.asarray(std_course, dtype=float).copy()
    suffix = "blend_oos_var" if mode == "oos" else "blend50_var"
    diagnostics: dict[str, object] = {}
    for index, cat in enumerate(SHORT_CATS):
        base_col = f"{cat}_base_var"
        new_col = f"{cat}_{suffix}"
        if base_col not in frame or new_col not in frame:
            raise ValueError(
                f"{feature_inputs_path} lacks {base_col} or {new_col}"
            )
        base = frame.loc[list(players), base_col].to_numpy(float)
        new = frame.loc[list(players), new_col].to_numpy(float)
        if not np.isfinite(base).all() or not np.isfinite(new).all():
            raise ValueError(f"Non-finite frozen dispersion variance for {cat}")
        if np.any(base <= 0.0) or np.any(new <= 0.0):
            raise ValueError(f"Non-positive frozen dispersion variance for {cat}")
        ratio = np.sqrt(new / base)
        output[:, index] *= ratio
        diagnostics[cat] = {
            "std_ratio_min": float(ratio.min()),
            "std_ratio_mean": float(ratio.mean()),
            "std_ratio_max": float(ratio.max()),
        }
    diagnostics["mode"] = mode
    diagnostics["feature_inputs"] = str(feature_inputs_path.resolve())
    diagnostics["feature_inputs_sha256"] = sha256_file(feature_inputs_path)
    return np.ascontiguousarray(output), diagnostics


def infer_course_multipliers(
    fixture: Mapping[str, np.ndarray],
    dists_path: Path,
    players: tuple[str, ...],
    *,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Infer fixture course multipliers when its stds predate an overlay."""

    frame = pd.read_csv(dists_path)
    frame["player_name"] = frame["player_name"].map(normalise_player)
    std = frame.pivot(index="player_name", columns="category_clean", values="std")
    missing = sorted(set(players) - set(std.index))
    if missing:
        raise ValueError(f"Distribution file misses fixture players: {missing}")
    base = std.loc[list(players), list(CAT_ORDER)].to_numpy(float)
    ratios = np.asarray(fixture["std_course"], float) / base
    spread = np.ptp(ratios, axis=0)
    stable = bool(np.all(spread <= tolerance))
    return ratios.mean(axis=0), {
        "stable_player_invariant": stable,
        "max_player_spread": spread.tolist(),
        "dists_path": str(dists_path.resolve()),
        "dists_sha256": sha256_file(dists_path),
    }


def load_frozen_production_spec(
    fixture_path: Path,
    *,
    round_pars: np.ndarray | float = 70.0,
    week_latent_sd: float | None = None,
    dispersion_feature_inputs: Path | None = None,
    dispersion_mode: str = "oos",
    base_dists_path: Path | None = None,
) -> FrozenProductionSpec:
    """Load a new_sim dump without importing or executing production code."""

    fixture_path = Path(fixture_path)
    fixture = _load_npz(fixture_path)
    required = {
        "mu", "std_course", "eff_skew", "l_corr", "my_pred_base",
        "r2_mu", "r3_mu", "r4_mu", "player_names",
    }
    missing = required - set(fixture)
    if missing:
        raise ValueError(f"Frozen fixture lacks keys: {sorted(missing)}")

    players = tuple(normalise_player(value) for value in fixture["player_names"].tolist())
    if len(players) != len(set(players)):
        raise ValueError("Frozen fixture player names are not unique")
    n = len(players)
    mu = np.asarray(fixture["mu"], dtype=float)
    std_course = np.asarray(fixture["std_course"], dtype=float)
    skew = np.asarray(fixture["eff_skew"], dtype=float)
    l_corr = np.asarray(fixture["l_corr"], dtype=float)
    if mu.shape != (n, 4) or std_course.shape != (n, 4) or skew.shape != (n, 4):
        raise ValueError("Frozen category arrays must all have shape [players, 4]")
    if l_corr.shape != (4, 4):
        raise ValueError("Frozen l_corr must have shape [4, 4]")

    my_pred = np.asarray(fixture["my_pred_base"], dtype=float)
    if not np.allclose(mu.sum(axis=1), my_pred, atol=1e-10, rtol=0.0):
        raise ValueError("Fixture category means are not re-centered to my_pred_base")

    weather_r1 = np.asarray(fixture.get("weather_delta_r1", np.zeros(n)), dtype=float)
    weather_r2 = np.asarray(fixture.get("weather_delta_r2", np.zeros(n)), dtype=float)
    # new_sim's pre-event convention: positive weather delta is worse and is
    # subtracted from the total SG mean. R3/R4 have no pre-event weather arrays.
    target_sg = np.column_stack(
        [
            my_pred - weather_r1,
            np.asarray(fixture["r2_mu"], float) - weather_r2,
            np.asarray(fixture["r3_mu"], float),
            np.asarray(fixture["r4_mu"], float),
        ]
    )
    pars = _as_player_round(round_pars, n, 4)

    metadata: dict[str, object] = {
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": sha256_file(fixture_path),
        "category_order": list(CAT_ORDER),
        "mean_policy": "R1=my_pred-weather_r1; R2=r2_mu-weather_r2; R3=r3_mu; R4=r4_mu",
        "category_means_recentered_to_my_pred": True,
    }
    if base_dists_path is not None:
        mults, audit = infer_course_multipliers(
            fixture, Path(base_dists_path), players
        )
        metadata["inferred_course_multipliers"] = dict(zip(CAT_ORDER, mults.tolist()))
        metadata["course_multiplier_audit"] = audit
    if dispersion_feature_inputs is not None:
        std_course, dispersion_audit = apply_frozen_dispersion_variance(
            std_course,
            players,
            Path(dispersion_feature_inputs),
            mode=dispersion_mode,
        )
        metadata["dispersion_overlay"] = dispersion_audit
    else:
        metadata["dispersion_overlay"] = {"mode": "fixture_as_is"}

    latent = (
        float(week_latent_sd)
        if week_latent_sd is not None
        else float(fixture.get("week_latent_sd", DEFAULT_WEEK_LATENT_SD))
    )
    if latent < 0.0:
        raise ValueError("week_latent_sd must be non-negative")
    metadata["week_latent_sd"] = latent
    return FrozenProductionSpec(
        player_names=players,
        category_mu=np.ascontiguousarray(mu),
        std_course=np.ascontiguousarray(std_course),
        effective_skew=np.ascontiguousarray(skew),
        l_corr=np.ascontiguousarray(l_corr),
        target_sg=np.ascontiguousarray(target_sg),
        round_pars=np.ascontiguousarray(pars),
        week_latent_sd=latent,
        metadata=metadata,
    )


def aggregate_hole_strokes(hole_strokes: np.ndarray) -> np.ndarray:
    """Aggregate ``[player, round, hole, particle]`` to round strokes."""

    values = np.asarray(hole_strokes, dtype=float)
    if values.ndim != 4:
        raise ValueError(
            "hole_strokes must have shape [player, round, hole, particle]"
        )
    if not np.isfinite(values).all():
        raise ValueError("hole_strokes contains incomplete/non-finite particles")
    if values.shape[2] != 18:
        raise ValueError(f"Expected 18 complete holes per round; got {values.shape[2]}")
    return np.ascontiguousarray(values.sum(axis=2))


def _orthogonal_week_draw(
    centered_sg: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return unit-SD week draws orthogonal to every round residual vector.

    Orthogonalisation makes the adapter's marginal variance identity exact in
    a finite frozen particle set, rather than only true in expectation.
    """

    players, rounds, particles = centered_sg.shape
    if particles <= rounds + 2:
        raise ValueError("Need more particles than rounds + 2 for latent orthogonalisation")
    rng = np.random.default_rng(seed)
    output = np.empty((players, particles), dtype=float)
    ones = np.ones((particles, 1), dtype=float)
    for player in range(players):
        design = np.column_stack([ones, centered_sg[player].T])
        draw = rng.standard_normal(particles)
        coefficient, *_ = np.linalg.lstsq(design, draw, rcond=None)
        residual = draw - design @ coefficient
        residual -= residual.mean()
        sd = residual.std(ddof=0)
        if not np.isfinite(sd) or sd <= 1e-12:
            raise ValueError("Unable to construct an orthogonal week latent")
        output[player] = residual / sd
    return output


def anchor_continuous_and_add_week_latent(
    round_strokes: np.ndarray,
    *,
    round_pars: np.ndarray | float,
    target_sg: np.ndarray,
    week_latent_sd: float = DEFAULT_WEEK_LATENT_SD,
    seed: int = 1245,
    min_idiosyncratic_variance_fraction: float = MIN_IDIOSYNCRATIC_VARIANCE_FRACTION,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Anchor means and add a production-style shared player-week latent.

    Input/output shape is ``[player, round, particle]``. The raw spatial
    distribution supplies shape and marginal variance; its mean is removed.
    The week draw is shared across all four rounds for a player/particle. Raw
    residuals are shrunk so each one-round variance remains unchanged whenever
    the requested latent variance is feasible. The production floor of 5% is
    mirrored for lower-variance rows.
    """

    strokes = np.asarray(round_strokes, dtype=float)
    if strokes.ndim != 3:
        raise ValueError("round_strokes must have shape [player, round, particle]")
    if not np.isfinite(strokes).all():
        raise ValueError("round_strokes contains incomplete/non-finite particles")
    players, rounds, _ = strokes.shape
    pars = _as_player_round(round_pars, players, rounds)
    targets = _as_player_round(target_sg, players, rounds)
    if week_latent_sd < 0.0:
        raise ValueError("week_latent_sd must be non-negative")
    if not 0.0 < min_idiosyncratic_variance_fraction <= 1.0:
        raise ValueError("min_idiosyncratic_variance_fraction must be in (0, 1]")

    raw_sg = pars[:, :, None] - strokes
    centered = raw_sg - raw_sg.mean(axis=2, keepdims=True)
    raw_var = centered.var(axis=2, ddof=0)
    if np.any(raw_var <= 1e-12):
        where = np.argwhere(raw_var <= 1e-12).tolist()
        raise ValueError(f"Degenerate round distributions at player/round rows {where}")

    latent_var = float(week_latent_sd) ** 2
    shrink_sq = np.clip(
        1.0 - latent_var / raw_var,
        min_idiosyncratic_variance_fraction,
        1.0,
    )
    shrink = np.sqrt(shrink_sq)
    if week_latent_sd > 0.0:
        latent_z = _orthogonal_week_draw(centered, seed)
        latent = latent_z[:, None, :] * float(week_latent_sd)
    else:
        latent_z = np.zeros((players, strokes.shape[2]), dtype=float)
        latent = latent_z[:, None, :]

    calibrated_sg = (
        targets[:, :, None]
        + centered * shrink[:, :, None]
        + latent
    )
    calibrated_strokes = pars[:, :, None] - calibrated_sg
    output_var = calibrated_sg.var(axis=2, ddof=0)
    diagnostics: dict[str, np.ndarray | float] = {
        "raw_mean_sg": raw_sg.mean(axis=2),
        "target_mean_sg": targets,
        "output_mean_sg": calibrated_sg.mean(axis=2),
        "raw_round_variance": raw_var,
        "output_round_variance": output_var,
        "idiosyncratic_shrink": shrink,
        "week_latent_z": latent_z,
        "week_latent_sd": float(week_latent_sd),
        "variance_floor_rows": float(np.sum(shrink_sq <= min_idiosyncratic_variance_fraction + 1e-12)),
    }
    return np.ascontiguousarray(calibrated_strokes), diagnostics


def empirical_integer_pmf(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted integer support and empirical probabilities."""

    values = np.asarray(samples, dtype=float).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Discrete stroke samples contain non-finite values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-8, rtol=0.0):
        raise ValueError(
            "Discrete PMF anchoring requires integer stroke samples; use the "
            "continuous diagnostic adapter only for continuous SG samples"
        )
    support, counts = np.unique(rounded.astype(np.int16), return_counts=True)
    if len(support) < 2:
        raise ValueError("Discrete stroke distribution has degenerate support")
    probabilities = counts.astype(float) / counts.sum()
    return support.astype(float), probabilities


def _tilt_probabilities(
    support: np.ndarray,
    probabilities: np.ndarray,
    target_mean: float,
    *,
    power: float = 1.0,
) -> np.ndarray:
    """Power-temper then exponentially tilt a PMF to an exact target mean."""

    x = np.asarray(support, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(x) != len(p) or len(x) < 2 or np.any(p <= 0.0):
        raise ValueError("Invalid discrete PMF")
    if not x[0] < target_mean < x[-1]:
        if np.isclose(target_mean, x[0], atol=1e-12):
            output = np.zeros_like(p)
            output[0] = 1.0
            return output
        if np.isclose(target_mean, x[-1], atol=1e-12):
            output = np.zeros_like(p)
            output[-1] = 1.0
            return output
        raise ValueError(
            f"Target mean {target_mean:.6f} is outside PMF support "
            f"[{x[0]:.0f}, {x[-1]:.0f}]"
        )
    log_base = float(power) * np.log(p)

    def weights(parameter: float) -> np.ndarray:
        log_weight = log_base + parameter * x
        log_weight -= log_weight.max()
        weight = np.exp(log_weight)
        return weight / weight.sum()

    def difference(parameter: float) -> float:
        q = weights(parameter)
        return float(q @ x - target_mean)

    low, high = -1.0, 1.0
    while difference(low) > 0.0 and low > -1024.0:
        low *= 2.0
    while difference(high) < 0.0 and high < 1024.0:
        high *= 2.0
    parameter = brentq(difference, low, high, xtol=1e-12, rtol=1e-12)
    return weights(parameter)


def _pmf_moments(support: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    mean = float(probabilities @ support)
    variance = float(probabilities @ ((support - mean) ** 2))
    return mean, variance


def _standard_normal_nodes(
    particles: int,
    bins: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a finite, exactly standardized normal-quantile latent grid."""

    # Leave enough particles in every latent stratum for the conditional
    # integer PMF/copula remap below. Thirty-one bins is useful at production
    # particle counts but is pathological for a 48- or 64-particle research
    # smoke (one or two paths per conditional distribution).
    bins = max(3, min(int(bins), particles, max(3, particles // 25)))
    base_nodes = ndtri((np.arange(bins, dtype=float) + 0.5) / bins)
    labels = np.arange(particles, dtype=int) % bins
    rng.shuffle(labels)
    expanded = base_nodes[labels]
    expanded = (expanded - expanded.mean()) / expanded.std(ddof=0)
    # Every occurrence of a label maps to the same standardized node.
    nodes = np.array([expanded[labels == index][0] for index in range(bins)])
    counts = np.bincount(labels, minlength=bins)
    weights = counts.astype(float) / particles
    # Unequal final-bin counts can very slightly move weighted moments. Recenter
    # the unique nodes using their actual finite-sample weights.
    nodes -= weights @ nodes
    nodes /= np.sqrt(weights @ (nodes**2))
    return nodes, weights, labels


def _conditional_mixture(
    support: np.ndarray,
    empirical_probabilities: np.ndarray,
    target_strokes: float,
    week_nodes_sg: np.ndarray,
    node_weights: np.ndarray,
    power: float,
) -> tuple[np.ndarray, float, float, float]:
    """Build P(score | week node), returning weights and marginal moments."""

    # Mathematically, tilting the power-tempered PMF first to ``target_strokes``
    # and then to a week-conditional mean is equivalent to one exponential
    # tilt of ``empirical_probabilities ** power`` to that conditional mean.
    # Do it in one step.  Materialising the intermediate PMF can underflow its
    # rare integer-score tails to exact zero at large powers (real 20k-particle
    # round PMFs exercise this), after which the second tilt incorrectly
    # rejects or loses support that still has finite log mass.
    conditional = np.empty((len(week_nodes_sg), len(support)), dtype=float)
    clipped_mass = 0.0
    conditional_means = np.empty(len(week_nodes_sg), dtype=float)
    conditional_vars = np.empty(len(week_nodes_sg), dtype=float)
    epsilon = 1e-8
    for index, week_sg in enumerate(week_nodes_sg):
        # Positive player-week form improves SG and lowers strokes.
        desired = float(target_strokes - week_sg)
        clipped = float(np.clip(desired, support[0] + epsilon, support[-1] - epsilon))
        if clipped != desired:
            clipped_mass += float(node_weights[index])
        conditional[index] = _tilt_probabilities(
            support,
            empirical_probabilities,
            clipped,
            power=power,
        )
        conditional_means[index], conditional_vars[index] = _pmf_moments(
            support, conditional[index]
        )
    marginal_mean = float(node_weights @ conditional_means)
    marginal_variance = float(
        node_weights @ (
            conditional_vars + (conditional_means - marginal_mean) ** 2
        )
    )
    return conditional, marginal_mean, marginal_variance, clipped_mass


def _mean_calibrated_conditional_mixture(
    support: np.ndarray,
    empirical_probabilities: np.ndarray,
    target_strokes: float,
    week_nodes_sg: np.ndarray,
    node_weights: np.ndarray,
    power: float,
) -> tuple[np.ndarray, float, float, float]:
    """Solve the latent-mixture intercept so its marginal mean is exact.

    When the production target lies near the finite empirical support, one or
    more week-latent conditional means must be clipped. Centering those nodes
    naively on ``target_strokes`` then moves the marginal mean. The mixture is
    monotone in its pre-clipping center, so a one-dimensional intercept solve
    restores the exact production mean before the outer variance solve.
    """

    x = np.asarray(support, float)
    week = np.asarray(week_nodes_sg, float)
    if not x[0] < float(target_strokes) < x[-1]:
        raise ValueError(
            f"Target mean {target_strokes:.6f} is outside mixture support "
            f"[{x[0]:.0f}, {x[-1]:.0f}]"
        )
    cache: dict[float, tuple[np.ndarray, float, float, float]] = {}

    def mixture(center: float) -> tuple[np.ndarray, float, float, float]:
        key = float(center)
        if key not in cache:
            cache[key] = _conditional_mixture(
                x,
                empirical_probabilities,
                key,
                week,
                node_weights,
                power,
            )
        return cache[key]

    def difference(center: float) -> float:
        return mixture(center)[1] - float(target_strokes)

    padding = float(np.max(np.abs(week))) + max(float(x[-1] - x[0]), 1.0)
    low = float(x[0] - padding)
    high = float(x[-1] + padding)
    center = brentq(difference, low, high, xtol=1e-11, rtol=1e-11)
    return mixture(float(center))


def _fit_variance_preserving_power(
    support: np.ndarray,
    probabilities: np.ndarray,
    target_strokes: float,
    week_nodes_sg: np.ndarray,
    node_weights: np.ndarray,
    target_variance: float,
) -> tuple[float, np.ndarray, float, float, float]:
    """Temper the idiosyncratic PMF so latent mixing preserves round variance."""

    cache: dict[float, tuple[np.ndarray, float, float, float]] = {}

    def mixture(log_power: float) -> tuple[np.ndarray, float, float, float]:
        key = float(log_power)
        if key not in cache:
            cache[key] = _mean_calibrated_conditional_mixture(
                support,
                probabilities,
                target_strokes,
                week_nodes_sg,
                node_weights,
                power=float(np.exp(log_power)),
            )
        return cache[key]

    def difference(log_power: float) -> float:
        return mixture(log_power)[2] - target_variance

    low, high = np.log(0.03), np.log(100.0)
    low_diff, high_diff = difference(low), difference(high)
    if low_diff == 0.0:
        selected = low
    elif high_diff == 0.0:
        selected = high
    elif low_diff * high_diff < 0.0:
        selected = brentq(difference, low, high, xtol=1e-8, rtol=1e-8)
    else:
        # Some sparse supports cannot attain the requested moment after a mean
        # shift. Choose the closest support-preserving solution and expose the
        # residual in diagnostics instead of silently changing support.
        result = minimize_scalar(
            lambda value: difference(value) ** 2,
            bounds=(low, high),
            method="bounded",
            options={"xatol": 1e-7},
        )
        selected = float(result.x)
    conditional, mean, variance, clipped_mass = mixture(float(selected))
    return float(np.exp(selected)), conditional, mean, variance, clipped_mass


def _stratified_draw(
    support: np.ndarray,
    probabilities: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.int16)
    offset = rng.random()
    uniforms = (np.arange(size, dtype=float) + offset) / size
    rng.shuffle(uniforms)
    indices = np.searchsorted(np.cumsum(probabilities), uniforms, side="right")
    return support[np.minimum(indices, len(support) - 1)].astype(np.int16)


def _copula_preserving_stratified_draw(
    support: np.ndarray,
    probabilities: np.ndarray,
    raw_values: np.ndarray,
    tie_breaker: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a conditional integer PMF while retaining the raw path ordering.

    The old adapter independently shuffled each round inside every week-latent
    bin.  That reproduced each marginal but erased any upstream cross-round
    dependence supplied by a hole/path simulator. Here stratified quantiles
    are assigned monotonically to the raw paths.  A player-level shared
    tie-breaker supplies the randomized distributional transform for discrete
    ties without discarding particle identity across rounds.
    """

    values = np.asarray(raw_values, dtype=float)
    ties = np.asarray(tie_breaker, dtype=float)
    size = len(values)
    if size == 0:
        return np.empty(0, dtype=np.int16)
    if ties.shape != values.shape:
        raise ValueError("tie_breaker must match raw_values")
    offset = rng.random()
    uniforms = (np.arange(size, dtype=float) + offset) / size
    indices = np.searchsorted(np.cumsum(probabilities), uniforms, side="right")
    quantiles = support[np.minimum(indices, len(support) - 1)].astype(np.int16)
    order = np.lexsort((ties, values))
    output = np.empty(size, dtype=np.int16)
    output[order] = quantiles
    return output


def anchor_discrete_pmf_with_week_latent(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_strokes: float,
    week_latent_sd: float = DEFAULT_WEEK_LATENT_SD,
    latent_bins: int = 61,
    latent_particles: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calibrate one round PMF when cross-round particle paths are unavailable.

    The returned PMF is the marginal mixture over a standardized normal-
    quantile week latent. It is suitable for round CRPS/log density/H2H. The
    probabilities correspond to ``diagnostics["calibrated_support"]``: the
    input integer support translated by a single integer location shift. It
    cannot by itself represent the cross-round joint; callers need aligned
    particles from :func:`anchor_discrete_rounds_with_week_latent` for 72-hole
    totals and finish markets.
    """

    x = np.asarray(support, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if len(x) != len(p) or len(x) < 2 or np.any(p < 0.0) or p.sum() <= 0.0:
        raise ValueError("Invalid discrete round PMF")
    if not np.allclose(x, np.rint(x), atol=1e-8, rtol=0.0):
        raise ValueError("Round PMF support must be integer strokes")
    order = np.argsort(x)
    x, p = x[order], p[order]
    keep = p > 0.0
    x, p = x[keep], p[keep]
    p = p / p.sum()
    raw_mean, raw_variance = _pmf_moments(x, p)
    integer_support_shift = int(np.rint(float(target_strokes) - raw_mean))
    calibrated_support = x + integer_support_shift
    if latent_particles is None:
        bins = max(3, int(latent_bins))
        nodes = ndtri((np.arange(bins, dtype=float) + 0.5) / bins)
        nodes = (nodes - nodes.mean()) / nodes.std(ddof=0)
        node_weights = np.full(bins, 1.0 / bins)
    else:
        if int(latent_particles) < 3:
            raise ValueError("latent_particles must be at least three")
        nodes, node_weights, _ = _standard_normal_nodes(
            int(latent_particles), int(latent_bins), np.random.default_rng(0)
        )
        bins = len(nodes)
    (
        power,
        conditional,
        expected_mean,
        expected_variance,
        clipped_mass,
    ) = _fit_variance_preserving_power(
        calibrated_support,
        p,
        float(target_strokes),
        nodes * float(week_latent_sd),
        node_weights,
        raw_variance,
    )
    marginal = node_weights @ conditional
    marginal /= marginal.sum()
    return marginal, {
        "raw_mean_strokes": raw_mean,
        "target_mean_strokes": float(target_strokes),
        "integer_support_shift": float(integer_support_shift),
        "post_shift_residual_mean_adjustment": float(
            target_strokes - (raw_mean + integer_support_shift)
        ),
        "calibrated_support": calibrated_support.copy(),
        "mixture_expected_mean_strokes": expected_mean,
        "raw_round_variance": raw_variance,
        "mixture_expected_round_variance": expected_variance,
        "power_temperature": power,
        "conditional_mean_clipped_mass": clipped_mass,
        "week_latent_sd": float(week_latent_sd),
        "latent_bins": float(bins),
        "latent_particles": (
            None if latent_particles is None else float(latent_particles)
        ),
    }


def anchor_discrete_rounds_with_week_latent(
    round_strokes: np.ndarray,
    *,
    round_pars: np.ndarray | float,
    target_sg: np.ndarray,
    week_latent_sd: float = DEFAULT_WEEK_LATENT_SD,
    seed: int = 1245,
    latent_bins: int = 31,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Primary adapter: integer PMF anchoring plus shared week latent.

    For each player/round, the complete empirical integer support is first
    translated by the nearest integer location shift. This preserves all
    relative tails, ties, variance, and path ranks while leaving at most a
    half-stroke residual mean adjustment. The translated PMF is then
    power-tempered and exponentially tilted: the linear tilt solves the exact
    production target mean and the power is root-solved so the mixture over
    the shared week latent preserves the raw marginal variance. Every
    conditional distribution keeps the translated integer support, including
    its blow-up tail.
    """

    strokes = np.asarray(round_strokes, dtype=float)
    if strokes.ndim != 3:
        raise ValueError("round_strokes must have shape [player, round, particle]")
    if not np.isfinite(strokes).all():
        raise ValueError("round_strokes contains incomplete/non-finite particles")
    players, rounds, particles = strokes.shape
    pars = _as_player_round(round_pars, players, rounds)
    targets = _as_player_round(target_sg, players, rounds)
    if week_latent_sd < 0.0:
        raise ValueError("week_latent_sd must be non-negative")

    rng = np.random.default_rng(seed)
    output = np.empty(strokes.shape, dtype=np.int16)
    raw_mean = np.empty((players, rounds), dtype=float)
    raw_variance = np.empty((players, rounds), dtype=float)
    mixture_mean = np.empty((players, rounds), dtype=float)
    mixture_variance = np.empty((players, rounds), dtype=float)
    output_mean = np.empty((players, rounds), dtype=float)
    output_variance = np.empty((players, rounds), dtype=float)
    powers = np.empty((players, rounds), dtype=float)
    clipped_mass = np.empty((players, rounds), dtype=float)
    integer_support_shift = np.empty((players, rounds), dtype=np.int16)
    residual_mean_adjustment = np.empty((players, rounds), dtype=float)
    marginal_support = np.empty((players, rounds), dtype=object)
    marginal_probability = np.empty((players, rounds), dtype=object)
    latent_assignments = np.empty((players, particles), dtype=np.int16)
    latent_values = np.empty((players, particles), dtype=float)

    for player in range(players):
        nodes, node_weights, labels = _standard_normal_nodes(
            particles, latent_bins, rng
        )
        # Tie order is round-specific.  Sharing it across rounds would invent
        # dependence for otherwise independent integer paths merely because
        # golf scores contain many ties.  Dependence in distinct raw score
        # ranks is still carried through monotonically.
        tie_breaker = rng.random((rounds, particles))
        latent_assignments[player] = labels
        latent_values[player] = nodes[labels]
        week_nodes = nodes * float(week_latent_sd)
        for round_index in range(rounds):
            raw_support, probabilities = empirical_integer_pmf(
                strokes[player, round_index]
            )
            raw_mean[player, round_index], raw_variance[player, round_index] = _pmf_moments(
                raw_support, probabilities
            )
            target_strokes = float(pars[player, round_index] - targets[player, round_index])
            # Skill anchoring is primarily a location change. Translate the
            # complete integer support first, preserving every relative tail,
            # tie and raw path rank, then use the exponential/latent adapter
            # only for the sub-stroke residual. Without this step a 3-4 stroke
            # player/course mean gap can force the target against the finite
            # 128-particle support boundary and destroy the variance shape.
            support_shift = int(np.rint(target_strokes - raw_mean[player, round_index]))
            support = raw_support + support_shift
            integer_support_shift[player, round_index] = support_shift
            residual_mean_adjustment[player, round_index] = (
                target_strokes
                - (raw_mean[player, round_index] + support_shift)
            )
            (
                power,
                conditional,
                expected_mean,
                expected_variance,
                clipped,
            ) = _fit_variance_preserving_power(
                support,
                probabilities,
                target_strokes,
                week_nodes,
                node_weights,
                raw_variance[player, round_index],
            )
            powers[player, round_index] = power
            mixture_mean[player, round_index] = expected_mean
            mixture_variance[player, round_index] = expected_variance
            clipped_mass[player, round_index] = clipped
            marginal = np.asarray(node_weights @ conditional, dtype=float)
            marginal /= marginal.sum()
            marginal_support[player, round_index] = support.astype(np.int16).copy()
            marginal_probability[player, round_index] = marginal.copy()
            for node_index in range(len(nodes)):
                positions = np.flatnonzero(labels == node_index)
                output[player, round_index, positions] = _copula_preserving_stratified_draw(
                    support,
                    conditional[node_index],
                    strokes[player, round_index, positions],
                    tie_breaker[round_index, positions],
                    rng,
                )
            output_mean[player, round_index] = output[player, round_index].mean()
            output_variance[player, round_index] = output[player, round_index].var(ddof=0)

    target_strokes = pars - targets
    diagnostics: dict[str, np.ndarray | float] = {
        "raw_mean_strokes": raw_mean,
        "target_mean_strokes": target_strokes,
        "mixture_expected_mean_strokes": mixture_mean,
        "sample_mean_strokes": output_mean,
        "raw_round_variance": raw_variance,
        "mixture_expected_round_variance": mixture_variance,
        "sample_round_variance": output_variance,
        "power_temperature": powers,
        "conditional_mean_clipped_mass": clipped_mass,
        "integer_support_shift": integer_support_shift,
        "post_shift_residual_mean_adjustment": residual_mean_adjustment,
        "marginal_support": marginal_support,
        "marginal_probability": marginal_probability,
        "week_latent_z": latent_values,
        "week_latent_assignment": latent_assignments,
        "week_latent_sd": float(week_latent_sd),
        "latent_bins": float(latent_bins),
        "max_abs_expected_mean_error": float(np.max(np.abs(mixture_mean - target_strokes))),
        "max_abs_expected_variance_error": float(np.max(np.abs(mixture_variance - raw_variance))),
    }
    return np.ascontiguousarray(output), diagnostics


# Backward-compatible name for research callers. The discrete adapter is the
# decision/default path; the continuous version remains explicitly named above.
anchor_and_add_week_latent = anchor_discrete_rounds_with_week_latent


def variance_ratio(round_strokes: np.ndarray) -> np.ndarray:
    """Return ``Var(72h total) / sum(Var(round))`` for each player."""

    values = np.asarray(round_strokes, dtype=float)
    if values.ndim != 3:
        raise ValueError("round_strokes must have shape [player, round, particle]")
    denominator = values.var(axis=2, ddof=0).sum(axis=1)
    numerator = values.sum(axis=1).var(axis=1, ddof=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > 0.0,
    )


def audit_spec(spec: FrozenProductionSpec) -> dict[str, object]:
    variance = spec.category_round_variance
    expected_ratio = 1.0 + (spec.n_rounds - 1.0) * spec.week_latent_sd**2 / variance
    corr = spec.correlation
    return {
        "players": spec.n_players,
        "rounds": spec.n_rounds,
        "week_latent_sd": spec.week_latent_sd,
        "target_sg_field_mean_by_round": spec.target_sg.mean(axis=0).tolist(),
        "category_std_course_mean": dict(zip(CAT_ORDER, spec.std_course.mean(axis=0).tolist())),
        "category_correlation": corr.tolist(),
        "category_total_round_sd_mean": float(np.sqrt(variance).mean()),
        "category_total_round_sd_range": [float(np.sqrt(variance).min()), float(np.sqrt(variance).max())],
        "expected_72h_variance_ratio_mean": float(expected_ratio.mean()),
        "expected_72h_variance_ratio_range": [float(expected_ratio.min()), float(expected_ratio.max())],
        "metadata": dict(spec.metadata),
    }


def save_spec_audit(spec: FrozenProductionSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit_spec(spec), indent=2), encoding="utf-8")

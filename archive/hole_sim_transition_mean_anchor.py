"""Research-only production-category mean anchor for a recursive shot chain.

The shot chain changes the number and kind of later shots when an earlier
transition changes.  Consequently, matching OTT/APP/ARG/PUTT means is a joint
absorbing-chain calibration problem, not four independent shifts and not a
post-round recentering exercise.

This module owns only the numerical boundary:

* de-tilt complete historical transition atoms with a fixed-reference reward;
* reapply a conservative target-player quality tilt;
* add one fitted natural-parameter correction per category; and
* solve those four corrections jointly against an immutable production SG
  vector using a deterministic common-random expectation callback.

It deliberately does not import or modify a production simulator.  A caller
must expose category SG accumulated inside the physical chain.  The returned
parameters change transition probabilities; they must never be followed by a
score translation or post-round mean correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np


ANCHOR_CONTRACT = "joint_recursive_production_category_mean_anchor_v1"
ATOM_WEIGHT_CONTRACT = "whole_atom_donor_detilt_target_reapply_lambda_v1"
CRN_CONTRACT = "immutable_common_random_expectation_v1"
LATENT_POLICY = "player_week_and_round_latents_deferred_outside_transition_anchor"
CLIPPING_POLICY = "fail_closed_if_any_donor_or_target_log_is_clipped"
CATEGORIES = ("ott", "app", "arg", "putt")
CATEGORY_INDEX = {category: index for index, category in enumerate(CATEGORIES)}
DEFAULT_TRANSFER_BETA = (0.9, 0.2, 0.3, 0.0)


def _vector(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (len(CATEGORIES),) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {len(CATEGORIES)}-vector")
    return result


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _stable_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _quality_weight_config_sha256(
    transfer_beta: Sequence[float],
    max_abs_log_weight: float,
    reward_reference_sha256: str,
) -> str:
    """Hash every static choice that can alter quality-weight admission."""

    return _stable_json_sha256(
        {
            "atom_weight_contract": ATOM_WEIGHT_CONTRACT,
            "clipping_policy": CLIPPING_POLICY,
            "transfer_beta": _vector(transfer_beta, "transfer_beta").tolist(),
            "max_abs_log_weight": float(max_abs_log_weight),
            "reward_reference_sha256": str(reward_reference_sha256),
        }
    )


@dataclass(frozen=True)
class CommonRandomContext:
    """Immutable identity for the random lattice reused by every solve call.

    The payload itself can be a counter-RNG definition or a hash of frozen
    uniform arrays.  It remains owned by the callback; this object makes any
    accidental seed/sample/support drift visible to the solver.
    """

    seed: int
    simulations: int
    semantic_key: str
    payload_sha256: str
    contract: str = CRN_CONTRACT

    def __post_init__(self) -> None:
        if int(self.simulations) <= 0:
            raise ValueError("CRN simulations must be positive")
        if not str(self.semantic_key).strip():
            raise ValueError("CRN semantic_key must be non-empty")
        if not _is_sha256(self.payload_sha256):
            raise ValueError("CRN payload_sha256 must be a lowercase SHA-256")
        if self.contract != CRN_CONTRACT:
            raise ValueError(f"Unexpected CRN contract: {self.contract!r}")

    @property
    def digest(self) -> str:
        return _stable_json_sha256(asdict(self))


@dataclass(frozen=True)
class TransitionQualityParameters:
    """Natural parameters presented to a physical expectation callback."""

    transfer_beta: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    lambda_correction: tuple[float, float, float, float]
    reward_reference_sha256: str
    max_abs_log_weight: float = 30.0
    quality_weight_config_sha256: str = ""
    latent_policy: str = LATENT_POLICY
    atom_weight_contract: str = ATOM_WEIGHT_CONTRACT

    def __post_init__(self) -> None:
        _vector(self.transfer_beta, "transfer_beta")
        _vector(self.target_ability_sg, "target_ability_sg")
        _vector(self.lambda_correction, "lambda_correction")
        if not _is_sha256(self.reward_reference_sha256):
            raise ValueError("reward_reference_sha256 must be a lowercase SHA-256")
        if not math.isfinite(self.max_abs_log_weight) or self.max_abs_log_weight <= 0.0:
            raise ValueError("max_abs_log_weight must be finite and positive")
        expected_config_sha = _quality_weight_config_sha256(
            self.transfer_beta,
            self.max_abs_log_weight,
            self.reward_reference_sha256,
        )
        if not self.quality_weight_config_sha256:
            object.__setattr__(
                self, "quality_weight_config_sha256", expected_config_sha
            )
        if self.quality_weight_config_sha256 != expected_config_sha:
            raise ValueError("quality_weight_config_sha256 does not match its inputs")
        if self.latent_policy != LATENT_POLICY:
            raise ValueError("Week/round latent effects are not owned by this anchor")
        if self.atom_weight_contract != ATOM_WEIGHT_CONTRACT:
            raise ValueError("Unexpected atom-weight contract")

    @property
    def effective_target_log_slope(self) -> np.ndarray:
        """Final category slopes; the production mean is not added again."""

        return (
            _vector(self.transfer_beta, "transfer_beta")
            * _vector(self.target_ability_sg, "target_ability_sg")
            + _vector(self.lambda_correction, "lambda_correction")
        )

    @property
    def parameter_sha256(self) -> str:
        return _stable_json_sha256(asdict(self))


@dataclass(frozen=True)
class ChainExpectation:
    """Deterministic callback result for one common-random parameter point.

    ``support_lower_sg`` and ``support_upper_sg`` must be structural bounds,
    not the extrema merely observed in the current Monte Carlo sample.
    """

    mean_category_sg: tuple[float, float, float, float]
    mean_opportunities: tuple[float, float, float, float]
    support_lower_sg: tuple[float, float, float, float]
    support_upper_sg: tuple[float, float, float, float]
    simulations: int
    crn_sha256: str
    reward_reference_sha256: str
    support_sha256: str
    quality_weight_config_sha256: str
    quality_parameter_sha256: str
    clipped_donor_logs: int
    clipped_target_logs: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)


ExpectationCallback = Callable[
    [TransitionQualityParameters, CommonRandomContext], ChainExpectation
]


@dataclass(frozen=True)
class MeanAnchorProblem:
    """Frozen player-round calibration request.

    ``target_ability_sg`` shapes transitions through the transferred quality
    coefficients.  ``target_category_sg`` is the sole mean owner.  The fitted
    lambda absorbs any mismatch between the first quantity's implied chain
    mean and the exact production vector, which prevents double-counting.
    """

    target_category_sg: tuple[float, float, float, float]
    target_ability_sg: tuple[float, float, float, float]
    reward_reference_sha256: str
    support_sha256: str
    context_key: str
    production_total_sg: float | None = None
    transfer_beta: tuple[float, float, float, float] = DEFAULT_TRANSFER_BETA
    lambda_lower: tuple[float, float, float, float] = (-8.0, -8.0, -8.0, -8.0)
    lambda_upper: tuple[float, float, float, float] = (8.0, 8.0, 8.0, 8.0)
    category_tolerance_sg: float = 2.5e-5
    finite_difference_step: float = 0.03
    max_abs_log_weight: float = 30.0
    jacobian_rank_absolute_tolerance: float = 1e-10
    maximum_jacobian_condition_number: float = 1e7
    max_newton_iterations: int = 12
    max_broyden_iterations: int = 16
    max_fallback_evaluations: int = 160
    max_coordinate_step: float = 2.0
    deterministic_tolerance: float = 0.0
    latent_policy: str = LATENT_POLICY
    contract: str = ANCHOR_CONTRACT

    def __post_init__(self) -> None:
        target = _vector(self.target_category_sg, "target_category_sg")
        _vector(self.target_ability_sg, "target_ability_sg")
        beta = _vector(self.transfer_beta, "transfer_beta")
        lower = _vector(self.lambda_lower, "lambda_lower")
        upper = _vector(self.lambda_upper, "lambda_upper")
        if (lower >= upper).any():
            raise ValueError("Every lambda lower bound must be below its upper bound")
        if (beta < 0.0).any():
            raise ValueError("Transferred quality coefficients must be non-negative")
        if beta[CATEGORY_INDEX["putt"]] != 0.0:
            raise ValueError("PUTT transfer beta is intentionally disabled initially")
        if not _is_sha256(self.reward_reference_sha256):
            raise ValueError("reward_reference_sha256 must be a lowercase SHA-256")
        if not _is_sha256(self.support_sha256):
            raise ValueError("support_sha256 must be a lowercase SHA-256")
        if not str(self.context_key).strip():
            raise ValueError("context_key must identify an event/player/round")
        if self.production_total_sg is not None:
            supplied = float(self.production_total_sg)
            if not math.isfinite(supplied):
                raise ValueError("production_total_sg must be finite")
            if not math.isclose(supplied, float(target.sum()), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    "Production total disagrees with the four category means; "
                    "the anchor may not invent a second total-mean owner"
                )
        if self.category_tolerance_sg <= 0.0:
            raise ValueError("category_tolerance_sg must be positive")
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        if not math.isfinite(self.max_abs_log_weight) or self.max_abs_log_weight <= 0.0:
            raise ValueError("max_abs_log_weight must be finite and positive")
        if (
            not math.isfinite(self.jacobian_rank_absolute_tolerance)
            or self.jacobian_rank_absolute_tolerance <= 0.0
        ):
            raise ValueError("jacobian_rank_absolute_tolerance must be finite and positive")
        if (
            not math.isfinite(self.maximum_jacobian_condition_number)
            or self.maximum_jacobian_condition_number <= 1.0
        ):
            raise ValueError(
                "maximum_jacobian_condition_number must be finite and greater than one"
            )
        if min(
            self.max_newton_iterations,
            self.max_broyden_iterations,
            self.max_fallback_evaluations,
        ) < 0:
            raise ValueError("Solver iteration limits cannot be negative")
        if self.max_coordinate_step <= 0.0:
            raise ValueError("max_coordinate_step must be positive")
        if self.deterministic_tolerance != 0.0:
            raise ValueError(
                "deterministic_tolerance must be zero; every repeated callback "
                "must be bit-identical"
            )
        if self.latent_policy != LATENT_POLICY:
            raise ValueError(
                "Player-week and round latent effects are explicitly deferred"
            )
        if self.contract != ANCHOR_CONTRACT:
            raise ValueError(f"Unexpected anchor contract: {self.contract!r}")

    def parameters(self, lambda_correction: Sequence[float]) -> TransitionQualityParameters:
        correction = _vector(lambda_correction, "lambda_correction")
        return TransitionQualityParameters(
            transfer_beta=tuple(map(float, self.transfer_beta)),
            target_ability_sg=tuple(map(float, self.target_ability_sg)),
            lambda_correction=tuple(map(float, correction)),
            reward_reference_sha256=self.reward_reference_sha256,
            max_abs_log_weight=float(self.max_abs_log_weight),
            quality_weight_config_sha256=self.quality_weight_config_sha256,
            latent_policy=self.latent_policy,
        )

    @property
    def quality_weight_config_sha256(self) -> str:
        return _quality_weight_config_sha256(
            self.transfer_beta,
            self.max_abs_log_weight,
            self.reward_reference_sha256,
        )

    @property
    def problem_sha256(self) -> str:
        return _stable_json_sha256(asdict(self))


@dataclass(frozen=True)
class AtomMassResult:
    probability: np.ndarray
    neutral_probability_mass: np.ndarray
    neutral_multiplier: np.ndarray
    target_multiplier: np.ndarray
    donor_group_mass_before: np.ndarray
    donor_group_mass_after: np.ndarray
    clipped_donor_logs: int
    clipped_target_logs: int
    reward_reference_sha256: str
    quality_weight_config_sha256: str
    quality_parameter_sha256: str
    clipping_policy: str = CLIPPING_POLICY
    contract: str = ATOM_WEIGHT_CONTRACT


def whole_atom_probabilities(
    *,
    base_mass: Sequence[float],
    fixed_reference_reward: Sequence[float],
    donor_ability_sg: Sequence[float],
    donor_group_id: Sequence[object],
    category: str,
    parameters: TransitionQualityParameters,
    max_abs_log_weight: float | None = None,
) -> AtomMassResult:
    """De-tilt donor atoms, reapply target quality and apply fitted lambda.

    One call represents one conditional transition support, hence one SG
    category.  All fields of an atom (endpoint, lie, hazard, cost, penalty and
    terminal status) remain inseparable; only the complete atom's mass changes.

    Donor de-tilting is normalized within the supplied donor event/state group
    using *base-mass-weighted* means.  Thus each donor group's hierarchy mass
    is preserved exactly before target reapplication.
    """

    canonical = str(category).strip().casefold()
    if canonical not in CATEGORY_INDEX:
        raise ValueError(f"Unknown SG category: {category!r}")
    base = np.asarray(base_mass, dtype=float)
    reward = np.asarray(fixed_reference_reward, dtype=float)
    donor_ability = np.asarray(donor_ability_sg, dtype=float)
    groups = np.asarray(donor_group_id).astype(str)
    if base.ndim != 1 or not len(base):
        raise ValueError("Transition support must contain at least one atom")
    if reward.shape != base.shape or donor_ability.shape != base.shape or groups.shape != base.shape:
        raise ValueError("Atom mass, reward, donor ability and group arrays must align")
    if not np.isfinite(base).all() or not np.isfinite(reward).all() or not np.isfinite(donor_ability).all():
        raise ValueError("Atom mass/reward/ability arrays must be finite")
    if (base < 0.0).any() or float(base.sum()) <= 0.0:
        raise ValueError("Base atom mass must be non-negative with positive total")
    if np.any(np.char.str_len(groups) == 0):
        raise ValueError("Every atom needs a donor event/state normalization group")
    configured_limit = float(parameters.max_abs_log_weight)
    if max_abs_log_weight is not None and not math.isclose(
        float(max_abs_log_weight), configured_limit, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "Atom log-weight limit disagrees with quality provenance config"
        )
    max_abs_log_weight = configured_limit
    if not math.isfinite(max_abs_log_weight) or max_abs_log_weight <= 0.0:
        raise ValueError("max_abs_log_weight must be finite and positive")

    base = base / base.sum()
    index = CATEGORY_INDEX[canonical]
    beta = float(parameters.transfer_beta[index])
    donor_log_raw = -beta * donor_ability * reward
    donor_log = np.clip(donor_log_raw, -max_abs_log_weight, max_abs_log_weight)
    clipped_donor_logs = int(np.count_nonzero(donor_log != donor_log_raw))
    if clipped_donor_logs:
        raise RuntimeError(
            "Donor quality log-weight clipping is not admissible under "
            f"{CLIPPING_POLICY}: clipped_atoms={clipped_donor_logs}"
        )
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    group_mass_before = np.bincount(inverse, weights=base, minlength=group_count)

    # Group-specific max subtraction is algebraically neutral and prevents
    # overflow before preserving the exact declared hierarchy mass.
    group_max = np.full(group_count, -np.inf, dtype=float)
    np.maximum.at(group_max, inverse, donor_log)
    detilt = base * np.exp(donor_log - group_max[inverse])
    detilt_total = np.bincount(inverse, weights=detilt, minlength=group_count)
    if (detilt_total <= 0.0).any() or not np.isfinite(detilt_total).all():
        raise ValueError("Donor de-tilt collapsed a normalization group")
    neutral_mass = detilt * (group_mass_before / detilt_total)[inverse]
    group_mass_after = np.bincount(
        inverse, weights=neutral_mass, minlength=group_count
    )
    if not np.allclose(group_mass_after, group_mass_before, atol=2e-15, rtol=2e-15):
        raise AssertionError("Donor de-tilt did not preserve hierarchy group mass")

    effective_slope = float(parameters.effective_target_log_slope[index])
    target_log_raw = effective_slope * reward
    target_log = np.clip(target_log_raw, -max_abs_log_weight, max_abs_log_weight)
    clipped_target_logs = int(np.count_nonzero(target_log != target_log_raw))
    if clipped_target_logs:
        raise RuntimeError(
            "Target quality log-weight clipping is not admissible under "
            f"{CLIPPING_POLICY}: clipped_atoms={clipped_target_logs}"
        )
    target_shift = target_log - float(target_log.max())
    target_multiplier = np.exp(target_shift)
    final_mass = neutral_mass * target_multiplier
    total = float(final_mass.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Target reapplication collapsed the transition support")
    probability = np.ascontiguousarray(final_mass / total)
    neutral_multiplier = np.divide(
        neutral_mass,
        base,
        out=np.zeros_like(neutral_mass),
        where=base > 0.0,
    )
    return AtomMassResult(
        probability=probability,
        neutral_probability_mass=np.ascontiguousarray(neutral_mass),
        neutral_multiplier=np.ascontiguousarray(neutral_multiplier),
        target_multiplier=np.ascontiguousarray(target_multiplier),
        donor_group_mass_before=np.ascontiguousarray(group_mass_before),
        donor_group_mass_after=np.ascontiguousarray(group_mass_after),
        clipped_donor_logs=clipped_donor_logs,
        clipped_target_logs=clipped_target_logs,
        reward_reference_sha256=parameters.reward_reference_sha256,
        quality_weight_config_sha256=parameters.quality_weight_config_sha256,
        quality_parameter_sha256=parameters.parameter_sha256,
    )


@dataclass(frozen=True)
class MeanAnchorResult:
    converged: bool
    production_admissible: bool
    status: str
    lambda_correction: tuple[float, float, float, float]
    effective_target_log_slope: tuple[float, float, float, float]
    target_category_sg: tuple[float, float, float, float]
    achieved_category_sg: tuple[float, float, float, float]
    residual_category_sg: tuple[float, float, float, float]
    mean_opportunities: tuple[float, float, float, float]
    jacobian: tuple[tuple[float, ...], ...]
    methods_attempted: tuple[str, ...]
    callback_evaluations: int
    unique_parameter_points: int
    support_status: str
    diagnostics: Mapping[str, object]
    crn_sha256: str
    reward_reference_sha256: str
    support_sha256: str
    quality_weight_config_sha256: str
    quality_parameter_sha256: str
    anchor_problem_sha256: str
    clipping_policy: str
    context_key: str
    latent_policy: str = LATENT_POLICY
    post_round_recenter_applied: bool = False
    contract: str = ANCHOR_CONTRACT

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class _ExpectationEvaluator:
    def __init__(
        self,
        problem: MeanAnchorProblem,
        callback: ExpectationCallback,
        crn: CommonRandomContext,
    ) -> None:
        self.problem = problem
        self.callback = callback
        self.crn = crn
        self.cache: dict[tuple[float, ...], ChainExpectation] = {}
        self.callback_evaluations = 0
        self.best_key: tuple[float, ...] | None = None
        self.best_norm = math.inf
        self.best_feasible_key: tuple[float, ...] | None = None
        self.best_feasible_norm = math.inf
        self.reference_bounds: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _key(values: np.ndarray) -> tuple[float, ...]:
        return tuple(map(float, np.asarray(values, dtype=np.float64)))

    def _validate(
        self,
        result: ChainExpectation,
        parameters: TransitionQualityParameters,
    ) -> None:
        mean = _vector(result.mean_category_sg, "callback mean_category_sg")
        opportunities = _vector(result.mean_opportunities, "callback mean_opportunities")
        lower = _vector(result.support_lower_sg, "callback support_lower_sg")
        upper = _vector(result.support_upper_sg, "callback support_upper_sg")
        if (opportunities < 0.0).any():
            raise ValueError("Callback mean opportunities cannot be negative")
        if (lower > upper).any() or (mean < lower - 1e-12).any() or (mean > upper + 1e-12).any():
            raise ValueError("Callback mean lies outside its structural support bounds")
        if int(result.simulations) != int(self.crn.simulations):
            raise ValueError("Callback changed the frozen simulation count")
        if result.crn_sha256 != self.crn.digest:
            raise ValueError("Callback did not use the frozen common-random context")
        if result.reward_reference_sha256 != self.problem.reward_reference_sha256:
            raise ValueError("Callback reward reference changed during calibration")
        if result.support_sha256 != self.problem.support_sha256:
            raise ValueError("Callback transition support changed during calibration")
        if result.quality_weight_config_sha256 != self.problem.quality_weight_config_sha256:
            raise ValueError("Callback quality-weight config changed during calibration")
        if result.quality_parameter_sha256 != parameters.parameter_sha256:
            raise ValueError("Callback quality-parameter provenance does not match lambda")
        for label, count in (
            ("donor", result.clipped_donor_logs),
            ("target", result.clipped_target_logs),
        ):
            if isinstance(count, (bool, np.bool_)) or int(count) != count or int(count) < 0:
                raise ValueError(f"Callback {label} clipped-log count is invalid")
            if int(count):
                raise RuntimeError(
                    f"Callback reported {label} log-weight clipping; calibrated "
                    f"output is not admissible under {CLIPPING_POLICY}"
                )
        if self.reference_bounds is None:
            self.reference_bounds = (lower.copy(), upper.copy())
        else:
            previous_lower, previous_upper = self.reference_bounds
            if not np.array_equal(lower, previous_lower) or not np.array_equal(upper, previous_upper):
                raise ValueError("Structural support bounds changed with lambda")

    def _uncached(self, values: np.ndarray) -> ChainExpectation:
        parameters = self.problem.parameters(values)
        result = self.callback(parameters, self.crn)
        if not isinstance(result, ChainExpectation):
            raise TypeError("Expectation callback must return ChainExpectation")
        self.callback_evaluations += 1
        self._validate(result, parameters)
        return result

    def _repeated_deterministic(self, values: np.ndarray) -> ChainExpectation:
        first = self._uncached(values)
        second = self._uncached(values)
        fields = (
            "mean_category_sg",
            "mean_opportunities",
            "support_lower_sg",
            "support_upper_sg",
        )
        for name in fields:
            left = np.asarray(getattr(first, name), dtype=float)
            right = np.asarray(getattr(second, name), dtype=float)
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Expectation callback is not deterministic for {name}; "
                    "the same CRN and lambda produced different results"
                )
        scalar_fields = (
            "simulations",
            "crn_sha256",
            "reward_reference_sha256",
            "support_sha256",
            "quality_weight_config_sha256",
            "quality_parameter_sha256",
            "clipped_donor_logs",
            "clipped_target_logs",
        )
        for name in scalar_fields:
            if getattr(first, name) != getattr(second, name):
                raise RuntimeError(
                    f"Expectation callback is not deterministic for {name}; "
                    "the same CRN and lambda produced different results"
                )
        return first

    def assert_deterministic(self, values: np.ndarray) -> ChainExpectation:
        """Evaluate once through the every-point repeated-callback gate."""

        return self.evaluate(values)

    def _record_best(self, values: np.ndarray, result: ChainExpectation) -> None:
        residual = _vector(result.mean_category_sg, "mean") - _vector(
            self.problem.target_category_sg, "target"
        )
        norm = float(np.linalg.norm(residual))
        if norm < self.best_norm:
            self.best_norm = norm
            self.best_key = self._key(values)
        if (
            float(np.max(np.abs(residual)))
            <= float(self.problem.category_tolerance_sg)
            and norm < self.best_feasible_norm
        ):
            self.best_feasible_norm = norm
            self.best_feasible_key = self._key(values)

    def evaluate(self, values: np.ndarray) -> ChainExpectation:
        key = self._key(values)
        if key not in self.cache:
            # Every unique point, including every nonzero finite-difference,
            # line-search and fallback point, is called twice and compared.
            self.cache[key] = self._repeated_deterministic(values)
            self._record_best(values, self.cache[key])
        return self.cache[key]

    def residual(self, values: np.ndarray) -> np.ndarray:
        result = self.evaluate(values)
        return _vector(result.mean_category_sg, "mean") - _vector(
            self.problem.target_category_sg, "target"
        )


def _finite_difference_jacobian(
    evaluator: _ExpectationEvaluator,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    step: float,
) -> np.ndarray:
    size = len(CATEGORIES)
    jacobian = np.empty((size, size), dtype=float)
    for column in range(size):
        left = values.copy()
        right = values.copy()
        left[column] = max(lower[column], values[column] - step)
        right[column] = min(upper[column], values[column] + step)
        width = float(right[column] - left[column])
        if width <= 0.0:
            jacobian[:, column] = 0.0
            continue
        jacobian[:, column] = (
            evaluator.residual(right) - evaluator.residual(left)
        ) / width
    return jacobian


def _bounded_step(
    values: np.ndarray,
    step: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum: float,
) -> np.ndarray:
    limited = np.clip(np.asarray(step, dtype=float), -maximum, maximum)
    return np.clip(values + limited, lower, upper)


def _line_search(
    evaluator: _ExpectationEvaluator,
    values: np.ndarray,
    residual: np.ndarray,
    proposal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    baseline = float(np.linalg.norm(residual))
    direction = proposal - values
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        candidate = np.clip(values + scale * direction, lower, upper)
        if np.array_equal(candidate, values):
            continue
        candidate_residual = evaluator.residual(candidate)
        if float(np.linalg.norm(candidate_residual)) < baseline:
            return candidate, candidate_residual, True
    return values, residual, False


def _support_status(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[str, list[str]]:
    below = target < lower - 1e-12
    above = target > upper + 1e-12
    categories = [
        CATEGORIES[index]
        for index in np.flatnonzero(below | above)
    ]
    return (
        "infeasible_structural_category_bounds" if categories else "within_structural_category_bounds",
        categories,
    )


def _make_result(
    *,
    problem: MeanAnchorProblem,
    evaluator: _ExpectationEvaluator,
    crn: CommonRandomContext,
    values: np.ndarray,
    expectation: ChainExpectation,
    jacobian: np.ndarray,
    methods: list[str],
    converged: bool,
    status: str,
    support_status: str,
    diagnostics: Mapping[str, object],
) -> MeanAnchorResult:
    parameters = problem.parameters(values)
    achieved = _vector(expectation.mean_category_sg, "achieved")
    target = _vector(problem.target_category_sg, "target")
    residual = achieved - target
    return MeanAnchorResult(
        converged=bool(converged),
        production_admissible=bool(converged),
        status=str(status),
        lambda_correction=tuple(map(float, values)),
        effective_target_log_slope=tuple(
            map(float, parameters.effective_target_log_slope)
        ),
        target_category_sg=tuple(map(float, target)),
        achieved_category_sg=tuple(map(float, achieved)),
        residual_category_sg=tuple(map(float, residual)),
        mean_opportunities=tuple(map(float, expectation.mean_opportunities)),
        jacobian=tuple(tuple(map(float, row)) for row in np.asarray(jacobian)),
        methods_attempted=tuple(methods),
        callback_evaluations=int(evaluator.callback_evaluations),
        unique_parameter_points=int(len(evaluator.cache)),
        support_status=str(support_status),
        diagnostics=dict(diagnostics),
        crn_sha256=crn.digest,
        reward_reference_sha256=problem.reward_reference_sha256,
        support_sha256=problem.support_sha256,
        quality_weight_config_sha256=problem.quality_weight_config_sha256,
        quality_parameter_sha256=parameters.parameter_sha256,
        anchor_problem_sha256=problem.problem_sha256,
        clipping_policy=CLIPPING_POLICY,
        context_key=problem.context_key,
    )


def solve_production_category_mean_anchor(
    problem: MeanAnchorProblem,
    callback: ExpectationCallback,
    crn: CommonRandomContext,
    *,
    initial_lambda: Sequence[float] | None = None,
) -> MeanAnchorResult:
    """Jointly solve transition tilts for exact production category means.

    The algorithm tries damped bounded Newton, projected Broyden and bounded
    nonlinear least squares, always on the same callback/support/CRN identity.
    Failure is returned with structural, opportunity and local-rank evidence;
    it is never hidden by translating completed round scores.
    """

    if not isinstance(problem, MeanAnchorProblem):
        raise TypeError("problem must be MeanAnchorProblem")
    if not isinstance(crn, CommonRandomContext):
        raise TypeError("crn must be CommonRandomContext")
    lower = _vector(problem.lambda_lower, "lambda_lower")
    upper = _vector(problem.lambda_upper, "lambda_upper")
    values = np.zeros(len(CATEGORIES), dtype=float) if initial_lambda is None else _vector(
        initial_lambda, "initial_lambda"
    )
    values = np.clip(values, lower, upper)
    target = _vector(problem.target_category_sg, "target_category_sg")
    evaluator = _ExpectationEvaluator(problem, callback, crn)
    initial = evaluator.assert_deterministic(values)
    structural_lower = _vector(initial.support_lower_sg, "support_lower_sg")
    structural_upper = _vector(initial.support_upper_sg, "support_upper_sg")
    support_status, infeasible_categories = _support_status(
        target, structural_lower, structural_upper
    )
    empty_jacobian = np.full((len(CATEGORIES), len(CATEGORIES)), np.nan)
    if infeasible_categories:
        return _make_result(
            problem=problem,
            evaluator=evaluator,
            crn=crn,
            values=values,
            expectation=initial,
            jacobian=empty_jacobian,
            methods=["structural_support_preflight"],
            converged=False,
            status="infeasible_support",
            support_status=support_status,
            diagnostics={
                "infeasible_categories": infeasible_categories,
                "support_lower_sg": structural_lower.tolist(),
                "support_upper_sg": structural_upper.tolist(),
                "target_category_sg": target.tolist(),
                "mean_owner": "exact_production_category_vector",
                "post_round_recenter": "forbidden",
            },
        )

    tolerance = float(problem.category_tolerance_sg)
    residual = evaluator.residual(values)
    methods: list[str] = []
    jacobian = empty_jacobian
    converged = float(np.max(np.abs(residual))) <= tolerance
    status = "converged_at_initial_parameters" if converged else "unresolved"

    if not converged and problem.max_newton_iterations:
        methods.append("damped_bounded_newton")
        for _ in range(int(problem.max_newton_iterations)):
            jacobian = _finite_difference_jacobian(
                evaluator,
                values,
                lower,
                upper,
                float(problem.finite_difference_step),
            )
            delta = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)[0]
            proposal = _bounded_step(
                values,
                delta,
                lower,
                upper,
                float(problem.max_coordinate_step),
            )
            candidate, candidate_residual, improved = _line_search(
                evaluator, values, residual, proposal, lower, upper
            )
            if not improved:
                break
            values, residual = candidate, candidate_residual
            if float(np.max(np.abs(residual))) <= tolerance:
                converged = True
                status = "converged_bounded_newton"
                break

    if not converged and problem.max_broyden_iterations:
        methods.append("projected_broyden")
        if not np.isfinite(jacobian).all():
            jacobian = _finite_difference_jacobian(
                evaluator,
                values,
                lower,
                upper,
                float(problem.finite_difference_step),
            )
        approximation = jacobian.copy()
        for _ in range(int(problem.max_broyden_iterations)):
            delta = np.linalg.lstsq(approximation, -residual, rcond=1e-10)[0]
            proposal = _bounded_step(
                values,
                delta,
                lower,
                upper,
                float(problem.max_coordinate_step),
            )
            candidate, candidate_residual, improved = _line_search(
                evaluator, values, residual, proposal, lower, upper
            )
            if not improved:
                break
            displacement = candidate - values
            response = candidate_residual - residual
            denominator = float(displacement @ displacement)
            if denominator > 1e-14:
                approximation = approximation + np.outer(
                    response - approximation @ displacement,
                    displacement,
                ) / denominator
            values, residual = candidate, candidate_residual
            jacobian = approximation
            if float(np.max(np.abs(residual))) <= tolerance:
                converged = True
                status = "converged_projected_broyden"
                break

    if not converged and problem.max_fallback_evaluations:
        methods.append("bounded_nonlinear_least_squares_fallback")
        try:
            from scipy.optimize import least_squares

            fallback = least_squares(
                evaluator.residual,
                values,
                jac=lambda point: _finite_difference_jacobian(
                    evaluator,
                    np.asarray(point, dtype=float),
                    lower,
                    upper,
                    float(problem.finite_difference_step),
                ),
                bounds=(lower, upper),
                max_nfev=int(problem.max_fallback_evaluations),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                x_scale="jac",
            )
            fallback_values = np.clip(np.asarray(fallback.x, dtype=float), lower, upper)
            fallback_residual = evaluator.residual(fallback_values)
            if np.linalg.norm(fallback_residual) <= np.linalg.norm(residual):
                values, residual = fallback_values, fallback_residual
            converged = float(np.max(np.abs(residual))) <= tolerance
            if converged:
                status = "converged_bounded_least_squares"
        except (ImportError, ValueError, np.linalg.LinAlgError) as error:
            status = f"fallback_unavailable_or_failed:{type(error).__name__}"

    # Admission is lexicographic: any componentwise-feasible cached point beats
    # every infeasible point, even when the latter has lower L2.  L2 breaks
    # ties only within the feasible or infeasible class.
    selected_key = (
        evaluator.best_feasible_key
        if evaluator.best_feasible_key is not None
        else evaluator.best_key
    )
    if selected_key is not None:
        selected_values = np.asarray(selected_key, dtype=float)
        selected_residual = evaluator.residual(selected_values)
        current_feasible = float(np.max(np.abs(residual))) <= tolerance
        selected_feasible = float(np.max(np.abs(selected_residual))) <= tolerance
        should_restore = (
            (selected_feasible and not current_feasible)
            or (
                selected_feasible == current_feasible
                and np.linalg.norm(selected_residual) < np.linalg.norm(residual)
            )
        )
        if should_restore:
            values, residual = selected_values, selected_residual
    mean_matched = float(np.max(np.abs(residual))) <= tolerance
    converged = mean_matched
    if converged and not status.startswith("converged"):
        status = "converged_componentwise_feasible_cached_point"
    elif not converged and status.startswith("converged"):
        status = "final_point_failed_component_tolerance"
    expectation = evaluator.evaluate(values)
    jacobian = _finite_difference_jacobian(
        evaluator,
        values,
        lower,
        upper,
        float(problem.finite_difference_step),
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank_tolerance = float(problem.jacobian_rank_absolute_tolerance)
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    largest_singular = float(singular_values.max(initial=0.0))
    smallest_singular = float(singular_values.min(initial=math.inf))
    condition_number = (
        largest_singular / smallest_singular
        if smallest_singular > 0.0 and math.isfinite(smallest_singular)
        else math.inf
    )
    full_rank = rank == len(CATEGORIES)
    well_conditioned = (
        full_rank
        and math.isfinite(condition_number)
        and condition_number <= float(problem.maximum_jacobian_condition_number)
    )
    # Exact mean residual is necessary but not sufficient: a locally
    # unidentified or numerically unstable four-vector cannot be admitted.
    converged = bool(mean_matched and full_rank and well_conditioned)
    opportunities = _vector(expectation.mean_opportunities, "mean_opportunities")
    sensitivity = np.linalg.norm(jacobian, axis=0)
    active_lower = np.isclose(values, lower, atol=1e-8)
    active_upper = np.isclose(values, upper, atol=1e-8)
    unconstrained_local_step = np.linalg.lstsq(
        jacobian, -residual, rcond=1e-10
    )[0]
    outward_lower = active_lower & (unconstrained_local_step < -1e-10)
    outward_upper = active_upper & (unconstrained_local_step > 1e-10)
    blocked_bound_parameters = outward_lower | outward_upper
    unresolved_categories = [
        CATEGORIES[index]
        for index in np.flatnonzero(np.abs(residual) > tolerance)
    ]
    unresolved_mask = np.abs(residual) > tolerance
    coupling_floor = max(
        float(np.max(np.abs(jacobian), initial=0.0)) * 1e-10,
        1e-12,
    )
    bound_coupling = (
        np.abs(jacobian) > coupling_floor
    ) & unresolved_mask[:, None] & blocked_bound_parameters[None, :]
    bound_limited_residual_mask = bound_coupling.any(axis=1)
    bound_parameter_to_residual = {
        CATEGORIES[column]: [
            CATEGORIES[row]
            for row in np.flatnonzero(bound_coupling[:, column])
        ]
        for column in np.flatnonzero(blocked_bound_parameters)
    }
    zero_response_categories = [
        CATEGORIES[index]
        for index in range(len(CATEGORIES))
        if opportunities[index] <= 1e-12 or sensitivity[index] <= 1e-10
    ]

    if not full_rank:
        status = "unresolved_rank_deficient_endogenous_support"
        support_status = "local_category_response_rank_deficient"
    elif not well_conditioned:
        status = "unresolved_ill_conditioned_endogenous_support"
        support_status = "local_category_response_conditioning_failed"
    elif converged:
        support_status = "joint_target_matched_within_structural_support"
    elif zero_response_categories:
        status = "infeasible_or_unidentified_zero_opportunity_support"
        support_status = "local_support_has_zero_quality_response"
    elif bound_limited_residual_mask.any():
        # This is evidence about the final local linearization only.  A
        # non-monotone response can have another in-bounds root even when the
        # local Newton direction points outward at an active bound, so do not
        # promote this diagnostic to a global infeasibility certificate.
        status = "locally_bound_limited_unresolved"
        support_status = "local_solver_point_bound_limited"
    elif status == "unresolved":
        status = "solver_failed_to_match_joint_target"

    diagnostics = {
        "target_total_sg": float(target.sum()),
        "achieved_total_sg": float(
            _vector(expectation.mean_category_sg, "mean_category_sg").sum()
        ),
        "max_abs_category_residual_sg": float(np.max(np.abs(residual))),
        "l2_residual_sg": float(np.linalg.norm(residual)),
        "unresolved_categories": unresolved_categories,
        "zero_response_categories": zero_response_categories,
        "jacobian_rank": rank,
        "jacobian_full_rank": full_rank,
        "jacobian_rank_absolute_tolerance": rank_tolerance,
        "jacobian_condition_number": condition_number,
        "maximum_jacobian_condition_number": float(
            problem.maximum_jacobian_condition_number
        ),
        "jacobian_conditioning_passed": well_conditioned,
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_column_norm": sensitivity.tolist(),
        "active_lower_bounds": [
            CATEGORIES[index] for index in np.flatnonzero(active_lower)
        ],
        "active_upper_bounds": [
            CATEGORIES[index] for index in np.flatnonzero(active_upper)
        ],
        "unconstrained_local_lambda_step": unconstrained_local_step.tolist(),
        "blocked_bound_parameters": [
            CATEGORIES[index]
            for index in np.flatnonzero(blocked_bound_parameters)
        ],
        "bound_limited_residual_categories": [
            CATEGORIES[index]
            for index in np.flatnonzero(bound_limited_residual_mask)
        ],
        "bound_parameter_to_residual_categories": bound_parameter_to_residual,
        "structural_support_lower_sg": structural_lower.tolist(),
        "structural_support_upper_sg": structural_upper.tolist(),
        "mean_owner": "exact_production_category_vector",
        "quality_shape_owner": "beta_times_target_ability_plus_fitted_lambda",
        "donor_quality_owner": "whole_atom_beta_detilt_with_group_mass_preservation",
        "post_round_recenter": "forbidden",
        "latent_effects": LATENT_POLICY,
        "putt_transfer_beta": 0.0,
        "putt_anchor_lambda_may_be_nonzero": True,
        "clipping_policy": CLIPPING_POLICY,
        "clipped_donor_logs": int(expectation.clipped_donor_logs),
        "clipped_target_logs": int(expectation.clipped_target_logs),
        "quality_weight_config_sha256": problem.quality_weight_config_sha256,
        "quality_parameter_sha256": problem.parameters(values).parameter_sha256,
        "admission_gates": {
            "componentwise_mean_match": bool(mean_matched),
            "full_jacobian_rank": bool(full_rank),
            "jacobian_conditioning": bool(well_conditioned),
            "zero_log_weight_clipping": True,
            "all_unique_points_repeated_deterministically": True,
        },
        "callback_diagnostics": dict(expectation.diagnostics),
    }
    return _make_result(
        problem=problem,
        evaluator=evaluator,
        crn=crn,
        values=values,
        expectation=expectation,
        jacobian=jacobian,
        methods=methods,
        converged=converged,
        status=status,
        support_status=support_status,
        diagnostics=diagnostics,
    )


__all__ = [
    "ANCHOR_CONTRACT",
    "ATOM_WEIGHT_CONTRACT",
    "CATEGORIES",
    "CLIPPING_POLICY",
    "CRN_CONTRACT",
    "DEFAULT_TRANSFER_BETA",
    "LATENT_POLICY",
    "AtomMassResult",
    "ChainExpectation",
    "CommonRandomContext",
    "ExpectationCallback",
    "MeanAnchorProblem",
    "MeanAnchorResult",
    "TransitionQualityParameters",
    "solve_production_category_mean_anchor",
    "whole_atom_probabilities",
]

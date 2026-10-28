from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_transition_mean_anchor as anchor


REWARD_SHA = sha256(b"synthetic-fixed-reference-green-value-v1").hexdigest()


class SyntheticAbsorbingRound:
    """Smooth exact expectation over a frozen empirical course lattice.

    The common random numbers choose exogenous hole variants.  Conditional on
    that frozen lattice, complete transition supports are integrated exactly.
    APP quality changes ARG opportunities, and APP/ARG both change PUTT
    opportunities, giving the calibration a genuinely coupled Jacobian.
    """

    def __init__(self, seed: int = 1701, simulations: int = 12_000) -> None:
        rng = np.random.default_rng(seed)
        variants = rng.integers(0, 3, size=(simulations, 18), dtype=np.int8)
        self.variant_counts = np.bincount(variants.ravel(), minlength=3).astype(float)
        self.variant_probability = self.variant_counts / self.variant_counts.sum()
        payload_sha = sha256(variants.tobytes()).hexdigest()
        self.crn = anchor.CommonRandomContext(
            seed=seed,
            simulations=simulations,
            semantic_key="synthetic|player=7|round=1|18_hole_variant_lattice",
            payload_sha256=payload_sha,
        )
        self.spec = {
            "tee": {
                "reward": [-0.55, 0.10, 0.68],
                "base": [0.29, 0.46, 0.25],
                "ability": [-0.35, 0.05, 0.62],
                "group": ["e1", "e1", "e2"],
                "next": ["app_hard", "app_easy", "app_easy"],
            },
            "app_easy": {
                "reward": [-0.72, 0.08, 0.76],
                "base": [0.27, 0.44, 0.29],
                "ability": [-0.42, 0.00, 0.55],
                "group": ["e1", "e1", "e2"],
                "next": ["arg", "putt", "putt"],
            },
            "app_hard": {
                "reward": [-0.84, -0.06, 0.57],
                "base": [0.36, 0.40, 0.24],
                "ability": [-0.47, -0.03, 0.49],
                "group": ["e1", "e1", "e2"],
                "next": ["arg", "arg", "putt"],
            },
            "arg": {
                "reward": [-0.58, 0.04, 0.63],
                "base": [0.31, 0.43, 0.26],
                "ability": [-0.40, 0.02, 0.51],
                "group": ["e1", "e1", "e2"],
                "next": ["putt", "putt", "terminal"],
            },
            "putt": {
                "reward": [-0.37, 0.03, 0.46],
                "base": [0.24, 0.47, 0.29],
                "ability": [-0.33, 0.01, 0.44],
                "group": ["e1", "e1", "e2"],
                "next": ["putt", "terminal", "terminal"],
            },
        }
        self.support_sha = sha256(
            json.dumps(self.spec, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _category(state: str) -> str:
        if state == "tee":
            return "ott"
        if state.startswith("app"):
            return "app"
        return state

    def _state_expectation(
        self,
        state: str,
        parameters: anchor.TransitionQualityParameters,
        variant: int,
        putt_depth: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if state == "terminal" or (state == "putt" and putt_depth >= 4):
            return np.zeros(4), np.zeros(4)
        category = self._category(state)
        spec = self.spec[state]
        base = np.asarray(spec["base"], float).copy()
        # Exogenous course variants alter support mass but are frozen across
        # all parameter evaluations by the CRN lattice.
        shift = (variant - 1) * 0.025
        base[0] += shift
        base[2] -= shift
        mass = anchor.whole_atom_probabilities(
            base_mass=base,
            fixed_reference_reward=spec["reward"],
            donor_ability_sg=spec["ability"],
            donor_group_id=spec["group"],
            category=category,
            parameters=parameters,
        ).probability
        gain = np.zeros(4)
        opportunities = np.zeros(4)
        category_index = anchor.CATEGORIES.index(category)
        opportunities[category_index] = 1.0
        rewards = np.asarray(spec["reward"], float)
        gain[category_index] = float(mass @ rewards)
        for probability, next_state in zip(mass, spec["next"]):
            next_depth = putt_depth + 1 if state == "putt" else putt_depth
            child_gain, child_opportunities = self._state_expectation(
                str(next_state), parameters, variant, next_depth
            )
            gain += float(probability) * child_gain
            opportunities += float(probability) * child_opportunities
        return gain, opportunities

    def __call__(
        self,
        parameters: anchor.TransitionQualityParameters,
        crn: anchor.CommonRandomContext,
    ) -> anchor.ChainExpectation:
        if crn.digest != self.crn.digest:
            raise AssertionError("synthetic callback received a different CRN")
        round_gain = np.zeros(4)
        round_opportunities = np.zeros(4)
        for variant, probability in enumerate(self.variant_probability):
            gain, opportunities = self._state_expectation(
                "tee", parameters, variant
            )
            round_gain += 18.0 * probability * gain
            round_opportunities += 18.0 * probability * opportunities
        return anchor.ChainExpectation(
            mean_category_sg=tuple(map(float, round_gain)),
            mean_opportunities=tuple(map(float, round_opportunities)),
            support_lower_sg=(-9.9, -15.2, -10.5, -26.7),
            support_upper_sg=(12.3, 13.7, 11.4, 33.2),
            simulations=crn.simulations,
            crn_sha256=crn.digest,
            reward_reference_sha256=REWARD_SHA,
            support_sha256=self.support_sha,
            quality_weight_config_sha256=parameters.quality_weight_config_sha256,
            quality_parameter_sha256=parameters.parameter_sha256,
            clipped_donor_logs=0,
            clipped_target_logs=0,
            diagnostics={"analytic_conditional_on_crn_lattice": True},
        )


def _problem(
    model: SyntheticAbsorbingRound,
    target: np.ndarray,
    ability: tuple[float, float, float, float],
    **overrides,
) -> anchor.MeanAnchorProblem:
    fields = dict(
        target_category_sg=tuple(map(float, target)),
        target_ability_sg=ability,
        reward_reference_sha256=REWARD_SHA,
        support_sha256=model.support_sha,
        context_key="synthetic|player=7|round=1",
        production_total_sg=float(target.sum()),
        category_tolerance_sg=2e-9,
        finite_difference_step=0.02,
    )
    fields.update(overrides)
    return anchor.MeanAnchorProblem(**fields)


def test_whole_atom_detilt_preserves_group_mass_and_tilts_complete_atoms() -> None:
    parameters = anchor.TransitionQualityParameters(
        transfer_beta=anchor.DEFAULT_TRANSFER_BETA,
        target_ability_sg=(0.5, 0.2, -0.1, 0.3),
        lambda_correction=(0.25, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
    )
    result = anchor.whole_atom_probabilities(
        base_mass=[0.10, 0.25, 0.20, 0.45],
        fixed_reference_reward=[-0.8, 0.4, -0.2, 0.7],
        donor_ability_sg=[-0.5, 0.3, -0.1, 0.8],
        donor_group_id=["event_a", "event_a", "event_b", "event_b"],
        category="ott",
        parameters=parameters,
    )
    assert np.isclose(result.probability.sum(), 1.0)
    assert np.array_equal(
        result.donor_group_mass_before, result.donor_group_mass_after
    )
    assert result.clipped_donor_logs == 0
    assert result.clipped_target_logs == 0
    assert result.reward_reference_sha256 == REWARD_SHA
    assert result.quality_weight_config_sha256 == parameters.quality_weight_config_sha256
    assert result.quality_parameter_sha256 == parameters.parameter_sha256
    assert result.clipping_policy == anchor.CLIPPING_POLICY
    assert result.contract == anchor.ATOM_WEIGHT_CONTRACT


def test_joint_absorbing_chain_anchor_matches_all_four_production_means() -> None:
    model = SyntheticAbsorbingRound()
    ability = (0.48, 0.31, -0.12, 0.27)
    true_lambda = np.asarray([0.23, -0.34, 0.29, 0.17])
    template = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=ability,
        reward_reference_sha256=REWARD_SHA,
        support_sha256=model.support_sha,
        context_key="synthetic-target-builder",
    )
    target = np.asarray(
        model(template.parameters(true_lambda), model.crn).mean_category_sg
    )
    problem = _problem(model, target, ability)

    # A diagonal one-pass correction ignores APP -> ARG/PUTT and ARG -> PUTT
    # opportunity coupling and therefore cannot match the vector.
    zero = np.zeros(4)
    base = np.asarray(model(problem.parameters(zero), model.crn).mean_category_sg)
    step = 0.02
    jacobian = np.column_stack(
        [
            (
                np.asarray(
                    model(
                        problem.parameters(np.eye(4)[column] * step), model.crn
                    ).mean_category_sg
                )
                - base
            )
            / step
            for column in range(4)
        ]
    )
    diagonal_lambda = np.divide(
        target - base,
        np.diag(jacobian),
        out=np.zeros(4),
        where=np.abs(np.diag(jacobian)) > 1e-10,
    )
    diagonal_residual = (
        np.asarray(
            model(problem.parameters(diagonal_lambda), model.crn).mean_category_sg
        )
        - target
    )
    off_diagonal = jacobian - np.diag(np.diag(jacobian))
    assert np.linalg.norm(off_diagonal) > 0.02
    assert np.max(np.abs(diagonal_residual)) > 1e-4

    result = anchor.solve_production_category_mean_anchor(
        problem, model, model.crn
    )
    assert result.converged, result.as_dict()
    assert max(abs(value) for value in result.residual_category_sg) <= 2e-9
    assert np.allclose(result.lambda_correction, true_lambda, atol=2e-7, rtol=0.0)
    assert np.max(np.abs(np.asarray(result.jacobian) - np.diag(np.diag(result.jacobian)))) > 1e-3
    assert result.post_round_recenter_applied is False
    assert result.latent_policy == anchor.LATENT_POLICY
    assert result.diagnostics["mean_owner"] == "exact_production_category_vector"
    assert result.diagnostics["putt_transfer_beta"] == 0.0
    assert result.diagnostics["putt_anchor_lambda_may_be_nonzero"] is True


def test_structurally_infeasible_target_fails_before_numerical_solver() -> None:
    model = SyntheticAbsorbingRound()
    target = np.asarray([20.0, 0.0, 0.0, 0.0])
    problem = _problem(model, target, (0.0, 0.0, 0.0, 0.0))
    result = anchor.solve_production_category_mean_anchor(problem, model, model.crn)
    assert not result.converged
    assert result.status == "infeasible_support"
    assert result.support_status == "infeasible_structural_category_bounds"
    assert result.diagnostics["infeasible_categories"] == ["ott"]
    assert result.methods_attempted == ("structural_support_preflight",)


@pytest.mark.parametrize(
    ("overrides", "expected_status"),
    [
        ({"max_newton_iterations": 0}, "converged_projected_broyden"),
        (
            {"max_newton_iterations": 0, "max_broyden_iterations": 0},
            "converged_bounded_least_squares",
        ),
    ],
)
def test_bounded_broyden_and_least_squares_fallbacks(
    overrides: dict[str, int], expected_status: str
) -> None:
    model = SyntheticAbsorbingRound(simulations=2_000)
    ability = (0.48, 0.31, -0.12, 0.27)
    true_lambda = np.asarray([0.23, -0.34, 0.29, 0.17])
    template = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=ability,
        reward_reference_sha256=REWARD_SHA,
        support_sha256=model.support_sha,
        context_key="fallback-target-builder",
    )
    target = np.asarray(
        model(template.parameters(true_lambda), model.crn).mean_category_sg
    )
    result = anchor.solve_production_category_mean_anchor(
        _problem(model, target, ability, **overrides), model, model.crn
    )
    assert result.converged, result.as_dict()
    assert result.status == expected_status
    assert max(abs(value) for value in result.residual_category_sg) <= 2e-9


def test_callback_must_be_exactly_deterministic_on_common_random_lattice() -> None:
    model = SyntheticAbsorbingRound(simulations=500)
    ability = (0.0, 0.0, 0.0, 0.0)
    target = np.asarray(
        model(
            anchor.MeanAnchorProblem(
                target_category_sg=(0.0, 0.0, 0.0, 0.0),
                target_ability_sg=ability,
                reward_reference_sha256=REWARD_SHA,
                support_sha256=model.support_sha,
                context_key="target",
            ).parameters(np.zeros(4)),
            model.crn,
        ).mean_category_sg
    )
    problem = _problem(model, target, ability)
    calls = 0

    def unstable(parameters, crn):
        nonlocal calls
        calls += 1
        result = model(parameters, crn)
        mean = np.asarray(result.mean_category_sg)
        mean[0] += calls * 1e-10
        return anchor.ChainExpectation(
            mean_category_sg=tuple(mean),
            mean_opportunities=result.mean_opportunities,
            support_lower_sg=result.support_lower_sg,
            support_upper_sg=result.support_upper_sg,
            simulations=result.simulations,
            crn_sha256=result.crn_sha256,
            reward_reference_sha256=result.reward_reference_sha256,
            support_sha256=result.support_sha256,
            quality_weight_config_sha256=result.quality_weight_config_sha256,
            quality_parameter_sha256=result.quality_parameter_sha256,
            clipped_donor_logs=result.clipped_donor_logs,
            clipped_target_logs=result.clipped_target_logs,
        )

    with pytest.raises(RuntimeError, match="not deterministic"):
        anchor.solve_production_category_mean_anchor(problem, unstable, model.crn)


def test_every_nonzero_solver_point_is_repeated_and_checked() -> None:
    support_sha = sha256(b"nonzero-determinism-regression").hexdigest()
    crn = anchor.CommonRandomContext(
        seed=813,
        simulations=1_000,
        semantic_key="regression|nonzero_determinism|round=1",
        payload_sha256=sha256(b"nonzero-determinism-crn").hexdigest(),
    )
    calls: dict[tuple[float, ...], int] = {}

    def callback(parameters, received_crn):
        values = np.asarray(parameters.lambda_correction, float)
        key = tuple(values)
        calls[key] = calls.get(key, 0) + 1
        mean = np.asarray([0.5 + values[0], values[1], values[2], values[3]])
        if np.any(values != 0.0) and calls[key] % 2 == 0:
            mean[0] += 1e-7
        return _algebraic_expectation(
            mean,
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )

    problem = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key="regression|nonzero_determinism|round=1",
        production_total_sg=0.0,
        deterministic_tolerance=0.0,
    )
    with pytest.raises(RuntimeError, match="not deterministic"):
        anchor.solve_production_category_mean_anchor(problem, callback, crn)
    assert calls[(0.0, 0.0, 0.0, 0.0)] == 2
    assert any(np.any(np.asarray(key) != 0.0) and count == 2 for key, count in calls.items())


@pytest.mark.parametrize(
    ("diagonal", "condition_limit", "expected_status", "expected_rank"),
    [
        (
            (1.0, 1.0, 1.0, 0.0),
            1e7,
            "unresolved_rank_deficient_endogenous_support",
            3,
        ),
        (
            (1.0, 1.0, 1.0, 1e-5),
            1e4,
            "unresolved_ill_conditioned_endogenous_support",
            4,
        ),
    ],
)
def test_rank_and_conditioning_are_hard_gates_even_at_zero_residual(
    diagonal: tuple[float, float, float, float],
    condition_limit: float,
    expected_status: str,
    expected_rank: int,
) -> None:
    support_sha = sha256(repr(diagonal).encode("utf-8")).hexdigest()
    crn = anchor.CommonRandomContext(
        seed=814,
        simulations=1_000,
        semantic_key=f"regression|jacobian_gate|{diagonal}",
        payload_sha256=sha256(b"jacobian-gate-crn").hexdigest(),
    )
    matrix = np.diag(np.asarray(diagonal, float))

    def callback(parameters, received_crn):
        values = np.asarray(parameters.lambda_correction, float)
        return _algebraic_expectation(
            matrix @ values,
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )

    problem = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key=f"regression|jacobian_gate|{diagonal}",
        production_total_sg=0.0,
        jacobian_rank_absolute_tolerance=1e-10,
        maximum_jacobian_condition_number=condition_limit,
    )
    result = anchor.solve_production_category_mean_anchor(problem, callback, crn)
    assert max(abs(value) for value in result.residual_category_sg) == 0.0
    assert result.converged is False
    assert result.production_admissible is False
    assert result.status == expected_status
    assert result.diagnostics["jacobian_rank"] == expected_rank
    assert result.diagnostics["admission_gates"]["componentwise_mean_match"] is True
    assert result.diagnostics["admission_gates"]["full_jacobian_rank"] is (
        expected_rank == 4
    )
    assert result.diagnostics["admission_gates"]["jacobian_conditioning"] is False


@pytest.mark.parametrize(("donor_ability", "target_ability", "message"), [
    (100.0, 0.0, "Donor quality log-weight clipping"),
    (0.0, 100.0, "Target quality log-weight clipping"),
])
def test_atom_log_weight_clipping_fails_closed(
    donor_ability: float,
    target_ability: float,
    message: str,
) -> None:
    parameters = anchor.TransitionQualityParameters(
        transfer_beta=anchor.DEFAULT_TRANSFER_BETA,
        target_ability_sg=(target_ability, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
    )
    with pytest.raises(RuntimeError, match=message):
        anchor.whole_atom_probabilities(
            base_mass=[0.5, 0.5],
            fixed_reference_reward=[1.0, -1.0],
            donor_ability_sg=[donor_ability, donor_ability],
            donor_group_id=["event", "event"],
            category="ott",
            parameters=parameters,
        )


def test_clipping_policy_is_hashed_and_callback_clipping_is_rejected() -> None:
    support_sha = sha256(b"clipping-provenance-regression").hexdigest()
    base_fields = dict(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key="regression|clipping_provenance|round=1",
        production_total_sg=0.0,
    )
    problem_30 = anchor.MeanAnchorProblem(**base_fields, max_abs_log_weight=30.0)
    problem_20 = anchor.MeanAnchorProblem(**base_fields, max_abs_log_weight=20.0)
    assert problem_30.quality_weight_config_sha256 != problem_20.quality_weight_config_sha256
    assert problem_30.problem_sha256 != problem_20.problem_sha256
    assert (
        problem_30.parameters(np.zeros(4)).parameter_sha256
        != problem_20.parameters(np.zeros(4)).parameter_sha256
    )

    crn = anchor.CommonRandomContext(
        seed=815,
        simulations=1_000,
        semantic_key="regression|clipping_provenance|round=1",
        payload_sha256=sha256(b"clipping-provenance-crn").hexdigest(),
    )

    def clipped_callback(parameters, received_crn):
        result = _algebraic_expectation(
            np.zeros(4),
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )
        return anchor.ChainExpectation(
            **{
                **result.__dict__,
                "clipped_target_logs": 1,
            }
        )

    with pytest.raises(RuntimeError, match="log-weight clipping"):
        anchor.solve_production_category_mean_anchor(
            problem_30, clipped_callback, crn
        )


def test_no_second_total_owner_or_week_latent_can_enter_problem() -> None:
    model = SyntheticAbsorbingRound(simulations=100)
    with pytest.raises(ValueError, match="total disagrees"):
        anchor.MeanAnchorProblem(
            target_category_sg=(0.1, 0.2, 0.3, 0.4),
            target_ability_sg=(0.0, 0.0, 0.0, 0.0),
            reward_reference_sha256=REWARD_SHA,
            support_sha256=model.support_sha,
            context_key="bad-total",
            production_total_sg=2.0,
        )
    with pytest.raises(ValueError, match="explicitly deferred"):
        anchor.MeanAnchorProblem(
            target_category_sg=(0.1, 0.2, 0.3, 0.4),
            target_ability_sg=(0.0, 0.0, 0.0, 0.0),
            reward_reference_sha256=REWARD_SHA,
            support_sha256=model.support_sha,
            context_key="bad-latent",
            latent_policy="apply_week_latent_inside_transition_solver",
        )


def _algebraic_expectation(
    mean: np.ndarray,
    *,
    parameters: anchor.TransitionQualityParameters,
    crn: anchor.CommonRandomContext,
    support_sha: str,
) -> anchor.ChainExpectation:
    return anchor.ChainExpectation(
        mean_category_sg=tuple(map(float, mean)),
        mean_opportunities=(18.0, 18.0, 6.0, 20.0),
        support_lower_sg=(-20.0, -20.0, -20.0, -20.0),
        support_upper_sg=(20.0, 20.0, 20.0, 20.0),
        simulations=crn.simulations,
        crn_sha256=crn.digest,
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        quality_weight_config_sha256=parameters.quality_weight_config_sha256,
        quality_parameter_sha256=parameters.parameter_sha256,
        clipped_donor_logs=0,
        clipped_target_logs=0,
    )


def test_componentwise_feasible_cache_point_beats_lower_l2_infeasible_probe() -> None:
    """Regression: lower L2 does not beat an admitted componentwise point."""

    step = 0.03
    support_sha = sha256(b"cached-l2-componentwise-regression").hexdigest()
    crn = anchor.CommonRandomContext(
        seed=811,
        simulations=1_000,
        semantic_key="regression|cached_l2|round=1",
        payload_sha256=sha256(b"cached-l2-crn").hexdigest(),
    )

    def callback(parameters, received_crn):
        values = np.asarray(parameters.lambda_correction, float)
        if np.allclose(values, 0.0, atol=1e-13, rtol=0.0):
            residual = np.asarray([1.0, 0.0, 0.0, 0.0])
        elif np.allclose(values, [step, 0.0, 0.0, 0.0], atol=1e-13, rtol=0.0):
            # Lowest cached L2, but OTT is outside the 0.001 component gate.
            residual = np.asarray([0.0011, 0.0, 0.0, 0.0])
        elif np.allclose(values, [-step, 0.0, 0.0, 0.0], atol=1e-13, rtol=0.0):
            residual = np.asarray([0.1211, 0.0, 0.0, 0.0])
        elif np.allclose(values, [0.5, 0.0, 0.0, 0.0], atol=1e-10, rtol=0.0):
            # Newton accepts this point componentwise, but its L2 is larger.
            residual = np.full(4, 0.0009)
        else:
            residual = np.asarray(
                [1.0 - 2.0 * values[0], values[1], values[2], values[3]]
            )
        return _algebraic_expectation(
            residual,
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )

    problem = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key="regression|cached_l2|round=1",
        production_total_sg=0.0,
        category_tolerance_sg=0.001,
        finite_difference_step=step,
        max_newton_iterations=1,
        max_broyden_iterations=0,
        max_fallback_evaluations=0,
    )
    result = anchor.solve_production_category_mean_anchor(
        problem, callback, crn
    )
    assert result.lambda_correction == pytest.approx((0.5, 0.0, 0.0, 0.0))
    assert max(abs(value) for value in result.residual_category_sg) == pytest.approx(
        0.0009
    )
    assert result.converged is True
    assert result.production_admissible is True
    assert result.status == "converged_bounded_newton"
    assert result.diagnostics["unresolved_categories"] == []


def test_bound_attribution_uses_cross_category_jacobian_coupling() -> None:
    """Regression: APP lambda can bind an unresolved OTT mean equation."""

    support_sha = sha256(b"swapped-bound-coordinate-regression").hexdigest()
    crn = anchor.CommonRandomContext(
        seed=812,
        simulations=1_000,
        semantic_key="regression|swapped_bound|round=1",
        payload_sha256=sha256(b"swapped-bound-crn").hexdigest(),
    )
    swapped = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    root_outside_bounds = np.asarray([0.0, -2.0, 0.0, 0.0])

    def callback(parameters, received_crn):
        values = np.asarray(parameters.lambda_correction, float)
        residual = swapped @ (values - root_outside_bounds)
        return _algebraic_expectation(
            residual,
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )

    problem = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key="regression|swapped_bound|round=1",
        production_total_sg=0.0,
        lambda_lower=(-1.0, -1.0, -1.0, -1.0),
        lambda_upper=(1.0, 1.0, 1.0, 1.0),
        category_tolerance_sg=0.001,
        max_newton_iterations=3,
        max_broyden_iterations=2,
        max_fallback_evaluations=20,
    )
    result = anchor.solve_production_category_mean_anchor(
        problem, callback, crn
    )
    assert result.converged is False
    assert result.lambda_correction[1] == pytest.approx(-1.0)
    assert result.residual_category_sg[0] == pytest.approx(1.0)
    assert result.status == "locally_bound_limited_unresolved"
    assert result.support_status == "local_solver_point_bound_limited"
    assert result.diagnostics["active_lower_bounds"] == ["app"]
    assert result.diagnostics["unresolved_categories"] == ["ott"]
    assert result.diagnostics["blocked_bound_parameters"] == ["app"]
    assert result.diagnostics["bound_limited_residual_categories"] == ["ott"]
    assert result.diagnostics["bound_parameter_to_residual_categories"] == {
        "app": ["ott"]
    }


def test_local_bound_limit_does_not_claim_global_infeasibility() -> None:
    """A locally stuck bound point can coexist with another in-bounds root."""

    support_sha = sha256(b"nonmonotone-known-root-bound-regression").hexdigest()
    crn = anchor.CommonRandomContext(
        seed=813,
        simulations=1_000,
        semantic_key="regression|nonmonotone_bound|round=1",
        payload_sha256=sha256(b"nonmonotone-bound-crn").hexdigest(),
    )

    def callback(parameters, received_crn):
        values = np.asarray(parameters.lambda_correction, float)
        residual = np.asarray(
            [
                (values[0] - 0.5) * (values[0] + 2.0),
                values[1],
                values[2],
                values[3],
            ]
        )
        return _algebraic_expectation(
            residual,
            parameters=parameters,
            crn=received_crn,
            support_sha=support_sha,
        )

    problem = anchor.MeanAnchorProblem(
        target_category_sg=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_SHA,
        support_sha256=support_sha,
        context_key="regression|nonmonotone_bound|round=1",
        production_total_sg=0.0,
        lambda_lower=(-1.0, -1.0, -1.0, -1.0),
        lambda_upper=(1.0, 1.0, 1.0, 1.0),
    )
    known_root = np.asarray([0.5, 0.0, 0.0, 0.0])
    assert callback(
        problem.parameters(known_root), crn
    ).mean_category_sg == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=0.0)

    # Use the production defaults for all three solver iteration limits.  The
    # lower bound is a constrained local least-squares minimum even though the
    # distinct +0.5 root is inside the same declared bounds.
    result = anchor.solve_production_category_mean_anchor(
        problem,
        callback,
        crn,
        initial_lambda=(-1.0, 0.0, 0.0, 0.0),
    )
    assert problem.max_newton_iterations == 12
    assert problem.max_broyden_iterations == 16
    assert problem.max_fallback_evaluations == 160
    assert result.converged is False
    assert result.production_admissible is False
    assert result.lambda_correction[0] == pytest.approx(-1.0, abs=1e-8)
    assert result.status == "locally_bound_limited_unresolved"
    assert result.support_status == "local_solver_point_bound_limited"
    assert result.diagnostics["jacobian_full_rank"] is True
    assert result.diagnostics["jacobian_conditioning_passed"] is True
    assert result.diagnostics["active_lower_bounds"] == ["ott"]
    assert result.diagnostics["blocked_bound_parameters"] == ["ott"]
    assert result.diagnostics["bound_limited_residual_categories"] == ["ott"]
    assert result.diagnostics["bound_parameter_to_residual_categories"] == {
        "ott": ["ott"]
    }

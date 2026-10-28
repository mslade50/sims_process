from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_compact_app_whole_atom_selector_v1 as compact
import hole_sim_donor_ability_policy as donor_policy
import hole_sim_transition_mean_anchor as anchor


REWARD_REFERENCE_SHA256 = (
    "a70ebc4aa3ec0dab67eb4d69722e48ec3f1fd6ec390643b53f21a63d6c45219b"
)
BRIDGES = tuple(f"compact-app-{index:02d}" for index in range(11))


def _stable_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _policy_rows() -> pd.DataFrame:
    availability = (
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
    )
    raw = (None, 0.4, None, 3.25, None, -0.6, -2.75, None, None, 0.7, None)
    skill_date = (
        None,
        "2024-01-01T00:00:00Z",
        None,
        "2024-01-02T00:00:00Z",
        "2024-01-03T00:00:00Z",
        "2024-01-04T00:00:00Z",
        "2024-01-05T00:00:00Z",
        None,
        "2024-01-06T00:00:00Z",
        "2024-01-07T00:00:00Z",
        None,
    )
    donor_event = ("event-a",) * 6 + ("event-b",) * 5
    donor_start = ("2024-01-11T00:00:00Z",) * 6 + (
        "2024-01-18T00:00:00Z",
    ) * 5
    frame = pd.DataFrame(
        {
            "bridge_row_id": BRIDGES,
            "category": ("app",) * 11,
            "donor_player_id": tuple(f"app-donor-{index:02d}" for index in range(11)),
            "donor_event_key": donor_event,
            "donor_event_start": donor_start,
            "target_event_key": ("synthetic-app-target",) * 11,
            "target_event_start": ("2024-03-07T00:00:00Z",) * 11,
            "donor_skill_date": skill_date,
            "donor_ability_source_column": ("sg_app",) * 11,
            "donor_ability_available": availability,
            "donor_ability_raw_strict_prior": raw,
        }
    )
    applied = donor_policy.apply_frozen_policy(frame)
    return applied.set_index("bridge_row_id", drop=False).loc[list(BRIDGES)].reset_index(
        drop=True
    )


def _support() -> compact.CompactAppSupport:
    policy = _policy_rows()
    frozen_spatial_mass = 0.8
    first_proposal_mass = frozen_spatial_mass * float(np.nextafter(0.4, 1.0))
    return compact.CompactAppSupport(
        support_key="synthetic-compact-app|support=000",
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
        frozen_holeout_probability=0.2,
        atom_id=tuple(f"app-atom-{index:02d}" for index in range(11)),
        base_mass=(
            0.1,
            0.1,
            0.0,
            0.032,
            0.048,
            first_proposal_mass - 0.08,
            0.12,
            0.12,
            0.04,
            0.12,
            frozen_spatial_mass - first_proposal_mass - 0.40,
        ),
        fixed_reference_reward=(
            1.8,
            -1.6,
            -0.2,
            -1.2,
            0.8,
            1.5,
            -0.6,
            1.1,
            0.4,
            -1.4,
            2.0,
        ),
        bridge_row_id=tuple(policy["bridge_row_id"]),
        category=tuple(policy["category"]),
        donor_player_id=tuple(policy["donor_player_id"]),
        donor_event_key=tuple(policy["donor_event_key"]),
        donor_event_start=tuple(policy["donor_event_start"]),
        target_event_key=tuple(policy["target_event_key"]),
        target_event_start=tuple(policy["target_event_start"]),
        donor_skill_date=tuple(policy["donor_skill_date"]),
        donor_ability_source_column=tuple(policy["donor_ability_source_column"]),
        donor_ability_available=tuple(policy["donor_ability_available"]),
        donor_ability_raw_strict_prior=tuple(policy["donor_ability_raw_strict_prior"]),
        donor_ability_value_kind=tuple(policy["donor_ability_value_kind"]),
        donor_ability_sg=tuple(policy["donor_ability_sg"]),
        donor_ability_was_clipped=tuple(policy["donor_ability_was_clipped"]),
        donor_ability_policy_sha256=tuple(policy["donor_ability_policy_sha256"]),
        donor_ability_row_sha256=tuple(policy["donor_ability_row_sha256"]),
        original_donor_group_id=("group-a",) * 6 + ("group-b",) * 5,
        terminal=(True, True) + (False,) * 9,
        terminal_donor_order=(0, 1) + (-1,) * 9,
        spatial_proposal_order=(-1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2),
        surface_order=(-1, -1, 0, 1, 1, 2, 0, 0, 1, 1, 0),
        hazard_order=(-1, -1, 0, 0, 1, 0, 0, 1, 0, 1, 0),
        frozen_terminal_residual_cdf_upper=(0.5, 1.0) + (0.0,) * 9,
        frozen_spatial_proposal_cdf_upper=(0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9, 1.0),
        frozen_surface_cdf_upper=(0.0, 0.0, 0.0, 0.25, 0.25, 1.0, 0.6, 0.6, 1.0, 1.0, 1.0),
        frozen_hazard_cdf_upper=(0.0, 0.0, 1.0, 0.4, 1.0, 1.0, 0.5, 1.0, 0.25, 1.0, 1.0),
        boundary_only=(False, False, True) + (False,) * 8,
        complete_atom=(True,) * 11,
        stroke_cost=(1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1),
        next_state_key=(
            "",
            "",
            "state-boundary",
            "state-a",
            "state-b",
            "state-c",
            "state-d",
            "state-e",
            "state-f",
            "state-g",
            "state-h",
        ),
    )


def _zero_parameters() -> compact.AppWeightingParameters:
    return compact.AppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.8, 0.2, -0.1, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )


def _quality_parameters() -> compact.AppWeightingParameters:
    return compact.AppWeightingParameters(
        transfer_beta=(0.0, 0.9, 0.0, 0.0),
        target_ability_sg=(0.8, 0.2, -0.1, 0.0),
        lambda_correction=(0.0, 0.15, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )


def _reviewer_parameters(
    target_app: float, lambda_app: float
) -> compact.AppWeightingParameters:
    return compact.AppWeightingParameters(
        transfer_beta=(0.0, 0.1, 0.0, 0.0),
        target_ability_sg=(0.0, target_app, 0.0, 0.0),
        lambda_correction=(0.0, lambda_app, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )


def _admit(
    support: compact.CompactAppSupport | None = None,
) -> compact.CompactAppSelectorSession:
    return compact.admit_synthetic_app_support(
        _support() if support is None else support,
        authority_manifest=compact.synthetic_app_authority_manifest(),
    )


def _replace_sequence(
    support: compact.CompactAppSupport, name: str, values: tuple[object, ...]
) -> compact.CompactAppSupport:
    return replace(support, **{name: values})


def test_closed_fixture_admits_complete_mixed_policy_app_support() -> None:
    session = _admit()
    readiness = session.readiness
    support = session.support_copy()
    assert readiness.rows == 11
    assert readiness.terminal_donors == 2
    assert readiness.spatial_proposals == 3
    assert readiness.original_donor_groups == 2
    assert readiness.mixed_policy_state_groups == 2
    assert readiness.observed_rows == 5
    assert readiness.neutral_imputed_rows == 6
    assert readiness.availability_preserved
    assert readiness.positive_zero_preserved
    assert readiness.synthetic_fixture_only
    assert readiness.weighting_ready and readiness.selector_ready
    assert not readiness.real_chain_ready and not readiness.integration_ready
    assert support.original_donor_group_id == ("group-a",) * 6 + ("group-b",) * 5
    imputed = np.asarray(support.donor_ability_value_kind) == donor_policy.IMPUTED_VALUE_KIND
    ability = np.asarray(support.donor_ability_sg, dtype=float)
    assert np.equal(ability[imputed], 0.0).all()
    assert not np.signbit(ability[imputed]).any()


def test_original_mixed_groups_and_positive_zero_reach_private_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _admit()
    support = session.support_copy()
    captured: list[tuple[np.ndarray, tuple[object, ...], str]] = []
    original = anchor.whole_atom_probabilities

    def spy(**kwargs: object) -> anchor.AtomMassResult:
        captured.append(
            (
                np.asarray(kwargs["donor_ability_sg"], dtype=float).copy(),
                tuple(kwargs["donor_group_id"]),
                str(kwargs["category"]),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(compact, "_whole_atom_probabilities", spy)
    weighted = session.weight(_quality_parameters())
    assert len(captured) == 1
    ability, groups, category = captured[0]
    imputed = np.asarray(support.donor_ability_value_kind) == donor_policy.IMPUTED_VALUE_KIND
    assert np.equal(ability[imputed], 0.0).all()
    assert not np.signbit(ability[imputed]).any()
    assert groups == support.original_donor_group_id
    assert category == "app"
    assert np.allclose(
        weighted.donor_group_mass_before,
        weighted.donor_group_mass_after,
        atol=2e-15,
        rtol=2e-15,
    )


def test_nonzero_terminal_probability_and_all_atom_factors_come_from_final_mass() -> None:
    session = _admit()
    support = session.support_copy()
    weighted = session.weight(_quality_parameters())
    probability = np.asarray(weighted.probability)
    terminal = np.asarray(support.terminal, dtype=bool)
    factorized = np.zeros(len(probability))
    factorized[terminal] = weighted.terminal_probability * (
        np.asarray(weighted.terminal_cdf_upper)[terminal]
        - np.asarray(weighted.terminal_cdf_lower)[terminal]
    )
    factorized[~terminal] = weighted.spatial_probability * (
        np.asarray(weighted.spatial_proposal_cdf_upper)[~terminal]
        - np.asarray(weighted.spatial_proposal_cdf_lower)[~terminal]
    ) * (
        np.asarray(weighted.surface_cdf_upper)[~terminal]
        - np.asarray(weighted.surface_cdf_lower)[~terminal]
    ) * (
        np.asarray(weighted.hazard_cdf_upper)[~terminal]
        - np.asarray(weighted.hazard_cdf_lower)[~terminal]
    )
    assert weighted.factorization_from_final_probability
    assert weighted.exact_spatial_mass_reconstruction
    assert weighted.binary64_pathwise_exact
    assert weighted.selection_mode == (
        "weighted_app_binary64_pathwise_from_unchanged_flat_probability_v2"
    )
    terminal_prefix = 0.0
    for value in probability[terminal]:
        terminal_prefix = float(terminal_prefix + float(value))
    assert weighted.terminal_probability == terminal_prefix
    assert weighted.terminal_unconditional_cdf_upper[0] == probability[0]
    assert weighted.terminal_unconditional_cdf_upper[1] == terminal_prefix
    spatial_exact = Fraction(weighted.spatial_probability_exact_ratio)
    for index in np.flatnonzero(~terminal):
        if support.boundary_only[int(index)]:
            assert probability[index] == 0.0
            assert weighted.surface_conditional_exact_ratio[index] is None
            assert weighted.hazard_conditional_exact_ratio[index] is None
            continue
        reconstructed_exact = (
            spatial_exact
            * Fraction(weighted.spatial_proposal_conditional_exact_ratio[index])
            * Fraction(weighted.surface_conditional_exact_ratio[index])
            * Fraction(weighted.hazard_conditional_exact_ratio[index])
        )
        assert reconstructed_exact == Fraction.from_float(float(probability[index]))
    assert np.isclose(
        weighted.terminal_probability,
        float(probability[terminal].sum()) / float(probability.sum()),
        atol=0.0,
        rtol=2e-15,
    )
    assert np.isclose(
        weighted.spatial_probability,
        float(probability[~terminal].sum()) / float(probability.sum()),
        atol=0.0,
        rtol=2e-15,
    )
    assert np.allclose(factorized, probability, atol=0.0, rtol=2e-12)
    assert not np.isclose(weighted.terminal_probability, 0.2, atol=1e-10, rtol=0.0)


def test_nonzero_flat_and_hierarchical_selection_match() -> None:
    session = _admit()
    weighted = session.weight(_quality_parameters())
    p = weighted.terminal_probability
    vectors = (
        (0.0, 0.0, 0.0, 0.0),
        (p * 0.3, 0.9, 0.9, 0.9),
        (p * 0.8, 0.1, 0.2, 0.3),
        (p, 0.0, 0.0, 0.0),
        (float(np.nextafter(p, 1.0)), 0.41, 0.60, 0.50),
        (0.99, 0.73, float(np.nextafter(0.5, 1.0)), 0.99),
    )
    for values in vectors:
        result = session.select(
            _quality_parameters(), compact.AppSemanticUniforms(*values)
        )
        assert result.flat_hierarchical_equal
        assert result.flat_atom_id == result.hierarchical_atom_id == result.atom_id


@pytest.mark.parametrize(
    ("target_app", "lambda_app", "expected"),
    [
        (
            -2.0,
            1.0,
            {
                "terminal": "0x1.8aa75ce133df5p-3",
                "terminal_first": "0x1.70c1ab0de48afp-3",
                "proposal_0": "0x1.08bc4038fff55p-1",
                "proposal_1": "0x1.8cb083ea1e143p-1",
                "surface_p0": "0x1.be44b70bfb42bp-4",
                "surface_p1": "0x1.9109a29d4cc7bp-1",
                "hazard_p1s0": "0x1.7bc15eed2909cp-3",
            },
        ),
        (
            0.0,
            0.1,
            {
                "terminal": "0x1.7de09dbfb6684p-3",
                "terminal_first": "0x1.b23e96860dbe0p-4",
                "proposal_0": "0x1.b63dadd723834p-2",
                "proposal_1": "0x1.c56d9a491057dp-1",
                "surface_p0": "0x1.cac869651deb4p-3",
                "surface_p1": "0x1.32bfed7330c0cp-1",
                "hazard_p1s0": "0x1.b646a54e7b8dfp-2",
            },
        ),
    ],
)
def test_reviewer_binary64_thresholds_use_unchanged_probability_and_nextafter(
    target_app: float,
    lambda_app: float,
    expected: dict[str, str],
) -> None:
    session = _admit()
    support = session.support_copy()
    parameters = _reviewer_parameters(target_app, lambda_app)
    weighted = session.weight(parameters)
    probability = np.asarray(weighted.probability, dtype=np.float64)

    anchor_parameters, _ = compact._validated_weighting_parameters(parameters)
    direct = anchor.whole_atom_probabilities(
        base_mass=support.base_mass,
        fixed_reference_reward=support.fixed_reference_reward,
        donor_ability_sg=support.donor_ability_sg,
        donor_group_id=support.original_donor_group_id,
        category="app",
        parameters=anchor_parameters,
    )
    assert probability.tobytes() == np.asarray(
        direct.probability, dtype=np.float64
    ).tobytes()
    terminal_prefix = float(probability[0] + probability[1])
    assert weighted.terminal_probability == terminal_prefix
    assert terminal_prefix.hex() == expected["terminal"]
    assert weighted.terminal_unconditional_cdf_upper[0] == probability[0]
    assert probability[0].hex() == expected["terminal_first"]
    assert weighted.terminal_unconditional_cdf_upper[1] == terminal_prefix
    assert weighted.spatial_proposal_cdf_upper[2].hex() == expected["proposal_0"]
    assert weighted.spatial_proposal_cdf_upper[6].hex() == expected["proposal_1"]
    assert weighted.surface_cdf_upper[3].hex() == expected["surface_p0"]
    assert weighted.surface_cdf_upper[6].hex() == expected["surface_p1"]
    assert weighted.hazard_cdf_upper[6].hex() == expected["hazard_p1s0"]

    exact = tuple(Fraction.from_float(float(value)) for value in probability)
    spatial_mass = sum(exact[2:], Fraction(0, 1))
    proposal_zero_mass = sum(exact[2:6], Fraction(0, 1))
    exact_proposal_cut = proposal_zero_mass / spatial_mass
    proposal_cut = weighted.spatial_proposal_cdf_upper[2]
    assert Fraction.from_float(float(np.nextafter(proposal_cut, 0.0))) < (
        exact_proposal_cut
    ) <= Fraction.from_float(proposal_cut)

    exact_surface_cut = sum(exact[3:5], Fraction(0, 1)) / proposal_zero_mass
    surface_cut = weighted.surface_cdf_upper[3]
    assert Fraction.from_float(surface_cut) <= exact_surface_cut < Fraction.from_float(
        float(np.nextafter(surface_cut, 1.0))
    )

    exact_hazard_cut = exact[6] / (exact[6] + exact[7])
    hazard_cut = weighted.hazard_cdf_upper[6]
    assert Fraction.from_float(float(np.nextafter(hazard_cut, 0.0))) < (
        exact_hazard_cut
    ) <= Fraction.from_float(hazard_cut)

    def selected(
        holeout: float, proposal: float = 0.0, surface: float = 0.0, hazard: float = 0.0
    ) -> compact.AppSelectionResult:
        return session.select(
            parameters,
            compact.AppSemanticUniforms(holeout, proposal, surface, hazard),
        )

    q0 = float(probability[0])
    assert selected(float(np.nextafter(q0, 0.0))).terminal_donor_order == 0
    assert selected(q0).terminal_donor_order == 1
    assert selected(float(np.nextafter(q0, 1.0))).terminal_donor_order == 1
    assert selected(float(np.nextafter(terminal_prefix, 0.0))).terminal
    assert not selected(terminal_prefix).terminal
    assert not selected(float(np.nextafter(terminal_prefix, 1.0))).terminal

    assert selected(
        terminal_prefix, float(np.nextafter(proposal_cut, 0.0)), 0.1, 0.1
    ).spatial_proposal_order == 0
    assert selected(terminal_prefix, proposal_cut, 0.1, 0.1).spatial_proposal_order == 1
    assert selected(
        terminal_prefix, float(np.nextafter(proposal_cut, 1.0)), 0.1, 0.1
    ).spatial_proposal_order == 1

    assert selected(terminal_prefix, 0.0, surface_cut, 0.1).surface_order == 1
    assert selected(
        terminal_prefix, 0.0, float(np.nextafter(surface_cut, 1.0)), 0.1
    ).surface_order == 2

    assert selected(
        terminal_prefix, 0.7, 0.0, float(np.nextafter(hazard_cut, 0.0))
    ).hazard_order == 0
    assert selected(terminal_prefix, 0.7, 0.0, hazard_cut).hazard_order == 1

    if target_app == -2.0:
        spatial_sum = float(sum(map(float, probability[2:])))
        old_normalized_p = terminal_prefix / (terminal_prefix + spatial_sum)
        assert old_normalized_p.hex() == "0x1.8aa75ce133df3p-3"
        assert selected(old_normalized_p).terminal
    else:
        old_residual_cdf = float(q0 / terminal_prefix)
        old_multiply_cut = float(terminal_prefix * old_residual_cdf)
        assert old_multiply_cut == float(np.nextafter(q0, 0.0))
        assert selected(old_multiply_cut).terminal_donor_order == 0
        assert selected(q0).terminal_donor_order == 1


@pytest.mark.parametrize(("target_app", "lambda_app"), [(-2.0, 1.0), (0.0, 0.1)])
def test_bounded_binary64_cut_grid_matches_independent_probability_oracle(
    target_app: float, lambda_app: float
) -> None:
    session = _admit()
    support = session.support_copy()
    weighted = session.weight(_reviewer_parameters(target_app, lambda_app))
    hierarchy = {
        name: getattr(weighted, name)
        for name in (
            "terminal_probability",
            "terminal_cdf_lower",
            "terminal_cdf_upper",
            "terminal_unconditional_cdf_lower",
            "terminal_unconditional_cdf_upper",
            "spatial_proposal_cdf_lower",
            "spatial_proposal_cdf_upper",
            "surface_cdf_lower",
            "surface_cdf_upper",
            "hazard_cdf_lower",
            "hazard_cdf_upper",
        )
    }

    def lattice(cuts: set[float]) -> tuple[float, ...]:
        retained = {0.0, 0.123456789, float(np.nextafter(1.0, 0.0))}
        for cut in cuts:
            if 0.0 < cut < 1.0:
                retained.add(float(np.nextafter(cut, 0.0)))
                retained.add(float(cut))
                upper = float(np.nextafter(cut, 1.0))
                if upper < 1.0:
                    retained.add(upper)
        return tuple(sorted(retained))

    terminal_cuts = {
        weighted.terminal_unconditional_cdf_upper[i]
        for i, value in enumerate(support.terminal)
        if value
    }
    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    proposal_cuts = {
        weighted.spatial_proposal_cdf_upper[i]
        for i in spatial_indices
        if weighted.spatial_proposal_cdf_upper[i] < 1.0
    }
    surface_cuts = {
        weighted.surface_cdf_upper[i]
        for i in spatial_indices
        if 0.0 < weighted.surface_cdf_upper[i] < 1.0
    }
    hazard_cuts = {
        weighted.hazard_cdf_upper[i]
        for i in spatial_indices
        if 0.0 < weighted.hazard_cdf_upper[i] < 1.0
    }
    grid = product(
        lattice(terminal_cuts),
        lattice(proposal_cuts),
        lattice(surface_cuts),
        lattice(hazard_cuts),
    )
    checked = 0
    for uniforms in grid:
        hierarchical, _ = compact._hierarchical_select(
            support, hierarchy, uniforms, allow_boundary_only=False
        )
        direct = compact._probability_direct_select(
            support, weighted.probability, uniforms
        )
        assert hierarchical == direct
        checked += 1
    assert 1_000 <= checked <= 20_000


def test_zero_fast_path_replays_holeout_and_terminal_residual_boundaries() -> None:
    session = _admit()

    def selected(holeout: float) -> compact.AppSelectionResult:
        return session.select(
            _zero_parameters(), compact.AppSemanticUniforms(holeout, 0.2, 0.2, 0.2)
        )

    at_zero = selected(0.0)
    terminal_exact = selected(0.1)
    terminal_below = selected(float(np.nextafter(0.1, 0.0)))
    terminal_above = selected(float(np.nextafter(0.1, 1.0)))
    holeout_exact = selected(0.2)
    holeout_below = selected(float(np.nextafter(0.2, 0.0)))
    holeout_above = selected(float(np.nextafter(0.2, 1.0)))
    assert at_zero.terminal and at_zero.terminal_donor_order == 0
    assert at_zero.terminal_residual_uniform == 0.0
    assert terminal_exact.terminal_donor_order == 1
    assert terminal_below.terminal_donor_order == 0
    assert terminal_above.terminal_donor_order == 1
    assert not holeout_exact.terminal
    assert holeout_below.terminal and holeout_below.terminal_donor_order == 1
    assert not holeout_above.terminal
    assert holeout_exact.terminal_residual_uniform is None
    assert all(
        result.terminal_probability == 0.2
        for result in (
            at_zero,
            terminal_exact,
            terminal_below,
            terminal_above,
            holeout_exact,
            holeout_below,
            holeout_above,
        )
    )


def test_zero_fast_path_retains_every_frozen_scalar_and_cdf_exactly() -> None:
    session = _admit()
    support = session.support_copy()
    weighted = session.weight(_zero_parameters())
    spatial_indices = [i for i, value in enumerate(support.terminal) if not value]
    frozen_numeric_boundary = support.frozen_spatial_proposal_cdf_upper[2]
    base_numeric_boundary = sum(support.base_mass[2:6]) / sum(
        support.base_mass[i] for i in spatial_indices
    )
    assert base_numeric_boundary > frozen_numeric_boundary
    assert weighted.selection_mode == compact.APP_ZERO_FAST_PATH_CONTRACT
    assert not weighted.exact_spatial_mass_reconstruction
    assert weighted.spatial_probability_exact_ratio is None
    assert weighted.terminal_probability == support.frozen_holeout_probability
    assert weighted.terminal_cdf_upper == support.frozen_terminal_residual_cdf_upper
    assert (
        weighted.spatial_proposal_cdf_upper
        == support.frozen_spatial_proposal_cdf_upper
    )
    assert weighted.surface_cdf_upper == support.frozen_surface_cdf_upper
    assert weighted.hazard_cdf_upper == support.frozen_hazard_cdf_upper


def test_zero_fast_path_replays_numeric_proposal_and_strict_surface_hazard_cdfs() -> None:
    session = _admit()

    def selected(proposal: float, surface: float, hazard: float) -> compact.AppSelectionResult:
        return session.select(
            _zero_parameters(),
            compact.AppSemanticUniforms(0.2, proposal, surface, hazard),
        )

    proposal_exact = selected(0.4, 0.2, 0.2)
    proposal_below = selected(float(np.nextafter(0.4, 0.0)), 0.2, 0.2)
    proposal_above = selected(float(np.nextafter(0.4, 1.0)), 0.2, 0.2)
    assert proposal_exact.spatial_proposal_order == 1
    assert proposal_below.spatial_proposal_order == 0
    assert proposal_above.spatial_proposal_order == 1

    surface_exact = selected(0.5, 0.6, 0.0)
    surface_below = selected(0.5, float(np.nextafter(0.6, 0.0)), 0.0)
    surface_above = selected(0.5, float(np.nextafter(0.6, 1.0)), 0.0)
    assert surface_exact.surface_order == 0
    assert surface_below.surface_order == 0
    assert surface_above.surface_order == 1

    hazard_exact = selected(0.5, 0.2, 0.5)
    hazard_below = selected(0.5, 0.2, float(np.nextafter(0.5, 0.0)))
    hazard_above = selected(0.5, 0.2, float(np.nextafter(0.5, 1.0)))
    assert hazard_exact.hazard_order == 1
    assert hazard_below.hazard_order == 0
    assert hazard_above.hazard_order == 1
    assert all(
        result.flat_hierarchical_equal
        for result in (
            proposal_exact,
            proposal_below,
            proposal_above,
            surface_exact,
            surface_below,
            surface_above,
            hazard_exact,
            hazard_below,
            hazard_above,
        )
    )


def test_zero_mass_surface_sentinel_replays_only_at_zero_tilt() -> None:
    session = _admit()
    exact = session.select(
        _zero_parameters(), compact.AppSemanticUniforms(0.2, 0.2, 0.0, 0.0)
    )
    above = session.select(
        _zero_parameters(),
        compact.AppSemanticUniforms(0.2, 0.2, float(np.nextafter(0.0, 1.0)), 0.0),
    )
    tilted = session.select(
        _quality_parameters(), compact.AppSemanticUniforms(0.99, 0.0, 0.0, 0.0)
    )
    assert exact.boundary_only and exact.surface_order == 0
    assert not above.boundary_only and above.surface_order == 1
    assert not tilted.boundary_only
    assert tilted.surface_order == 1


def test_honest_policy_clipping_and_exact_imputation_are_bound() -> None:
    support = _admit().support_copy()
    raw = np.asarray(
        [np.nan if value is None else value for value in support.donor_ability_raw_strict_prior]
    )
    clipped = np.asarray(support.donor_ability_was_clipped, dtype=bool)
    applied = np.asarray(support.donor_ability_sg, dtype=float)
    assert applied[int(np.nanargmax(raw))] == 2.0
    assert applied[int(np.nanargmin(raw))] == -2.0
    assert clipped[int(np.nanargmax(raw))] and clipped[int(np.nanargmin(raw))]

    flags = list(support.donor_ability_was_clipped)
    flags[int(np.nanargmax(raw))] = False
    with pytest.raises(compact.CompactAppSupportUnavailable, match="attestation failed"):
        _admit(_replace_sequence(support, "donor_ability_was_clipped", tuple(flags)))


@pytest.mark.parametrize("attack", ["availability", "negative_zero", "split_group"])
def test_imputed_state_and_original_group_mutations_fail_closed(attack: str) -> None:
    support = _admit().support_copy()
    imputed_index = support.donor_ability_value_kind.index(
        donor_policy.IMPUTED_VALUE_KIND
    )
    if attack == "availability":
        values = list(support.donor_ability_available)
        values[imputed_index] = True
        forged = _replace_sequence(support, "donor_ability_available", tuple(values))
        message = "attestation failed|availability"
    elif attack == "negative_zero":
        values = list(support.donor_ability_sg)
        values[imputed_index] = -0.0
        forged = _replace_sequence(support, "donor_ability_sg", tuple(values))
        message = r"positive.*0\.0"
    else:
        values = list(support.original_donor_group_id)
        values[imputed_index] = f"{values[imputed_index]}|imputed"
        forged = _replace_sequence(support, "original_donor_group_id", tuple(values))
        message = "outside the closed synthetic authority"
    with pytest.raises(compact.CompactAppSupportUnavailable, match=message):
        _admit(forged)


def test_clipping_log_cap_negative_beta_and_huge_uniform_fail_closed() -> None:
    session = _admit()
    with pytest.raises(compact.CompactAppSupportUnavailable, match="log cap differs"):
        session.weight(replace(_quality_parameters(), max_abs_log_weight=31.0))
    with pytest.raises(compact.CompactAppSupportUnavailable, match="cannot be negative"):
        session.weight(
            replace(_quality_parameters(), transfer_beta=(0.0, -0.1, 0.0, 0.0))
        )
    with pytest.raises(
        compact.CompactAppSupportUnavailable,
        match="clipping is not admissible|weighting rejected",
    ):
        session.weight(
            replace(_quality_parameters(), transfer_beta=(0.0, 100.0, 0.0, 0.0))
        )
    with pytest.raises(compact.CompactAppSupportUnavailable, match="finite"):
        session.select(
            _zero_parameters(), compact.AppSemanticUniforms(10**10000, 0.0, 0.0, 0.0)
        )


def test_positive_node_cdf_rounding_collapse_fails_closed() -> None:
    parameters = compact.AppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, -11.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )
    with pytest.raises(
        compact.CompactAppSupportUnavailable,
        match="collapsed|zero-width|unreachable|no distinct binary64",
    ):
        _admit().weight(parameters)


def test_atom_terminal_and_spatial_hierarchy_caps_fail_closed() -> None:
    support = _admit().support_copy()
    additions = compact._MAX_ATOMS + 1 - len(support.atom_id)
    updates: dict[str, tuple[object, ...]] = {}
    for field in fields(compact.CompactAppSupport):
        if field.name in {
            "support_key",
            "reward_reference_sha256",
            "frozen_holeout_probability",
        }:
            continue
        sequence = tuple(getattr(support, field.name))
        updates[field.name] = sequence + (sequence[-1],) * additions
    with pytest.raises(compact.CompactAppSupportUnavailable, match="atom cap invalid"):
        _admit(replace(support, **updates))

    terminal = (True,) * 5 + (False,) * 6
    with pytest.raises(compact.CompactAppSupportUnavailable, match="Terminal donor cap"):
        _admit(replace(support, terminal=terminal))

    proposal = list(support.spatial_proposal_order)
    proposal[2:] = list(range(9))
    with pytest.raises(compact.CompactAppSupportUnavailable, match="proposal cap"):
        _admit(replace(support, spatial_proposal_order=tuple(proposal)))


def test_terminal_holeout_cost_and_multiple_donor_contract_fail_closed() -> None:
    support = _admit().support_copy()
    costs = list(support.stroke_cost)
    costs[0] = 2
    with pytest.raises(compact.CompactAppSupportUnavailable, match="one-stroke holeout"):
        _admit(replace(support, stroke_cost=tuple(costs)))

    terminal = list(support.terminal)
    terminal[1] = False
    with pytest.raises(compact.CompactAppSupportUnavailable, match="multiple terminal"):
        _admit(replace(support, terminal=tuple(terminal)))


def test_malformed_cdf_and_structurally_valid_resealed_candidate_are_rejected() -> None:
    support = _admit().support_copy()
    invalid_hazard = list(support.frozen_hazard_cdf_upper)
    invalid_hazard[3] = 1.0
    with pytest.raises(compact.CompactAppSupportUnavailable, match="ordered CDF"):
        _admit(
            _replace_sequence(
                support, "frozen_hazard_cdf_upper", tuple(invalid_hazard)
            )
        )

    changed = replace(support, support_key="synthetic-compact-app|support=caller")
    canonical = compact._canonicalize_support(changed)
    forged_authority = deepcopy(compact.synthetic_app_authority_manifest())
    manifest = forged_authority["manifest"]
    assert isinstance(manifest, dict)
    manifest["support_sha256"] = compact._support_sha256(canonical)
    manifest["frozen_cdf_sha256"] = compact._frozen_cdf_sha256(canonical)
    forged_authority["manifest_sha256"] = _stable_sha(manifest)
    with pytest.raises(compact.CompactAppSupportUnavailable, match="closed source-pinned"):
        compact.admit_synthetic_app_support(
            changed, authority_manifest=forged_authority
        )


def test_authority_session_parameter_uniform_and_plain_string_exact_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileStr(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError(f"hostile equality dispatched: {other!r}")

    authority = compact.synthetic_app_authority_manifest()
    hostile_authority = deepcopy(authority)
    hostile_manifest = hostile_authority["manifest"]
    assert isinstance(hostile_manifest, dict)
    hostile_manifest["scope"] = HostileStr(str(hostile_manifest["scope"]))
    with pytest.raises(compact.CompactAppSupportUnavailable, match="non-plain"):
        compact.admit_synthetic_app_support(
            _support(), authority_manifest=hostile_authority
        )

    primitive_calls: list[bool] = []

    def forbidden_primitive(**_: object) -> object:
        primitive_calls.append(True)
        raise AssertionError("weighting primitive must not run")

    monkeypatch.setattr(compact, "_whole_atom_probabilities", forbidden_primitive)
    session = _admit()

    class ParameterSubclass(compact.AppWeightingParameters):
        pass

    class UniformSubclass(compact.AppSemanticUniforms):
        pass

    bad_parameters = ParameterSubclass(**_zero_parameters().__dict__)
    with pytest.raises(compact.CompactAppSupportUnavailable, match="wrong exact type"):
        session.weight(bad_parameters)
    with pytest.raises(compact.CompactAppSupportUnavailable, match="wrong exact type"):
        session.select(
            _zero_parameters(), UniformSubclass(0.0, 0.0, 0.0, 0.0)
        )
    assert primitive_calls == []

    class SessionSubclass(compact.CompactAppSelectorSession):
        pass

    forged_session = object.__new__(SessionSubclass)
    forged_session._support = session.support_copy()
    forged_session._authority = session._authority
    forged_session._readiness = session.readiness
    with pytest.raises(compact.CompactAppSupportUnavailable, match="session.*exact type"):
        forged_session.weight(_zero_parameters())
    assert primitive_calls == []


def test_authority_cycle_fails_with_domain_error() -> None:
    authority = compact.synthetic_app_authority_manifest()
    manifest = authority["manifest"]
    assert isinstance(manifest, dict)
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    manifest["cycle"] = cycle
    with pytest.raises(compact.CompactAppSupportUnavailable, match="container cycle"):
        compact.admit_synthetic_app_support(_support(), authority_manifest=authority)


def test_readiness_is_recomputed_from_live_closed_session() -> None:
    session = _admit()
    honest = session.readiness
    session._readiness = replace(
        honest,
        synthetic_fixture_only=False,
        real_chain_ready=True,
        integration_ready=True,
        blockers=(),
    )
    retained = session.readiness
    assert retained == honest
    assert retained.synthetic_fixture_only
    assert not retained.real_chain_ready
    assert not retained.integration_ready
    assert retained.blockers


def test_frozen_cdf_digest_binds_support_identity_and_atom_order() -> None:
    support = _admit().support_copy()
    renamed = replace(support, support_key="renamed-synthetic-app-support")
    assert compact._frozen_cdf_sha256(renamed) != compact._frozen_cdf_sha256(support)

    atom_ids = list(support.atom_id)
    atom_ids[0], atom_ids[1] = atom_ids[1], atom_ids[0]
    reordered_identity = replace(support, atom_id=tuple(atom_ids))
    assert compact._frozen_cdf_sha256(reordered_identity) != compact._frozen_cdf_sha256(
        support
    )

def test_candidate_derived_private_authority_cannot_reseal_session() -> None:
    admitted = _admit()
    support = admitted.support_copy()
    changed = replace(support, support_key="caller-resealed-support")
    canonical = compact._canonicalize_support(changed)
    forged_authority = compact._TrustedSyntheticAppAuthority(
        manifest_sha256=admitted.readiness.authority_manifest_sha256,
        runtime_binding_sha256=admitted.readiness.runtime_binding_sha256,
        support_sha256=compact._support_sha256(canonical),
        frozen_cdf_sha256=compact._frozen_cdf_sha256(canonical),
        reward_reference_sha256=canonical.reward_reference_sha256,
    )
    forged = object.__new__(compact.CompactAppSelectorSession)
    forged._support = canonical
    forged._authority = forged_authority
    forged._readiness = admitted.readiness
    with pytest.raises(compact.CompactAppSupportUnavailable, match="authority changed"):
        forged.weight(_zero_parameters())


def test_direct_session_and_public_primitive_solver_routes_are_absent() -> None:
    with pytest.raises(compact.CompactAppSupportUnavailable, match="minted only"):
        compact.CompactAppSelectorSession()
    assert not hasattr(compact, "anchor")
    assert not hasattr(compact, "whole_atom_probabilities")
    assert not hasattr(compact, "solve_mean_anchor")
    assert not hasattr(compact, "solve")
    assert not hasattr(compact, "simulate")
    assert "_whole_atom_probabilities" not in compact.__all__


def test_authority_binds_superseding_tee_and_keeps_explicit_integration_hold() -> None:
    authority = compact.synthetic_app_authority_manifest()
    manifest = authority["manifest"]
    assert isinstance(manifest, dict)
    bindings = {
        row["descriptor"]: row["source_sha256"]
        for row in manifest["source_bindings"]
    }
    assert bindings == {
        "archive/hole_sim_transition_mean_anchor.py": (
            "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28"
        ),
        "archive/hole_sim_donor_ability_policy.py": (
            "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
        ),
        "archive/hole_sim_tee_app_complete_atom_support.py": (
            "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498"
        ),
        "archive/hole_sim_compact_physical_support_selector_v1.py": (
            "a5043c1fe1d32331ab8484d4faf99bb06a30c4c7ade1d0e5ed52b31ecbd3aff6"
        ),
    }
    assert manifest["supported_categories"] == ["app"]
    assert manifest["weighting_selector_contract"].endswith("selector_v2")
    assert manifest["selection_boundary_contract"].endswith("pathwise_v2")
    assert (
        manifest["tilted_probability_policy"]
        == "unchanged_primitive_binary64_atom_vector_v1"
    )
    assert (
        manifest["tilted_oracle"]
        == "independent_direct_flat_probability_fraction_oracle_v1"
    )
    assert manifest["terminal_atoms_allowed"] is True
    assert manifest["multiple_terminal_donors_required"] is True
    assert manifest["candidate_derived_digests_accepted"] is False
    assert manifest["real_chain_ready"] is False
    assert manifest["integration_ready"] is False
    assert compact.INTEGRATION_STATUS.endswith("integration_hold")

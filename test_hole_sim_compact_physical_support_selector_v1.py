from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_compact_physical_support_selector_v1 as compact
import hole_sim_donor_ability_policy as donor_policy
import hole_sim_transition_mean_anchor as anchor


REWARD_REFERENCE_SHA256 = (
    "a70ebc4aa3ec0dab67eb4d69722e48ec3f1fd6ec390643b53f21a63d6c45219b"
)
BRIDGES = tuple(f"compact-{index:02d}" for index in range(9))


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
    availability = (False, True, False, True, False, True, True, False, False)
    raw = (None, 0.4, None, 3.25, None, -0.6, -2.75, None, None)
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
    )
    donor_event = ("event-a",) * 5 + ("event-b",) * 4
    donor_start = ("2024-01-11T00:00:00Z",) * 5 + (
        "2024-01-18T00:00:00Z",
    ) * 4
    frame = pd.DataFrame(
        {
            "bridge_row_id": BRIDGES,
            "category": ("ott",) * 9,
            "donor_player_id": tuple(f"donor-{index:02d}" for index in range(9)),
            "donor_event_key": donor_event,
            "donor_event_start": donor_start,
            "target_event_key": ("synthetic-target",) * 9,
            "target_event_start": ("2024-03-07T00:00:00Z",) * 9,
            "donor_skill_date": skill_date,
            "donor_ability_source_column": ("sg_ott",) * 9,
            "donor_ability_available": availability,
            "donor_ability_raw_strict_prior": raw,
        }
    )
    applied = donor_policy.apply_frozen_policy(frame)
    return applied.set_index("bridge_row_id", drop=False).loc[list(BRIDGES)].reset_index(
        drop=True
    )


def _support() -> compact.CompactPhysicalSupport:
    policy = _policy_rows()
    first_proposal_mass = float(np.nextafter(0.4, 1.0))
    base_mass = (
        0.0,
        0.04,
        0.06,
        first_proposal_mass - 0.10,
        0.15,
        0.15,
        0.05,
        0.15,
        1.0 - first_proposal_mass - 0.50,
    )
    return compact.CompactPhysicalSupport(
        support_key="synthetic-compact|support=000",
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
        atom_id=tuple(f"atom-{index:02d}" for index in range(9)),
        base_mass=base_mass,
        fixed_reference_reward=(-0.2, -1.2, 0.8, 1.5, -0.6, 1.1, 0.4, -1.4, 2.0),
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
        original_donor_group_id=("group-a",) * 5 + ("group-b",) * 4,
        proposal_order=(0, 0, 0, 0, 1, 1, 1, 1, 2),
        surface_order=(0, 1, 1, 2, 0, 0, 1, 1, 0),
        hazard_order=(0, 0, 1, 0, 0, 1, 0, 1, 0),
        frozen_proposal_cdf_upper=(0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9, 0.9, 1.0),
        frozen_surface_cdf_upper=(0.0, 0.25, 0.25, 1.0, 0.6, 0.6, 1.0, 1.0, 1.0),
        frozen_hazard_cdf_upper=(1.0, 0.4, 1.0, 1.0, 0.5, 1.0, 0.25, 1.0, 1.0),
        boundary_only=(True,) + (False,) * 8,
        complete_atom=(True,) * 9,
        terminal=(False,) * 9,
        stroke_cost=(1, 1, 2, 1, 1, 2, 1, 1, 1),
        next_state_key=(
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


def _zero_parameters() -> compact.WeightingParameters:
    return compact.WeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.8, 0.2, -0.1, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )


def _quality_parameters() -> compact.WeightingParameters:
    return compact.WeightingParameters(
        transfer_beta=(0.9, 0.0, 0.0, 0.0),
        target_ability_sg=(0.8, 0.2, -0.1, 0.0),
        lambda_correction=(0.15, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )


def _admit(
    support: compact.CompactPhysicalSupport | None = None,
) -> compact.CompactWeightingSelectorSession:
    return compact.admit_synthetic_support(
        _support() if support is None else support,
        authority_manifest=compact.synthetic_authority_manifest(),
    )


def _replace_sequence(
    support: compact.CompactPhysicalSupport, name: str, values: tuple[object, ...]
) -> compact.CompactPhysicalSupport:
    return replace(support, **{name: values})


def _forged_session(
    support: compact.CompactPhysicalSupport,
    authority: object,
) -> compact.CompactWeightingSelectorSession:
    admitted = _admit()
    forged = object.__new__(compact.CompactWeightingSelectorSession)
    forged._support = support
    forged._authority = authority
    forged._readiness = admitted.readiness
    return forged


def test_closed_fixture_admits_mixed_policy_states_without_splitting_groups() -> None:
    session = _admit()
    readiness = session.readiness
    support = session.support_copy()
    assert readiness.rows == 9
    assert readiness.proposals == 3
    assert readiness.original_donor_groups == 2
    assert readiness.mixed_policy_state_groups == 2
    assert readiness.observed_rows == 4
    assert readiness.neutral_imputed_rows == 5
    assert readiness.availability_preserved
    assert readiness.positive_zero_preserved
    assert readiness.synthetic_fixture_only
    assert readiness.weighting_ready and readiness.selector_ready
    assert not readiness.real_chain_ready
    assert not readiness.integration_ready
    assert len(readiness.blockers) == 4
    assert support.original_donor_group_id == ("group-a",) * 5 + ("group-b",) * 4
    imputed = np.asarray(support.donor_ability_value_kind) == donor_policy.IMPUTED_VALUE_KIND
    ability = np.asarray(support.donor_ability_sg, dtype=float)
    available = np.asarray(support.donor_ability_available, dtype=bool)
    assert not available[imputed].any()
    assert np.equal(ability[imputed], 0.0).all()
    assert not np.signbit(ability[imputed]).any()


def test_exact_positive_zero_and_original_groups_reach_private_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _admit()
    support = session.support_copy()
    captured: list[tuple[np.ndarray, tuple[object, ...]]] = []
    original = anchor.whole_atom_probabilities

    def spy(**kwargs: object) -> anchor.AtomMassResult:
        captured.append(
            (
                np.asarray(kwargs["donor_ability_sg"], dtype=float).copy(),
                tuple(kwargs["donor_group_id"]),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(compact, "_whole_atom_probabilities", spy)
    weighted = session.weight(_quality_parameters())
    assert len(captured) == 1
    ability, groups = captured[0]
    imputed = np.asarray(support.donor_ability_value_kind) == donor_policy.IMPUTED_VALUE_KIND
    assert np.equal(ability[imputed], 0.0).all()
    assert not np.signbit(ability[imputed]).any()
    assert groups == support.original_donor_group_id
    assert np.allclose(
        weighted.donor_group_mass_before,
        weighted.donor_group_mass_after,
        atol=2e-15,
        rtol=2e-15,
    )


def test_nonzero_probabilities_factor_exactly_and_flat_selection_matches_hierarchy() -> None:
    session = _admit()
    weighted = session.weight(_quality_parameters())
    probability = np.asarray(weighted.probability)
    factorized = (
        np.asarray(weighted.proposal_probability)
        * np.asarray(weighted.surface_conditional_probability)
        * np.asarray(weighted.hazard_conditional_probability)
    )
    assert weighted.factorization_from_final_probability
    assert weighted.selection_mode == "weighted_rosenblatt_from_final_flat_probability_v1"
    assert np.allclose(factorized, probability, atol=2e-15, rtol=2e-15)
    assert not np.allclose(probability, np.asarray(_support().base_mass), atol=1e-10, rtol=0.0)
    for proposal_u, surface_u, hazard_u in (
        (0.0, 0.0, 0.0),
        (0.17, 0.51, 0.23),
        (0.41, 0.60, 0.50),
        (0.73, np.nextafter(0.5, 1.0), 0.99),
        (np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0), 0.5),
    ):
        result = session.select(
            _quality_parameters(),
            compact.SemanticUniforms(proposal_u, surface_u, hazard_u),
        )
        assert result.flat_hierarchical_equal
        assert result.flat_atom_id == result.hierarchical_atom_id == result.atom_id


def test_leading_zero_surface_boundary_replays_only_on_frozen_zero_path() -> None:
    session = _admit()
    support = session.support_copy()
    assert support.base_mass[0] == 0.0
    assert support.boundary_only[0] is True
    assert support.surface_order[0] == 0
    assert support.frozen_surface_cdf_upper[0] == 0.0

    exact = session.select(
        _zero_parameters(), compact.SemanticUniforms(0.2, 0.0, 0.0)
    )
    above = session.select(
        _zero_parameters(),
        compact.SemanticUniforms(0.2, np.nextafter(0.0, 1.0), 0.0),
    )
    assert exact.boundary_only is True
    assert exact.surface_order == 0
    assert above.boundary_only is False
    assert above.surface_order == 1
    assert exact.flat_hierarchical_equal and above.flat_hierarchical_equal

    tilted = session.weight(_quality_parameters())
    assert tilted.probability[0] == 0.0
    assert tilted.surface_conditional_probability[0] == 0.0
    tilted_zero = session.select(
        _quality_parameters(), compact.SemanticUniforms(0.2, 0.0, 0.0)
    )
    assert tilted_zero.boundary_only is False
    assert tilted_zero.surface_order == 1
    assert tilted_zero.flat_hierarchical_equal


def test_beta_lambda_zero_replays_frozen_proposal_boundary_and_nextafter() -> None:
    session = _admit()
    weighted = session.weight(_zero_parameters())
    support = session.support_copy()
    assert weighted.selection_mode == compact.ZERO_FAST_PATH_CONTRACT
    assert not weighted.factorization_from_final_probability
    assert weighted.probability[3] > 0.3
    assert support.frozen_proposal_cdf_upper[0] == 0.4

    def selected(uniform: float) -> compact.SelectionResult:
        return session.select(
            _zero_parameters(), compact.SemanticUniforms(uniform, 0.0, 0.0)
        )

    exact = selected(0.4)
    below = selected(float(np.nextafter(0.4, 0.0)))
    above = selected(float(np.nextafter(0.4, 1.0)))
    assert exact.proposal_order == 1
    assert below.proposal_order == 0
    assert above.proposal_order == 1
    assert exact.flat_hierarchical_equal
    assert below.flat_hierarchical_equal
    assert above.flat_hierarchical_equal


def test_beta_lambda_zero_preserves_surface_and_hazard_parent_boundaries() -> None:
    session = _admit()
    # Proposal 1 is selected by u=0.5.  Its surface boundary is exactly 0.6,
    # where the parent uses strict `u > cdf` and therefore retains surface 0.
    def surface_selected(uniform: float) -> compact.SelectionResult:
        return session.select(
            _zero_parameters(), compact.SemanticUniforms(0.5, uniform, 0.0)
        )

    surface_exact = surface_selected(0.6)
    surface_below = surface_selected(float(np.nextafter(0.6, 0.0)))
    surface_above = surface_selected(float(np.nextafter(0.6, 1.0)))
    assert surface_exact.surface_order == 0
    assert surface_below.surface_order == 0
    assert surface_above.surface_order == 1

    # Hazard selection retains right-CDF semantics: exact boundary advances.
    def hazard_selected(uniform: float) -> compact.SelectionResult:
        return session.select(
            _zero_parameters(), compact.SemanticUniforms(0.5, 0.2, uniform)
        )

    hazard_exact = hazard_selected(0.5)
    hazard_below = hazard_selected(float(np.nextafter(0.5, 0.0)))
    hazard_above = hazard_selected(float(np.nextafter(0.5, 1.0)))
    assert hazard_exact.hazard_order == 1
    assert hazard_below.hazard_order == 0
    assert hazard_above.hazard_order == 1
    assert all(
        result.flat_hierarchical_equal
        for result in (
            surface_exact,
            surface_below,
            surface_above,
            hazard_exact,
            hazard_below,
            hazard_above,
        )
    )


def test_honest_policy_clipping_is_admitted_but_forged_clipping_is_not() -> None:
    support = _admit().support_copy()
    raw = np.asarray(
        [np.nan if value is None else value for value in support.donor_ability_raw_strict_prior]
    )
    clipped = np.asarray(support.donor_ability_was_clipped, dtype=bool)
    applied = np.asarray(support.donor_ability_sg, dtype=float)
    assert applied[int(np.nanargmax(raw))] == 2.0
    assert applied[int(np.nanargmin(raw))] == -2.0
    assert clipped[int(np.nanargmax(raw))]
    assert clipped[int(np.nanargmin(raw))]

    forged_flags = list(support.donor_ability_was_clipped)
    forged_flags[int(np.nanargmax(raw))] = False
    forged = _replace_sequence(
        support, "donor_ability_was_clipped", tuple(forged_flags)
    )
    with pytest.raises(compact.CompactSupportUnavailable, match="attestation failed"):
        _admit(forged)


@pytest.mark.parametrize("attack", ["availability", "negative_zero", "split_group"])
def test_imputed_availability_zero_and_original_group_regressions(attack: str) -> None:
    support = _admit().support_copy()
    imputed_index = support.donor_ability_value_kind.index(
        donor_policy.IMPUTED_VALUE_KIND
    )
    if attack == "availability":
        values = list(support.donor_ability_available)
        values[imputed_index] = True
        forged = _replace_sequence(support, "donor_ability_available", tuple(values))
        match = "attestation failed|cannot become available"
    elif attack == "negative_zero":
        values = list(support.donor_ability_sg)
        values[imputed_index] = -0.0
        forged = _replace_sequence(support, "donor_ability_sg", tuple(values))
        match = "positive zero"
    else:
        values = list(support.original_donor_group_id)
        for index, kind in enumerate(support.donor_ability_value_kind):
            if kind == donor_policy.IMPUTED_VALUE_KIND:
                values[index] = f"{values[index]}|imputed"
        forged = _replace_sequence(support, "original_donor_group_id", tuple(values))
        match = "outside the closed synthetic authority"
    with pytest.raises(compact.CompactSupportUnavailable, match=match):
        _admit(forged)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("complete_atom", False, "complete atom"),
        ("stroke_cost", 0, "positive stroke cost"),
        ("next_state_key", "", "nonterminal atoms need a next state"),
    ],
)
def test_complete_physical_terminal_cost_and_next_state_regressions(
    field: str, replacement: object, message: str
) -> None:
    support = _admit().support_copy()
    values = list(getattr(support, field))
    values[-1] = replacement
    forged = _replace_sequence(support, field, tuple(values))
    with pytest.raises(compact.CompactSupportUnavailable, match=message):
        _admit(forged)


def test_boundary_flag_is_bound_and_app_support_fails_closed() -> None:
    support = _admit().support_copy()
    boundary = list(support.boundary_only)
    boundary[0] = False
    forged = replace(support, boundary_only=tuple(boundary))
    assert compact._frozen_cdf_sha256(forged) != compact._frozen_cdf_sha256(support)
    with pytest.raises(
        compact.CompactSupportUnavailable,
        match="boundary-only zeros|Base mass",
    ):
        _admit(forged)

    app = replace(support, category=("app",) * len(support.category))
    with pytest.raises(compact.CompactSupportUnavailable, match="TEE-only"):
        _admit(app)


def test_nonterminal_empty_next_state_is_rejected() -> None:
    support = _admit().support_copy()
    next_state = list(support.next_state_key)
    next_state[0] = ""
    with pytest.raises(compact.CompactSupportUnavailable, match="Nonterminal|nonterminal"):
        _admit(_replace_sequence(support, "next_state_key", tuple(next_state)))


def test_terminal_tee_atom_is_rejected_even_when_terminal_state_is_consistent() -> None:
    session = _admit()
    selected = session.select(
        _zero_parameters(), compact.SemanticUniforms(0.95, 0.5, 0.5)
    )
    assert selected.atom_id == "atom-08"
    assert selected.terminal is False
    assert selected.next_state_key == "state-h"

    support = session.support_copy()
    terminal = list(support.terminal)
    next_state = list(support.next_state_key)
    terminal[-1] = True
    next_state[-1] = ""
    forged = replace(
        support,
        terminal=tuple(terminal),
        next_state_key=tuple(next_state),
    )
    with pytest.raises(
        compact.CompactSupportUnavailable,
        match="TEE compact support cannot contain terminal atoms",
    ):
        _admit(forged)


def test_atom_and_hierarchy_caps_fail_before_candidate_digest_authority() -> None:
    support = _admit().support_copy()
    additions = compact._MAX_ATOMS + 1 - len(support.atom_id)
    updates: dict[str, tuple[object, ...]] = {}
    for field in fields(compact.CompactPhysicalSupport):
        if field.name in {"support_key", "reward_reference_sha256"}:
            continue
        sequence = tuple(getattr(support, field.name))
        updates[field.name] = sequence + (sequence[-1],) * additions
    oversized = replace(support, **updates)
    with pytest.raises(compact.CompactSupportUnavailable, match="atom cap exceeded"):
        _admit(oversized)

    proposals = tuple(range(len(support.proposal_order)))
    too_many_proposals = replace(support, proposal_order=proposals)
    with pytest.raises(compact.CompactSupportUnavailable, match="Proposal cap exceeded"):
        _admit(too_many_proposals)


def test_log_cap_mismatch_and_weight_clipping_fail_closed() -> None:
    session = _admit()
    wrong_cap = replace(_quality_parameters(), max_abs_log_weight=31.0)
    with pytest.raises(compact.CompactSupportUnavailable, match="log cap differs"):
        session.weight(wrong_cap)
    clipped = replace(
        _quality_parameters(),
        transfer_beta=(100.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(
        compact.CompactSupportUnavailable,
        match="clipping is not admissible|weighting rejected",
    ):
        session.weight(clipped)


def test_positive_node_cdf_rounding_collapse_fails_closed() -> None:
    # No log is clipped: the OTT target logs span [-22, 15.4].  The tiny
    # positive final proposal nevertheless rounds behind a cumulative 1.0 and
    # therefore cannot be represented by a distinct binary64 CDF interval.
    parameters = compact.WeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(-11.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )
    with pytest.raises(
        compact.CompactSupportUnavailable,
        match="Positive proposal mass collapsed to a zero-width CDF interval",
    ):
        _admit().weight(parameters)


def test_frozen_cdf_mutation_and_resealed_self_authority_are_rejected() -> None:
    support = _admit().support_copy()
    cdf = list(support.frozen_proposal_cdf_upper)
    cdf[:4] = [0.41, 0.41, 0.41, 0.41]
    changed = replace(
        support,
        frozen_proposal_cdf_upper=tuple(cdf),
        base_mass=(0.0, 0.041, 0.0615, 0.3075, 0.147, 0.147, 0.049, 0.147, 0.10),
    )
    # The changed CDF is structurally valid.  Even a caller-created adjacent
    # authority carrying freshly recomputed support/CDF/manifest hashes cannot
    # replace the source-pinned approval lock.
    canonical = compact._canonicalize_support(changed)
    forged_authority = deepcopy(compact.synthetic_authority_manifest())
    manifest = forged_authority["manifest"]
    assert isinstance(manifest, dict)
    manifest["support_sha256"] = compact._support_sha256(canonical)
    manifest["frozen_cdf_sha256"] = compact._frozen_cdf_sha256(canonical)
    forged_authority["manifest_sha256"] = _stable_sha(manifest)
    with pytest.raises(compact.CompactSupportUnavailable, match="closed source-pinned"):
        compact.admit_synthetic_support(
            changed, authority_manifest=forged_authority
        )


def test_invalid_frozen_cdf_is_rejected_before_digest_check() -> None:
    support = _admit().support_copy()
    values = list(support.frozen_hazard_cdf_upper)
    values[1] = 1.0
    invalid = _replace_sequence(support, "frozen_hazard_cdf_upper", tuple(values))
    with pytest.raises(compact.CompactSupportUnavailable, match="ordered CDF"):
        _admit(invalid)


def test_wrong_authority_and_runtime_source_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = compact.synthetic_authority_manifest()
    forged = deepcopy(authority)
    manifest = forged["manifest"]
    assert isinstance(manifest, dict)
    manifest["production_enabled"] = True
    forged["manifest_sha256"] = _stable_sha(manifest)
    with pytest.raises(compact.CompactSupportUnavailable, match="closed source-pinned"):
        compact.admit_synthetic_support(_support(), authority_manifest=forged)

    original = compact._sha256_file

    def wrong_anchor(path: Path) -> str:
        if Path(path).name == "hole_sim_transition_mean_anchor.py":
            return "f" * 64
        return original(path)

    monkeypatch.setattr(compact, "_sha256_file", wrong_anchor)
    with pytest.raises(compact.CompactSupportUnavailable, match="source binding changed"):
        compact.admit_synthetic_support(
            _support(), authority_manifest=authority
        )


def test_private_session_cannot_be_forged_with_candidate_derived_authority() -> None:
    admitted = _admit()
    support = admitted.support_copy()
    groups = list(support.original_donor_group_id)
    groups[0] = "caller-resealed-group"
    changed = replace(support, original_donor_group_id=tuple(groups))
    canonical = compact._canonicalize_support(changed)
    forged_authority = compact._TrustedSyntheticAuthority(
        manifest_sha256=admitted.readiness.authority_manifest_sha256,
        runtime_binding_sha256=admitted.readiness.runtime_binding_sha256,
        support_sha256=compact._support_sha256(canonical),
        frozen_cdf_sha256=compact._frozen_cdf_sha256(canonical),
        reward_reference_sha256=canonical.reward_reference_sha256,
    )
    forged_session = object.__new__(compact.CompactWeightingSelectorSession)
    forged_session._support = canonical
    forged_session._authority = forged_authority
    forged_session._readiness = admitted.readiness
    with pytest.raises(compact.CompactSupportUnavailable, match="authority changed"):
        forged_session.weight(_zero_parameters())


def test_authority_subclass_equality_forgery_rejects_before_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _admit()
    support = admitted.support_copy()
    groups = list(support.original_donor_group_id)
    groups[0] = "hostile-resealed-boundary-group"
    changed = compact._canonicalize_support(
        replace(support, original_donor_group_id=tuple(groups))
    )
    equality_calls: list[object] = []

    class HostileAuthority(compact._TrustedSyntheticAuthority):
        def __eq__(self, other: object) -> bool:
            equality_calls.append(other)
            return True

    authority = HostileAuthority(
        manifest_sha256=admitted.readiness.authority_manifest_sha256,
        runtime_binding_sha256=admitted.readiness.runtime_binding_sha256,
        support_sha256=compact._support_sha256(changed),
        frozen_cdf_sha256=compact._frozen_cdf_sha256(changed),
        reward_reference_sha256=changed.reward_reference_sha256,
    )
    primitive_calls: list[bool] = []

    def forbidden_primitive(**_: object) -> object:
        primitive_calls.append(True)
        raise AssertionError("weighting primitive must not run")

    monkeypatch.setattr(compact, "_whole_atom_probabilities", forbidden_primitive)
    with pytest.raises(compact.CompactSupportUnavailable, match="authority changed"):
        _forged_session(changed, authority).weight(_zero_parameters())
    assert equality_calls == []
    assert primitive_calls == []


def test_authority_str_subclass_field_rejects_without_dispatched_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _admit()
    closed = admitted._authority
    equality_calls: list[object] = []

    class HostileStr(str):
        def __eq__(self, other: object) -> bool:
            equality_calls.append(other)
            return True

    forged = compact._TrustedSyntheticAuthority(
        manifest_sha256=HostileStr(closed.manifest_sha256),
        runtime_binding_sha256=closed.runtime_binding_sha256,
        support_sha256=closed.support_sha256,
        frozen_cdf_sha256=closed.frozen_cdf_sha256,
        reward_reference_sha256=closed.reward_reference_sha256,
    )
    primitive_calls: list[bool] = []

    def forbidden_primitive(**_: object) -> object:
        primitive_calls.append(True)
        raise AssertionError("weighting primitive must not run")

    monkeypatch.setattr(compact, "_whole_atom_probabilities", forbidden_primitive)
    with pytest.raises(compact.CompactSupportUnavailable, match="authority changed"):
        _forged_session(admitted.support_copy(), forged).weight(_zero_parameters())
    assert equality_calls == []
    assert primitive_calls == []


def test_parameter_and_uniform_subclasses_reject_before_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapBypass(compact.WeightingParameters):
        def _anchor_parameters(self) -> anchor.TransitionQualityParameters:
            return anchor.TransitionQualityParameters(
                transfer_beta=self.transfer_beta,
                target_ability_sg=self.target_ability_sg,
                lambda_correction=self.lambda_correction,
                reward_reference_sha256=self.reward_reference_sha256,
                max_abs_log_weight=100.0,
            )

    class DivergentUniforms(compact.SemanticUniforms):
        def validated(self) -> tuple[float, float, float]:
            return (0.99, 0.99, 0.99)

    bypass = CapBypass(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REWARD_REFERENCE_SHA256,
    )
    primitive_calls: list[bool] = []

    def forbidden_primitive(**_: object) -> object:
        primitive_calls.append(True)
        raise AssertionError("weighting primitive must not run")

    monkeypatch.setattr(compact, "_whole_atom_probabilities", forbidden_primitive)
    session = _admit()
    with pytest.raises(compact.CompactSupportUnavailable, match="wrong exact type"):
        session.weight(bypass)
    with pytest.raises(compact.CompactSupportUnavailable, match="wrong exact type"):
        session.select(_zero_parameters(), DivergentUniforms(0.0, 0.0, 0.0))
    assert primitive_calls == []

    monkeypatch.setattr(compact, "_whole_atom_probabilities", anchor.whole_atom_probabilities)
    honest = session.select(
        _zero_parameters(), compact.SemanticUniforms(np.float64(0.2), 0.0, 0.0)
    )
    assert type(honest.semantic_uniforms) is compact.SemanticUniforms
    assert honest.semantic_uniforms == compact.SemanticUniforms(0.2, 0.0, 0.0)


def test_session_subclass_cannot_override_live_support_before_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted = _admit()

    class HostileSession(compact.CompactWeightingSelectorSession):
        def _live_support(self) -> compact.CompactPhysicalSupport:
            return admitted.support_copy()

    forged = object.__new__(HostileSession)
    forged._support = admitted.support_copy()
    forged._authority = admitted._authority
    forged._readiness = admitted.readiness
    primitive_calls: list[bool] = []

    def forbidden_primitive(**_: object) -> object:
        primitive_calls.append(True)
        raise AssertionError("weighting primitive must not run")

    monkeypatch.setattr(compact, "_whole_atom_probabilities", forbidden_primitive)
    with pytest.raises(compact.CompactSupportUnavailable, match="session.*exact type"):
        forged.weight(_zero_parameters())
    assert primitive_calls == []


def test_direct_session_and_public_raw_weight_or_solver_routes_are_absent() -> None:
    with pytest.raises(compact.CompactSupportUnavailable, match="minted only"):
        compact.CompactWeightingSelectorSession()
    assert not hasattr(compact, "anchor")
    assert not hasattr(compact, "whole_atom_probabilities")
    assert not hasattr(compact, "solve_mean_anchor")
    assert not hasattr(compact, "solve")
    assert not hasattr(compact, "simulate")
    assert "_whole_atom_probabilities" not in compact.__all__


def test_uniforms_and_reward_reference_fail_closed() -> None:
    session = _admit()
    for value in (np.nextafter(0.0, -1.0), 1.0, np.nan):
        with pytest.raises(compact.CompactSupportUnavailable, match="uniform"):
            session.select(
                _zero_parameters(), compact.SemanticUniforms(value, 0.0, 0.0)
            )
    wrong_reward = replace(_zero_parameters(), reward_reference_sha256="f" * 64)
    with pytest.raises(compact.CompactSupportUnavailable, match="reward reference"):
        session.weight(wrong_reward)


def test_authority_binds_all_three_sources_and_keeps_integration_hold() -> None:
    authority = compact.synthetic_authority_manifest()
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
    }
    assert manifest["scope"] == "exact_source_pinned_synthetic_fixture_only"
    assert manifest["supported_categories"] == ["ott"]
    assert manifest["tee_terminal_contract"] == "all_complete_atoms_nonterminal_v1"
    assert manifest["tee_terminal_atoms_allowed"] is False
    assert manifest["app_selection_ready"] is False
    assert manifest["real_chain_ready"] is False
    assert manifest["integration_ready"] is False
    assert compact.INTEGRATION_STATUS.endswith("integration_hold")

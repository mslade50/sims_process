from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_compact_app_whole_atom_selector_v1 as synthetic
import hole_sim_donor_ability_policy as policy
import hole_sim_real_app_whole_atom_selector_lowry_r1h8 as real
import hole_sim_transition_mean_chain_adapter as chain_v3


REFERENCE_SHA256 = "a" * 64
PUTTING_SHA256 = "b" * 64


class _IntImpostor(int):
    pass


class _FloatImpostor(float):
    pass


class _StringImpostor(str):
    pass


class _HostileReference(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _policy_rows() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "bridge_row_id": ("bridge-p0", "bridge-p1", "bridge-terminal"),
            "category": ("app", "app", "app"),
            "donor_player_id": ("donor-observed", "donor-imputed", "donor-terminal"),
            "donor_event_key": ("event-a", "event-a", "event-b"),
            "donor_event_start": (
                "2024-01-11T00:00:00Z",
                "2024-01-11T00:00:00Z",
                "2024-01-18T00:00:00Z",
            ),
            "target_event_key": ("target", "target", "target"),
            "target_event_start": (
                "2024-03-07T00:00:00Z",
                "2024-03-07T00:00:00Z",
                "2024-03-07T00:00:00Z",
            ),
            "donor_skill_date": (
                "2024-01-01T00:00:00Z",
                None,
                "2024-01-02T00:00:00Z",
            ),
            "donor_ability_source_column": ("sg_app", "sg_app", "sg_app"),
            "donor_ability_available": (True, False, True),
            "donor_ability_raw_strict_prior": (0.6, None, -0.4),
        }
    )
    for column in (
        "donor_event_start",
        "target_event_start",
        "donor_skill_date",
    ):
        raw[column] = pd.to_datetime(raw[column], utc=True)
    return policy.apply_frozen_policy(raw).set_index("bridge_row_id", drop=False)


def _fixture() -> pd.DataFrame:
    policies = _policy_rows()
    full_probability = np.asarray(
        [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=float
    )
    full_cdf = np.cumsum(full_probability)
    probability_json = json.dumps(
        full_probability.tolist(), separators=(",", ":"), allow_nan=False
    )
    cdf_json = json.dumps(full_cdf.tolist(), separators=(",", ":"), allow_nan=False)
    rows: list[dict[str, object]] = []
    rewards = {
        0: (-1.2, -0.3, 0.7, 1.5),
        1: (-1.8, -0.6, 0.4, 1.1),
    }
    proposal_mass = {0: 0.5, 1: 0.3}
    proposal_cdf = {0: (0.0, 0.625), 1: (0.625, 1.0)}
    policy_key = {0: "bridge-p0", 1: "bridge-p1"}
    surface_definitions = (
        (0, "green", 0.5, ((-1, "not_applicable", "none", 1.0),)),
        (1, "fairway", 0.25, ((-1, "not_applicable", "none", 1.0),)),
        (
            7,
            "penalty",
            0.25,
            (
                (0, "hazard-a", "resolved_complete_atom", 0.4),
                (1, "hazard-b", "resolved_complete_atom", 0.6),
            ),
        ),
    )
    for proposal in (0, 1):
        policy_row = policies.loc[policy_key[proposal]]
        reward_index = 0
        for surface, label, surface_probability, hazard_rows in surface_definitions:
            surface_lower = 0.0 if surface == 0 else float(full_cdf[surface - 1])
            surface_upper = float(full_cdf[surface])
            hazard_running = 0.0
            for hazard_order, hazard_key, status, hazard_probability in hazard_rows:
                hazard_lower = hazard_running
                hazard_running = float(hazard_running + hazard_probability)
                mechanics = float(surface_probability * hazard_probability)
                row: dict[str, object] = {
                    "query_key": "fixture-lowry-r1h8-app",
                    "complete_row_order": len(rows),
                    "atom_id": f"p{proposal}-s{surface}-h{hazard_order}",
                    "category": "app",
                    "complete_atom": True,
                    "boundary_only": False,
                    "proposal_row_order": proposal,
                    "proposal_probability": proposal_mass[proposal],
                    "proposal_branch": "spatial",
                    "terminal": False,
                    "mechanics_conditional_probability": mechanics,
                    "quality_base_mass": proposal_mass[proposal] * mechanics,
                    "fixed_reference_reward": rewards[proposal][reward_index],
                    "start_state_key": "fairway|150.000000yd",
                    "declared_app_base_holeout_probability": 0.2,
                    "frozen_spatial_cdf_lower": proposal_cdf[proposal][0],
                    "frozen_spatial_cdf_upper": proposal_cdf[proposal][1],
                    "frozen_terminal_residual_cdf_lower": np.nan,
                    "frozen_terminal_residual_cdf_upper": np.nan,
                    "surface_class_count": 9,
                    "surface_full_probability_json": probability_json,
                    "surface_full_cdf_upper_json": cdf_json,
                    "surface_class_index": surface,
                    "surface_class_label": label,
                    "surface_probability": surface_probability,
                    "surface_cdf_lower": surface_lower,
                    "surface_cdf_upper": surface_upper,
                    "hazard_resolution_status": status,
                    "hazard_donor_order": hazard_order,
                    "hazard_donor_key": hazard_key,
                    "hazard_probability": hazard_probability,
                    "hazard_cdf_lower": hazard_lower,
                    "hazard_cdf_upper": hazard_running,
                    "stroke_cost": 2 if status == "resolved_complete_atom" else 1,
                    "end_state_key": f"state-p{proposal}-s{surface}-h{hazard_order}",
                    "next_support_key": f"next-p{proposal}-s{surface}-h{hazard_order}",
                    "reference_model_sha256": REFERENCE_SHA256,
                    "reference_putting_sha256": PUTTING_SHA256,
                }
                for column in (*policy.INPUT_COLUMNS, *policy.OUTPUT_COLUMNS):
                    row[column] = policy_row[column]
                row["donor_quality_group_id"] = chain_v3.canonical_donor_quality_group_id(
                    row
                )
                rows.append(row)
                reward_index += 1
    terminal_policy = policies.loc["bridge-terminal"]
    terminal: dict[str, object] = {
        "query_key": "fixture-lowry-r1h8-app",
        "complete_row_order": len(rows),
        "atom_id": "terminal-a",
        "category": "app",
        "complete_atom": True,
        "boundary_only": False,
        "proposal_row_order": 2,
        "proposal_probability": 0.2,
        "proposal_branch": "terminal",
        "terminal": True,
        "mechanics_conditional_probability": 1.0,
        "quality_base_mass": 0.2,
        "fixed_reference_reward": 2.2,
        "start_state_key": "fairway|150.000000yd",
        "declared_app_base_holeout_probability": 0.2,
        "frozen_spatial_cdf_lower": np.nan,
        "frozen_spatial_cdf_upper": np.nan,
        "frozen_terminal_residual_cdf_lower": 0.0,
        "frozen_terminal_residual_cdf_upper": 1.0,
        "surface_class_count": 0,
        "surface_full_probability_json": "[]",
        "surface_full_cdf_upper_json": "[]",
        "surface_class_index": -1,
        "surface_class_label": "explicit_app_holeout",
        "surface_probability": 1.0,
        "surface_cdf_lower": 0.0,
        "surface_cdf_upper": 1.0,
        "hazard_resolution_status": "none",
        "hazard_donor_order": -1,
        "hazard_donor_key": "not_applicable",
        "hazard_probability": 1.0,
        "hazard_cdf_lower": 0.0,
        "hazard_cdf_upper": 1.0,
        "stroke_cost": 1,
        "end_state_key": "terminal",
        "next_support_key": "",
        "reference_model_sha256": REFERENCE_SHA256,
        "reference_putting_sha256": PUTTING_SHA256,
    }
    for column in (*policy.INPUT_COLUMNS, *policy.OUTPUT_COLUMNS):
        terminal[column] = terminal_policy[column]
    terminal["donor_quality_group_id"] = chain_v3.canonical_donor_quality_group_id(
        terminal
    )
    rows.append(terminal)
    return pd.DataFrame(rows)


def _two_terminal_fixture() -> pd.DataFrame:
    frame = _fixture()
    first_index = int(frame.index[frame["terminal"]].item())
    frame.loc[first_index, "proposal_probability"] = 0.1
    frame.loc[first_index, "quality_base_mass"] = 0.1
    frame.loc[first_index, "frozen_terminal_residual_cdf_upper"] = 0.5
    second = frame.loc[first_index].copy()
    second["complete_row_order"] = len(frame)
    second["atom_id"] = "terminal-b"
    second["proposal_row_order"] = 3
    second["proposal_probability"] = 0.1
    second["quality_base_mass"] = 0.1
    second["fixed_reference_reward"] = -0.8
    second["frozen_terminal_residual_cdf_lower"] = 0.5
    second["frozen_terminal_residual_cdf_upper"] = 1.0
    return pd.concat([frame, second.to_frame().T], ignore_index=True)


def _proposal_cap_fixture(spatial_proposals: int) -> pd.DataFrame:
    seed = _fixture()
    spatial_template = seed.loc[seed["proposal_row_order"].eq(0)].copy(deep=True)
    terminal = seed.loc[seed["terminal"]].copy(deep=True)
    pieces: list[pd.DataFrame] = []
    for proposal in range(spatial_proposals):
        lower = float(proposal / spatial_proposals)
        upper = float((proposal + 1) / spatial_proposals)
        mass = float(0.8 * (upper - lower))
        rows = spatial_template.copy(deep=True)
        rows["proposal_row_order"] = proposal
        rows["proposal_probability"] = mass
        rows["frozen_spatial_cdf_lower"] = lower
        rows["frozen_spatial_cdf_upper"] = upper
        rows["quality_base_mass"] = (
            mass * rows["mechanics_conditional_probability"].to_numpy(float)
        )
        rows["atom_id"] = [
            f"cap-p{proposal}-s{surface}-h{hazard}"
            for surface, hazard in zip(
                rows["surface_class_index"], rows["hazard_donor_order"], strict=True
            )
        ]
        rows["end_state_key"] = [
            f"state-cap-p{proposal}-s{surface}-h{hazard}"
            for surface, hazard in zip(
                rows["surface_class_index"], rows["hazard_donor_order"], strict=True
            )
        ]
        rows["next_support_key"] = [
            f"next-cap-p{proposal}-s{surface}-h{hazard}"
            for surface, hazard in zip(
                rows["surface_class_index"], rows["hazard_donor_order"], strict=True
            )
        ]
        pieces.append(rows)
    terminal["proposal_row_order"] = spatial_proposals
    pieces.append(terminal)
    frame = pd.concat(pieces, ignore_index=True)
    frame["complete_row_order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _zero() -> real.RealAppWeightingParameters:
    return real.RealAppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.24, 0.74, 0.04, 0.09),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=REFERENCE_SHA256,
    )


def _tilted(*, app_only: bool = False) -> real.RealAppWeightingParameters:
    return real.RealAppWeightingParameters(
        transfer_beta=(0.0, 0.2, 0.0, 0.0) if app_only else (0.9, 0.2, 0.3, 0.0),
        target_ability_sg=(0.24, 0.74, 0.04, 0.09),
        lambda_correction=(0.0, 0.1, 0.0, 0.0),
        reward_reference_sha256=REFERENCE_SHA256,
    )


def test_empty_authority_fails_before_runtime_or_artifact_access(monkeypatch) -> None:
    manifest = real.real_app_authority_manifest()
    assert manifest["records"] == []
    assert manifest["real_support_admitted"] is False
    assert manifest["weighting_enabled"] is False
    assert manifest["selection_enabled"] is False
    assert manifest["production_enabled"] is False
    assert manifest["anchor_ready"] is False
    monkeypatch.setattr(
        real,
        "_validate_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not be read")),
    )
    with pytest.raises(real.RealAppSelectorUnavailable, match="No real Lowry"):
        real.admit_lowry_r1h8_app_support()


def test_runtime_is_exactly_source_pinned() -> None:
    observed = real._validate_runtime()
    assert len(observed) == 64
    assert real._runtime_payload()["semantic_uniforms"] == [
        "holeout",
        "proposal",
        "surface",
        "hazard",
    ]
    assert {item[2] for item in real._SOURCE_BINDINGS} == {
        "e0606e0818fdeb625f582b3379b109d50d9485e2866268930b68b08c4dd238e7",
        "55071ec977770897de18d0761d02fef32ce640343f760ad1f40e63eeb553be28",
        "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339",
        "597f068d68cee7580a964afc687b29a66ad3cfe29b8c2fa47b11172abedf6498",
        "ed12a40ad7a65117cf00d846997686a511d85b3cdf5537e3d5671b1464e96460",
        "84137c1bc80b198225796cee40ad6449810f72491d0194f2314e033e52734cff",
    }


def test_public_namespace_is_closed_to_declared_api() -> None:
    public = {name for name in vars(real) if not name.startswith("_")}
    assert public == set(real.__all__)


def test_real_shaped_fixture_weights_once_across_original_groups() -> None:
    frame = _fixture()
    state = real._weight_fixture(frame, _tilted())
    weighted = state.weighted
    assert weighted.zero_fast_path is False
    assert weighted.selection_mode == real.REAL_APP_TILTED_PATH_CONTRACT
    assert weighted.whole_atom_vector_preserved is True
    assert weighted.group_wide_weighting is True
    assert weighted.clipped_donor_logs == 0
    assert weighted.clipped_target_logs == 0
    assert sum(weighted.probability) == pytest.approx(1.0, abs=3e-13)
    assert np.allclose(
        weighted.donor_group_mass_before,
        weighted.donor_group_mass_after,
        atol=2e-15,
        rtol=2e-15,
    )
    assert sorted(weighted.donor_group_mass_before) == pytest.approx([0.2, 0.8])
    group_states = frame.groupby("donor_quality_group_id")[
        "donor_ability_value_kind"
    ].nunique()
    assert int((group_states > 1).sum()) == 1


def test_non_app_parameters_are_category_isolated_bit_for_bit() -> None:
    frame = _fixture()
    full = real._weight_fixture(frame, _tilted(app_only=False)).weighted
    app_only = real._weight_fixture(frame, _tilted(app_only=True)).weighted
    assert np.array_equal(full.probability, app_only.probability)
    assert np.array_equal(
        full.neutral_probability_mass, app_only.neutral_probability_mass
    )


def test_non_app_only_coefficients_retain_exact_frozen_zero_path() -> None:
    frame = _fixture()
    non_app_only = real.RealAppWeightingParameters(
        transfer_beta=(0.9, 0.0, 0.3, 0.2),
        target_ability_sg=(0.24, 0.74, 0.04, 0.09),
        lambda_correction=(0.4, 0.0, 0.1, 0.7),
        reward_reference_sha256=REFERENCE_SHA256,
    )
    zero = real._weight_fixture(frame, _zero())
    isolated = real._weight_fixture(frame, non_app_only)
    assert isolated.weighted.zero_fast_path is True
    assert isolated.weighted.selection_mode == real.REAL_APP_ZERO_PATH_CONTRACT
    assert np.array_equal(
        isolated.weighted.probability, zero.weighted.probability
    )
    assert isolated.hierarchy == zero.hierarchy
    uniforms = real.RealAppSemanticUniforms(
        0.2, 0.625, float(np.nextafter(0.75, 1.0)), 0.4
    )
    isolated_pick = real._select_fixture(frame, non_app_only, uniforms)
    zero_pick = real._select_fixture(frame, _zero(), uniforms)
    assert isolated_pick.atom_id == zero_pick.atom_id
    assert isolated_pick.selection_mode == zero_pick.selection_mode


def test_reward_reference_rejects_hostile_string_subclass_before_equality() -> None:
    parameters = real.RealAppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.24, 0.74, 0.04, 0.09),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=_HostileReference("f" * 64),
    )
    with pytest.raises(real.RealAppSelectorUnavailable, match="exact lowercase SHA"):
        real._weight_fixture(_fixture(), parameters)

    uppercase = real.RealAppWeightingParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.24, 0.74, 0.04, 0.09),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256="A" * 64,
    )
    with pytest.raises(real.RealAppSelectorUnavailable, match="exact lowercase SHA"):
        real._weight_fixture(_fixture(), uppercase)


def test_zero_path_preserves_all_four_boundary_rules() -> None:
    frame = _fixture()
    zero_state = real._weight_fixture(frame, _zero())
    expected_base = frame["quality_base_mass"].to_numpy(float)
    expected_base = expected_base / expected_base.sum()
    assert zero_state.weighted.zero_fast_path is True
    assert zero_state.weighted.selection_mode == real.REAL_APP_ZERO_PATH_CONTRACT
    assert np.allclose(
        zero_state.weighted.probability,
        expected_base,
        atol=2e-15,
        rtol=2e-15,
    )
    terminal = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(0.1, 0.99, 0.99, 0.99),
    )
    assert terminal.atom_id == "terminal-a"
    assert terminal.terminal is True
    assert terminal.terminal_residual_uniform == 0.5

    # Holeout equality is spatial; proposal equality advances right.
    proposal_right = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(0.2, 0.625, 0.5, 0.0),
    )
    assert proposal_right.proposal_row_order == 1
    assert proposal_right.surface_class_index == 0

    # Surface equality stays left, then one ULP above advances.
    surface_equal = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(0.2, 0.1, 0.5, 0.0),
    )
    surface_after = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(
            0.2, 0.1, float(np.nextafter(0.5, 1.0)), 0.0
        ),
    )
    assert surface_equal.surface_class_index == 0
    assert surface_after.surface_class_index == 1

    # Repeated zero surface slots are retained: equality is class 1; one ULP
    # above jumps over zero classes 2..6 to original class 7.
    plateau_equal = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(0.2, 0.1, 0.75, 0.0),
    )
    plateau_after = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(
            0.2, 0.1, float(np.nextafter(0.75, 1.0)), 0.0
        ),
    )
    assert plateau_equal.surface_class_index == 1
    assert plateau_after.surface_class_index == 7

    # Hazard equality advances right.
    hazard_right = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(
            0.2, 0.1, float(np.nextafter(0.75, 1.0)), 0.4
        ),
    )
    assert hazard_right.atom_id == "p0-s7-h1"
    assert hazard_right.hazard_donor_order == 1
    assert hazard_right.flat_hierarchical_equal is True


def test_one_terminal_holeout_residual_reuses_same_strict_uniform() -> None:
    frame = _fixture()
    terminal_u = float(np.nextafter(0.2, 0.0))
    before = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(terminal_u, 0.0, 0.0, 0.0),
    )
    at = real._select_fixture(
        frame,
        _zero(),
        real.RealAppSemanticUniforms(0.2, 0.0, 0.0, 0.0),
    )
    assert before.atom_id == "terminal-a"
    assert before.terminal_residual_uniform == terminal_u / 0.2
    assert before.semantic_uniforms.holeout == terminal_u
    assert at.terminal is False
    assert at.terminal_residual_uniform is None
    assert at.semantic_uniforms.holeout == 0.2


def test_tilted_one_terminal_prefix_and_residual_are_trivially_equivalent() -> None:
    frame = _fixture()
    state = real._weight_fixture(frame, _tilted())
    assert state.weighted.zero_fast_path is False
    assert len(state.hierarchy.terminal_indices) == 1
    terminal_probability = state.weighted.terminal_probability
    assert state.hierarchy.terminal_unconditional_upper == (terminal_probability,)
    assert state.hierarchy.terminal_conditional.upper == (1.0,)
    assert 0.0 < terminal_probability < 1.0
    for holeout in (
        0.0,
        float(np.nextafter(terminal_probability, 0.0)),
    ):
        retained = real._uniforms(
            real.RealAppSemanticUniforms(holeout, 0.0, 0.0, 0.0)
        )
        hierarchical, residual = real._hierarchical_index(state, retained)
        direct = real._direct_index(state, retained)
        assert hierarchical == direct
        assert residual == holeout / terminal_probability
    at = real._select_fixture(
        frame,
        _tilted(),
        real.RealAppSemanticUniforms(
            terminal_probability, 0.0, 0.0, 0.0
        ),
    )
    assert at.terminal is False
    assert at.terminal_residual_uniform is None
    assert at.semantic_uniforms.holeout == terminal_probability


def test_multiple_terminal_atoms_are_outside_fixed_real_h8_contract() -> None:
    with pytest.raises(real.RealAppSelectorUnavailable, match="exactly one terminal"):
        real._canonicalize_fixture(_two_terminal_fixture())


def test_tilted_hierarchy_matches_independent_fraction_oracle_at_every_cut() -> None:
    state = real._weight_fixture(_fixture(), _tilted())
    probes: list[real.RealAppSemanticUniforms] = [
        real.RealAppSemanticUniforms(0.0, 0.0, 0.0, 0.0),
        real.RealAppSemanticUniforms(
            float(np.nextafter(state.weighted.terminal_probability, 1.0)),
            0.0,
            0.0,
            0.0,
        ),
        real.RealAppSemanticUniforms(
            state.weighted.terminal_probability, 0.0, 0.0, 0.0
        ),
    ]
    for cut in state.hierarchy.proposal.upper[:-1]:
        for value in (np.nextafter(cut, 0.0), cut, np.nextafter(cut, 1.0)):
            probes.append(
                real.RealAppSemanticUniforms(
                    state.weighted.terminal_probability, float(value), 0.0, 0.0
                )
            )
    proposal_lower = 0.0
    for proposal_slot, (proposal, node) in enumerate(state.hierarchy.surfaces):
        proposal_upper = state.hierarchy.proposal.upper[proposal_slot]
        proposal_u = (proposal_lower + proposal_upper) / 2.0
        proposal_lower = proposal_upper
        for cut in sorted(value for value in set(node.upper[:-1]) if 0.0 < value < 1.0):
            for value in (np.nextafter(cut, 0.0), cut, np.nextafter(cut, 1.0)):
                probes.append(
                    real.RealAppSemanticUniforms(
                        state.weighted.terminal_probability,
                        float(proposal_u),
                        float(value),
                        0.0,
                    )
                )
        for (hazard_proposal, surface), hazard_node in state.hierarchy.hazards:
            if hazard_proposal != proposal or len(hazard_node.upper) == 1:
                continue
            surface_upper = node.upper[surface]
            surface_lower = 0.0 if surface == 0 else node.upper[surface - 1]
            surface_u = (
                0.0
                if surface_lower == 0.0
                else float(np.nextafter(surface_lower, 1.0))
            )
            assert surface_u <= surface_upper
            for cut in hazard_node.upper[:-1]:
                for value in (np.nextafter(cut, 0.0), cut, np.nextafter(cut, 1.0)):
                    probes.append(
                        real.RealAppSemanticUniforms(
                            state.weighted.terminal_probability,
                            float(proposal_u),
                            surface_u,
                            float(value),
                        )
                    )
    assert len(probes) >= 20
    for semantic in probes:
        retained = real._uniforms(semantic)
        hierarchical, _ = real._hierarchical_index(state, retained)
        direct = real._direct_index(state, retained)
        assert hierarchical == direct


def test_synthetic_fraction_projection_fixture_matches_reviewed_primitive() -> None:
    masses = (
        Fraction.from_float(float(np.nextafter(0.2, 1.0))),
        Fraction(0, 1),
        Fraction.from_float(0.3),
        Fraction.from_float(0.5),
    )
    strict = real._fraction_node(
        (0, 1, 2, 3), masses, boundary="strict", allow_zero=True
    )
    denominator = sum(masses, Fraction(0, 1))
    running = Fraction(0, 1)
    expected = []
    for mass in masses:
        running += mass
        expected.append(
            synthetic._binary64_ratio_cut(running, denominator, boundary="strict")
        )
    assert strict.upper == tuple(expected)
    assert strict.upper[0] == strict.upper[1]


def test_fixture_mutations_fail_closed() -> None:
    group_split = _fixture()
    group_split.loc[0, "donor_quality_group_id"] = "f" * 64
    with pytest.raises(real.RealAppSelectorUnavailable, match="normalization group"):
        real._canonicalize_fixture(group_split)

    dropped_zero_topology = _fixture()
    full = json.loads(dropped_zero_topology.loc[0, "surface_full_probability_json"])
    full.pop(5)
    dropped_zero_topology.loc[
        dropped_zero_topology["proposal_row_order"].eq(0),
        "surface_full_probability_json",
    ] = json.dumps(full, separators=(",", ":"))
    with pytest.raises(real.RealAppSelectorUnavailable, match="full-surface topology"):
        real._canonicalize_fixture(dropped_zero_topology)

    reordered = _fixture().iloc[::-1].reset_index(drop=True)
    with pytest.raises(real.RealAppSelectorUnavailable, match="source row order"):
        real._canonicalize_fixture(reordered)


def test_real_contract_rejects_eight_slot_and_leading_zero_surfaces() -> None:
    eight_slots = _fixture()
    mask = eight_slots["proposal_row_order"].eq(0)
    probability = json.loads(eight_slots.loc[mask, "surface_full_probability_json"].iloc[0])
    cdf = json.loads(eight_slots.loc[mask, "surface_full_cdf_upper_json"].iloc[0])
    eight_slots.loc[mask, "surface_class_count"] = 8
    eight_slots.loc[mask, "surface_full_probability_json"] = json.dumps(
        probability[:-1], separators=(",", ":")
    )
    eight_slots.loc[mask, "surface_full_cdf_upper_json"] = json.dumps(
        cdf[:-1], separators=(",", ":")
    )
    with pytest.raises(real.RealAppSelectorUnavailable, match="exactly nine"):
        real._canonicalize_fixture(eight_slots)

    leading_zero = _fixture()
    leading_zero = leading_zero.loc[
        ~leading_zero["atom_id"].eq("p0-s0-h-1")
    ].reset_index(drop=True)
    leading_zero["complete_row_order"] = np.arange(len(leading_zero), dtype=np.int64)
    mask = leading_zero["proposal_row_order"].eq(0)
    probability = np.asarray(
        [0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=float
    )
    cdf = np.cumsum(probability)
    leading_zero.loc[mask, "surface_full_probability_json"] = json.dumps(
        probability.tolist(), separators=(",", ":")
    )
    leading_zero.loc[mask, "surface_full_cdf_upper_json"] = json.dumps(
        cdf.tolist(), separators=(",", ":")
    )
    fairway = mask & leading_zero["surface_class_index"].eq(1)
    leading_zero.loc[fairway, "surface_probability"] = 0.75
    leading_zero.loc[fairway, "surface_cdf_lower"] = 0.0
    leading_zero.loc[fairway, "surface_cdf_upper"] = 0.75
    leading_zero.loc[fairway, "mechanics_conditional_probability"] = 0.75
    leading_zero.loc[fairway, "quality_base_mass"] = 0.375
    with pytest.raises(real.RealAppSelectorUnavailable, match="slot zero"):
        real._canonicalize_fixture(leading_zero)


def test_terminal_surface_hazard_and_stroke_sentinels_fail_closed() -> None:
    mutations = (
        ("surface_class_label", "green"),
        ("surface_probability", 0.5),
        ("surface_cdf_lower", 0.25),
        ("surface_cdf_upper", 0.75),
        ("hazard_donor_key", "hazard-a"),
        ("hazard_probability", 0.5),
        ("hazard_cdf_lower", 0.25),
        ("hazard_cdf_upper", 0.75),
        ("stroke_cost", 2),
    )
    for column, value in mutations:
        frame = _fixture()
        frame.loc[frame["terminal"], column] = value
        with pytest.raises(real.RealAppSelectorUnavailable, match="sentinels"):
            real._canonicalize_fixture(frame)


def test_cross_proposal_mechanics_mass_transfer_fails_closed() -> None:
    frame = _fixture()
    p0 = int(frame.index[frame["atom_id"].eq("p0-s0-h-1")].item())
    p1 = int(frame.index[frame["atom_id"].eq("p1-s0-h-1")].item())
    frame.loc[p0, "mechanics_conditional_probability"] = 0.4
    frame.loc[p0, "quality_base_mass"] = 0.2
    frame.loc[p1, "mechanics_conditional_probability"] = 2.0 / 3.0
    frame.loc[p1, "quality_base_mass"] = 0.2
    assert frame["quality_base_mass"].sum() == pytest.approx(1.0)
    with pytest.raises(real.RealAppSelectorUnavailable, match="surface times hazard"):
        real._canonicalize_fixture(frame)


def test_collapsed_positive_hazard_interval_fails_closed() -> None:
    frame = _fixture()
    mask = frame["atom_id"].isin(("p0-s7-h0", "p0-s7-h1"))
    indices = frame.index[mask].tolist()
    tiny = 2.0**-54
    hazard_probability = np.asarray((1.0, tiny), dtype=float)
    hazard_lower = np.asarray((0.0, 1.0), dtype=float)
    hazard_upper = np.asarray((1.0, 1.0), dtype=float)
    mechanics = 0.25 * hazard_probability
    frame.loc[indices, "hazard_probability"] = hazard_probability
    frame.loc[indices, "hazard_cdf_lower"] = hazard_lower
    frame.loc[indices, "hazard_cdf_upper"] = hazard_upper
    frame.loc[indices, "mechanics_conditional_probability"] = mechanics
    frame.loc[indices, "quality_base_mass"] = 0.5 * mechanics
    with pytest.raises(real.RealAppSelectorUnavailable, match="distinct binary64"):
        real._canonicalize_fixture(frame)


def test_spatial_labels_strokes_and_live_keys_fail_closed() -> None:
    no_hazard_cost = _fixture()
    no_hazard_cost.loc[no_hazard_cost["atom_id"].eq("p0-s0-h-1"), "stroke_cost"] = 7
    with pytest.raises(real.RealAppSelectorUnavailable, match="No-hazard"):
        real._canonicalize_fixture(no_hazard_cost)

    resolved_cost = _fixture()
    resolved_cost.loc[resolved_cost["atom_id"].eq("p0-s7-h0"), "stroke_cost"] = 1
    with pytest.raises(real.RealAppSelectorUnavailable, match="penalty stroke"):
        real._canonicalize_fixture(resolved_cost)

    label = _fixture()
    label.loc[label["atom_id"].eq("p0-s1-h-1"), "surface_class_label"] = "bogus"
    with pytest.raises(real.RealAppSelectorUnavailable, match="index-to-label"):
        real._canonicalize_fixture(label)

    for column, value in (("end_state_key", "terminal"), ("next_support_key", "")):
        frame = _fixture()
        frame.loc[~frame["terminal"], column] = value
        with pytest.raises(real.RealAppSelectorUnavailable, match="live end-state"):
            real._canonicalize_fixture(frame)


@pytest.mark.parametrize("column", real._INTEGRAL_SCALAR_COLUMNS)
def test_integral_contract_columns_reject_float_storage(column: str) -> None:
    frame = _fixture()
    frame[column] = frame[column].astype(float)
    with pytest.raises(real.RealAppSelectorUnavailable, match=column):
        real._canonicalize_fixture(frame)


@pytest.mark.parametrize(
    ("column", "impostor"),
    (
        ("complete_atom", 1),
        ("complete_atom", "yes"),
        ("boundary_only", 0),
        ("terminal", 0),
    ),
)
def test_boolean_contract_columns_reject_truthy_impostors(
    column: str, impostor: object
) -> None:
    frame = _fixture()
    frame[column] = frame[column].astype(object)
    frame.at[0, column] = impostor
    with pytest.raises(real.RealAppSelectorUnavailable, match=column):
        real._canonicalize_fixture(frame)


def test_numeric_json_and_string_contracts_reject_coercible_impostors() -> None:
    numeric = _fixture()
    numeric["proposal_probability"] = numeric["proposal_probability"].astype(object)
    numeric.at[0, "proposal_probability"] = "0.5"
    with pytest.raises(real.RealAppSelectorUnavailable, match="genuine finite numeric"):
        real._canonicalize_fixture(numeric)

    key = _fixture()
    key["query_key"] = pd.Series([7] * len(key), dtype=object)
    with pytest.raises(real.RealAppSelectorUnavailable, match="genuine string"):
        real._canonicalize_fixture(key)

    with pytest.raises(real.RealAppSelectorUnavailable, match="genuine finite JSON"):
        real._json_vector("[true,false]", "adversarial JSON")


def test_scalar_families_reject_python_numeric_and_string_subclasses() -> None:
    mutations = (
        ("complete_row_order", _IntImpostor(0)),
        ("proposal_probability", _FloatImpostor(0.5)),
        ("query_key", _StringImpostor("fixture-lowry-r1h8-app")),
    )
    for column, value in mutations:
        frame = _fixture()
        frame[column] = frame[column].astype(object)
        frame.at[0, column] = value
        with pytest.raises(real.RealAppSelectorUnavailable, match=column):
            real._canonicalize_fixture(frame)


def test_persisted_timestamp_and_nat_shape_normalizes_to_utc() -> None:
    frame = _fixture()
    assert str(frame["donor_event_start"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["target_event_start"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["donor_skill_date"].dtype) == "datetime64[ns, UTC]"
    canonical = real._canonicalize_fixture(frame)
    assert str(canonical["donor_event_start"].dtype) == "datetime64[ns, UTC]"
    assert str(canonical["target_event_start"].dtype) == "datetime64[ns, UTC]"
    assert str(canonical["donor_skill_date"].dtype) == "datetime64[ns, UTC]"
    assert canonical.loc[
        canonical["donor_ability_value_kind"].eq(policy.IMPUTED_VALUE_KIND),
        "donor_skill_date",
    ].isna().all()
    assert len(real._support_sha(canonical)) == 64

    numpy_temporal = frame.copy(deep=True)
    for column in ("donor_event_start", "target_event_start", "donor_skill_date"):
        numpy_temporal[column] = pd.Series(
            [
                np.datetime64("NaT")
                if pd.isna(value)
                else value.tz_convert("UTC").tz_localize(None).to_datetime64()
                for value in frame[column]
            ],
            dtype=object,
        )
    numpy_canonical = real._canonicalize_fixture(numpy_temporal)
    assert real._support_sha(numpy_canonical) == real._support_sha(canonical)


def test_temporal_type_and_strict_chronology_attacks_fail_closed() -> None:
    malformed = _fixture()
    malformed["donor_event_start"] = malformed["donor_event_start"].astype(object)
    malformed.at[0, "donor_event_start"] = "2024-01-11T00:00:00Z"
    with pytest.raises(real.RealAppSelectorUnavailable, match="exact pandas/NumPy"):
        real._canonicalize_fixture(malformed)

    naive = _fixture()
    naive["donor_event_start"] = naive["donor_event_start"].astype(object)
    naive.at[0, "donor_event_start"] = pd.Timestamp("2024-01-11")
    with pytest.raises(real.RealAppSelectorUnavailable, match="timezone-aware"):
        real._canonicalize_fixture(naive)

    equal_event = _fixture()
    equal_event.at[0, "donor_event_start"] = equal_event.at[0, "target_event_start"]
    with pytest.raises(real.RealAppSelectorUnavailable, match="strictly prior"):
        real._canonicalize_fixture(equal_event)

    future_skill = _fixture()
    future_skill.at[0, "donor_skill_date"] = future_skill.at[
        0, "donor_event_start"
    ]
    with pytest.raises(real.RealAppSelectorUnavailable, match="strictly prior"):
        real._canonicalize_fixture(future_skill)


def test_spatial_terminal_cdf_requires_both_missing_sentinels() -> None:
    frame = _fixture()
    frame.loc[~frame["terminal"], "frozen_terminal_residual_cdf_lower"] = 0.123
    with pytest.raises(real.RealAppSelectorUnavailable, match="terminal CDF"):
        real._canonicalize_fixture(frame)


def test_fixed_h8_hole_and_unused_surface_slots_remain_positive_zero() -> None:
    frame = _fixture()
    drop = frame["atom_id"].eq("p0-s7-h1")
    frame = frame.loc[~drop].reset_index(drop=True)
    frame["complete_row_order"] = np.arange(len(frame), dtype=np.int64)
    mask = frame["proposal_row_order"].eq(0)
    probability = np.asarray(
        [0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=float
    )
    cdf = np.cumsum(probability)
    frame.loc[mask, "surface_full_probability_json"] = json.dumps(
        probability.tolist(), separators=(",", ":")
    )
    frame.loc[mask, "surface_full_cdf_upper_json"] = json.dumps(
        cdf.tolist(), separators=(",", ":")
    )
    hole = frame["atom_id"].eq("p0-s7-h0")
    frame.loc[hole, "atom_id"] = "p0-s6-h-1"
    frame.loc[hole, "surface_class_index"] = 6
    frame.loc[hole, "surface_class_label"] = "hole"
    frame.loc[hole, "surface_probability"] = 0.25
    frame.loc[hole, "surface_cdf_lower"] = 0.75
    frame.loc[hole, "surface_cdf_upper"] = 1.0
    frame.loc[hole, "hazard_resolution_status"] = "none"
    frame.loc[hole, "hazard_donor_order"] = -1
    frame.loc[hole, "hazard_donor_key"] = "not_applicable"
    frame.loc[hole, "hazard_probability"] = 1.0
    frame.loc[hole, "hazard_cdf_lower"] = 0.0
    frame.loc[hole, "hazard_cdf_upper"] = 1.0
    frame.loc[hole, "mechanics_conditional_probability"] = 0.25
    frame.loc[hole, "quality_base_mass"] = 0.125
    frame.loc[hole, "stroke_cost"] = 1
    frame.loc[hole, "end_state_key"] = "hole|0.000000yd"
    frame.loc[hole, "next_support_key"] = "next-p0-s6-h-1"
    with pytest.raises(real.RealAppSelectorUnavailable, match="slots 5, 6, and 8"):
        real._canonicalize_fixture(frame)


def test_real_scale_903_spatial_proposals_close_with_pinned_mass_tolerance() -> None:
    frame = _proposal_cap_fixture(903)
    canonical = real._canonicalize_fixture(frame)
    assert len(canonical) == 903 * 4 + 1
    assert canonical["proposal_row_order"].nunique() == 904
    assert canonical.loc[~canonical["terminal"], "proposal_row_order"].nunique() == 903


def test_905_total_proposals_exceed_fixed_real_h8_cap() -> None:
    frame = _proposal_cap_fixture(904)
    assert frame["proposal_row_order"].nunique() == 905
    with pytest.raises(real.RealAppSelectorUnavailable, match="Proposal source order"):
        real._canonicalize_fixture(frame)


def test_full_contract_digest_binds_selection_topology() -> None:
    original = _fixture()
    changed = _fixture()
    first = int(changed.index[changed["atom_id"].eq("p0-s7-h0")].item())
    second = int(changed.index[changed["atom_id"].eq("p0-s7-h1")].item())
    changed.loc[first, ["hazard_donor_order", "hazard_donor_key"]] = [1, "hazard-b"]
    changed.loc[second, ["hazard_donor_order", "hazard_donor_key"]] = [0, "hazard-a"]
    changed.loc[first, ["hazard_cdf_lower", "hazard_cdf_upper"]] = [0.6, 1.0]
    changed.loc[second, ["hazard_cdf_lower", "hazard_cdf_upper"]] = [0.0, 0.6]
    original_valid = real._canonicalize_fixture(original)
    changed_valid = real._canonicalize_fixture(changed)
    for column in (
        "atom_id",
        "quality_base_mass",
        "fixed_reference_reward",
        "donor_ability_sg",
    ):
        assert original_valid[column].equals(changed_valid[column])
    assert real._support_sha(original_valid) != real._support_sha(changed_valid)
    uniforms = real.RealAppSemanticUniforms(
        0.2, 0.0, float(np.nextafter(0.75, 1.0)), 0.2
    )
    before = real._select_fixture(original, _zero(), uniforms)
    after = real._select_fixture(changed, _zero(), uniforms)
    assert before.atom_id == "p0-s7-h0"
    assert after.atom_id == "p0-s7-h1"


def test_registry_mutation_cannot_self_authorize(monkeypatch) -> None:
    monkeypatch.setattr(real, "_AUTHORITY_RECORDS", ({"candidate": "self"},))
    with pytest.raises(real.RealAppSelectorUnavailable, match="registry lock"):
        real.real_app_authority_manifest()
    with pytest.raises(real.RealAppSelectorUnavailable, match="registry lock"):
        real.admit_lowry_r1h8_app_support()

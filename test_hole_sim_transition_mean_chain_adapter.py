from __future__ import annotations

from dataclasses import asdict
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

import hole_sim_transition_mean_anchor as anchor
import hole_sim_transition_mean_chain_adapter as adapter


def _contract_kwargs(atoms: pd.DataFrame, reward_sha: str) -> dict[str, object]:
    return {
        "expected_reward_reference_sha256": reward_sha,
        "expected_donor_source_sha256": str(atoms["donor_source_sha256"].iloc[0]),
        "expected_target_event_key": str(atoms["target_event_key"].iloc[0]),
        "expected_target_event_start": atoms["target_event_start"].iloc[0],
    }


def _model():
    atoms, roots, reward_sha = adapter.build_synthetic_128_state_atoms()
    crn = anchor.CommonRandomContext(
        seed=128,
        simulations=128,
        semantic_key="test|synthetic128|exact_support_lattice",
        payload_sha256=sha256(b"synthetic128-test-lattice").hexdigest(),
    )
    model = adapter.PhysicalChainExpectationAdapter(
        atoms,
        start_support_keys=roots,
        crn=crn,
        **_contract_kwargs(atoms, reward_sha),
    )
    return atoms, roots, reward_sha, crn, model


def _parameters(reward_sha: str, correction=(0.0, 0.0, 0.0, 0.0)):
    return anchor.TransitionQualityParameters(
        transfer_beta=anchor.DEFAULT_TRANSFER_BETA,
        target_ability_sg=(0.42, 0.18, -0.11, 0.07),
        lambda_correction=correction,
        reward_reference_sha256=reward_sha,
    )


def _reseal_physical_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    out = adapter.seal_physical_atom_hashes(frame)
    out = adapter.seal_app_holeout_equivalence(out)
    return adapter.seal_quality_input_hashes(out)


def test_128_state_callback_accumulates_categories_and_telescopes_physical_strokes() -> None:
    atoms, roots, reward_sha, crn, model = _model()
    result = model(_parameters(reward_sha, (0.16, -0.21, 0.19, 0.12)), crn)
    repeated = model(_parameters(reward_sha, (0.16, -0.21, 0.19, 0.12)), crn)

    assert len(model.supports) == 128
    assert len(atoms) == 512
    assert len(roots) == 18
    assert asdict(result) == asdict(repeated)
    assert result.diagnostics["analytic_complete_support_integration"] is True
    assert result.diagnostics["explicit_app_holeout_atoms"] == 36
    assert result.diagnostics["maximum_app_base_holeout_mass_residual"] == 0.0
    assert result.diagnostics["donor_quality_group_contract"].endswith(
        "excludes_player"
    )
    assert result.diagnostics["independent_app_holeout_probability"] == 0.0
    assert abs(result.diagnostics["max_atom_telescoping_residual"]) < 1e-12
    assert abs(result.diagnostics["round_telescoping_residual"]) < 1e-10
    assert abs(result.diagnostics["stroke_identity_residual"]) < 1e-10
    expected_from_strokes = (
        result.diagnostics["starting_reference_value"]
        - result.diagnostics["expected_physical_strokes"]
    )
    assert np.isclose(sum(result.mean_category_sg), expected_from_strokes, atol=1e-10)
    assert np.all(np.asarray(result.mean_category_sg) >= np.asarray(result.support_lower_sg))
    assert np.all(np.asarray(result.mean_category_sg) <= np.asarray(result.support_upper_sg))
    assert np.isclose(
        result.diagnostics["expected_physical_strokes"],
        result.diagnostics["expected_causal_shots"]
        + result.diagnostics["expected_penalty_strokes"],
        atol=1e-10,
    )


def test_multi_donor_event_state_group_detilts_and_preserves_mean_one_mass() -> None:
    _, roots, reward_sha, _, model = _model()
    support = model.supports[roots[0]]
    groups = support["donor_quality_group_id"].to_numpy(str)
    players = support["donor_player_id"].to_numpy(str)
    events = support["donor_event_key"].to_numpy(str)

    assert len(np.unique(players)) == 4
    assert len(np.unique(groups)) == 2
    for event in np.unique(events):
        event_rows = events == event
        assert len(np.unique(players[event_rows])) == 2
        assert len(np.unique(groups[event_rows])) == 1

    nonzero = _parameters(reward_sha)
    zero_beta = anchor.TransitionQualityParameters(
        transfer_beta=(0.0, 0.0, 0.0, 0.0),
        target_ability_sg=(0.0, 0.0, 0.0, 0.0),
        lambda_correction=(0.0, 0.0, 0.0, 0.0),
        reward_reference_sha256=reward_sha,
    )
    kwargs = {
        "base_mass": support["quality_base_mass"].to_numpy(float),
        "fixed_reference_reward": support["fixed_reference_reward"].to_numpy(float),
        "donor_ability_sg": support["donor_ability_sg"].to_numpy(float),
        "donor_group_id": groups,
        "category": "ott",
    }
    neutral = anchor.whole_atom_probabilities(parameters=nonzero, **kwargs)
    beta_zero = anchor.whole_atom_probabilities(parameters=zero_beta, **kwargs)
    base = support["quality_base_mass"].to_numpy(float)
    base /= base.sum()

    assert np.allclose(beta_zero.neutral_probability_mass, base, atol=1e-15, rtol=0.0)
    assert not np.allclose(neutral.neutral_probability_mass, base, atol=1e-10, rtol=0.0)
    for group_id in np.unique(groups):
        rows = groups == group_id
        before = float(base[rows].sum())
        after = float(neutral.neutral_probability_mass[rows].sum())
        weighted_mean_multiplier = float(
            np.sum(base[rows] * neutral.neutral_multiplier[rows]) / before
        )
        assert np.isclose(after, before, atol=2e-15, rtol=2e-15)
        assert np.isclose(weighted_mean_multiplier, 1.0, atol=2e-15, rtol=2e-15)


def test_exact_adapter_drives_joint_mean_anchor_without_post_round_recentering() -> None:
    _, _, reward_sha, crn, model = _model()
    hidden = np.asarray([0.16, -0.21, 0.19, 0.12])
    target = model(_parameters(reward_sha, tuple(hidden)), crn)
    problem = anchor.MeanAnchorProblem(
        target_category_sg=target.mean_category_sg,
        target_ability_sg=(0.42, 0.18, -0.11, 0.07),
        reward_reference_sha256=reward_sha,
        support_sha256=model.support_sha256,
        context_key="test|synthetic128|solver",
        production_total_sg=float(sum(target.mean_category_sg)),
        category_tolerance_sg=2e-9,
        finite_difference_step=0.02,
    )
    solved = anchor.solve_production_category_mean_anchor(problem, model, crn)

    assert solved.converged, solved.as_dict()
    assert solved.production_admissible
    assert np.max(np.abs(solved.residual_category_sg)) <= 2e-9
    assert np.allclose(solved.lambda_correction, hidden, atol=3e-7, rtol=0.0)
    assert solved.post_round_recenter_applied is False
    assert solved.diagnostics["jacobian_full_rank"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.assign(implicit_holeout_probability=np.where(
                frame["category"].eq("app"), 0.01, 0.0
            )),
            "Independent APP holeout probability",
        ),
        (
            lambda frame: frame.assign(donor_ability_available=False),
            "Missing donor ability",
        ),
        (
            lambda frame: frame.assign(quality_reward_available=False),
            "Missing fixed-reference reward",
        ),
    ],
)
def test_quality_inputs_fail_closed_instead_of_becoming_zero(mutation, message) -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    broken = mutation(atoms.copy())
    with pytest.raises(adapter.QualitySupportUnavailable, match=message):
        adapter.validate_complete_atom_frame(
            broken, **_contract_kwargs(atoms, reward_sha)
        )


def test_app_holeout_must_be_an_explicit_complete_atom() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    index = atoms.index[
        atoms["category"].eq("app") & atoms["terminal"]
    ][0]
    atoms.loc[index, "atom_kind"] = "standard"
    atoms = _reseal_physical_semantics(atoms)
    with pytest.raises(adapter.QualitySupportUnavailable, match="APP holeout atom"):
        adapter.validate_complete_atom_frame(
            atoms, **_contract_kwargs(atoms, reward_sha)
        )


def test_penalty_strokes_and_reward_must_share_one_complete_atom() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    index = atoms.index[atoms["penalty_strokes"].gt(0)][0]
    atoms.loc[index, "stroke_cost"] = 1
    with pytest.raises(adapter.QualitySupportUnavailable, match="stroke_cost"):
        adapter.validate_complete_atom_frame(
            atoms, **_contract_kwargs(atoms, reward_sha)
        )


def test_donor_source_detects_ability_mutation_before_quality_can_be_reused() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    atoms.loc[0, "donor_ability_sg"] += 0.01
    with pytest.raises(
        adapter.QualitySupportUnavailable,
        match="conflicting snapshot|donor_source_sha256",
    ):
        adapter.validate_complete_atom_frame(
            atoms, **_contract_kwargs(atoms, reward_sha)
        )


def test_graph_end_state_must_equal_next_support_start() -> None:
    atoms, roots, reward_sha = adapter.build_synthetic_128_state_atoms()
    index = atoms.index[~atoms["terminal"]][0]
    atoms.loc[index, "end_state_key"] = "wrong-state"
    atoms.loc[index, "end_reference_value"] = atoms.loc[index, "end_reference_value"]
    atoms.loc[index, "fixed_reference_reward"] = (
        atoms.loc[index, "start_reference_value"]
        - atoms.loc[index, "stroke_cost"]
        - atoms.loc[index, "end_reference_value"]
    )
    atoms = adapter.seal_complete_atom_contracts(atoms)
    reward_sha = adapter.reward_reference_sha256(atoms)
    crn = anchor.CommonRandomContext(
        seed=1,
        simulations=1,
        semantic_key="broken-edge",
        payload_sha256=sha256(b"broken-edge").hexdigest(),
    )
    with pytest.raises(adapter.QualitySupportUnavailable, match="end state"):
        adapter.PhysicalChainExpectationAdapter(
            atoms,
            start_support_keys=roots,
            crn=crn,
            **_contract_kwargs(atoms, reward_sha),
        )


def test_existing_support_contracts_report_exact_readiness_blockers() -> None:
    precompute = adapter.current_support_readiness(
        "precompute.SupportArrays", category="app"
    )
    streaming = adapter.current_support_readiness(
        "streaming.NumericSupport", category="app"
    )
    markov = adapter.current_support_readiness(
        "markov.TransitionPool", category="arg"
    )

    assert not precompute.anchor_ready
    assert "app_holeout_is_separate_scalar_not_complete_atom" in precompute.blockers
    assert not streaming.anchor_ready
    assert "reward_or_strict_prior_donor_ability_provenance_missing" in streaming.blockers
    assert not markov.anchor_ready
    assert "selected_donor_identity_is_discarded_by_pool" in markov.blockers
    assert "sg_field_is_not_bound_to_frozen_reward_reference_hash" in markov.blockers
    with pytest.raises(adapter.QualitySupportUnavailable):
        adapter.assert_current_support_anchor_ready(
            "streaming.NumericSupport", category="app"
        )


def test_production_mean_is_an_external_validated_input_only() -> None:
    frame = adapter.seal_production_target_hashes(pd.DataFrame(
        [
            {
                "event_key": "synthetic:R2026999",
                "player_id": "7",
                "round_no": 1,
                "target_ott_sg": 0.20,
                "target_app_sg": 0.30,
                "target_arg_sg": -0.10,
                "target_putt_sg": 0.05,
                "target_total_sg": 0.45,
            }
        ]
    ))
    validated = adapter.validate_production_target_frame(frame)
    assert validated.loc[0, "target_total_sg"] == 0.45

    broken = frame.copy()
    broken.loc[0, "target_total_sg"] = 0.46
    with pytest.raises(ValueError, match="do not sum"):
        adapter.validate_production_target_frame(broken)


@pytest.mark.parametrize(
    "field",
    [
        "start_state_key",
        "end_state_key",
        "category",
        "strokes_and_penalty",
        "terminal_and_holed",
        "next_support_key",
        "donor_event_key",
        "donor_player_id",
    ],
)
def test_physical_atom_hash_rejects_every_prior_unbound_identity_exploit(field: str) -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    broken = atoms.copy()
    index = broken.index[~broken["terminal"]][0]
    if field in {"start_state_key", "end_state_key", "next_support_key"}:
        broken.loc[index, field] = f"tampered-{field}"
    elif field == "category":
        broken.loc[index, "category"] = "app"
    elif field == "strokes_and_penalty":
        broken.loc[index, "penalty_strokes"] += 1
        broken.loc[index, "stroke_cost"] += 1
    elif field == "terminal_and_holed":
        broken.loc[index, ["terminal", "holed"]] = True
        broken.loc[index, "end_state_key"] = adapter.TERMINAL_STATE_KEY
        broken.loc[index, "next_support_key"] = ""
    elif field in {"donor_event_key", "donor_player_id"}:
        broken.loc[index, field] = f"tampered-{field}"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(field)

    with pytest.raises(adapter.QualitySupportUnavailable, match="physical_atom_sha256"):
        adapter.validate_complete_atom_frame(
            broken, **_contract_kwargs(atoms, reward_sha)
        )


def test_reward_reference_hash_cannot_be_a_caller_chosen_valid_sha() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    forged = "a" * 64
    broken = atoms.copy()
    broken["reward_reference_sha256"] = forged
    broken = adapter.seal_quality_input_hashes(broken)
    kwargs = _contract_kwargs(atoms, reward_sha)
    kwargs["expected_reward_reference_sha256"] = forged

    with pytest.raises(
        adapter.QualitySupportUnavailable,
        match="canonical full value table",
    ):
        adapter.validate_complete_atom_frame(broken, **kwargs)


def test_reward_reference_hash_binds_descriptor_version_and_full_state_table() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    original_kwargs = _contract_kwargs(atoms, reward_sha)

    version_changed = atoms.copy()
    version_changed["reward_reference_version"] = "forged-version"
    version_changed = adapter.seal_reward_reference_hash(version_changed)
    version_changed = adapter.seal_quality_input_hashes(version_changed)
    with pytest.raises(adapter.QualitySupportUnavailable, match="reference changed"):
        adapter.validate_complete_atom_frame(version_changed, **original_kwargs)

    table_changed = atoms.copy()
    root_state = str(table_changed.loc[0, "start_state_key"])
    rows = table_changed["start_state_key"].eq(root_state)
    table_changed.loc[rows, "start_reference_value"] += 0.01
    table_changed.loc[rows, "fixed_reference_reward"] = (
        table_changed.loc[rows, "start_reference_value"]
        - table_changed.loc[rows, "stroke_cost"]
        - table_changed.loc[rows, "end_reference_value"]
    )
    table_changed = adapter.seal_complete_atom_contracts(table_changed)
    with pytest.raises(adapter.QualitySupportUnavailable, match="reference changed"):
        adapter.validate_complete_atom_frame(table_changed, **original_kwargs)


def test_production_hash_is_content_derived_and_keys_deduplicate_canonically() -> None:
    base = pd.DataFrame(
        [
            {
                "event_key": " Synthetic:Event ",
                "player_id": " PLAYER-7 ",
                "round_no": 2,
                "target_ott_sg": 0.2,
                "target_app_sg": 0.3,
                "target_arg_sg": -0.1,
                "target_putt_sg": 0.05,
                "target_total_sg": 0.45,
            }
        ]
    )
    sealed = adapter.seal_production_target_hashes(base)
    validated = adapter.validate_production_target_frame(sealed)
    assert validated.loc[0, "event_key"] == "synthetic:event"
    assert validated.loc[0, "player_id"] == "player-7"

    forged = sealed.copy()
    forged["production_input_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="canonical player-round content"):
        adapter.validate_production_target_frame(forged)

    stale = sealed.copy()
    stale.loc[0, "target_app_sg"] += 0.01
    stale.loc[0, "target_total_sg"] += 0.01
    with pytest.raises(ValueError, match="stale"):
        adapter.validate_production_target_frame(stale)

    duplicate = pd.concat(
        [
            sealed,
            sealed.assign(event_key="  SYNTHETIC:EVENT  ", player_id=" player-7 "),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicated after canonicalization"):
        adapter.validate_production_target_frame(duplicate)


def test_donor_source_and_snapshot_hashes_are_content_verified() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    original_kwargs = _contract_kwargs(atoms, reward_sha)

    source_changed = atoms.copy()
    identity = source_changed.loc[
        0, ["donor_event_key", "donor_player_id", "category"]
    ]
    same_snapshot = (
        source_changed["donor_event_key"].eq(identity["donor_event_key"])
        & source_changed["donor_player_id"].eq(identity["donor_player_id"])
        & source_changed["category"].eq(identity["category"])
    )
    source_changed.loc[same_snapshot, "donor_ability_sg"] += 0.02
    source_changed = adapter.seal_donor_provenance(source_changed)
    source_changed = adapter.seal_quality_input_hashes(source_changed)
    with pytest.raises(adapter.QualitySupportUnavailable, match="donor source changed"):
        adapter.validate_complete_atom_frame(source_changed, **original_kwargs)

    snapshot_forged = atoms.copy()
    snapshot_forged.loc[0, "donor_snapshot_sha256"] = "c" * 64
    snapshot_forged = adapter.seal_quality_input_hashes(snapshot_forged)
    with pytest.raises(adapter.QualitySupportUnavailable, match="donor_snapshot_sha256"):
        adapter.validate_complete_atom_frame(snapshot_forged, **original_kwargs)


def test_donor_event_mapping_cutoff_and_internal_group_are_fail_closed() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    kwargs = _contract_kwargs(atoms, reward_sha)

    conflicting_start = atoms.copy()
    conflicting_start.loc[0, "donor_event_start"] = "2024-01-12T00:00:00Z"
    conflicting_start = adapter.seal_quality_input_hashes(conflicting_start)
    with pytest.raises(adapter.QualitySupportUnavailable, match="multiple canonical start"):
        adapter.validate_complete_atom_frame(conflicting_start, **kwargs)

    leaked = atoms.copy()
    leaked["target_event_start"] = "2024-01-05T00:00:00Z"
    leaked = adapter.seal_quality_input_hashes(leaked)
    with pytest.raises(adapter.QualitySupportUnavailable, match="target-event cutoff"):
        adapter.validate_complete_atom_frame(leaked, **kwargs)

    invented_group = atoms.copy()
    invented_group.loc[0, "donor_quality_group_id"] = "caller-controls-this-group"
    invented_group = adapter.seal_quality_input_hashes(invented_group)
    with pytest.raises(adapter.QualitySupportUnavailable, match="internally derived"):
        adapter.validate_complete_atom_frame(invented_group, **kwargs)


def test_app_holeout_is_terminal_and_holed_iff_with_per_support_equivalence() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    kwargs = _contract_kwargs(atoms, reward_sha)

    nonterminal_holeout = atoms.copy()
    index = nonterminal_holeout.index[
        nonterminal_holeout["category"].eq("app")
        & ~nonterminal_holeout["terminal"]
    ][0]
    nonterminal_holeout.loc[index, "atom_kind"] = "app_holeout"
    nonterminal_holeout = _reseal_physical_semantics(nonterminal_holeout)
    with pytest.raises(adapter.QualitySupportUnavailable, match="iff"):
        adapter.validate_complete_atom_frame(nonterminal_holeout, **kwargs)

    terminal_not_holed = atoms.copy()
    terminal_index = terminal_not_holed.index[
        terminal_not_holed["category"].eq("app")
        & terminal_not_holed["terminal"]
    ][0]
    terminal_not_holed.loc[terminal_index, "holed"] = False
    terminal_not_holed = _reseal_physical_semantics(terminal_not_holed)
    with pytest.raises(adapter.QualitySupportUnavailable, match="Terminal and holed"):
        adapter.validate_complete_atom_frame(terminal_not_holed, **kwargs)

    stale_equivalence = atoms.copy()
    app_support = stale_equivalence.loc[
        stale_equivalence["category"].eq("app"), "support_key"
    ].iloc[0]
    stale_equivalence.loc[
        stale_equivalence["support_key"].eq(app_support),
        "app_holeout_equivalence_sha256",
    ] = "d" * 64
    stale_equivalence = adapter.seal_quality_input_hashes(stale_equivalence)
    with pytest.raises(adapter.QualitySupportUnavailable, match="equivalence digest"):
        adapter.validate_complete_atom_frame(stale_equivalence, **kwargs)


def test_app_declared_holeout_mass_matches_atoms_and_zero_requires_explicit_zero() -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    kwargs = _contract_kwargs(atoms, reward_sha)
    app_support = str(
        atoms.loc[atoms["category"].eq("app"), "support_key"].iloc[0]
    )
    support_rows = atoms["support_key"].eq(app_support)

    wrong_mass = atoms.copy()
    wrong_mass.loc[
        support_rows, "declared_app_base_holeout_probability"
    ] = 0.51
    wrong_mass = adapter.seal_app_holeout_equivalence(wrong_mass)
    wrong_mass = adapter.seal_quality_input_hashes(wrong_mass)
    with pytest.raises(adapter.QualitySupportUnavailable, match="actual base holeout mass"):
        adapter.validate_complete_atom_frame(wrong_mass, **kwargs)

    no_holeout = atoms.copy()
    terminal_rows = support_rows & no_holeout["terminal"]
    progress = no_holeout.loc[support_rows & ~no_holeout["terminal"]].iloc[0]
    no_holeout.loc[terminal_rows, "terminal"] = False
    no_holeout.loc[terminal_rows, "holed"] = False
    no_holeout.loc[terminal_rows, "atom_kind"] = "standard"
    no_holeout.loc[terminal_rows, "end_state_key"] = progress["end_state_key"]
    no_holeout.loc[terminal_rows, "next_support_key"] = progress["next_support_key"]
    no_holeout.loc[terminal_rows, "end_reference_value"] = progress[
        "end_reference_value"
    ]
    no_holeout.loc[terminal_rows, "fixed_reference_reward"] = (
        no_holeout.loc[terminal_rows, "start_reference_value"]
        - no_holeout.loc[terminal_rows, "stroke_cost"]
        - no_holeout.loc[terminal_rows, "end_reference_value"]
    )
    no_holeout = _reseal_physical_semantics(no_holeout)
    with pytest.raises(adapter.QualitySupportUnavailable, match="coverage must exist iff"):
        adapter.validate_complete_atom_frame(no_holeout, **kwargs)

    explicit_zero = no_holeout.copy()
    explicit_zero.loc[
        support_rows, "declared_app_base_holeout_probability"
    ] = 0.0
    explicit_zero = adapter.seal_app_holeout_equivalence(explicit_zero)
    explicit_zero = adapter.seal_quality_input_hashes(explicit_zero)
    validated = adapter.validate_complete_atom_frame(explicit_zero, **kwargs)
    assert not validated.loc[
        validated["support_key"].eq(app_support), "terminal"
    ].any()


def test_public_contract_loader_round_trips_parquet_and_gates_csv(tmp_path: Path) -> None:
    atoms, _, reward_sha = adapter.build_synthetic_128_state_atoms()
    kwargs = _contract_kwargs(atoms, reward_sha)
    parquet_path = tmp_path / "complete_atoms.parquet"
    csv_path = tmp_path / "complete_atoms.csv"
    atoms.to_parquet(parquet_path, index=False, engine="pyarrow", compression="zstd")
    atoms.to_csv(csv_path, index=False, lineterminator="\n")

    parquet = adapter.load_complete_atom_contract(parquet_path, **kwargs)
    assert adapter._records_sha(
        parquet, adapter.SUPPORT_HASH_COLUMNS, adapter.ATOM_SCHEMA_CONTRACT
    ) == adapter._records_sha(
        adapter.validate_complete_atom_frame(atoms, **kwargs),
        adapter.SUPPORT_HASH_COLUMNS,
        adapter.ATOM_SCHEMA_CONTRACT,
    )

    with pytest.raises(adapter.QualitySupportUnavailable, match="CSV atom loading is unsafe"):
        adapter.load_complete_atom_contract(csv_path, **kwargs)
    with pytest.warns(adapter.UnsafeContractSerializationWarning, match="round-trip"):
        round_trip_csv = adapter.load_complete_atom_contract(
            csv_path,
            allow_round_trip_csv=True,
            **kwargs,
        )
    assert len(round_trip_csv) == len(atoms)


def test_smoke_manifest_self_attests_sources_inputs_and_every_nonself_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "adapter-smoke-v2"
    manifest_path = adapter.run_128_state_smoke(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_start_equals_end"] is True
    assert manifest["source_sha256_start"] == manifest["source_sha256_end"]
    assert set(manifest["source_sha256_start"]) == {
        "adapter",
        "adapter_test",
        "anchor",
    }
    assert manifest["input_start_equals_end"] is True
    assert manifest["input_start"] == manifest["input_end"]
    assert set(manifest["input_start"]) == {
        "canonical_complete_atom_frame",
        "canonical_physical_atom_set",
        "canonical_full_reward_reference",
        "canonical_strict_prior_donor_source",
        "canonical_production_player_round",
        "hole_start_supports",
        "crn_payload",
    }
    for digest in manifest["input_start"].values():
        assert adapter._is_sha256(digest)

    generated_nonself = {
        path.name for path in output_dir.iterdir()
    }.difference({"manifest.json", "manifest.sha256"})
    assert manifest["artifact_start_equals_end"] is True
    assert manifest["artifact_start"] == manifest["artifact_end"]
    assert generated_nonself == set(manifest["artifact_start"])
    for name, digest in manifest["artifact_start"].items():
        assert sha256((output_dir / name).read_bytes()).hexdigest() == digest

    assert manifest["detached_manifest_self_check_passed"] is True
    payload_digest = manifest.pop("manifest_payload_sha256")
    assert adapter._stable_sha(manifest) == payload_digest
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    assert (output_dir / "manifest.sha256").read_text(encoding="ascii") == (
        f"{manifest_digest}  manifest.json\n"
    )
    with pytest.raises(FileExistsError, match="not fresh"):
        adapter.run_128_state_smoke(output_dir)

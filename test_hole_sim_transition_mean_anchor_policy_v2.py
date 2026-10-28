from __future__ import annotations

from copy import deepcopy
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

import hole_sim_donor_ability_policy as donor_policy
import hole_sim_transition_mean_anchor as anchor
import hole_sim_transition_mean_anchor_policy_v2 as seam
import hole_sim_transition_mean_chain_adapter as v3
import hole_sim_transition_mean_chain_adapter_v4 as v4


FROZEN_FIXTURE_AUTHORITY_SHA256 = (
    "e6177dc0ed5f7545af06d4f61a23dd8458abdcbb98f2cda1f52e981f0330562c"
)


def _stable_sha(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _root_sha(manifest: dict[str, object]) -> str:
    return _stable_sha(
        {
            "contract": seam.ROOT_SUPPORT_CONTRACT,
            "support_inventory_sha256": manifest["support_inventory_sha256"],
            "start_support_keys": manifest["start_support_keys"],
            "expected_round_holes": manifest["expected_round_holes"],
            "target_event_key": manifest["target_event_key"],
            "target_event_start": manifest["target_event_start"],
        }
    )


def _authority_manifest(
    atoms: pd.DataFrame,
    roots: tuple[str, ...],
    reward_sha: str,
) -> tuple[dict[str, object], str]:
    manifest: dict[str, object] = {
        "contract": seam.AUTHORITY_CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "authority_source_descriptor": seam.APPROVED_AUTHORITY_SOURCE_DESCRIPTOR,
        "policy_support_mapping": dict(
            seam.EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT
        ),
        "anchor_contract": anchor.ANCHOR_CONTRACT,
        "anchor_source_descriptor": "archive/hole_sim_transition_mean_anchor.py",
        "anchor_source_sha256": seam.ANCHOR_SOURCE_SHA256,
        "adapter_contract": v4.ATOM_SCHEMA_CONTRACT,
        "adapter_source_descriptor": (
            "archive/hole_sim_transition_mean_chain_adapter_v4.py"
        ),
        "adapter_source_sha256": seam.ADAPTER_SOURCE_SHA256,
        "v3_adapter_contract": v3.ATOM_SCHEMA_CONTRACT,
        "v3_adapter_source_descriptor": seam.V3_ADAPTER_SOURCE_DESCRIPTOR,
        "v3_adapter_source_sha256": seam.V3_ADAPTER_SOURCE_SHA256,
        "policy_contract": v4.FROZEN_POLICY_CONTRACT,
        "policy_identity_sha256": v4.FROZEN_POLICY_IDENTITY_SHA256,
        "policy_source_descriptor": v4.FROZEN_POLICY_SOURCE_DESCRIPTOR,
        "policy_source_sha256": seam.POLICY_SOURCE_SHA256,
        "support_inventory_contract": v4.SUPPORT_INVENTORY_CONTRACT,
        "support_inventory_sha256": str(
            atoms["support_inventory_sha256"].iloc[0]
        ),
        "root_support_contract": seam.ROOT_SUPPORT_CONTRACT,
        "root_support_sha256": "",
        "start_support_keys": list(roots),
        "expected_round_holes": len(roots),
        "reward_reference_sha256": reward_sha,
        "donor_source_sha256": str(atoms["donor_source_sha256"].iloc[0]),
        "target_event_key": v3._canonical_text(
            atoms["target_event_key"].iloc[0], "target_event_key"
        ),
        "target_event_start": v3._canonical_timestamp(
            atoms["target_event_start"].iloc[0], "target_event_start"
        ),
    }
    manifest["root_support_sha256"] = _root_sha(manifest)
    return manifest, _stable_sha(manifest)


def _parameters(
    reward_sha: str,
    *,
    beta: tuple[float, float, float, float] = anchor.DEFAULT_TRANSFER_BETA,
    target: tuple[float, float, float, float] = (0.42, 0.18, -0.11, 0.07),
    correction: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> anchor.TransitionQualityParameters:
    return anchor.TransitionQualityParameters(
        transfer_beta=beta,
        target_ability_sg=target,
        lambda_correction=correction,
        reward_reference_sha256=reward_sha,
    )


@pytest.fixture(scope="module")
def admitted_contract() -> dict[str, object]:
    original, roots, reward_sha = v3.build_synthetic_128_state_atoms()
    base = original.copy(deep=True)
    event = base.loc[base["category"].eq("ott"), "donor_event_key"].iloc[0]
    imputed = base["donor_event_key"].eq(event) & base["category"].eq("ott")
    undated_player = base.loc[imputed, "donor_player_id"].iloc[0]
    base.loc[imputed, "donor_ability_available"] = False
    base.loc[imputed, "donor_ability_sg"] = np.nan
    base.loc[
        imputed & base["donor_player_id"].eq(undated_player), "donor_skill_date"
    ] = None
    policy_rows = v4.build_policy_rows_from_atoms(base)
    atoms = v4.attest_policy_v2_donor_rows(base, policy_rows)
    manifest, manifest_sha = _authority_manifest(atoms, roots, reward_sha)
    assert manifest_sha == FROZEN_FIXTURE_AUTHORITY_SHA256
    session = seam.admit_policy_v2_weighting(
        atoms,
        authority_manifest=manifest,
    )
    support_kind_count = atoms.groupby("support_key")[
        "donor_ability_value_kind"
    ].nunique()
    support_key = str(support_kind_count[support_kind_count.eq(2)].index[0])
    return {
        "original": original,
        "base": base,
        "atoms": atoms,
        "roots": roots,
        "reward_sha": reward_sha,
        "manifest": manifest,
        "manifest_sha": FROZEN_FIXTURE_AUTHORITY_SHA256,
        "session": session,
        "support_key": support_key,
    }


def test_explicit_policy_mapping_and_candidate_digests_cannot_unlock_admission(
    admitted_contract: dict[str, object],
) -> None:
    atoms = admitted_contract["atoms"]
    manifest = admitted_contract["manifest"]
    assert dict(seam.EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT) == {
        v4.FROZEN_POLICY_CONTRACT: v4.FROZEN_POLICY_IDENTITY_SHA256
    }
    with pytest.raises(TypeError):
        seam.EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT["forged"] = "0" * 64
    for adjacent_digest in (
        v4.FROZEN_POLICY_IDENTITY_SHA256,
        str(atoms["support_inventory_sha256"].iloc[0]),
        str(manifest["root_support_sha256"]),
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            seam.admit_policy_v2_weighting(
                atoms,
                authority_manifest=manifest,
                expected_authority_manifest_sha256=adjacent_digest,
            )
    manifest_with_self_authority = deepcopy(manifest)
    manifest_with_self_authority["authority_manifest_sha256"] = _stable_sha(manifest)
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable, match="manifest fields differ"
    ):
        seam.admit_policy_v2_weighting(
            atoms, authority_manifest=manifest_with_self_authority
        )


def test_direct_unvalidated_session_and_array_weighting_are_not_exposed(
    admitted_contract: dict[str, object],
) -> None:
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable, match="must be minted"
    ):
        seam.PolicyV2WeightingSession(
            admitted_contract["atoms"], object(), object()
        )
    assert not hasattr(seam, "whole_atom_probabilities")
    assert not hasattr(seam, "solve")
    assert not hasattr(seam, "simulate")
    assert not hasattr(seam, "anchor")
    assert not hasattr(seam, "v3")
    assert not hasattr(seam, "v4")
    assert not hasattr(seam, "donor_policy")
    assert not hasattr(seam, "sys")
    assert not hasattr(seam, "_AUTHORITY_TOKEN")
    assert not hasattr(seam, "_SESSION_TOKEN")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "anchor_source_sha256",
            "0" * 64,
            "wrong anchor_source_sha256",
        ),
        (
            "adapter_source_sha256",
            "1" * 64,
            "wrong adapter_source_sha256",
        ),
        (
            "v3_adapter_source_sha256",
            "3" * 64,
            "wrong v3_adapter_source_sha256",
        ),
        (
            "policy_source_sha256",
            "2" * 64,
            "wrong policy_source_sha256",
        ),
    ],
)
def test_wrong_bound_source_is_rejected_by_closed_authority_schema(
    admitted_contract: dict[str, object],
    field: str,
    value: str,
    message: str,
) -> None:
    forged = deepcopy(admitted_contract["manifest"])
    forged[field] = value
    with pytest.raises(seam.PolicyV2AnchorContractUnavailable, match=message):
        seam.admit_policy_v2_weighting(
            admitted_contract["atoms"],
            authority_manifest=forged,
        )


def test_wrong_policy_mapping_is_rejected_by_closed_authority_schema(
    admitted_contract: dict[str, object],
) -> None:
    forged = deepcopy(admitted_contract["manifest"])
    forged["policy_support_mapping"] = {
        v4.FROZEN_POLICY_CONTRACT: "f" * 64,
        "forged_policy": "e" * 64,
    }
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable, match="Wrong or expanded"
    ):
        seam.admit_policy_v2_weighting(
            admitted_contract["atoms"],
            authority_manifest=forged,
        )


def test_wrong_inventory_is_rejected_before_candidate_rows_can_authorize_it(
    admitted_contract: dict[str, object],
) -> None:
    forged = deepcopy(admitted_contract["manifest"])
    forged["support_inventory_sha256"] = "f" * 64
    forged["root_support_sha256"] = _root_sha(forged)
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="outside the closed synthetic approval lock: support_inventory_sha256",
    ):
        seam.admit_policy_v2_weighting(
            admitted_contract["atoms"],
            authority_manifest=forged,
        )


def test_wrong_root_digest_is_rejected_by_root_content_binding(
    admitted_contract: dict[str, object],
) -> None:
    forged = deepcopy(admitted_contract["manifest"])
    forged["root_support_sha256"] = "f" * 64
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable, match="Root-support digest is stale"
    ):
        seam.admit_policy_v2_weighting(
            admitted_contract["atoms"],
            authority_manifest=forged,
        )


def test_wrong_but_self_consistent_root_list_is_outside_closed_approval(
    admitted_contract: dict[str, object],
) -> None:
    forged = deepcopy(admitted_contract["manifest"])
    roots = list(forged["start_support_keys"])
    all_supports = set(admitted_contract["atoms"]["support_key"].astype(str))
    replacement = sorted(all_supports.difference(roots))[0]
    roots[0] = replacement
    forged["start_support_keys"] = roots
    forged["root_support_sha256"] = _root_sha(forged)
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="outside the closed synthetic approval lock: root_support_sha256",
    ):
        seam.admit_policy_v2_weighting(
            admitted_contract["atoms"],
            authority_manifest=forged,
        )


def test_fully_resealed_graph_mutation_fails_original_inventory_authority(
    admitted_contract: dict[str, object],
) -> None:
    changed = admitted_contract["atoms"].copy(deep=True)
    row = changed.index[changed["category"].eq("ott")][0]
    changed.loc[row, "spatial_model_weight"] = (
        float(changed.loc[row, "spatial_model_weight"]) * 1.01
    )
    changed.loc[row, "quality_base_mass"] = (
        float(changed.loc[row, "base_hierarchy_mass"])
        * float(changed.loc[row, "spatial_model_weight"])
    )
    changed = v4.seal_policy_v2_contracts(changed)
    assert (
        changed["support_inventory_sha256"].iloc[0]
        != admitted_contract["atoms"]["support_inventory_sha256"].iloc[0]
    )
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="Canonical full support inventory changed",
    ):
        seam.admit_policy_v2_weighting(
            changed,
            authority_manifest=admitted_contract["manifest"],
        )


def test_mutated_resealed_graph_cannot_self_authorize_with_fresh_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, roots, reward_sha = v3.build_synthetic_128_state_atoms()
    event = base.loc[base["category"].eq("ott"), "donor_event_key"].iloc[0]
    imputed = base["donor_event_key"].eq(event) & base["category"].eq("ott")
    player = base.loc[imputed, "donor_player_id"].iloc[0]
    base.loc[imputed, "donor_ability_available"] = False
    base.loc[imputed, "donor_ability_sg"] = np.nan
    base.loc[
        imputed & base["donor_player_id"].eq(player), "donor_skill_date"
    ] = None
    atoms = v4.attest_policy_v2_donor_rows(
        base, v4.build_policy_rows_from_atoms(base)
    )
    row = atoms.index[atoms["category"].eq("ott")][0]
    atoms.loc[row, "spatial_model_weight"] = (
        float(atoms.loc[row, "spatial_model_weight"]) * 1.01
    )
    atoms.loc[row, "quality_base_mass"] = (
        float(atoms.loc[row, "base_hierarchy_mass"])
        * float(atoms.loc[row, "spatial_model_weight"])
    )
    atoms = v4.seal_policy_v2_contracts(atoms)
    self_authorized, self_authorized_sha = _authority_manifest(
        atoms, roots, reward_sha
    )
    assert self_authorized_sha != seam.APPROVED_AUTHORITY_MANIFEST_SHA256
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="outside the closed synthetic approval lock: support_inventory_sha256",
    ):
        seam.admit_policy_v2_weighting(
            atoms, authority_manifest=self_authorized
        )
    monkeypatch.setattr(
        seam, "APPROVED_AUTHORITY_MANIFEST_SHA256", self_authorized_sha
    )
    monkeypatch.setattr(
        seam,
        "APPROVED_SUPPORT_INVENTORY_SHA256",
        str(atoms["support_inventory_sha256"].iloc[0]),
    )
    monkeypatch.setattr(
        seam,
        "APPROVED_ROOT_SUPPORT_SHA256",
        str(self_authorized["root_support_sha256"]),
    )
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="Public approval-lock report constants changed",
    ):
        seam.admit_policy_v2_weighting(
            atoms, authority_manifest=self_authorized
        )


@pytest.mark.parametrize("alteration", ["availability", "date"])
def test_imputed_availability_or_date_alteration_fails_v4_readmission(
    admitted_contract: dict[str, object],
    alteration: str,
) -> None:
    forged = admitted_contract["atoms"].copy(deep=True)
    imputed = forged["donor_ability_value_kind"].eq(
        donor_policy.IMPUTED_VALUE_KIND
    )
    row = forged.index[imputed][0]
    if alteration == "availability":
        forged.loc[row, "donor_ability_available"] = True
    else:
        forged.loc[row, "donor_skill_date"] = "2020-01-01T00:00:00Z"
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable, match="Adapter-v4 admission failed"
    ):
        seam.admit_policy_v2_weighting(
            forged,
            authority_manifest=admitted_contract["manifest"],
        )


def test_v4_valid_group_that_mixes_observed_and_imputed_is_rejected_by_seam() -> None:
    base, roots, reward_sha = v3.build_synthetic_128_state_atoms()
    identity = base.loc[base["category"].eq("ott")].iloc[0]
    rows = (
        base["donor_event_key"].eq(identity["donor_event_key"])
        & base["donor_player_id"].eq(identity["donor_player_id"])
        & base["category"].eq(identity["category"])
    )
    base.loc[rows, "donor_ability_available"] = False
    base.loc[rows, "donor_ability_sg"] = np.nan
    policy_rows = v4.build_policy_rows_from_atoms(base)
    atoms = v4.attest_policy_v2_donor_rows(base, policy_rows)
    assert (
        atoms.groupby("donor_quality_group_id")["donor_ability_value_kind"]
        .nunique()
        .max()
        == 2
    )
    manifest, _ = _authority_manifest(atoms, roots, reward_sha)
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="outside the closed synthetic approval lock",
    ):
        seam.admit_policy_v2_weighting(
            atoms,
            authority_manifest=manifest,
        )


def test_exact_imputed_zero_reaches_frozen_weighting_and_group_mass_is_preserved(
    admitted_contract: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = admitted_contract["session"]
    support_key = admitted_contract["support_key"]
    support = session.atoms.loc[session.atoms["support_key"].eq(support_key)]
    imputed = support["donor_ability_value_kind"].eq(
        donor_policy.IMPUTED_VALUE_KIND
    ).to_numpy(bool)
    assert imputed.any()
    assert not support.loc[imputed, "donor_ability_available"].astype(bool).any()
    assert np.equal(support.loc[imputed, "donor_ability_sg"], 0.0).all()
    assert not np.signbit(support.loc[imputed, "donor_ability_sg"].to_numpy(float)).any()

    captured: list[np.ndarray] = []
    original_weighting = anchor.whole_atom_probabilities

    def spy(**kwargs: object) -> anchor.AtomMassResult:
        captured.append(np.asarray(kwargs["donor_ability_sg"], dtype=float).copy())
        return original_weighting(**kwargs)

    monkeypatch.setattr(seam, "_whole_atom_probabilities", spy)
    zero = session.weight_support(
        support_key,
        _parameters(
            admitted_contract["reward_sha"],
            beta=(0.0, 0.0, 0.0, 0.0),
            target=(0.0, 0.0, 0.0, 0.0),
        ),
    )
    quality = session.weight_support(
        support_key,
        _parameters(
            admitted_contract["reward_sha"],
            correction=(0.35, 0.0, 0.0, 0.0),
        ),
    )
    assert len(captured) == 2
    assert np.array_equal(captured[0], support["donor_ability_sg"].to_numpy(float))
    assert np.equal(captured[0][imputed], 0.0).all()
    assert not np.signbit(captured[0][imputed]).any()

    base = support["quality_base_mass"].to_numpy(float)
    base /= base.sum()
    assert np.allclose(zero.neutral_probability_mass, base, atol=1e-15, rtol=0.0)
    assert np.allclose(zero.probability, base, atol=1e-15, rtol=0.0)
    assert not np.allclose(quality.probability, base, atol=1e-10, rtol=0.0)
    for result in (zero, quality):
        assert np.allclose(
            result.donor_group_mass_before,
            result.donor_group_mass_after,
            atol=2e-15,
            rtol=2e-15,
        )


def test_readiness_preserves_availability_and_retains_integration_hold(
    admitted_contract: dict[str, object],
) -> None:
    session = admitted_contract["session"]
    readiness = session.readiness
    atoms = session.atoms
    imputed = atoms["donor_ability_value_kind"].eq(
        donor_policy.IMPUTED_VALUE_KIND
    )
    assert readiness.rows == 512
    assert readiness.supports == 128
    assert readiness.roots == 18
    assert readiness.observed_rows == 476
    assert readiness.imputed_rows == readiness.unavailable_rows == 36
    assert readiness.nullable_skill_date_rows == 18
    assert readiness.availability_preserved
    assert readiness.approval_lock_sha256 == seam.APPROVAL_LOCK_SHA256
    assert readiness.approval_lock_status == seam.APPROVAL_LOCK_STATUS
    assert readiness.synthetic_fixture_only
    assert readiness.weighting_ready
    assert not readiness.real_chain_ready
    assert not readiness.integration_ready
    assert readiness.integration_blockers == (
        "no_policy_v2_physical_chain_expectation_callback",
        "no_policy_v2_mean_solve_or_simulation_integration",
    )
    assert not atoms.loc[imputed, "donor_ability_available"].astype(bool).any()
    copied = session.atoms
    copied.loc[copied.index[0], "donor_ability_available"] = not bool(
        copied.loc[copied.index[0], "donor_ability_available"]
    )
    assert session.atoms.equals(atoms)


def test_frozen_runtime_binding_hashes_all_four_executed_sources() -> None:
    runtime = seam.validate_frozen_runtime_bindings()
    assert runtime["anchor_source_sha256"] == seam.ANCHOR_SOURCE_SHA256
    assert runtime["adapter_source_sha256"] == seam.ADAPTER_SOURCE_SHA256
    assert runtime["v3_adapter_source_sha256"] == seam.V3_ADAPTER_SOURCE_SHA256
    assert runtime["policy_source_sha256"] == seam.POLICY_SOURCE_SHA256
    assert runtime["policy_support_mapping"] == dict(
        seam.EXPLICIT_DONOR_ABILITY_POLICY_SUPPORT
    )
    assert runtime["runtime_binding_sha256"] == (
        "be1aceea9d1f8e94dae04b3bb8382e7f2e7c2c80d9813cf48a8b3b814b027b56"
    )


def test_runtime_rejects_v3_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_file_sha256 = seam._file_sha256

    def forged_file_sha256(path: Path) -> str:
        if path.name == "hole_sim_transition_mean_chain_adapter.py":
            return "0" * 64
        return real_file_sha256(path)

    monkeypatch.setattr(seam, "_file_sha256", forged_file_sha256)
    with pytest.raises(
        seam.PolicyV2AnchorContractUnavailable,
        match="v3_adapter_source_sha256 changed",
    ):
        seam.validate_frozen_runtime_bindings()

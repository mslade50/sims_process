from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_donor_ability_policy as donor_policy
import hole_sim_transition_mean_chain_adapter as v3
import hole_sim_transition_mean_chain_adapter_v4 as v4


def _kwargs(atoms: pd.DataFrame, reward_sha: str) -> dict[str, object]:
    return {
        "expected_reward_reference_sha256": reward_sha,
        "expected_donor_source_sha256": str(atoms["donor_source_sha256"].iloc[0]),
        "expected_support_inventory_sha256": str(
            atoms["support_inventory_sha256"].iloc[0]
        ),
        "expected_target_event_key": str(atoms["target_event_key"].iloc[0]),
        "expected_target_event_start": atoms["target_event_start"].iloc[0],
    }


def _identity_mask(frame: pd.DataFrame, identity: pd.Series) -> pd.Series:
    return (
        frame["donor_event_key"].eq(identity["donor_event_key"])
        & frame["donor_player_id"].eq(identity["donor_player_id"])
        & frame["category"].eq(identity["category"])
    )


def _reseal_physical_and_app(frame: pd.DataFrame) -> pd.DataFrame:
    """Model an attacker who recomputes every accessible v3 physical/APP seal."""

    return v3.seal_app_holeout_equivalence(v3.seal_physical_atom_hashes(frame))


@pytest.fixture(scope="module")
def policy_v2_contracts() -> dict[str, object]:
    base, _, reward_sha = v3.build_synthetic_128_state_atoms()

    observed_policy = v4.build_policy_rows_from_atoms(base)
    observed = v4.attest_policy_v2_donor_rows(base, observed_policy)

    mixed_base = base.copy(deep=True)
    identities = mixed_base[
        ["donor_event_key", "donor_player_id", "category"]
    ].drop_duplicates().reset_index(drop=True)
    dated_identity = identities.iloc[0]
    undated_identity = identities.iloc[1]
    dated_mask = _identity_mask(mixed_base, dated_identity)
    undated_mask = _identity_mask(mixed_base, undated_identity)
    for mask in (dated_mask, undated_mask):
        mixed_base.loc[mask, "donor_ability_available"] = False
        mixed_base.loc[mask, "donor_ability_sg"] = np.nan
    mixed_base.loc[undated_mask, "donor_skill_date"] = None
    mixed_policy = v4.build_policy_rows_from_atoms(mixed_base)
    mixed = v4.attest_policy_v2_donor_rows(mixed_base, mixed_policy)

    return {
        "reward_sha": reward_sha,
        "observed": observed,
        "observed_policy": observed_policy,
        "mixed": mixed,
        "mixed_base": mixed_base,
        "mixed_policy": mixed_policy,
        "dated_identity": dated_identity,
        "undated_identity": undated_identity,
    }


def test_v4_accepts_only_attested_observed_or_exact_neutral_zero_and_fails_closed(
    policy_v2_contracts: dict[str, object],
) -> None:
    reward_sha = str(policy_v2_contracts["reward_sha"])
    observed = policy_v2_contracts["observed"]
    mixed = policy_v2_contracts["mixed"]

    observed_valid = v4.validate_policy_v2_atom_frame(
        observed, **_kwargs(observed, reward_sha)
    )
    observed_readiness = v4.policy_v2_anchor_readiness(
        observed, **_kwargs(observed, reward_sha)
    )
    assert observed_valid["donor_ability_value_kind"].eq(
        donor_policy.OBSERVED_VALUE_KIND
    ).all()
    assert observed_readiness.anchor_ready
    assert observed_readiness.imputed_rows == 0

    mixed_valid = v4.validate_policy_v2_atom_frame(
        mixed, **_kwargs(mixed, reward_sha)
    )
    imputed = mixed_valid["donor_ability_value_kind"].eq(
        donor_policy.IMPUTED_VALUE_KIND
    )
    assert imputed.any()
    assert not mixed_valid.loc[imputed, "donor_ability_available"].astype(bool).any()
    assert mixed_valid.loc[imputed, "donor_ability_sg"].eq(0.0).all()
    assert set(mixed_valid.loc[imputed, "donor_skill_date"].isna()) == {False, True}
    assert mixed_valid.loc[imputed, "donor_ability_original_available"].eq(
        mixed_valid.loc[imputed, "donor_ability_available"]
    ).all()

    readiness = v4.policy_v2_anchor_readiness(
        mixed, **_kwargs(mixed, reward_sha)
    )
    assert not readiness.anchor_ready
    assert readiness.availability_preserved
    assert readiness.imputed_rows == readiness.unavailable_rows
    assert readiness.anchor_api_explicit_policy_support is False
    assert readiness.blockers == (
        "anchor_api_lacks_explicit_frozen_policy_v2_neutral_zero_support",
    )
    with pytest.raises(v4.PolicyV2ContractUnavailable, match="anchor_api_lacks"):
        v4.assert_policy_v2_anchor_ready(
            mixed, **_kwargs(mixed, reward_sha)
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("donor_ability_policy_sha256", "f" * 64, "exact frozen policy-v2"),
        ("donor_ability_policy_source_sha256", "e" * 64, "exact frozen policy-v2"),
        ("donor_ability_policy_contract", "forged-policy-v2", "exact frozen policy-v2"),
    ],
)
def test_forged_policy_identity_or_source_is_rejected_before_resealing(
    policy_v2_contracts: dict[str, object],
    column: str,
    value: object,
    message: str,
) -> None:
    forged = policy_v2_contracts["mixed"].copy(deep=True)
    forged[column] = value
    with pytest.raises(v4.PolicyV2ContractUnavailable, match=message):
        v4.seal_policy_v2_contracts(forged)


def test_zero_without_exact_policy_row_attestation_is_rejected(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    identity = forged[
        ["donor_event_key", "donor_player_id", "category"]
    ].drop_duplicates().iloc[0]
    rows = _identity_mask(forged, identity)
    forged.loc[rows, "donor_ability_available"] = False
    forged.loc[rows, "donor_ability_original_available"] = False
    forged.loc[rows, "donor_ability_raw_strict_prior"] = None
    forged.loc[rows, "donor_ability_value_kind"] = donor_policy.IMPUTED_VALUE_KIND
    forged.loc[rows, "donor_ability_sg"] = 0.0
    forged.loc[rows, "donor_ability_was_clipped"] = False
    forged.loc[rows, "donor_ability_snapshot_kind"] = v4.IMPUTED_SNAPSHOT_KIND
    # The observed policy row attestation is intentionally retained.
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="row attestation failed|values or hashes changed",
    ):
        v4.seal_policy_v2_contracts(forged)


@pytest.mark.parametrize(
    ("current_value", "original_value"),
    [
        (None, "2024-01-02T00:00:00Z"),
        ("2023-12-31T00:00:00Z", "2024-01-02T00:00:00Z"),
        ("2024-01-02T00:00:00Z", None),
        ("2024-01-02T00:00:00Z", "2023-12-31T00:00:00Z"),
    ],
)
def test_erased_or_invented_skill_dates_fail_original_date_binding(
    policy_v2_contracts: dict[str, object],
    current_value: object,
    original_value: object,
) -> None:
    forged = policy_v2_contracts["mixed"].copy(deep=True)
    identity = policy_v2_contracts["dated_identity"]
    rows = _identity_mask(forged, identity)
    forged.loc[rows, "donor_skill_date"] = current_value
    forged.loc[rows, "donor_ability_original_skill_date"] = original_value
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="erased or invented the original donor skill date",
    ):
        v4.seal_policy_v2_contracts(forged)


def test_recomputed_valid_date_change_still_fails_original_expected_source(
    policy_v2_contracts: dict[str, object],
) -> None:
    reward_sha = str(policy_v2_contracts["reward_sha"])
    original = policy_v2_contracts["mixed"]
    changed_base = policy_v2_contracts["mixed_base"].copy(deep=True)
    identity = policy_v2_contracts["dated_identity"]
    rows = _identity_mask(changed_base, identity)
    changed_base.loc[rows, "donor_skill_date"] = None
    changed_policy = v4.build_policy_rows_from_atoms(changed_base)
    changed = v4.attest_policy_v2_donor_rows(changed_base, changed_policy)

    kwargs = _kwargs(original, reward_sha)
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="Canonical policy-v2 donor source changed",
    ):
        v4.validate_policy_v2_atom_frame(changed, **kwargs)


def test_observed_imputed_mixing_for_one_donor_identity_is_rejected(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    identity = forged[
        ["donor_event_key", "donor_player_id", "category"]
    ].drop_duplicates().iloc[0]
    positions = list(forged.index[_identity_mask(forged, identity)])
    assert len(positions) > 1
    row = positions[0]
    forged.loc[row, "donor_ability_available"] = False
    forged.loc[row, "donor_ability_original_available"] = False
    forged.loc[row, "donor_ability_raw_strict_prior"] = None
    forged.loc[row, "donor_ability_value_kind"] = donor_policy.IMPUTED_VALUE_KIND
    forged.loc[row, "donor_ability_sg"] = 0.0
    forged.loc[row, "donor_ability_was_clipped"] = False
    forged.loc[row, "donor_ability_snapshot_kind"] = v4.IMPUTED_SNAPSHOT_KIND
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="mixes observed and imputed",
    ):
        v4.seal_policy_v2_contracts(forged)


def test_imputed_availability_cannot_be_silently_enabled(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["mixed"].copy(deep=True)
    imputed = forged["donor_ability_value_kind"].eq(
        donor_policy.IMPUTED_VALUE_KIND
    )
    forged.loc[imputed, "donor_ability_available"] = True
    forged.loc[imputed, "donor_ability_original_available"] = True
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="row attestation failed|imputation changed honest availability",
    ):
        v4.seal_policy_v2_contracts(forged)


def test_reseal_cannot_authorize_negative_hierarchy_mass(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott")][0]
    forged.loc[row, "base_hierarchy_mass"] = -abs(
        float(forged.loc[row, "base_hierarchy_mass"])
    )
    forged.loc[row, "quality_base_mass"] = (
        float(forged.loc[row, "base_hierarchy_mass"])
        * float(forged.loc[row, "spatial_model_weight"])
    )
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="positive hierarchy mass"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_resolved_hazard_on_nonpenalty_atom(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[
        forged["category"].eq("ott") & forged["penalty_strokes"].eq(0)
    ][0]
    forged.loc[row, "hazard_resolution_status"] = "resolved_complete_atom"
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="Non-penalty atoms"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_terminal_atom_with_live_next_support(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott") & forged["terminal"]][0]
    forged.loc[row, "next_support_key"] = forged.loc[
        ~forged["terminal"], "next_support_key"
    ].iloc[0]
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="Terminal atoms cannot point"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_nonterminal_atom_without_next_support(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott") & ~forged["terminal"]][0]
    forged.loc[row, "next_support_key"] = ""
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="Nonterminal atoms need a next support"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_mixed_categories_inside_one_support(
    policy_v2_contracts: dict[str, object],
) -> None:
    observed = policy_v2_contracts["observed"]
    target_support = observed.loc[observed["category"].eq("ott"), "support_key"].iloc[0]
    injected = observed.loc[observed["category"].eq("arg")].iloc[[0]].copy()
    injected["support_key"] = target_support
    forged = pd.concat([observed, injected], ignore_index=True)
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="does not have one category"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_app_terminal_mass_different_from_declaration(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    app_terminal = forged["category"].eq("app") & forged["terminal"]
    row = forged.index[app_terminal][0]
    forged.loc[row, "base_hierarchy_mass"] = (
        float(forged.loc[row, "base_hierarchy_mass"]) * 1.5
    )
    forged.loc[row, "quality_base_mass"] = (
        float(forged.loc[row, "base_hierarchy_mass"])
        * float(forged.loc[row, "spatial_model_weight"])
    )
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="actual base holeout mass"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_forged_app_holeout_representation(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("app")][0]
    forged.loc[row, "holeout_representation"] = "forged_side_channel"
    with pytest.raises(
        v4.PolicyV2ContractUnavailable,
        match="APP support does not declare explicit holeout atoms",
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_atom_pointing_to_missing_support(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott") & ~forged["terminal"]][0]
    forged.loc[row, "next_support_key"] = "missing-support"
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="points to missing support"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_end_state_mismatching_next_support(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott") & ~forged["terminal"]][0]
    end_state = str(forged.loc[row, "end_state_key"])
    support_states = forged.groupby("support_key")["start_state_key"].first()
    alternative = support_states.index[support_states.ne(end_state)][0]
    forged.loc[row, "next_support_key"] = alternative
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="end state does not match"
    ):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_reseal_cannot_authorize_support_cycle(
    policy_v2_contracts: dict[str, object],
) -> None:
    forged = policy_v2_contracts["observed"].copy(deep=True)
    row = forged.index[forged["category"].eq("ott") & ~forged["terminal"]][0]
    support_key = str(forged.loc[row, "support_key"])
    start_state = str(forged.loc[row, "start_state_key"])
    start_reference = float(forged.loc[row, "start_reference_value"])
    stroke_cost = float(forged.loc[row, "stroke_cost"])
    forged.loc[row, "next_support_key"] = support_key
    forged.loc[row, "end_state_key"] = start_state
    forged.loc[row, "end_reference_value"] = start_reference
    forged.loc[row, "fixed_reference_reward"] = -stroke_cost
    with pytest.raises(v4.PolicyV2ContractUnavailable, match="contains a cycle"):
        v4.seal_policy_v2_contracts(_reseal_physical_and_app(forged))


def test_external_inventory_digest_rejects_a_fully_valid_resealed_inventory_change(
    policy_v2_contracts: dict[str, object],
) -> None:
    reward_sha = str(policy_v2_contracts["reward_sha"])
    original = policy_v2_contracts["observed"]
    changed = original.copy(deep=True)
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
        != original["support_inventory_sha256"].iloc[0]
    )
    with pytest.raises(
        v4.PolicyV2ContractUnavailable, match="Canonical full support inventory changed"
    ):
        v4.validate_policy_v2_atom_frame(
            changed, **_kwargs(original, reward_sha)
        )


def test_readiness_is_stable_and_json_serializable(
    policy_v2_contracts: dict[str, object],
) -> None:
    mixed = policy_v2_contracts["mixed"]
    reward_sha = str(policy_v2_contracts["reward_sha"])
    first = v4.policy_v2_anchor_readiness(mixed, **_kwargs(mixed, reward_sha))
    second = v4.policy_v2_anchor_readiness(mixed, **_kwargs(mixed, reward_sha))
    assert asdict(first) == asdict(second)
    assert first.policy_identity_sha256 == v4.FROZEN_POLICY_IDENTITY_SHA256


def test_frozen_policy_runtime_hashes_match_sealed_v2() -> None:
    runtime = v4.validate_frozen_policy_runtime()
    assert runtime == {
        "contract": v4.FROZEN_POLICY_CONTRACT,
        "status": "policy_evidence_only_not_real_chain_ready",
        "identity_sha256": (
            "f30e0161a1767d42f01dc0a33f617d93327286e3f49c59287c9b751686ba5bc7"
        ),
        "source_descriptor": "archive/hole_sim_donor_ability_policy.py",
        "source_sha256": (
            "3f9593333a79ffde258ec1efb149294bd1b036721c30cfcf5d47cf5ea2aa7339"
        ),
    }

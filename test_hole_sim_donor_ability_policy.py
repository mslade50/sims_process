from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from archive import hole_sim_donor_ability_policy as policy


EXPECTED_V1_MANIFEST_SHA256 = (
    "9481b51931395fa4740a7c98abdfd337e6a21035f6c3185586ace11a150dfe6c"
)
EXPECTED_V1_OUTPUT_INDEX_SHA256 = (
    "9b0e4967b051af3dcb83a0b4c799bf2819c851f22f7acdd1a568d913ac5573d9"
)
EXPECTED_V2_MANIFEST_SHA256 = (
    "315639db6cf633b6a8ee2d938b54bd7737900dd34abc5f43a27277c4904ca287"
)
EXPECTED_V2_OUTPUT_INDEX_SHA256 = (
    "c34be930e3fbff61ad8237c9233a4e1f1db6e1c475b6e7a048cd3c1b9d12497b"
)


def _raw() -> pd.DataFrame:
    return policy.synthetic_bridge_fixture()


def _row(frame: pd.DataFrame, row_id: str) -> pd.Series:
    return frame.loc[frame["bridge_row_id"].eq(row_id)].iloc[0]


def test_policy_identity_binds_mapping_missingness_clip_and_evidence() -> None:
    payload = policy.policy_payload()
    policy.validate_policy_payload(payload)
    identity = payload["policy"]
    assert identity["category_ability_mapping"] == {
        "ott": "sg_ott",
        "app": "sg_app",
        "arg": "ag_arg",
        "putt": "ag_putt",
    }
    assert identity["observed_value"]["clip_min"] == -2.0
    assert identity["observed_value"]["clip_max"] == 2.0
    assert identity["imputed_value"]["applied_value"] == 0.0
    assert identity["imputed_value"]["skill_date"] == (
        "preserve_actual_nullable_value"
    )
    assert identity["imputed_value"]["null_skill_date_allowed"] is True
    assert identity["imputed_value"]["nonnull_skill_date_cutoff"] == (
        "donor_skill_date < donor_event_start"
    )
    assert identity["identifier_admission"]["required"] == [
        "donor_event_key",
        "donor_player_id",
        "target_event_key",
    ]
    assert identity["beta_zero_identity"]["donor_weight_multiplier"] == 1.0
    assert identity["bound_evidence_sha256"] == {
        "evaluator_source": policy.EVALUATOR_SOURCE_SHA256,
        "evaluator_manifest": policy.EVALUATOR_MANIFEST_SHA256,
        "complete_event_selection_manifest": (
            policy.COMPLETE_EVENT_SELECTION_MANIFEST_SHA256
        ),
    }


def test_bound_evidence_matches_the_three_frozen_inputs() -> None:
    evidence = policy.validate_bound_evidence()
    assert evidence["evaluator_source"]["sha256"] == policy.EVALUATOR_SOURCE_SHA256
    assert evidence["evaluator_manifest"]["sha256"] == policy.EVALUATOR_MANIFEST_SHA256
    assert evidence["complete_event_selection_manifest"]["sha256"] == (
        policy.COMPLETE_EVENT_SELECTION_MANIFEST_SHA256
    )


def test_observed_values_use_raw_strict_prior_ability_clipped_to_two() -> None:
    output = policy.apply_frozen_policy(_raw())

    inside = _row(output, "observed_inside_clip")
    assert bool(inside["donor_ability_available"]) is True
    assert inside["donor_ability_value_kind"] == policy.OBSERVED_VALUE_KIND
    assert inside["donor_ability_raw_strict_prior"] == pytest.approx(1.25)
    assert inside["donor_ability_sg"] == pytest.approx(1.25)
    assert bool(inside["donor_ability_was_clipped"]) is False

    high = _row(output, "observed_clip_high")
    assert high["donor_ability_raw_strict_prior"] == pytest.approx(3.75)
    assert high["donor_ability_sg"] == pytest.approx(2.0)
    assert bool(high["donor_ability_was_clipped"]) is True

    low = _row(output, "observed_clip_low")
    assert low["donor_ability_raw_strict_prior"] == pytest.approx(-2.75)
    assert low["donor_ability_sg"] == pytest.approx(-2.0)
    assert bool(low["donor_ability_was_clipped"]) is True
    policy.validate_applied_frame(output)


def test_missing_values_preserve_true_nullable_date_and_use_exactly_zero() -> None:
    output = policy.apply_frozen_policy(_raw())
    dated = _row(output, "missing_dated_neutral_zero")
    assert bool(dated["donor_ability_available"]) is False
    assert dated["donor_ability_value_kind"] == policy.IMPUTED_VALUE_KIND
    assert dated["donor_skill_date"] == "2023-08-23T00:00:00Z"
    assert pd.isna(dated["donor_ability_raw_strict_prior"])
    assert dated["donor_ability_sg"] == 0.0
    assert bool(dated["donor_ability_was_clipped"]) is False

    undated = _row(output, "missing_undated_neutral_zero")
    assert bool(undated["donor_ability_available"]) is False
    assert undated["donor_ability_value_kind"] == policy.IMPUTED_VALUE_KIND
    assert pd.isna(undated["donor_skill_date"])
    assert pd.isna(undated["donor_ability_raw_strict_prior"])
    assert undated["donor_ability_sg"] == 0.0
    assert bool(undated["donor_ability_was_clipped"]) is False


def test_nullable_skill_date_is_bound_into_row_provenance() -> None:
    dated = _raw().loc[lambda value: value["bridge_row_id"].eq(
        "missing_dated_neutral_zero"
    )].copy()
    dated.loc[:, "bridge_row_id"] = "same_identity"
    undated = dated.copy()
    undated.loc[:, "donor_skill_date"] = None
    dated_hash = policy.apply_frozen_policy(dated).iloc[0][
        "donor_ability_row_sha256"
    ]
    undated_hash = policy.apply_frozen_policy(undated).iloc[0][
        "donor_ability_row_sha256"
    ]
    assert dated_hash != undated_hash


def test_applied_validation_rejects_erased_or_invented_missing_skill_dates() -> None:
    output = policy.apply_frozen_policy(_raw())
    dated_index = output.index[
        output["bridge_row_id"].eq("missing_dated_neutral_zero")
    ][0]
    erased = output.copy()
    erased.loc[dated_index, "donor_skill_date"] = None
    with pytest.raises(policy.DonorAbilityPolicyError, match="values or hashes changed"):
        policy.validate_applied_frame(erased)

    undated_index = output.index[
        output["bridge_row_id"].eq("missing_undated_neutral_zero")
    ][0]
    invented = output.copy()
    invented.loc[undated_index, "donor_skill_date"] = "2024-08-21T00:00:00Z"
    with pytest.raises(policy.DonorAbilityPolicyError, match="values or hashes changed"):
        policy.validate_applied_frame(invented)


@pytest.mark.parametrize(
    ("availability", "raw_value", "skill_date", "message"),
    [
        (True, None, "2023-01-01T00:00:00Z", "undeclared missing"),
        (True, float("nan"), "2023-01-01T00:00:00Z", "undeclared missing"),
        (True, 0.25, None, "true skill date"),
        (False, 0.0, None, "marks a present raw ability unavailable"),
        (
            False,
            None,
            "2023-01-01T00:00:00Z",
            "dated missing ability requires",
        ),
    ],
)
def test_partial_or_undeclared_missingness_fails_closed(
    availability: bool,
    raw_value: object,
    skill_date: object,
    message: str,
) -> None:
    frame = _raw().iloc[[0]].copy()
    frame.loc[:, "donor_ability_available"] = availability
    frame["donor_ability_raw_strict_prior"] = pd.Series(
        [raw_value], index=frame.index, dtype="object"
    )
    frame.loc[:, "donor_skill_date"] = skill_date
    with pytest.raises(policy.DonorAbilityPolicyError, match=message):
        policy.apply_frozen_policy(frame)


def test_availability_must_be_an_explicit_boolean() -> None:
    frame = _raw().iloc[[0]].copy()
    frame["donor_ability_available"] = "True"
    with pytest.raises(policy.DonorAbilityPolicyError, match="explicit boolean"):
        policy.apply_frozen_policy(frame)


@pytest.mark.parametrize(
    "identifier_column",
    ["donor_event_key", "donor_player_id", "target_event_key"],
)
@pytest.mark.parametrize("bad_value", [None, float("nan"), 123, "", "   "])
def test_required_identifiers_reject_missing_or_blank_nontext_values(
    identifier_column: str,
    bad_value: object,
) -> None:
    frame = _raw().iloc[[0]].copy()
    frame[identifier_column] = pd.Series(
        [bad_value], index=frame.index, dtype="object"
    )
    with pytest.raises(policy.DonorAbilityPolicyError, match=identifier_column):
        policy.apply_frozen_policy(frame)


@pytest.mark.parametrize(
    "identifier_column",
    ["donor_event_key", "donor_player_id", "target_event_key"],
)
def test_required_identifiers_must_arrive_in_canonical_form(
    identifier_column: str,
) -> None:
    frame = _raw().iloc[[0]].copy()
    original = str(frame.iloc[0][identifier_column])
    frame.loc[:, identifier_column] = f" {original}"
    with pytest.raises(policy.DonorAbilityPolicyError, match="already be canonical"):
        policy.apply_frozen_policy(frame)

    frame = _raw().iloc[[0]].copy()
    frame.loc[:, identifier_column] = "１２３"
    with pytest.raises(policy.DonorAbilityPolicyError, match="already be canonical"):
        policy.apply_frozen_policy(frame)


@pytest.mark.parametrize(
    ("skill_date", "donor_start", "target_start", "message"),
    [
        (
            "2023-02-02T00:00:00Z",
            "2023-02-02T00:00:00Z",
            "2023-03-02T00:00:00Z",
            "donor_skill_date < donor_event_start",
        ),
        (
            "2023-02-03T00:00:00Z",
            "2023-02-02T00:00:00Z",
            "2023-03-02T00:00:00Z",
            "donor_skill_date < donor_event_start",
        ),
        (
            "2023-01-02T00:00:00Z",
            "2023-03-02T00:00:00Z",
            "2023-03-02T00:00:00Z",
            "donor_event_start < target_event_start",
        ),
        (
            "2023-01-02T00:00:00Z",
            "2023-04-02T00:00:00Z",
            "2023-03-02T00:00:00Z",
            "donor_event_start < target_event_start",
        ),
    ],
)
def test_same_date_or_future_evidence_fails_closed(
    skill_date: str,
    donor_start: str,
    target_start: str,
    message: str,
) -> None:
    frame = _raw().iloc[[0]].copy()
    frame.loc[:, "donor_skill_date"] = skill_date
    frame.loc[:, "donor_event_start"] = donor_start
    frame.loc[:, "target_event_start"] = target_start
    with pytest.raises(policy.DonorAbilityPolicyError, match=message):
        policy.apply_frozen_policy(frame)


@pytest.mark.parametrize(
    "skill_date",
    [
        "2023-08-24T00:00:00Z",
        "2023-08-25T00:00:00Z",
    ],
)
def test_dated_missing_same_date_or_future_fails_closed(skill_date: str) -> None:
    frame = _raw().loc[lambda value: value["bridge_row_id"].eq(
        "missing_dated_neutral_zero"
    )].copy()
    frame.loc[:, "donor_skill_date"] = skill_date
    with pytest.raises(policy.DonorAbilityPolicyError, match="dated missing ability"):
        policy.apply_frozen_policy(frame)


def test_blank_missing_values_are_not_treated_as_declared_nulls() -> None:
    frame = _raw().loc[lambda value: value["bridge_row_id"].eq(
        "missing_undated_neutral_zero"
    )].copy()
    frame.loc[:, "donor_skill_date"] = " "
    with pytest.raises(policy.DonorAbilityPolicyError, match="valid timestamp"):
        policy.apply_frozen_policy(frame)

    frame = _raw().loc[lambda value: value["bridge_row_id"].eq(
        "missing_undated_neutral_zero"
    )].copy()
    frame["donor_ability_raw_strict_prior"] = pd.Series(
        [""], index=frame.index, dtype="object"
    )
    with pytest.raises(
        policy.DonorAbilityPolicyError,
        match="marks a present raw ability unavailable",
    ):
        policy.apply_frozen_policy(frame)


def test_wrong_category_mapping_fails_closed() -> None:
    frame = _raw().iloc[[0]].copy()
    frame.loc[:, "donor_ability_source_column"] = "sg_app"
    with pytest.raises(policy.DonorAbilityPolicyError, match="category mapping"):
        policy.apply_frozen_policy(frame)


def test_changed_policy_identity_fails_closed() -> None:
    changed = copy.deepcopy(policy.policy_payload())
    changed["policy"]["imputed_value"]["applied_value"] = 0.01
    with pytest.raises(policy.DonorAbilityPolicyError, match="differs"):
        policy.validate_policy_payload(changed)


def test_wrong_bound_source_hash_fails_closed(tmp_path: Path) -> None:
    changed_source = tmp_path / "evaluate_shot_quality_tilt.py"
    changed_source.write_bytes(policy.EVALUATOR_SOURCE_PATH.read_bytes() + b"\n# changed\n")
    with pytest.raises(policy.DonorAbilityPolicyError, match="wrong evaluator source"):
        policy.validate_bound_evidence(evaluator_source=changed_source)


def test_wrong_bound_manifest_hashes_fail_closed(tmp_path: Path) -> None:
    changed_evaluator = tmp_path / "evaluator_manifest.json"
    changed_evaluator.write_bytes(policy.EVALUATOR_MANIFEST_PATH.read_bytes() + b"\n")
    with pytest.raises(policy.DonorAbilityPolicyError, match="wrong evaluator manifest"):
        policy.validate_bound_evidence(evaluator_manifest=changed_evaluator)

    changed_selection = tmp_path / "selection_manifest.json"
    changed_selection.write_bytes(
        policy.COMPLETE_EVENT_SELECTION_MANIFEST_PATH.read_bytes() + b"\n"
    )
    with pytest.raises(
        policy.DonorAbilityPolicyError,
        match="wrong complete-event selection manifest",
    ):
        policy.validate_bound_evidence(
            complete_event_selection_manifest=changed_selection
        )


def test_nonzero_imputation_and_other_changed_outputs_fail_closed() -> None:
    output = policy.apply_frozen_policy(_raw())
    imputed_index = output.index[
        output["donor_ability_value_kind"].eq(policy.IMPUTED_VALUE_KIND)
    ][0]
    changed = output.copy()
    changed.loc[imputed_index, "donor_ability_sg"] = 0.01
    with pytest.raises(policy.DonorAbilityPolicyError, match="nonzero"):
        policy.validate_applied_frame(changed)

    changed = output.copy()
    changed.loc[0, "donor_ability_sg"] = 1.24
    with pytest.raises(policy.DonorAbilityPolicyError, match="values or hashes changed"):
        policy.validate_applied_frame(changed)

    changed = output.copy()
    changed.loc[0, "donor_ability_policy_sha256"] = "0" * 64
    with pytest.raises(policy.DonorAbilityPolicyError, match="values or hashes changed"):
        policy.validate_applied_frame(changed)


def test_beta_zero_is_an_exact_identity_for_both_value_kinds() -> None:
    output = policy.apply_frozen_policy(_raw())
    for ability in output["donor_ability_sg"]:
        assert policy.donor_weight_multiplier(0.0, ability, 3.0) == 1.0
        assert policy.donor_weight_multiplier(-0.0, ability, -2.5) == 1.0
    assert policy.donor_weight_multiplier(0.5, 1.0, 2.0) == pytest.approx(
        0.36787944117144233
    )
    with pytest.raises(policy.DonorAbilityPolicyError, match="outside"):
        policy.donor_weight_multiplier(0.0, 2.01, 1.0)


def test_bounded_artifact_build_validates_and_detects_tampering(tmp_path: Path) -> None:
    output_dir = tmp_path / "policy_artifact"
    summary = policy.build_sealed_artifact(output_dir)
    assert summary["status"] == policy.POLICY_STATUS
    assert summary["rows"] == 5
    assert summary["artifact_start_equals_end"] is True
    assert summary["bound_evidence_start_equals_end"] is True
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        policy.SEALED_ARTIFACT_FILES
    )
    assert policy.validate_sealed_artifact(output_dir) == summary

    evaluated_path = output_dir / "bridge_fixture_evaluated.csv"
    evaluated_path.write_bytes(evaluated_path.read_bytes() + b"\n")
    with pytest.raises(policy.DonorAbilityPolicyError, match="changed after sealing"):
        policy.validate_sealed_artifact(output_dir)


@pytest.mark.parametrize("intruder_kind", ["extra_file", "nested_directory"])
def test_seal_rejects_exhaustive_output_closure_violations(
    tmp_path: Path,
    intruder_kind: str,
) -> None:
    output_dir = tmp_path / intruder_kind
    policy.build_sealed_artifact(output_dir)
    if intruder_kind == "extra_file":
        (output_dir / "undeclared.txt").write_text("undeclared\n", encoding="utf-8")
    else:
        (output_dir / "undeclared_nested").mkdir()
    with pytest.raises(
        policy.DonorAbilityPolicyError,
        match="recursive output closure changed",
    ):
        policy.validate_sealed_artifact(output_dir)


def test_builder_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "already_exists"
    output_dir.mkdir()
    with pytest.raises(policy.DonorAbilityPolicyError, match="refusing to overwrite"):
        policy.build_sealed_artifact(output_dir)


def test_v1_sealed_artifact_bytes_are_preserved() -> None:
    assert policy.sha256_file(policy.V1_ARTIFACT_DIR / "manifest.json") == (
        EXPECTED_V1_MANIFEST_SHA256
    )
    assert policy.sha256_file(policy.V1_ARTIFACT_DIR / "OUTPUT_SHA256.json") == (
        EXPECTED_V1_OUTPUT_INDEX_SHA256
    )


def test_repository_v2_sealed_artifact_is_frozen() -> None:
    summary = policy.validate_sealed_artifact(policy.DEFAULT_ARTIFACT_DIR)
    assert summary["manifest_sha256"] == EXPECTED_V2_MANIFEST_SHA256
    assert summary["output_index_sha256"] == EXPECTED_V2_OUTPUT_INDEX_SHA256
    assert summary["rows"] == 5
    assert summary["status"] == "policy_evidence_only_not_real_chain_ready"

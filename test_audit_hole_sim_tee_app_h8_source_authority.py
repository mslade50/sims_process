from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import audit_hole_sim_tee_app_h8_source_authority as audit


def _candidate() -> dict[str, object]:
    return json.loads(
        (audit.CANDIDATE / "authority_candidate.json").read_text(encoding="utf-8")
    )


def test_fixed_scope_and_non_authorizing_status_are_exact() -> None:
    assert audit.CANDIDATE.name == (
        "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_v1"
    )
    assert audit.OUTPUT.name == (
        "hole_sim_tee_app_source_authority_lowry_r1h8_raw_app_candidate_audit_v1"
    )
    assert audit.EXPECTED_BUILDER_SHA256 == (
        "809c163db737bc4d628101d4f4687543863ffd7a995f5ee9ece04d5d88c47182"
    )
    assert audit.EXPECTED_SEAM_SHA256 == (
        "70c41fdb11aa2ae369caa474cfeb9c8fab423d740acd3d19f81c35a3d7a1da83"
    )
    assert audit.STATUS == "review_evidence_only_not_authority"


def test_cli_has_only_default_build_or_validate_fixed_output() -> None:
    parser = audit.build_parser()
    assert vars(parser.parse_args([])) == {"validate_only": False}
    assert vars(parser.parse_args(["--validate-only"])) == {"validate_only": True}
    for forbidden in (
        ["--output-dir", "elsewhere"],
        ["--candidate", "elsewhere"],
        ["--digest", "0" * 64],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(forbidden)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_ready", True),
        ("production_enabled", True),
        ("authority_id", "candidate-derived"),
        ("review_lock_sha256", "d" * 64),
        ("candidate_digest_is_not_authority", False),
        ("admission_status", "admitted"),
    ],
)
def test_candidate_status_mutations_fail_closed(field: str, value: object) -> None:
    candidate = _candidate()
    audit._assert_non_authorizing(candidate)
    candidate[field] = value
    with pytest.raises(audit.AuditError, match="fail-closed"):
        audit._assert_non_authorizing(candidate)


def test_candidate_and_query_identities_are_independently_reproduced() -> None:
    candidate = _candidate()
    query = candidate["query"]
    assert isinstance(query, dict)
    assert audit._candidate_id(candidate) == audit.EXPECTED_CANDIDATE_ID
    assert audit._query_identities(query, candidate) == {
        "query_context_sha256": audit.EXPECTED_QUERY_CONTEXT_SHA256,
        "query_key": audit.EXPECTED_QUERY_KEY,
    }


def test_ordered_inventory_hasher_rejects_schema_and_order_mutations() -> None:
    frame = pd.DataFrame(
        {
            "inventory_record_order": np.arange(3, dtype=np.int64),
            "value": ["alpha", "beta", "gamma"],
        }
    )
    digest = audit._ordered_frame_hash(frame, "synthetic-v1", ("value",))
    mutated = frame.copy()
    mutated.loc[1, "value"] = "changed"
    assert audit._ordered_frame_hash(mutated, "synthetic-v1", ("value",)) != digest
    reordered = frame.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(audit.AuditError, match="not contiguous"):
        audit._ordered_frame_hash(reordered, "synthetic-v1", ("value",))
    with pytest.raises(audit.AuditError, match="schema"):
        audit._ordered_frame_hash(frame[["value", "inventory_record_order"]], "synthetic-v1", ("value",))


def test_policy_replay_matches_all_904_rows_exactly() -> None:
    raw = pd.read_parquet(
        audit.CANDIDATE / "parents" / "h8_proposal_support_atoms.parquet"
    )
    stored = pd.read_parquet(audit.CANDIDATE / "proposal_policy_v2.parquet")
    replayed = audit._independent_policy_expected(raw)
    pd.testing.assert_frame_equal(
        replayed,
        stored,
        check_dtype=False,
        check_exact=True,
        check_categorical=False,
    )
    assert len(replayed) == 904
    assert int(replayed["donor_ability_available"].astype(bool).sum()) == 853
    assert int((~replayed["donor_ability_available"].astype(bool)).sum()) == 51
    assert int(replayed["donor_ability_was_clipped"].astype(bool).sum()) == 1


def test_auditor_does_not_import_builder_seam_or_run_pipeline() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not any(
        name.endswith("hole_sim_tee_app_h8_source_authority")
        or name.endswith("hole_sim_tee_app_complete_atom_support")
        for name in imports
    )
    assert calls.isdisjoint(
        {
            "prepare_streaming_event",
            "CompleteAtomExpansion",
            "expand_complete_atoms",
            "run_simulation",
            "simulate",
        }
    )


def test_sealed_artifact_validates_when_present() -> None:
    if not audit.OUTPUT.exists():
        pytest.skip("fixed fresh audit artifact has not been built yet")
    result = audit.validate_audit_artifact(deep=False)
    sealed = result["audit"]
    assert sealed["authority_ready"] is False
    assert sealed["production_enabled"] is False
    assert sealed["audit_digest_is_not_authority"] is True
    assert sealed["candidate_non_authorizing_exact"] is True

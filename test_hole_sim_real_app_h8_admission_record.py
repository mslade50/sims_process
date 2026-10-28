from __future__ import annotations

from hashlib import sha256
import inspect
from pathlib import Path
import sys

import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_real_app_h8_admission_session as admission


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_record_review_slot_is_fixed_and_admission_has_no_inputs() -> None:
    assert admission._RECORD_TEST_RELATIVE == Path(__file__).name
    assert len(inspect.signature(admission.admit_lowry_r1h8_app_support).parameters) == 0


def test_current_or_future_record_binds_both_review_tests() -> None:
    manifest = admission.real_app_h8_authority_manifest()
    if manifest["authority_record_present"] is False:
        assert manifest["records"] == []
        return

    assert manifest["authority_record_present"] is True
    assert len(manifest["records"]) == 1
    record = manifest["records"][0]
    logic = record["logic"]
    sources = record["review_sources"]
    dormant_test = Path(admission.__file__).resolve().parent.parent / admission._TEST_RELATIVE
    record_test = Path(__file__).resolve()
    assert logic["reviewed_dormant_test_sha256"] == _sha(dormant_test)
    assert logic["reviewed_record_test_sha256"] == _sha(record_test)
    assert sources[admission._TEST_RELATIVE] == logic["reviewed_dormant_test_sha256"]
    assert sources[admission._RECORD_TEST_RELATIVE] == logic[
        "reviewed_record_test_sha256"
    ]


def test_current_or_future_source_projects_to_the_reviewed_dormant_bytes() -> None:
    raw = Path(admission.__file__).read_bytes()
    projected = admission._project_dormant_source(raw)
    manifest = admission.real_app_h8_authority_manifest()
    if manifest["authority_record_present"] is False:
        assert projected == raw
        return

    record = manifest["records"][0]
    assert sha256(projected).hexdigest() == record["logic"][
        "reviewed_dormant_wrapper_sha256"
    ]
    assert projected != raw


def test_authority_record_never_enables_out_of_scope_capabilities() -> None:
    manifest = admission.real_app_h8_authority_manifest()
    for field in (
        "callback_enabled",
        "chain_ready",
        "anchor_ready",
        "mean_solve_launched",
        "simulation_launched",
        "production_enabled",
    ):
        assert manifest[field] is False
    assert manifest["research_only"] is True


def test_empty_registry_still_fails_closed() -> None:
    if admission.real_app_h8_authority_manifest()["authority_record_present"] is True:
        return
    with pytest.raises(
        admission.RealAppH8AdmissionUnavailable,
        match="No reviewed real-H8 selector authority",
    ):
        admission.admit_lowry_r1h8_app_support()

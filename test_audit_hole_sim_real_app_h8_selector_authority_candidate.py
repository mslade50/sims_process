from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


REPO = Path(__file__).resolve().parent
MODULE_PATH = (
    REPO
    / "archive"
    / "audit_hole_sim_real_app_h8_selector_authority_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("real_app_h8_candidate_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _reseal_candidate(root: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("candidate_id", None)
    payload["candidate_id"] = audit._candidate_id(body)
    _write_json(root / "authority_candidate.json", payload)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["candidate_id"] = payload["candidate_id"]
    manifest["payload_files"] = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": audit._sha256_file(root / name),
        }
        for name in ("REPORT.md", "authority_candidate.json")
    }
    _write_json(root / "manifest.json", manifest)
    index = {
        "contract": audit.OUTPUT_INDEX_CONTRACT,
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": audit._sha256_file(root / name),
            }
            for name in ("REPORT.md", "authority_candidate.json", "manifest.json")
        },
    }
    _write_json(root / audit.OUTPUT_INDEX, index)


def _reseal_audit(root: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("audit_id", None)
    payload["audit_id"] = audit._audit_id(body)
    _write_json(root / "audit.json", payload)
    (root / "REPORT.md").write_text(
        audit._audit_report(payload),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "contract": audit.AUDIT_MANIFEST_CONTRACT,
        "audit_id": payload["audit_id"],
        "decision": payload["decision"],
        "audit_status": payload["audit_status"],
        "reviewed_candidate_id": payload["reviewed_candidate_id"],
        "candidate_artifact_review_passed": payload[
            "candidate_artifact_review_passed"
        ],
        "data_only_authority_record_ready": payload[
            "data_only_authority_record_ready"
        ],
        **{
            field: payload[field]
            for field in audit._FAIL_CLOSED_FLAGS
        },
        "research_only": payload["research_only"],
        "payload_files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": audit._sha256_file(root / name),
            }
            for name in ("REPORT.md", "audit.json")
        },
    }
    _write_json(root / "manifest.json", manifest)
    index = {
        "contract": audit.OUTPUT_INDEX_CONTRACT,
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": audit._sha256_file(root / name),
            }
            for name in ("REPORT.md", "audit.json", "manifest.json")
        },
    }
    _write_json(root / audit.OUTPUT_INDEX, index)


@pytest.fixture(scope="module")
def copied_candidate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("real-app-h8-audit-candidate") / "candidate"
    shutil.copytree(audit._FIXED_CANDIDATE_ROOT, root)
    return root


@pytest.fixture(scope="module")
def copied_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("real-app-h8-audit-source") / "source"
    shutil.copytree(audit._FIXED_SOURCE_ROOT, root)
    return root


@pytest.fixture(scope="module")
def real_evidence() -> dict[str, object]:
    return audit.collect_independent_audit()


@pytest.fixture(scope="module")
def built_temp_audit(
    tmp_path_factory: pytest.TempPathFactory,
    copied_candidate: Path,
    copied_source: Path,
    real_evidence: dict[str, object],
) -> Path:
    output = tmp_path_factory.mktemp("real-app-h8-built-audit") / "audit"
    patch = pytest.MonkeyPatch()
    patch.setattr(audit, "_FIXED_CANDIDATE_ROOT", copied_candidate)
    patch.setattr(audit, "_FIXED_SOURCE_ROOT", copied_source)
    patch.setattr(audit, "_FIXED_AUDIT_ROOT", output)
    patch.setattr(
        audit,
        "collect_independent_audit",
        lambda: copy.deepcopy(real_evidence),
    )
    try:
        assert audit.build_audit_artifact() == output
    finally:
        patch.undo()
    return output


def test_real_audit_output_has_not_been_built() -> None:
    assert not audit._FIXED_AUDIT_ROOT.exists()


def test_contract_has_fixed_paths_and_closed_cli() -> None:
    assert audit._FIXED_CANDIDATE_ROOT == (
        REPO
        / "permanent_data"
        / "hole_sim_real_app_h8_selector_authority_candidate_v1"
    )
    assert audit._FIXED_AUDIT_ROOT == (
        REPO
        / "permanent_data"
        / "hole_sim_real_app_h8_selector_authority_candidate_audit_v1"
    )
    parser = audit.build_parser()
    assert parser.parse_args(["--validate-only"]).validate_only is True
    assert parser.parse_args(["--audit"]).audit is True
    for args in (
        [],
        ["--validate-only", "--audit"],
        ["--audit", "--output", "elsewhere"],
        ["--validate-only", "--candidate", "elsewhere"],
        ["--audit", "--cap", "1"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_auditor_never_imports_or_calls_builder_selector_or_runtime() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not any(
        "hole_sim_real_app_h8_selector_authority_candidate" in name
        or "whole_atom_selector" in name
        for name in imported
    )
    assert not called.intersection(
        {
            "validate_candidate_artifact",
            "validate_fixed_source",
            "build_candidate",
            "weight_support",
            "select_support",
            "admit_support",
            "run_callback",
            "solve_mean",
            "run_simulation",
        }
    )


def test_exact_schema_projection_topology_and_fixed_pins() -> None:
    assert len(audit.EXPECTED_ARROW_SCHEMA) == 150
    assert audit.EXPECTED_ARROW_SCHEMA[0] == ("arm", "string")
    assert audit.EXPECTED_ARROW_SCHEMA[-1] == ("terminal", "bool")
    assert len(audit.SEMANTIC_SELECTOR_COLUMNS) == 54
    assert len(set(audit.SEMANTIC_SELECTOR_COLUMNS)) == 54
    assert len(audit.TOPOLOGY_COLUMNS) == 30
    assert audit.EXPECTED_CANDIDATE_ID == (
        "b5552aaea4e7c216b8bae8dc9ccdb68d122df54523d7b8faf1ddb70664dde942"
    )
    assert audit.EXPECTED_CANDIDATE_INDEX_SHA256 == (
        "ee2c086ee7b42a816f33d1894251e01eb4a58bc1d673b7a866491931fe553b43"
    )
    assert audit.EXPECTED_CANONICAL_FULL_SUPPORT_SHA256 == (
        "b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7"
    )
    assert audit.EXPECTED_CANONICAL_PROJECTION_SHA256 == (
        "a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889"
    )
    assert audit.EXPECTED_CANONICAL_SELECTOR_SUPPORT_SHA256 == (
        "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
    )
    assert audit.EXPECTED_CANONICAL_TOPOLOGY_SHA256 == (
        "9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484"
    )


def test_auditor_binds_its_own_focused_test() -> None:
    assert audit.EXPECTED_AUDITOR_TEST_SHA256 == audit._sha256_file(Path(__file__))


def test_fixed_candidate_closure_and_identity_reproduce_exactly(
    copied_candidate: Path,
) -> None:
    state = audit._candidate_tree_state(copied_candidate, enforce_fixed=True)
    assert state == {
        "files": 4,
        "bytes": 7_203,
        "file_state_sha256": (
            "b5c2781ace83f40e88c2a0837393398e9457ca6d50615a94e4d2f2fcfafb786f"
        ),
        "closure_sha256": (
            "9150acd26758cf4118c57a7ad9ba47a44af1cb266bbfd67ce173a3546271ba6b"
        ),
    }
    candidate = audit._validate_candidate_artifact(copied_candidate)
    assert candidate["candidate_id"] == audit.EXPECTED_CANDIDATE_ID


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authority_ready",), True),
        (("research_only",), 1),
        (("fixed_caps", "complete_atoms"), "35218"),
        (("fixed_input_identity", "outer_output_index_sha256"), "0" * 64),
        (("external_review_requirements", "candidate_id_is_not_authority"), False),
        (("validation", "canonical_projection_sha256"), "0" * 64),
        (("validation", "counts", "complete_atoms"), 0),
        (("source_sha256_start", "builder_source_sha256"), "0" * 64),
    ],
)
def test_fully_resealed_candidate_widening_is_rejected(
    copied_candidate: Path,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    root = tmp_path / "candidate"
    shutil.copytree(copied_candidate, root)
    payload = json.loads((root / "authority_candidate.json").read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _reseal_candidate(root, payload)
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit._assert_candidate(payload)
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit._validate_candidate_artifact(root)


def test_candidate_rejects_unindexed_addition(
    copied_candidate: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "candidate-extra"
    shutil.copytree(copied_candidate, root)
    (root / "extra.txt").write_text("not indexed\n", encoding="utf-8")
    with pytest.raises(audit.SelectorAuthorityCandidateError, match="closure"):
        audit._candidate_tree_state(root, enforce_fixed=True)


def test_original_support_rejects_extra_and_byte_mutation(
    copied_source: Path,
    tmp_path: Path,
) -> None:
    extra = tmp_path / "source-extra"
    shutil.copytree(copied_source, extra)
    (extra / "extra.txt").write_text("not indexed\n", encoding="utf-8")
    with pytest.raises(audit.SelectorAuthorityCandidateError, match="closure"):
        audit._artifact_closure_state(extra, enforce_fixed=True)

    changed = tmp_path / "source-byte-change"
    shutil.copytree(copied_source, changed)
    part = changed / "complete_atom_support" / "parts" / "part-000004.parquet"
    with part.open("r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(audit.SelectorAuthorityCandidateError, match="SHA-256"):
        audit._artifact_closure_state(changed, enforce_fixed=True)


def test_full_independent_rederivation_matches_all_counts_and_hashes(
    real_evidence: dict[str, object],
) -> None:
    assert real_evidence["candidate_sha256_start"] == real_evidence[
        "candidate_sha256_end"
    ]
    assert real_evidence["original_support_sha256_start"] == real_evidence[
        "original_support_sha256_end"
    ]
    assert real_evidence["builder_source_sha256_start"] == real_evidence[
        "builder_source_sha256_end"
    ]
    assert real_evidence["auditor_source_sha256_start"] == real_evidence[
        "auditor_source_sha256_end"
    ]
    assert real_evidence["independent_validation"] == audit._EXPECTED_VALIDATION


def test_candidate_tree_toctou_is_rejected(
    copied_candidate: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "candidate-toctou"
    shutil.copytree(copied_candidate, root)
    monkeypatch.setattr(audit, "_FIXED_CANDIDATE_ROOT", root)

    def mutate_during_validation(_root: Path) -> dict[str, object]:
        (root / "REPORT.md").write_text("changed\n", encoding="utf-8")
        return {
            "artifact_sha256_start": copy.deepcopy(audit._EXPECTED_ORIGINAL_SUMMARY),
            "artifact_sha256_end": copy.deepcopy(audit._EXPECTED_ORIGINAL_SUMMARY),
            "validation": copy.deepcopy(audit._EXPECTED_VALIDATION),
        }

    monkeypatch.setattr(
        audit,
        "_validation_summary",
        mutate_during_validation,
    )
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit.collect_independent_audit()


def test_temp_audit_is_flat_sealed_and_strictly_non_authorizing(
    built_temp_audit: Path,
) -> None:
    result = audit.validate_audit_artifact(built_temp_audit, deep=False)
    assert result["decision"] == audit.AUDIT_DECISION
    assert result["data_only_authority_record_ready"] is False
    assert result["authority_id"] is None
    assert result["authority_lock_sha256"] is None
    assert result["research_only"] is True
    assert result["candidate_artifact_review_passed"] is True
    assert result["original_support_review_passed"] is True
    assert result["next_review_requirements"] == audit._AUDIT_REQUIREMENTS
    assert all(result[field] is False for field in audit._FAIL_CLOSED_FLAGS)
    files, directories = audit._closed_tree(built_temp_audit)
    assert not directories
    assert set(files) == {"OUTPUT_SHA256.json", "REPORT.md", "audit.json", "manifest.json"}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authority_ready",), True),
        (("weighting_enabled",), 1),
        (("data_only_authority_record_ready",), True),
        (("next_review_requirements", "audit_id_is_not_authority"), False),
        (("next_review_requirements", "separate_data_only_authority_record_required"), False),
        (("independent_validation", "canonical_topology_sha256"), "0" * 64),
        (("original_support_sha256_end", "closure_sha256"), "0" * 64),
        (("auditor_source_sha256_end", "auditor_source_sha256"), "0" * 64),
    ],
)
def test_fully_resealed_audit_widening_is_rejected(
    built_temp_audit: Path,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    root = tmp_path / "audit"
    shutil.copytree(built_temp_audit, root)
    payload = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _reseal_audit(root, payload)
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit.validate_audit_artifact(root, deep=False)


def test_fully_resealed_report_rewording_is_rejected(
    built_temp_audit: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit-report"
    shutil.copytree(built_temp_audit, root)
    (root / "REPORT.md").write_text("GO for production.\n", encoding="utf-8")
    payload = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["payload_files"]["REPORT.md"] = {
        "bytes": (root / "REPORT.md").stat().st_size,
        "sha256": audit._sha256_file(root / "REPORT.md"),
    }
    _write_json(root / "manifest.json", manifest)
    index = json.loads((root / audit.OUTPUT_INDEX).read_text(encoding="utf-8"))
    index["files"]["REPORT.md"] = manifest["payload_files"]["REPORT.md"]
    index["files"]["manifest.json"] = {
        "bytes": (root / "manifest.json").stat().st_size,
        "sha256": audit._sha256_file(root / "manifest.json"),
    }
    _write_json(root / audit.OUTPUT_INDEX, index)
    with pytest.raises(audit.SelectorAuthorityCandidateError, match="report"):
        audit.validate_audit_artifact(root, deep=False)


def test_audit_tree_toctou_is_rejected(
    built_temp_audit: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "audit-toctou"
    shutil.copytree(built_temp_audit, root)
    payload = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    evidence_fields = (
        "candidate_sha256_start",
        "candidate_sha256_end",
        "original_support_sha256_start",
        "original_support_sha256_end",
        "builder_source_sha256_start",
        "builder_source_sha256_end",
        "auditor_source_sha256_start",
        "auditor_source_sha256_end",
        "independent_validation",
    )

    def mutate_during_deep_validation() -> dict[str, object]:
        (root / "REPORT.md").write_text("changed\n", encoding="utf-8")
        return {field: copy.deepcopy(payload[field]) for field in evidence_fields}

    monkeypatch.setattr(
        audit,
        "collect_independent_audit",
        mutate_during_deep_validation,
    )
    with pytest.raises(audit.SelectorAuthorityCandidateError, match="start/end"):
        audit.validate_audit_artifact(root, deep=True)


def test_build_refuses_preexisting_fixed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    monkeypatch.setattr(audit, "_FIXED_AUDIT_ROOT", output)
    with pytest.raises(FileExistsError, match="must be absent"):
        audit.build_audit_artifact()


def test_build_rejects_postwrite_candidate_change(
    copied_candidate: Path,
    copied_source: Path,
    real_evidence: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    shutil.copytree(copied_candidate, candidate_root)
    output = tmp_path / "audit"
    monkeypatch.setattr(audit, "_FIXED_CANDIDATE_ROOT", candidate_root)
    monkeypatch.setattr(audit, "_FIXED_SOURCE_ROOT", copied_source)
    monkeypatch.setattr(audit, "_FIXED_AUDIT_ROOT", output)
    monkeypatch.setattr(
        audit,
        "collect_independent_audit",
        lambda: copy.deepcopy(real_evidence),
    )
    original_write = audit._write_output_index

    def write_then_mutate(root: Path) -> None:
        original_write(root)
        (candidate_root / "REPORT.md").write_text("changed\n", encoding="utf-8")

    monkeypatch.setattr(audit, "_write_output_index", write_then_mutate)
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit.build_audit_artifact()


def test_build_rejects_postwrite_original_support_change(
    copied_candidate: Path,
    copied_source: Path,
    real_evidence: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    shutil.copytree(copied_source, source_root)
    output = tmp_path / "audit"
    monkeypatch.setattr(audit, "_FIXED_CANDIDATE_ROOT", copied_candidate)
    monkeypatch.setattr(audit, "_FIXED_SOURCE_ROOT", source_root)
    monkeypatch.setattr(audit, "_FIXED_AUDIT_ROOT", output)
    monkeypatch.setattr(
        audit,
        "collect_independent_audit",
        lambda: copy.deepcopy(real_evidence),
    )
    original_write = audit._write_output_index

    def write_then_mutate(root: Path) -> None:
        original_write(root)
        (source_root / "REPORT.md").write_text("changed\n", encoding="utf-8")

    monkeypatch.setattr(audit, "_write_output_index", write_then_mutate)
    with pytest.raises(audit.SelectorAuthorityCandidateError):
        audit.build_audit_artifact()

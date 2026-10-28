from __future__ import annotations

import ast
from difflib import SequenceMatcher
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest


REPO = Path(__file__).resolve().parent
MODULE_PATH = (
    REPO
    / "archive"
    / "audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
SPEC = importlib.util.spec_from_file_location("real_app_h8_candidate_audit_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _attestation() -> dict[str, object]:
    return {
        "contract": "fixed_one_read_launcher_attestation_v2",
        "launcher_relative_path": (
            "archive/run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
        ),
        "launcher_test_relative_path": (
            "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
        ),
        "launcher_sha256": _sha(audit.LAUNCHER_SOURCE),
        "launcher_test_sha256": _sha(audit.LAUNCHER_TEST),
    }


@pytest.fixture(scope="session")
def sealed() -> dict[str, object]:
    candidate_bytes, candidate_state = audit._capture_pinned(
        audit.FIXED_CANDIDATE, audit.CANDIDATE_FILES, (), "candidate"
    )
    original_bytes, original_state = audit._capture_pinned(
        audit.FIXED_ORIGINAL,
        audit.ORIGINAL_FILES,
        ("complete_atom_support", "complete_atom_support/parts"),
        "original",
    )
    candidate = audit._validate_candidate(candidate_bytes)
    buffers = audit._validate_original(original_bytes)
    frame, schema = audit._load_from_captures(buffers)
    vectors = audit._vector_checks(frame, schema)
    return {
        "candidate_bytes": candidate_bytes,
        "candidate_state": candidate_state,
        "candidate": candidate,
        "original_bytes": original_bytes,
        "original_state": original_state,
        "frame": frame,
        "schema": schema,
        "vectors": vectors,
    }


def _evidence(sealed: dict[str, object]) -> dict[str, object]:
    source = audit._source_state()
    launcher = audit._launcher_state(_attestation())
    return {
        "candidate_bytes_start": sealed["candidate_state"],
        "candidate_bytes_end": sealed["candidate_state"],
        "original_bytes_start": sealed["original_state"],
        "original_bytes_end": sealed["original_state"],
        "source_start": source,
        "source_end": source,
        "launcher_start": launcher,
        "launcher_end": launcher,
        "external_canonical_evidence": dict(audit.EXTERNAL_CANONICAL_EVIDENCE),
        "vectorized_invariants": sealed["vectors"],
    }


def _copy_capture(root: Path, captured: dict[str, bytes]) -> None:
    root.mkdir()
    for relative, raw in captured.items():
        target = root / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _reseal_output(root: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("audit_id", None)
    rebuilt = {"audit_id": audit._audit_id(body), **body}
    audit_raw = _write_json(root / "audit.json", rebuilt)
    report_raw = audit._report(rebuilt).encode()
    (root / "REPORT.md").write_bytes(report_raw)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["audit_id"] = rebuilt["audit_id"]
    for flag in audit.FAIL_CLOSED:
        manifest[flag] = rebuilt[flag]
    manifest["payload_files"] = {
        "REPORT.md": {"bytes": len(report_raw), "sha256": sha256(report_raw).hexdigest()},
        "audit.json": {"bytes": len(audit_raw), "sha256": sha256(audit_raw).hexdigest()},
    }
    manifest_raw = _write_json(root / "manifest.json", manifest)
    index = {
        "contract": audit.INDEX_CONTRACT,
        "files": {
            "REPORT.md": manifest["payload_files"]["REPORT.md"],
            "audit.json": manifest["payload_files"]["audit.json"],
            "manifest.json": {
                "bytes": len(manifest_raw),
                "sha256": sha256(manifest_raw).hexdigest(),
            },
        },
    }
    _write_json(root / "OUTPUT_SHA256.json", index)


def _build_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed: dict[str, object],
    name: str,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / name
    monkeypatch.setattr(audit, "FIXED_OUTPUT", root)
    monkeypatch.setattr(audit, "_collect", lambda _att: _evidence(sealed))
    result = audit._build(_attestation())
    return root, result


def test_fixed_inputs_scope_and_v1_preservation(sealed: dict[str, object]) -> None:
    assert audit.SCOPE == "independent_byte_and_vectorized_invariant_audit_v2"
    assert audit.FIXED_OUTPUT == (
        REPO
        / "permanent_data"
        / "hole_sim_real_app_h8_selector_authority_candidate_audit_v2"
    )
    assert not audit.FIXED_OUTPUT.exists()
    assert _sha(
        REPO / "archive" / "audit_hole_sim_real_app_h8_selector_authority_candidate.py"
    ) == audit.V1_DIAGNOSTIC_PIN
    assert sealed["candidate"]["candidate_id"] == audit.EXPECTED_CANDIDATE_ID


def test_initial_input_capture_reads_every_byte_image_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    reads: dict[Path, int] = {}

    def counted(path: Path) -> bytes:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    audit._capture_pinned(audit.FIXED_CANDIDATE, audit.CANDIDATE_FILES, (), "candidate")
    audit._capture_pinned(
        audit.FIXED_ORIGINAL,
        audit.ORIGINAL_FILES,
        ("complete_atom_support", "complete_atom_support/parts"),
        "original",
    )
    expected = {
        (audit.FIXED_CANDIDATE / Path(path)).resolve()
        for path in audit.CANDIDATE_FILES
    } | {
        (audit.FIXED_ORIGINAL / Path(path)).resolve()
        for path in audit.ORIGINAL_FILES
    }
    assert set(reads) == expected
    assert set(reads.values()) == {1}


def test_parquet_path_is_bufferreader_and_schema_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    sealed: dict[str, object],
) -> None:
    seen: list[object] = []
    original = audit.pq.ParquetFile

    def wrapped(source: object, *args: object, **kwargs: object) -> object:
        seen.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(audit.pq, "ParquetFile", wrapped)
    frame, schema = audit._load_from_captures(
        audit._validate_original(sealed["original_bytes"])
    )
    assert len(seen) == 5 and all(isinstance(item, pa.BufferReader) for item in seen)
    assert frame.shape == (35_218, 86)
    expected = tuple(
        (name, kind, True)
        for name, kind in (item.split("@", 1) for item in audit.EXPECTED_SCHEMA.split(";"))
    )
    assert audit._schema_tuple(schema) == expected
    assert schema.equals(sealed["schema"], check_metadata=True)


def test_explicit_ndjson_contracts_and_vectorized_counts(
    sealed: dict[str, object],
) -> None:
    vectors = sealed["vectors"]
    assert vectors["audit_only_semantic_root_sha256"] == audit.EXPECTED_SEMANTIC_ROOT
    assert vectors["ndjson_components"] == audit.EXPECTED_NDJSON
    assert vectors["counts"] == {
        "complete_atoms": 35_218,
        "proposals": 904,
        "spatial_proposals": 903,
        "terminal_atoms": 1,
        "atoms_per_spatial_proposal": 39,
        "surface_records": 903,
        "materialized_surface_branches": 5_418,
        "resolved_hazard_atoms": 30_702,
        "no_hazard_spatial_atoms": 4_515,
        "observed_ability_atoms": 33_229,
        "imputed_ability_atoms": 1_989,
        "observed_ability_proposals": 853,
        "imputed_ability_proposals": 51,
        "clipped_ability_atoms": 39,
        "donor_quality_groups": 6,
        "mixed_policy_groups": 4,
    }
    contract = audit.EXPECTED_NDJSON["schema"]["contract"]
    records = audit._schema_records(sealed["schema"])
    framed = contract.encode("ascii") + b"\n" + b"".join(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
        for record in records
    )
    assert len(framed) == 14_888
    assert sha256(framed).hexdigest() == audit.EXPECTED_NDJSON["schema"]["sha256"]


@pytest.mark.parametrize(
    "attack",
    [
        "row_order",
        "proposal_group_split",
        "surface_cdf",
        "hazard_order",
        "quality_factor",
        "spatial_weight",
        "imputed_ability",
        "clipped_flag",
        "ability_date",
        "reference_reward",
        "ability_record_split",
    ],
)
def test_vectorized_invariants_reject_hostile_atom_mutations(
    attack: str,
    sealed: dict[str, object],
) -> None:
    frame = sealed["frame"].copy(deep=True)
    if attack == "row_order":
        frame.loc[0, "complete_row_order"] = 1
    elif attack == "proposal_group_split":
        frame.loc[0, "donor_quality_group_id"] = next(iter(audit.GROUP_EXPECTATIONS))
    elif attack == "surface_cdf":
        row = frame.index[(~frame["terminal"]) & (frame["surface_class_index"] == 0)][0]
        frame.loc[row, "surface_cdf_upper"] += 0.01
    elif attack == "hazard_order":
        row = frame.index[frame["hazard_resolution_status"] == "resolved_complete_atom"][0]
        frame.loc[row, "hazard_donor_order"] = 33
    elif attack == "quality_factor":
        frame.loc[0, "quality_base_mass"] += 0.01
    elif attack == "spatial_weight":
        frame.loc[0, "spatial_model_weight"] = 0.5
    elif attack == "imputed_ability":
        proposal = int(frame.loc[~frame["donor_ability_available"], "proposal_row_order"].iloc[0])
        frame.loc[frame["proposal_row_order"] == proposal, "donor_ability_sg"] = 0.25
    elif attack == "clipped_flag":
        frame.loc[frame["proposal_row_order"] == 137, "donor_ability_was_clipped"] = False
    elif attack == "ability_date":
        frame.loc[frame["proposal_row_order"] == 0, "donor_skill_date"] = pd.Timestamp(
            "2026-06-25", tz="UTC"
        )
    elif attack == "reference_reward":
        frame.loc[0, "fixed_reference_reward"] += 0.25
    else:
        frame.loc[0, "donor_player_name"] = "hostile split"
    with pytest.raises(audit.AuditV2Error):
        audit._vector_checks(frame, sealed["schema"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authority_ready",), True),
        (("fixed_caps", "complete_atoms"), "35218"),
        (("fixed_input_identity", "outer_manifest_sha256"), "0" * 64),
        (("external_review_requirements", "candidate_id_is_not_authority"), False),
        (("validation", "canonical_full_support_sha256"), "0" * 64),
        (("authority_evidence_sha256",), "0" * 64),
    ],
)
def test_strict_candidate_maps_reject_nested_mutations(
    path: tuple[str, ...],
    value: object,
    sealed: dict[str, object],
) -> None:
    captured = dict(sealed["candidate_bytes"])
    candidate = json.loads(captured["authority_candidate.json"])
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    captured["authority_candidate.json"] = (
        json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    ).encode()
    with pytest.raises(audit.AuditV2Error):
        audit._validate_candidate(captured)


def test_fully_resealed_candidate_and_closure_widening_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed: dict[str, object],
) -> None:
    root = tmp_path / "candidate"
    _copy_capture(root, sealed["candidate_bytes"])
    candidate = json.loads((root / "authority_candidate.json").read_bytes())
    candidate["authority_ready"] = True
    candidate_raw = _write_json(root / "authority_candidate.json", candidate)
    manifest = json.loads((root / "manifest.json").read_bytes())
    manifest["authority_ready"] = True
    manifest["payload_files"]["authority_candidate.json"] = {
        "bytes": len(candidate_raw), "sha256": sha256(candidate_raw).hexdigest()
    }
    manifest_raw = _write_json(root / "manifest.json", manifest)
    index = json.loads((root / "OUTPUT_SHA256.json").read_bytes())
    for name in ("REPORT.md", "authority_candidate.json", "manifest.json"):
        raw = (root / name).read_bytes()
        index["files"][name] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    _write_json(root / "OUTPUT_SHA256.json", index)
    monkeypatch.setattr(audit, "FIXED_CANDIDATE", root)
    with pytest.raises(audit.AuditV2Error, match="byte pin"):
        audit._capture_pinned(root, audit.CANDIDATE_FILES, (), "candidate")
    (root / "unreviewed.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(audit.AuditV2Error, match="closure"):
        audit._capture_pinned(root, audit.CANDIDATE_FILES, (), "candidate")


def test_original_byte_and_recursive_closure_mutations_are_rejected(
    tmp_path: Path,
    sealed: dict[str, object],
) -> None:
    root = tmp_path / "original"
    _copy_capture(root, sealed["original_bytes"])
    part = root / Path(audit.PARTS[0][0])
    raw = bytearray(part.read_bytes())
    raw[-1] ^= 1
    part.write_bytes(raw)
    with pytest.raises(audit.AuditV2Error, match="byte pin"):
        audit._capture_pinned(
            root,
            audit.ORIGINAL_FILES,
            ("complete_atom_support", "complete_atom_support/parts"),
            "original",
        )
    (root / "complete_atom_support" / "parts" / "extra.bin").write_bytes(b"x")
    with pytest.raises(audit.AuditV2Error, match="closure"):
        audit._capture_pinned(
            root,
            audit.ORIGINAL_FILES,
            ("complete_atom_support", "complete_atom_support/parts"),
            "original",
        )


def test_validate_only_rechecks_start_end_without_emitting(
    monkeypatch: pytest.MonkeyPatch,
    sealed: dict[str, object],
) -> None:
    monkeypatch.setattr(audit, "_vector_checks", lambda _frame, _schema: sealed["vectors"])
    result = audit.run_fixed_audit_v2("validate-only", _attestation())
    assert result == {
        "mode": "validate-only",
        "scope": audit.SCOPE,
        "audit_only_semantic_root_sha256": audit.EXPECTED_SEMANTIC_ROOT,
        "status": "HOLD_source_reviews_required_no_audit_emitted",
    }
    assert not audit.FIXED_OUTPUT.exists()


def test_flat_non_authorizing_audit_output_validates_on_temp_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed: dict[str, object],
) -> None:
    root, result = _build_temp(monkeypatch, tmp_path, sealed, "valid-audit")
    payload = json.loads((root / "audit.json").read_bytes())
    assert result["decision"] == audit.DECISION
    assert result["audit_status"] == audit.AUDIT_STATUS
    assert result["final_audit_tree"] == audit._validate_output(root, payload)
    assert set(path.name for path in root.iterdir()) == {
        "OUTPUT_SHA256.json", "REPORT.md", "audit.json", "manifest.json"
    }
    assert payload["data_only_authority_record_ready"] is False
    assert payload["authority_id"] is None
    assert payload["authority_lock_sha256"] is None
    assert all(payload[field] is False for field in audit.FAIL_CLOSED)


@pytest.mark.parametrize("attack", ["authority", "nested_review", "report", "index", "extra"])
def test_written_audit_hostile_mutations_fail_closed(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed: dict[str, object],
) -> None:
    root, _result = _build_temp(monkeypatch, tmp_path, sealed, f"hostile-{attack}")
    expected = json.loads((root / "audit.json").read_bytes())
    if attack == "authority":
        expected["authority_ready"] = True
        _reseal_output(root, expected)
    elif attack == "nested_review":
        expected["next_review_requirements"]["audit_id_is_not_authority"] = False
        _reseal_output(root, expected)
    elif attack == "report":
        (root / "REPORT.md").write_text("hostile\n", encoding="utf-8")
    elif attack == "index":
        index = json.loads((root / "OUTPUT_SHA256.json").read_bytes())
        index["files"]["REPORT.md"]["sha256"] = "0" * 64
        _write_json(root / "OUTPUT_SHA256.json", index)
    else:
        (root / "unsealed.txt").write_text("widen", encoding="utf-8")
    with pytest.raises(audit.AuditV2Error):
        audit._validate_output(root, expected)


def test_report_mutation_during_last_source_check_is_detected_by_final_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sealed: dict[str, object],
) -> None:
    root = tmp_path / "last-check-report-attack"
    monkeypatch.setattr(audit, "FIXED_OUTPUT", root)
    monkeypatch.setattr(audit, "_collect", lambda _att: _evidence(sealed))
    original = audit._final_source_recheck

    def hostile(evidence: object, attestation: object) -> None:
        original(evidence, attestation)
        (root / "REPORT.md").write_bytes(b"mutated during last source check\n")

    monkeypatch.setattr(audit, "_final_source_recheck", hostile)
    with pytest.raises(audit.AuditV2Error, match="final audit tree"):
        audit._build(_attestation())


def test_preexisting_output_and_widened_modes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "preexisting"
    root.mkdir()
    monkeypatch.setattr(audit, "FIXED_OUTPUT", root)
    with pytest.raises(FileExistsError):
        audit._build(_attestation())
    with pytest.raises(audit.AuditV2Error):
        audit.run_fixed_audit_v2("build", _attestation())
    with pytest.raises(audit.AuditV2Error):
        audit.run_fixed_audit_v2("validate-only", dict(_attestation(), extra=True))


def _function_source(tree: ast.Module, source: str, names: set[str]) -> str:
    chunks = [
        ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    return "\n".join(chunks)


def test_source_is_independent_and_semantically_different() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("selector" in name or "candidate" in name for name in imports)
    assert "hash_pandas_object" not in source
    assert "EXPECTED_AUDIT_DIGESTS" not in source
    assert "_vector_digest" not in source
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "BufferReader"
        for node in ast.walk(tree)
    )
    forbidden_calls = {
        "solve", "simulate", "select", "admit", "callback", "run_selector",
        "validate_candidate_artifact", "validate_selector_support",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden_calls)

    semantic_names = {
        "_validate_candidate", "_validate_original", "_load_from_captures",
        "_group_inventory", "_schema_records", "_topology_records",
        "_mass_records", "_group_records", "_ability_records",
        "_reference_records", "_ndjson_components", "_vector_checks", "_collect",
    }
    current = _function_source(tree, source, semantic_names)
    for relative in (
        "archive/hole_sim_real_app_h8_selector_authority_candidate.py",
        "archive/audit_hole_sim_real_app_h8_selector_authority_candidate.py",
    ):
        other_source = (REPO / relative).read_text(encoding="utf-8")
        other_tree = ast.parse(other_source)
        other = _function_source(
            other_tree,
            other_source,
            {
                node.name
                for node in other_tree.body
                if isinstance(node, ast.FunctionDef)
            },
        )
        matcher = SequenceMatcher(
            None, current.splitlines(), other.splitlines(), autojunk=False
        )
        assert matcher.ratio() < 0.35
        assert matcher.find_longest_match().size < 15
        old_functions = {
            ast.dump(node, include_attributes=False)
            for node in other_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        assert all(
            ast.dump(node, include_attributes=False) not in old_functions
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in semantic_names
        )


def test_auditor_binds_its_exact_focused_test() -> None:
    assert _sha(Path(__file__)) == audit.AUDITOR_TEST_SHA256

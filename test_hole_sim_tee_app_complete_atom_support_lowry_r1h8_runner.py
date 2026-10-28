from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_tee_app_complete_atom_support_lowry_r1h8_runner as runner


def _fake_audit() -> dict[str, object]:
    return {
        **runner.EXPECTED_COUNTS,
        "query_key": runner.EXPECTED_QUERY_KEY,
        "support_key": runner.EXPECTED_SUPPORT_KEY,
        "quality_mass": 1.000000000000006,
        "terminal_mass": 0.002,
        "anchor_ready": False,
    }


def _fake_admission() -> dict[str, object]:
    source = {"runner_source": "a" * 64}
    artifact = {"candidate": "b" * 64, "candidate_closed_files": 24}
    return {
        "candidate": {
            "candidate_id": runner.EXPECTED_CANDIDATE_ID,
            "source_inventory_sha256": runner.EXPECTED_SOURCE_INVENTORY_SHA256,
            "expansion_audit": _fake_audit(),
        },
        "audit": {"audit_id": runner.EXPECTED_AUDIT_ID},
        "authority": {"authority_id": runner.EXPECTED_AUTHORITY_ID},
        "source_sha256_start": source,
        "source_sha256_end": source,
        "artifact_sha256_start": artifact,
        "artifact_sha256_end": artifact,
    }


def _fake_inner_manifest() -> dict[str, object]:
    return {
        "contract": runner.complete.PERSISTENCE_CONTRACT,
        "production_enabled": False,
        "requests": 1,
        "complete_atoms": runner.EXPECTED_COUNTS["complete_atoms"],
        "expansion_batch_rows": runner.EXPANSION_BATCH_ROWS,
        "max_rows_per_part": runner.MAX_ROWS_PER_PART,
        "max_hazard_donors_per_surface": runner.MAX_HAZARD_DONORS_PER_SURFACE,
        "absolute_max_hazard_provider_donors": (
            runner.complete.ABSOLUTE_MAX_HAZARD_PROVIDER_DONORS
        ),
        "max_complete_atoms_per_proposal": runner.MAX_COMPLETE_ATOMS_PER_PROPOSAL,
        "peak_buffered_rows": runner.MAX_ROWS_PER_PART,
        "source_inventory_sha256": runner.EXPECTED_SOURCE_INVENTORY_SHA256,
        "source_inventory_authority_id": runner.EXPECTED_AUTHORITY_ID,
        "source_inventory_authority_scope": runner.EXPECTED_AUTHORITY_SCOPE,
        "source_inventory_authority_ready": True,
        "parts": [
            {
                "path": f"parts/part-{index:06d}.parquet",
                "rows": rows,
            }
            for index, rows in enumerate((8_192, 8_192, 8_192, 8_192, 2_450))
        ],
        "inventory": [_fake_audit()],
    }


def _fake_inner_validation() -> dict[str, object]:
    return {
        "contract": "persisted_complete_tee_app_atom_validation_v4",
        "queries": 1,
        "parts": 5,
        "complete_atoms": runner.EXPECTED_COUNTS["complete_atoms"],
    }


def _build_fake_outer(root: Path, admission: Mapping[str, object]) -> None:
    inner = root / runner.INNER_NAME
    (inner / "parts").mkdir(parents=True)
    inner_manifest = _fake_inner_manifest()
    for part in inner_manifest["parts"]:  # type: ignore[index]
        (inner / part["path"]).write_bytes(b"parquet-placeholder")  # type: ignore[index]
    runner._write_json(inner / "manifest.json", inner_manifest)
    runner._write_json(
        inner / runner.OUTPUT_INDEX,
        {"contract": "non_circular_output_sha256_v1", "files": {}},
    )
    manifest = {
        "contract": runner.CONTRACT,
        "research_only": True,
        "production_enabled": False,
        "anchor_ready": False,
        "simulation_launched": False,
        "mean_solve_launched": False,
        "candidate_id": runner.EXPECTED_CANDIDATE_ID,
        "audit_id": runner.EXPECTED_AUDIT_ID,
        "authority_id": runner.EXPECTED_AUTHORITY_ID,
        "authority_record_sha256": runner.EXPECTED_AUTHORITY_RECORD_SHA256,
        "authority_lock_sha256": runner.EXPECTED_AUTHORITY_LOCK_SHA256,
        "query_key": runner.EXPECTED_QUERY_KEY,
        "support_key": runner.EXPECTED_SUPPORT_KEY,
        "complete_atoms": runner.EXPECTED_COUNTS["complete_atoms"],
        "terminal_atoms": runner.EXPECTED_COUNTS["terminal_atoms"],
        "imputed_ability_atoms": runner.EXPECTED_COUNTS["imputed_ability_atoms"],
        "inner_relative_path": runner.INNER_NAME,
        "inner_output_index_sha256": runner._sha256_file(
            inner / runner.OUTPUT_INDEX
        ),
        "inner_validation": _fake_inner_validation(),
        "source_sha256_start": admission["source_sha256_start"],
        "source_sha256_end": admission["source_sha256_end"],
        "artifact_sha256_start": admission["artifact_sha256_start"],
        "artifact_sha256_end": admission["artifact_sha256_end"],
    }
    runner._write_json(root / "manifest.json", manifest)
    (root / "REPORT.md").write_text(runner._outer_report(), encoding="utf-8")
    runner._write_outer_index(root)


def test_cli_has_only_required_fixed_actions() -> None:
    parser = runner.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--validate-only", "--materialize"}
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["--validate-only"]).validate_only is True
    assert parser.parse_args(["--materialize"]).materialize is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--validate-only", "--materialize"])


def test_real_validate_only_preflight_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = tmp_path / "must-remain-absent"
    monkeypatch.setattr(runner, "OUTPUT", absent)
    result = runner.preflight()
    assert result["ready"] is True
    assert result["output_exists"] is False
    assert result["materialization_launched"] is False
    assert result["simulation_launched"] is False
    assert result["mean_solve_launched"] is False
    assert not absent.exists()


def test_existing_output_refuses_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(
        runner,
        "_admit_fixed_inputs",
        lambda: pytest.fail("admission must not run for an existing output"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        runner.materialize()


def test_source_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    real = runner._sha256_file

    def changed(path: Path) -> str:
        if Path(path).resolve() == Path(runner.complete.__file__).resolve():
            return "0" * 64
        return real(path)

    monkeypatch.setattr(runner, "_sha256_file", changed)
    with pytest.raises(runner.FixedH8RunnerError, match="complete_seam"):
        runner._source_state()


def test_authority_lock_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = json.loads(
        (runner.CANDIDATE / "authority_candidate.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        runner.complete, "_SOURCE_PINNED_AUTHORITY_LOCK_SHA256", "0" * 64
    )
    with pytest.raises(
        (runner.FixedH8RunnerError, runner.complete.CompleteAtomSupportError),
        match="authority lock changed",
    ):
        runner._authority_record(candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proposal_atoms", 903),
        ("surface_source_inventory_records", 902),
        ("hazard_source_inventory_records", 30_701),
        ("complete_atoms", 35_217),
        ("terminal_atoms", 0),
        ("imputed_ability_atoms", 1_988),
    ],
)
def test_candidate_count_drift_fails_closed(field: str, value: int) -> None:
    candidate = {
        "candidate_id": runner.EXPECTED_CANDIDATE_ID,
        "source_inventory_sha256": runner.EXPECTED_SOURCE_INVENTORY_SHA256,
        "expansion_audit": _fake_audit(),
    }
    candidate["expansion_audit"][field] = value  # type: ignore[index]
    with pytest.raises(runner.FixedH8RunnerError, match="Candidate count changed"):
        runner._assert_candidate(candidate)


def test_inner_cap_drift_fails_closed() -> None:
    manifest = _fake_inner_manifest()
    manifest["max_hazard_donors_per_surface"] = 8_193
    with pytest.raises(runner.FixedH8RunnerError, match="Inner cap changed"):
        runner._assert_inner(manifest, _fake_inner_validation())


def test_inner_numeric_string_rejects_instead_of_coercing() -> None:
    manifest = _fake_inner_manifest()
    manifest["expansion_batch_rows"] = "8192"
    with pytest.raises(runner.FixedH8RunnerError, match="Inner cap changed"):
        runner._assert_inner(manifest, _fake_inner_validation())

    candidate = {
        "candidate_id": runner.EXPECTED_CANDIDATE_ID,
        "source_inventory_sha256": runner.EXPECTED_SOURCE_INVENTORY_SHA256,
        "expansion_audit": _fake_audit(),
    }
    candidate["expansion_audit"]["proposal_atoms"] = "904"  # type: ignore[index]
    with pytest.raises(runner.FixedH8RunnerError, match="Candidate count changed"):
        runner._assert_candidate(candidate)


def test_all_fixed_numeric_fields_reject_bool_float_and_string() -> None:
    inner_fields = {
        "expansion_batch_rows": 8_192,
        "max_rows_per_part": 8_192,
        "max_hazard_donors_per_surface": 8_192,
        "absolute_max_hazard_provider_donors": 8_192,
        "max_complete_atoms_per_proposal": 16_384,
        "requests": 1,
        "complete_atoms": 35_218,
    }
    for field, expected in inner_fields.items():
        for hostile in (True, float(expected) + 0.9, str(expected)):
            manifest = _fake_inner_manifest()
            manifest[field] = hostile
            with pytest.raises(runner.FixedH8RunnerError):
                runner._assert_inner(manifest, _fake_inner_validation())

    for field, expected in runner.EXPECTED_COUNTS.items():
        for hostile in (True, float(expected) + 0.9, str(expected)):
            candidate = {
                "candidate_id": runner.EXPECTED_CANDIDATE_ID,
                "source_inventory_sha256": runner.EXPECTED_SOURCE_INVENTORY_SHA256,
                "expansion_audit": _fake_audit(),
            }
            candidate["expansion_audit"][field] = hostile  # type: ignore[index]
            with pytest.raises(runner.FixedH8RunnerError):
                runner._assert_candidate(candidate)


def test_part_and_validation_counts_require_exact_builtin_ints() -> None:
    for hostile in (True, 8192.9, "8192"):
        manifest = _fake_inner_manifest()
        manifest["parts"][0]["rows"] = hostile  # type: ignore[index]
        with pytest.raises(runner.FixedH8RunnerError, match="part row count"):
            runner._assert_inner(manifest, _fake_inner_validation())
    for field, expected in (("queries", 1), ("parts", 5), ("complete_atoms", 35_218)):
        for hostile in (True, float(expected) + 0.9, str(expected)):
            validation = _fake_inner_validation()
            validation[field] = hostile
            with pytest.raises(runner.FixedH8RunnerError, match="validation count"):
                runner._assert_inner(_fake_inner_manifest(), validation)


def test_outer_index_binds_nested_inner_index_and_extra_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outer"
    inner = root / runner.INNER_NAME
    inner.mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "REPORT.md").write_text("report\n", encoding="utf-8")
    (inner / "manifest.json").write_text("{}\n", encoding="utf-8")
    (inner / runner.OUTPUT_INDEX).write_text("{}\n", encoding="utf-8")
    runner._write_outer_index(root)
    index = runner._validate_outer_index(root)
    assert f"{runner.INNER_NAME}/{runner.OUTPUT_INDEX}" in index["files"]
    (inner / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(runner.FixedH8RunnerError, match="tree closure changed"):
        runner._validate_outer_index(root)


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("research_only", 1),
        ("complete_atoms", 35_218.0),
        ("terminal_atoms", True),
        ("imputed_ability_atoms", 1_989.0),
    ],
)
def test_outer_identity_aliases_reject_after_full_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    hostile: object,
) -> None:
    output = tmp_path / "outer"
    admission = _fake_admission()
    _build_fake_outer(output, admission)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "_admit_fixed_inputs", lambda: admission)
    monkeypatch.setattr(
        runner.complete,
        "validate_persisted_complete_atom_artifact",
        lambda inner: _fake_inner_validation(),
    )
    manifest = runner._read_json(output / "manifest.json")
    manifest[field] = hostile
    runner._write_json(output / "manifest.json", manifest)
    runner._write_outer_index(output)
    with pytest.raises(runner.FixedH8RunnerError, match="Outer identity changed"):
        runner.validate_materialized()


def test_outer_inner_validation_and_provenance_aliases_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outer"
    admission = _fake_admission()
    _build_fake_outer(output, admission)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "_admit_fixed_inputs", lambda: admission)
    monkeypatch.setattr(
        runner.complete,
        "validate_persisted_complete_atom_artifact",
        lambda inner: _fake_inner_validation(),
    )
    manifest = runner._read_json(output / "manifest.json")
    manifest["inner_validation"]["complete_atoms"] = 35_218.0  # type: ignore[index]
    runner._write_json(output / "manifest.json", manifest)
    runner._write_outer_index(output)
    with pytest.raises(runner.FixedH8RunnerError, match="inner validation"):
        runner.validate_materialized()

    output2 = tmp_path / "outer-provenance"
    _build_fake_outer(output2, admission)
    monkeypatch.setattr(runner, "OUTPUT", output2)
    manifest = runner._read_json(output2 / "manifest.json")
    manifest["artifact_sha256_end"]["candidate_closed_files"] = 24.0  # type: ignore[index]
    runner._write_json(output2 / "manifest.json", manifest)
    runner._write_outer_index(output2)
    with pytest.raises(runner.FixedH8RunnerError, match="Outer provenance changed"):
        runner.validate_materialized()


@pytest.mark.parametrize("hostile", [True, 3.9, "3"])
def test_outer_index_byte_count_requires_builtin_int(
    tmp_path: Path, hostile: object
) -> None:
    root = tmp_path / "outer"
    inner = root / runner.INNER_NAME
    inner.mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "REPORT.md").write_text("report\n", encoding="utf-8")
    (inner / runner.OUTPUT_INDEX).write_text("{}\n", encoding="utf-8")
    runner._write_outer_index(root)
    index = runner._read_json(root / runner.OUTPUT_INDEX)
    index["files"]["manifest.json"]["bytes"] = hostile  # type: ignore[index]
    runner._write_json(root / runner.OUTPUT_INDEX, index)
    with pytest.raises(runner.FixedH8RunnerError, match="byte count changed"):
        runner._validate_outer_index(root)


def test_root_reserved_index_path_rejects_but_nested_is_allowed() -> None:
    with pytest.raises(runner.FixedH8RunnerError, match="Unsafe"):
        runner._safe_relative(runner.OUTPUT_INDEX)
    assert runner._safe_relative(
        f"{runner.INNER_NAME}/{runner.OUTPUT_INDEX}"
    ) == f"{runner.INNER_NAME}/{runner.OUTPUT_INDEX}"


def test_materialize_passes_only_fixed_caps_and_seals_nested_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "outer"
    admission = _fake_admission()
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "_admit_fixed_inputs", lambda: admission)
    monkeypatch.setattr(runner, "_load_request", lambda candidate: object())
    monkeypatch.setattr(
        runner, "_source_state", lambda: admission["source_sha256_start"]
    )
    monkeypatch.setattr(
        runner, "_artifact_state", lambda: admission["artifact_sha256_start"]
    )

    def fake_persist(requests: object, inner: Path, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        inner.mkdir(parents=True)
        parts = inner / "parts"
        parts.mkdir()
        manifest = _fake_inner_manifest()
        for part in manifest["parts"]:  # type: ignore[index]
            path = inner / part["path"]  # type: ignore[index]
            path.write_bytes(b"parquet-placeholder")
        runner._write_json(inner / "manifest.json", manifest)
        runner._write_json(
            inner / runner.OUTPUT_INDEX,
            {"contract": "non_circular_output_sha256_v1", "files": {}},
        )
        return manifest

    monkeypatch.setattr(
        runner.complete, "persist_complete_atom_requests", fake_persist
    )
    monkeypatch.setattr(
        runner.complete,
        "validate_persisted_complete_atom_artifact",
        lambda inner: _fake_inner_validation(),
    )
    expected_result = {"valid": True, "mocked": True}
    monkeypatch.setattr(runner, "validate_materialized", lambda: expected_result)
    assert runner.materialize() == expected_result
    assert captured == {
        "expansion_batch_rows": 8_192,
        "max_rows_per_part": 8_192,
        "max_hazard_donors_per_surface": 8_192,
        "max_complete_atoms_per_proposal": 16_384,
    }
    index = runner._read_json(output / runner.OUTPUT_INDEX)
    assert f"{runner.INNER_NAME}/{runner.OUTPUT_INDEX}" in index["files"]


def test_validate_only_main_never_calls_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "absent"
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "preflight", lambda: {"ready": True})
    monkeypatch.setattr(
        runner, "materialize", lambda: pytest.fail("materialize must not run")
    )
    monkeypatch.setattr(sys, "argv", ["runner", "--validate-only"])
    runner.main()
    assert json.loads(capsys.readouterr().out) == {"ready": True}
    assert not output.exists()


def test_runner_source_is_nonempty_and_currently_compiles() -> None:
    source = Path(runner.__file__).read_bytes()
    assert source
    assert sha256(source).hexdigest() == runner._sha256_file(Path(runner.__file__))

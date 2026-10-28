from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import textwrap

import pytest


REPO = Path(__file__).resolve().parent
MODULE_PATH = (
    REPO
    / "archive"
    / "run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py"
)
SPEC = importlib.util.spec_from_file_location("real_app_h8_audit_launcher_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fake_auditor_source(behavior: str = "normal") -> str:
    return textwrap.dedent(
        f'''\
        from hashlib import sha256
        import json
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        OUTPUT = ROOT / "permanent_data" / "hole_sim_real_app_h8_selector_authority_candidate_audit_v2"
        BEHAVIOR = {behavior!r}

        def _tree():
            files = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
            directories = sorted(path for path in OUTPUT.rglob("*") if path.is_dir())
            inventory = []
            total = 0
            for path in files:
                payload = path.read_bytes()
                total += len(payload)
                inventory.append({{
                    "path": path.relative_to(OUTPUT).as_posix(),
                    "bytes": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                }})
            encoded = json.dumps(
                inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            return {{
                "files": len(files),
                "directories": len(directories),
                "bytes": total,
                "inventory_sha256": sha256(encoded).hexdigest(),
            }}

        def _validate_result():
            return {{
                "mode": "validate-only",
                "scope": "independent_byte_and_vectorized_invariant_audit_v2",
                "audit_only_semantic_root_sha256": "b8312616d491bdce5e97fa438e95a8b9ead36ce1cb0fb9d7274605cfe3675dae",
                "status": "HOLD_source_reviews_required_no_audit_emitted",
            }}

        def _mutate_fixed_source():
            targets = {{
                "mutate_auditor_source": Path(__file__),
                "mutate_auditor_test": ROOT / "test_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py",
                "mutate_launcher_source": ROOT / "archive" / "run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py",
                "mutate_launcher_test": ROOT / "test_run_audit_hole_sim_real_app_h8_selector_authority_candidate_v2.py",
            }}
            if BEHAVIOR in targets:
                targets[BEHAVIOR].write_bytes(b"post-capture mutation")

        def run_fixed_audit_v2(mode, launcher_attestation):
            _mutate_fixed_source()
            if BEHAVIOR == "mutate_attestation":
                launcher_attestation["contract"] = "hostile"
            if mode == "validate-only":
                if BEHAVIOR == "validate_writes":
                    OUTPUT.mkdir(parents=False, exist_ok=False)
                    (OUTPUT / "forbidden.txt").write_text("write", encoding="utf-8")
                return _validate_result()
            OUTPUT.mkdir(parents=False, exist_ok=False)
            for name, payload in (
                ("OUTPUT_SHA256.json", b"index"),
                ("REPORT.md", b"report"),
                ("audit.json", b"audit"),
                ("manifest.json", b"manifest"),
            ):
                (OUTPUT / name).write_bytes(payload)
            tree = _tree()
            return {{
                "mode": "audit",
                "scope": "independent_byte_and_vectorized_invariant_audit_v2",
                "decision": "GO_for_separate_data_only_authority_record_review_only",
                "audit_status": "candidate_byte_and_vectorized_audit_passed_record_review_required",
                "audit_id": "a" * 64,
                "output": str(OUTPUT),
                "final_audit_tree": tree,
            }}

        def final_audit_tree_v2():
            if BEHAVIOR == "mutate_tree_during_final":
                (OUTPUT / "audit.json").write_bytes(b"mutated after sealed return")
            return _tree()
        '''
    )


def _configure_fixed_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    behavior: str = "normal",
) -> dict[str, Path]:
    repo = tmp_path / "repo"
    archive = repo / "archive"
    permanent_data = repo / "permanent_data"
    archive.mkdir(parents=True)
    permanent_data.mkdir()

    paths = {
        "auditor_source": repo / launcher.AUDITOR_RELATIVE_PATH,
        "auditor_test": repo / launcher.AUDITOR_TEST_RELATIVE_PATH,
        "launcher_source": repo / launcher.LAUNCHER_RELATIVE_PATH,
        "launcher_test": repo / launcher.LAUNCHER_TEST_RELATIVE_PATH,
        "audit_output": repo / launcher.AUDIT_OUTPUT_RELATIVE_PATH,
    }
    paths["auditor_source"].write_text(
        _fake_auditor_source(behavior), encoding="utf-8", newline="\n"
    )
    paths["auditor_test"].write_bytes(b"frozen auditor test\n")
    paths["launcher_source"].write_bytes(b"externally reviewed launcher\n")
    paths["launcher_test"].write_bytes(b"externally reviewed launcher test\n")

    monkeypatch.setattr(launcher, "REPO", repo)
    monkeypatch.setattr(launcher, "ARCHIVE", archive)
    monkeypatch.setattr(launcher, "AUDITOR_SOURCE", paths["auditor_source"])
    monkeypatch.setattr(launcher, "AUDITOR_TEST", paths["auditor_test"])
    monkeypatch.setattr(launcher, "LAUNCHER_SOURCE", paths["launcher_source"])
    monkeypatch.setattr(launcher, "LAUNCHER_TEST", paths["launcher_test"])
    monkeypatch.setattr(launcher, "_FIXED_AUDIT_ROOT", paths["audit_output"])
    monkeypatch.setattr(
        launcher, "EXPECTED_AUDITOR_SOURCE_SHA256", _sha(paths["auditor_source"])
    )
    monkeypatch.setattr(
        launcher, "EXPECTED_AUDITOR_TEST_SHA256", _sha(paths["auditor_test"])
    )
    return paths


def test_frozen_auditor_pair_matches_launcher_pins() -> None:
    assert launcher.EXPECTED_AUDITOR_SOURCE_SHA256 == _sha(launcher.AUDITOR_SOURCE)
    assert launcher.EXPECTED_AUDITOR_TEST_SHA256 == _sha(launcher.AUDITOR_TEST)
    assert launcher._HEX64.fullmatch(launcher.EXPECTED_AUDITOR_SOURCE_SHA256)
    assert launcher._HEX64.fullmatch(launcher.EXPECTED_AUDITOR_TEST_SHA256)


def test_contract_has_fixed_paths_and_closed_cli() -> None:
    assert launcher.LAUNCHER_SOURCE == REPO / launcher.LAUNCHER_RELATIVE_PATH
    assert launcher.LAUNCHER_TEST == REPO / launcher.LAUNCHER_TEST_RELATIVE_PATH
    assert launcher.AUDITOR_SOURCE == REPO / launcher.AUDITOR_RELATIVE_PATH
    assert launcher.AUDITOR_TEST == REPO / launcher.AUDITOR_TEST_RELATIVE_PATH
    assert launcher._FIXED_AUDIT_ROOT == REPO / launcher.AUDIT_OUTPUT_RELATIVE_PATH

    parser = launcher.build_parser()
    option_sets = {
        tuple(action.option_strings)
        for action in parser._actions
        if action.option_strings
    }
    assert option_sets == {("-h", "--help"), ("--validate-only",), ("--audit",)}
    assert parser.parse_args(["--validate-only"]).validate_only is True
    assert parser.parse_args(["--audit"]).audit is True
    for widened in (
        [],
        ["--audit", "--validate-only"],
        ["--audit", "--path", "elsewhere"],
        ["--audit", "--digest", "0" * 64],
        ["--audit", "--cap", "1"],
        ["--audit", "--output", "elsewhere"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(widened)


def test_captured_auditor_bytes_execute_after_disk_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_fixed_layout(monkeypatch, tmp_path)
    captured, states = launcher._capture_fixed_sources()
    paths["auditor_source"].write_text(
        "raise AssertionError('swapped auditor was executed')\n",
        encoding="utf-8",
        newline="\n",
    )
    namespace = launcher._exec_captured_auditor(captured)
    result = namespace["run_fixed_audit_v2"](
        "validate-only", launcher._launcher_attestation(states)
    )
    assert result["status"] == launcher.EXPECTED_VALIDATE_STATUS


def test_exec_path_has_no_source_reopen_or_dynamic_loader() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "importlib" not in imports
    assert "runpy" not in imports
    assert "subprocess" not in imports

    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_exec_captured_auditor"
    )
    called_names = {
        node.func.id
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "compile" in called_names
    assert "exec" in called_names
    assert "open" not in called_names
    assert "__import__" not in called_names
    assert all(
        not (
            isinstance(node, ast.Attribute)
            and node.attr in {"read_bytes", "read_text", "open"}
        )
        for node in ast.walk(execute)
    )


def test_validate_only_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_fixed_layout(monkeypatch, tmp_path)
    result = launcher.run_fixed("validate-only")
    assert result == {
        "mode": "validate-only",
        "scope": launcher.EXPECTED_SCOPE,
        "audit_only_semantic_root_sha256": launcher.EXPECTED_SEMANTIC_ROOT,
        "status": launcher.EXPECTED_VALIDATE_STATUS,
    }
    assert not paths["audit_output"].exists()


def test_validate_only_rejects_a_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_fixed_layout(monkeypatch, tmp_path, behavior="validate_writes")
    with pytest.raises(launcher.FixedAuditLauncherError, match="emitted"):
        launcher.run_fixed("validate-only")
    assert paths["audit_output"].exists()


@pytest.mark.parametrize(
    "behavior,label",
    (
        ("mutate_auditor_source", "auditor source postcheck"),
        ("mutate_auditor_test", "auditor test postcheck"),
        ("mutate_launcher_source", "launcher source postcheck"),
        ("mutate_launcher_test", "launcher test postcheck"),
    ),
)
def test_postcheck_rejects_source_or_test_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    behavior: str,
    label: str,
) -> None:
    _configure_fixed_layout(monkeypatch, tmp_path, behavior=behavior)
    with pytest.raises(launcher.FixedAuditLauncherError, match=label):
        launcher.run_fixed("validate-only")


def test_audit_refuses_preexisting_output_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_fixed_layout(monkeypatch, tmp_path)
    paths["audit_output"].mkdir()
    with pytest.raises(FileExistsError, match="must be absent"):
        launcher.run_fixed("audit")


def test_fake_audit_seal_and_final_readback_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_fixed_layout(monkeypatch, tmp_path)
    result = launcher.run_fixed("audit")
    assert result["mode"] == "audit"
    assert result["output"] == str(paths["audit_output"])
    assert result["final_audit_tree"]["files"] == 4
    assert result["final_audit_tree"]["directories"] == 0


def test_audit_tree_mutation_during_last_postcheck_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_fixed_layout(monkeypatch, tmp_path, behavior="mutate_tree_during_final")
    with pytest.raises(launcher.FixedAuditLauncherError, match="changed after"):
        launcher.run_fixed("audit")


def test_auditor_cannot_mutate_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_fixed_layout(monkeypatch, tmp_path, behavior="mutate_attestation")
    with pytest.raises(launcher.FixedAuditLauncherError, match="mutated"):
        launcher.run_fixed("validate-only")


def test_hostile_mode_context_and_result_types_are_rejected() -> None:
    class DictSubclass(dict):
        pass

    valid_attestation = {
        "contract": launcher.LAUNCH_ATTESTATION_CONTRACT,
        "launcher_relative_path": launcher.LAUNCHER_RELATIVE_PATH,
        "launcher_test_relative_path": launcher.LAUNCHER_TEST_RELATIVE_PATH,
        "launcher_sha256": "1" * 64,
        "launcher_test_sha256": "2" * 64,
    }
    with pytest.raises(TypeError):
        launcher.run_fixed(True)
    with pytest.raises(TypeError):
        launcher._validate_attestation(DictSubclass(valid_attestation))
    with pytest.raises(launcher.FixedAuditLauncherError):
        launcher._validate_attestation(
            {**valid_attestation, "launcher_sha256": b"1" * 64}
        )
    with pytest.raises(launcher.FixedAuditLauncherError):
        launcher._validate_result(
            "validate-only",
            DictSubclass(
                {
                    "mode": "validate-only",
                    "scope": launcher.EXPECTED_SCOPE,
                    "audit_only_semantic_root_sha256": launcher.EXPECTED_SEMANTIC_ROOT,
                    "status": launcher.EXPECTED_VALIDATE_STATUS,
                }
            ),
        )
    with pytest.raises(launcher.FixedAuditLauncherError):
        launcher._validate_final_tree(
            {
                "files": False,
                "directories": 0,
                "bytes": 1,
                "inventory_sha256": "3" * 64,
            },
            "hostile tree",
        )


def test_captured_namespace_rejects_callable_objects() -> None:
    payload = b'''\
class Hostile:
    def __call__(self, *args):
        return {}
run_fixed_audit_v2 = Hostile()
final_audit_tree_v2 = Hostile()
'''
    with pytest.raises(launcher.FixedAuditLauncherError, match="exact function"):
        launcher._exec_captured_auditor(payload)

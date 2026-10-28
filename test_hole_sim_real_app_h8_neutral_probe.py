from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_real_app_h8_neutral_probe as probe


SOURCE = ARCHIVE / "hole_sim_real_app_h8_neutral_probe.py"
DOWNSTREAM = {
    "callback_ready": False,
    "chain_ready": False,
    "anchor_ready": False,
    "mean_solve_ready": False,
    "simulation_ready": False,
    "production_ready": False,
    "research_only": True,
}


class _FakeSession:
    def __init__(self) -> None:
        self.calls = {"readiness": 0, "weight": 0, "select": 0}
        self._selected = _fake_selections()

    @property
    def readiness(self):
        self.calls["readiness"] += 1
        return SimpleNamespace(
            authority_id=probe._AUTHORITY_ID,
            authority_record_sha256=probe._AUTHORITY_RECORD_SHA256,
            authority_lock_sha256=probe._AUTHORITY_LOCK_SHA256,
            selector_runtime_sha256=probe._SELECTOR_RUNTIME_SHA256,
            selector_support_sha256=probe._SELECTOR_SUPPORT_SHA256,
            complete_atoms=35_218,
            proposals=904,
            spatial_proposals=903,
            terminal_atoms=1,
            real_support_admitted=True,
            weighting_ready=True,
            selection_ready=True,
            **DOWNSTREAM,
        )

    def weight(self, parameters):
        self.calls["weight"] += 1
        self.parameters = parameters
        group = tuple(float.fromhex(value) for value in probe._GROUP_MASS)
        total = float.fromhex(probe._ONE_FSUM)
        return SimpleNamespace(
            runtime_binding_sha256=probe._SELECTOR_RUNTIME_SHA256,
            support_sha256=probe._SELECTOR_SUPPORT_SHA256,
            parameter_sha256=probe._PARAMETER_SHA256,
            probability=(total,),
            neutral_probability_mass=(total,),
            donor_group_mass_before=group,
            donor_group_mass_after=group,
            terminal_probability=float.fromhex(probe._TERMINAL),
            clipped_donor_logs=0,
            clipped_target_logs=0,
            zero_fast_path=True,
            selection_mode=probe._SELECTION_MODE,
        )

    def select(self, parameters, uniforms):
        self.calls["select"] += 1
        assert parameters is self.parameters
        self.uniforms = getattr(self, "uniforms", []) + [uniforms]
        return self._selected[self.calls["select"] - 1]


def _fake_selections() -> list[SimpleNamespace]:
    values = []
    for expected in probe._EXPECTED_SELECTIONS:
        atom = expected["atom_id"]
        residual_hex = expected["terminal_residual_uniform_hex"]
        values.append(
            SimpleNamespace(
                **{
                    key: value
                    for key, value in expected.items()
                    if key != "terminal_residual_uniform_hex"
                },
                terminal_probability=float.fromhex(probe._TERMINAL),
                terminal_residual_uniform=(
                    None if residual_hex is None else float.fromhex(residual_hex)
                ),
                selection_mode=probe._SELECTION_MODE,
                hierarchical_atom_id=atom,
                flat_atom_id=atom,
                flat_hierarchical_equal=True,
            )
        )
    return values


def _install_fake_admission(monkeypatch) -> tuple[_FakeSession, dict[str, object]]:
    session = _FakeSession()
    calls: dict[str, object] = {"admission": 0, "parameters": [], "uniforms": []}

    def parameters(**values):
        calls["parameters"].append(values)
        return object()

    def uniforms(*values):
        calls["uniforms"].append(values)
        return values

    def admit():
        calls["admission"] += 1
        return session

    monkeypatch.setattr(probe.admission, "RealAppWeightingParameters", parameters)
    monkeypatch.setattr(probe.admission, "RealAppSemanticUniforms", uniforms)
    monkeypatch.setattr(probe.admission, "admit_lowry_r1h8_app_support", admit)
    return session, calls


def _load_bundle(root: Path) -> tuple[dict, dict, dict, dict[str, bytes]]:
    raw = {path.name: path.read_bytes() for path in root.iterdir()}
    return (
        json.loads(raw["probe.json"]),
        json.loads(raw["manifest.json"]),
        json.loads(raw["OUTPUT_SHA256.json"]),
        raw,
    )


def test_source_has_fixed_zero_argument_api_and_closed_domain_calls() -> None:
    assert probe.__all__ == (
        "NeutralProbeError",
        "validate_only",
        "materialize",
        "main",
    )
    assert len(inspect.signature(probe.validate_only).parameters) == 0
    assert len(inspect.signature(probe.materialize).parameters) == 0
    assert len(inspect.signature(probe.main).parameters) == 0
    assert probe._OUTPUT_RELATIVE == (
        "permanent_data/hole_sim_real_app_h8_neutral_probe_v1"
    )

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {
        "argparse",
        "json",
        "math",
        "os",
        "stat",
        "tempfile",
        "hole_sim_real_app_h8_admission_session",
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported_from == {"__future__", "hashlib", "pathlib"}

    domain_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr
        in {
            "admit_lowry_r1h8_app_support",
            "weight",
            "select",
            "callback",
            "run_chain",
            "anchor",
            "solve",
            "simulate",
            "production",
        }
    }
    assert domain_calls == {"admit_lowry_r1h8_app_support", "weight", "select"}
    admission_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "admission"
    }
    assert admission_calls == {
        "RealAppWeightingParameters",
        "RealAppSemanticUniforms",
        "admit_lowry_r1h8_app_support",
    }


def test_validate_absent_is_read_only_ready_and_never_admits(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "missing"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        probe.admission,
        "admit_lowry_r1h8_app_support",
        lambda: (_ for _ in ()).throw(AssertionError("admission called")),
    )
    assert probe.validate_only() == {
        "contract": probe.PROBE_CONTRACT,
        "status": "ready_to_materialize",
        "output": probe._OUTPUT_RELATIVE,
    }
    assert not output.exists()


def test_preexisting_output_refuses_before_admission_or_parameter_construction(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "belongs-to-user"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)

    def bomb(*_args, **_kwargs):
        raise AssertionError("probe operation called")

    monkeypatch.setattr(probe.admission, "RealAppWeightingParameters", bomb)
    monkeypatch.setattr(probe.admission, "RealAppSemanticUniforms", bomb)
    monkeypatch.setattr(probe.admission, "admit_lowry_r1h8_app_support", bomb)
    with pytest.raises(probe.NeutralProbeError, match="already exists"):
        probe.materialize()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_dangling_output_refuses_lexically_before_any_runtime_call(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "dangling-output"
    assert not output.exists()
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    monkeypatch.setattr(probe, "_lexists", lambda path: path == output)

    def bomb(*_args, **_kwargs):
        raise AssertionError("probe operation called")

    monkeypatch.setattr(probe.admission, "RealAppWeightingParameters", bomb)
    monkeypatch.setattr(probe.admission, "RealAppSemanticUniforms", bomb)
    monkeypatch.setattr(probe.admission, "admit_lowry_r1h8_app_support", bomb)
    with pytest.raises(probe.NeutralProbeError, match="already exists"):
        probe.materialize()


def test_mock_materialize_has_exact_calls_schema_hex_values_and_seals(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "hole_sim_real_app_h8_neutral_probe_v1"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    session, calls = _install_fake_admission(monkeypatch)

    returned = probe.materialize()
    assert calls["admission"] == 1
    assert session.calls == {"readiness": 1, "weight": 1, "select": 4}
    assert calls["parameters"] == [
        {
            "transfer_beta": (0.0, 0.0, 0.0, 0.0),
            "target_ability_sg": (0.0, 0.0, 0.0, 0.0),
            "lambda_correction": (0.0, 0.0, 0.0, 0.0),
            "reward_reference_sha256": probe._REWARD_REFERENCE_SHA256,
        }
    ]
    expected_uniforms = [
        tuple(float.fromhex(value) for value in row) for row in probe._UNIFORM_HEX
    ]
    assert calls["uniforms"] == expected_uniforms
    assert session.uniforms == expected_uniforms

    payload, manifest, index, raw = _load_bundle(output)
    assert returned == payload
    assert set(raw) == probe._OUTPUT_FILES
    assert set(payload) == {
        "contract",
        "authority",
        "scope",
        "parameters",
        "masses",
        "selections",
        "evaluation_counts",
        "downstream",
    }
    assert set(payload["authority"]) == {
        "authority_id",
        "authority_record_sha256",
        "authority_lock_sha256",
        "wrapper_sha256",
        "selector_runtime_sha256",
        "selector_support_sha256",
    }
    assert set(payload["scope"]) == {
        "query_key",
        "support_key",
        "target_player_id",
        "complete_atoms",
        "proposals",
        "spatial_proposals",
        "terminal_atoms",
    }
    assert set(payload["parameters"]) == {"parameter_sha256", "mode"}
    assert set(payload["masses"]) == {
        "probability_fsum_hex",
        "neutral_probability_fsum_hex",
        "terminal_probability_hex",
        "donor_group_mass_before_hex",
        "donor_group_mass_after_hex",
        "max_atom_probability_delta_hex",
        "max_donor_group_delta_hex",
        "clipped_donor_logs",
        "clipped_target_logs",
    }
    assert payload["masses"] == {
        "probability_fsum_hex": probe._ONE_FSUM,
        "neutral_probability_fsum_hex": probe._ONE_FSUM,
        "terminal_probability_hex": probe._TERMINAL,
        "donor_group_mass_before_hex": list(probe._GROUP_MASS),
        "donor_group_mass_after_hex": list(probe._GROUP_MASS),
        "max_atom_probability_delta_hex": probe._ZERO,
        "max_donor_group_delta_hex": probe._ZERO,
        "clipped_donor_logs": 0,
        "clipped_target_logs": 0,
    }
    assert payload["parameters"] == {
        "parameter_sha256": probe._PARAMETER_SHA256,
        "mode": "neutral_zero_fast_path",
    }
    assert payload["evaluation_counts"] == {
        "admission": 1,
        "readiness": 1,
        "explicit_weight": 1,
        "select": 4,
        "underlying_weight_evaluations": 5,
    }
    assert payload["downstream"] == DOWNSTREAM
    assert len(payload["selections"]) == 4

    selection_fields = {
        "semantic_uniforms_hex",
        "atom_id",
        "complete_row_order",
        "terminal",
        "proposal_row_order",
        "surface_class_index",
        "hazard_donor_order",
        "stroke_cost",
        "end_state_key",
        "next_support_key",
        "terminal_probability_hex",
        "terminal_residual_uniform_hex",
        "selection_mode",
        "hierarchical_atom_id",
        "flat_atom_id",
        "flat_hierarchical_equal",
    }
    for observed, expected, uniforms in zip(
        payload["selections"],
        probe._EXPECTED_SELECTIONS,
        probe._UNIFORM_HEX,
        strict=True,
    ):
        assert set(observed) == selection_fields
        assert observed["semantic_uniforms_hex"] == list(uniforms)
        assert observed["terminal_probability_hex"] == probe._TERMINAL
        assert observed["selection_mode"] == probe._SELECTION_MODE
        for field, value in expected.items():
            assert observed[field] == value
        assert observed["hierarchical_atom_id"] == expected["atom_id"]
        assert observed["flat_atom_id"] == expected["atom_id"]
        assert observed["flat_hierarchical_equal"] is True

    assert manifest == {
        "contract": probe.MANIFEST_CONTRACT,
        "artifact": "hole_sim_real_app_h8_neutral_probe_v1",
        "status": "verified_research_only_no_downstream_readiness",
        "payload_files": {
            "probe.json": {
                "bytes": len(raw["probe.json"]),
                "sha256": sha256(raw["probe.json"]).hexdigest(),
            }
        },
        **DOWNSTREAM,
    }
    assert index == {
        "contract": probe.OUTPUT_CONTRACT,
        "files": {
            name: {"bytes": len(raw[name]), "sha256": sha256(raw[name]).hexdigest()}
            for name in ("probe.json", "manifest.json")
        },
    }
    assert probe.validate_only()["status"] == "validated"


def test_publish_race_uses_lexical_check_and_leaves_failed_temp_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "probe"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    _install_fake_admission(monkeypatch)
    checks = 0

    def lexists(path: Path) -> bool:
        nonlocal checks
        if path == output:
            checks += 1
            return checks >= 2
        return probe.os.path.lexists(probe.os.fspath(path))

    monkeypatch.setattr(probe, "_lexists", lexists)
    with pytest.raises(probe.NeutralProbeError, match="failed temporary evidence left"):
        probe.materialize()
    assert checks == 2
    assert not output.exists()
    evidence = list(tmp_path.glob(".probe.*"))
    assert len(evidence) == 1
    assert {path.name for path in evidence[0].iterdir()} == probe._OUTPUT_FILES


@pytest.mark.parametrize(
    ("target_name", "mode", "attributes"),
    [
        (None, stat.S_IFLNK | 0o777, 0),
        (None, stat.S_IFDIR | 0o700, 0x400),
        ("probe.json", stat.S_IFLNK | 0o777, 0),
        ("probe.json", stat.S_IFREG | 0o600, 0x400),
        ("manifest.json", stat.S_IFLNK | 0o777, 0),
        ("manifest.json", stat.S_IFREG | 0o600, 0x400),
        ("OUTPUT_SHA256.json", stat.S_IFLNK | 0o777, 0),
        ("OUTPUT_SHA256.json", stat.S_IFREG | 0o600, 0x400),
    ],
)
def test_validate_rejects_root_and_every_file_symlink_or_reparse_without_read(
    tmp_path: Path,
    monkeypatch,
    target_name: str | None,
    mode: int,
    attributes: int,
) -> None:
    output = tmp_path / "probe"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    _install_fake_admission(monkeypatch)
    probe.materialize()
    real_lstat = probe._lstat_no_follow
    reads: list[Path] = []

    def attacked_lstat(path: Path, label: str):
        if (target_name is None and path == output) or (
            target_name is not None and path == output / target_name
        ):
            return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return real_lstat(path, label)

    real_read_bytes = Path.read_bytes

    def observed_read(path: Path) -> bytes:
        reads.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(probe, "_lstat_no_follow", attacked_lstat)
    monkeypatch.setattr(Path, "read_bytes", observed_read)
    with pytest.raises(probe.NeutralProbeError, match="non-symlink, non-reparse"):
        probe.validate_only()
    assert reads == []


@pytest.mark.parametrize(
    "raw",
    [
        b'{"contract":"a","contract":"b"}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
    ],
)
def test_strict_json_rejects_duplicates_and_nonfinite_constants(raw: bytes) -> None:
    with pytest.raises(
        probe.NeutralProbeError,
        match="strict finite duplicate-free JSON",
    ):
        probe._strict_json(raw, "attacked.json")


def test_validate_rejects_extra_file_and_false_flag_widening(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "probe"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", output)
    _install_fake_admission(monkeypatch)
    probe.materialize()

    (output / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(probe.NeutralProbeError, match="closure"):
        probe.validate_only()

    widened = tmp_path / "widened"
    monkeypatch.setattr(probe, "_OUTPUT_ROOT", widened)
    _install_fake_admission(monkeypatch)
    payload = probe.materialize()
    payload["downstream"]["production_ready"] = True
    for name, raw in probe._bundle_bytes(payload).items():
        (widened / name).write_bytes(raw)
    with pytest.raises(probe.NeutralProbeError, match="downstream"):
        probe.validate_only()

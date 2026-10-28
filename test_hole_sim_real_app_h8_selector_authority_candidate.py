from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest


REPO = Path(__file__).resolve().parent
ARCHIVE = REPO / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_real_app_h8_selector_authority_candidate as candidate


REAL_SOURCE = (
    REPO
    / "permanent_data"
    / "hole_sim_tee_app_complete_atom_support_lowry_r1h8_raw_app_v1"
)
REAL_OUTPUT = (
    REPO
    / "permanent_data"
    / "hole_sim_real_app_h8_selector_authority_candidate_v1"
)


@pytest.fixture(scope="module")
def copied_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("real-app-h8-candidate") / "fixed-source-copy"
    shutil.copytree(REAL_SOURCE, root)
    return root


@pytest.fixture(scope="module")
def copied_frame(copied_source: Path):
    _outer, inner = candidate._validate_manifests(copied_source)
    _tables, frame = candidate._read_and_validate_parts(copied_source, inner)
    return frame


@pytest.fixture(scope="module")
def copied_evidence(copied_source: Path) -> dict[str, object]:
    source = candidate._source_state()
    return {
        **candidate._validation_summary(copied_source),
        "source_sha256_start": source,
        "source_sha256_end": source,
    }


@pytest.fixture(scope="module")
def built_temp_candidate(
    copied_source: Path,
    copied_evidence: dict[str, object],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    output = tmp_path_factory.mktemp("real-app-h8-built-candidate") / "candidate"
    original_input = candidate._FIXED_INPUT_ROOT
    original_output = candidate._FIXED_OUTPUT_ROOT
    original_validate = candidate.validate_fixed_source
    try:
        candidate._FIXED_INPUT_ROOT = copied_source
        candidate._FIXED_OUTPUT_ROOT = output
        candidate.validate_fixed_source = lambda: copy.deepcopy(copied_evidence)
        assert candidate.build_candidate() == output
    finally:
        candidate._FIXED_INPUT_ROOT = original_input
        candidate._FIXED_OUTPUT_ROOT = original_output
        candidate.validate_fixed_source = original_validate
    return output


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reseal_candidate(root: Path, payload: dict[str, object]) -> None:
    body = dict(payload)
    body.pop("candidate_id", None)
    payload["candidate_id"] = candidate._candidate_id(body)
    _write_json(root / "authority_candidate.json", payload)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["candidate_id"] = payload["candidate_id"]
    for name in ("REPORT.md", "authority_candidate.json"):
        path = root / name
        manifest["payload_files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": candidate._sha256_file(path),
        }
    _write_json(root / "manifest.json", manifest)

    index = json.loads((root / candidate.OUTPUT_INDEX).read_text(encoding="utf-8"))
    for name in ("REPORT.md", "authority_candidate.json", "manifest.json"):
        path = root / name
        index["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": candidate._sha256_file(path),
        }
    _write_json(root / candidate.OUTPUT_INDEX, index)


def test_real_candidate_output_has_not_been_built() -> None:
    assert candidate._FIXED_OUTPUT_ROOT == REAL_OUTPUT
    assert not REAL_OUTPUT.exists()


def test_contract_has_fixed_input_output_and_closed_cli() -> None:
    assert candidate._FIXED_INPUT_ROOT == REAL_SOURCE
    parser = candidate.build_parser()
    assert vars(parser.parse_args(["--validate-only"])) == {
        "validate_only": True,
        "build": False,
    }
    assert vars(parser.parse_args(["--build"])) == {
        "validate_only": False,
        "build": True,
    }
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--build", "--validate-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--build", "--output", "elsewhere"])


def test_source_never_imports_or_calls_runtime_selector_paths() -> None:
    tree = ast.parse(Path(candidate.__file__).read_text(encoding="utf-8"))
    forbidden_import_fragments = (
        "whole_atom_selector_lowry_r1h8",
        "transition_mean_anchor",
        "round_sim",
        "new_sim",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                all(fragment not in alias.name for fragment in forbidden_import_fragments)
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            assert all(
                fragment not in (node.module or "")
                for fragment in forbidden_import_fragments
            )
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in {
                "admit_lowry_r1h8_app_support",
                "_weight_fixture",
                "_select_fixture",
                "weight_real_app_support",
                "select_real_app_atom",
                "run_callback",
                "solve_mean",
                "simulate",
            }


def test_frozen_schema_and_semantic_projection_are_exact() -> None:
    assert len(candidate.EXPECTED_ARROW_SCHEMA) == 150
    assert len(candidate.SEMANTIC_SELECTOR_COLUMNS) == 54
    assert len(set(candidate.SEMANTIC_SELECTOR_COLUMNS)) == 54
    assert candidate.EXPECTED_ARROW_SCHEMA[0] == ("arm", "string")
    assert candidate.EXPECTED_ARROW_SCHEMA[-1] == ("terminal", "bool")
    assert tuple(name for name, _dtype in candidate.EXPECTED_ARROW_SCHEMA).index(
        "donor_event_start"
    ) == 25


def test_builder_binds_its_own_focused_test() -> None:
    relative = "test_hole_sim_real_app_h8_selector_authority_candidate.py"
    assert candidate.EXPECTED_SOURCE_BINDINGS[relative] == candidate._sha256_file(
        REPO / relative
    )


def test_temp_copy_validates_with_exact_counts_and_hashes(
    copied_evidence: dict[str, object],
) -> None:
    result = copied_evidence
    validation = result["validation"]
    assert result["artifact_sha256_start"] == result["artifact_sha256_end"]
    assert validation["counts"] == {
        "complete_atoms": 35_218,
        "proposals": 904,
        "spatial_proposals": 903,
        "terminal_atoms": 1,
        "surface_records": 903,
        "materialized_surface_branches": 5_418,
        "hazard_atoms": 30_702,
        "imputed_ability_atoms": 1_989,
        "donor_quality_groups": 6,
        "mixed_policy_groups": 4,
    }
    assert validation["full_schema_sha256"] == (
        "90f5176de0cbf1c52118a94a47ba35299bcdf9755e77c6350e88bc0342c65855"
    )
    assert validation["semantic_projection_schema_sha256"] == (
        "83b0b77a1894d6949d9236ca40bafe9fd9cb779d975dfb511933a82f7047d2a0"
    )
    assert validation["semantic_projection_sha256"] == (
        "1fa34f66cc3925ebc7d8ab12f6c8039629dfa3671f9369903a85ee1cf2217a4b"
    )
    assert validation["topology_sha256"] == (
        "7a7f6847c029dd1382084e1bcfc43c8454c1a09e93572a426e1bc19c237df9e2"
    )
    assert validation["group_policy_topology_sha256"] == (
        "b13eeb67a48bffbbc2cc8c728b3767f86bb032c85d26b873307f424bdbb02985"
    )
    assert validation["canonical_full_schema_sha256"] == (
        "113ef43b17f03ad9d9d7c9112d0636a87445ca9b97738935bec969f98b7b6a4d"
    )
    assert validation["canonical_projection_schema_sha256"] == (
        "855c4b57abfe5ea0f29e23f1e3ffab82e58b186f09358bb1adcff17eb59c87f2"
    )
    assert validation["canonical_full_support_sha256"] == (
        "b66b8703d1b156aedbadded022a9ba1f1bdd821c7f8b585103fa4f8de747dce7"
    )
    assert validation["canonical_projection_sha256"] == (
        "a9957e4e20ed0259ce90a5abf27d29b7d4dcc8ee7259362e41c7ba7c9c178889"
    )
    assert validation["canonical_selector_support_sha256"] == (
        "f5fd780ea6c2a3e2795c8197f45f3486cd4a03606bac8e8c17b0548d7a07dc4c"
    )
    assert validation["canonical_topology_sha256"] == (
        "9e8e3a3128aa12ddae281ed0b37e036bee833232aa01ca9b93bd0370552af484"
    )


def test_frame_rejects_a_split_whole_proposal_group(copied_frame) -> None:
    broken = copied_frame.copy()
    broken.loc[0, "donor_quality_group_id"] = "0" * 64
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="split across donor groups",
    ):
        candidate._validate_frame(broken)


def test_frame_rejects_availability_split_as_a_new_group(copied_frame) -> None:
    broken = copied_frame.copy()
    unavailable = broken.index[~broken["donor_ability_available"]][0]
    proposal = broken.loc[unavailable, "proposal_row_order"]
    broken.loc[broken["proposal_row_order"] == proposal, "donor_quality_group_id"] = (
        "f" * 64
    )
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="Six-group policy-v2 topology changed",
    ):
        candidate._validate_frame(broken)


def test_frame_rejects_frozen_spatial_cdf_split(copied_frame) -> None:
    broken = copied_frame.copy()
    broken.loc[1, "frozen_spatial_cdf_lower"] = 0.25
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="Frozen spatial CDF split within proposal",
    ):
        candidate._validate_frame(broken)


def test_frame_rejects_terminal_sentinel_change(copied_frame) -> None:
    broken = copied_frame.copy()
    broken.loc[broken["terminal"], "next_support_key"] = "not-terminal"
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="Terminal next_support_key changed",
    ):
        candidate._validate_frame(broken)


def test_frame_rejects_surface_cdf_or_hazard_identity_change(copied_frame) -> None:
    surface_broken = copied_frame.copy()
    surface_broken.loc[0, "surface_cdf_upper"] = 0.5
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="surface CDF bounds changed",
    ):
        candidate._validate_frame(surface_broken)

    hazard_broken = copied_frame.copy()
    penalty = hazard_broken.index[
        (hazard_broken["proposal_row_order"] == 0)
        & (hazard_broken["surface_class_index"] == 7)
    ]
    hazard_broken.loc[penalty[1], "hazard_donor_key"] = hazard_broken.loc[
        penalty[0], "hazard_donor_key"
    ]
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="hazard keys/stroke cost changed",
    ):
        candidate._validate_frame(hazard_broken)


def test_frame_rejects_whole_atom_mass_factorization_change(copied_frame) -> None:
    broken = copied_frame.copy()
    broken.loc[0, "quality_base_mass"] += 1e-8
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="quality base mass|factorization",
    ):
        candidate._validate_frame(broken)


def test_fail_closed_flags_reject_numeric_boolean_alias() -> None:
    payload = {
        "contract": candidate.CONTRACT,
        "admission_status": candidate.ADMISSION_STATUS,
        "authority_id": None,
        "authority_lock_sha256": None,
        **candidate._FAIL_CLOSED_FLAGS,
    }
    payload["weighting_enabled"] = 0
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="exact boolean False",
    ):
        candidate._assert_non_authorizing(payload)


def test_temp_copy_rejects_an_unindexed_addition(
    copied_source: Path, tmp_path: Path
) -> None:
    root = tmp_path / "source-with-extra"
    shutil.copytree(copied_source, root)
    (root / "unindexed.txt").write_text("not closed\n", encoding="utf-8")
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="closure changed",
    ):
        candidate._validation_summary(root)


def test_temp_candidate_build_is_hashes_counts_only_and_remains_hold(
    built_temp_candidate: Path,
) -> None:
    output = built_temp_candidate
    assert sorted(path.name for path in output.iterdir()) == [
        "OUTPUT_SHA256.json",
        "REPORT.md",
        "authority_candidate.json",
        "manifest.json",
    ]
    payload = json.loads(
        (output / "authority_candidate.json").read_text(encoding="utf-8")
    )
    candidate._assert_non_authorizing(payload)
    assert payload["validation"]["counts"]["complete_atoms"] == 35_218
    assert all(payload[field] is False for field in candidate._FAIL_CLOSED_FLAGS)
    assert payload["research_only"] is True
    assert "rows" not in payload
    assert "uniforms" not in payload
    assert "parameters" not in payload
    assert REAL_OUTPUT.exists() is False
    widened = dict(payload)
    widened["mystery_ready"] = True
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="field set changed",
    ):
        candidate._assert_non_authorizing(widened)


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    (
        (
            "fixed-input",
            lambda payload: payload["fixed_input_identity"].__setitem__(
                "outer_output_index_sha256", "0" * 64
            ),
            "fixed input identity",
        ),
        (
            "authority-evidence",
            lambda payload: payload.__setitem__(
                "authority_evidence_sha256", "1" * 64
            ),
            "authority evidence SHA-256",
        ),
        (
            "source-bindings",
            lambda payload: payload.__setitem__(
                "source_bindings_sha256", "2" * 64
            ),
            "source bindings SHA-256",
        ),
        (
            "fixed-caps-type",
            lambda payload: payload["fixed_caps"].__setitem__("parts", "5"),
            "fixed caps",
        ),
        (
            "review-bypass",
            lambda payload: payload["external_review_requirements"].__setitem__(
                "candidate_id_is_not_authority", False
            ),
            "external-review requirements",
        ),
    ),
)
def test_fully_resealed_nested_candidate_widening_is_rejected(
    built_temp_candidate: Path,
    tmp_path: Path,
    label: str,
    mutate,
    message: str,
) -> None:
    root = tmp_path / f"resealed-{label}"
    shutil.copytree(built_temp_candidate, root)
    payload = json.loads(
        (root / "authority_candidate.json").read_text(encoding="utf-8")
    )
    mutate(payload)
    _reseal_candidate(root, payload)
    with pytest.raises(candidate.SelectorAuthorityCandidateError, match=message):
        candidate.validate_candidate_artifact(root, verify_fixed_source=False)


def test_candidate_tree_change_during_long_source_verify_is_rejected(
    built_temp_candidate: Path,
    copied_evidence: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "candidate-tree-toctou"
    shutil.copytree(built_temp_candidate, root)

    def mutate_during_verify() -> dict[str, object]:
        (root / "REPORT.md").write_text(
            "changed during fixed-source validation\n", encoding="utf-8"
        )
        payload = json.loads(
            (root / "authority_candidate.json").read_text(encoding="utf-8")
        )
        _reseal_candidate(root, payload)
        return copy.deepcopy(copied_evidence)

    monkeypatch.setattr(candidate, "validate_fixed_source", mutate_during_verify)
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="candidate tree start/end closure",
    ):
        candidate.validate_candidate_artifact(root, verify_fixed_source=True)


def test_build_rejects_postwrite_fixed_input_change(
    copied_source: Path,
    copied_evidence: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    late_source = tmp_path / "late-source-change"
    shutil.copytree(copied_source, late_source)
    output = tmp_path / "candidate-after-input-change"
    original_write_index = candidate._write_output_index

    def write_then_change_source(root: Path) -> None:
        original_write_index(root)
        (late_source / "late-unindexed-evidence.txt").write_text(
            "changed after evidence capture\n", encoding="utf-8"
        )

    monkeypatch.setattr(candidate, "_FIXED_INPUT_ROOT", late_source)
    monkeypatch.setattr(candidate, "_FIXED_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        candidate,
        "validate_fixed_source",
        lambda: copy.deepcopy(copied_evidence),
    )
    monkeypatch.setattr(candidate, "_write_output_index", write_then_change_source)
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="closure changed",
    ):
        candidate.build_candidate()


def test_build_rejects_postwrite_source_binding_change(
    copied_source: Path,
    copied_evidence: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "candidate-after-code-change"
    changed_source = copy.deepcopy(copied_evidence["source_sha256_end"])
    changed_source["source_closure_sha256"] = "f" * 64
    monkeypatch.setattr(candidate, "_FIXED_INPUT_ROOT", copied_source)
    monkeypatch.setattr(candidate, "_FIXED_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        candidate,
        "validate_fixed_source",
        lambda: copy.deepcopy(copied_evidence),
    )
    monkeypatch.setattr(candidate, "_source_state", lambda: changed_source)
    with pytest.raises(
        candidate.SelectorAuthorityCandidateError,
        match="post-write source closure",
    ):
        candidate.build_candidate()


def test_build_refuses_a_preexisting_fixed_output(
    copied_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    monkeypatch.setattr(candidate, "_FIXED_INPUT_ROOT", copied_source)
    monkeypatch.setattr(candidate, "_FIXED_OUTPUT_ROOT", output)
    with pytest.raises(FileExistsError, match="must be absent"):
        candidate.build_candidate()

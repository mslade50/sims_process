from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ARCHIVE = Path(__file__).resolve().parent / "archive"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import hole_sim_donor_ability_policy as policy  # noqa: E402
import hole_sim_population_transition_support_all18 as runner  # noqa: E402


def _atoms() -> pd.DataFrame:
    donor_start = pd.Timestamp("2025-06-19", tz="UTC")
    target_start = pd.Timestamp("2026-06-25", tz="UTC")
    return pd.DataFrame(
        {
            "support_key": ["support-a", "support-a", "support-b"],
            "atom_id": ["atom-1", "atom-2", "atom-3"],
            "category": ["arg", "putt", "putt"],
            "donor_player_id": ["10", "11", "12"],
            "donor_event_key": ["pga:R2025001"] * 3,
            "donor_event_start": [donor_start] * 3,
            "target_event_key": ["pga:R2026034"] * 3,
            "target_event_start": [target_start] * 3,
            "donor_skill_date": [
                pd.Timestamp("2025-06-18", tz="UTC"),
                pd.Timestamp("2025-06-17", tz="UTC"),
                pd.NaT,
            ],
            "donor_ability_available": [True, False, False],
            "donor_ability_sg": [2.5, np.nan, np.nan],
        }
    )


def test_real_parent_dry_run_is_closed_and_nonlaunching(tmp_path: Path) -> None:
    output = tmp_path / "unused-output"
    result = runner.dry_run(runner.DEFAULT_PARENT, output)
    assert result["admission_valid"] is True
    assert result["simulation_launched"] is False
    assert result["parent"]["files_verified"] == 7
    assert result["selected_population_rows"] == 9562
    assert result["expected_partitions"] == 72
    assert result["unique_queries"] == 6978
    assert result["maximum_atoms"] == 6978 * 64
    assert result["output_absent"] is True
    assert not output.exists()


def test_parent_dependency_graph_is_hash_bound() -> None:
    manifest = json.loads(
        (runner.DEFAULT_PARENT / "manifest.json").read_text(encoding="utf-8")
    )
    attestation = runner.validate_frozen_sources(manifest)
    assert attestation["parent_dependencies_all_exact"] is True
    assert attestation["derived_sources_all_exact"] is True
    changed = json.loads(json.dumps(manifest))
    changed["inputs"]["dependency_sha256_end"]["streaming_source"] = "0" * 64
    with pytest.raises(runner.PopulationAll18Error, match="streaming_source"):
        runner.validate_frozen_sources(changed)
    changed = json.loads(json.dumps(manifest))
    changed["inputs"]["dependency_sha256_end"].pop("fixed_reference_source")
    with pytest.raises(runner.PopulationAll18Error, match="key set"):
        runner.validate_frozen_sources(changed)


def test_partition_population_ledger_is_exact_and_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["raw", "raw", "normalized", "raw"],
            "round_no": [1, 1, 1, 1],
            "hole_no": [1, 1, 1, 2],
            "category": ["putt", "arg", "arg", "putt"],
            "particle_id": [2, 1, 0, 3],
            "atom_sequence": [4, 2, 1, 3],
        }
    )
    parts = runner.partition_population_ledger(frame)
    assert [key for key, _ in parts] == [
        ("normalized", 1, 1, "arg"),
        ("raw", 1, 1, "arg"),
        ("raw", 1, 1, "putt"),
        ("raw", 1, 2, "putt"),
    ]
    assert sum(len(part) for _, part in parts) == len(frame)


def test_partition_rejects_nonpopulation_category() -> None:
    frame = pd.DataFrame(
        {
            "arm": ["raw"],
            "round_no": [1],
            "hole_no": [1],
            "category": ["app"],
            "particle_id": [0],
            "atom_sequence": [0],
        }
    )
    with pytest.raises(ValueError, match="non-ARG/PUTT"):
        runner.partition_population_ledger(frame)


def test_policy_overlay_preserves_missingness_and_clips_only_observed() -> None:
    overlay = runner.build_policy_overlay(_atoms())
    audit = runner.validate_policy_overlay(overlay, _atoms())
    assert audit == {
        "rows": 3,
        "observed_rows": 1,
        "imputed_rows": 2,
        "clipped_rows": 1,
        "dated_missing_rows": 1,
        "undated_missing_rows": 1,
        "policy_identity_sha256": runner.EXPECTED_ABILITY_POLICY_IDENTITY,
        "all_exact": True,
    }
    assert overlay["donor_ability_sg"].tolist() == [2.0, 0.0, 0.0]
    assert overlay["donor_ability_available"].tolist() == [True, False, False]
    assert overlay.loc[1, "donor_skill_date"] == pd.Timestamp(
        "2025-06-17", tz="UTC"
    )
    assert pd.isna(overlay.loc[2, "donor_skill_date"])
    assert overlay["donor_ability_policy_sha256"].eq(
        policy.policy_payload()["policy_identity_sha256"]
    ).all()


def test_policy_overlay_rejects_duplicate_support_atom_identity() -> None:
    atoms = _atoms()
    atoms.loc[1, ["support_key", "atom_id"]] = atoms.loc[
        0, ["support_key", "atom_id"]
    ].to_numpy()
    with pytest.raises(runner.PopulationAll18Error, match="not unique"):
        runner.build_policy_overlay(atoms)


def test_policy_overlay_mutation_rejects() -> None:
    atoms = _atoms()
    overlay = runner.build_policy_overlay(atoms)
    overlay.loc[1, "donor_ability_sg"] = 0.25
    with pytest.raises(policy.DonorAbilityPolicyError):
        runner.validate_policy_overlay(overlay, atoms)


def test_partition_relative_dir_is_closed() -> None:
    assert runner.partition_relative_dir(("raw", 1, 8, "arg")) == Path(
        "partitions/raw/R1H08/arg"
    )
    with pytest.raises(ValueError):
        runner.partition_relative_dir(("raw", 2, 8, "arg"))
    with pytest.raises(ValueError):
        runner.partition_relative_dir(("other", 1, 8, "arg"))


def test_rss_cap_is_hard_ceiling() -> None:
    assert runner.validate_rss_cap(4 * 2**30) == 4 * 2**30
    assert runner.validate_rss_cap(2 * 2**30) == 2 * 2**30
    with pytest.raises(ValueError, match="positive"):
        runner.validate_rss_cap(0)
    with pytest.raises(ValueError, match="cannot exceed"):
        runner.validate_rss_cap(4 * 2**30 + 1)


def test_verified_parquet_roundtrip_binds_input(tmp_path: Path) -> None:
    frame = runner.build_policy_overlay(_atoms())
    path = tmp_path / "policy.parquet"
    persisted, logical = runner.write_verified_parquet(frame, path)
    pd.testing.assert_frame_equal(
        persisted,
        frame,
        check_dtype=False,
        check_exact=True,
        check_categorical=False,
    )
    assert logical


def test_output_index_requires_exact_recursive_set(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    payload = nested / "payload.txt"
    payload.write_text("bound\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    index = {
        "contract": runner.OUTPUT_INDEX_CONTRACT,
        "directories": ["nested"],
        "files": {
            "manifest.json": {
                "bytes": manifest.stat().st_size,
                "sha256": sha256(manifest.read_bytes()).hexdigest(),
            },
            "nested/payload.txt": {
                "bytes": payload.stat().st_size,
                "sha256": sha256(payload.read_bytes()).hexdigest(),
            },
        },
    }
    (tmp_path / "OUTPUT_SHA256.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    assert runner.verify_output_index(tmp_path)["files_verified"] == 2
    extra = tmp_path / "nested" / "extra.txt"
    extra.write_text("not indexed\n", encoding="utf-8")
    with pytest.raises(runner.PopulationAll18Error, match="exact file closure"):
        runner.verify_output_index(tmp_path)


def test_output_index_does_not_exempt_nested_index_name(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    payload = nested / "payload.txt"
    payload.write_text("bound\n", encoding="utf-8")
    index = {
        "contract": runner.OUTPUT_INDEX_CONTRACT,
        "directories": ["nested"],
        "files": {
            "nested/payload.txt": {
                "bytes": payload.stat().st_size,
                "sha256": sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    (tmp_path / "OUTPUT_SHA256.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    assert runner.verify_output_index(tmp_path)["files_verified"] == 1

    nested_index = nested / "OUTPUT_SHA256.json"
    nested_index.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.PopulationAll18Error, match="exact file closure"):
        runner.verify_output_index(tmp_path)

    rebuilt = runner._output_index(tmp_path)
    assert "nested/OUTPUT_SHA256.json" in rebuilt["files"]


def test_output_index_rejects_unindexed_empty_directory(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("bound\n", encoding="utf-8")
    index = {
        "contract": runner.OUTPUT_INDEX_CONTRACT,
        "directories": [],
        "files": {
            "payload.txt": {
                "bytes": payload.stat().st_size,
                "sha256": sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }
    (tmp_path / "OUTPUT_SHA256.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    assert runner.verify_output_index(tmp_path)["files_verified"] == 1
    (tmp_path / "extra-directory").mkdir()
    with pytest.raises(runner.PopulationAll18Error, match="directory closure"):
        runner.verify_output_index(tmp_path)

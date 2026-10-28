from __future__ import annotations

from pathlib import Path

import pytest

from archive import research_execution_provenance as provenance


def test_execution_provenance_snapshots_source_and_records_runtime(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_bytes(b"print('frozen evaluator')\n")
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"immutable input")
    output = tmp_path / "output"
    state = provenance.begin(
        output_dir=output,
        primary_source=source,
        input_paths=[input_path],
        cli=["python", "evaluator.py", "--sealed"],
    )
    snapshot = output / "execution_provenance" / "00__evaluator.py"
    assert snapshot.read_bytes() == source.read_bytes()
    payload = provenance.finalize(state)
    assert payload["complete"]
    assert payload["verified_unchanged_before_complete_manifest"]
    assert payload["cli"] == ["python", "evaluator.py", "--sealed"]
    assert payload["runtime"]["numpy"]
    assert payload["inputs"][0]["sha256"] == provenance.sha256(input_path)


def test_execution_provenance_refuses_midrun_input_change(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_bytes(b"pass\n")
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(b"before")
    state = provenance.begin(
        output_dir=tmp_path / "output",
        primary_source=source,
        input_paths=[input_path],
    )
    input_path.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="input changed"):
        provenance.finalize(state)


def test_execution_provenance_refuses_midrun_preserved_source_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluator.py"
    source.write_bytes(b"pass\n")
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(b"before")
    output = tmp_path / "output"
    state = provenance.begin(
        output_dir=output,
        primary_source=source,
        input_paths=[input_path],
    )
    (output / "execution_provenance" / "00__evaluator.py").write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="preserved evaluator source changed"):
        provenance.finalize(state)


def test_assert_current_rejects_mutation_after_finalize(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_bytes(b"pass\n")
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(b"before")
    state = provenance.begin(
        output_dir=tmp_path / "output",
        primary_source=source,
        input_paths=[input_path],
    )
    provenance.finalize(state)
    input_path.write_bytes(b"after finalize")
    with pytest.raises(RuntimeError, match="input changed"):
        provenance.assert_current(state)

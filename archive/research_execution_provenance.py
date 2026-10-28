"""Immutable execution provenance for long-running research evaluators."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ExecutionProvenance:
    output_dir: Path
    primary_source: Path
    source_fingerprints: tuple[dict[str, Any], ...]
    source_copy_fingerprints: tuple[dict[str, Any], ...]
    input_fingerprints: tuple[dict[str, Any], ...]
    start_utc: str
    cli: tuple[str, ...]
    pid: int


def begin(
    *,
    output_dir: Path,
    primary_source: Path,
    input_paths: Iterable[Path],
    code_paths: Iterable[Path] = (),
    cli: Sequence[str] | None = None,
) -> ExecutionProvenance:
    """Snapshot code bytes and fingerprint every declared direct input."""

    output_dir = output_dir.resolve()
    provenance_dir = output_dir / "execution_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    unique_sources = list(
        dict.fromkeys(
            path.resolve() for path in (primary_source, *tuple(code_paths))
        )
    )
    source_fingerprints = tuple(fingerprint(path) for path in unique_sources)
    unique_inputs = list(dict.fromkeys(path.resolve() for path in input_paths))
    input_fingerprints = tuple(fingerprint(path) for path in unique_inputs)
    source_copy_fingerprints: list[dict[str, Any]] = []
    for index, source in enumerate(unique_sources):
        destination = provenance_dir / f"{index:02d}__{source.name}"
        shutil.copyfile(source, destination)
        if sha256(destination) != source_fingerprints[index]["sha256"]:
            raise AssertionError(f"source snapshot copy mismatch: {source}")
        source_copy_fingerprints.append(fingerprint(destination))
    start_utc = pd.Timestamp.utcnow().isoformat()
    state = ExecutionProvenance(
        output_dir=output_dir,
        primary_source=primary_source.resolve(),
        source_fingerprints=source_fingerprints,
        source_copy_fingerprints=tuple(source_copy_fingerprints),
        input_fingerprints=input_fingerprints,
        start_utc=start_utc,
        cli=tuple(sys.argv if cli is None else cli),
        pid=os.getpid(),
    )
    _atomic_json(
        provenance_dir / "execution_provenance.incomplete.json",
        {
            "complete": False,
            "start_utc": start_utc,
            "pid": state.pid,
            "cli": list(state.cli),
            "primary_source": str(state.primary_source),
            "sources": list(source_fingerprints),
            "preserved_sources": source_copy_fingerprints,
            "inputs": list(input_fingerprints),
        },
    )
    return state


def _verify_unchanged(records: Sequence[dict[str, Any]], label: str) -> None:
    failures: list[dict[str, Any]] = []
    for original in records:
        path = Path(str(original["path"]))
        try:
            current = fingerprint(path)
        except FileNotFoundError:
            failures.append({"path": str(path), "reason": "missing"})
            continue
        changed = {
            field: {"start": original[field], "end": current[field]}
            for field in ("sha256", "bytes", "mtime_ns")
            if current[field] != original[field]
        }
        if changed:
            failures.append({"path": str(path), "changed": changed})
    if failures:
        raise RuntimeError(f"{label} changed during execution: {failures}")


def assert_current(state: ExecutionProvenance) -> None:
    """Re-hash the complete execution snapshot without writing a manifest."""

    _verify_unchanged(state.source_fingerprints, "evaluator source")
    _verify_unchanged(state.source_copy_fingerprints, "preserved evaluator source")
    _verify_unchanged(state.input_fingerprints, "evaluator input")
    for source, preserved in zip(
        state.source_fingerprints, state.source_copy_fingerprints
    ):
        if source["sha256"] != preserved["sha256"] or source["bytes"] != preserved["bytes"]:
            raise RuntimeError("preserved evaluator source copy no longer matches source")


def finalize(state: ExecutionProvenance) -> dict[str, Any]:
    """Re-hash code and inputs; write completion provenance only if immutable."""

    assert_current(state)
    end_utc = pd.Timestamp.utcnow().isoformat()
    payload = {
        "complete": True,
        "start_utc": state.start_utc,
        "end_utc": end_utc,
        "pid": state.pid,
        "cli": list(state.cli),
        "primary_source": str(state.primary_source),
        "sources": list(state.source_fingerprints),
        "preserved_sources": list(state.source_copy_fingerprints),
        "inputs": list(state.input_fingerprints),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "verified_unchanged_before_complete_manifest": True,
    }
    path = state.output_dir / "execution_provenance" / "execution_provenance.json"
    _atomic_json(path, payload)
    return payload

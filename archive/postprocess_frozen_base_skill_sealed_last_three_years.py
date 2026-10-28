"""Provenance-bound exact-last-three-years slice of a completed sealed run.

This research-only postprocessor does not score, select, or refit a model.  It
accepts only a *completed* full sealed matchup output, verifies every recorded
artifact and the full frozen handoff/input authority, and then filters the
already materialized canonical probabilities and bets to 2023-08-22 through
2026-08-22 inclusive.  Proper scores and betting results are recomputed from
those event-level records.  The source sealed evaluator is intentionally left
unchanged.

The program is safe to import: sealed outcomes are read only from ``run()``.
Use a fresh output directory for every execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import backtest_base_skill_market_models as market_eval  # noqa: E402
from archive import evaluate_complete_base_skill_market_candidates as discovery  # noqa: E402
from archive import evaluate_frozen_base_skill_sealed_matchups as sealed  # noqa: E402
from archive import market_backtest_scoring as market_scoring  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402
from archive.base_skill_frozen_handoff import (  # noqa: E402
    FrozenHandoff,
    validate_frozen_handoff_chain,
)


ROOT = REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
DEFAULT_OUTPUT = ROOT / "model_eval_complete_sealed_exact_3y_v1_provenance"
SLICE_START = pd.Timestamp("2023-08-22")
SLICE_END = pd.Timestamp("2026-08-22")
SOURCE_START = sealed.SEALED_START
SOURCE_END = sealed.SEALED_END
OUTPUT_CONTRACT = "base_skill_frozen_sealed_exact_3y_postprocess_v1"
SOURCE_CONTRACT = sealed.OUTPUT_CONTRACT
SAMPLE = "sealed_exact_2023-08-22_2026-08-22"
BASELINE = sealed.STRICT_PT_BASELINE

# These pins deliberately make the postprocessor stop if the audited sealed
# semantics or any directly reused scoring/provenance implementation changes.
# Updating a pin requires a new reviewed postprocessor version and fresh output.
DEPENDENCY_PINS: Mapping[str, tuple[int, str]] = {
    "archive/evaluate_frozen_base_skill_sealed_matchups.py": (
        126_344,
        "b9d5d74621568497b85ab73d641617ab17877f9107d2d7c3eaaf1097198c0500",
    ),
    "archive/backtest_base_skill_market_models.py": (
        114_608,
        "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d",
    ),
    "archive/base_skill_frozen_handoff.py": (
        43_142,
        "b072b2fdf18769db99e65f633ea24c28c5491fa1c659b0da300b59d7c9fc3a4f",
    ),
    "archive/evaluate_complete_base_skill_market_candidates.py": (
        175_983,
        "1818f08cd03a8bde71058223686798c5eaf4aa68f714664671e7f42eb7442c49",
    ),
    "archive/research_execution_provenance.py": (
        6_596,
        "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034",
    ),
    "archive/market_backtest_scoring.py": (
        35_158,
        "53681535e0d1da7bde9cccede5ac09d7943e7493160c1ac2a248a0c0bfae2608",
    ),
}
CODE_PATHS = tuple((REPO / relative).resolve() for relative in DEPENDENCY_PINS)

REQUIRED_SOURCE_ARTIFACTS = {
    "preflight.json",
    "audit.json",
    "candidate_roles_and_metadata.csv",
    "sealed_logical_probabilities.parquet",
    "proper_scores.csv",
    "paired_common_proper_scores.csv",
    "proper_score_event_deltas.parquet",
    "betting_performance.csv",
    "event_betting_performance.parquet",
    "money_event_deltas.parquet",
    "paired_event_bootstrap.csv",
    "block_max_stat_reality_check.csv",
    "execution_provenance/execution_provenance.json",
}


@dataclass(frozen=True)
class SourceAuthority:
    output_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    artifact_paths: Mapping[str, Path]
    preflight: Mapping[str, Any]
    audit: Mapping[str, Any]
    execution: Mapping[str, Any]
    handoff: FrozenHandoff
    panel_authority: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _strict_numeric(
    series: pd.Series, label: str, *, allow_null: bool = False
) -> pd.Series:
    raw = series.copy()
    invalid_type = raw.notna() & ~raw.map(
        lambda value: isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if invalid_type.any():
        raise ValueError(f"{label} contains a nonnumeric or boolean value")
    numeric = pd.to_numeric(raw, errors="coerce")
    if not allow_null and numeric.isna().any():
        raise ValueError(f"{label} contains a null value")
    return numeric


def verify_dependency_pins(
    pins: Mapping[str, tuple[int, str]] = DEPENDENCY_PINS,
) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_hash) in pins.items():
        path = (REPO / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = int(path.stat().st_size)
        actual_hash = _sha256(path)
        if actual_bytes != int(expected_bytes) or actual_hash != str(expected_hash).lower():
            raise ValueError(f"dependency source pin mismatch: {relative}")
        audit[relative] = {
            "path": str(path),
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }
    return audit


def _safe_artifact_path(output_dir: Path, raw: object) -> tuple[str, Path]:
    text = str(raw)
    candidate = Path(text)
    if (
        not text
        or candidate.is_absolute()
        or bool(candidate.drive)
        or bool(candidate.root)
        or "\\" in text
    ):
        raise ValueError(f"unsafe source artifact path: {text!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe source artifact path: {text!r}")
    resolved = (output_dir / candidate).resolve()
    try:
        relative = resolved.relative_to(output_dir.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"source artifact escapes output directory: {text!r}") from error
    if relative != candidate.as_posix():
        raise ValueError(f"source artifact path is not canonical: {text!r}")
    return relative, resolved


def validate_artifact_inventory(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    required: Iterable[str] = REQUIRED_SOURCE_ARTIFACTS,
) -> dict[str, Path]:
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("completed sealed manifest has no artifact inventory")
    paths: dict[str, Path] = {}
    canonical_targets: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("sealed artifact record is malformed")
        if set(("path", "bytes", "sha256")) - set(record):
            raise ValueError("sealed artifact record lacks path/bytes/sha256")
        relative, path = _safe_artifact_path(output_dir, record["path"])
        if relative in paths or path in canonical_targets:
            raise ValueError(f"duplicate sealed artifact record: {relative}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sealed artifact is missing or is a symlink: {relative}")
        raw_bytes = record["bytes"]
        if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool):
            raise ValueError(f"sealed artifact byte count is not an integer: {relative}")
        expected_bytes = int(raw_bytes)
        expected_hash = str(record["sha256"]).lower()
        if (
            expected_bytes < 0
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"sealed artifact metadata is malformed: {relative}")
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_hash:
            raise ValueError(f"sealed artifact bytes differ from manifest: {relative}")
        paths[relative] = path
        canonical_targets.add(path)
    missing = sorted(set(required) - set(paths))
    if missing:
        raise ValueError(f"completed sealed output lacks required artifacts: {missing}")
    if not any(name.startswith("bet_partitions/") and name.endswith(".parquet") for name in paths):
        raise ValueError("completed sealed output has no materialized bet partitions")
    symlinks = [path for path in output_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("sealed output contains an unmanifested symlink")
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != (output_dir / "manifest.json").resolve()
    }
    declared = set(paths)
    if actual != declared:
        raise ValueError(
            "sealed output file inventory differs from manifest: "
            f"untracked={sorted(actual - declared)[:10]}, missing={sorted(declared - actual)[:10]}"
        )
    return paths


def _verified_json(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _verify_record(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(record.get("sha256", "")).lower()
    raw_bytes = record.get("bytes", -1)
    if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool):
        raise ValueError(f"{label} has malformed provenance byte count")
    expected_bytes = int(raw_bytes)
    if (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
        or expected_bytes < 0
    ):
        raise ValueError(f"{label} has malformed provenance metadata")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_hash:
        raise ValueError(f"{label} no longer matches recorded provenance")


def validate_completed_source_output(
    output_dir: Path, expected_manifest_sha256: str
) -> SourceAuthority:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected_parent_hash = str(expected_manifest_sha256).strip().lower()
    if len(expected_parent_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_parent_hash
    ):
        raise ValueError("--expected-sealed-manifest-sha256 must be 64 hex characters")
    if _sha256(manifest_path) != expected_parent_hash:
        raise ValueError("source sealed manifest differs from the externally supplied hash")
    manifest = _verified_json(manifest_path, "sealed manifest")
    if manifest.get("contract") != SOURCE_CONTRACT:
        raise ValueError("source output has the wrong sealed contract")
    if manifest.get("complete") is not True or manifest.get("research_only") is not True:
        raise ValueError("source sealed output is not complete research-only output")
    for key in (
        "sealed_data_used_for_selection",
        "sealed_results_used_for_selection",
        "uses_closing_odds_as_feature",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"source sealed manifest violates {key}=False")
    if manifest.get("frozen_before_sealed_scoring") is not True:
        raise ValueError("source model was not frozen before sealed scoring")
    paths = validate_artifact_inventory(output_dir, manifest)
    preflight = _verified_json(paths["preflight.json"], "sealed preflight")
    audit = _verified_json(paths["audit.json"], "sealed audit")
    execution = _verified_json(
        paths["execution_provenance/execution_provenance.json"],
        "sealed execution provenance",
    )
    if execution.get("complete") is not True or execution.get(
        "verified_unchanged_before_complete_manifest"
    ) is not True:
        raise ValueError("source sealed execution provenance is incomplete")
    for record in execution.get("inputs", []):
        if not isinstance(record, Mapping):
            raise ValueError("source execution input record is malformed")
        _verify_record(Path(str(record.get("path", ""))).resolve(), record, "source execution input")
    source_records = execution.get("sources")
    if not isinstance(source_records, list):
        raise ValueError("source execution provenance lacks sources")
    expected_evaluator = DEPENDENCY_PINS[
        "archive/evaluate_frozen_base_skill_sealed_matchups.py"
    ][1]
    if expected_evaluator not in {
        str(record.get("sha256", "")).lower()
        for record in source_records
        if isinstance(record, Mapping)
    }:
        raise ValueError("source run did not execute the pinned sealed evaluator")
    preserved = execution.get("preserved_sources")
    if not isinstance(preserved, list) or len(preserved) != len(source_records):
        raise ValueError("source execution preserved-source closure is incomplete")
    for source_record, copy_record in zip(source_records, preserved):
        if not isinstance(source_record, Mapping) or not isinstance(copy_record, Mapping):
            raise ValueError("source execution source record is malformed")
        _verify_record(
            Path(str(source_record.get("path", ""))).resolve(),
            source_record,
            "source evaluator dependency",
        )
        _verify_record(Path(str(copy_record.get("path", ""))).resolve(), copy_record, "preserved source")
        if (
            str(source_record.get("sha256", "")).lower()
            != str(copy_record.get("sha256", "")).lower()
            or int(source_record.get("bytes", -1)) != int(copy_record.get("bytes", -2))
        ):
            raise ValueError("source and preserved source bytes disagree")

    compact_manifest_path = Path(str(preflight.get("compact_manifest", ""))).resolve()
    selection_path = Path(str(preflight.get("selection_manifest", ""))).resolve()
    if _sha256(compact_manifest_path) != str(preflight.get("compact_manifest_sha256", "")).lower():
        raise ValueError("sealed preflight compact-manifest hash mismatch")
    if _sha256(selection_path) != str(preflight.get("selection_manifest_sha256", "")).lower():
        raise ValueError("sealed preflight selection-manifest hash mismatch")
    handoff = validate_frozen_handoff_chain(compact_manifest_path)
    if handoff.selection_manifest_path != selection_path:
        raise ValueError("source sealed output points to a different frozen selection")
    if str(manifest.get("selected_model")) != handoff.selected_name:
        raise ValueError("source sealed selected model differs from frozen handoff")
    if str(preflight.get("selected_model")) != handoff.selected_name:
        raise ValueError("sealed preflight selected model differs from frozen handoff")
    if str(manifest.get("selection_manifest_sha256", "")).lower() != _sha256(selection_path):
        raise ValueError("source manifest selection hash differs from handoff")
    if str(manifest.get("compact_manifest_sha256", "")).lower() != _sha256(compact_manifest_path):
        raise ValueError("source manifest compact hash differs from handoff")
    if str(manifest.get("source_prediction_sha256", "")).lower() != _sha256(
        handoff.source_prediction_path
    ):
        raise ValueError("source manifest prediction hash differs from handoff")

    panel = preflight.get("panel_authority")
    if not isinstance(panel, Mapping):
        raise ValueError("sealed preflight lacks panel authority")
    round_panel = Path(str(panel.get("round_panel_path", ""))).resolve()
    event_crosswalk = Path(str(panel.get("event_crosswalk_path", ""))).resolve()
    panel_authority = sealed.validate_sealed_panel_authorities(
        handoff, round_panel, event_crosswalk
    )
    for field in (
        "root_panel_manifest_sha256",
        "round_panel_sha256",
        "event_crosswalk_sha256",
    ):
        if str(manifest.get(field, "")).lower() != str(panel_authority[field]).lower():
            raise ValueError(f"source manifest {field} differs from current authority")
    if audit.get("production_files_changed") is not False:
        raise ValueError("source audit does not preserve production")
    if audit.get("sealed_candidate_selection_performed") is not False:
        raise ValueError("source audit reports sealed candidate selection")
    if audit.get("sealed_period") != f"{SOURCE_START.date()}/{SOURCE_END.date()}":
        raise ValueError("source sealed audit has the wrong full period")
    if int(audit.get("canonical_probability_duplicates", -1)) != 0:
        raise ValueError("source audit reports duplicate canonical probabilities")
    if int(audit.get("push_pnl_violations", -1)) != 0:
        raise ValueError("source audit reports push settlement violations")
    if sorted(map(str, manifest.get("evaluated_candidate_names", []))) != sorted(
        map(str, preflight.get("evaluated_candidate_names", []))
    ):
        raise ValueError("source manifest/preflight candidate menus disagree")
    expected_execution_inputs = {
        Path(path).resolve()
        for path in (*handoff.verified_paths, *panel_authority["verified_paths"])
    }
    observed_execution_inputs = {
        Path(str(record["path"])).resolve()
        for record in execution.get("inputs", [])
        if isinstance(record, Mapping)
    }
    if observed_execution_inputs != expected_execution_inputs:
        raise ValueError(
            "source execution input set differs from full handoff/panel authority"
        )
    if manifest.get("opener_blend_sensitivity") != preflight.get(
        "opener_blend_sensitivity"
    ):
        raise ValueError("source manifest/preflight opener sensitivity disagree")
    for key, expected in (
        ("books", list(market_eval.BOOKS)),
        ("line_policies", list(sealed.LINE_POLICIES)),
        ("price_bases", list(sealed.PRICE_BASES)),
        ("sizing_rules", list(sealed.SIZINGS)),
        ("reliability_policies", list(sealed.RELIABILITY_POLICIES)),
    ):
        if audit.get(key) != expected:
            raise ValueError(f"source audit has the wrong frozen {key}")
    return SourceAuthority(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        artifact_paths=paths,
        preflight=preflight,
        audit=audit,
        execution=execution,
        handoff=handoff,
        panel_authority=panel_authority,
    )


def load_authorized_quotes(authority: SourceAuthority) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = authority.handoff.selection
    quote_entry = selection.get("inputs", {}).get("normalized_quotes")
    if not isinstance(quote_entry, Mapping):
        raise ValueError("frozen selection lacks normalized-quotes authority")
    quote_path = Path(str(quote_entry.get("path", ""))).resolve()
    _, quotes, quote_audit = sealed.load_catalog_and_quotes(
        selection,
        authority.handoff.candidate_catalog_path,
        quote_path.parent,
    )
    recorded = authority.preflight.get("market_audit")
    if not isinstance(recorded, Mapping):
        raise ValueError("sealed preflight lacks market audit")
    for key in ("quote_rows", "events", "markets", "books", "date_min", "date_max"):
        if quote_audit.get(key) != recorded.get(key):
            raise ValueError(f"revalidated quote authority differs on {key}")
    quotes = quotes.copy()
    quotes["event_start_date"] = pd.to_datetime(
        quotes["event_start_date"], errors="raise"
    ).dt.normalize()
    sliced = quotes.loc[
        quotes["event_start_date"].between(SLICE_START, SLICE_END)
    ].copy()
    if sliced.empty or set(sealed.MARKETS) - set(sliced["market"].astype(str)):
        raise ValueError("exact three-year slice lacks both frozen matchup markets")
    return sliced, quote_audit


def validate_and_slice_logical(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
        "sigma",
        "tie_threshold",
    }
    _require_columns(frame, required, "sealed logical probabilities")
    work = frame.copy()
    work["event_start_date"] = pd.to_datetime(
        work["event_start_date"], errors="raise"
    ).dt.normalize()
    if work["event_start_date"].isna().any() or not work[
        "event_start_date"
    ].between(SOURCE_START, SOURCE_END).all():
        raise ValueError("sealed logical probabilities exceed the full source period")
    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    if work[keys].isna().any().any() or work.duplicated(keys).any():
        raise ValueError("sealed canonical probability keys are null or duplicated")
    if set(work["market"].astype(str)) - set(sealed.MARKETS):
        raise ValueError("sealed logical probabilities contain an unexpected market")
    for column in ("canonical_probability", "canonical_push_probability", "sigma", "tie_threshold"):
        work[column] = _strict_numeric(work[column], f"sealed logical {column}")
        if not np.isfinite(work[column]).all():
            raise ValueError(f"sealed logical {column} is non-finite")
    raw_outcome = work["canonical_outcome"]
    outcome = _strict_numeric(
        raw_outcome, "sealed logical outcomes", allow_null=True
    )
    invalid_outcome = outcome.notna() & ~outcome.isin([0.0, 0.5, 1.0])
    if invalid_outcome.any():
        raise ValueError("sealed logical outcomes are outside {0, 0.5, 1}")
    work["canonical_outcome"] = outcome
    outcome_keys = ["mapped_event_key", "market", "pair_key"]
    if (
        work.groupby(outcome_keys, observed=True)["canonical_outcome"]
        .nunique(dropna=False)
        .gt(1)
        .any()
    ):
        raise ValueError("canonical outcomes disagree across candidate rows")
    probability = work["canonical_probability"]
    push = work["canonical_push_probability"]
    if (
        probability.lt(0.0).any()
        or probability.gt(1.0).any()
        or push.lt(0.0).any()
        or push.ge(1.0).any()
        or (probability + push).gt(1.0 + 1e-12).any()
        or work["sigma"].le(0.0).any()
        or work["tie_threshold"].lt(0.0).any()
    ):
        raise ValueError("sealed logical probability simplex or mapper parameters are invalid")
    if "probability_fit_end" not in work:
        raise ValueError("sealed logical probabilities lack frozen mapper fit end")
    if work["probability_fit_end"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).any():
        raise ValueError("sealed probability mapper fit end contains a boolean")
    fit_end = pd.to_datetime(work["probability_fit_end"], errors="coerce").dt.normalize()
    if fit_end.isna().any() or fit_end.gt(sealed.DISCOVERY_END).any():
        raise ValueError("sealed probabilities use a mapper fit after discovery")
    work["probability_fit_end"] = fit_end
    sliced = work.loc[work["event_start_date"].between(SLICE_START, SLICE_END)].copy()
    if sliced.empty:
        raise ValueError("sealed logical probabilities contain no exact three-year rows")
    event_dates = sliced[["mapped_event_key", "event_start_date"]].drop_duplicates()
    if event_dates.duplicated("mapped_event_key").any():
        raise ValueError("canonical event key maps to multiple dates in exact slice")
    return sliced


def canonical_quote_benchmark(quotes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "mapped_event_key",
        "event_start_date",
        "market",
        "book",
        "pair_key",
        "p1_id",
        "p2_id",
        "p1_close",
        "p2_close",
        "p1_outcome",
        "p2_outcome",
    }
    _require_columns(quotes, required, "authorized quotes")
    work = quotes.copy()
    market_eval.validate_normalized_matchup_quotes(work)
    work["p1_id"] = work["p1_id"].map(market_eval._stable_id)
    work["p2_id"] = work["p2_id"].map(market_eval._stable_id)
    left = work["pair_key"].astype(str).str.split("|", regex=False).str[0]
    flip = work["p1_id"].ne(left)
    work["canonical_left_close"] = np.where(flip, work["p2_close"], work["p1_close"])
    work["canonical_right_close"] = np.where(flip, work["p1_close"], work["p2_close"])
    p1_outcome = pd.to_numeric(work["p1_outcome"], errors="coerce")
    work["quote_canonical_outcome"] = np.where(
        flip & p1_outcome.isin([0.0, 1.0]), 1.0 - p1_outcome, p1_outcome
    )
    rank = {"pinnacle": 0, "betcris": 1, "betonline": 2}
    work["_book_rank"] = work["book"].map(rank).fillna(99)
    keys = ["mapped_event_key", "market", "pair_key"]
    for column in ("quote_canonical_outcome",):
        if work.groupby(keys, observed=True)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"authorized quotes disagree across books for {column}")
    output = (
        work.sort_values(keys + ["_book_rank"], kind="mergesort")
        .drop_duplicates(keys, keep="first")
        [
            [
                *keys,
                "canonical_left_close",
                "canonical_right_close",
                "quote_canonical_outcome",
            ]
        ]
        .copy()
    )
    if output.duplicated(keys).any():
        raise AssertionError("canonical quote benchmark is not unique")
    valid_close = market_scoring.valid_american_odds_mask(
        output["canonical_left_close"]
    ) & market_scoring.valid_american_odds_mask(output["canonical_right_close"])
    if not valid_close.all():
        raise ValueError("canonical selected close benchmark contains invalid American odds")
    return output


def proper_scores_from_logical(
    logical: pd.DataFrame, quote_benchmark: pd.DataFrame
) -> pd.DataFrame:
    keys = ["mapped_event_key", "market", "pair_key"]
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].merge(
        quote_benchmark, on=keys, how="left", validate="many_to_one"
    )
    if resolved.empty:
        raise ValueError("exact slice has no resolved canonical matchup outcomes")
    if resolved[["canonical_left_close", "canonical_right_close"]].isna().any().any():
        raise ValueError("proper-score close benchmark is incomplete")
    if not np.allclose(
        resolved["canonical_outcome"], resolved["quote_canonical_outcome"]
    ):
        raise ValueError("logical and quote canonical outcomes disagree")
    rows: list[dict[str, Any]] = []
    for (candidate, market), group in resolved.groupby(
        ["candidate", "market"], observed=True
    ):
        y = group["canonical_outcome"].to_numpy(float)
        p = (
            group["canonical_probability"]
            / (1.0 - group["canonical_push_probability"])
        ).clip(market_eval.EPS, 1.0 - market_eval.EPS).to_numpy(float)
        scores = group[["mapped_event_key"]].copy()
        scores["log_loss"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        scores["brier"] = np.square(p - y)
        event_scores = scores.groupby("mapped_event_key", observed=True)[
            ["log_loss", "brier"]
        ].mean()
        implied_left = np.asarray(
            market_scoring.american_implied_probability(group["canonical_left_close"]),
            dtype=float,
        )
        implied_right = np.asarray(
            market_scoring.american_implied_probability(group["canonical_right_close"]),
            dtype=float,
        )
        market_p = np.clip(
            implied_left / (implied_left + implied_right),
            market_eval.EPS,
            1.0 - market_eval.EPS,
        )
        market_loss = -(y * np.log(market_p) + (1.0 - y) * np.log(1.0 - market_p))
        market_events = pd.DataFrame(
            {"event": group["mapped_event_key"].to_numpy(), "loss": market_loss}
        ).groupby("event", observed=True)["loss"].mean()
        sigma = group["sigma"].drop_duplicates()
        tie = group["tie_threshold"].drop_duplicates()
        if len(sigma) != 1 or len(tie) != 1:
            raise ValueError("frozen mapper parameters vary inside candidate/market")
        rows.append(
            {
                "sample": SAMPLE,
                "candidate": candidate,
                "market": market,
                "events": int(event_scores.index.nunique()),
                "resolved_quotes": int(len(group)),
                "event_equal_log_loss": float(event_scores["log_loss"].mean()),
                "event_equal_brier": float(event_scores["brier"].mean()),
                "event_equal_devig_close_log_loss": float(market_events.mean()),
                "mean_probability": float(p.mean()),
                "observed_win_rate": float(y.mean()),
                "sigma": float(sigma.iloc[0]),
                "tie_threshold": float(tie.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _exact_bool(series: pd.Series, label: str) -> pd.Series:
    if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{label} must contain exact booleans")
    return series.astype(bool)


def validate_materialized_bets(frame: pd.DataFrame, source_partition: str) -> pd.DataFrame:
    required = {
        "candidate",
        "event_key",
        "event_start_date",
        "round_market",
        "market",
        "pair_key",
        "selection_key",
        "book",
        "close_odds",
        "probability",
        "push_probability",
        "tie_rule",
        "outcome_numeric",
        "ev_per_unit",
        "stake",
        "settlement_status",
        "pnl",
        "return_amount",
        "paid_fraction",
        "is_graded",
        "pair_reliability_ok",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
        "comparison_candidate",
        "opportunity_mode",
    }
    _require_columns(frame, required, f"bet partition {source_partition}")
    work = frame.copy()
    work["event_start_date"] = pd.to_datetime(
        work["event_start_date"], errors="raise"
    ).dt.normalize()
    if work["event_start_date"].isna().any() or not work[
        "event_start_date"
    ].between(SOURCE_START, SOURCE_END).all():
        raise ValueError("materialized bets exceed the full source period")
    if set(work["round_market"].astype(str)) - set(sealed.MARKETS):
        raise ValueError("materialized bets contain an unexpected matchup market")
    if set(work["book"].astype(str)) - set(market_eval.BOOKS):
        raise ValueError("materialized bets contain an unexpected sportsbook")
    if set(work["price_basis"].astype(str)) - set(sealed.PRICE_BASES):
        raise ValueError("materialized bets contain an unexpected price basis")
    if set(work["line_policy"].astype(str)) - set(sealed.LINE_POLICIES):
        raise ValueError("materialized bets contain an unexpected line policy")
    if set(work["sizing"].astype(str)) - set(sealed.SIZINGS):
        raise ValueError("materialized bets contain an unexpected sizing")
    if set(work["reliability_policy"].astype(str)) - set(sealed.RELIABILITY_POLICIES):
        raise ValueError("materialized bets contain an unexpected reliability policy")
    if set(work["opportunity_mode"].astype(str)) - {
        "all_available",
        "pairwise_common_with_strict_pt_baseline",
    }:
        raise ValueError("materialized bets contain an unexpected opportunity mode")
    all_available = work["opportunity_mode"].eq("all_available")
    if not work.loc[all_available, "comparison_candidate"].astype(str).eq(
        work.loc[all_available, "candidate"].astype(str)
    ).all():
        raise ValueError("all-available bet has a different comparison candidate")
    pairwise = work["opportunity_mode"].eq(
        "pairwise_common_with_strict_pt_baseline"
    )
    if work.loc[pairwise, "comparison_candidate"].astype(str).eq(BASELINE).any():
        raise ValueError("pairwise bet uses the baseline as its comparison menu")
    valid_pairwise_candidate = work.loc[pairwise, "candidate"].astype(str).eq(
        BASELINE
    ) | work.loc[pairwise, "candidate"].astype(str).eq(
        work.loc[pairwise, "comparison_candidate"].astype(str)
    )
    if not valid_pairwise_candidate.all():
        raise ValueError("pairwise bet candidate is outside its comparison menu")
    pair_reliable = _exact_bool(
        work["pair_reliability_ok"], "materialized pair_reliability_ok"
    )
    fixed = work["line_policy"].isin(market_eval.BOOKS)
    if not work.loc[fixed, "book"].eq(work.loc[fixed, "line_policy"]).all():
        raise ValueError("fixed-book partition contains a different execution book")
    numeric = ["close_odds", "probability", "push_probability", "ev_per_unit", "stake"]
    for column in numeric:
        work[column] = _strict_numeric(work[column], f"materialized bet {column}")
        if not np.isfinite(work[column]).all():
            raise ValueError(f"materialized bet {column} is non-finite")
    if (
        ~market_scoring.valid_american_odds_mask(work["close_odds"])
    ).any() or work["stake"].le(0.0).any():
        raise ValueError("materialized bet price or stake is invalid")
    if (
        work["probability"].lt(0.0).any()
        or work["push_probability"].lt(0.0).any()
        or (work["probability"] + work["push_probability"]).gt(1.0 + 1e-12).any()
        or work["ev_per_unit"].lt(market_eval.EV_GATE - 1e-12).any()
    ):
        raise ValueError("materialized bet probability/EV gate is invalid")
    unreliable_large_edge = work["ev_per_unit"].ge(0.10) & ~pair_reliable
    if (
        unreliable_large_edge
        & work["reliability_policy"].eq("suppress_unreliable_ge10pct")
    ).any():
        raise ValueError("materialized bet violates frozen reliability suppression")
    if (
        unreliable_large_edge
        & work["candidate"].astype(str).str.endswith("__reliability_gate")
    ).any():
        raise ValueError("legacy reliability-gate candidate retains a suppressed bet")
    if work["settlement_status"].eq("ungraded").any():
        raise ValueError("materialized exact-slice bets contain ungraded outcomes")
    raw_outcome_numeric = work["outcome_numeric"]
    outcome_numeric = _strict_numeric(
        raw_outcome_numeric,
        "materialized matchup outcome_numeric",
        allow_null=True,
    )
    if (outcome_numeric.notna() & ~outcome_numeric.isin([0.0, 0.5, 1.0])).any():
        raise ValueError("materialized matchup outcome_numeric is not exactly 0, 0.5, or 1")
    expected_numeric_by_status = {
        "win": 1.0,
        "loss": 0.0,
        "push": 0.5,
        "tie_loss": 0.5,
    }
    for status, expected_numeric in expected_numeric_by_status.items():
        status_rows = work["settlement_status"].eq(status)
        if status_rows.any() and not outcome_numeric.loc[status_rows].eq(
            expected_numeric
        ).all():
            raise ValueError(
                f"materialized {status} settlement disagrees with exact outcome_numeric"
            )
    allowed_statuses = {*expected_numeric_by_status, "void"}
    if set(work["settlement_status"].astype(str)) - allowed_statuses:
        raise ValueError("materialized matchup settlement status is unsupported")
    for column in ("pnl", "return_amount", "paid_fraction"):
        numeric_settlement = _strict_numeric(
            work[column],
            f"graded materialized bet {column}",
            allow_null=(column == "paid_fraction"),
        )
        if not np.isfinite(numeric_settlement.loc[numeric_settlement.notna()]).all():
            raise ValueError(f"graded materialized bet {column} is non-finite")
    flat = work["sizing"].eq("flat_1u")
    expected_stake = np.where(
        work["close_odds"].lt(0.0), work["close_odds"].abs() / 100.0, 1.0
    )
    if not np.array_equal(
        work.loc[flat, "stake"].to_numpy(float),
        np.ones(int(flat.sum()), dtype=float),
    ):
        raise ValueError("flat sizing stake differs from 1u")
    if not np.array_equal(
        work.loc[~flat, "stake"].to_numpy(float),
        np.asarray(expected_stake[~flat], dtype=float),
    ):
        raise ValueError("production risk-to-win stake is malformed")
    effective_push = np.where(
        work["tie_rule"].map(market_scoring.ties_lose).to_numpy(),
        0.0,
        work["push_probability"].to_numpy(float),
    )
    decimal = np.asarray(
        market_scoring.american_decimal_odds(work["close_odds"]), dtype=float
    )
    expected_ev = work["probability"].to_numpy(float) * (decimal - 1.0) - (
        1.0 - work["probability"].to_numpy(float) - effective_push
    )
    if not np.array_equal(
        work["ev_per_unit"].to_numpy(float), np.asarray(expected_ev, dtype=float)
    ):
        raise ValueError("materialized bet EV does not reconstruct")
    recomputed = market_scoring.settle_bets(work)
    if not recomputed["settlement_status"].astype(str).eq(
        work["settlement_status"].astype(str)
    ).all():
        raise ValueError("materialized bet settlement status does not reconstruct")
    if not np.array_equal(
        pd.to_numeric(recomputed["pnl"], errors="coerce").to_numpy(float),
        pd.to_numeric(work["pnl"], errors="coerce").to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError("materialized bet P&L does not reconstruct")
    for column in ("return_amount", "paid_fraction"):
        if not np.array_equal(
            pd.to_numeric(recomputed[column], errors="coerce").to_numpy(float),
            pd.to_numeric(work[column], errors="coerce").to_numpy(float),
            equal_nan=True,
        ):
            raise ValueError(f"materialized bet {column} does not reconstruct")
    observed_graded = _exact_bool(work["is_graded"], "materialized is_graded")
    if not recomputed["is_graded"].astype(bool).eq(observed_graded).all():
        raise ValueError("materialized bet graded flag does not reconstruct")
    pushes = work["settlement_status"].eq("push")
    if pushes.any() and (
        work.loc[pushes, "pnl"].ne(0.0).any()
        or work.loc[pushes, "return_amount"].ne(work.loc[pushes, "stake"]).any()
        or work.loc[pushes, "paid_fraction"].notna().any()
    ):
        raise ValueError("push settlement is not an exact zero-P&L stake refund")
    identity = [
        "candidate",
        "event_key",
        "round_market",
        "pair_key",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    if work.duplicated(identity).any():
        raise ValueError("materialized bet slice/direction keys are duplicated")
    work["source_partition"] = source_partition
    return work


def load_and_slice_bets(
    authority: SourceAuthority, candidate_names: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_specs: dict[str, tuple[str, str]] = {}
    for candidate in candidate_names:
        relative = (
            f"bet_partitions/all_available__{sealed._safe_filename(candidate)}.parquet"
        )
        if relative in expected_specs:
            raise ValueError("candidate names collide after safe filename normalization")
        expected_specs[relative] = ("all_available", candidate)
    for candidate in candidate_names:
        if candidate == BASELINE:
            continue
        relative = (
            f"bet_partitions/pairwise_common__{sealed._safe_filename(candidate)}.parquet"
        )
        if relative in expected_specs:
            raise ValueError("candidate partition filenames collide")
        expected_specs[relative] = (
            "pairwise_common_with_strict_pt_baseline",
            candidate,
        )
    expected = set(expected_specs)
    observed = {
        relative
        for relative in authority.artifact_paths
        if relative.startswith("bet_partitions/") and relative.endswith(".parquet")
    }
    if observed != expected:
        raise ValueError(
            "source bet-partition menu differs from evaluated candidate menu: "
            f"missing={sorted(expected-observed)[:10]}, extra={sorted(observed-expected)[:10]}"
        )
    pieces: list[pd.DataFrame] = []
    source_rows = 0
    for relative in sorted(observed):
        partition = pd.read_parquet(authority.artifact_paths[relative])
        source_rows += len(partition)
        if partition.empty:
            continue
        checked = validate_materialized_bets(partition, relative)
        expected_mode, expected_comparison = expected_specs[relative]
        if set(checked["opportunity_mode"].astype(str)) != {expected_mode}:
            raise ValueError(f"bet rows migrated to the wrong partition mode: {relative}")
        if set(checked["comparison_candidate"].astype(str)) != {
            expected_comparison
        }:
            raise ValueError(f"bet rows migrated to the wrong comparison partition: {relative}")
        expected_candidates = (
            {expected_comparison}
            if expected_mode == "all_available"
            else {BASELINE, expected_comparison}
        )
        if set(checked["candidate"].astype(str)) - expected_candidates:
            raise ValueError(f"bet rows contain a candidate outside partition: {relative}")
        pieces.append(
            checked.loc[
                checked["event_start_date"].between(SLICE_START, SLICE_END)
            ].copy()
        )
    bets = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if bets.empty:
        raise ValueError("exact three-year slice contains no materialized bets")
    return bets, {
        "source_partition_files": len(observed),
        "source_materialized_bet_rows": int(source_rows),
        "slice_materialized_bet_rows": int(len(bets)),
        "slice_events_with_bets": int(bets["event_key"].nunique()),
        "push_pnl_violations": int(
            (bets["settlement_status"].eq("push") & bets["pnl"].ne(0.0)).sum()
        ),
    }


def validate_bets_against_authorities(
    bets: pd.DataFrame, quotes: pd.DataFrame, logical: pd.DataFrame
) -> dict[str, Any]:
    """Bind every materialized bet to a quote, side, price, and probability."""

    _require_columns(
        bets,
        [
            "candidate",
            "selection_key",
            "selection_id",
            "opponent_id",
            "event_key",
            "event_start_date",
            "round_market",
            "pair_key",
            "book",
            "market",
            "price_basis",
            "line_policy",
            "close_odds",
            "tie_rule",
            "outcome_numeric",
            "outcome",
            "probability",
            "push_probability",
        ],
        "materialized authority binding",
    )
    _require_columns(
        quotes,
        [
            "quote_id",
            "mapped_event_key",
            "event_start_date",
            "market",
            "book",
            "pair_key",
            "p1_id",
            "p2_id",
            "p1_open",
            "p2_open",
            "p1_close",
            "p2_close",
            "p1_outcome",
            "p2_outcome",
            "p1_outcome_text",
            "p2_outcome_text",
            "tie_rule",
        ],
        "authorized quote binding",
    )
    if not bets["market"].astype(str).eq("tournament_matchup").all():
        raise ValueError("materialized bet market is not exactly tournament_matchup")
    parsed_rows: list[tuple[str, str]] = []
    for candidate, selection_key in bets[["candidate", "selection_key"]].itertuples(
        index=False, name=None
    ):
        prefix = f"{candidate}|"
        text = str(selection_key)
        if not text.startswith(prefix):
            raise ValueError("selection_key does not begin with its exact candidate")
        quote_side, separator, side = text[len(prefix) :].rpartition("|")
        if not separator or not quote_side or side not in {"p1", "p2"}:
            raise ValueError("selection_key does not encode an authorized quote side")
        parsed_rows.append((quote_side, side))
    bound = bets.copy()
    bound[["_quote_id", "_side"]] = pd.DataFrame(
        parsed_rows, index=bound.index
    )
    quote_work = quotes.copy()
    quote_work["event_start_date"] = pd.to_datetime(
        quote_work["event_start_date"], errors="raise"
    ).dt.normalize()
    if quote_work["quote_id"].astype(str).duplicated().any():
        raise ValueError("authorized quote IDs are not unique")
    quote_work["quote_id"] = quote_work["quote_id"].astype(str)
    rename = {
        column: f"quote_{column}"
        for column in quote_work.columns
        if column != "quote_id"
    }
    bound = bound.merge(
        quote_work.rename(columns=rename),
        left_on="_quote_id",
        right_on="quote_id",
        how="left",
        validate="many_to_one",
    )
    if bound["quote_id"].isna().any():
        raise ValueError("selection_key refers to a quote outside frozen authority")
    p1_side = bound["_side"].eq("p1")
    expected_selection = np.where(p1_side, bound["quote_p1_id"], bound["quote_p2_id"])
    expected_opponent = np.where(p1_side, bound["quote_p2_id"], bound["quote_p1_id"])
    observed_selection = bound["selection_id"].map(market_eval._stable_id)
    observed_opponent = bound["opponent_id"].map(market_eval._stable_id)
    expected_selection = pd.Series(expected_selection, index=bound.index).map(
        market_eval._stable_id
    )
    expected_opponent = pd.Series(expected_opponent, index=bound.index).map(
        market_eval._stable_id
    )
    if not observed_selection.eq(expected_selection).all() or not observed_opponent.eq(
        expected_opponent
    ).all():
        raise ValueError("materialized selection/opponent IDs differ from quote side")
    identity_checks = {
        "event_key": "quote_mapped_event_key",
        "round_market": "quote_market",
        "pair_key": "quote_pair_key",
        "book": "quote_book",
    }
    for observed_column, quote_column in identity_checks.items():
        if not bound[observed_column].astype(str).eq(
            bound[quote_column].astype(str)
        ).all():
            raise ValueError(f"materialized {observed_column} differs from quote authority")
    if not pd.to_datetime(bound["event_start_date"], errors="raise").dt.normalize().eq(
        pd.to_datetime(bound["quote_event_start_date"], errors="raise").dt.normalize()
    ).all():
        raise ValueError("materialized event date differs from quote authority")
    if not bound["tie_rule"].fillna("").astype(str).eq(
        bound["quote_tie_rule"].fillna("").astype(str)
    ).all():
        raise ValueError("materialized tie rule differs from quote authority")
    quote_outcome = np.where(
        p1_side, bound["quote_p1_outcome"], bound["quote_p2_outcome"]
    )
    if not np.array_equal(
        pd.to_numeric(bound["outcome_numeric"], errors="coerce").to_numpy(float),
        pd.to_numeric(pd.Series(quote_outcome), errors="coerce").to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError("materialized outcome differs from authorized quote side")
    quote_outcome_text = np.where(
        p1_side,
        bound["quote_p1_outcome_text"],
        bound["quote_p2_outcome_text"],
    )
    observed_outcome_text = bound["outcome"].fillna("<NULL>").astype(str)
    expected_outcome_text = pd.Series(quote_outcome_text, index=bound.index).fillna(
        "<NULL>"
    ).astype(str)
    if not observed_outcome_text.eq(expected_outcome_text).all():
        raise ValueError("materialized outcome text differs from authorized quote side")
    quote_execution_odds = np.where(
        p1_side,
        np.where(
            bound["price_basis"].eq("open"),
            bound["quote_p1_open"],
            bound["quote_p1_close"],
        ),
        np.where(
            bound["price_basis"].eq("open"),
            bound["quote_p2_open"],
            bound["quote_p2_close"],
        ),
    )
    if not np.array_equal(
        bound["close_odds"].to_numpy(float),
        np.asarray(quote_execution_odds, dtype=float),
    ):
        raise ValueError("materialized execution odds differ from quote authority")

    probability_keys = ["candidate", "event_key", "round_market", "pair_key"]
    probability_lookup = logical[
        [
            "candidate",
            "mapped_event_key",
            "market",
            "pair_key",
            "canonical_probability",
            "canonical_push_probability",
        ]
    ].rename(
        columns={"mapped_event_key": "event_key", "market": "round_market"}
    )
    if probability_lookup.duplicated(probability_keys).any():
        raise ValueError("canonical probability authority is duplicated")
    bound = bound.merge(
        probability_lookup,
        on=probability_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_logical"),
    )
    if bound["canonical_probability"].isna().any():
        raise ValueError("materialized bet has no canonical probability authority")
    canonical_left = bound["pair_key"].astype(str).str.split(
        "|", regex=False
    ).str[0]
    bound_selection = bound["selection_id"].map(market_eval._stable_id)
    selection_is_left = bound_selection.reset_index(drop=True).eq(
        canonical_left.reset_index(drop=True)
    ).to_numpy()
    canonical_probability = bound["canonical_probability"].to_numpy(float)
    canonical_push = bound["canonical_push_probability"].to_numpy(float)
    expected_probability = np.where(
        selection_is_left,
        canonical_probability,
        1.0 - canonical_probability - canonical_push,
    )
    if not np.allclose(
        bound["probability"].to_numpy(float),
        expected_probability,
        rtol=0.0,
        atol=2e-15,
    ) or not np.array_equal(
        bound["push_probability"].to_numpy(float), canonical_push
    ):
        raise ValueError("materialized probability differs from canonical authority")

    # Reconstruct each offered selection line so best-of-three must really be
    # the best valid archived execution price, with the pinned lexical tie-break.
    long_parts: list[pd.DataFrame] = []
    for price_basis in sealed.PRICE_BASES:
        valid = market_scoring.valid_american_odds_mask(
            quote_work["p1_open" if price_basis == "open" else "p1_close"]
        ) & market_scoring.valid_american_odds_mask(
            quote_work["p2_open" if price_basis == "open" else "p2_close"]
        )
        if price_basis == "open":
            valid &= market_scoring.valid_american_odds_mask(
                quote_work["p1_close"]
            ) & market_scoring.valid_american_odds_mask(quote_work["p2_close"])
        local = quote_work.loc[valid].copy()
        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            piece = local[
                ["mapped_event_key", "market", "pair_key", "book", f"{side}_id", f"{opponent}_id", f"{side}_{price_basis}"]
            ].rename(
                columns={
                    "mapped_event_key": "event_key",
                    f"{side}_id": "selection_id",
                    f"{opponent}_id": "opponent_id",
                    f"{side}_{price_basis}": "execution_odds",
                }
            )
            piece["selection_id"] = piece["selection_id"].map(market_eval._stable_id)
            piece["opponent_id"] = piece["opponent_id"].map(market_eval._stable_id)
            piece["price_basis"] = price_basis
            long_parts.append(piece)
    offered_lines = pd.concat(long_parts, ignore_index=True)
    best_keys = [
        "event_key",
        "market",
        "pair_key",
        "selection_id",
        "opponent_id",
        "price_basis",
    ]
    best = (
        offered_lines.sort_values(
            [*best_keys, "execution_odds", "book"],
            ascending=[True] * len(best_keys) + [False, True],
            kind="mergesort",
        )
        .drop_duplicates(best_keys, keep="first")
        .rename(
            columns={"book": "best_book", "execution_odds": "best_execution_odds"}
        )
    )
    selected_best = bound.loc[bound["line_policy"].eq("best_of_three")].copy()
    if not selected_best.empty:
        selected_best["selection_id"] = selected_best["selection_id"].map(
            market_eval._stable_id
        )
        selected_best["opponent_id"] = selected_best["opponent_id"].map(
            market_eval._stable_id
        )
        selected_best = selected_best.merge(
            best[[*best_keys, "best_book", "best_execution_odds"]],
            left_on=[
                "event_key",
                "round_market",
                "pair_key",
                "selection_id",
                "opponent_id",
                "price_basis",
            ],
            right_on=best_keys,
            how="left",
            validate="many_to_one",
        )
        if selected_best["best_book"].isna().any() or not selected_best["book"].eq(
            selected_best["best_book"]
        ).all() or not np.array_equal(
            selected_best["close_odds"].to_numpy(float),
            selected_best["best_execution_odds"].to_numpy(float),
        ):
            raise ValueError("best-of-three bet is not the pinned best valid price")
    return {
        "bet_rows_bound_to_quote_identity": int(len(bound)),
        "bet_rows_bound_to_canonical_probability": int(len(bound)),
        "best_of_three_rows_reconstructed": int(
            bets["line_policy"].eq("best_of_three").sum()
        ),
    }


def summarize_bets(
    bets: pd.DataFrame, candidate_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance: list[dict[str, Any]] = []
    event_parts: list[pd.DataFrame] = []
    arrangements: list[tuple[str, str, str]] = [
        ("all_available", candidate, candidate) for candidate in candidate_names
    ]
    arrangements.extend(
        (
            "pairwise_common_with_strict_pt_baseline",
            comparison,
            candidate,
        )
        for comparison in candidate_names
        if comparison != BASELINE
        for candidate in (BASELINE, comparison)
    )
    for opportunity_mode, comparison, candidate in arrangements:
        base = bets.loc[
            bets["opportunity_mode"].eq(opportunity_mode)
            & bets["comparison_candidate"].eq(comparison)
            & bets["candidate"].eq(candidate)
        ]
        for market in sealed.MARKETS:
            for price_basis in sealed.PRICE_BASES:
                for line_policy in sealed.LINE_POLICIES:
                    for sizing in sealed.SIZINGS:
                        for reliability_policy in sealed.RELIABILITY_POLICIES:
                            group = base.loc[
                                base["round_market"].eq(market)
                                & base["price_basis"].eq(price_basis)
                                & base["line_policy"].eq(line_policy)
                                & base["sizing"].eq(sizing)
                                & base["reliability_policy"].eq(reliability_policy)
                            ].copy()
                            stake = float(group["stake"].sum()) if not group.empty else 0.0
                            pnl = float(group["pnl"].sum()) if not group.empty else 0.0
                            action = (
                                group.loc[~group["settlement_status"].eq("void")]
                                if not group.empty
                                else group
                            )
                            action_stake = (
                                float(action["stake"].sum()) if not action.empty else 0.0
                            )
                            performance.append(
                                {
                                    "candidate": candidate,
                                    "comparison_candidate": comparison,
                                    "market": market,
                                    "opportunity_mode": opportunity_mode,
                                    "price_basis": price_basis,
                                    "line_policy": line_policy,
                                    "sizing": sizing,
                                    "reliability_policy": reliability_policy,
                                    "events_with_bets": int(group["event_key"].nunique()) if not group.empty else 0,
                                    "bets": int(len(group)),
                                    "stake": stake,
                                    "pnl": pnl,
                                    "roi": pnl / stake if stake else np.nan,
                                    "roi_parent_semantics": pnl / stake if stake else np.nan,
                                    "action_stake_excluding_void": action_stake,
                                    "roi_action_excluding_void": pnl / action_stake if action_stake else np.nan,
                                    "max_drawdown": float(market_eval._max_drawdown(group)) if not group.empty else 0.0,
                                    "mean_nominal_ev": float(group["ev_per_unit"].mean()) if not group.empty else np.nan,
                                    "mean_open_to_close_clv_pp": float(group["open_to_close_clv_pp"].mean()) if not group.empty else np.nan,
                                    "pushes": int(group["settlement_status"].eq("push").sum()) if not group.empty else 0,
                                }
                            )
                            if group.empty:
                                continue
                            event = (
                                group.groupby(
                                    ["event_key", "event_start_date"], observed=True
                                )
                                .agg(
                                    pnl=("pnl", "sum"),
                                    stake=("stake", "sum"),
                                    bets=("pnl", "size"),
                                )
                                .reset_index()
                            )
                            event["candidate"] = candidate
                            event["comparison_candidate"] = comparison
                            event["market"] = market
                            event["opportunity_mode"] = opportunity_mode
                            event["price_basis"] = price_basis
                            event["line_policy"] = line_policy
                            event["sizing"] = sizing
                            event["reliability_policy"] = reliability_policy
                            event_parts.append(event)
    performance_frame = pd.DataFrame(performance)
    event_frame = (
        pd.concat(event_parts, ignore_index=True, sort=False)
        if event_parts
        else pd.DataFrame()
    )
    keys = [
        "candidate",
        "comparison_candidate",
        "market",
        "opportunity_mode",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    if performance_frame.duplicated(keys).any():
        raise AssertionError("money summary grid is duplicated")
    expected_rows = (
        len(candidate_names) + 2 * (len(candidate_names) - 1)
    ) * len(sealed.MARKETS) * len(sealed.PRICE_BASES) * len(
        sealed.LINE_POLICIES
    ) * len(sealed.SIZINGS) * len(sealed.RELIABILITY_POLICIES)
    if len(performance_frame) != expected_rows:
        raise AssertionError("money summary grid is incomplete")
    return performance_frame, event_frame


def build_money_event_deltas_exact(
    event_performance: pd.DataFrame,
    quotes: pd.DataFrame,
    candidate_names: Sequence[str],
    baseline_candidate: str,
) -> pd.DataFrame:
    """Rebuild the sealed money comparison grid with true no-bet counts.

    ``zero_bet_event`` deliberately preserves the parent evaluator's historical
    zero-*P&L* label.  ``true_no_bet_event`` is the correctly named count-based
    field used for audit only; inference includes every row either way.
    """

    event_catalog = (
        quotes[["mapped_event_key", "event_start_date", "market"]]
        .drop_duplicates()
        .rename(columns={"mapped_event_key": "event_key"})
    )
    if event_catalog.duplicated(["event_key", "market"]).any():
        raise ValueError("offered event/market catalog is ambiguous")
    slice_columns = [
        "market",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    rows: list[pd.DataFrame] = []
    for candidate in candidate_names:
        if candidate == baseline_candidate:
            continue
        for opportunity_mode in (
            "all_available",
            "pairwise_common_with_strict_pt_baseline",
        ):
            if opportunity_mode == "all_available":
                challenger_rows = event_performance.loc[
                    event_performance["opportunity_mode"].eq(opportunity_mode)
                    & event_performance["candidate"].eq(candidate)
                    & event_performance["comparison_candidate"].eq(candidate)
                ]
                baseline_rows = event_performance.loc[
                    event_performance["opportunity_mode"].eq(opportunity_mode)
                    & event_performance["candidate"].eq(baseline_candidate)
                    & event_performance["comparison_candidate"].eq(
                        baseline_candidate
                    )
                ]
            else:
                comparison = event_performance["comparison_candidate"].eq(candidate)
                challenger_rows = event_performance.loc[
                    comparison
                    & event_performance["opportunity_mode"].eq(opportunity_mode)
                    & event_performance["candidate"].eq(candidate)
                ]
                baseline_rows = event_performance.loc[
                    comparison
                    & event_performance["opportunity_mode"].eq(opportunity_mode)
                    & event_performance["candidate"].eq(baseline_candidate)
                ]
            for values in (
                (market, price_basis, line_policy, sizing, reliability_policy)
                for market in sealed.MARKETS
                for price_basis in sealed.PRICE_BASES
                for line_policy in sealed.LINE_POLICIES
                for sizing in sealed.SIZINGS
                for reliability_policy in sealed.RELIABILITY_POLICIES
            ):
                filters = dict(zip(slice_columns, values))
                events = event_catalog.loc[
                    event_catalog["market"].eq(filters["market"])
                ].copy()
                challenger = challenger_rows.copy()
                baseline = baseline_rows.copy()
                for column, value in filters.items():
                    challenger = challenger.loc[challenger[column].eq(value)]
                    baseline = baseline.loc[baseline[column].eq(value)]
                challenger = challenger[["event_key", "pnl", "bets"]].rename(
                    columns={"pnl": "candidate_pnl", "bets": "candidate_bets"}
                )
                baseline = baseline[["event_key", "pnl", "bets"]].rename(
                    columns={"pnl": "baseline_pnl", "bets": "baseline_bets"}
                )
                merged = events.merge(
                    baseline, on="event_key", how="left", validate="one_to_one"
                ).merge(
                    challenger, on="event_key", how="left", validate="one_to_one"
                )
                merged[["baseline_pnl", "candidate_pnl"]] = merged[
                    ["baseline_pnl", "candidate_pnl"]
                ].fillna(0.0)
                merged[["baseline_bets", "candidate_bets"]] = merged[
                    ["baseline_bets", "candidate_bets"]
                ].fillna(0).astype(int)
                merged["event_delta"] = (
                    merged["candidate_pnl"] - merged["baseline_pnl"]
                )
                merged["candidate"] = candidate
                merged["baseline"] = baseline_candidate
                merged["opportunity_mode"] = opportunity_mode
                for column, value in filters.items():
                    merged[column] = value
                merged["metric"] = "pnl_delta"
                merged["zero_bet_event"] = (
                    merged["candidate_pnl"].eq(0.0)
                    & merged["baseline_pnl"].eq(0.0)
                )
                merged["true_no_bet_event"] = (
                    merged["candidate_bets"].eq(0)
                    & merged["baseline_bets"].eq(0)
                )
                rows.append(merged)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    hypothesis = [
        "candidate",
        "opportunity_mode",
        "market",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    if output.empty or output.duplicated([*hypothesis, "event_key"]).any():
        raise ValueError("exact money event-delta panel is empty or duplicated")
    counts = output.groupby(hypothesis, observed=True).size()
    expected_by_market = event_catalog.groupby("market", observed=True).size()
    for keys, count in counts.items():
        if int(count) != int(expected_by_market.loc[keys[2]]):
            raise ValueError("money event-delta grid is not rectangular")
    return output


def assert_filtered_parent_deltas_match(
    rebuilt: pd.DataFrame,
    parent: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    label: str,
) -> None:
    for frame in (rebuilt, parent):
        _require_columns(frame, [*key_columns, *value_columns], label)
        if frame.duplicated(list(key_columns)).any():
            raise ValueError(f"{label} contains duplicate comparison keys")
    left = rebuilt[[*key_columns, *value_columns]].sort_values(
        list(key_columns), kind="mergesort"
    ).reset_index(drop=True)
    right = parent[[*key_columns, *value_columns]].sort_values(
        list(key_columns), kind="mergesort"
    ).reset_index(drop=True)
    if len(left) != len(right) or not left[list(key_columns)].astype(str).equals(
        right[list(key_columns)].astype(str)
    ):
        raise ValueError(f"rebuilt {label} keys differ from filtered parent output")
    for column in value_columns:
        if pd.api.types.is_bool_dtype(left[column]) or pd.api.types.is_bool_dtype(
            right[column]
        ):
            left_bool = _exact_bool(left[column], f"rebuilt {label}.{column}")
            right_bool = _exact_bool(right[column], f"parent {label}.{column}")
            if not left_bool.eq(right_bool).all():
                raise ValueError(f"rebuilt {label} differs in {column}")
        else:
            left_numeric = _strict_numeric(
                left[column], f"rebuilt {label}.{column}", allow_null=True
            )
            right_numeric = _strict_numeric(
                right[column], f"parent {label}.{column}", allow_null=True
            )
            if not np.array_equal(
                left_numeric.to_numpy(float),
                right_numeric.to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError(f"rebuilt {label} differs in {column}")


def _proper_by_year(logical: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year, group in logical.groupby(logical["event_start_date"].dt.year, observed=True):
        summary = proper_scores_from_logical(group, benchmark)
        summary["year"] = int(year)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _artifact_manifest(output: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _write_report(
    output: Path,
    performance: pd.DataFrame,
    selected_name: str,
    audit: Mapping[str, Any],
) -> None:
    primary = performance.loc[
        performance["candidate"].eq(selected_name)
        & performance["comparison_candidate"].eq(selected_name)
        & performance["opportunity_mode"].eq("all_available")
        & performance["price_basis"].eq("close")
        & performance["line_policy"].eq("best_of_three")
        & performance["sizing"].eq("flat_1u")
        & performance["reliability_policy"].eq("ungated")
    ]
    lines = [
        "# Exact last-three-years sealed matchup slice",
        "",
        f"Fixed inclusive period: {SLICE_START.date()} through {SLICE_END.date()}.",
        "The frozen model, mapper, bet gate, prices, outcomes, and settlements are unchanged from the completed full sealed run.",
        "",
        "| Market | Bets | Stake | P&L | ROI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"| {row['market']} | {int(row['bets'])} | {row['stake']:.2f} | "
            f"{row['pnl']:+.2f} | {row['roi']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Integrity:",
            f"- Source artifact files verified: {audit['source_artifact_files_verified']}",
            f"- Canonical probability rows: {audit['slice_canonical_probability_rows']}",
            f"- Materialized bet rows: {audit['slice_materialized_bet_rows']}",
            f"- Whole-event bootstrap draws: {audit['bootstrap_draws']}",
            "- Zero-bet events are explicitly padded before paired money inference.",
            "- 2/3/4-event circular moving-block max-stat semantics are imported from and pinned to the sealed evaluator.",
        ]
    )
    (output / "EXACT_3Y_MATCHUP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    if args.bootstrap < 10_000:
        raise ValueError("exact three-year postprocess requires --bootstrap >= 10000")
    if tuple(args.block_sizes) != (2, 3, 4):
        raise ValueError("exact postprocess requires frozen block sizes 2,3,4")
    if SLICE_END - SLICE_START != pd.Timedelta(days=1096):
        raise AssertionError("fixed inclusive three-year window changed")
    source_output = args.sealed_output.resolve()
    output = args.output_dir.resolve()
    try:
        output.relative_to(source_output)
    except ValueError:
        pass
    else:
        raise ValueError(
            "postprocess output must not equal or be nested inside the immutable sealed output"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"postprocess output directory is not empty; use a new versioned path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    dependency_audit = verify_dependency_pins()

    # Validate once before provenance enumeration, fingerprint the complete
    # source closure, then validate the same bytes again to close the TOCTOU gap.
    pre = validate_completed_source_output(
        source_output, args.expected_sealed_manifest_sha256
    )
    pre_manifest_hash = _sha256(pre.manifest_path)
    provenance_inputs = [
        pre.manifest_path,
        *pre.artifact_paths.values(),
        *pre.handoff.verified_paths,
        *pre.panel_authority["verified_paths"],
    ]
    provenance_state = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__).resolve(),
        input_paths=provenance_inputs,
        code_paths=CODE_PATHS,
    )
    authority = validate_completed_source_output(
        source_output, args.expected_sealed_manifest_sha256
    )
    if _sha256(authority.manifest_path) != pre_manifest_hash:
        raise RuntimeError("source sealed manifest changed across provenance begin")

    candidate_names = list(map(str, authority.manifest["evaluated_candidate_names"]))
    if len(candidate_names) != len(set(candidate_names)) or BASELINE not in candidate_names:
        raise ValueError("source candidate menu is duplicated or lacks strict-PT baseline")
    quotes, quote_audit = load_authorized_quotes(authority)
    logical_full = pd.read_parquet(
        authority.artifact_paths["sealed_logical_probabilities.parquet"]
    )
    logical = validate_and_slice_logical(logical_full)
    if set(logical["candidate"].astype(str)) != set(candidate_names):
        raise ValueError("exact-slice canonical candidate menu differs from source manifest")
    benchmark = canonical_quote_benchmark(quotes)
    logical_event_dates = logical[
        ["mapped_event_key", "market", "event_start_date"]
    ].drop_duplicates()
    quoted_event_dates = quotes[
        ["mapped_event_key", "market", "event_start_date"]
    ].drop_duplicates()
    if logical_event_dates.duplicated(["mapped_event_key", "market"]).any() or quoted_event_dates.duplicated(
        ["mapped_event_key", "market"]
    ).any():
        raise ValueError("logical or quoted event/market dates are ambiguous")
    if not logical_event_dates.sort_values(
        ["mapped_event_key", "market"], kind="mergesort"
    ).reset_index(drop=True).equals(
        quoted_event_dates.sort_values(
            ["mapped_event_key", "market"], kind="mergesort"
        ).reset_index(drop=True)
    ):
        raise ValueError("canonical probability and offered-quote event universes differ")
    proper = proper_scores_from_logical(logical, benchmark)
    paired = discovery.paired_common_proper_scores(logical, BASELINE)
    proper_year = _proper_by_year(logical, benchmark)
    proper_event = sealed.paired_proper_event_deltas(logical, BASELINE)
    parent_proper_event = pd.read_parquet(
        authority.artifact_paths["proper_score_event_deltas.parquet"]
    )
    parent_proper_event["event_start_date"] = pd.to_datetime(
        parent_proper_event["event_start_date"], errors="raise"
    ).dt.normalize()
    parent_proper_event = parent_proper_event.loc[
        parent_proper_event["event_start_date"].between(SLICE_START, SLICE_END)
    ].copy()
    proper_delta_keys = [
        "candidate",
        "baseline",
        "market",
        "event_key",
        "event_start_date",
        "metric",
    ]
    assert_filtered_parent_deltas_match(
        proper_event,
        parent_proper_event,
        key_columns=proper_delta_keys,
        value_columns=["event_delta"],
        label="proper-score event deltas",
    )

    bets, bet_audit = load_and_slice_bets(authority, candidate_names)
    bet_event_dates = bets[
        ["event_key", "round_market", "event_start_date"]
    ].drop_duplicates()
    bet_date_check = bet_event_dates.merge(
        quoted_event_dates.rename(
            columns={"mapped_event_key": "event_key", "market": "round_market"}
        ),
        on=["event_key", "round_market"],
        how="left",
        suffixes=("_bet", "_quote"),
        validate="many_to_one",
    )
    if bet_date_check["event_start_date_quote"].isna().any() or not bet_date_check[
        "event_start_date_bet"
    ].eq(bet_date_check["event_start_date_quote"]).all():
        raise ValueError("materialized bet event/date identity differs from quote authority")
    bet_authority_audit = validate_bets_against_authorities(bets, quotes, logical)
    performance, event_performance = summarize_bets(bets, candidate_names)
    parent_event_performance = pd.read_parquet(
        authority.artifact_paths["event_betting_performance.parquet"]
    )
    parent_event_performance["event_start_date"] = pd.to_datetime(
        parent_event_performance["event_start_date"], errors="raise"
    ).dt.normalize()
    parent_event_performance = parent_event_performance.loc[
        parent_event_performance["event_start_date"].between(
            SLICE_START, SLICE_END
        )
    ].copy()
    event_performance_keys = [
        "candidate",
        "comparison_candidate",
        "market",
        "opportunity_mode",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
        "event_key",
        "event_start_date",
    ]
    assert_filtered_parent_deltas_match(
        event_performance,
        parent_event_performance,
        key_columns=event_performance_keys,
        value_columns=["pnl", "stake", "bets"],
        label="event betting performance",
    )
    money_event = build_money_event_deltas_exact(
        event_performance, quotes, candidate_names, BASELINE
    )
    if money_event.empty or not money_event["event_start_date"].between(
        SLICE_START, SLICE_END
    ).all():
        raise ValueError("zero-padded money event panel is empty or outside fixed slice")
    hypothesis = [
        "candidate",
        "opportunity_mode",
        "market",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    parent_money_event = pd.read_parquet(
        authority.artifact_paths["money_event_deltas.parquet"]
    )
    parent_money_event["event_start_date"] = pd.to_datetime(
        parent_money_event["event_start_date"], errors="raise"
    ).dt.normalize()
    parent_money_event = parent_money_event.loc[
        parent_money_event["event_start_date"].between(SLICE_START, SLICE_END)
    ].copy()
    money_delta_keys = [*hypothesis, "baseline", "event_key", "event_start_date", "metric"]
    assert_filtered_parent_deltas_match(
        money_event,
        parent_money_event,
        key_columns=money_delta_keys,
        value_columns=[
            "baseline_pnl",
            "candidate_pnl",
            "event_delta",
            "zero_bet_event",
        ],
        label="money event deltas",
    )

    money_bootstrap = sealed.event_cluster_bootstrap(
        money_event,
        draws=args.bootstrap,
        seed=args.seed,
        better_when="positive",
    )
    proper_bootstrap = sealed.event_cluster_bootstrap(
        proper_event,
        draws=args.bootstrap,
        seed=args.seed + 1,
        better_when="negative",
    )
    bootstrap = pd.concat(
        [money_bootstrap, proper_bootstrap], ignore_index=True, sort=False
    )
    reality = sealed.block_max_stat_reality_check(
        money_event,
        draws=args.bootstrap,
        seed=args.seed + 2,
        block_sizes=args.block_sizes,
    )

    metadata = pd.read_csv(
        authority.artifact_paths["candidate_roles_and_metadata.csv"]
    )
    if metadata["candidate"].astype(str).duplicated().any():
        raise ValueError("source candidate metadata is duplicated")
    for name, frame in (
        ("proper", proper),
        ("paired", paired),
        ("proper_year", proper_year),
        ("performance", performance),
    ):
        if "candidate" in frame:
            merged = frame.merge(
                metadata, on="candidate", how="left", validate="many_to_one"
            )
            if merged["candidate_role"].isna().any():
                raise ValueError(f"{name} rows lack source candidate metadata")
            if name == "proper":
                proper = merged
            elif name == "paired":
                paired = merged
            elif name == "proper_year":
                proper_year = merged
            else:
                performance = merged

    _atomic_parquet(logical, output / "sealed_logical_probabilities_exact_3y.parquet")
    _atomic_parquet(bets, output / "sealed_materialized_bets_exact_3y.parquet")
    _atomic_csv(proper, output / "proper_scores.csv")
    _atomic_csv(paired, output / "paired_common_proper_scores.csv")
    _atomic_csv(proper_year, output / "proper_scores_by_year.csv")
    _atomic_parquet(proper_event, output / "proper_score_event_deltas.parquet")
    _atomic_csv(performance, output / "betting_performance.csv")
    _atomic_parquet(
        event_performance, output / "event_betting_performance.parquet"
    )
    _atomic_parquet(money_event, output / "money_event_deltas.parquet")
    _atomic_csv(bootstrap, output / "paired_event_bootstrap.csv")
    _atomic_csv(reality, output / "block_max_stat_reality_check.csv")

    audit = {
        "contract": OUTPUT_CONTRACT,
        "research_only": True,
        "production_files_changed": False,
        "source_sealed_output": str(authority.output_dir),
        "source_sealed_manifest_sha256": _sha256(authority.manifest_path),
        "source_contract": SOURCE_CONTRACT,
        "source_artifact_files_verified": len(authority.artifact_paths),
        "source_full_handoff_verified": bool(authority.handoff.audit["full_hash_chain_verified"]),
        "source_input_authority_verified": True,
        "dependency_source_pins": dependency_audit,
        "slice_start": SLICE_START.date().isoformat(),
        "slice_end": SLICE_END.date().isoformat(),
        "inclusive": True,
        "calendar_days_difference": int((SLICE_END - SLICE_START).days),
        "markets": list(sealed.MARKETS),
        "books": list(market_eval.BOOKS),
        "line_policies": list(sealed.LINE_POLICIES),
        "price_bases": list(sealed.PRICE_BASES),
        "sizing_rules": list(sealed.SIZINGS),
        "reliability_policies": list(sealed.RELIABILITY_POLICIES),
        "evaluated_candidate_names": candidate_names,
        "slice_events": int(logical["mapped_event_key"].nunique()),
        "slice_event_date_min": logical["event_start_date"].min().date().isoformat(),
        "slice_event_date_max": logical["event_start_date"].max().date().isoformat(),
        "slice_canonical_probability_rows": int(len(logical)),
        "slice_canonical_probability_duplicates": int(
            logical.duplicated(
                ["candidate", "mapped_event_key", "market", "pair_key"]
            ).sum()
        ),
        "quote_authority": quote_audit,
        "materialized_bet_authority": bet_authority_audit,
        **bet_audit,
        "zero_padded_money_event_rows": int(len(money_event)),
        "zero_bet_money_event_rows": int(money_event["zero_bet_event"].sum()),
        "true_no_bet_money_event_rows": int(money_event["true_no_bet_event"].sum()),
        "zero_bet_event_field_semantics": "legacy_both_pnl_zero_not_both_bet_count_zero",
        "bootstrap_draws": int(args.bootstrap),
        "bootstrap_seed_money": int(args.seed),
        "bootstrap_seed_proper": int(args.seed + 1),
        "max_stat_seed_base": int(args.seed + 2),
        "bootstrap_cluster": "whole_event_including_zero_bet_events",
        "block_sizes_events": list(args.block_sizes),
        "max_stat_null_centered": True,
        "source_evaluator_unchanged": True,
        "model_refit": False,
        "model_reselection": False,
        "probability_mapper_refit": False,
        "bet_reselection": False,
        "closing_odds_used_as_features": False,
    }
    _atomic_json(output / "audit.json", audit)
    _write_report(
        output, performance, str(authority.manifest["selected_model"]), audit
    )

    execution_audit = execution_provenance.finalize(provenance_state)
    audit["execution_provenance"] = {
        "path": "execution_provenance/execution_provenance.json",
        "start_utc": execution_audit["start_utc"],
        "end_utc": execution_audit["end_utc"],
        "pid": execution_audit["pid"],
        "verified_unchanged_before_complete_manifest": True,
    }
    _atomic_json(output / "audit.json", audit)
    execution_provenance.assert_current(provenance_state)
    manifest = {
        "contract": OUTPUT_CONTRACT,
        "complete": True,
        "research_only": True,
        "production_files_changed": False,
        "source_sealed_manifest_sha256": _sha256(authority.manifest_path),
        "source_selected_model": authority.manifest["selected_model"],
        "source_full_handoff_verified": True,
        "source_input_authority_verified": True,
        "slice_start": SLICE_START.date().isoformat(),
        "slice_end": SLICE_END.date().isoformat(),
        "inclusive": True,
        "bootstrap_draws": int(args.bootstrap),
        "block_sizes_events": list(args.block_sizes),
        "dependency_source_pins": dependency_audit,
        "execution_provenance": audit["execution_provenance"],
        "artifacts": _artifact_manifest(output),
    }
    execution_provenance.assert_current(provenance_state)
    _atomic_json(output / "manifest.json", manifest)
    print(f"Completed exact last-three-years sealed matchup postprocess: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sealed-output",
        type=Path,
        required=True,
        help="Completed full sealed evaluator output directory (not a preflight run).",
    )
    parser.add_argument(
        "--expected-sealed-manifest-sha256",
        required=True,
        help=(
            "Externally captured SHA-256 of the completed parent manifest.json; "
            "prevents a self-consistent parent-output rewrite."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--block-sizes",
        type=lambda value: tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        ),
        default=(2, 3, 4),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

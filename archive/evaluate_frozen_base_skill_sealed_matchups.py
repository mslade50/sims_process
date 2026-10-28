"""Fail-closed sealed matchup evaluation for a frozen base-skill challenger.

This program never selects a model.  It accepts only a compact finalist or
diagnostic-challenger handoff that was frozen from the 2021-22 discovery run,
verifies the compact prediction, source prediction, and selection-manifest
hash chain, then scores 2023-01-01 through 2026-08-22 with the frozen 2021-22
probability mapper.  Closing odds are evaluation data only.

The locked weather/clean-IMG siblings are descriptive ablations.  Their sealed
results may explain a model but may not change the frozen selection decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import backtest_base_skill_market_models as market_eval  # noqa: E402
from archive.base_skill_frozen_handoff import (  # noqa: E402
    FrozenHandoff,
    validate_frozen_handoff_chain,
)
from archive import evaluate_complete_base_skill_market_candidates as discovery  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402
from archive.market_backtest_scoring import (  # noqa: E402
    american_implied_probability,
    valid_american_odds_mask,
)


ROOT = REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
DEFAULT_FINALIST_DIR = ROOT / "model_completion_v3_provenance" / "frozen_finalist"
DEFAULT_SELECTION = (
    ROOT / "model_eval_complete_discovery_v3_provenance" / "selection_manifest.json"
)
DEFAULT_CACHE = ROOT / "model_eval_clean_dg_plumbing_v4"
DEFAULT_CATALOG = ROOT / "model_completion_v3_provenance" / "candidate_catalog.csv"
DEFAULT_ROUND_PANEL = ROOT / "player_round_panel.parquet"
DEFAULT_EVENT_CROSSWALK = ROOT / "odds_event_crosswalk.parquet"
DEFAULT_ROOT_PANEL_MANIFEST = ROOT / "manifest.json"
DEFAULT_OUTPUT = ROOT / "model_eval_complete_sealed_v3_provenance"

# Independently audited frozen outcome-panel authority.  The sealed evaluator
# consumes only the round panel and event crosswalk, but the latter must be
# authorized by the exact immutable root-panel manifest rather than by an
# arbitrary self-consistent CLI file.
EXPECTED_ROOT_PANEL_MANIFEST_SHA256 = (
    "cd8dea51658a7899fa0c2d2dd534d52f456485e1528290c1fba27b33c471d3db"
)
EXPECTED_ROOT_PANEL_MANIFEST_BYTES = 12_767
EXPECTED_ROOT_PANEL_PURPOSE = (
    "Research-only frozen outcome panel for walk-forward base-skill and market "
    "evaluation."
)
EXPECTED_ROOT_EVENT_CROSSWALK_SHA256 = (
    "60959cfb194f27769c914d367a8bf3697a74966ef5ffa48a839722109d24267d"
)
EXPECTED_ROOT_EVENT_CROSSWALK_ROWS = 302

SEALED_START = pd.Timestamp("2023-01-01")
SEALED_END = pd.Timestamp("2026-08-22")
DISCOVERY_END = pd.Timestamp("2022-12-31")
FINALIST_CONTRACT = "base_skill_frozen_finalist_v1"
SELECTION_CONTRACT = "base_skill_discovery_selection_v1"
OUTPUT_CONTRACT = "base_skill_frozen_sealed_matchup_v1"
STRICT_PT_BASELINE = "current_architecture_pt_v2_approx_control"
STRICT_LEGACY_CONTROL = "legacy_skill_est_pt_current_coefficients_control"
RETROSPECTIVE_CONTROL = "legacy_pred_step2_exact_retrospective_control"
MARKETS = ("72h", "r1")
PRICE_BASES = ("close", "open")
LINE_POLICIES = (*market_eval.BOOKS, "best_of_three")
SIZINGS = ("flat_1u", "production_risk_to_win")
RELIABILITY_POLICIES = ("ungated", "suppress_unreliable_ge10pct")
EDGE_BINS = [-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf]
EDGE_LABELS = ["<0%", "0-3%", "3-5%", "5-10%", ">10%"]
SKILL_BINS = [-np.inf, 0.0, 0.5, 1.0, 1.5, np.inf]
SKILL_LABELS = ["<0", "0-0.5", "0.5-1.0", "1.0-1.5", ">=1.5"]
SKILL_GAP_BINS = [-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf]
SKILL_GAP_LABELS = ["<-1", "-1--0.5", "-0.5-0", "0-0.5", "0.5-1", ">=1"]
HISTORY_BINS = [-0.5, 0.5, 10.0, 20.0, 40.0, 80.0, np.inf]
HISTORY_LABELS = ["0", "1-9", "10-19", "20-39", "40-79", ">=80"]
ODDS_EVENT_RE = re.compile(r"^pga:R\d{7}$")
PROVENANCE_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            REPO / "archive" / "base_skill_frozen_handoff.py",
            Path(discovery.__file__).resolve(),
            *(
                Path(path).resolve()
                for path in discovery.DISCOVERY_PROVENANCE_SOURCE_CLOSURE
            ),
        )
    )
)

REQUIRED_COMPACT_COLUMNS = {
    "event_key",
    "dg_id",
    "event_start",
    "candidate",
    "finalist_core",
    "control_core",
    "split",
    "prediction_asof",
    "chronology_ok",
}
REQUIRED_SOURCE_COLUMNS = {
    "candidate",
    "event_key",
    "dg_id_key",
    "event_start_date",
    "prediction",
    "prediction_units",
    "prediction_asof",
    "training_max_event_end",
    "chronology_ok",
    "canonical_event_key",
}


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


def write_sealed_complete_manifest(
    path: Path,
    payload: Mapping[str, Any],
    provenance_state: execution_provenance.ExecutionProvenance,
) -> None:
    """Write a terminal sealed/preflight manifest only on the current snapshot."""

    execution_provenance.assert_current(provenance_state)
    _atomic_json(path, payload)


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


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _resolve_path(parent: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else parent / path


def _selection_cutoff(payload: Mapping[str, Any]) -> pd.Timestamp:
    for key in (
        "selection_window_end",
        "selection_period_end",
        "selection_cutoff",
        "discovery_end",
    ):
        parsed = pd.to_datetime(payload.get(key), errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    raise ValueError("selection manifest has no parseable discovery cutoff")


def _manifest_selected_name(payload: Mapping[str, Any]) -> str:
    finalists = payload.get("finalists")
    if isinstance(finalists, list) and finalists:
        first = finalists[0]
        return str(first.get("candidate") or first.get("name")) if isinstance(first, Mapping) else str(first)
    value = payload.get("selected_candidate") or payload.get("finalist_name")
    if value:
        return str(value)
    diagnostic = payload.get("diagnostic_challenger")
    if isinstance(diagnostic, Mapping):
        diagnostic = diagnostic.get("candidate") or diagnostic.get("name")
    if diagnostic:
        return str(diagnostic)
    raise ValueError("selection manifest declares neither finalist nor diagnostic challenger")


def _verify_selection_artifacts(selection_path: Path, selection: Mapping[str, Any]) -> int:
    verified = 0
    for entry in selection.get("artifacts", []):
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("selection manifest contains a malformed artifact record")
        path = _resolve_path(selection_path.parent, entry["path"])
        if not path.exists() or _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"selection artifact hash mismatch: {path}")
        if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"selection artifact size mismatch: {path}")
        verified += 1
    return verified


def validate_frozen_handoff(
    compact_manifest_path: Path,
    expected_selection_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, str]:
    """Delegate every sealed handoff decision to the shared fail-closed chain."""

    handoff = validate_frozen_handoff_chain(compact_manifest_path.resolve())
    if (
        expected_selection_path is not None
        and handoff.selection_manifest_path != expected_selection_path.resolve()
    ):
        raise ValueError("compact handoff points to a different selection manifest")
    return _handoff_tuple(handoff)


def _handoff_tuple(
    handoff: FrozenHandoff,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, str]:
    """Preserve the historical sealed-evaluator tuple API after shared validation."""

    compact = dict(handoff.compact)
    compact["_verified_handoff_paths"] = [str(path) for path in handoff.verified_paths]
    compact["_verified_completion_inputs"] = handoff.audit.get(
        "completion_input_bindings", {}
    )
    compact["_provenance_chain_audit"] = handoff.audit
    compact["_verified_selection_artifacts"] = handoff.audit.get(
        "selection_artifacts_verified", 0
    )
    return (
        compact,
        dict(handoff.selection),
        handoff.compact_prediction_path,
        handoff.selection_manifest_path,
        handoff.selected_name,
    )


def validate_sealed_panel_authorities(
    handoff: FrozenHandoff,
    round_panel_path: Path,
    event_crosswalk_path: Path,
    *,
    panel_manifest_path: Path = DEFAULT_ROOT_PANEL_MANIFEST,
) -> dict[str, Any]:
    """Bind sealed weather inputs to completion and the frozen root panel."""

    round_panel_path = round_panel_path.resolve()
    event_crosswalk_path = event_crosswalk_path.resolve()
    panel_manifest_path = panel_manifest_path.resolve()
    completion_inputs = handoff.audit.get("completion_input_bindings")
    if not isinstance(completion_inputs, Mapping):
        raise ValueError("shared handoff lacks completion input bindings")
    round_binding = completion_inputs.get("round_panel")
    if not isinstance(round_binding, Mapping):
        raise ValueError("shared handoff lacks completion round_panel authority")
    declared_round_path = Path(str(round_binding.get("path", ""))).resolve()
    declared_round_hash = str(round_binding.get("sha256", "")).lower()
    if declared_round_path != round_panel_path:
        raise ValueError("sealed --round-panel differs from the completion authority")
    if not round_panel_path.is_file() or _sha256(round_panel_path) != declared_round_hash:
        raise ValueError("sealed round-panel hash differs from the completion authority")

    if panel_manifest_path != DEFAULT_ROOT_PANEL_MANIFEST.resolve():
        raise ValueError("sealed panel-manifest path differs from the code-pinned authority")
    if not panel_manifest_path.is_file():
        raise FileNotFoundError(panel_manifest_path)
    if panel_manifest_path.stat().st_size != EXPECTED_ROOT_PANEL_MANIFEST_BYTES:
        raise ValueError("root panel manifest byte size differs from the audited authority")
    if _sha256(panel_manifest_path) != EXPECTED_ROOT_PANEL_MANIFEST_SHA256:
        raise ValueError("root panel manifest hash differs from the audited authority")
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("purpose") != EXPECTED_ROOT_PANEL_PURPOSE:
        raise ValueError("root panel manifest purpose differs from the audited authority")
    if panel_manifest.get("production_changes") is not False:
        raise ValueError("root panel manifest does not explicitly preserve production")
    if panel_manifest.get("parameters") != {
        "end": "2026-08-22",
        "history_start": "2015-04-01",
        "img_signal_model_version": "predictive_skill_pga_validated_v2",
        "market_start": "2020-01-01",
        "tour": "pga",
    }:
        raise ValueError("root panel manifest parameters differ from the audited authority")
    crosswalk_entry = panel_manifest.get("outputs", {}).get("odds_event_crosswalk")
    if not isinstance(crosswalk_entry, Mapping):
        raise ValueError("root panel manifest lacks odds_event_crosswalk output authority")
    declared_crosswalk_path = Path(str(crosswalk_entry.get("path", ""))).resolve()
    declared_crosswalk_hash = str(crosswalk_entry.get("sha256", "")).lower()
    if event_crosswalk_path != DEFAULT_EVENT_CROSSWALK.resolve():
        raise ValueError("sealed --event-crosswalk differs from the code-pinned authority")
    if declared_crosswalk_path != event_crosswalk_path:
        raise ValueError("root panel manifest event-crosswalk path mismatch")
    if declared_crosswalk_hash != EXPECTED_ROOT_EVENT_CROSSWALK_SHA256:
        raise ValueError("root panel manifest event-crosswalk hash mismatch")
    if int(crosswalk_entry.get("rows", -1)) != EXPECTED_ROOT_EVENT_CROSSWALK_ROWS:
        raise ValueError("root panel manifest event-crosswalk row count mismatch")
    if (
        not event_crosswalk_path.is_file()
        or _sha256(event_crosswalk_path) != EXPECTED_ROOT_EVENT_CROSSWALK_SHA256
    ):
        raise ValueError("sealed event-crosswalk bytes differ from the audited authority")
    return {
        "round_panel_path": str(round_panel_path),
        "round_panel_sha256": declared_round_hash,
        "event_crosswalk_path": str(event_crosswalk_path),
        "event_crosswalk_sha256": EXPECTED_ROOT_EVENT_CROSSWALK_SHA256,
        "root_panel_manifest_path": str(panel_manifest_path),
        "root_panel_manifest_sha256": EXPECTED_ROOT_PANEL_MANIFEST_SHA256,
        "verified_paths": tuple(
            sorted(
                {round_panel_path, event_crosswalk_path, panel_manifest_path}, key=str
            )
        ),
    }


def revalidate_sealed_handoff_after_begin(
    compact_manifest_path: Path,
    expected_selection_path: Path,
    *,
    expected_handoff_paths: Iterable[Path],
    expected_authority_paths: Iterable[Path],
    provenance_state: execution_provenance.ExecutionProvenance,
    round_panel_path: Path,
    event_crosswalk_path: Path,
    panel_manifest_path: Path = DEFAULT_ROOT_PANEL_MANIFEST,
) -> tuple[FrozenHandoff, dict[str, Any]]:
    """Close the validation/fingerprinting TOCTOU gap before sealed reads."""

    handoff = validate_frozen_handoff_chain(compact_manifest_path.resolve())
    if handoff.selection_manifest_path != expected_selection_path.resolve():
        raise ValueError("post-begin compact handoff changed its selection manifest")
    expected_handoff = {Path(path).resolve() for path in expected_handoff_paths}
    actual_handoff = {Path(path).resolve() for path in handoff.verified_paths}
    if actual_handoff != expected_handoff:
        raise ValueError("post-begin frozen handoff resolved-path set changed")
    authority = validate_sealed_panel_authorities(
        handoff,
        round_panel_path,
        event_crosswalk_path,
        panel_manifest_path=panel_manifest_path,
    )
    expected_authority = {Path(path).resolve() for path in expected_authority_paths}
    actual_authority = {
        Path(path).resolve() for path in authority["verified_paths"]
    }
    if actual_authority != expected_authority:
        raise ValueError("post-begin sealed panel-authority path set changed")
    tracked = {
        Path(str(record["path"])).resolve()
        for record in provenance_state.input_fingerprints
    }
    expected_tracked = expected_handoff | expected_authority
    if tracked != expected_tracked:
        raise ValueError(
            "execution provenance input set differs from the complete sealed input set"
        )
    if not actual_handoff.issubset(tracked):
        raise ValueError("post-begin frozen handoff includes an untracked path")
    return handoff, authority


def _candidate_names(
    selected_name: str, selection: Mapping[str, Any]
) -> tuple[list[str], dict[str, str]]:
    locked = selection["locked_diagnostic_ablation_set"]
    ablations = [str(value) for value in locked.get("included_candidates", [])]
    names = list(
        dict.fromkeys(
            [
                selected_name,
                STRICT_PT_BASELINE,
                STRICT_LEGACY_CONTROL,
                RETROSPECTIVE_CONTROL,
                *ablations,
            ]
        )
    )
    roles = {name: "locked_descriptive_ablation" for name in ablations}
    roles[selected_name] = (
        "promoted_finalist"
        if bool(selection.get("promotion_gate_passed"))
        else "diagnostic_challenger"
    )
    roles[STRICT_PT_BASELINE] = "strict_point_in_time_architecture_control"
    roles[STRICT_LEGACY_CONTROL] = "strict_prior_legacy_input_current_coefficients_control"
    roles[RETROSPECTIVE_CONTROL] = "exact_current_code_retrospective_control"
    return names, roles


def _source_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def load_frozen_predictions(
    compact: Mapping[str, Any],
    compact_prediction_path: Path,
    selection: Mapping[str, Any],
    selected_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], dict[str, Any]]:
    """Load only prespecified names and verify selected/control compact values."""

    compact_predictions = pd.read_parquet(compact_prediction_path)
    _require_columns(compact_predictions, REQUIRED_COMPACT_COLUMNS, "compact predictions")
    if compact_predictions.duplicated(["event_key", "dg_id"]).any():
        raise ValueError("compact predictions contain duplicate event/player keys")
    if set(compact_predictions["candidate"].astype(str)) != {selected_name}:
        raise ValueError("compact rows do not contain exactly the selected candidate")
    compact_predictions["event_start"] = pd.to_datetime(
        compact_predictions["event_start"], errors="coerce"
    ).dt.normalize()
    compact_predictions["prediction_asof"] = pd.to_datetime(
        compact_predictions["prediction_asof"], errors="coerce"
    ).dt.normalize()
    chronology_bad = (
        ~compact_predictions["chronology_ok"].fillna(False).astype(bool)
        | compact_predictions["prediction_asof"].isna()
        | compact_predictions["prediction_asof"].gt(compact_predictions["event_start"])
    )
    if chronology_bad.any():
        raise ValueError("compact prediction chronology violation")
    if not compact_predictions["event_key"].astype(str).map(
        lambda value: bool(ODDS_EVENT_RE.fullmatch(value))
    ).all():
        raise ValueError("compact event keys are not canonical odds event keys")

    selection_source_path = _verified_selection_input(selection, "predictions")
    source_path = Path(str(compact.get("source_prediction_file", "")))
    if not source_path.is_absolute():
        source_path = compact_prediction_path.parent / source_path
    source_path = source_path.resolve()
    compact_source_sha = str(compact.get("source_prediction_sha256", "")).lower()
    selection_source_sha = str(
        selection.get("inputs", {}).get("predictions", {}).get("sha256", "")
    ).lower()
    if source_path != selection_source_path.resolve():
        raise ValueError(
            "compact source prediction path differs from discovery selection input"
        )
    if not compact_source_sha or compact_source_sha != selection_source_sha:
        raise ValueError(
            "compact source prediction hash differs from discovery selection input"
        )
    source_manifest_path = _verified_selection_input(selection, "candidate_manifest")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _, builder_prediction_sha = discovery._required_manifest_output(
        source_manifest,
        source_manifest_path,
        "player_event_predictions.parquet",
        expected_path=source_path,
    )
    if builder_prediction_sha != compact_source_sha:
        raise ValueError(
            "builder, discovery, and compact prediction hashes do not form one chain"
        )
    if not source_path.exists() or _sha256(source_path) != compact_source_sha:
        raise ValueError("hashed source prediction file is missing or changed")
    source_schema = _source_columns(source_path)
    missing_source = sorted(REQUIRED_SOURCE_COLUMNS - set(source_schema))
    if missing_source:
        raise ValueError(f"source predictions missing columns: {missing_source}")
    names, roles = _candidate_names(selected_name, selection)
    optional = [
        column
        for column in (
            "dg_id",
            "event_start",
            "reliability_ok",
            "cross_tour_fallback_reason",
            "history_rounds",
            "dg_history_rounds",
            "days_since_last_round",
            "prior_event_withdrawal",
            "img_history_rounds",
            "img_history_coverage",
            "days_since_last_img_round",
            "weather_arm",
            "target_arm",
        )
        if column in source_schema
    ]
    read_columns = sorted(REQUIRED_SOURCE_COLUMNS | set(optional))
    source = pd.read_parquet(
        source_path,
        columns=read_columns,
        filters=[("candidate", "in", names)],
    )
    source["event_start_date"] = pd.to_datetime(
        source["event_start_date"], errors="coerce"
    ).dt.normalize()
    source["prediction_asof"] = pd.to_datetime(
        source["prediction_asof"], errors="coerce"
    ).dt.normalize()
    source["training_max_event_end"] = pd.to_datetime(
        source["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    source["dg_id_key"] = source["dg_id_key"].astype(str)
    source = source.loc[
        source["event_start_date"].between(SEALED_START, SEALED_END)
    ].copy()
    if source.empty:
        raise ValueError("source predictions contain no sealed rows")
    missing_candidates = sorted(set(names) - set(source["candidate"].astype(str)))
    if missing_candidates:
        raise ValueError(f"prespecified sealed candidates missing: {missing_candidates}")
    if source.duplicated(["candidate", "event_key", "dg_id_key"]).any():
        raise ValueError("source predictions contain duplicate sealed keys")
    chronology_bad = (
        ~source["chronology_ok"].fillna(False).astype(bool)
        | source["prediction_asof"].isna()
        | source["prediction_asof"].gt(source["event_start_date"])
        | (
            source["training_max_event_end"].notna()
            & source["training_max_event_end"].ge(source["event_start_date"])
        )
    )
    if chronology_bad.any():
        raise ValueError(f"sealed source chronology violations: {int(chronology_bad.sum())}")
    discovery.validate_prediction_units(source)
    if not np.isfinite(pd.to_numeric(source["prediction"], errors="coerce")).all():
        raise ValueError("sealed source predictions contain non-finite skill")

    # Cross-check compact values against the exact hashed source on sealed rows.
    compact_sealed = compact_predictions.loc[
        compact_predictions["event_start"].between(SEALED_START, SEALED_END)
    ].copy()
    if compact_sealed.empty:
        raise ValueError("compact handoff contains no sealed prediction rows")
    if not compact_sealed["split"].astype(str).str.casefold().eq("sealed").all():
        raise ValueError("compact sealed-date rows are not labeled split='sealed'")
    compact_sealed["dg_id_key"] = compact_sealed["dg_id"].map(
        market_eval._stable_id
    )
    if compact_sealed["dg_id_key"].eq("").any():
        raise ValueError("compact sealed handoff contains blank golfer IDs")
    selected_source = source.loc[source["candidate"].eq(selected_name)].copy()
    baseline_source = source.loc[source["candidate"].eq(STRICT_PT_BASELINE)].copy()
    for label, frame in (
        ("selected", selected_source),
        ("strict-PT control", baseline_source),
    ):
        if frame.empty:
            raise ValueError(f"source predictions contain no sealed {label} rows")
        if frame.duplicated(["canonical_event_key", "dg_id_key"]).any():
            raise ValueError(f"source sealed {label} canonical keys are not unique")
    compact_keys = pd.MultiIndex.from_frame(
        compact_sealed[["event_key", "dg_id_key"]].astype(str)
    )
    selected_keys = pd.MultiIndex.from_frame(
        selected_source[["canonical_event_key", "dg_id_key"]].astype(str)
    )
    baseline_keys = pd.MultiIndex.from_frame(
        baseline_source[["canonical_event_key", "dg_id_key"]].astype(str)
    )
    if set(compact_keys) != set(selected_keys) or set(compact_keys) != set(
        baseline_keys
    ):
        raise ValueError(
            "compact sealed keys do not equal the full selected/control source menus"
        )
    selected_compare = compact_sealed.merge(
        selected_source[
            [
                "canonical_event_key",
                "dg_id_key",
                "event_start_date",
                "prediction_asof",
                "prediction",
            ]
        ].rename(
            columns={
                "event_start_date": "source_event_start_date",
                "prediction_asof": "source_prediction_asof",
            }
        ),
        left_on=["event_key", "dg_id_key"],
        right_on=["canonical_event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    if selected_compare["prediction"].isna().any() or not np.allclose(
        selected_compare["finalist_core"], selected_compare["prediction"], atol=1e-12
    ):
        raise ValueError("compact finalist values disagree with hashed source predictions")
    if not selected_compare["event_start"].eq(
        selected_compare["source_event_start_date"]
    ).all() or not selected_compare["prediction_asof"].eq(
        selected_compare["source_prediction_asof"]
    ).all():
        raise ValueError("compact finalist dates disagree with hashed source predictions")

    baseline_compare = compact_sealed.merge(
        baseline_source[
            ["canonical_event_key", "dg_id_key", "event_start_date", "prediction"]
        ].rename(columns={"event_start_date": "baseline_event_start_date"}),
        left_on=["event_key", "dg_id_key"],
        right_on=["canonical_event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    if baseline_compare["prediction"].isna().any() or not np.allclose(
        baseline_compare["control_core"], baseline_compare["prediction"], atol=1e-12
    ):
        raise ValueError("compact strict-PT control disagrees with hashed source predictions")
    if not baseline_compare["event_start"].eq(
        baseline_compare["baseline_event_start_date"]
    ).all():
        raise ValueError("compact strict-PT event dates disagree with source predictions")
    audit = {
        "compact_rows": int(len(compact_predictions)),
        "compact_sealed_rows": int(len(compact_sealed)),
        "source_sealed_rows": int(len(source)),
        "candidate_names": names,
        "candidate_roles": roles,
        "source_predictions_sha256": _sha256(source_path),
        "selection_source_path_and_hash_match": True,
        "compact_sealed_full_menu_match": True,
        "compact_source_value_matches": True,
    }
    return source, compact_predictions, names, roles, audit


def _verified_selection_input(
    selection: Mapping[str, Any],
    key: str,
    expected_path: Path | None = None,
) -> Path:
    entry = selection.get("inputs", {}).get(key)
    if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"selection manifest lacks hashed input {key!r}")
    path = Path(str(entry["path"])).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"sealed evaluator {key} path differs from discovery input")
    if not path.exists() or _sha256(path) != str(entry["sha256"]):
        raise ValueError(f"selection input hash mismatch: {key}")
    return path


def load_catalog_and_quotes(
    selection: Mapping[str, Any],
    catalog_path: Path,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _verified_selection_input(selection, "candidate_catalog", catalog_path)
    source_manifest_path = _verified_selection_input(selection, "candidate_manifest")
    frozen_catalog_path = _verified_selection_input(
        selection, "frozen_candidate_catalog"
    )
    if _sha256(frozen_catalog_path) != discovery.EXPECTED_FROZEN_CATALOG_SHA256:
        raise ValueError("sealed evaluator received an unapproved frozen catalog")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    discovery._required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog.csv",
        expected_path=catalog_path,
    )
    declared_frozen_path, _ = discovery._required_manifest_output(
        source_manifest,
        source_manifest_path,
        "candidate_catalog_frozen.json",
        expected_path=frozen_catalog_path,
    )
    quote_path = cache_dir / "normalized_matchup_quotes.parquet"
    _verified_selection_input(selection, "normalized_quotes", quote_path)
    cache_manifest_path = cache_dir / "manifest.json"
    _verified_selection_input(
        selection, "evaluation_cache_manifest", cache_manifest_path
    )
    catalog = pd.read_csv(catalog_path)
    _require_columns(
        catalog,
        {
            "name",
            "family",
            "source",
            "weather_arm",
            "target_arm",
            "selection_eligible",
            "sensitivity_only",
            "control",
        },
        "candidate catalog",
    )
    if catalog["name"].astype(str).duplicated().any():
        raise ValueError("candidate catalog names are not unique")
    for column in ("selection_eligible", "sensitivity_only", "control"):
        catalog[column] = catalog[column].map(discovery._as_bool)
    frozen_catalog_audit = discovery.validate_frozen_catalog_contract(
        catalog, source_manifest, declared_frozen_path
    )
    quotes = pd.read_parquet(quote_path)
    _require_columns(
        quotes,
        {
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
            "tie_rule",
        },
        "normalized matchup quotes",
    )
    quotes["event_start_date"] = pd.to_datetime(
        quotes["event_start_date"], errors="coerce"
    ).dt.normalize()
    quotes["p1_id"] = quotes["p1_id"].map(market_eval._stable_id)
    quotes["p2_id"] = quotes["p2_id"].map(market_eval._stable_id)
    quotes = quotes.loc[
        quotes["event_start_date"].between(SEALED_START, SEALED_END)
        & quotes["market"].isin(MARKETS)
        & quotes["book"].isin(market_eval.BOOKS)
    ].copy()
    if quotes.empty:
        raise ValueError("normalized matchup cache contains no sealed quotes")
    if set(MARKETS) - set(quotes["market"].astype(str)):
        raise ValueError("sealed cache does not contain both 72h and R1 matchups")
    if quotes["mapped_event_key"].isna().any():
        raise ValueError("sealed quotes contain unmapped events")
    if quotes.duplicated("quote_id").any():
        raise ValueError("sealed quote IDs are not unique")
    logical_book_keys = ["mapped_event_key", "market", "book", "pair_key"]
    if quotes.duplicated(logical_book_keys).any():
        raise ValueError("sealed logical pair/book keys are not unique")
    quote_integrity_audit = market_eval.validate_normalized_matchup_quotes(quotes)
    valid_open = valid_american_odds_mask(
        pd.to_numeric(quotes["p1_open"], errors="coerce")
    ) & valid_american_odds_mask(pd.to_numeric(quotes["p2_open"], errors="coerce"))
    valid_close = valid_american_odds_mask(
        pd.to_numeric(quotes["p1_close"], errors="coerce")
    ) & valid_american_odds_mask(pd.to_numeric(quotes["p2_close"], errors="coerce"))
    audit = {
        "quote_rows": int(len(quotes)),
        "events": int(quotes["mapped_event_key"].nunique()),
        "markets": sorted(quotes["market"].astype(str).unique()),
        "books": sorted(quotes["book"].astype(str).unique()),
        "date_min": quotes["event_start_date"].min().date().isoformat(),
        "date_max": quotes["event_start_date"].max().date().isoformat(),
        "quote_key_duplicates": 0,
        "logical_book_key_duplicates": 0,
        "quote_identity_and_outcome_integrity": quote_integrity_audit,
        "frozen_catalog": frozen_catalog_audit,
        "same_book_price_coverage": {
            "quote_rows": int(len(quotes)),
            "valid_two_sided_open_rows": int(valid_open.sum()),
            "invalid_two_sided_open_rows": int((~valid_open).sum()),
            "valid_two_sided_close_rows": int(valid_close.sum()),
            "invalid_two_sided_close_rows": int((~valid_close).sum()),
            "strict_conversion_after_filter": True,
        },
    }
    return catalog, quotes, audit


def frozen_probability_mapper(
    selection: Mapping[str, Any], candidate_names: Sequence[str]
) -> pd.DataFrame:
    raw = selection.get("final_frozen_probability_parameters")
    if not isinstance(raw, list) or not raw:
        raise ValueError("selection manifest has no frozen probability mapper")
    mapper = pd.DataFrame(raw)
    _require_columns(
        mapper,
        {
            "candidate",
            "market",
            "fit_end",
            "sigma",
            "tie_threshold",
            "frozen_before_sealed_scoring",
        },
        "frozen probability mapper",
    )
    mapper["fit_end"] = pd.to_datetime(mapper["fit_end"], errors="coerce").dt.normalize()
    mapper["sigma"] = pd.to_numeric(mapper["sigma"], errors="coerce")
    mapper["tie_threshold"] = pd.to_numeric(
        mapper["tie_threshold"], errors="coerce"
    )
    mapper = mapper.loc[
        mapper["candidate"].astype(str).isin(candidate_names)
        & mapper["market"].astype(str).isin(MARKETS)
    ].copy()
    expected = pd.MultiIndex.from_product(
        [list(candidate_names), list(MARKETS)], names=["candidate", "market"]
    )
    actual = pd.MultiIndex.from_frame(mapper[["candidate", "market"]])
    missing = expected.difference(actual)
    extra_duplicates = mapper.duplicated(["candidate", "market"])
    if len(missing) or extra_duplicates.any():
        raise ValueError(
            f"frozen mapper missing={list(missing)}, duplicates={int(extra_duplicates.sum())}"
        )
    bad = (
        mapper["fit_end"].isna()
        | mapper["fit_end"].gt(DISCOVERY_END)
        | ~mapper["frozen_before_sealed_scoring"].map(discovery._as_bool)
        | ~np.isfinite(mapper["sigma"])
        | mapper["sigma"].le(0)
        | ~np.isfinite(mapper["tie_threshold"])
        | mapper["tie_threshold"].lt(0)
    )
    if bad.any():
        raise ValueError("one or more frozen mapper rows violate the discovery-only gate")
    mapper["sealed_data_used_for_mapper"] = False
    return mapper.sort_values(["candidate", "market"]).reset_index(drop=True)


def quote_coverage_audit(
    quotes: pd.DataFrame, predictions: pd.DataFrame, candidate_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[pd.DataFrame] = []
    total_logical = (
        quotes.sort_values(["mapped_event_key", "market", "pair_key", "book"])
        .drop_duplicates(["mapped_event_key", "market", "pair_key"])
    )
    for candidate in candidate_names:
        keys = predictions.loc[
            predictions["candidate"].eq(candidate), ["event_key", "dg_id_key"]
        ].drop_duplicates()
        left = keys.rename(
            columns={"event_key": "mapped_event_key", "dg_id_key": "p1_id"}
        ).assign(p1_available=True)
        right = keys.rename(
            columns={"event_key": "mapped_event_key", "dg_id_key": "p2_id"}
        ).assign(p2_available=True)
        checked = quotes.merge(
            left,
            on=["mapped_event_key", "p1_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            right,
            on=["mapped_event_key", "p2_id"],
            how="left",
            validate="many_to_one",
        )
        checked["p1_available"] = checked["p1_available"].fillna(False)
        checked["p2_available"] = checked["p2_available"].fillna(False)
        checked["both_available"] = checked["p1_available"] & checked["p2_available"]
        for (year, market), group in checked.groupby(
            [checked["event_start_date"].dt.year, "market"], observed=True
        ):
            raw_logical = total_logical.loc[
                total_logical["event_start_date"].dt.year.eq(year)
                & total_logical["market"].eq(market)
            ]
            covered_logical = group.loc[group["both_available"]].drop_duplicates(
                ["mapped_event_key", "market", "pair_key"]
            )
            offered_players = set(group["p1_id"]) | set(group["p2_id"])
            missing_players = set(
                group.loc[~group["p1_available"], "p1_id"]
            ) | set(group.loc[~group["p2_available"], "p2_id"])
            rows.append(
                {
                    "candidate": candidate,
                    "year": int(year),
                    "market": market,
                    "quote_rows": int(len(group)),
                    "covered_quote_rows": int(group["both_available"].sum()),
                    "quote_coverage": float(group["both_available"].mean()),
                    "logical_pairs": int(len(raw_logical)),
                    "covered_logical_pairs": int(len(covered_logical)),
                    "logical_pair_coverage": (
                        float(len(covered_logical) / len(raw_logical))
                        if len(raw_logical)
                        else np.nan
                    ),
                    "offered_players": int(len(offered_players)),
                    "missing_players": int(len(missing_players)),
                }
            )
        missing = checked.loc[~checked["both_available"]].copy()
        if not missing.empty:
            missing["candidate"] = candidate
            missing["missing_reason"] = np.select(
                [
                    ~missing["p1_available"] & ~missing["p2_available"],
                    ~missing["p1_available"],
                    ~missing["p2_available"],
                ],
                ["both_players_absent", "p1_absent", "p2_absent"],
                default="unknown",
            )
            missing_rows.append(
                missing[
                    [
                        "candidate",
                        "quote_id",
                        "mapped_event_key",
                        "event_start_date",
                        "market",
                        "book",
                        "pair_key",
                        "p1_id",
                        "p2_id",
                        "missing_reason",
                    ]
                ]
            )
    detail = (
        pd.concat(missing_rows, ignore_index=True)
        if missing_rows
        else pd.DataFrame(
            columns=[
                "candidate",
                "quote_id",
                "mapped_event_key",
                "event_start_date",
                "market",
                "book",
                "pair_key",
                "p1_id",
                "p2_id",
                "missing_reason",
            ]
        )
    )
    return pd.DataFrame(rows), detail


def enforce_primary_quote_coverage(
    coverage: pd.DataFrame, selected_candidate: str
) -> None:
    """Fail closed if the frozen model or fair control drops an offered quote."""

    required = {selected_candidate, STRICT_PT_BASELINE}
    primary = coverage.loc[coverage["candidate"].isin(required)].copy()
    missing_candidates = required - set(primary["candidate"].astype(str))
    if missing_candidates:
        raise ValueError(
            f"primary quote coverage audit lacks candidates: {sorted(missing_candidates)}"
        )
    bad = primary.loc[
        primary["covered_quote_rows"].ne(primary["quote_rows"])
        | primary["covered_logical_pairs"].ne(primary["logical_pairs"])
        | primary["missing_players"].ne(0)
    ]
    if not bad.empty:
        raise ValueError(
            "selected/strict-PT quote coverage is incomplete: "
            + bad[
                [
                    "candidate",
                    "year",
                    "market",
                    "covered_quote_rows",
                    "quote_rows",
                    "missing_players",
                ]
            ].to_json(orient="records")
        )


def score_with_frozen_mapper(
    quotes: pd.DataFrame,
    predictions: pd.DataFrame,
    mapper: pd.DataFrame,
    roles: Mapping[str, str],
) -> pd.DataFrame:
    paired = market_eval.attach_skill_differences(quotes, predictions)
    if paired.empty:
        raise ValueError("no sealed predictions mapped to matchup quotes")
    pieces: list[pd.DataFrame] = []
    for row in mapper.to_dict(orient="records"):
        group = paired.loc[
            paired["candidate"].eq(row["candidate"])
            & paired["market"].eq(row["market"])
        ]
        if group.empty:
            continue
        piece = market_eval.score_probabilities(
            group, float(row["sigma"]), float(row["tie_threshold"])
        )
        piece["probability_fit_end"] = pd.Timestamp(row["fit_end"])
        piece["probability_calibration_source"] = row.get("calibration_source")
        piece["candidate_role"] = piece["candidate"].map(roles)
        pieces.append(piece)
    if not pieces:
        raise ValueError("frozen mapper produced no sealed probabilities")
    scored = pd.concat(pieces, ignore_index=True)
    if scored.duplicated(["candidate", "quote_id"]).any():
        raise ValueError("sealed candidate/quote probability keys are not unique")
    if not scored["probability_fit_end"].le(DISCOVERY_END).all():
        raise ValueError("sealed probabilities contain a post-discovery mapper")
    if set(scored["candidate"].astype(str)) != set(roles):
        missing = sorted(set(roles) - set(scored["candidate"].astype(str)))
        raise ValueError(f"one or more frozen candidates received no sealed score: {missing}")
    return scored


def frozen_opener_blend_choice(
    selection: Mapping[str, Any],
    selected_candidate: str,
) -> dict[str, Any]:
    """Validate and return one discovery-frozen opener choice."""
    sensitivity = selection.get("opener_blend_sensitivity")
    if not isinstance(sensitivity, Mapping):
        raise ValueError("selection manifest lacks frozen opener-blend sensitivity")
    if sensitivity.get("neutral_base_redefined") is not False:
        raise ValueError("opener blend may not redefine the neutral base")
    choices = sensitivity.get("choices")
    if not isinstance(choices, list):
        raise ValueError("selection manifest opener choices are malformed")
    matches = [
        row
        for row in choices
        if isinstance(row, Mapping)
        and str(row.get("candidate")) == selected_candidate
    ]
    if len(matches) != 1:
        raise ValueError("selected candidate lacks one unique frozen opener weight")
    choice = matches[0]
    if (
        pd.Timestamp(choice.get("selection_window_end")) > DISCOVERY_END
        or choice.get("sealed_data_used_for_selection") is not False
        or choice.get("neutral_base_redefined") is not False
        or choice.get("same_book_open_close_required") is not True
    ):
        raise ValueError("frozen opener choice violates discovery-only provenance")
    weight = float(choice["selected_opener_weight"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError("frozen opener weight lies outside [0,1]")
    return dict(choice)


def append_frozen_opener_blend_sensitivity(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    selection: Mapping[str, Any],
    selected_candidate: str,
    roles: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], dict[str, Any]]:
    """Apply the discovery-frozen opener weight without using close prices.

    The discovery rule first chooses one best valid opener across the frozen
    three-book menu for each *canonical* logical pair.  That one blended
    canonical probability is then copied to every eligible execution-book
    quote for the pair.  Execution odds (and therefore same-book CLV) remain
    the original quote-row values; they never affect the blended probability.
    """

    choice = frozen_opener_blend_choice(selection, selected_candidate)
    weight = float(choice["selected_opener_weight"])
    source = scored.loc[scored["candidate"].eq(selected_candidate)].copy()
    if source.empty:
        raise ValueError("selected candidate has no sealed quote rows")
    source_quote_rows = int(len(source))
    source_logical_pairs = int(
        source[["mapped_event_key", "market", "pair_key"]]
        .drop_duplicates()
        .shape[0]
    )
    source["book"] = source["book"].astype(str).str.strip().str.lower()
    execution = source.loc[source["book"].isin(market_eval.BOOKS)].copy()
    if execution.empty:
        raise ValueError("selected candidate has no execution rows at the frozen books")

    # Validate all execution rows before any odds-based filtering, so malformed
    # or book-dependent model probabilities cannot hide behind a missing opener.
    market_eval._canonical_logical_pairs(execution)

    # Reuse the exact discovery selector.  This is intentionally a direct call
    # rather than a second implementation whose tie-break or validity rules
    # could drift after the discovery decision was frozen.
    reference, opener_rule_audit = (
        discovery.best_available_same_book_opening_pairs(source)
    )
    nonpush = 1.0 - pd.to_numeric(
        reference["canonical_push_probability"], errors="coerce"
    )
    model = pd.to_numeric(reference["canonical_probability"], errors="coerce")
    valid_model = model.notna() & nonpush.notna() & nonpush.gt(market_eval.EPS)
    invalid_model_rows = int((~valid_model).sum())
    reference = reference.loc[valid_model].copy()
    nonpush = nonpush.loc[valid_model].to_numpy(float)
    model = model.loc[valid_model].to_numpy(float)
    if reference.empty:
        raise ValueError("selected candidate has no valid opener/model probability rows")

    # The selector has already filtered invalid two-sided prices.  Conversion
    # occurs only after that filter, preserving the strict-odds fail-closed rule.
    left = np.asarray(
        american_implied_probability(reference["canonical_left_open"]),
        dtype=float,
    )
    right = np.asarray(
        american_implied_probability(reference["canonical_right_open"]),
        dtype=float,
    )
    open_novig = np.clip(
        left / (left + right), market_eval.EPS, 1.0 - market_eval.EPS
    )
    model_conditional = np.clip(
        model / nonpush,
        market_eval.EPS,
        1.0 - market_eval.EPS,
    )
    blended_conditional = expit(
        (1.0 - weight) * logit(model_conditional) + weight * logit(open_novig)
    )

    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    reference_map = reference[
        keys
        + [
            "book",
            "canonical_left_open",
            "canonical_right_open",
            "canonical_left_close",
            "canonical_right_close",
            "canonical_push_probability",
        ]
    ].copy()
    reference_map = reference_map.rename(
        columns={
            "book": "opener_blend_reference_book",
            "canonical_left_open": "opener_blend_reference_canonical_left_open",
            "canonical_right_open": "opener_blend_reference_canonical_right_open",
            "canonical_left_close": "opener_blend_reference_canonical_left_close",
            "canonical_right_close": "opener_blend_reference_canonical_right_close",
            "canonical_push_probability": "opener_blend_canonical_push_probability",
        }
    )
    reference_map["opener_blend_canonical_conditional_probability"] = (
        blended_conditional
    )
    if reference_map.duplicated(keys).any():
        raise AssertionError("frozen opener reference keys are not unique")

    # Preserve each row's execution/open/close fields.  Only the probability is
    # propagated from the one canonical discovery-rule reference for the pair.
    base = execution.merge(reference_map, on=keys, how="inner", validate="many_to_one")
    if base.empty:
        raise ValueError("no execution rows map to a valid frozen opener reference")
    canonical_left = base["pair_key"].str.split("|", regex=False).str[0]
    flip = base["p1_id"].map(market_eval._stable_id).ne(canonical_left)
    canonical_conditional = base[
        "opener_blend_canonical_conditional_probability"
    ].to_numpy(float)
    row_conditional = np.where(flip, 1.0 - canonical_conditional, canonical_conditional)
    push = base["opener_blend_canonical_push_probability"].to_numpy(float)

    sensitivity_name = f"{selected_candidate}__frozen_opener_blend"
    base["source_candidate"] = selected_candidate
    base["candidate"] = sensitivity_name
    base["push_probability"] = push
    base["p1_pricing_probability"] = row_conditional
    base["p1_probability"] = row_conditional * (1.0 - push)
    base["candidate_role"] = "frozen_discovery_opener_blend_sensitivity"
    base["probability_calibration_source"] = (
        base["probability_calibration_source"].fillna("frozen_discovery_mapper").astype(str)
        + "+discovery_frozen_same_book_open_blend"
    )
    base["opener_blend_weight"] = weight
    prediction_clone = predictions.loc[
        predictions["candidate"].eq(selected_candidate)
    ].copy()
    prediction_clone["source_candidate"] = selected_candidate
    prediction_clone["candidate"] = sensitivity_name
    output_predictions = pd.concat(
        [predictions, prediction_clone], ignore_index=True, sort=False
    )
    output_scored = pd.concat([scored, base], ignore_index=True, sort=False)
    output_roles = dict(roles)
    output_roles[sensitivity_name] = "frozen_discovery_opener_blend_sensitivity"
    audit = {
        "candidate": sensitivity_name,
        "source_candidate": selected_candidate,
        "selected_opener_weight": weight,
        "selection_window_end": str(choice["selection_window_end"]),
        "sealed_data_used_for_selection": False,
        "neutral_base_redefined": False,
        "uses_closing_odds_as_feature": False,
        "source_quote_rows": source_quote_rows,
        "frozen_three_book_execution_quote_rows": int(len(execution)),
        "disallowed_execution_book_rows": int(source_quote_rows - len(execution)),
        "same_book_open_novig_rows": int(len(reference_map)),
        "invalid_or_missing_same_book_open_rows": int(
            opener_rule_audit["invalid_same_book_open_rows"]
        ),
        "invalid_or_missing_same_book_close_rows": int(
            opener_rule_audit["invalid_same_book_close_rows"]
        ),
        "invalid_model_probability_rows": invalid_model_rows,
        "source_logical_pairs": source_logical_pairs,
        "valid_same_book_open_logical_pairs": int(len(reference_map)),
        "propagated_execution_quote_rows": int(len(base)),
        "propagated_execution_logical_pairs": int(
            base[["mapped_event_key", "market", "pair_key"]]
            .drop_duplicates()
            .shape[0]
        ),
        "canonical_probability_is_book_invariant": True,
        "opener_reference_rule": opener_rule_audit,
        "strict_odds_conversion_after_validity_filter": True,
        "asynchronous_openers_not_execution_proof": True,
    }
    return (
        output_scored,
        output_predictions,
        [*roles.keys(), sensitivity_name],
        output_roles,
        audit,
    )


def paired_proper_event_deltas(
    logical: pd.DataFrame, baseline_candidate: str
) -> pd.DataFrame:
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    resolved["conditional_probability"] = (
        resolved["canonical_probability"]
        / (1.0 - resolved["canonical_push_probability"])
    ).clip(market_eval.EPS, 1.0 - market_eval.EPS)
    resolved["log_loss"] = -(
        resolved["canonical_outcome"] * np.log(resolved["conditional_probability"])
        + (1.0 - resolved["canonical_outcome"])
        * np.log(1.0 - resolved["conditional_probability"])
    )
    event_dates = (
        logical[["mapped_event_key", "event_start_date"]]
        .drop_duplicates()
        .rename(columns={"mapped_event_key": "event_key"})
    )
    if event_dates.duplicated("event_key").any():
        raise ValueError("proper-score event/date mapping is ambiguous")
    rows: list[pd.DataFrame] = []
    for market in MARKETS:
        base = resolved.loc[
            resolved["candidate"].eq(baseline_candidate)
            & resolved["market"].eq(market),
            ["mapped_event_key", "pair_key", "canonical_outcome", "log_loss"],
        ].rename(columns={"log_loss": "baseline_log_loss"})
        for candidate in sorted(
            set(resolved.loc[resolved["market"].eq(market), "candidate"])
            - {baseline_candidate}
        ):
            challenger = resolved.loc[
                resolved["candidate"].eq(candidate)
                & resolved["market"].eq(market),
                ["mapped_event_key", "pair_key", "canonical_outcome", "log_loss"],
            ].rename(
                columns={
                    "canonical_outcome": "candidate_outcome",
                    "log_loss": "candidate_log_loss",
                }
            )
            common = base.merge(
                challenger,
                on=["mapped_event_key", "pair_key"],
                how="inner",
                validate="one_to_one",
            )
            if common.empty:
                continue
            if not np.allclose(
                common["canonical_outcome"], common["candidate_outcome"]
            ):
                raise ValueError("proper-score outcomes disagree on common pairs")
            event = (
                common.assign(
                    delta=common["candidate_log_loss"]
                    - common["baseline_log_loss"]
                )
                .groupby("mapped_event_key", observed=True)["delta"]
                .mean()
                .rename("event_delta")
                .reset_index()
            )
            event = event.rename(columns={"mapped_event_key": "event_key"}).merge(
                event_dates, on="event_key", how="left", validate="one_to_one"
            )
            event["candidate"] = candidate
            event["baseline"] = baseline_candidate
            event["market"] = market
            event["metric"] = "event_equal_log_loss_delta"
            rows.append(event)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_weather_profiles(
    selection: Mapping[str, Any],
    round_panel_path: Path,
    event_crosswalk_path: Path = DEFAULT_EVENT_CROSSWALK,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build post-event weather attribution without outcome-selected flags."""

    source_manifest_path = _verified_selection_input(
        selection, "candidate_manifest"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    round_entry = source_manifest.get("inputs", {}).get("round_panel")
    if not isinstance(round_entry, Mapping):
        raise ValueError("candidate source manifest lacks its round-panel input")
    declared_round_path = Path(str(round_entry.get("path", ""))).resolve()
    if declared_round_path != round_panel_path.resolve():
        raise ValueError("weather attribution panel differs from candidate source panel")
    if not round_panel_path.exists() or _sha256(round_panel_path) != str(
        round_entry.get("sha256")
    ):
        raise ValueError("weather attribution round-panel hash mismatch")
    schema = set(_source_columns(round_panel_path))
    desired = [
        "event_key",
        "year",
        "event_id",
        "event_start_date",
        "round_num",
        "weather_field_mean_wind",
        "weather_field_mean_dew",
        "weather_field_mean_temp",
        "weather_field_mean_rain",
        "weather_delta_wind",
        "weather_delta_dew",
        "weather_delta_temp",
        "weather_delta_rain",
    ]
    missing = sorted(set(desired) - schema)
    if missing:
        raise ValueError(f"round panel lacks covariate weather columns: {missing}")
    rounds = pd.read_parquet(round_panel_path, columns=desired)
    rounds["event_start_date"] = pd.to_datetime(
        rounds["event_start_date"], errors="coerce"
    ).dt.normalize()
    rounds = rounds.loc[
        rounds["event_start_date"].between(SEALED_START, SEALED_END)
    ].copy()
    for column in desired[5:]:
        rounds[column] = pd.to_numeric(rounds[column], errors="coerce")

    if not event_crosswalk_path.exists():
        raise FileNotFoundError(event_crosswalk_path)
    crosswalk = pd.read_parquet(
        event_crosswalk_path,
        columns=[
            "odds_calendar_year",
            "odds_event_id",
            "local_event_key",
            "event_identity_status",
            "event_identity_ambiguous",
            "local_event_has_multiple_odds_ids",
        ],
    )
    unsafe_identity = (
        crosswalk["event_identity_status"].ne("matched")
        | crosswalk["event_identity_ambiguous"].fillna(True).astype(bool)
        | crosswalk["local_event_has_multiple_odds_ids"].fillna(True).astype(bool)
        | crosswalk["local_event_key"].isna()
    )
    if unsafe_identity.any():
        raise ValueError("weather event crosswalk contains unsafe identity rows")
    crosswalk["canonical_event_key"] = (
        "pga:R"
        + pd.to_numeric(crosswalk["odds_calendar_year"], errors="raise")
        .astype(int)
        .astype(str)
        + pd.to_numeric(crosswalk["odds_event_id"], errors="raise")
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )
    if crosswalk.duplicated("local_event_key").any() or crosswalk.duplicated(
        "canonical_event_key"
    ).any():
        raise ValueError("weather event crosswalk keys are not one-to-one")

    def spread(series: pd.Series) -> float:
        values = series.dropna().to_numpy(float)
        return float(np.quantile(values, 0.9) - np.quantile(values, 0.1)) if len(values) else np.nan

    profiles = rounds.groupby("event_key", observed=True).agg(
        local_year=("year", "first"),
        local_event_id=("event_id", "first"),
        event_start_date=("event_start_date", "first"),
        weather_round_rows=("round_num", "size"),
        field_wind_mean=("weather_field_mean_wind", "mean"),
        field_wind_max=("weather_field_mean_wind", "max"),
        field_dew_mean=("weather_field_mean_dew", "mean"),
        field_temp_mean=("weather_field_mean_temp", "mean"),
        field_rain_mean=("weather_field_mean_rain", "mean"),
        wind_delta_p90_p10=("weather_delta_wind", spread),
        dew_delta_p90_p10=("weather_delta_dew", spread),
        temp_delta_p90_p10=("weather_delta_temp", spread),
        rain_delta_p90_p10=("weather_delta_rain", spread),
    ).reset_index().rename(columns={"event_key": "local_event_key"})
    profiles = profiles.merge(
        crosswalk[["local_event_key", "canonical_event_key"]],
        on="local_event_key",
        how="left",
        validate="one_to_one",
    )
    # The round panel contains PGA events for which no archived matchup board
    # exists.  They are not scoring inputs and therefore need no odds identity.
    # Retain only audited matchup-event mappings here; attach_weather_to_bets()
    # independently fails closed if any quoted event is absent from this set.
    unmapped_panel_profiles = int(profiles["canonical_event_key"].isna().sum())
    profiles = profiles.loc[profiles["canonical_event_key"].notna()].copy()
    if profiles.empty:
        raise ValueError("weather profiles contain no audited matchup-event identities")
    profiles["weather_wind_bucket"] = pd.cut(
        profiles["field_wind_mean"],
        bins=[-np.inf, 8.0, 12.0, 16.0, np.inf],
        labels=["<8", "8-12", "12-16", ">=16"],
        right=False,
    ).astype("string").fillna("missing")
    audit = {
        "round_panel_sha256": _sha256(round_panel_path),
        "event_crosswalk_sha256": _sha256(event_crosswalk_path),
        "weather_profiles": int(len(profiles)),
        "unquoted_panel_event_profiles_excluded": unmapped_panel_profiles,
        "uses_weather_excluded": False,
        "uses_outcomes": False,
        "wind_bucket_thresholds": [8.0, 12.0, 16.0],
    }
    return profiles, audit


def _price_view(scored: pd.DataFrame, price_basis: str) -> pd.DataFrame:
    if price_basis not in PRICE_BASES:
        raise ValueError(f"unknown price basis: {price_basis}")
    output = scored.copy()
    output["p1_reference_close"] = pd.to_numeric(output["p1_close"], errors="coerce")
    output["p2_reference_close"] = pd.to_numeric(output["p2_close"], errors="coerce")
    if price_basis == "open":
        output["p1_close"] = pd.to_numeric(output["p1_open"], errors="coerce")
        output["p2_close"] = pd.to_numeric(output["p2_open"], errors="coerce")
    else:
        output["p1_close"] = pd.to_numeric(output["p1_close"], errors="coerce")
        output["p2_close"] = pd.to_numeric(output["p2_close"], errors="coerce")
    valid = valid_american_odds_mask(output["p1_close"]) & (
        valid_american_odds_mask(output["p2_close"])
    )
    if price_basis == "open":
        valid &= valid_american_odds_mask(output["p1_reference_close"]) & (
            valid_american_odds_mask(output["p2_reference_close"])
        )
    return output.loc[valid].copy()


def _selection_price_lookup(price_view: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side, opponent in (("p1", "p2"), ("p2", "p1")):
        piece = price_view[
            [
                "candidate",
                "quote_id",
                f"{side}_close",
                f"{opponent}_close",
                f"{side}_reference_close",
                f"{opponent}_reference_close",
            ]
        ].copy()
        piece["selection_key"] = (
            piece["candidate"].astype(str)
            + "|"
            + piece["quote_id"].astype(str)
            + f"|{side}"
        )
        piece = piece.rename(
            columns={
                f"{side}_close": "execution_selection_odds",
                f"{opponent}_close": "execution_opponent_odds",
                f"{side}_reference_close": "same_book_close_selection_odds",
                f"{opponent}_reference_close": "same_book_close_opponent_odds",
            }
        )
        pieces.append(
            piece[
                [
                    "selection_key",
                    "execution_selection_odds",
                    "execution_opponent_odds",
                    "same_book_close_selection_odds",
                    "same_book_close_opponent_odds",
                ]
            ]
        )
    lookup = pd.concat(pieces, ignore_index=True)
    if lookup.duplicated("selection_key").any():
        raise ValueError("same-book price lookup keys are not unique")
    return lookup


def select_bet_slice(
    scored: pd.DataFrame,
    *,
    price_basis: str,
    line_policy: str,
    sizing: str,
    reliability_policy: str,
    min_ev: float = market_eval.EV_GATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Settle one frozen portfolio slice and explicitly apply reliability."""

    if reliability_policy not in RELIABILITY_POLICIES:
        raise ValueError(f"unknown reliability policy: {reliability_policy}")
    view = _price_view(scored, price_basis)
    if view.empty:
        return pd.DataFrame(), pd.DataFrame()
    bets = market_eval.select_and_settle_bets(
        view, line_policy, sizing, min_ev=min_ev
    )
    if bets.empty:
        return bets, bets.copy()
    bets = bets.merge(
        _selection_price_lookup(view),
        on="selection_key",
        how="left",
        validate="one_to_one",
    )
    execution_selection_implied = np.asarray(
        american_implied_probability(bets["execution_selection_odds"]), dtype=float
    )
    execution_opponent_implied = np.asarray(
        american_implied_probability(bets["execution_opponent_odds"]), dtype=float
    )
    close_selection_implied = np.asarray(
        american_implied_probability(bets["same_book_close_selection_odds"]),
        dtype=float,
    )
    close_opponent_implied = np.asarray(
        american_implied_probability(bets["same_book_close_opponent_odds"]),
        dtype=float,
    )
    execution_novig = execution_selection_implied / (
        execution_selection_implied + execution_opponent_implied
    )
    close_novig = close_selection_implied / (
        close_selection_implied + close_opponent_implied
    )
    bets["raw_implied_open_to_close_move_pp"] = (
        close_selection_implied - execution_selection_implied
    )
    bets["open_to_close_novig_clv_pp"] = close_novig - execution_novig
    bets["open_to_close_clv_pp"] = bets["open_to_close_novig_clv_pp"]
    bets["price_basis"] = price_basis
    bets["reliability_policy"] = reliability_policy
    suppress = pd.Series(False, index=bets.index)
    if reliability_policy == "suppress_unreliable_ge10pct":
        if "pair_reliability_ok" not in bets:
            raise ValueError("reliability policy requires pair_reliability_ok")
        suppress = (
            bets["ev_per_unit"].ge(0.10)
            & ~bets["pair_reliability_ok"].fillna(False).astype(bool)
        )
    suppressed = bets.loc[suppress].copy()
    if not suppressed.empty:
        suppressed["suppression_reason"] = (
            "nominal_ev_ge_10pct_and_pair_reliability_false"
        )
    kept = bets.loc[~suppress].copy()
    if not kept.empty and "pair_reliability_ok" not in kept:
        raise AssertionError("selected bets lost pair reliability provenance")
    return kept, suppressed


def _player_attribute_lookup(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "event_key",
        "dg_id_key",
        "prediction",
        *[
            column
            for column in (
                "reliability_ok",
                "cross_tour_fallback_reason",
                "history_rounds",
                "dg_history_rounds",
                "days_since_last_round",
                "prior_event_withdrawal",
                "img_history_rounds",
                "img_history_coverage",
                "days_since_last_img_round",
            )
            if column in predictions
        ],
    ]
    output = predictions[columns].drop_duplicates(
        ["candidate", "event_key", "dg_id_key"]
    )
    return output


def _outcome_lookup(scored: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side, opponent in (("p1", "p2"), ("p2", "p1")):
        piece = scored[
            [
                "candidate",
                "quote_id",
                f"{side}_outcome_text",
                f"{opponent}_outcome_text",
            ]
        ].copy()
        piece["selection_key"] = (
            piece["candidate"].astype(str)
            + "|"
            + piece["quote_id"].astype(str)
            + f"|{side}"
        )
        piece = piece.rename(
            columns={
                f"{side}_outcome_text": "selection_outcome_text",
                f"{opponent}_outcome_text": "opponent_outcome_text",
            }
        )
        pieces.append(
            piece[
                ["selection_key", "selection_outcome_text", "opponent_outcome_text"]
            ]
        )
    output = pd.concat(pieces, ignore_index=True).drop_duplicates("selection_key")
    return output


def _withdrawal_text(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.lower()
    return normalized.str.contains(r"\b(?:wd|withdraw|withdrawn)\b", regex=True)


def enrich_bets(
    bets: pd.DataFrame,
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    if bets.empty:
        return bets.copy()
    output = bets.merge(
        _outcome_lookup(scored), on="selection_key", how="left", validate="many_to_one"
    )
    attrs = _player_attribute_lookup(predictions)
    selection = attrs.rename(
        columns={
            "dg_id_key": "selection_id",
            **{
                column: f"selection_{column}"
                for column in attrs.columns
                if column not in {"candidate", "event_key", "dg_id_key"}
            },
        }
    )
    opponent = attrs.rename(
        columns={
            "dg_id_key": "opponent_id",
            **{
                column: f"opponent_{column}"
                for column in attrs.columns
                if column not in {"candidate", "event_key", "dg_id_key"}
            },
        }
    )
    output = output.merge(
        selection,
        on=["candidate", "event_key", "selection_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        opponent,
        on=["candidate", "event_key", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    baseline_columns = [
        column for column in attrs.columns if column != "candidate"
    ]
    baseline_raw = attrs.loc[
        attrs["candidate"].eq(STRICT_PT_BASELINE), baseline_columns
    ].copy()
    baseline = baseline_raw.rename(
        columns={
            "dg_id_key": "selection_id",
            **{
                column: f"baseline_selection_{column}"
                for column in baseline_raw.columns
                if column not in {"event_key", "dg_id_key"}
            },
        }
    )
    output = output.merge(
        baseline,
        on=["event_key", "selection_id"],
        how="left",
        validate="many_to_one",
    )
    baseline_opponent = baseline_raw.rename(
        columns={
            "dg_id_key": "opponent_id",
            **{
                column: f"baseline_opponent_{column}"
                for column in baseline_raw.columns
                if column not in {"event_key", "dg_id_key"}
            },
        }
    )
    output = output.merge(
        baseline_opponent,
        on=["event_key", "opponent_id"],
        how="left",
        validate="many_to_one",
    )
    output["baseline_skill_bucket"] = pd.cut(
        output["baseline_selection_prediction"],
        bins=SKILL_BINS,
        labels=SKILL_LABELS,
        right=False,
    ).astype("string").fillna("missing")
    output["baseline_skill_gap"] = (
        output["baseline_selection_prediction"]
        - output["baseline_opponent_prediction"]
    )
    output["baseline_skill_gap_bucket"] = pd.cut(
        output["baseline_skill_gap"],
        bins=SKILL_GAP_BINS,
        labels=SKILL_GAP_LABELS,
        right=False,
    ).astype("string").fillna("missing")
    history = pd.to_numeric(
        output.get(
            "baseline_selection_history_rounds",
            output.get(
                "baseline_selection_dg_history_rounds",
                pd.Series(np.nan, index=output.index),
            ),
        ),
        errors="coerce",
    )
    output["baseline_selection_history_bucket"] = pd.cut(
        history,
        bins=HISTORY_BINS,
        labels=HISTORY_LABELS,
        right=False,
    ).astype("string").fillna("missing")
    img_coverage = pd.to_numeric(
        output.get(
            "baseline_selection_img_history_coverage",
            pd.Series(np.nan, index=output.index),
        ),
        errors="coerce",
    )
    output["baseline_selection_img_coverage_bucket"] = np.select(
        [
            img_coverage.isna(),
            img_coverage.lt(0.0),
            img_coverage.eq(0.0),
            img_coverage.lt(0.25),
            img_coverage.lt(0.50),
            img_coverage.lt(0.75),
            img_coverage.le(1.0),
        ],
        [
            "missing",
            "invalid_negative",
            "0",
            "0-25%",
            "25-50%",
            "50-75%",
            "75-100%",
        ],
        default="invalid_above_100%",
    )
    output["baseline_selection_cross_tour_fallback_reason"] = output.get(
        "baseline_selection_cross_tour_fallback_reason",
        pd.Series("missing", index=output.index),
    ).fillna("missing").astype(str)
    selection_wd = _withdrawal_text(output["selection_outcome_text"])
    opponent_wd = _withdrawal_text(output["opponent_outcome_text"])
    output["current_withdrawal_state"] = np.select(
        [selection_wd & opponent_wd, selection_wd, opponent_wd],
        ["both_withdrew", "selection_withdrew", "opponent_withdrew"],
        default="neither",
    )
    selection_prior = output.get(
        "selection_prior_event_withdrawal", pd.Series(False, index=output.index)
    ).fillna(False).astype(bool)
    opponent_prior = output.get(
        "opponent_prior_event_withdrawal", pd.Series(False, index=output.index)
    ).fillna(False).astype(bool)
    output["prior_withdrawal_state"] = np.select(
        [selection_prior & opponent_prior, selection_prior, opponent_prior],
        ["both_prior_wd", "selection_prior_wd", "opponent_prior_wd"],
        default="neither",
    )
    output = attach_weather_to_bets(output, weather)
    output["weather_wind_bucket"] = output["weather_wind_bucket"].fillna("missing")
    return output


def attach_weather_to_bets(
    bets: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Join either local or canonical bet-event identity and fail on misses."""

    required = {"local_event_key", "canonical_event_key", "weather_round_rows"}
    missing_columns = sorted(required - set(weather.columns))
    if missing_columns:
        raise ValueError(f"weather profiles lack join columns: {missing_columns}")
    if weather[["local_event_key", "canonical_event_key"]].isna().any().any():
        raise ValueError("weather event identity contains null join keys")
    if weather.duplicated("local_event_key", keep=False).any() or weather.duplicated(
        "canonical_event_key", keep=False
    ).any():
        raise ValueError("weather event identity is ambiguous before join-key expansion")
    local = weather.copy()
    local["weather_join_key"] = local["local_event_key"].astype(str)
    canonical = weather.copy()
    canonical["weather_join_key"] = canonical["canonical_event_key"].astype(str)
    lookup = pd.concat([local, canonical], ignore_index=True)
    if lookup.duplicated("weather_join_key", keep=False).any():
        raise ValueError("weather join keys are ambiguous")
    output = bets.merge(
        lookup.drop(columns="event_start_date", errors="ignore"),
        left_on="event_key",
        right_on="weather_join_key",
        how="left",
        validate="many_to_one",
    ).drop(columns="weather_join_key")
    missing = output["weather_round_rows"].isna()
    if missing.any():
        examples = sorted(output.loc[missing, "event_key"].astype(str).unique())[:10]
        raise ValueError(
            f"weather attribution failed for {int(missing.sum())} bet rows: {examples}"
        )
    return output


def summarize_portfolio(
    bets: pd.DataFrame,
    *,
    expected_candidates: Sequence[str],
    expected_markets: Sequence[str],
    comparison_candidate: str,
    opportunity_mode: str,
    price_basis: str,
    line_policy: str,
    sizing: str,
    reliability_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    event_rows: list[pd.DataFrame] = []
    bad_player_rows: list[pd.DataFrame] = []
    for candidate in expected_candidates:
        candidate_frame = bets.loc[bets["candidate"].eq(candidate)] if not bets.empty else bets
        for market in expected_markets:
            group = (
                candidate_frame.loc[candidate_frame["round_market"].eq(market)].copy()
                if not candidate_frame.empty
                else candidate_frame.copy()
            )
            stake = float(group["stake"].sum()) if not group.empty else 0.0
            pnl = float(group["pnl"].sum()) if not group.empty else 0.0
            performance_rows.append(
                {
                    "candidate": candidate,
                    "comparison_candidate": comparison_candidate,
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
                    "max_drawdown": float(market_eval._max_drawdown(group)) if not group.empty else 0.0,
                    "mean_nominal_ev": float(group["ev_per_unit"].mean()) if not group.empty else np.nan,
                    "mean_open_to_close_clv_pp": float(group["open_to_close_clv_pp"].mean()) if not group.empty else np.nan,
                    "pushes": int(group["settlement_status"].eq("push").sum()) if not group.empty else 0,
                }
            )
            if group.empty:
                continue
            event = group.groupby(
                ["event_key", "event_start_date"], observed=True
            ).agg(pnl=("pnl", "sum"), stake=("stake", "sum"), bets=("pnl", "size")).reset_index()
            event["candidate"] = candidate
            event["comparison_candidate"] = comparison_candidate
            event["market"] = market
            event["opportunity_mode"] = opportunity_mode
            event["price_basis"] = price_basis
            event["line_policy"] = line_policy
            event["sizing"] = sizing
            event["reliability_policy"] = reliability_policy
            event_rows.append(event)
            dimensions = {
                "year": group["event_start_date"].dt.year.astype(str),
                "book": group["book"].astype(str),
                "edge": pd.cut(
                    group["ev_per_unit"], EDGE_BINS, labels=EDGE_LABELS, right=False
                ).astype(str),
                "skill": group["baseline_skill_bucket"].astype(str),
                "skill_gap": group["baseline_skill_gap_bucket"].astype(str),
                "selection_history": group[
                    "baseline_selection_history_bucket"
                ].astype(str),
                "img_history_coverage": group[
                    "baseline_selection_img_coverage_bucket"
                ].astype(str),
                "cross_tour_fallback": group[
                    "baseline_selection_cross_tour_fallback_reason"
                ].astype(str),
                "reliability": group["pair_reliability_ok"].fillna(False).astype(str),
                "current_wd": group["current_withdrawal_state"].astype(str),
                "prior_wd": group["prior_withdrawal_state"].astype(str),
                "weather_wind": group["weather_wind_bucket"].astype(str),
            }
            for breakdown_type, buckets in dimensions.items():
                local = group.assign(_bucket=buckets)
                for bucket, bucket_group in local.groupby("_bucket", observed=True):
                    bucket_stake = float(bucket_group["stake"].sum())
                    bucket_pnl = float(bucket_group["pnl"].sum())
                    breakdown_rows.append(
                        {
                            "candidate": candidate,
                            "comparison_candidate": comparison_candidate,
                            "market": market,
                            "opportunity_mode": opportunity_mode,
                            "price_basis": price_basis,
                            "line_policy": line_policy,
                            "sizing": sizing,
                            "reliability_policy": reliability_policy,
                            "breakdown_type": breakdown_type,
                            "bucket": str(bucket),
                            "events": int(bucket_group["event_key"].nunique()),
                            "bets": int(len(bucket_group)),
                            "stake": bucket_stake,
                            "pnl": bucket_pnl,
                            "roi": bucket_pnl / bucket_stake if bucket_stake else np.nan,
                        }
                    )
            worst_events = event.nsmallest(20, "pnl")["event_key"]
            bad = group.loc[group["event_key"].isin(worst_events)].groupby(
                ["event_key", "event_start_date", "selection_id", "selection_name"],
                observed=True,
                dropna=False,
            ).agg(pnl=("pnl", "sum"), stake=("stake", "sum"), bets=("pnl", "size")).reset_index()
            bad["candidate"] = candidate
            bad["comparison_candidate"] = comparison_candidate
            bad["market"] = market
            bad["opportunity_mode"] = opportunity_mode
            bad["price_basis"] = price_basis
            bad["line_policy"] = line_policy
            bad["sizing"] = sizing
            bad["reliability_policy"] = reliability_policy
            bad_player_rows.append(bad)
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(breakdown_rows),
        pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame(),
        pd.concat(bad_player_rows, ignore_index=True) if bad_player_rows else pd.DataFrame(),
    )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def materialize_frozen_portfolios(
    scored: pd.DataFrame,
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    candidate_names: Sequence[str],
    baseline_candidate: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write bounded candidate partitions; retain only aggregated frames."""

    bet_dir = output_dir / "bet_partitions"
    bet_dir.mkdir(parents=True, exist_ok=True)
    performance_parts: list[pd.DataFrame] = []
    breakdown_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    bad_player_parts: list[pd.DataFrame] = []
    suppression_rows: list[dict[str, Any]] = []

    def one_partition(
        partition_scored: pd.DataFrame,
        expected_candidates: Sequence[str],
        comparison_candidate: str,
        opportunity_mode: str,
        path: Path,
    ) -> None:
        bet_parts: list[pd.DataFrame] = []
        for price_basis in PRICE_BASES:
            for line_policy in LINE_POLICIES:
                for sizing in SIZINGS:
                    for reliability_policy in RELIABILITY_POLICIES:
                        bets, suppressed = select_bet_slice(
                            partition_scored,
                            price_basis=price_basis,
                            line_policy=line_policy,
                            sizing=sizing,
                            reliability_policy=reliability_policy,
                        )
                        if not suppressed.empty:
                            for (candidate, market), count in suppressed.groupby(
                                ["candidate", "round_market"], observed=True
                            ).size().items():
                                suppression_rows.append(
                                    {
                                        "candidate": candidate,
                                        "comparison_candidate": comparison_candidate,
                                        "market": market,
                                        "opportunity_mode": opportunity_mode,
                                        "price_basis": price_basis,
                                        "line_policy": line_policy,
                                        "sizing": sizing,
                                        "reliability_policy": reliability_policy,
                                        "suppressed_bets": int(count),
                                        "suppression_reason": (
                                            "nominal_ev_ge_10pct_and_pair_reliability_false"
                                        ),
                                    }
                                )
                        if not bets.empty:
                            pushes = bets["settlement_status"].eq("push")
                            if pushes.any() and not np.allclose(
                                bets.loc[pushes, "pnl"], 0.0, atol=0.0
                            ):
                                raise ValueError("push settlement produced nonzero P&L")
                            bets = enrich_bets(
                                bets, partition_scored, predictions, weather
                            )
                            bets["comparison_candidate"] = comparison_candidate
                            bets["opportunity_mode"] = opportunity_mode
                            bet_parts.append(bets)
                        summaries = summarize_portfolio(
                            bets,
                            expected_candidates=expected_candidates,
                            expected_markets=MARKETS,
                            comparison_candidate=comparison_candidate,
                            opportunity_mode=opportunity_mode,
                            price_basis=price_basis,
                            line_policy=line_policy,
                            sizing=sizing,
                            reliability_policy=reliability_policy,
                        )
                        for collection, frame in zip(
                            (
                                performance_parts,
                                breakdown_parts,
                                event_parts,
                                bad_player_parts,
                            ),
                            summaries,
                        ):
                            if not frame.empty:
                                collection.append(frame)
        partition_bets = (
            pd.concat(bet_parts, ignore_index=True)
            if bet_parts
            else pd.DataFrame()
        )
        _atomic_parquet(partition_bets, path)

    # All-available: one compact partition per candidate, no repeated baseline.
    for candidate in candidate_names:
        candidate_scored = scored.loc[scored["candidate"].eq(candidate)].copy()
        one_partition(
            candidate_scored,
            [candidate],
            candidate,
            "all_available",
            bet_dir / f"all_available__{_safe_filename(candidate)}.parquet",
        )

    # Pairwise common menus: only the baseline and one challenger at a time.
    baseline_quotes = set(
        scored.loc[scored["candidate"].eq(baseline_candidate), "quote_id"]
    )
    for candidate in candidate_names:
        if candidate == baseline_candidate:
            continue
        candidate_quotes = set(
            scored.loc[scored["candidate"].eq(candidate), "quote_id"]
        )
        common_quotes = baseline_quotes & candidate_quotes
        common_scored = scored.loc[
            scored["candidate"].isin([baseline_candidate, candidate])
            & scored["quote_id"].isin(common_quotes)
        ].copy()
        one_partition(
            common_scored,
            [baseline_candidate, candidate],
            candidate,
            "pairwise_common_with_strict_pt_baseline",
            bet_dir / f"pairwise_common__{_safe_filename(candidate)}.parquet",
        )

    def concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return (
        concat(performance_parts),
        concat(breakdown_parts),
        concat(event_parts),
        concat(bad_player_parts),
        pd.DataFrame(suppression_rows),
    )


def build_money_event_deltas(
    event_performance: pd.DataFrame,
    quotes: pd.DataFrame,
    candidate_names: Sequence[str],
    baseline_candidate: str,
) -> pd.DataFrame:
    if event_performance.empty:
        return pd.DataFrame()
    event_catalog = (
        quotes[["mapped_event_key", "event_start_date", "market"]]
        .drop_duplicates()
        .rename(columns={"mapped_event_key": "event_key"})
    )
    rows: list[pd.DataFrame] = []
    slice_columns = [
        "market",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
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
            slice_values = (
                (market, price_basis, line_policy, sizing, reliability_policy)
                for market in MARKETS
                for price_basis in PRICE_BASES
                for line_policy in LINE_POLICIES
                for sizing in SIZINGS
                for reliability_policy in RELIABILITY_POLICIES
            )
            for values in slice_values:
                filters = dict(zip(slice_columns, values))
                market = str(filters["market"])
                events = event_catalog.loc[event_catalog["market"].eq(market)].copy()
                challenger = challenger_rows.copy()
                baseline = baseline_rows.copy()
                for column, value in filters.items():
                    challenger = challenger.loc[challenger[column].eq(value)]
                    baseline = baseline.loc[baseline[column].eq(value)]
                challenger = challenger[["event_key", "pnl"]].rename(
                    columns={"pnl": "candidate_pnl"}
                )
                baseline = baseline[["event_key", "pnl"]].rename(
                    columns={"pnl": "baseline_pnl"}
                )
                merged = events.merge(
                    baseline, on="event_key", how="left", validate="one_to_one"
                ).merge(
                    challenger, on="event_key", how="left", validate="one_to_one"
                )
                merged[["baseline_pnl", "candidate_pnl"]] = merged[
                    ["baseline_pnl", "candidate_pnl"]
                ].fillna(0.0)
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
                rows.append(merged)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not output.empty:
        hypothesis = [
            "candidate",
            "opportunity_mode",
            "market",
            "price_basis",
            "line_policy",
            "sizing",
            "reliability_policy",
        ]
        if output.duplicated([*hypothesis, "event_key"]).any():
            raise ValueError("money event-delta keys are not unique")
    return output


def event_cluster_bootstrap(
    event_deltas: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    better_when: str,
) -> pd.DataFrame:
    if draws < 10_000:
        raise ValueError("sealed event bootstrap requires at least 10,000 draws")
    if event_deltas.empty:
        return pd.DataFrame()
    if better_when not in {"positive", "negative"}:
        raise ValueError("better_when must be positive or negative")
    metadata = [
        column
        for column in (
            "metric",
            "candidate",
            "baseline",
            "opportunity_mode",
            "market",
            "price_basis",
            "line_policy",
            "sizing",
            "reliability_policy",
        )
        if column in event_deltas
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for keys, group in event_deltas.groupby(metadata, observed=True, dropna=False):
        values = group.sort_values(["event_start_date", "event_key"])[
            "event_delta"
        ].to_numpy(float)
        if not len(values):
            continue
        samples = np.empty(draws, dtype=float)
        batch_size = 500
        for start in range(0, draws, batch_size):
            stop = min(start + batch_size, draws)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            if str(group["metric"].iloc[0]) == "pnl_delta":
                samples[start:stop] = values[indices].sum(axis=1)
            else:
                samples[start:stop] = values[indices].mean(axis=1)
        estimate = float(values.sum()) if str(group["metric"].iloc[0]) == "pnl_delta" else float(values.mean())
        row = dict(zip(metadata, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "events": int(len(values)),
                "zero_bet_events": int(group.get("zero_bet_event", pd.Series(False, index=group.index)).sum()),
                "estimate": estimate,
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "probability_better": float(
                    np.mean(samples > 0.0)
                    if better_when == "positive"
                    else np.mean(samples < 0.0)
                ),
                "draws": int(draws),
                "cluster": "whole_event",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def block_max_stat_reality_check(
    money_event_deltas: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    block_sizes: Sequence[int] = (2, 3, 4),
) -> pd.DataFrame:
    """Studentized moving-block max-stat check across frozen money slices."""

    if draws < 10_000:
        raise ValueError("sealed max-stat check requires at least 10,000 draws")
    if money_event_deltas.empty:
        return pd.DataFrame()
    hypothesis_columns = [
        "candidate",
        "opportunity_mode",
        "market",
        "price_basis",
        "line_policy",
        "sizing",
        "reliability_policy",
    ]
    work = money_event_deltas.copy()
    work["hypothesis_id"] = work[hypothesis_columns].astype(str).agg("|".join, axis=1)
    metadata = work[["hypothesis_id", *hypothesis_columns]].drop_duplicates(
        "hypothesis_id"
    )
    rows: list[dict[str, Any]] = []
    for market in MARKETS:
        market_work = work.loc[work["market"].eq(market)].copy()
        if market_work.empty:
            continue
        event_order = (
            market_work[["event_key", "event_start_date"]]
            .drop_duplicates()
            .sort_values(["event_start_date", "event_key"])["event_key"]
            .tolist()
        )
        matrix = market_work.pivot(
            index="event_key", columns="hypothesis_id", values="event_delta"
        ).reindex(event_order).fillna(0.0)
        values = matrix.to_numpy(float)
        observed_mean = values.mean(axis=0)
        standard_error = values.std(axis=0, ddof=1) / math.sqrt(len(values))
        valid_scale = np.where(standard_error > 1e-12, standard_error, np.inf)
        observed_t = observed_mean / valid_scale
        centered = values - observed_mean
        for block_size in block_sizes:
            if block_size < 1 or block_size > len(values):
                raise ValueError(f"invalid moving block size {block_size}")
            rng = np.random.default_rng(seed + int(block_size) * 1009)
            max_stats = np.empty(draws, dtype=float)
            batch_size = 100
            block_count = math.ceil(len(values) / block_size)
            offsets = np.arange(block_size)
            for start in range(0, draws, batch_size):
                stop = min(start + batch_size, draws)
                starts = rng.integers(
                    0, len(values), size=(stop - start, block_count)
                )
                indices = (
                    starts[..., None] + offsets[None, None, :]
                ) % len(values)
                indices = indices.reshape(stop - start, -1)[:, : len(values)]
                sample_means = centered[indices].mean(axis=1)
                sample_t = sample_means / valid_scale
                max_stats[start:stop] = np.max(sample_t, axis=1)
            for index, hypothesis_id in enumerate(matrix.columns):
                meta = metadata.loc[
                    metadata["hypothesis_id"].eq(hypothesis_id)
                ].iloc[0]
                rows.append(
                    {
                        **{
                            column: meta[column] for column in hypothesis_columns
                        },
                        "block_size_events": int(block_size),
                        "events": int(len(values)),
                        "hypotheses_in_family": int(values.shape[1]),
                        "observed_mean_event_delta": float(observed_mean[index]),
                        "observed_studentized_stat": float(observed_t[index]),
                        "max_stat_adjusted_p": float(
                            np.mean(max_stats >= observed_t[index])
                        ),
                        "max_stat_critical_95": float(
                            np.quantile(max_stats, 0.95)
                        ),
                        "draws": int(draws),
                        "null_centered": True,
                    }
                )
    return pd.DataFrame(rows)


def _candidate_metadata(
    catalog: pd.DataFrame, names: Sequence[str], roles: Mapping[str, str]
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "name",
            "family",
            "source",
            "weather_arm",
            "target_arm",
            "refit_schedule",
            "alpha",
            "short_game_penalty",
            "control",
            "sensitivity_only",
            "selection_eligible",
        )
        if column in catalog
    ]
    metadata = catalog.loc[catalog["name"].astype(str).isin(names), metadata_columns].copy()
    missing = sorted(set(names) - set(metadata["name"].astype(str)))
    if missing:
        raise ValueError(f"sealed candidate names absent from hashed catalog: {missing}")
    metadata = metadata.rename(columns={"name": "candidate"})
    metadata["candidate_role"] = metadata["candidate"].map(roles)
    metadata["sealed_selection_eligible"] = False
    metadata["sealed_result_can_reselect"] = False
    return metadata


def append_sensitivity_metadata(
    metadata: pd.DataFrame,
    *,
    source_candidate: str,
    sensitivity_candidate: str,
    role: str,
) -> pd.DataFrame:
    """Clone frozen source metadata while categorically barring reselection."""

    source = metadata.loc[metadata["candidate"].eq(source_candidate)].copy()
    if len(source) != 1:
        raise ValueError(
            "sensitivity metadata requires one unique frozen source candidate"
        )
    source["source_candidate"] = source_candidate
    source["candidate"] = sensitivity_candidate
    source["candidate_role"] = role
    if "control" in source:
        source["control"] = False
    if "sensitivity_only" in source:
        source["sensitivity_only"] = True
    if "selection_eligible" in source:
        source["selection_eligible"] = False
    source["sealed_selection_eligible"] = False
    source["sealed_result_can_reselect"] = False
    source["sealed_sensitivity_type"] = "frozen_opener_market_blend"
    output = pd.concat([metadata, source], ignore_index=True, sort=False)
    if output["candidate"].duplicated().any():
        raise ValueError("candidate metadata is not unique after sensitivity append")
    return output


def _attach_candidate_metadata(
    frame: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    if frame.empty or "candidate" not in frame:
        return frame
    return frame.merge(metadata, on="candidate", how="left", validate="many_to_one")


def _reliability_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["year"] = work["event_start_date"].dt.year
    if "reliability_ok" not in work:
        work["reliability_ok"] = False
    work["reliability_ok"] = work["reliability_ok"].fillna(False).astype(bool)
    if "cross_tour_fallback_reason" not in work:
        work["cross_tour_fallback_reason"] = "unavailable"
    rows: list[dict[str, Any]] = []
    for (candidate, year, reason), group in work.groupby(
        ["candidate", "year", "cross_tour_fallback_reason"],
        observed=True,
        dropna=False,
    ):
        rows.append(
            {
                "candidate": candidate,
                "year": int(year),
                "cross_tour_fallback_reason": str(reason),
                "rows": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "players": int(group["dg_id_key"].nunique()),
                "reliable_rows": int(group["reliability_ok"].sum()),
                "reliable_rate": float(group["reliability_ok"].mean()),
                "prior_wd_rows": int(
                    group.get(
                        "prior_event_withdrawal",
                        pd.Series(False, index=group.index),
                    )
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "median_history_rounds": float(
                    pd.to_numeric(
                        group.get(
                            "history_rounds", pd.Series(np.nan, index=group.index)
                        ),
                        errors="coerce",
                    ).median()
                ),
                "median_days_since_last_round": float(
                    pd.to_numeric(
                        group.get(
                            "days_since_last_round",
                            pd.Series(np.nan, index=group.index),
                        ),
                        errors="coerce",
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _proper_by_year(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year, group in scored.groupby(scored["event_start_date"].dt.year, observed=True):
        summary = market_eval.proper_score_summary(group, "sealed_frozen")
        if not summary.empty:
            summary["year"] = int(year)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_report(
    output: Path,
    selected_name: str,
    roles: Mapping[str, str],
    performance: pd.DataFrame,
    proper: pd.DataFrame,
    bootstrap: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    selected_performance = performance.loc[
        performance["candidate"].eq(selected_name)
        & performance["opportunity_mode"].eq(
            "pairwise_common_with_strict_pt_baseline"
        )
        & performance["price_basis"].eq("close")
        & performance["line_policy"].eq("best_of_three")
        & performance["sizing"].eq("flat_1u")
        & performance["reliability_policy"].eq("ungated")
    ]
    lines = [
        "# Frozen base-skill sealed matchup evaluation",
        "",
        f"Frozen model: `{selected_name}` ({roles[selected_name]}).",
        "",
        "The 2023-26 sample did not select or retune any model or probability mapper. "
        "Weather and clean-IMG siblings were locked in the discovery manifest and are descriptive only.",
        "",
        "Closing prices were not features. Opening-price results use asynchronous archived openers and are not proof of simultaneous executability.",
        "",
        "## Primary fixed slice",
        "",
        "Pairwise-common, best of BetOnline/Betcris/Pinnacle close, flat 1u, ungated reliability sensitivity.",
        "",
    ]
    if selected_performance.empty:
        lines.append("No bets met the fixed 3% EV rule in this slice.")
    else:
        lines.append("| Market | Bets | Stake | P&L | ROI |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in selected_performance.to_dict(orient="records"):
            lines.append(
                f"| {row['market']} | {int(row['bets'])} | {row['stake']:.2f} | "
                f"{row['pnl']:+.2f} | {row['roi']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Sealed events: {audit['market_input']['events']}",
            f"- Canonical probability duplicates: {audit['canonical_probability_duplicates']}",
            f"- Bootstrap draws: {audit['bootstrap_draws']}",
            f"- Exact push P&L violations: {audit['push_pnl_violations']}",
            "- Ordinary bootstrap resamples whole events and includes zero-bet events.",
            "- The 2/3/4-event moving-block max-stat output controls descriptively for clustered bad weeks and multiple frozen slices.",
            "",
            f"Full proper-score rows: {len(proper):,}; bootstrap rows: {len(bootstrap):,}.",
        ]
    )
    (output / "SEALED_MATCHUP_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.bootstrap < 10_000:
        raise ValueError("sealed evaluation requires --bootstrap >= 10000")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"sealed output directory is not empty; use a new versioned path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    compact_manifest_path = args.finalist_dir.resolve() / "manifest.json"
    expected_selection = args.selection_manifest.resolve()
    # Validate the entire shared chain before enumerating execution inputs.  A
    # second identical validation immediately after begin() closes the gap in
    # which a manifest could otherwise redirect the evaluator to valid but
    # untracked bytes.
    pre_handoff = validate_frozen_handoff_chain(compact_manifest_path)
    if pre_handoff.selection_manifest_path != expected_selection:
        raise ValueError("compact handoff points to a different selection manifest")
    pre_authority = validate_sealed_panel_authorities(
        pre_handoff,
        args.round_panel.resolve(),
        args.event_crosswalk.resolve(),
    )
    pre_handoff_paths = tuple(pre_handoff.verified_paths)
    pre_authority_paths = tuple(pre_authority["verified_paths"])
    provenance_inputs = [*pre_handoff_paths, *pre_authority_paths]
    provenance_state = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__),
        input_paths=provenance_inputs,
        code_paths=PROVENANCE_CODE_PATHS,
    )
    post_handoff, panel_authority_audit = revalidate_sealed_handoff_after_begin(
        compact_manifest_path,
        expected_selection,
        expected_handoff_paths=pre_handoff_paths,
        expected_authority_paths=pre_authority_paths,
        provenance_state=provenance_state,
        round_panel_path=args.round_panel.resolve(),
        event_crosswalk_path=args.event_crosswalk.resolve(),
    )
    compact, selection, compact_predictions_path, selection_path, selected_name = (
        _handoff_tuple(post_handoff)
    )
    predictions, _, names, roles, prediction_audit = load_frozen_predictions(
        compact,
        compact_predictions_path,
        selection,
        selected_name,
    )
    catalog, quotes, market_audit = load_catalog_and_quotes(
        selection, args.candidate_catalog.resolve(), args.evaluation_cache.resolve()
    )
    metadata = _candidate_metadata(catalog, names, roles)
    mapper = frozen_probability_mapper(selection, names)
    opener_choice = frozen_opener_blend_choice(selection, selected_name)
    weather, weather_audit = load_weather_profiles(
        selection,
        args.round_panel.resolve(),
        args.event_crosswalk.resolve(),
    )
    quoted_events = quotes[["mapped_event_key"]].drop_duplicates().rename(
        columns={"mapped_event_key": "event_key"}
    )
    quoted_weather = attach_weather_to_bets(quoted_events, weather)
    weather_audit["quoted_events"] = int(len(quoted_events))
    weather_audit["quoted_events_with_profiles"] = int(len(quoted_weather))
    weather_audit["quoted_event_profile_coverage"] = float(
        len(quoted_weather) / len(quoted_events)
    )
    coverage, coverage_missing = quote_coverage_audit(quotes, predictions, names)
    enforce_primary_quote_coverage(coverage, selected_name)

    preflight = {
        "contract": OUTPUT_CONTRACT,
        "complete": False,
        "preflight_complete": True,
        "research_only": True,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "compact_manifest": str(compact_manifest_path),
        "compact_manifest_sha256": _sha256(compact_manifest_path),
        "compact_predictions": str(compact_predictions_path),
        "compact_predictions_sha256": _sha256(compact_predictions_path),
        "selected_model": selected_name,
        "selected_model_role": roles[selected_name],
        "candidate_names": names,
        "frozen_opener_blend_choice": opener_choice,
        "sealed_selection_performed": False,
        "prediction_audit": prediction_audit,
        "market_audit": market_audit,
        "weather_audit": weather_audit,
        "panel_authority": {
            key: value
            for key, value in panel_authority_audit.items()
            if key != "verified_paths"
        },
        "frozen_handoff": post_handoff.audit,
    }
    _atomic_json(output / "preflight.json", preflight)
    mapper.to_csv(output / "frozen_probability_mapper.csv", index=False)
    coverage.to_csv(output / "coverage_by_candidate_year_market.csv", index=False)
    _atomic_parquet(coverage_missing, output / "coverage_missing_quotes.parquet")
    metadata.to_csv(output / "candidate_roles_and_metadata.csv", index=False)
    weather.to_csv(output / "sealed_event_weather_profiles.csv", index=False)
    if args.preflight_only:
        execution_audit = execution_provenance.finalize(provenance_state)
        preflight["execution_provenance"] = {
            "path": "execution_provenance/execution_provenance.json",
            "start_utc": execution_audit["start_utc"],
            "end_utc": execution_audit["end_utc"],
            "pid": execution_audit["pid"],
            "verified_unchanged": True,
        }
        write_sealed_complete_manifest(
            output / "preflight.json", preflight, provenance_state
        )
        print(f"Sealed preflight complete: {output}")
        return

    scored = score_with_frozen_mapper(quotes, predictions, mapper, roles)
    (
        scored,
        predictions,
        names,
        roles,
        opener_blend_audit,
    ) = append_frozen_opener_blend_sensitivity(
        scored,
        predictions,
        selection,
        selected_name,
        roles,
    )
    opener_name = str(opener_blend_audit["candidate"])
    metadata = append_sensitivity_metadata(
        metadata,
        source_candidate=selected_name,
        sensitivity_candidate=opener_name,
        role=roles[opener_name],
    )
    coverage, coverage_missing = quote_coverage_audit(quotes, predictions, names)
    enforce_primary_quote_coverage(coverage, selected_name)
    coverage.to_csv(output / "coverage_by_candidate_year_market.csv", index=False)
    _atomic_parquet(coverage_missing, output / "coverage_missing_quotes.parquet")
    metadata.to_csv(output / "candidate_roles_and_metadata.csv", index=False)
    preflight["evaluated_candidate_names"] = names
    preflight["opener_blend_sensitivity"] = opener_blend_audit
    _atomic_json(output / "preflight.json", preflight)
    logical = market_eval._canonical_logical_pairs(scored)
    canonical_keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    canonical_duplicates = int(logical.duplicated(canonical_keys).sum())
    if canonical_duplicates:
        raise ValueError("canonical sealed probabilities are not unique")
    logical_columns = [
        column
        for column in (
            "candidate",
            "candidate_role",
            "mapped_event_key",
            "event_start_date",
            "market",
            "pair_key",
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
            "sigma",
            "tie_threshold",
            "probability_fit_end",
        )
        if column in logical
    ]
    _atomic_parquet(
        logical[logical_columns], output / "sealed_logical_probabilities.parquet"
    )
    raw_proper = market_eval.proper_score_summary(scored, "sealed_frozen")
    calibration = market_eval.calibration_summary(scored, "sealed_frozen")
    paired_proper = discovery.paired_common_proper_scores(
        logical, STRICT_PT_BASELINE
    )
    proper_year = _proper_by_year(scored)
    proper_event_deltas = paired_proper_event_deltas(logical, STRICT_PT_BASELINE)
    raw_proper = _attach_candidate_metadata(raw_proper, metadata)
    calibration = _attach_candidate_metadata(calibration, metadata)
    paired_proper = _attach_candidate_metadata(paired_proper, metadata)
    proper_year = _attach_candidate_metadata(proper_year, metadata)
    raw_proper.to_csv(output / "proper_scores.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    paired_proper.to_csv(output / "paired_common_proper_scores.csv", index=False)
    proper_year.to_csv(output / "proper_scores_by_year.csv", index=False)
    _atomic_parquet(proper_event_deltas, output / "proper_score_event_deltas.parquet")

    performance, breakdowns, event_performance, bad_players, suppressed = (
        materialize_frozen_portfolios(
            scored,
            predictions,
            weather,
            names,
            STRICT_PT_BASELINE,
            output,
        )
    )
    performance = _attach_candidate_metadata(performance, metadata)
    breakdowns = _attach_candidate_metadata(breakdowns, metadata)
    event_performance = _attach_candidate_metadata(event_performance, metadata)
    bad_players = _attach_candidate_metadata(bad_players, metadata)
    suppressed = _attach_candidate_metadata(suppressed, metadata)
    performance.to_csv(output / "betting_performance.csv", index=False)
    breakdowns.to_csv(output / "betting_breakdowns.csv", index=False)
    _atomic_parquet(event_performance, output / "event_betting_performance.parquet")
    suppressed.to_csv(output / "reliability_suppression_audit.csv", index=False)
    selected_bad = bad_players.loc[
        bad_players["candidate"].isin([selected_name, STRICT_PT_BASELINE])
    ].copy()
    selected_bad.to_csv(output / "bad_week_golfer_losses.csv", index=False)
    worst_events = (
        event_performance.sort_values("pnl", kind="mergesort")
        .groupby(
            [
                "candidate",
                "comparison_candidate",
                "market",
                "opportunity_mode",
                "price_basis",
                "line_policy",
                "sizing",
                "reliability_policy",
            ],
            observed=True,
            group_keys=False,
        )
        .head(20)
    )
    worst_events.to_csv(output / "worst_event_performance.csv", index=False)
    reliability = _attach_candidate_metadata(
        _reliability_coverage(predictions), metadata
    )
    reliability.to_csv(output / "prediction_reliability_coverage.csv", index=False)

    money_event_deltas = build_money_event_deltas(
        event_performance, quotes, names, STRICT_PT_BASELINE
    )
    _atomic_parquet(money_event_deltas, output / "money_event_deltas.parquet")
    money_bootstrap = event_cluster_bootstrap(
        money_event_deltas,
        draws=args.bootstrap,
        seed=args.seed,
        better_when="positive",
    )
    proper_bootstrap = event_cluster_bootstrap(
        proper_event_deltas,
        draws=args.bootstrap,
        seed=args.seed + 1,
        better_when="negative",
    )
    bootstrap = pd.concat(
        [money_bootstrap, proper_bootstrap], ignore_index=True, sort=False
    )
    bootstrap.to_csv(output / "paired_event_bootstrap.csv", index=False)
    reality = block_max_stat_reality_check(
        money_event_deltas,
        draws=args.bootstrap,
        seed=args.seed + 2,
        block_sizes=args.block_sizes,
    )
    reality.to_csv(output / "block_max_stat_reality_check.csv", index=False)

    push_violations = 0
    partition_count = 0
    total_bets = 0
    for path in (output / "bet_partitions").glob("*.parquet"):
        partition_count += 1
        partition = pd.read_parquet(
            path, columns=["settlement_status", "pnl"]
        )
        total_bets += len(partition)
        push_violations += int(
            (
                partition["settlement_status"].eq("push")
                & partition["pnl"].ne(0.0)
            ).sum()
        )
    if push_violations:
        raise ValueError("one or more materialized pushes have nonzero P&L")
    audit = {
        "research_only": True,
        "production_files_changed": False,
        "sealed_period": f"{SEALED_START.date()}/{SEALED_END.date()}",
        "sealed_candidate_selection_performed": False,
        "sealed_results_can_promote_or_reselect": False,
        "selected_model": selected_name,
        "selected_model_role": roles[selected_name],
        "promotion_gate_passed_on_discovery": bool(
            selection.get("promotion_gate_passed")
        ),
        "locked_diagnostic_ablation_set": selection[
            "locked_diagnostic_ablation_set"
        ],
        "closing_odds_used_as_features": False,
        "opening_price_executability": "asynchronous_sensitivity_only",
        "probability_mapper_fit_end_max": mapper["fit_end"].max().date().isoformat(),
        "probability_mapper_refit_on_sealed": False,
        "market_input": market_audit,
        "prediction_input": prediction_audit,
        "weather_attribution": weather_audit,
        "panel_authority": {
            key: value
            for key, value in panel_authority_audit.items()
            if key != "verified_paths"
        },
        "frozen_handoff": post_handoff.audit,
        "opener_blend_sensitivity": opener_blend_audit,
        "evaluated_candidate_names": names,
        "canonical_probability_rows": int(len(logical)),
        "canonical_probability_duplicates": canonical_duplicates,
        "bet_partition_files": partition_count,
        "materialized_bet_rows": int(total_bets),
        "push_pnl_violations": int(push_violations),
        "bootstrap_draws": int(args.bootstrap),
        "bootstrap_cluster": "whole_event_including_zero_bet_events",
        "block_sizes_events": list(args.block_sizes),
        "max_stat_null_centered": True,
        "fixed_ev_gate": float(market_eval.EV_GATE),
        "books": list(market_eval.BOOKS),
        "line_policies": list(LINE_POLICIES),
        "price_bases": list(PRICE_BASES),
        "sizing_rules": list(SIZINGS),
        "reliability_policies": list(RELIABILITY_POLICIES),
    }
    _atomic_json(output / "audit.json", audit)
    _write_report(
        output,
        selected_name,
        roles,
        performance,
        raw_proper,
        bootstrap,
        audit,
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

    artifacts: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "contract": OUTPUT_CONTRACT,
        "complete": True,
        "research_only": True,
        "selected_model": selected_name,
        "selected_model_role": roles[selected_name],
        "selection_manifest_sha256": _sha256(selection_path),
        "compact_manifest_sha256": _sha256(compact_manifest_path),
        "compact_prediction_sha256": _sha256(compact_predictions_path),
        "source_prediction_sha256": compact["source_prediction_sha256"],
        "root_panel_manifest_sha256": panel_authority_audit[
            "root_panel_manifest_sha256"
        ],
        "round_panel_sha256": panel_authority_audit["round_panel_sha256"],
        "event_crosswalk_sha256": panel_authority_audit[
            "event_crosswalk_sha256"
        ],
        "selection_window_end": _selection_cutoff(selection).date().isoformat(),
        "sealed_data_used_for_selection": False,
        "sealed_results_used_for_selection": False,
        "frozen_before_sealed_scoring": True,
        "uses_closing_odds_as_feature": False,
        "locked_diagnostic_ablation_set": selection[
            "locked_diagnostic_ablation_set"
        ],
        "opener_blend_sensitivity": opener_blend_audit,
        "evaluated_candidate_names": names,
        "bootstrap_draws": int(args.bootstrap),
        "execution_provenance": audit["execution_provenance"],
        "artifacts": artifacts,
    }
    write_sealed_complete_manifest(
        output / "manifest.json", manifest, provenance_state
    )
    print(f"Completed frozen sealed matchup evaluation: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalist-dir", type=Path, default=DEFAULT_FINALIST_DIR)
    parser.add_argument(
        "--selection-manifest", type=Path, default=DEFAULT_SELECTION
    )
    parser.add_argument("--evaluation-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--candidate-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--round-panel", type=Path, default=DEFAULT_ROUND_PANEL)
    parser.add_argument(
        "--event-crosswalk", type=Path, default=DEFAULT_EVENT_CROSSWALK
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
    parser.add_argument("--preflight-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

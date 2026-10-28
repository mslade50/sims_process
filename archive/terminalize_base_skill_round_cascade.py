"""Terminalize a numerically complete cascade run after report-only failure.

This research-only utility never resumes fitting, pricing, settlement, or
bootstrap work.  It accepts only the exact artifact inventory written by
``evaluate_base_skill_round_cascade.py`` after its numeric audit succeeded but
before its report/manifest could be written.  The motivating failure was the
optional ``tabulate`` dependency used only by ``DataFrame.to_markdown``.

The terminalizer:

* requires the interrupted directory to remain non-terminal;
* fingerprints every numeric artifact and the original incomplete provenance;
* revalidates the original source/input provenance records;
* independently checks selection, chronology, settlement, sample, and
  bootstrap invariants;
* calls the original report function with a deterministic local Markdown-table
  formatter; and
* writes a fresh provenance directory and terminal manifest which reference,
  but do not mutate or relabel, the interrupted numeric directory.

Nothing here writes to a production input or external service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evaluate_base_skill_round_cascade as cascade  # noqa: E402
import research_execution_provenance as execution_provenance  # noqa: E402


ROOT = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_SOURCE = ROOT / "model_eval_round_cascade_v2_full_20260823"
DEFAULT_OUTPUT = ROOT / "model_eval_round_cascade_v2_terminalized_20260823"
CONTRACT = "base_skill_round_cascade_terminalization_v1"
EXPECTED_EVALUATOR_SHA256 = (
    "8544ad2c0076c95b737e4dc214c1bd5d2e05a5c9068f6242a9403f8794938202"
)
EXPECTED_SELECTED = {
    "complete__dynamic_kalman__xtour__w0__event_batch__q0.0005",
    "current_architecture_pt_v2_approx_control",
}
EXPECTED_NUMERIC_ARTIFACTS = frozenset(
    {
        "all_offered_edge_bets.parquet",
        "all_offered_edge_bucket_performance.csv",
        "audit.json",
        "bad_week_golfers.csv",
        "bad_weeks.csv",
        "bet_bucket_performance.csv",
        "bets.parquet",
        "c3_coefficients.csv",
        "c3_discovery_hyperparameter_scores.csv",
        "calibration.csv",
        "candidate_round_coverage.csv",
        "candidate_selection.csv",
        "cascade_player_round_predictions.parquet",
        "counterfactual_zero_gate_bets.parquet",
        "counterfactual_zero_gate_bucket_performance.csv",
        "discovery_probability_sigmas.csv",
        "event_cluster_bootstrap.csv",
        "model_quote_coverage.csv",
        "normalized_round_matchup_quotes.parquet",
        "performance.csv",
        "proper_scores.csv",
        "round_matchup_probabilities.parquet",
        "settlement_completeness_audit.json",
    }
)
FINAL_REPORT_NAME = "CASCADE_MARKET_EVAL_REPORT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


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
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _records_current(records: Iterable[Mapping[str, Any]], label: str) -> int:
    count = 0
    for record in records:
        path = Path(str(record.get("path", ""))).resolve()
        current = _fingerprint(path)
        changed = {
            key: {"declared": record.get(key), "current": current[key]}
            for key in ("sha256", "bytes", "mtime_ns")
            if record.get(key) != current[key]
        }
        if changed:
            raise ValueError(f"{label} changed after interrupted run: {path}: {changed}")
        count += 1
    return count


def validate_incomplete_provenance(
    source_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    provenance_dir = source_dir / "execution_provenance"
    incomplete_path = provenance_dir / "execution_provenance.incomplete.json"
    complete_path = provenance_dir / "execution_provenance.json"
    if complete_path.exists():
        raise ValueError("source run already has complete execution provenance")
    payload = json.loads(incomplete_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not False:
        raise ValueError("source execution provenance is not explicitly incomplete")
    sources = payload.get("sources")
    preserved = payload.get("preserved_sources")
    inputs = payload.get("inputs")
    if not all(isinstance(value, list) for value in (sources, preserved, inputs)):
        raise ValueError("incomplete provenance lacks source/preserved/input records")
    if len(sources) != len(preserved) or not sources:
        raise ValueError("incomplete provenance source snapshot inventory is malformed")
    source_count = _records_current(sources, "original source")
    preserved_count = _records_current(preserved, "preserved source")
    input_count = _records_current(inputs, "original input")
    for source_record, preserved_record in zip(sources, preserved):
        if (
            source_record.get("sha256") != preserved_record.get("sha256")
            or source_record.get("bytes") != preserved_record.get("bytes")
        ):
            raise ValueError("original preserved source does not match executed source")
    primary = Path(str(payload.get("primary_source", ""))).resolve()
    if primary != Path(str(sources[0]["path"])).resolve():
        raise ValueError("incomplete provenance primary source is not source record zero")
    if str(sources[0].get("sha256", "")).lower() != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("interrupted evaluator source hash is not the audited authority")
    original_snapshot = Path(str(preserved[0]["path"])).resolve()
    if original_snapshot.parent != provenance_dir.resolve():
        raise ValueError("original evaluator snapshot escapes interrupted provenance dir")
    audit = {
        "incomplete_provenance_path": str(incomplete_path.resolve()),
        "incomplete_provenance_sha256": _sha256(incomplete_path),
        "sources_verified": source_count,
        "preserved_sources_verified": preserved_count,
        "inputs_verified": input_count,
        "executed_evaluator_source_sha256": EXPECTED_EVALUATOR_SHA256,
        "executed_evaluator_snapshot": str(original_snapshot),
        "source_copy_pairs_exact": True,
    }
    paths = [incomplete_path, *[Path(str(x["path"])) for x in preserved]]
    return payload, audit, paths


def inventory_numeric_artifacts(source_dir: Path) -> list[dict[str, Any]]:
    if (source_dir / "manifest.json").exists():
        raise ValueError("interrupted numeric directory unexpectedly has a manifest")
    if (source_dir / FINAL_REPORT_NAME).exists():
        raise ValueError("interrupted numeric directory unexpectedly has a report")
    actual = {path.name for path in source_dir.iterdir() if path.is_file()}
    if actual != EXPECTED_NUMERIC_ARTIFACTS:
        raise ValueError(
            "interrupted numeric artifact inventory differs from authority: "
            f"missing={sorted(EXPECTED_NUMERIC_ARTIFACTS - actual)}, "
            f"extra={sorted(actual - EXPECTED_NUMERIC_ARTIFACTS)}"
        )
    return [
        {
            "path": name,
            "absolute_path": str((source_dir / name).resolve()),
            "bytes": int((source_dir / name).stat().st_size),
            "sha256": _sha256(source_dir / name),
        }
        for name in sorted(EXPECTED_NUMERIC_ARTIFACTS)
    ]


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_zero(payload: Mapping[str, Any], keys: Iterable[str]) -> None:
    failures = {key: payload.get(key) for key in keys if payload.get(key) != 0}
    if failures:
        raise ValueError(f"numeric audit has nonzero failure counters: {failures}")


def _date_max_not_after(values: pd.Series, cutoff: pd.Timestamp, label: str) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty or dates.max() > cutoff:
        raise ValueError(f"{label} exceeds discovery cutoff or is empty")
    return str(dates.max().date())


def validate_numeric_outputs(source_dir: Path) -> dict[str, Any]:
    audit = json.loads((source_dir / "audit.json").read_text(encoding="utf-8"))
    settlement = json.loads(
        (source_dir / "settlement_completeness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    expected_flags = {
        "research_only": True,
        "production_changed": False,
        "promotion_gate_passed": False,
        "sealed_results_used_for_selection": False,
        "target_or_upcoming_realized_weather_feature_used": False,
        "strict_c3_uses_current_cascade_prediction": False,
        "market_prices_used_for_cascade_probability_calibration": False,
        "opening_price_used_as_skill_or_cascade_feature": False,
        "closing_price_used_as_skill_or_cascade_feature": False,
        "bootstrap_includes_zero_bet_eligible_events": True,
    }
    for key, expected in expected_flags.items():
        _require_equal(audit.get(key), expected, f"audit flag {key}")
    _require_equal(audit.get("evaluator_source_sha256"), EXPECTED_EVALUATOR_SHA256, "evaluator hash")
    _require_equal(int(audit.get("bootstrap_draws", -1)), 10_000, "bootstrap draws")
    _require_equal(int(audit.get("seed", -1)), 20_260_822, "bootstrap seed")
    _require_equal(int(audit.get("models", -1)), 13, "model count")
    _require_equal(set(audit.get("selected_candidates", [])), EXPECTED_SELECTED, "selected candidates")
    _require_equal(audit.get("selection_role"), "diagnostic_challenger", "selection role")
    _require_zero(
        audit,
        (
            "duplicate_player_prediction_keys",
            "c3_training_boundary_violations",
            "live_lambda_selection_after_discovery_rows",
            "compatibility_quote_menu_drops",
            "ungraded_bets",
            "zero_gate_ungraded_bets",
            "all_offered_ungraded_bets",
            "nonzero_pnl_on_void_pushes",
            "probability_sigma_upper_boundary_hits",
        ),
    )
    for key in (
        "c1_formula_max_abs_error",
        "no_live_formula_max_abs_error",
        "live_shrink_formula_max_abs_error",
    ):
        if not math.isclose(float(audit.get(key, np.nan)), 0.0, abs_tol=1e-12):
            raise ValueError(f"numeric formula audit failed: {key}")
    if float(audit.get("probability_sum_max_abs_error", np.inf)) > 1e-12:
        raise ValueError("probability simplex audit failed")
    if float(audit.get("reverse_pair_symmetry_max_abs_error", np.inf)) > 1e-12:
        raise ValueError("reverse-pair symmetry audit failed")
    score_void = audit.get("score_void_classification", {})
    _require_equal(
        score_void.get("archived_vs_score_outcome_mismatches_when_both_scores_valid"),
        0,
        "score outcome mismatch count",
    )
    _require_equal(score_void.get("impossible_archived_outcome_pairs"), 0, "impossible outcomes")
    _require_equal(settlement.get("primary_ungraded_rows"), 0, "primary ungraded")
    _require_equal(settlement.get("zero_gate_ungraded_rows"), 0, "zero-gate ungraded")
    _require_equal(settlement.get("all_offered_ungraded_rows"), 0, "all-offered ungraded")
    _require_equal(settlement.get("primary_selected_rows"), audit.get("settled_bets"), "settled row count")

    selection = pd.read_csv(source_dir / "candidate_selection.csv")
    if len(selection) != 2 or not selection["selected"].fillna(False).astype(bool).all():
        raise ValueError("candidate selection must contain exactly two selected frozen rows")
    _require_equal(set(selection["candidate"].astype(str)), EXPECTED_SELECTED, "selection file candidates")
    if selection["promotion_gate_passed"].fillna(False).astype(bool).any():
        raise ValueError("terminalization cannot promote a diagnostic candidate")

    performance = pd.read_csv(source_dir / "performance.csv")
    key_columns = [
        "sample",
        "model",
        "round_market",
        "line_policy",
        "price_point",
        "sizing",
    ]
    _require_equal(len(performance), 3_120, "performance row count")
    _require_equal(set(performance["sample"]), {"discovery", "sealed", "comparable_3y"}, "performance samples")
    _require_equal(performance["model"].nunique(), 13, "performance models")
    if performance.duplicated(key_columns).any():
        raise ValueError("performance table has duplicate logical keys")

    bootstrap = pd.read_csv(source_dir / "event_cluster_bootstrap.csv")
    _require_equal(len(bootstrap), 2_080, "bootstrap row count")
    _require_equal(set(bootstrap["sample"]), {"sealed", "comparable_3y"}, "bootstrap samples")
    _require_equal(set(pd.to_numeric(bootstrap["draws"], errors="raise")), {10_000}, "bootstrap draws column")
    expected_boot_keys = set(
        map(
            tuple,
            performance.loc[performance["sample"].isin(["sealed", "comparable_3y"]), key_columns]
            .astype(str)
            .to_numpy(),
        )
    )
    actual_boot_keys = set(map(tuple, bootstrap[key_columns].astype(str).to_numpy()))
    _require_equal(actual_boot_keys, expected_boot_keys, "bootstrap/performance logical key closure")

    tuning = pd.read_csv(source_dir / "c3_discovery_hyperparameter_scores.csv")
    _require_equal(int(pd.to_numeric(tuning["sealed_rows_used_for_selection"], errors="raise").sum()), 0, "sealed tuning rows")
    tuning_max = _date_max_not_after(
        tuning["selection_max_event_date"], pd.Timestamp("2022-12-31"), "cascade tuning"
    )
    sigmas = pd.read_csv(source_dir / "discovery_probability_sigmas.csv")
    sigma_max = _date_max_not_after(
        sigmas["discovery_max_event_date"], pd.Timestamp("2022-12-31"), "probability calibration"
    )
    coefficients = pd.read_csv(source_dir / "c3_coefficients.csv")
    training_dates = pd.to_datetime(coefficients["training_max_event_end"], errors="coerce")
    prediction_cutoff = pd.to_datetime(
        pd.to_numeric(coefficients["prediction_year"], errors="raise").astype(int).astype(str)
        + "-01-01"
    )
    if training_dates.isna().any() or (training_dates >= prediction_cutoff).any():
        raise ValueError("C3 coefficient rows violate annual point-in-time boundary")

    proper = pd.read_csv(source_dir / "proper_scores.csv")
    _require_equal(set(proper["sample"]), {"discovery", "sealed", "comparable_3y"}, "proper-score samples")
    _require_equal(proper["model"].nunique(), 13, "proper-score models")

    event_dates = pd.read_parquet(source_dir / "bets.parquet", columns=["event_start_date"])[
        "event_start_date"
    ]
    event_dates = pd.to_datetime(event_dates, errors="coerce").dt.normalize()
    if event_dates.isna().any() or not event_dates.between(
        pd.Timestamp("2021-01-01"), pd.Timestamp("2026-08-22")
    ).all():
        raise ValueError("selected bets escape discovery/sealed evaluation calendar")
    comparable_rows = int(
        event_dates.between(pd.Timestamp("2023-08-22"), pd.Timestamp("2026-08-22")).sum()
    )
    sealed_rows = int(
        event_dates.between(pd.Timestamp("2023-01-01"), pd.Timestamp("2026-08-22")).sum()
    )
    if comparable_rows <= 0 or sealed_rows < comparable_rows:
        raise ValueError("exact comparable-three-year calendar has no valid rows")

    return {
        "numeric_audit_sha256": _sha256(source_dir / "audit.json"),
        "settlement_audit_sha256": _sha256(source_dir / "settlement_completeness_audit.json"),
        "research_only": True,
        "promotion_gate_passed": False,
        "selection_role": "diagnostic_challenger",
        "selected_candidates": sorted(EXPECTED_SELECTED),
        "performance_rows": int(len(performance)),
        "performance_logical_keys_unique": True,
        "bootstrap_rows": int(len(bootstrap)),
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20_260_822,
        "bootstrap_matches_sealed_and_comparable_performance_keys": True,
        "tuning_max_event_date": tuning_max,
        "probability_calibration_max_event_date": sigma_max,
        "c3_annual_point_in_time_boundary_violations": 0,
        "selected_bet_rows": int(len(event_dates)),
        "sealed_selected_bet_rows": sealed_rows,
        "comparable_3y_selected_bet_rows": comparable_rows,
        "comparable_3y_start": "2023-08-22",
        "sealed_start": "2023-01-01",
        "sealed_end": "2026-08-22",
        "primary_ungraded_rows": 0,
        "zero_gate_ungraded_rows": 0,
        "all_offered_ungraded_rows": 0,
        "target_or_upcoming_realized_weather_feature_used": False,
        "production_changed": False,
    }


def _markdown_scalar(value: Any, floatfmt: str | None) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        rendered = format(float(value), floatfmt or "g")
    else:
        rendered = str(value)
    return rendered.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def dataframe_to_markdown(frame: pd.DataFrame, *, index: bool, floatfmt: str | None) -> str:
    table = frame.reset_index() if index else frame.copy()
    headers = [_markdown_scalar(column, None) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in table.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(_markdown_scalar(value, floatfmt) for value in values)
            + " |"
        )
    return "\n".join(lines)


@contextmanager
def local_markdown_formatter():
    original = pd.DataFrame.to_markdown

    def replacement(
        frame: pd.DataFrame,
        buf: Any = None,
        *,
        mode: str = "wt",
        index: bool = True,
        storage_options: Any = None,
        **kwargs: Any,
    ) -> str | None:
        del storage_options
        rendered = dataframe_to_markdown(
            frame, index=index, floatfmt=kwargs.get("floatfmt")
        )
        if buf is None:
            return rendered
        if hasattr(buf, "write"):
            buf.write(rendered)
        else:
            Path(buf).write_text(rendered, encoding="utf-8")
        return None

    pd.DataFrame.to_markdown = replacement
    try:
        yield
    finally:
        pd.DataFrame.to_markdown = original


def render_report(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    audit = json.loads((source_dir / "audit.json").read_text(encoding="utf-8"))
    selection = pd.read_csv(source_dir / "candidate_selection.csv")
    proper = pd.read_csv(source_dir / "proper_scores.csv")
    performance = pd.read_csv(source_dir / "performance.csv")
    bootstrap = pd.read_csv(source_dir / "event_cluster_bootstrap.csv")
    tuning = pd.read_csv(source_dir / "c3_discovery_hyperparameter_scores.csv")
    coefficients = pd.read_csv(source_dir / "c3_coefficients.csv")
    with local_markdown_formatter():
        cascade.write_report(
            output_dir,
            audit,
            selection,
            proper,
            performance,
            bootstrap,
            tuning,
            coefficients,
        )
    report = output_dir / FINAL_REPORT_NAME
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("fallback report renderer produced no report")
    return {
        "path": str(report.resolve()),
        "bytes": int(report.stat().st_size),
        "sha256": _sha256(report),
        "formatter": "terminalizer_local_gfm_table_v1",
        "tabulate_available_at_terminalization": bool(
            importlib.util.find_spec("tabulate") is not None
        ),
    }


def _serialized_cli(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        str(Path(__file__).resolve()),
        "--source-dir",
        str(args.source_dir),
        "--output-dir",
        str(args.output_dir),
    ]


def run(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("terminalization output must differ from interrupted source")
    if output_dir.exists():
        raise FileExistsError(f"terminalization output must be fresh: {output_dir}")
    numeric_inventory = inventory_numeric_artifacts(source_dir)
    _original_provenance, original_audit, preserved_paths = (
        validate_incomplete_provenance(source_dir)
    )
    input_paths = [
        *(source_dir / entry["path"] for entry in numeric_inventory),
        *preserved_paths,
    ]
    state = execution_provenance.begin(
        output_dir=output_dir,
        primary_source=Path(__file__).resolve(),
        input_paths=input_paths,
        code_paths=(
            Path(cascade.__file__).resolve(),
            Path(original_audit["executed_evaluator_snapshot"]).resolve(),
            Path(execution_provenance.__file__).resolve(),
        ),
        cli=_serialized_cli(args),
    )
    numeric_validation = validate_numeric_outputs(source_dir)
    report_audit = render_report(source_dir, output_dir)
    readme = output_dir / "TERMINALIZATION_README.md"
    readme.write_text(
        "\n".join(
            [
                "# Cascade terminalization record",
                "",
                "Research only. No production input or external service was changed.",
                "",
                "The referenced cascade run completed all numeric work, including its ",
                "10,000-draw whole-event bootstrap and numeric audit, but exited before ",
                "its report/manifest because pandas' optional `tabulate` dependency was absent.",
                "",
                "This directory does not relabel the interrupted directory as complete. It ",
                "pins every interrupted numeric artifact, revalidates the original execution ",
                "provenance, checks the numeric contracts independently, renders the same report ",
                "with a local Markdown-table formatter, and supplies separate terminal provenance.",
                "",
                f"Interrupted numeric directory: `{source_dir}`",
            ]
        ),
        encoding="utf-8",
    )
    terminalization_audit = {
        "contract": CONTRACT,
        "complete": True,
        "research_only": True,
        "production_changed": False,
        "numeric_outputs_recomputed": False,
        "interrupted_directory_relabelled_complete": False,
        "terminalization_reason": "report-only ImportError: missing optional tabulate dependency",
        "interrupted_source_dir": str(source_dir),
        "terminal_output_dir": str(output_dir),
        "numeric_artifacts": numeric_inventory,
        "original_execution_provenance": original_audit,
        "numeric_validation": numeric_validation,
        "report": report_audit,
        "readme_sha256": _sha256(readme),
    }
    _atomic_json(output_dir / "terminalization_audit.json", terminalization_audit)
    execution_audit = execution_provenance.finalize(state)
    terminal_artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            terminal_artifacts.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    execution_provenance.assert_current(state)
    manifest = {
        "contract": CONTRACT,
        "complete": True,
        "research_only": True,
        "production_changed": False,
        "numeric_outputs_recomputed": False,
        "interrupted_directory_relabelled_complete": False,
        "interrupted_source_dir": str(source_dir),
        "numeric_artifacts": numeric_inventory,
        "numeric_audit_sha256": numeric_validation["numeric_audit_sha256"],
        "bootstrap_draws": numeric_validation["bootstrap_draws"],
        "bootstrap_seed": numeric_validation["bootstrap_seed"],
        "selection_role": numeric_validation["selection_role"],
        "promotion_gate_passed": False,
        "execution_provenance": execution_audit,
        "terminalization_audit_sha256": _sha256(
            output_dir / "terminalization_audit.json"
        ),
        "report_sha256": report_audit["sha256"],
        "artifacts": terminal_artifacts,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"Cascade terminalization complete: {output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

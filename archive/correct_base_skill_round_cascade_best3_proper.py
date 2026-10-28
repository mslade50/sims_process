"""Correct the model-scoping defect in cascade best-three proper summaries.

This research-only postprocessor reads the immutable, terminalized cascade's
hash-bound probability parquet.  It does not fit a model, recalibrate a
probability mapper, select bets, resettle outcomes, rerun a bootstrap, or
modify either the interrupted numeric directory or its terminal wrapper.

The original summary selected one best-priced quote per event/round/pair
before grouping by model.  That accidentally made models compete for rows.
This supplement selects the best book independently inside each model and
rebuilds only ``proper_scores.csv`` and ``calibration.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from archive import analyze_base_skill_weather_bad_weeks as terminal_adapter  # noqa: E402
from archive import evaluate_base_skill_round_cascade as cascade  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402


ROOT = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_TERMINAL_DIR = ROOT / "model_eval_round_cascade_v2_terminalized_20260823"
DEFAULT_PANEL_MANIFEST = ROOT / "manifest.json"
DEFAULT_OUTPUT = ROOT / "model_eval_round_cascade_v2_best3_proper_correction_20260823"
CONTRACT = "base_skill_round_cascade_best3_proper_correction_v1"
EXPECTED_TERMINAL_MANIFEST_SHA256 = (
    "6f36026612d99853b941e38696baa81a866f822c8ef1ea922d7bf5444816c139"
)
EXPECTED_PROBABILITY_SHA256 = (
    "097696176da84648e2dc5a028e0356d1417f70abb6eca803f635b33ca7fbb68d"
)
EXPECTED_EVALUATOR_SHA256 = (
    "8544ad2c0076c95b737e4dc214c1bd5d2e05a5c9068f6242a9403f8794938202"
)
EXPECTED_MODELS = 13
EXPECTED_SUMMARY_ROWS = 1_560
EXPECTED_BEST3_ROWS = 390
EXPECTED_FIXED_BOOK_ROWS = 1_170
EXPECTED_ORIGINAL_BEST3_ROWS = 54
EXPECTED_ORIGINAL_VALID_C0_BEST3_ROWS = 30
EXPECTED_ORIGINAL_INVALID_NON_C0_BEST3_ROWS = 24
C0_MODEL = "C0_current_cascade_retrospective"
SUMMARY_KEYS = ["sample", "model", "round_market", "line_policy", "price_point"]


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
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def model_scoped_best_price_market_rows(
    frame: pd.DataFrame, price_point: str
) -> pd.DataFrame:
    """Choose each model's best canonical-p1 quote and same-book p2 quote."""

    if price_point not in cascade.PRICE_POINTS:
        raise ValueError(price_point)
    if "model" not in frame:
        raise ValueError("best-price input lacks model")
    chosen = frame.copy()
    chosen["_book_priority"] = chosen["book"].map(cascade.BOOK_PRIORITY).fillna(99)
    time_column = f"{price_point}_time"
    chosen["_execution_time"] = pd.to_datetime(
        chosen.get(time_column), errors="coerce", utc=True
    )
    identity = ["model", "event_key", "round_market", "pair_key"]
    return (
        chosen.sort_values(
            [
                *identity,
                f"p1_{price_point}",
                "_book_priority",
                "_execution_time",
            ],
            ascending=[True, True, True, True, False, True, price_point == "open"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(identity, keep="first")
        .drop(columns=["_book_priority", "_execution_time"])
    )


def summarize_corrected_proper_scores(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proper_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_frame = scored.loc[cascade._sample_mask(scored, sample)].copy()
        for price_point in cascade.PRICE_POINTS:
            valid_price = cascade._valid_american(
                sample_frame[f"p1_{price_point}"]
            ) & cascade._valid_american(sample_frame[f"p2_{price_point}"])
            priced = sample_frame.loc[valid_price].copy()
            for model, model_frame in priced.groupby("model", observed=True):
                policies: list[tuple[str, pd.DataFrame]] = [
                    (book, model_frame.loc[model_frame["book"].eq(book)].copy())
                    for book in cascade.BOOKS
                ]
                policies.append(
                    (
                        "best_of_three",
                        model_scoped_best_price_market_rows(model_frame, price_point),
                    )
                )
                for line_policy, policy_frame in policies:
                    if policy_frame.empty:
                        continue
                    for round_label, group in [
                        *[
                            (
                                f"r{round_num}",
                                policy_frame.loc[
                                    policy_frame["round_num"].eq(round_num)
                                ],
                            )
                            for round_num in range(1, 5)
                        ],
                        ("pooled", policy_frame),
                    ]:
                        if group.empty:
                            continue
                        common = {
                            "sample": sample,
                            "model": model,
                            "candidate": group["candidate"].iloc[0],
                            "variant": group["variant"].iloc[0],
                            "round_market": round_label,
                            "line_policy": line_policy,
                            "price_point": price_point,
                        }
                        proper = cascade._proper_group(group, price_point)
                        if proper:
                            proper_rows.append({**common, **proper})
                        calibration = cascade._calibration_group(group)
                        if calibration:
                            calibration_rows.append({**common, **calibration})
    return pd.DataFrame(proper_rows), pd.DataFrame(calibration_rows)


def validate_summary_closure(
    frame: pd.DataFrame, *, models: Iterable[str], label: str
) -> None:
    model_set = set(map(str, models))
    required = [*SUMMARY_KEYS, "candidate", "variant"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")
    if len(frame) != EXPECTED_SUMMARY_ROWS or frame.duplicated(SUMMARY_KEYS).any():
        raise ValueError(
            f"{label} logical closure differs: rows={len(frame)}, "
            f"duplicates={int(frame.duplicated(SUMMARY_KEYS).sum())}"
        )
    if set(frame["model"].astype(str)) != model_set or len(model_set) != EXPECTED_MODELS:
        raise ValueError(f"{label} model menu differs from 13-model authority")
    expected = {
        (sample, model, round_market, line_policy, price_point)
        for sample in ("discovery", "sealed", "comparable_3y")
        for model in model_set
        for round_market in ("r1", "r2", "r3", "r4", "pooled")
        for line_policy in (*cascade.BOOKS, "best_of_three")
        for price_point in cascade.PRICE_POINTS
    }
    observed = set(map(tuple, frame[SUMMARY_KEYS].astype(str).itertuples(index=False, name=None)))
    if observed != expected:
        raise ValueError(
            f"{label} key closure differs; missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )
    best3 = frame.loc[frame["line_policy"].eq("best_of_three")]
    fixed = frame.loc[~frame["line_policy"].eq("best_of_three")]
    if len(best3) != EXPECTED_BEST3_ROWS or len(fixed) != EXPECTED_FIXED_BOOK_ROWS:
        raise ValueError(f"{label} fixed/best-three row split differs")


def _assert_matching_rows(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    label: str,
    tolerance: float = 1e-12,
) -> None:
    if expected.duplicated(SUMMARY_KEYS).any() or observed.duplicated(SUMMARY_KEYS).any():
        raise ValueError(f"{label} has duplicate logical keys")
    merged = expected.merge(
        observed,
        on=SUMMARY_KEYS,
        how="outer",
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label} key menus differ")
    shared = sorted((set(expected.columns) & set(observed.columns)) - set(SUMMARY_KEYS))
    for column in shared:
        left = merged[f"{column}_expected"]
        right = merged[f"{column}_observed"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            same = np.isclose(
                pd.to_numeric(left, errors="coerce").to_numpy(float),
                pd.to_numeric(right, errors="coerce").to_numpy(float),
                atol=tolerance,
                rtol=0.0,
                equal_nan=True,
            )
        else:
            same = left.fillna("__missing__").astype(str).eq(
                right.fillna("__missing__").astype(str)
            ).to_numpy()
        if not bool(np.all(same)):
            raise ValueError(f"{label} differs at {column}")


def validate_original_defect_and_unaffected_rows(
    corrected: pd.DataFrame,
    original: pd.DataFrame,
    *,
    label: str,
) -> dict[str, Any]:
    original_best3 = original.loc[original["line_policy"].eq("best_of_three")].copy()
    original_fixed = original.loc[~original["line_policy"].eq("best_of_three")].copy()
    corrected_best3 = corrected.loc[corrected["line_policy"].eq("best_of_three")].copy()
    corrected_fixed = corrected.loc[~corrected["line_policy"].eq("best_of_three")].copy()
    if (
        len(original_best3) != EXPECTED_ORIGINAL_BEST3_ROWS
        or len(original_fixed) != EXPECTED_FIXED_BOOK_ROWS
    ):
        raise ValueError(f"{label} original row counts differ from audited defect")
    _assert_matching_rows(corrected_fixed, original_fixed, label=f"{label} fixed books")
    c0_original = original_best3.loc[original_best3["model"].eq(C0_MODEL)]
    c0_corrected = corrected_best3.loc[corrected_best3["model"].eq(C0_MODEL)]
    if len(c0_original) != EXPECTED_ORIGINAL_VALID_C0_BEST3_ROWS:
        raise ValueError(f"{label} original C0 best-three row count differs")
    _assert_matching_rows(c0_corrected, c0_original, label=f"{label} C0 best three")
    non_c0 = original_best3.loc[~original_best3["model"].eq(C0_MODEL)].copy()
    if len(non_c0) != EXPECTED_ORIGINAL_INVALID_NON_C0_BEST3_ROWS:
        raise ValueError(f"{label} original invalid best-three row count differs")
    matching_keys = non_c0.merge(
        corrected_best3,
        on=SUMMARY_KEYS,
        how="left",
        suffixes=("_original", "_corrected"),
        validate="one_to_one",
        indicator=True,
    )
    if not matching_keys["_merge"].eq("both").all():
        raise ValueError(f"{label} invalid original rows lack corrected counterparts")
    if "resolved_quotes_original" in matching_keys:
        changed = pd.to_numeric(
            matching_keys["resolved_quotes_original"], errors="coerce"
        ).ne(pd.to_numeric(matching_keys["resolved_quotes_corrected"], errors="coerce"))
        if not changed.all():
            raise ValueError(f"{label} expected every non-C0 best-three menu to change")
    return {
        "original_rows": int(len(original)),
        "corrected_rows": int(len(corrected)),
        "original_fixed_book_rows_reproduced": int(len(original_fixed)),
        "original_valid_c0_best3_rows_reproduced": int(len(c0_original)),
        "original_invalid_non_c0_best3_rows_replaced": int(len(non_c0)),
        "original_missing_best3_rows_added": int(
            len(corrected_best3) - len(original_best3)
        ),
    }


def _write_report(output: Path, audit: Mapping[str, Any]) -> None:
    lines = [
        "# Corrected cascade best-of-three proper-score supplement",
        "",
        "Research only. The terminalized wrapper and interrupted numeric directory remain immutable.",
        "",
        "The correction scopes best-price selection by model and recomputes only proper-score and calibration summaries from the hash-bound probability parquet.",
        "",
        f"- Corrected proper-score rows: {audit['proper_scores']['corrected_rows']:,}",
        f"- Corrected calibration rows: {audit['calibration']['corrected_rows']:,}",
        f"- Best-of-three rows in each output: {EXPECTED_BEST3_ROWS:,}",
        f"- Fixed-book rows independently reproduced in each output: {EXPECTED_FIXED_BOOK_ROWS:,}",
        f"- Missing best-three rows restored: {audit['proper_scores']['original_missing_best3_rows_added']:,}",
        f"- Invalid surviving non-C0 best-three rows replaced: {audit['proper_scores']['original_invalid_non_c0_best3_rows_replaced']:,}",
        "",
        "No probabilities, bets, settlements, P&L, bootstrap draws, model parameters, or production files were changed.",
        "The complete status in this directory applies only to this correction supplement; the original evaluator remains explicitly incomplete.",
    ]
    (output / "BEST3_PROPER_CORRECTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace, *, argv: Sequence[str] | None = None) -> None:
    terminal_dir = args.terminal_dir.resolve()
    panel_manifest = args.panel_manifest.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"correction output must be a fresh path: {output}")
    if output == terminal_dir or terminal_dir in output.parents:
        raise ValueError("correction output must be disjoint from terminal wrapper")
    source_dir = Path(
        str(json.loads((terminal_dir / "manifest.json").read_text(encoding="utf-8"))[
            "interrupted_source_dir"
        ])
    ).resolve()
    if output == source_dir or source_dir in output.parents:
        raise ValueError("correction output must be disjoint from interrupted source")
    if _sha256(terminal_dir / "manifest.json") != EXPECTED_TERMINAL_MANIFEST_SHA256:
        raise ValueError("terminal cascade manifest is not the correction authority")
    if _sha256(Path(cascade.__file__).resolve()) != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("cascade evaluator source is not the correction authority")

    manifest, _, artifacts, context = terminal_adapter.validate_cascade_provenance(
        terminal_dir, panel_manifest
    )
    if (
        context.get("provenance_mode") != "terminal_wrapper_external_numeric_v1"
        or context.get("original_evaluator_complete") is not False
        or context.get("numeric_artifacts_external_read_only") is not True
    ):
        raise ValueError("correction requires the fail-closed terminal wrapper adapter")
    probability_path = artifacts["round_matchup_probabilities.parquet"]
    if _sha256(probability_path) != EXPECTED_PROBABILITY_SHA256:
        raise ValueError("cascade probability parquet is not the correction authority")
    input_paths = tuple(
        sorted(
            {
                (terminal_dir / "manifest.json").resolve(),
                *map(Path.resolve, context["direct_paths"]),
            },
            key=str,
        )
    )
    provenance = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__).resolve(),
        input_paths=input_paths,
        code_paths=(
            Path(cascade.__file__).resolve(),
            Path(terminal_adapter.__file__).resolve(),
            terminal_adapter.CASCADE_TERMINALIZER_PRIMARY_SOURCE.resolve(),
            Path(execution_provenance.__file__).resolve(),
        ),
        cli=tuple(sys.argv if argv is None else map(str, argv)),
    )
    # Revalidate the entire wrapper/original-input chain after the provenance snapshot.
    _, _, artifacts_after, context_after = terminal_adapter.validate_cascade_provenance(
        terminal_dir, panel_manifest
    )
    if (
        artifacts_after != artifacts
        or tuple(context_after["direct_paths"]) != tuple(context["direct_paths"])
    ):
        raise RuntimeError("terminal cascade mapping changed across provenance snapshot")

    scored = pd.read_parquet(probability_path)
    models = sorted(scored["model"].astype(str).unique())
    if len(models) != EXPECTED_MODELS:
        raise ValueError("probability parquet model count differs from authority")
    proper, calibration = summarize_corrected_proper_scores(scored)
    proper = proper.sort_values(SUMMARY_KEYS, kind="mergesort").reset_index(drop=True)
    calibration = calibration.sort_values(SUMMARY_KEYS, kind="mergesort").reset_index(
        drop=True
    )
    validate_summary_closure(proper, models=models, label="corrected proper scores")
    validate_summary_closure(calibration, models=models, label="corrected calibration")

    original_proper_path = artifacts["proper_scores.csv"]
    original_calibration_path = artifacts["calibration.csv"]
    original_proper = pd.read_csv(original_proper_path)
    original_calibration = pd.read_csv(original_calibration_path)
    proper_audit = validate_original_defect_and_unaffected_rows(
        proper, original_proper, label="proper scores"
    )
    calibration_audit = validate_original_defect_and_unaffected_rows(
        calibration, original_calibration, label="calibration"
    )

    _atomic_csv(proper, output / "proper_scores.csv")
    _atomic_csv(calibration, output / "calibration.csv")
    audit = {
        "contract": CONTRACT,
        "complete": True,
        "research_only": True,
        "production_changed": False,
        "source_terminal_manifest": {
            "path": str(terminal_dir / "manifest.json"),
            "sha256": _sha256(terminal_dir / "manifest.json"),
            "contract": manifest["contract"],
        },
        "source_probability_parquet": {
            "path": str(probability_path),
            "bytes": probability_path.stat().st_size,
            "sha256": _sha256(probability_path),
        },
        "source_original_proper_scores_sha256": _sha256(original_proper_path),
        "source_original_calibration_sha256": _sha256(original_calibration_path),
        "terminal_wrapper_complete": True,
        "original_evaluator_complete": False,
        "numeric_artifacts_external_read_only": True,
        "best_price_identity_includes_model": True,
        "models": models,
        "samples": ["discovery", "sealed", "comparable_3y"],
        "price_points": list(cascade.PRICE_POINTS),
        "line_policies": [*cascade.BOOKS, "best_of_three"],
        "round_markets": ["r1", "r2", "r3", "r4", "pooled"],
        "proper_scores": proper_audit,
        "calibration": calibration_audit,
        "model_refit": False,
        "model_reselection": False,
        "probability_mapper_refit": False,
        "bet_reselection": False,
        "settlement_recomputed": False,
        "pnl_recomputed": False,
        "bootstrap_recomputed": False,
        "terminal_or_interrupted_artifacts_mutated": False,
        "execution_provenance": {
            "path": "execution_provenance/execution_provenance.json"
        },
    }
    _atomic_json(output / "audit.json", audit)
    _write_report(output, audit)
    execution_audit = execution_provenance.finalize(provenance)
    artifacts_out = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts_out.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    execution_provenance.assert_current(provenance)
    output_manifest = {
        "contract": CONTRACT,
        "complete": True,
        "research_only": True,
        "production_changed": False,
        "source_terminal_manifest_sha256": EXPECTED_TERMINAL_MANIFEST_SHA256,
        "source_probability_sha256": EXPECTED_PROBABILITY_SHA256,
        "corrected_proper_score_rows": len(proper),
        "corrected_calibration_rows": len(calibration),
        "terminal_wrapper_complete": True,
        "original_evaluator_complete": False,
        "execution_provenance": execution_audit,
        "audit_sha256": _sha256(output / "audit.json"),
        "report_sha256": _sha256(output / "BEST3_PROPER_CORRECTION_REPORT.md"),
        "inputs": {
            "terminal_manifest": {
                "path": str(terminal_dir / "manifest.json"),
                "sha256": EXPECTED_TERMINAL_MANIFEST_SHA256,
            },
            "round_matchup_probabilities": {
                "path": str(probability_path),
                "sha256": EXPECTED_PROBABILITY_SHA256,
            },
            "original_proper_scores": {
                "path": str(original_proper_path),
                "sha256": _sha256(original_proper_path),
            },
            "original_calibration": {
                "path": str(original_calibration_path),
                "sha256": _sha256(original_calibration_path),
            },
        },
        "artifacts": artifacts_out,
    }
    _atomic_json(output / "manifest.json", output_manifest)
    print(f"Corrected best-three proper-score supplement complete: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-dir", type=Path, default=DEFAULT_TERMINAL_DIR)
    parser.add_argument(
        "--panel-manifest", type=Path, default=DEFAULT_PANEL_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args(), argv=sys.argv)

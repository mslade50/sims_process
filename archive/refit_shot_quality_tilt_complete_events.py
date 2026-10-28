"""Refit shot-quality beta selection after a predeclared completeness gate.

This research-only postprocessor consumes the frozen candidate-event panel
written by ``evaluate_shot_quality_tilt.py``.  It never reloads shot data and
never changes an event score.  It only removes event/category panels with too
few held-out atoms *before* both chronological tuning and testing, then
repeats the original nested beta-selection calculation.

The default 500-atom floor is intended to reject broken/incomplete ShotLink
event-category exports, not to optimize a coefficient.  Production code is
not imported or modified.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO / "permanent_data" / "shot_quality_tilt_eval_20260820_v2"
    / "candidate_event_scores.csv"
)
DEFAULT_OUTPUT = (
    REPO / "permanent_data" / "shot_quality_tilt_complete_events_20260821_v1"
)
PRIMARY_CATEGORIES = ("ott", "app", "arg")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_panel(frame: pd.DataFrame) -> np.ndarray:
    required = {
        "scenario", "donor_normalization", "category", "event_key",
        "event_name", "season", "beta", "shots", "logloss_delta_vs_beta0",
        "crps_delta_vs_beta0",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate panel is missing columns: {missing}")
    key = ["scenario", "donor_normalization", "category", "event_key"]
    duplicate = frame.duplicated(key + ["beta"])
    if duplicate.any():
        raise ValueError("candidate panel has duplicate event/category/beta rows")
    beta_grids = frame.groupby(key, observed=True)["beta"].agg(
        lambda values: tuple(np.sort(np.asarray(values, dtype=float)))
    )
    if beta_grids.nunique() != 1:
        raise ValueError("candidate panels do not share one exact beta grid")
    shot_counts = frame.groupby(key, observed=True)["shots"].nunique()
    if not shot_counts.eq(1).all():
        raise ValueError("shot count varies across beta within an event/category")
    betas = np.sort(frame["beta"].unique().astype(float))
    if not np.isclose(betas, 0.0).any():
        raise ValueError("beta-zero control is absent")
    beta_zero = frame[np.isclose(frame["beta"], 0.0)]
    if not np.allclose(beta_zero["logloss_delta_vs_beta0"], 0.0, atol=0.0):
        raise ValueError("beta-zero log-score parity failed")
    if not np.allclose(beta_zero["crps_delta_vs_beta0"], 0.0, atol=0.0):
        raise ValueError("beta-zero CRPS parity failed")
    return betas


def apply_completeness_gate(
    frame: pd.DataFrame, minimum_target_atoms: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if minimum_target_atoms <= 0:
        raise ValueError("minimum_target_atoms must be positive")
    key = ["scenario", "donor_normalization", "category", "event_key"]
    counts = (
        frame[np.isclose(frame["beta"], 0.0)][
            key + ["event_name", "season", "shots"]
        ]
        .copy()
        .rename(columns={"shots": "target_atoms"})
    )
    counts["complete"] = counts["target_atoms"].ge(minimum_target_atoms)
    accepted = counts[counts["complete"]][key]
    filtered = frame.merge(accepted, on=key, how="inner", validate="many_to_one")
    excluded = counts[~counts["complete"]].sort_values(key).reset_index(drop=True)
    return filtered, excluded


def rolling_selection(
    curves: pd.DataFrame,
    betas: np.ndarray,
    *,
    start_season: int,
    end_season: int,
    minimum_tune_events: int,
    beta_prior_events: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    group_cols = ["scenario", "donor_normalization", "category"]
    for keys, panel in curves.groupby(group_cols, observed=True):
        for test_year in range(start_season + 2, end_season + 1):
            tune = panel[panel["season"].lt(test_year)]
            test = panel[panel["season"].eq(test_year)]
            tune_events = int(tune["event_key"].nunique())
            if tune_events < minimum_tune_events or test.empty:
                continue
            objective = tune.groupby("beta", observed=True)[
                "logloss_delta_vs_beta0"
            ].mean()
            raw_beta = float(objective.idxmin())
            reliability = tune_events / (tune_events + beta_prior_events)
            shrunk_beta = raw_beta * reliability
            selected_beta = float(betas[np.argmin(np.abs(betas - shrunk_beta))])
            selected = test[np.isclose(test["beta"], selected_beta)].copy()
            if selected.empty:
                raise AssertionError("selected beta is absent from test panel")
            log_delta = selected["logloss_delta_vs_beta0"].to_numpy(float)
            crps_delta = selected["crps_delta_vs_beta0"].to_numpy(float)
            weights = selected["shots"].to_numpy(float)
            event_count = len(selected)
            log_se = (
                float(np.std(log_delta, ddof=1) / math.sqrt(event_count))
                if event_count > 1 else math.nan
            )
            crps_se = (
                float(np.std(crps_delta, ddof=1) / math.sqrt(event_count))
                if event_count > 1 else math.nan
            )
            fold_rows.append(
                {
                    "scenario": keys[0],
                    "donor_normalization": keys[1],
                    "category": keys[2],
                    "test_year": test_year,
                    "tune_events": tune_events,
                    "test_events": event_count,
                    "test_shots": int(selected["shots"].sum()),
                    "raw_best_beta": raw_beta,
                    "beta_reliability": float(reliability),
                    "selected_beta": selected_beta,
                    "equal_event_logloss_delta": float(np.mean(log_delta)),
                    "event_cluster_se": log_se,
                    "ci_low": float(np.mean(log_delta) - 1.96 * log_se),
                    "ci_high": float(np.mean(log_delta) + 1.96 * log_se),
                    "shot_weighted_logloss_delta": float(
                        np.average(log_delta, weights=weights)
                    ),
                    "favorable_logloss_events": int(np.sum(log_delta < 0)),
                    "equal_event_crps_delta": float(np.mean(crps_delta)),
                    "crps_event_cluster_se": crps_se,
                    "shot_weighted_crps_delta": float(
                        np.average(crps_delta, weights=weights)
                    ),
                    "favorable_crps_events": int(np.sum(crps_delta < 0)),
                }
            )
            selected["test_year"] = test_year
            selected["selected_beta"] = selected_beta
            selected_rows.append(selected)
    folds = pd.DataFrame(fold_rows)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return folds, selected


def _pooled_primary(selected: pd.DataFrame) -> pd.DataFrame:
    primary = selected[
        selected["scenario"].eq("all_valid")
        & selected["donor_normalization"].eq("event_normalized")
        & selected["category"].isin(PRIMARY_CATEGORIES)
    ]
    rows: list[dict[str, object]] = []
    for category, group in primary.groupby("category", observed=True):
        log_delta = group["logloss_delta_vs_beta0"].to_numpy(float)
        crps_delta = group["crps_delta_vs_beta0"].to_numpy(float)
        log_se = float(np.std(log_delta, ddof=1) / math.sqrt(len(group)))
        crps_se = float(np.std(crps_delta, ddof=1) / math.sqrt(len(group)))
        rows.append(
            {
                "category": category,
                "events": len(group),
                "shots": int(group["shots"].sum()),
                "selected_betas": ",".join(
                    f"{value:.1f}" for value in sorted(group["selected_beta"].unique())
                ),
                "equal_event_logloss_delta": float(np.mean(log_delta)),
                "ci_low": float(np.mean(log_delta) - 1.96 * log_se),
                "ci_high": float(np.mean(log_delta) + 1.96 * log_se),
                "favorable_logloss_events": int(np.sum(log_delta < 0)),
                "equal_event_crps_delta": float(np.mean(crps_delta)),
                "crps_ci_low": float(np.mean(crps_delta) - 1.96 * crps_se),
                "crps_ci_high": float(np.mean(crps_delta) + 1.96 * crps_se),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    if frame.empty:
        return "None."
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    excluded: pd.DataFrame,
    minimum_target_atoms: int,
) -> Path:
    pooled = _pooled_primary(selected)
    lines = [
        "# Complete-event shot quality-tilt selection",
        "",
        "This postprocessor repeats the original chronological beta selection after ",
        f"requiring at least {minimum_target_atoms:,} held-out atoms in each event/category.",
        "The candidate proper scores are unchanged; only incomplete panels are removed ",
        "before both tuning and testing.",
        "",
        "## Pooled primary results",
        "",
        _markdown_table(pooled),
        "",
        "## Excluded panels",
        "",
        _markdown_table(excluded, decimals=0),
        "",
        "This remains research evidence for conditional reward atoms. It does not validate ",
        "spatial endpoints, hazards, recursion, round distributions, or production enablement.",
    ]
    report = output / "COMPLETE_EVENT_QUALITY_TILT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-target-atoms", type=int, default=500)
    parser.add_argument("--minimum-tune-events", type=int, default=20)
    parser.add_argument("--beta-prior-events", type=float, default=20.0)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_hash_start = _sha256(Path(__file__))
    input_hash = _sha256(args.candidate_scores)
    panel = pd.read_csv(args.candidate_scores)
    betas = _validate_panel(panel)
    filtered, excluded = apply_completeness_gate(panel, args.minimum_target_atoms)
    folds, selected = rolling_selection(
        filtered,
        betas,
        start_season=args.start_season,
        end_season=args.end_season,
        minimum_tune_events=args.minimum_tune_events,
        beta_prior_events=args.beta_prior_events,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {
        "rolling_fold_results_complete.csv": args.output_dir / "rolling_fold_results_complete.csv",
        "selected_event_scores_complete.csv": args.output_dir / "selected_event_scores_complete.csv",
        "excluded_event_categories.csv": args.output_dir / "excluded_event_categories.csv",
    }
    folds.to_csv(outputs["rolling_fold_results_complete.csv"], index=False)
    selected.to_csv(outputs["selected_event_scores_complete.csv"], index=False)
    excluded.to_csv(outputs["excluded_event_categories.csv"], index=False)
    report = _write_report(
        args.output_dir, folds, selected, excluded, args.minimum_target_atoms
    )
    outputs[report.name] = report
    source_hash_end = _sha256(Path(__file__))
    if source_hash_start != source_hash_end:
        raise AssertionError("postprocessor source changed during execution")
    input_hash_end = _sha256(args.candidate_scores)
    if input_hash_end != input_hash:
        raise AssertionError("candidate score input changed during execution")
    manifest = {
        "contract": "nested_quality_tilt_complete_event_gate_v1",
        "research_only": True,
        "production_enabled": False,
        "candidate_scores": str(args.candidate_scores.resolve()),
        "candidate_scores_sha256": input_hash,
        "candidate_scores_sha256_at_end": input_hash_end,
        "source_sha256": source_hash_start,
        "source_sha256_at_end": source_hash_end,
        "minimum_target_atoms": args.minimum_target_atoms,
        "minimum_tune_events": args.minimum_tune_events,
        "beta_prior_events": args.beta_prior_events,
        "beta_grid": betas.tolist(),
        "input_rows": len(panel),
        "filtered_rows": len(filtered),
        "excluded_event_category_arms": len(excluded),
        "output_sha256": {name: _sha256(path) for name, path in outputs.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Excluded event/category/arm panels: {len(excluded)}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

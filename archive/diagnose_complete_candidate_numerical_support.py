"""No-market numerical support diagnostic for complete base-skill candidates.

This script deliberately reads only frozen feature snapshots and training labels.
It compares raw linear predictions with projections of each test covariate vector
onto support estimated from the corresponding strict-prior training fold.  It
does not select candidates or read betting-market data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import build_complete_base_skill_candidates as builder


DEFAULT_AUGMENTED = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
    "model_completion_v2_provenance/augmented_feature_snapshots.parquet"
)
DEFAULT_OUTPUT = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
    "model_completion_v3_numerical_support_diagnostic"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    projected = np.asarray(x_test, dtype=float).copy()
    train_norm = np.linalg.norm(x_train, axis=1)
    before_norm = np.linalg.norm(projected, axis=1)
    element_clipped_rows = np.zeros(len(projected), dtype=bool)
    radius = np.nan

    if "element001" in method:
        lower = np.quantile(x_train, 0.001, axis=0)
        upper = np.quantile(x_train, 0.999, axis=0)
        element_clipped_rows = np.any(
            (projected < lower[None, :]) | (projected > upper[None, :]), axis=1
        )
        projected = np.minimum(np.maximum(projected, lower[None, :]), upper[None, :])
    elif "element005" in method:
        lower = np.quantile(x_train, 0.005, axis=0)
        upper = np.quantile(x_train, 0.995, axis=0)
        element_clipped_rows = np.any(
            (projected < lower[None, :]) | (projected > upper[None, :]), axis=1
        )
        projected = np.minimum(np.maximum(projected, lower[None, :]), upper[None, :])

    radial_rows = np.zeros(len(projected), dtype=bool)
    if "radial999" in method:
        radius = float(np.quantile(train_norm, 0.999))
    elif "radial995" in method:
        radius = float(np.quantile(train_norm, 0.995))
    if np.isfinite(radius):
        current_norm = np.linalg.norm(projected, axis=1)
        radial_rows = current_norm > radius
        scale = np.minimum(1.0, radius / np.maximum(current_norm, 1e-12))
        projected *= scale[:, None]

    after_norm = np.linalg.norm(projected, axis=1)
    return projected, {
        "training_norm_q995": float(np.quantile(train_norm, 0.995)),
        "training_norm_q999": float(np.quantile(train_norm, 0.999)),
        "test_norm_max_before": float(before_norm.max(initial=0.0)),
        "test_norm_max_after": float(after_norm.max(initial=0.0)),
        "element_clipped_rows": int(element_clipped_rows.sum()),
        "radial_projected_rows": int(radial_rows.sum()),
        "projected_rows": int((element_clipped_rows | radial_rows).sum()),
    }


def _candidate(source: str, family: str, alpha: float) -> builder.CandidateSpec:
    return builder.CandidateSpec(
        name=f"diagnostic__{family}__{source}__a{alpha:g}",
        family=family,
        source=source,
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        alpha=alpha,
        description="no-market training-support diagnostic",
    )


def run(augmented_path: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    snapshots = pd.read_parquet(augmented_path)
    snapshots["event_start_date"] = pd.to_datetime(
        snapshots["event_start_date"], errors="raise"
    ).dt.normalize()
    snapshots["event_end_date"] = pd.to_datetime(
        snapshots["event_end_date"], errors="raise"
    ).dt.normalize()
    tests = snapshots.loc[
        snapshots["event_start_date"].ge(builder.PREDICTION_START)
    ].copy()
    tests["_refit_cutoff"] = builder._period_cutoff(
        tests["event_start_date"], "quarterly"
    )
    methods = (
        "raw",
        "radial999",
        "radial995",
        "element001",
        "element001_radial999",
        "element005",
        "element005_radial995",
    )
    specifications = (
        ("total", "ridge", 1.0),
        ("total", "robust_irls", 1.0),
        ("total", "robust_irls", 10.0),
        ("total_cat_img", "ridge", 1.0),
        ("total_cat_img", "robust_irls", 1.0),
        ("total_cat_img", "robust_irls", 10.0),
    )
    result_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []

    for source, family, alpha in specifications:
        candidate = _candidate(source, family, alpha)
        features = builder.candidate_feature_columns(snapshots, candidate)
        target_column, count_column = builder._target_columns(candidate)
        for cutoff, test in tests.groupby("_refit_cutoff", sort=True, observed=True):
            cutoff = pd.Timestamp(cutoff).normalize()
            train = snapshots.loc[
                snapshots["event_end_date"].lt(cutoff)
                & snapshots[target_column].notna()
                & pd.to_numeric(snapshots[count_column], errors="coerce").ge(2)
            ].copy()
            if len(train) < 500:
                raise RuntimeError(f"unexpected thin fold {cutoff}: {len(train)} rows")
            usable = [column for column in features if train[column].notna().any()]
            imputer = SimpleImputer(strategy="median", add_indicator=False)
            scaler = StandardScaler()
            train_imputed = imputer.fit_transform(train[usable])
            x_train = scaler.fit_transform(train_imputed)
            test_imputed = imputer.transform(test[usable])
            x_test = scaler.transform(test_imputed)
            y_train = pd.to_numeric(train[target_column], errors="raise").to_numpy(float)
            weights = builder._event_equal_weights(train)
            if family == "ridge":
                model: Any = Ridge(alpha=alpha).fit(
                    x_train, y_train, sample_weight=weights
                )
            else:
                model = builder._fit_regression(
                    candidate, x_train, y_train, weights
                )
            coefficient = np.asarray(model.coef_, dtype=float).reshape(-1)
            if not np.isfinite(coefficient).all():
                raise RuntimeError(f"nonfinite coefficient: {candidate.name} {cutoff}")
            for method in methods:
                projected, support = _project_test(x_train, x_test, method)
                prediction = np.asarray(model.predict(projected), dtype=float)
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"nonfinite prediction: {candidate.name} {cutoff}")
                result_rows.append(
                    {
                        "source": source,
                        "family": family,
                        "alpha": alpha,
                        "cutoff": cutoff,
                        "method": method,
                        "training_rows": len(train),
                        "test_rows": len(test),
                        "features": len(usable),
                        "prediction_min": float(prediction.min()),
                        "prediction_max": float(prediction.max()),
                        "prediction_abs_max": float(np.abs(prediction).max()),
                        "abs_gt_5": int((np.abs(prediction) > 5).sum()),
                        "abs_gt_10": int((np.abs(prediction) > 10).sum()),
                        "coefficient_abs_max": float(np.abs(coefficient).max()),
                        "coefficient_l2": float(np.linalg.norm(coefficient)),
                        **support,
                    }
                )

            raw_prediction = np.asarray(model.predict(x_test), dtype=float)
            order = np.argsort(np.abs(raw_prediction))[-3:][::-1]
            medians = np.asarray(imputer.statistics_, dtype=float)
            for test_position in order:
                row = test.iloc[int(test_position)]
                missing = test[usable].iloc[int(test_position)].isna().to_numpy(bool)
                extreme_id = (
                    f"{source}|{family}|{alpha:g}|{cutoff.date()}|"
                    f"{row['event_key']}|{row['dg_id_key']}"
                )
                extreme_rows.append(
                    {
                        "extreme_id": extreme_id,
                        "source": source,
                        "family": family,
                        "alpha": alpha,
                        "cutoff": cutoff,
                        "event_key": row["event_key"],
                        "event_start_date": row["event_start_date"],
                        "dg_id_key": row["dg_id_key"],
                        "prediction": float(raw_prediction[int(test_position)]),
                        "standardized_l2": float(np.linalg.norm(x_test[int(test_position)])),
                        "standardized_abs_max": float(np.abs(x_test[int(test_position)]).max()),
                        "raw_missing_features": int(missing.sum()),
                        "feature_count": len(usable),
                        "history_rounds": row.get("xtour_combined_r12__n_all", np.nan),
                        "days_since_last_round": row.get("xtour_days_since_last_round", np.nan),
                        "prior_event_withdrawal": row.get("prior_event_withdrawal", np.nan),
                        "fallback_reason": row.get("cross_tour_fallback_reason", ""),
                        "img_history_rounds": row.get("clean_img_warmup_history_rounds", np.nan),
                        "img_history_coverage": row.get("clean_img_warmup_history_coverage", np.nan),
                        "img_complete_rounds": row.get("clean_img_warmup_complete_rounds", np.nan),
                    }
                )
                contributions = x_test[int(test_position)] * coefficient
                top_features = np.argsort(np.abs(contributions))[-12:][::-1]
                for feature_position in top_features:
                    train_values = train_imputed[:, int(feature_position)]
                    contribution_rows.append(
                        {
                            "extreme_id": extreme_id,
                            "feature": usable[int(feature_position)],
                            "raw_value": test_imputed[int(test_position), int(feature_position)],
                            "raw_was_missing": bool(missing[int(feature_position)]),
                            "train_imputation_median": medians[int(feature_position)],
                            "train_q001": float(np.quantile(train_values, 0.001)),
                            "train_q005": float(np.quantile(train_values, 0.005)),
                            "train_q995": float(np.quantile(train_values, 0.995)),
                            "train_q999": float(np.quantile(train_values, 0.999)),
                            "standardized_value": x_test[int(test_position), int(feature_position)],
                            "coefficient": coefficient[int(feature_position)],
                            "contribution": contributions[int(feature_position)],
                        }
                    )

    folds = pd.DataFrame(result_rows)
    summary = (
        folds.groupby(["source", "family", "alpha", "method"], observed=True)
        .agg(
            folds=("cutoff", "size"),
            prediction_min=("prediction_min", "min"),
            prediction_max=("prediction_max", "max"),
            prediction_abs_max=("prediction_abs_max", "max"),
            abs_gt_5=("abs_gt_5", "sum"),
            abs_gt_10=("abs_gt_10", "sum"),
            projected_rows=("projected_rows", "sum"),
            element_clipped_rows=("element_clipped_rows", "sum"),
            radial_projected_rows=("radial_projected_rows", "sum"),
            max_test_norm_before=("test_norm_max_before", "max"),
            max_test_norm_after=("test_norm_max_after", "max"),
            max_coefficient_abs=("coefficient_abs_max", "max"),
            max_coefficient_l2=("coefficient_l2", "max"),
        )
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "support_projection_by_fold.csv"
    summary_path = output_dir / "support_projection_summary.csv"
    extreme_path = output_dir / "raw_extreme_rows.csv"
    contribution_path = output_dir / "raw_extreme_feature_contributions.csv"
    folds.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(extreme_rows).to_csv(extreme_path, index=False)
    pd.DataFrame(contribution_rows).to_csv(contribution_path, index=False)
    manifest = {
        "research_only": True,
        "market_data_read": False,
        "selection_performed": False,
        "strict_prior_training_support": True,
        "input": str(augmented_path.resolve()),
        "input_sha256": _sha256(augmented_path),
        "builder_source_sha256": _sha256(Path(builder.__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.perf_counter() - started,
        "methods": list(methods),
        "specifications": [list(specification) for specification in specifications],
        "outputs": {},
    }
    manifest_path = output_dir / "manifest.json"
    for path in (fold_path, summary_path, extreme_path, contribution_path):
        manifest["outputs"][path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(summary.to_string(index=False))
    print(f"manifest={manifest_path} sha256={_sha256(manifest_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", type=Path, default=DEFAULT_AUGMENTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.augmented, arguments.output_dir)

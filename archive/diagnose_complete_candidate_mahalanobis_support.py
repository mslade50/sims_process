"""No-market shrinkage-Mahalanobis support diagnostic.

The only support information used for a test fold is the strict-prior training
covariate distribution.  Betting data and test outcomes are never read.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import build_complete_base_skill_candidates as builder


AUGMENTED = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
    "model_completion_v2_provenance/augmented_feature_snapshots.parquet"
)
OUTPUT = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
    "model_completion_v3_mahalanobis_support_diagnostic"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate(source: str, family: str) -> builder.CandidateSpec:
    alpha = 1.0 if family == "ridge" else 10.0
    return builder.CandidateSpec(
        name=f"diagnostic__{family}__{source}__a{alpha:g}",
        family=family,
        source=source,
        weather_arm="w0",
        target_arm="observed",
        refit_schedule="quarterly",
        alpha=alpha,
        description="strict-prior numerical support diagnostic",
    )


def prediction_metrics(
    prediction: np.ndarray,
    raw_prediction: np.ndarray,
) -> dict[str, float | int]:
    return {
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_abs_max": float(np.abs(prediction).max()),
        "abs_gt_5": int((np.abs(prediction) > 5).sum()),
        "abs_gt_8": int((np.abs(prediction) > 8).sum()),
        "abs_gt_10": int((np.abs(prediction) > 10).sum()),
        "changed_rows": int(
            np.count_nonzero(
                ~np.isclose(prediction, raw_prediction, rtol=0, atol=1e-12)
            )
        ),
    }


def run() -> None:
    started = time.perf_counter()
    snapshots = pd.read_parquet(AUGMENTED)
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
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    for source in ("total", "total_cat_img"):
        base_candidate = candidate(source, "ridge")
        features = builder.candidate_feature_columns(snapshots, base_candidate)
        target, count = builder._target_columns(base_candidate)
        for cutoff, test in tests.groupby("_refit_cutoff", sort=True, observed=True):
            cutoff = pd.Timestamp(cutoff).normalize()
            train = snapshots.loc[
                snapshots["event_end_date"].lt(cutoff)
                & snapshots[target].notna()
                & pd.to_numeric(snapshots[count], errors="coerce").ge(2)
            ].copy()
            if len(train) < 500 or not train["event_end_date"].max() < cutoff:
                raise RuntimeError(f"invalid strict-prior fold {cutoff}")
            usable = [column for column in features if train[column].notna().any()]
            imputer = SimpleImputer(strategy="median", add_indicator=False)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(train[usable]))
            x_test = scaler.transform(imputer.transform(test[usable]))
            y_train = pd.to_numeric(train[target], errors="raise").to_numpy(float)
            weights = builder._event_equal_weights(train)

            covariance_started = time.perf_counter()
            covariance = LedoitWolf(assume_centered=False).fit(x_train)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance.covariance_)
            if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
                raise RuntimeError(f"nonpositive shrinkage covariance {source} {cutoff}")
            root = np.sqrt(eigenvalues)
            location = np.asarray(covariance.location_, dtype=float)
            train_pc = ((x_train - location) @ eigenvectors) / root
            test_pc = ((x_test - location) @ eigenvectors) / root
            train_norm = np.linalg.norm(train_pc, axis=1)
            raw_test_norm = np.linalg.norm(test_pc, axis=1)
            radius = float(np.quantile(train_norm, 0.995))

            radial_pc = test_pc.copy()
            radial_scale = np.minimum(
                1.0, radius / np.maximum(np.linalg.norm(radial_pc, axis=1), 1e-12)
            )
            radial_pc *= radial_scale[:, None]
            radial_x = (radial_pc * root) @ eigenvectors.T + location

            lower, upper = np.quantile(train_pc, [0.005, 0.995], axis=0)
            pc_capped = np.minimum(np.maximum(test_pc, lower[None, :]), upper[None, :])
            capped_norm = np.linalg.norm(pc_capped, axis=1)
            capped_scale = np.minimum(1.0, radius / np.maximum(capped_norm, 1e-12))
            pc_capped *= capped_scale[:, None]
            capped_x = (pc_capped * root) @ eigenvectors.T + location
            support_rows.append(
                {
                    "source": source,
                    "cutoff": cutoff,
                    "training_rows": len(train),
                    "test_rows": len(test),
                    "features": len(usable),
                    "training_max_event_end": train["event_end_date"].max(),
                    "ledoit_wolf_shrinkage": float(covariance.shrinkage_),
                    "covariance_min_eigenvalue": float(eigenvalues.min()),
                    "covariance_max_eigenvalue": float(eigenvalues.max()),
                    "training_whitened_norm_q995": radius,
                    "test_whitened_norm_max": float(raw_test_norm.max(initial=0.0)),
                    "radial_projected_rows": int((radial_scale < 1.0).sum()),
                    "pc_capped_rows": int(
                        np.any((test_pc < lower[None, :]) | (test_pc > upper[None, :]), axis=1).sum()
                    ),
                    "pc_cap_plus_radial_rows": int(
                        np.any(
                            (test_pc < lower[None, :])
                            | (test_pc > upper[None, :])
                            | (capped_scale[:, None] < 1.0),
                            axis=1,
                        ).sum()
                    ),
                    "covariance_seconds": time.perf_counter() - covariance_started,
                }
            )

            for family in ("ridge", "robust_irls"):
                spec = candidate(source, family)
                model: Any
                if family == "ridge":
                    model = Ridge(alpha=1.0).fit(
                        x_train, y_train, sample_weight=weights
                    )
                else:
                    model = builder._fit_regression(spec, x_train, y_train, weights)
                raw_prediction = np.asarray(model.predict(x_test), dtype=float)
                training_prediction = np.asarray(model.predict(x_train), dtype=float)
                median = float(np.median(training_prediction))
                mad = float(np.median(np.abs(training_prediction - median)))
                methods = {
                    "raw": raw_prediction,
                    "mahalanobis_radial995": np.asarray(model.predict(radial_x), dtype=float),
                    "mahalanobis_pc005_radial995": np.asarray(
                        model.predict(capped_x), dtype=float
                    ),
                    "training_prediction_mad6": np.clip(
                        raw_prediction, median - 6.0 * mad, median + 6.0 * mad
                    ),
                    "training_prediction_sigma6": np.clip(
                        raw_prediction,
                        median - 6.0 * 1.4826 * mad,
                        median + 6.0 * 1.4826 * mad,
                    ),
                }
                for method, prediction in methods.items():
                    if not np.isfinite(prediction).all():
                        raise RuntimeError(f"nonfinite {source} {family} {cutoff} {method}")
                    rows.append(
                        {
                            "source": source,
                            "family": family,
                            "alpha": spec.alpha,
                            "cutoff": cutoff,
                            "method": method,
                            "training_rows": len(train),
                            "test_rows": len(test),
                            "features": len(usable),
                            "training_prediction_median": median,
                            "training_prediction_mad": mad,
                            **prediction_metrics(prediction, raw_prediction),
                        }
                    )
            print(f"[mahalanobis] {source} {cutoff.date()} complete", flush=True)

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["source", "family", "alpha", "method"], observed=True)
        .agg(
            folds=("cutoff", "size"),
            rows=("test_rows", "sum"),
            prediction_min=("prediction_min", "min"),
            prediction_max=("prediction_max", "max"),
            prediction_abs_max=("prediction_abs_max", "max"),
            abs_gt_5=("abs_gt_5", "sum"),
            abs_gt_8=("abs_gt_8", "sum"),
            abs_gt_10=("abs_gt_10", "sum"),
            changed_rows=("changed_rows", "sum"),
        )
        .reset_index()
    )
    support = pd.DataFrame(support_rows)
    support_summary = (
        support.groupby("source", observed=True)
        .agg(
            folds=("cutoff", "size"),
            rows=("test_rows", "sum"),
            radial_projected_rows=("radial_projected_rows", "sum"),
            pc_capped_rows=("pc_capped_rows", "sum"),
            pc_cap_plus_radial_rows=("pc_cap_plus_radial_rows", "sum"),
            max_test_whitened_norm=("test_whitened_norm_max", "max"),
            max_training_radius=("training_whitened_norm_q995", "max"),
            max_shrinkage=("ledoit_wolf_shrinkage", "max"),
            min_covariance_eigenvalue=("covariance_min_eigenvalue", "min"),
            covariance_seconds=("covariance_seconds", "sum"),
        )
        .reset_index()
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "prediction_folds": OUTPUT / "prediction_methods_by_fold.csv",
        "prediction_summary": OUTPUT / "prediction_methods_summary.csv",
        "support_folds": OUTPUT / "mahalanobis_support_by_fold.csv",
        "support_summary": OUTPUT / "mahalanobis_support_summary.csv",
    }
    folds.to_csv(paths["prediction_folds"], index=False)
    summary.to_csv(paths["prediction_summary"], index=False)
    support.to_csv(paths["support_folds"], index=False)
    support_summary.to_csv(paths["support_summary"], index=False)
    manifest = {
        "research_only": True,
        "market_data_read": False,
        "selection_performed": False,
        "strict_prior_training_support": True,
        "input": str(AUGMENTED.resolve()),
        "input_sha256": sha256(AUGMENTED),
        "builder_source_sha256": sha256(Path(builder.__file__)),
        "covariance": "sklearn LedoitWolf on imputed/standardized training fold",
        "support_methods": [
            "whitened L2 q99.5 radial projection",
            "whitened per-PC q0.5/q99.5 then L2 q99.5 projection",
        ],
        "prediction_envelopes": [
            "training fitted-prediction median +/- 6 raw MAD",
            "training fitted-prediction median +/- 6 * 1.4826 MAD",
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {},
    }
    for path in paths.values():
        manifest["outputs"][path.name] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(support_summary.to_string(index=False), flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    run()

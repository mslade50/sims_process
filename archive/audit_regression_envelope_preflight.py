"""Audit the persisted all-regression envelope preflight without market data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path(
    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
    "model_completion_v3_regression_envelope_preflight"
)
EXPECTED_SOURCE_SHA256 = (
    "e15f0e83b740a05ac3ee445b20f512d721a533054c2aaa6f6d14e231477e3dfe"
)
EXPECTED_AUGMENTED_SHA256 = (
    "47d4d7580599ba0452ef53a4893802ee69f5690677c3179c9dde5ef0adf44a96"
)
EXPECTED_SUBSET_CATALOG_SHA256 = (
    "52547bbbd2793324b287f2d97aa785bc9ef6cbf92fe98a9817a7140d2dd9d4b5"
)
EXPECTED_CHALLENGERS = 88
EXPECTED_KEYS = 38_409
OVERALL_CLIP_GATE = 0.01
FOLD_CLIP_GATE = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest_path = OUTPUT / "manifest.json"
    prediction_path = OUTPUT / "player_event_predictions.parquet"
    fit_log_path = OUTPUT / "fit_log.csv"
    catalog_path = OUTPUT / "candidate_catalog.csv"
    frozen_catalog_path = OUTPUT / "candidate_catalog_frozen.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = pd.read_csv(catalog_path)
    catalog["control"] = catalog["control"].fillna(False).astype(bool)
    challengers = catalog.loc[~catalog["control"]].copy()
    challenger_names = set(challengers["name"].astype(str))
    require(len(challenger_names) == EXPECTED_CHALLENGERS, "not exactly 88 challengers")
    require(
        set(challengers["family"].astype(str)).issubset(
            {"ridge", "elastic_cd", "robust_irls"}
        ),
        "non-regression challenger entered all-regression preflight",
    )
    require(
        manifest["frozen_candidate_catalog_sha256"]
        == EXPECTED_SUBSET_CATALOG_SHA256,
        "subset catalog hash mismatch",
    )
    require(sha256(frozen_catalog_path) == EXPECTED_SUBSET_CATALOG_SHA256, "catalog bytes mismatch")

    provenance = manifest["execution_provenance"]
    require(provenance["all_unchanged"] is True, "execution inputs/source changed")
    require(provenance["source_unchanged"] is True, "source changed")
    require(provenance["source_copy_matches"] is True, "executed source copy mismatch")
    require(
        provenance["source_before"]["sha256"] == EXPECTED_SOURCE_SHA256
        and provenance["source_after"]["sha256"] == EXPECTED_SOURCE_SHA256,
        "executed source SHA mismatch",
    )
    require(
        provenance["inputs_before"]["reuse_augmented"]["sha256"]
        == EXPECTED_AUGMENTED_SHA256
        and provenance["inputs_after"]["reuse_augmented"]["sha256"]
        == EXPECTED_AUGMENTED_SHA256,
        "augmented panel SHA mismatch",
    )
    output_hashes: dict[str, str] = {}
    for name, record in manifest["outputs"].items():
        path = Path(record["path"])
        actual = sha256(path)
        require(actual == record["sha256"], f"output hash mismatch: {name}")
        output_hashes[name] = actual

    prediction_columns = [
        "candidate",
        "event_key",
        "dg_id_key",
        "event_start_date",
        "prediction_asof",
        "refit_cutoff",
        "training_max_event_end",
        "chronology_ok",
        "prediction",
        "raw_regression_component",
        "guarded_regression_component",
        "raw_regression_component_selection_eligible",
        "prediction_envelope_center",
        "prediction_envelope_weighted_mad",
        "prediction_envelope_robust_scale",
        "prediction_envelope_lower",
        "prediction_envelope_upper",
        "prediction_envelope_clipped",
        "prediction_envelope_clip_distance",
        "img_primary_weight",
        "img_primary_fallback_component",
        "img_primary_fallback_used",
    ]
    prediction = pd.read_parquet(prediction_path, columns=prediction_columns)
    prediction = prediction.loc[prediction["candidate"].isin(challenger_names)].copy()
    require(
        len(prediction) == EXPECTED_CHALLENGERS * EXPECTED_KEYS,
        "challenger prediction row count mismatch",
    )
    require(
        not prediction.duplicated(["candidate", "event_key", "dg_id_key"]).any(),
        "duplicate prediction keys",
    )
    first_name = sorted(challenger_names)[0]
    expected_keys = prediction.loc[
        prediction["candidate"].eq(first_name), ["event_key", "dg_id_key"]
    ].drop_duplicates()
    require(len(expected_keys) == EXPECTED_KEYS, "first candidate key count mismatch")
    key_codes = expected_keys.assign(_expected_key=True)
    keyed = prediction[["candidate", "event_key", "dg_id_key"]].merge(
        key_codes,
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    require(keyed["_expected_key"].fillna(False).all(), "candidate has extra key")
    key_counts = keyed.groupby("candidate", observed=True).size()
    require(
        key_counts.eq(EXPECTED_KEYS).all() and len(key_counts) == EXPECTED_CHALLENGERS,
        "candidate does not have exact expected menu",
    )

    numeric_columns = [
        "prediction",
        "raw_regression_component",
        "guarded_regression_component",
        "prediction_envelope_center",
        "prediction_envelope_weighted_mad",
        "prediction_envelope_robust_scale",
        "prediction_envelope_lower",
        "prediction_envelope_upper",
        "prediction_envelope_clip_distance",
    ]
    numeric = prediction[numeric_columns].apply(pd.to_numeric, errors="coerce")
    require(np.isfinite(numeric.to_numpy(float)).all(), "nonfinite regression prediction state")
    require(
        ~prediction["raw_regression_component_selection_eligible"]
        .fillna(True)
        .astype(bool)
        .any(),
        "raw component marked selection eligible",
    )
    expected_guard = np.clip(
        numeric["raw_regression_component"],
        numeric["prediction_envelope_lower"],
        numeric["prediction_envelope_upper"],
    )
    require(
        np.allclose(
            expected_guard,
            numeric["guarded_regression_component"],
            rtol=0,
            atol=1e-12,
        ),
        "guarded component differs from envelope clip",
    )
    expected_distance = np.abs(
        numeric["raw_regression_component"]
        - numeric["guarded_regression_component"]
    )
    require(
        np.allclose(
            expected_distance,
            numeric["prediction_envelope_clip_distance"],
            rtol=0,
            atol=1e-12,
        ),
        "clip distance mismatch",
    )
    clipped = prediction["prediction_envelope_clipped"].fillna(False).astype(bool)
    require(
        np.array_equal(clipped.to_numpy(), expected_distance.gt(0).to_numpy()),
        "clip flag mismatch",
    )
    require(prediction["chronology_ok"].fillna(False).astype(bool).all(), "chronology flag failure")
    event_start = pd.to_datetime(prediction["event_start_date"], errors="coerce")
    prediction_asof = pd.to_datetime(prediction["prediction_asof"], errors="coerce")
    refit_cutoff = pd.to_datetime(prediction["refit_cutoff"], errors="coerce")
    training_max = pd.to_datetime(prediction["training_max_event_end"], errors="coerce")
    require((prediction_asof <= event_start).all(), "prediction as-of chronology failure")
    require((refit_cutoff <= event_start).all(), "refit cutoff chronology failure")
    require((training_max < refit_cutoff).all(), "training event chronology failure")

    source_by_candidate = challengers.set_index("name")["source"].astype(str)
    img_mask = prediction["candidate"].map(source_by_candidate).eq("img_primary")
    non_img = ~img_mask
    require(
        np.allclose(
            numeric.loc[non_img, "prediction"],
            numeric.loc[non_img, "guarded_regression_component"],
            rtol=0,
            atol=1e-12,
        ),
        "non-IMG final differs from guard",
    )
    img_weight = pd.to_numeric(prediction.loc[img_mask, "img_primary_weight"], errors="coerce")
    img_fallback = pd.to_numeric(
        prediction.loc[img_mask, "img_primary_fallback_component"], errors="coerce"
    )
    require(
        np.isfinite(img_weight).all()
        and np.isfinite(img_fallback).all()
        and img_weight.between(0, 1, inclusive="both").all(),
        "IMG blend state invalid",
    )
    expected_img_final = (
        img_weight.to_numpy(float)
        * numeric.loc[img_mask, "guarded_regression_component"].to_numpy(float)
        + (1.0 - img_weight.to_numpy(float)) * img_fallback.to_numpy(float)
    )
    require(
        np.allclose(
            numeric.loc[img_mask, "prediction"],
            expected_img_final,
            rtol=0,
            atol=1e-12,
        ),
        "IMG guard-then-fallback blend failure",
    )

    prediction["prediction_envelope_clipped"] = clipped
    folds = (
        prediction.groupby(["candidate", "refit_cutoff"], observed=True)
        .agg(
            rows=("prediction", "size"),
            clips=("prediction_envelope_clipped", "sum"),
            raw_min=("raw_regression_component", "min"),
            raw_max=("raw_regression_component", "max"),
            guarded_min=("guarded_regression_component", "min"),
            guarded_max=("guarded_regression_component", "max"),
            final_min=("prediction", "min"),
            final_max=("prediction", "max"),
        )
        .reset_index()
    )
    folds["clip_rate"] = folds["clips"] / folds["rows"]
    worst_positions = folds.groupby("candidate", observed=True)["clip_rate"].idxmax()
    worst = folds.loc[
        worst_positions,
        ["candidate", "refit_cutoff", "rows", "clips", "clip_rate"],
    ].rename(
        columns={
            "refit_cutoff": "worst_fold",
            "rows": "worst_fold_rows",
            "clips": "worst_fold_clips",
            "clip_rate": "worst_fold_clip_rate",
        }
    )
    overall = (
        prediction.groupby("candidate", observed=True)
        .agg(
            rows=("prediction", "size"),
            clips=("prediction_envelope_clipped", "sum"),
            raw_min=("raw_regression_component", "min"),
            raw_max=("raw_regression_component", "max"),
            guarded_min=("guarded_regression_component", "min"),
            guarded_max=("guarded_regression_component", "max"),
            final_min=("prediction", "min"),
            final_max=("prediction", "max"),
            max_overshoot=("prediction_envelope_clip_distance", "max"),
        )
        .reset_index()
    )
    overall["overall_clip_rate"] = overall["clips"] / overall["rows"]
    overall = overall.merge(worst, on="candidate", how="left", validate="one_to_one")
    overall["fails_overall_gate"] = overall["overall_clip_rate"] > OVERALL_CLIP_GATE
    overall["fails_fold_gate"] = overall["worst_fold_clip_rate"] > FOLD_CLIP_GATE
    overall["guard_dependent"] = overall[
        ["fails_overall_gate", "fails_fold_gate"]
    ].any(axis=1)
    exclusion_names = sorted(overall.loc[overall["guard_dependent"], "candidate"])

    fit_log = pd.read_csv(fit_log_path)
    fit_log["refit_cutoff"] = pd.to_datetime(fit_log["refit_cutoff"], errors="coerce")
    fit_log["training_max_event_end"] = pd.to_datetime(
        fit_log["training_max_event_end"], errors="coerce"
    )
    require(fit_log["status"].eq("fit").all(), "non-fit regression fold")
    require(fit_log["converged"].fillna(False).astype(bool).all(), "nonconverged fold")
    require(fit_log["numerical_finite"].fillna(False).astype(bool).all(), "nonfinite fit telemetry")
    require(fit_log["convergence_warning_count"].fillna(-1).eq(0).all(), "fit warning count nonzero")
    require(
        (fit_log["training_max_event_end"] < fit_log["refit_cutoff"]).all(),
        "fit-log chronology failure",
    )
    for column in ("fit_seconds", "training_wmse", "max_abs_coefficient"):
        require(np.isfinite(pd.to_numeric(fit_log[column], errors="coerce")).all(), f"nonfinite {column}")
    elastic = fit_log.loc[fit_log["candidate"].str.contains("__elastic_cd__")].copy()
    require(len(elastic) == 108, "elastic fit count is not 108")
    require(pd.to_numeric(elastic["iterations"], errors="coerce").lt(50_000).all(), "elastic iteration budget failure")
    require(np.isfinite(pd.to_numeric(elastic["dual_gap_max"], errors="coerce")).all(), "elastic dual gap missing")
    require(pd.to_numeric(elastic["training_wmse"], errors="coerce").le(1_000).all(), "elastic WMSE gate failure")
    require(pd.to_numeric(elastic["max_abs_coefficient"], errors="coerce").le(100).all(), "elastic coefficient gate failure")

    log_clips = (
        fit_log.groupby(["candidate", "refit_cutoff"], observed=True)[
            "prediction_envelope_clip_count"
        ]
        .sum()
        .rename("log_clips")
        .reset_index()
    )
    clip_compare = folds.merge(
        log_clips, on=["candidate", "refit_cutoff"], how="left", validate="one_to_one"
    )
    require(
        clip_compare["clips"].eq(clip_compare["log_clips"]).all(),
        "fit-log clip counts differ from panel",
    )

    overall_path = OUTPUT / "envelope_admission_by_candidate.csv"
    fold_path = OUTPUT / "envelope_admission_by_fold.csv"
    overall.sort_values("candidate").to_csv(overall_path, index=False)
    folds.sort_values(["candidate", "refit_cutoff"]).to_csv(fold_path, index=False)
    audit = {
        "contract": "base_skill_regression_envelope_preflight_audit_v1",
        "research_only": True,
        "market_data_read": False,
        "selection_performed": False,
        "predeclared_gates": {
            "overall_clip_rate_max": OVERALL_CLIP_GATE,
            "any_fold_clip_rate_max": FOLD_CLIP_GATE,
        },
        "admission_passed": not exclusion_names,
        "guard_dependent_candidate_count": len(exclusion_names),
        "guard_dependent_candidates": exclusion_names,
        "overall_gate_failure_count": int(overall["fails_overall_gate"].sum()),
        "fold_gate_failure_count": int(overall["fails_fold_gate"].sum()),
        "max_overall_clip_rate": float(overall["overall_clip_rate"].max()),
        "max_fold_clip_rate": float(overall["worst_fold_clip_rate"].max()),
        "global_guarded_component_range": [
            float(overall["guarded_min"].min()),
            float(overall["guarded_max"].max()),
        ],
        "global_final_prediction_range": [
            float(overall["final_min"].min()),
            float(overall["final_max"].max()),
        ],
        "exact_menu_audit": {
            "challengers": EXPECTED_CHALLENGERS,
            "keys_per_challenger": EXPECTED_KEYS,
            "duplicate_keys": 0,
            "missing_or_extra_keys": 0,
        },
        "elastic_audit": {
            "fits": len(elastic),
            "all_converged": True,
            "warning_count": int(elastic["convergence_warning_count"].sum()),
            "max_iterations": int(elastic["iterations"].max()),
            "dual_gap_max": float(elastic["dual_gap_max"].max()),
            "training_wmse_range": [
                float(elastic["training_wmse"].min()),
                float(elastic["training_wmse"].max()),
            ],
            "max_abs_coefficient": float(elastic["max_abs_coefficient"].max()),
        },
        "chronology_violations": 0,
        "nonfinite_prediction_or_fit_values": 0,
        "execution_source_sha256": EXPECTED_SOURCE_SHA256,
        "augmented_input_sha256": EXPECTED_AUGMENTED_SHA256,
        "subset_catalog_sha256": EXPECTED_SUBSET_CATALOG_SHA256,
        "run_manifest_sha256": sha256(manifest_path),
        "run_elapsed_seconds": manifest["elapsed_seconds"],
        "run_peak_rss_gib": manifest["peak_rss_gib"],
        "run_output_hashes": output_hashes,
        "outputs": {
            overall_path.name: sha256(overall_path),
            fold_path.name: sha256(fold_path),
        },
    }
    audit_path = OUTPUT / "numerical_preflight_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"audit_sha256={sha256(audit_path)}")
    print(overall.loc[overall["guard_dependent"]].sort_values("candidate").to_string(index=False))


if __name__ == "__main__":
    main()

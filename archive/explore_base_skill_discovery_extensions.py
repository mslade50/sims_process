"""Discovery-only exploratory extensions for the completed base-skill rebuild.

This research script cannot promote a model and never reads retained rows after
2022-12-31.  It screens two post-hoc ideas before the sealed period is opened:

* retrospective re-fitting of the six current-code downstream contributions;
* one legacy-control blend diagnostic and one pre-frozen clean-model blend.

The bridge arm is explicitly non-promotable because several bridge exposures
embed full-archive current-code fits.  Its only purpose is deciding whether a
much heavier strict-point-in-time exposure rebuild is worth doing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive import backtest_base_skill_market_models as market_eval  # noqa: E402
from archive import analyze_base_skill_discovery_search_correction as search_correction  # noqa: E402
from archive import evaluate_complete_base_skill_market_candidates as discovery  # noqa: E402
from archive import research_execution_provenance as execution_provenance  # noqa: E402


ROOT = REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
DISCOVERY_DIR = ROOT / "model_eval_complete_discovery_v5_provenance"
DEFAULT_SELECTION = DISCOVERY_DIR / "selection_manifest.json"
DEFAULT_SCORECARD = DISCOVERY_DIR / "candidate_gate_scorecard.csv"
DEFAULT_OUTPUT = ROOT / "model_eval_complete_discovery_extension_v1_exploratory"
EXPECTED_SELECTION_SHA256 = "8d8ad4f3e08a13e86a3bb0eed8752bc392ab7e3df610ed9735735a6bb8fca6f6"
EXPECTED_DEPENDENCY_SHA256 = {
    "evaluate_complete_base_skill_market_candidates.py": "1818f08cd03a8bde71058223686798c5eaf4aa68f714664671e7f42eb7442c49",
    "evaluate_complete_base_skill_market.py": "92d298661c5f189703c960fcab54e2c7cda321e8d8bbc8351f3d615467374f47",
    "backtest_base_skill_market_models.py": "0870d1eb7dd9364b3e7090b2f61643ddca59e536641f6fdb18971b004f22529d",
    "historical_odds_cache.py": "58a7b774edc9a7e23aab772023846402c02f37396c1b8c7a0d3a4863feeb56c7",
    "market_backtest_scoring.py": "53681535e0d1da7bde9cccede5ac09d7943e7493160c1ac2a248a0c0bfae2608",
    "research_execution_provenance.py": "b5ed954a1c7578891614d319915848ca659ad7f2dd40fde77e146cda7eeb0034",
    "analyze_base_skill_discovery_search_correction.py": "21b3f0891a973e7a5f9c9a5e1bd85e7fc0ead2080814958fa184f622f55f2117",
}
BASELINE = "current_architecture_pt_v2_approx_control"
DYNAMIC = "complete__dynamic_kalman__xtour__w0__event_batch__q0.0005"
RICH_CLEAN = "complete__ridge__total_cat_img__w1_neutral_target__quarterly__a100"
CONTROL_BLEND = "extension__control_anchor_dynamic_kalman__ridge10"
CLEAN_BLEND = "extension__dynamic_kalman_total_cat_img_w1__ridge10"
BRIDGE_PREFIX = "extension__bridge_refit__"
BRIDGE_BASELINE = BRIDGE_PREFIX + BASELINE
DISCOVERY_END = pd.Timestamp("2022-12-31")
WARMUP_START = pd.Timestamp("2020-01-01")
ALPHAS = (1.0, 10.0, 100.0)
COMPONENTS = (
    "component_course_shape",
    "component_relative_dsle",
    "component_course_experience",
    "component_course_history",
    "component_major",
    "component_links",
)
MIN_TRAINING_EVENTS = 8
MIN_TRAINING_ROWS = 500
BLEND_ALPHA = 10.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260823


def validate_dependency_authority() -> dict[str, dict[str, Any]]:
    """Bind every scoring/search dependency to audited discovery-era bytes."""

    records: dict[str, dict[str, Any]] = {}
    for name, expected_hash in EXPECTED_DEPENDENCY_SHA256.items():
        path = (REPO / "archive" / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen dependency is missing: {path}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"frozen dependency hash changed for {name}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        records[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _authorized_path(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"authorized {label} is missing: {path}")
    if int(record.get("bytes", path.stat().st_size)) != path.stat().st_size:
        raise ValueError(f"authorized {label} byte count changed")
    if str(record.get("sha256", "")).lower() != _sha256(path):
        raise ValueError(f"authorized {label} hash changed")
    return path


def resolve_inputs(selection_path: Path, scorecard_path: Path) -> dict[str, Path]:
    if _sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise ValueError("selection manifest is not exact independently audited V5")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("complete") is not True or selection.get("research_only") is not True:
        raise ValueError("V5 selection is not complete research output")
    if selection.get("selection_uses_sealed") is not False or selection.get(
        "sealed_data_used_for_selection"
    ) is not False:
        raise ValueError("V5 selection does not exclude sealed selection")
    if selection.get("selection_window_end") != DISCOVERY_END.date().isoformat():
        raise ValueError("V5 selection cutoff changed")
    inputs = selection.get("inputs", {})
    required = {
        "predictions": "predictions",
        "candidate_catalog": "catalog",
        "candidate_manifest": "completion_manifest",
        "normalized_quotes": "quotes",
        "core_bridge": "bridge",
    }
    resolved = {
        target: _authorized_path(inputs[source], source)
        for source, target in required.items()
    }
    artifacts = {
        str(record.get("path")): record
        for record in selection.get("artifacts", [])
        if isinstance(record, Mapping)
    }
    relative_scorecard = scorecard_path.resolve().relative_to(selection_path.parent.resolve()).as_posix()
    if relative_scorecard not in artifacts:
        raise ValueError("scorecard is not V5-authorized")
    score_record = artifacts[relative_scorecard]
    if int(score_record.get("bytes", -1)) != scorecard_path.stat().st_size or str(
        score_record.get("sha256", "")
    ).lower() != _sha256(scorecard_path):
        raise ValueError("scorecard bytes/hash changed")
    resolved["scorecard"] = scorecard_path.resolve()
    resolved["selection"] = selection_path.resolve()

    completion = json.loads(resolved["completion_manifest"].read_text(encoding="utf-8"))
    snapshots_record = completion.get("outputs", {}).get("augmented_feature_snapshots.parquet")
    if not isinstance(snapshots_record, Mapping):
        raise ValueError("completion manifest lacks augmented snapshots")
    resolved["snapshots"] = _authorized_path(snapshots_record, "augmented snapshots")
    return resolved


def _event_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_key", observed=True)["event_key"].transform("size")
    return 1.0 / counts.to_numpy(float)


def validate_exact_prediction_panel(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    label: str,
) -> None:
    """Fail closed on exact candidate menus, domains, and prediction chronology."""

    required = {
        "candidate",
        "event_key",
        "dg_id_key",
        "event_start_date",
        "prediction",
        "prediction_asof",
        "chronology_ok",
        "training_max_event_end",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks prediction columns: {missing}")
    expected = {str(value) for value in candidates}
    observed = set(frame["candidate"].astype(str))
    if observed != expected:
        raise ValueError(
            f"{label} candidate family changed: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    keys = ["event_key", "dg_id_key"]
    if frame.duplicated(["candidate", *keys]).any():
        raise ValueError(f"{label} prediction keys are duplicated")
    if (
        frame["event_key"].astype("string").isna().any()
        or frame["event_key"].astype("string").str.strip().eq("").any()
        or frame["dg_id_key"].astype("string").isna().any()
        or frame["dg_id_key"].astype("string").str.strip().eq("").any()
    ):
        raise ValueError(f"{label} contains blank prediction keys")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{label} predictions are non-finite")
    if not pd.api.types.is_bool_dtype(frame["chronology_ok"].dtype):
        raise ValueError(f"{label} chronology_ok is not exact boolean")
    if not frame["chronology_ok"].all():
        raise ValueError(f"{label} contains a failed chronology flag")
    start = pd.to_datetime(frame["event_start_date"], errors="coerce").dt.normalize()
    asof = pd.to_datetime(frame["prediction_asof"], errors="coerce").dt.normalize()
    training_max = pd.to_datetime(
        frame["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    if start.isna().any() or asof.isna().any() or asof.gt(start).any():
        raise ValueError(f"{label} prediction as-of chronology is invalid")
    if training_max.notna().any() and training_max.loc[training_max.notna()].ge(
        start.loc[training_max.notna()]
    ).any():
        raise ValueError(f"{label} training chronology is invalid")
    metadata = frame.groupby(keys, observed=True).agg(
        event_dates=("event_start_date", "nunique"),
        prediction_asofs=("prediction_asof", "nunique"),
    )
    if metadata["event_dates"].ne(1).any() or metadata["prediction_asofs"].ne(1).any():
        raise ValueError(f"{label} candidate metadata differs on shared keys")
    reference_name = str(candidates[0])
    reference = pd.MultiIndex.from_frame(
        frame.loc[frame["candidate"].astype(str).eq(reference_name), keys]
    )
    for candidate in candidates[1:]:
        candidate_index = pd.MultiIndex.from_frame(
            frame.loc[frame["candidate"].astype(str).eq(str(candidate)), keys]
        )
        missing_keys = reference.difference(candidate_index)
        extra_keys = candidate_index.difference(reference)
        if len(missing_keys) or len(extra_keys):
            raise ValueError(
                f"{label} menu differs for {candidate}: "
                f"missing={len(missing_keys)}, extra={len(extra_keys)}"
            )


def ridge_identity_fit(
    x: np.ndarray,
    target_residual: np.ndarray,
    event_keys: Sequence[str],
    alpha: float,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(target_residual, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("ridge fit arrays have incompatible shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge fit arrays contain non-finite values")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError("ridge alpha must be finite and nonnegative")
    prior_value = (
        np.ones(x.shape[1], dtype=float)
        if prior is None
        else np.asarray(prior, dtype=float)
    )
    if prior_value.shape != (x.shape[1],) or not np.isfinite(prior_value).all():
        raise ValueError("ridge prior has incompatible shape or non-finite values")
    events = pd.Series(list(event_keys), dtype="string")
    if len(events) != len(x) or events.isna().any() or events.str.strip().eq("").any():
        raise ValueError("ridge event keys are incompatible or blank")
    counts = events.groupby(events, observed=True).transform("size").to_numpy(float)
    weights = 1.0 / counts
    gram = x.T @ (weights[:, None] * x) + float(alpha) * np.eye(x.shape[1])
    rhs = x.T @ (weights * y) + float(alpha) * prior_value
    return np.linalg.solve(gram, rhs)


def _eligible_training(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    count = pd.to_numeric(frame["label_dg_sg_adj_r1_r2_count"], errors="coerce")
    label = pd.to_numeric(frame["label_dg_sg_adj_r1_r2_mean"], errors="coerce")
    return frame.loc[
        frame["event_end_date"].lt(cutoff)
        & count.ge(2)
        & label.notna()
        & np.isfinite(label),
    ].copy()


def _fit_bridge_beta(train: pd.DataFrame, alpha: float) -> np.ndarray:
    residual = (
        train["label_dg_sg_adj_r1_r2_mean"].to_numpy(float)
        - train["base_prediction"].to_numpy(float)
    )
    return ridge_identity_fit(
        train.loc[:, COMPONENTS].to_numpy(float),
        residual,
        train["event_key"].astype(str),
        alpha,
    )


def choose_bridge_alpha(history: pd.DataFrame) -> tuple[float, int, dict[str, float]]:
    """Choose alpha by strict forward event validation inside prior history."""

    dates = sorted(pd.to_datetime(history["event_start_date"].dropna().unique()))
    losses = {alpha: [] for alpha in ALPHAS}
    folds = 0
    for date_value in dates:
        date = pd.Timestamp(date_value).normalize()
        train = _eligible_training(history, date)
        validation = history.loc[
            history["event_start_date"].eq(date)
            & pd.to_numeric(
                history["label_dg_sg_adj_r1_r2_count"], errors="coerce"
            ).ge(2)
            & pd.to_numeric(
                history["label_dg_sg_adj_r1_r2_mean"], errors="coerce"
            ).notna()
        ].copy()
        if (
            len(train) < MIN_TRAINING_ROWS
            or train["event_key"].nunique() < MIN_TRAINING_EVENTS
            or validation.empty
        ):
            continue
        folds += 1
        for alpha in ALPHAS:
            beta = _fit_bridge_beta(train, alpha)
            prediction = validation["base_prediction"].to_numpy(float) + (
                validation.loc[:, COMPONENTS].to_numpy(float) @ beta
            )
            error = prediction - validation["label_dg_sg_adj_r1_r2_mean"].to_numpy(float)
            validation_loss = pd.DataFrame(
                {
                    "event_key": validation["event_key"].astype(str).to_numpy(),
                    "squared_error": np.square(error),
                }
            )
            losses[alpha].extend(
                validation_loss.groupby("event_key", observed=True)[
                    "squared_error"
                ].mean().astype(float).tolist()
            )
    means = {
        str(alpha): float(np.mean(values)) if values else np.inf
        for alpha, values in losses.items()
    }
    selected = min(ALPHAS, key=lambda value: (means[str(value)], -value))
    return float(selected), int(folds), means


def build_bridge_extension(
    candidate_frame: pd.DataFrame,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for year in (2020, 2021, 2022):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        history = candidate_frame.loc[candidate_frame["event_start_date"].lt(year_start)]
        alpha, cv_folds, cv_losses = choose_bridge_alpha(history)
        for date_value, target in candidate_frame.loc[
            candidate_frame["event_start_date"].dt.year.eq(year)
        ].groupby("event_start_date", sort=True):
            date = pd.Timestamp(date_value).normalize()
            train = _eligible_training(candidate_frame, date)
            if len(train) >= MIN_TRAINING_ROWS and train["event_key"].nunique() >= MIN_TRAINING_EVENTS:
                beta = _fit_bridge_beta(train, alpha)
                source = "strict_prior_expanding_bridge_refit"
                training_max = train["event_end_date"].max()
            else:
                beta = np.ones(len(COMPONENTS), dtype=float)
                source = "identity_fallback_insufficient_prior_history"
                training_max = train["event_end_date"].max() if len(train) else pd.NaT
            piece = target[[
                "event_key",
                "dg_id_key",
                "event_start_date",
                "prediction_asof",
                "chronology_ok",
            ]].copy()
            piece["candidate"] = BRIDGE_PREFIX + candidate
            piece["prediction"] = target["base_prediction"].to_numpy(float) + (
                target.loc[:, COMPONENTS].to_numpy(float) @ beta
            )
            piece["parent_candidate"] = candidate
            piece["extension_class"] = "retrospective_bridge_refit_diagnostic"
            piece["selection_eligible"] = False
            piece["training_max_event_end"] = training_max
            piece["bridge_available"] = target["bridge_available"].to_numpy(bool)
            pieces.append(piece)
            logs.append(
                {
                    "candidate": candidate,
                    "extension_candidate": BRIDGE_PREFIX + candidate,
                    "target_event_start": date,
                    "selected_alpha": alpha,
                    "alpha_cv_folds_at_year_start": cv_folds,
                    "alpha_cv_losses": json.dumps(cv_losses, sort_keys=True),
                    "fit_source": source,
                    "training_rows": int(len(train)),
                    "training_events": int(train["event_key"].nunique()),
                    "training_max_event_end": training_max,
                    **{f"beta_{name}": float(value) for name, value in zip(COMPONENTS, beta)},
                }
            )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(logs)


def fit_blend_weight(
    anchor: np.ndarray,
    challenger: np.ndarray,
    outcome: np.ndarray,
    event_keys: Sequence[str],
    *,
    prior_weight: float,
    alpha: float = BLEND_ALPHA,
) -> float:
    anchor = np.asarray(anchor, dtype=float)
    challenger = np.asarray(challenger, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if (
        anchor.ndim != 1
        or challenger.ndim != 1
        or outcome.ndim != 1
        or len(anchor) == 0
        or len(anchor) != len(challenger)
        or len(anchor) != len(outcome)
    ):
        raise ValueError("blend fit arrays have incompatible shapes")
    if not (
        np.isfinite(anchor).all()
        and np.isfinite(challenger).all()
        and np.isfinite(outcome).all()
    ):
        raise ValueError("blend fit arrays contain non-finite values")
    if not np.isfinite(alpha) or float(alpha) < 0.0:
        raise ValueError("blend alpha must be finite and nonnegative")
    if not np.isfinite(prior_weight) or not 0.0 <= float(prior_weight) <= 1.0:
        raise ValueError("blend prior weight must be finite and in [0, 1]")
    x = challenger - anchor
    y = outcome - anchor
    events = pd.Series(list(event_keys), dtype="string")
    if len(events) != len(anchor) or events.isna().any() or events.str.strip().eq("").any():
        raise ValueError("blend event keys are incompatible or blank")
    counts = events.groupby(events, observed=True).transform("size").to_numpy(float)
    weights = 1.0 / counts
    denominator = float(np.sum(weights * np.square(x)) + alpha)
    numerator = float(np.sum(weights * x * y) + alpha * prior_weight)
    if denominator <= 0.0:
        raise ValueError("blend weight is unidentified")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def build_blend_extension(
    aligned: pd.DataFrame,
    *,
    name: str,
    anchor_column: str,
    challenger_column: str,
    prior_weight: float,
    selection_eligible: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for date_value, target in aligned.groupby("event_start_date", sort=True):
        date = pd.Timestamp(date_value).normalize()
        train = _eligible_training(aligned, date)
        if len(train) >= MIN_TRAINING_ROWS and train["event_key"].nunique() >= MIN_TRAINING_EVENTS:
            weight = fit_blend_weight(
                train[anchor_column].to_numpy(float),
                train[challenger_column].to_numpy(float),
                train["label_dg_sg_adj_r1_r2_mean"].to_numpy(float),
                train["event_key"].astype(str),
                prior_weight=prior_weight,
            )
            source = "strict_prior_event_equal_sg_label"
            training_max = train["event_end_date"].max()
        else:
            weight = float(prior_weight)
            source = "frozen_prior_weight_insufficient_history"
            training_max = train["event_end_date"].max() if len(train) else pd.NaT
        piece = target[[
            "event_key",
            "dg_id_key",
            "event_start_date",
            "prediction_asof",
            "chronology_ok",
        ]].copy()
        piece["candidate"] = name
        piece["prediction"] = target[anchor_column].to_numpy(float) + weight * (
            target[challenger_column].to_numpy(float)
            - target[anchor_column].to_numpy(float)
        )
        piece["parent_candidate"] = f"{anchor_column}|{challenger_column}"
        piece["extension_class"] = (
            "clean_challenger_blend" if selection_eligible else "legacy_control_blend_diagnostic"
        )
        piece["selection_eligible"] = bool(selection_eligible)
        piece["training_max_event_end"] = training_max
        piece["bridge_available"] = False
        pieces.append(piece)
        logs.append(
            {
                "candidate": name,
                "target_event_start": date,
                "blend_weight": weight,
                "prior_weight": prior_weight,
                "ridge_alpha": BLEND_ALPHA,
                "fit_source": source,
                "training_rows": int(len(train)),
                "training_events": int(train["event_key"].nunique()),
                "training_max_event_end": training_max,
            }
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(logs)


def score_prediction_set(
    quotes: pd.DataFrame,
    predictions: pd.DataFrame,
    baseline: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = market_eval.attach_skill_differences(quotes, predictions)
    mapper = discovery.fit_temporal_probability_mappers(
        paired,
        baseline,
        evaluation_years=discovery.EVALUATION_YEARS,
        min_pairs=discovery.FROZEN_MIN_CALIBRATION_PAIRS,
    )
    scored = discovery.score_temporal_discovery(paired, mapper)
    logical = discovery.canonical_discovery_probabilities(scored)
    proper = discovery.paired_common_proper_scores(logical, baseline)
    return scored, logical, proper, discovery.aggregate_candidate_scores(proper)


def event_loss_deltas(
    logical: pd.DataFrame,
    candidates: Sequence[str],
    baseline: str,
) -> pd.DataFrame:
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    required = {
        "candidate",
        "mapped_event_key",
        "event_start_date",
        "market",
        "pair_key",
        "canonical_outcome",
        "canonical_probability",
        "canonical_push_probability",
    }
    missing = sorted(required - set(resolved.columns))
    if missing:
        raise ValueError(f"logical probabilities lack event-delta columns: {missing}")
    numeric = resolved[
        [
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("logical probabilities contain non-finite values")
    if (
        numeric["canonical_probability"].lt(0.0).any()
        or numeric["canonical_probability"].gt(1.0).any()
        or numeric["canonical_push_probability"].lt(0.0).any()
        or numeric["canonical_push_probability"].ge(1.0).any()
    ):
        raise ValueError("logical probabilities are outside their domains")
    resolved["conditional_probability"] = (
        resolved["canonical_probability"]
        / (1.0 - resolved["canonical_push_probability"])
    ).clip(market_eval.EPS, 1.0 - market_eval.EPS)
    resolved["log_loss"] = -(
        resolved["canonical_outcome"] * np.log(resolved["conditional_probability"])
        + (1.0 - resolved["canonical_outcome"])
        * np.log(1.0 - resolved["conditional_probability"])
    )
    keys = ["mapped_event_key", "event_start_date", "market", "pair_key"]
    baseline_rows = resolved.loc[
        resolved["candidate"].eq(baseline), keys + ["canonical_outcome", "log_loss"]
    ].copy()
    if baseline_rows.empty or baseline_rows.duplicated(keys).any():
        raise ValueError("extension baseline menu is empty or duplicated")
    base = baseline_rows.rename(
        columns={
            "canonical_outcome": "baseline_outcome",
            "log_loss": "baseline_loss",
        }
    )
    baseline_index = pd.MultiIndex.from_frame(base[keys])
    pieces: list[pd.DataFrame] = []
    for candidate in candidates:
        challenger = resolved.loc[
            resolved["candidate"].eq(candidate),
            keys + ["canonical_outcome", "log_loss"],
        ].copy()
        if challenger.empty or challenger.duplicated(keys).any():
            raise ValueError(f"extension challenger menu is empty or duplicated: {candidate}")
        challenger_index = pd.MultiIndex.from_frame(challenger[keys])
        missing_keys = baseline_index.difference(challenger_index)
        extra_keys = challenger_index.difference(baseline_index)
        if len(missing_keys) or len(extra_keys):
            raise ValueError(
                f"extension menu differs from baseline: {candidate}; "
                f"missing={len(missing_keys)}, extra={len(extra_keys)}"
            )
        challenger = challenger.rename(
            columns={
                "canonical_outcome": "candidate_outcome",
                "log_loss": "candidate_loss",
            }
        )
        common = base.merge(challenger, on=keys, how="inner", validate="one_to_one")
        if not np.allclose(
            common["baseline_outcome"].to_numpy(float),
            common["candidate_outcome"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=False,
        ):
            raise ValueError(f"extension outcomes differ from baseline: {candidate}")
        common["loss_delta"] = common["candidate_loss"] - common["baseline_loss"]
        event_market = common.groupby(
            ["mapped_event_key", "event_start_date", "market"],
            observed=True,
            as_index=False,
        )["loss_delta"].mean()
        if not event_market.groupby("mapped_event_key")["market"].nunique().eq(2).all():
            raise ValueError(f"extension lacks both markets: {candidate}")
        pooled = event_market.groupby(
            ["mapped_event_key", "event_start_date"], observed=True, as_index=False
        )["loss_delta"].mean()
        pooled["candidate"] = candidate
        pieces.append(pooled)
    return pd.concat(pieces, ignore_index=True)


def year_stratified_search_inference(
    event_deltas: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"candidate", "mapped_event_key", "event_start_date", "loss_delta"}
    missing = sorted(required - set(event_deltas.columns))
    if missing:
        raise ValueError(f"event deltas lack search-inference columns: {missing}")
    if not isinstance(draws, (int, np.integer)) or int(draws) < 100:
        raise ValueError("bootstrap draws must be an integer of at least 100")
    if event_deltas.empty or event_deltas.duplicated(
        ["candidate", "mapped_event_key"]
    ).any():
        raise ValueError("extension event deltas are empty or duplicated")
    if not np.isfinite(
        pd.to_numeric(event_deltas["loss_delta"], errors="coerce").to_numpy(float)
    ).all():
        raise ValueError("extension event deltas contain non-finite values")
    candidates = sorted(event_deltas["candidate"].astype(str).unique())
    event_dates = event_deltas[["mapped_event_key", "event_start_date"]].drop_duplicates()
    if event_dates.duplicated("mapped_event_key").any():
        raise ValueError("an event key maps to multiple start dates")
    meta = event_dates
    meta = meta.sort_values(["event_start_date", "mapped_event_key"], kind="mergesort")
    events = meta["mapped_event_key"].astype(str).tolist()
    years = pd.to_datetime(meta["event_start_date"]).dt.year.to_numpy(int)
    matrix_frame = event_deltas.pivot(
        index="mapped_event_key", columns="candidate", values="loss_delta"
    ).reindex(index=events, columns=candidates)
    expected_rows = len(events) * len(candidates)
    if len(event_deltas) != expected_rows:
        raise ValueError("extension event menus are not exactly rectangular")
    matrix = matrix_frame.to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("extension event matrix is incomplete/non-finite")
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((draws, len(candidates)), dtype=float)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        weights = np.zeros((stop - start, len(events)), dtype=float)
        for year in sorted(set(years)):
            positions = np.flatnonzero(years == year)
            weights[:, positions] = rng.multinomial(
                len(positions), np.full(len(positions), 1.0 / len(positions)), size=stop - start
            )
        bootstrap[start:stop] = weights @ matrix / len(events)
    centered = bootstrap - observed
    best = int(np.argmin(observed))
    global_p = float((1 + np.sum(centered.min(axis=1) <= observed[best])) / (draws + 1))
    upper_increment = float(
        np.quantile(np.max(-centered, axis=1), 0.95, method="higher")
    )
    rows = []
    for index, candidate in enumerate(candidates):
        sample = bootstrap[:, index]
        rows.append(
            {
                "candidate": candidate,
                "observed_pooled_log_loss_delta": float(observed[index]),
                "ci_low": float(np.quantile(sample, 0.025)),
                "ci_high": float(np.quantile(sample, 0.975)),
                "null_centered_one_sided_p_improvement": float(
                    (1 + np.sum(centered[:, index] <= observed[index])) / (draws + 1)
                ),
                "simultaneous_95pct_upper_bound": float(observed[index] + upper_increment),
                "familywise_improvement_certified": bool(observed[index] + upper_increment < 0.0),
            }
        )
    audit = {
        "events": len(events),
        "events_by_year": {
            str(year): int(np.sum(years == year)) for year in sorted(set(years))
        },
        "candidates": len(candidates),
        "draws": draws,
        "seed": seed,
        "best_candidate": candidates[best],
        "best_observed_pooled_log_loss_delta": float(observed[best]),
        "year_stratified_reality_check_one_sided_p": global_p,
        "simultaneous_95pct_upper_increment": upper_increment,
        "any_familywise_improvement_certified": bool(
            any(row["familywise_improvement_certified"] for row in rows)
        ),
    }
    return pd.DataFrame(rows), audit


def comparison_hypothesis(candidate: str, baseline: str) -> str:
    return f"{candidate}___versus___{baseline}"


def relabel_comparison_deltas(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    output = frame.copy()
    names = set(output["candidate"].astype(str))
    if names != {candidate}:
        raise ValueError(
            f"comparison delta relabel expected only {candidate!r}, found {sorted(names)}"
        )
    output["model_candidate"] = candidate
    output["model_baseline"] = baseline
    output["candidate"] = comparison_hypothesis(candidate, baseline)
    return output


def inference_certified_under_all_resamples(
    inference: pd.DataFrame,
    audit: Mapping[str, Any],
    hypothesis: str,
) -> bool:
    row = inference.loc[inference["candidate"].eq(hypothesis)]
    if len(row) != 1 or not bool(row.iloc[0]["familywise_improvement_certified"]):
        return False
    for section in (
        "moving_event_block_sensitivity",
        "calendar_week_block_sensitivity",
    ):
        block_audits = audit.get(section, {})
        if set(block_audits) != {"1", "2", "3", "4"}:
            return False
        for block in block_audits.values():
            if block.get("status") != "complete":
                return False
            bounds = block.get("candidate_simultaneous_95pct_upper_bound", {})
            value = pd.to_numeric(pd.Series([bounds.get(hypothesis)]), errors="coerce").iloc[0]
            if not np.isfinite(value) or float(value) >= 0.0:
                return False
    return True


def _market_gate(proper: pd.DataFrame, candidate: str) -> bool:
    rows = proper.loc[proper["candidate"].eq(candidate)]
    numeric_columns = [
        "coverage_ratio_vs_baseline",
        "log_loss_delta",
        "brier_delta",
        "candidate_calibration_intercept",
        "baseline_calibration_intercept",
        "candidate_calibration_slope",
        "baseline_calibration_slope",
        "candidate_ece_decile",
        "baseline_ece_decile",
    ]
    return bool(
        len(rows) == 2
        and rows["market"].nunique() == 2
        and np.isfinite(rows[numeric_columns].to_numpy(float)).all()
        and rows["candidate_calibration_fit_success"].eq(True).all()
        and rows["baseline_calibration_fit_success"].eq(True).all()
        and rows["coverage_ratio_vs_baseline"].eq(1.0).all()
        and rows["log_loss_delta"].lt(0.0).all()
        and rows["brier_delta"].le(0.0).all()
        and (
            rows["candidate_calibration_intercept"].abs()
            <= rows["baseline_calibration_intercept"].abs()
        ).all()
        and (
            (rows["candidate_calibration_slope"] - 1.0).abs()
            <= (rows["baseline_calibration_slope"] - 1.0).abs()
        ).all()
        and rows["candidate_ece_decile"].le(rows["baseline_ece_decile"]).all()
    )


def _guardrail_gate(row: Mapping[str, Any]) -> bool:
    """Apply the frozen discovery money/CLV/edge/concentration no-worse gates."""

    required = {
        "close_pnl_delta": ("ge", 0.0),
        "close_roi_delta": ("ge", 0.0),
        "candidate_mean_open_to_close_clv": ("ge", 0.0),
        "open_to_close_clv_delta": ("ge", 0.0),
        "edge_violation_magnitude_delta": ("le", 0.0),
        "max_drawdown_delta": ("le", 0.0),
        "worst_event_loss_delta": ("le", 0.0),
    }
    for column, (direction, threshold) in required.items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            return False
        if direction == "ge" and float(value) < threshold:
            return False
        if direction == "le" and float(value) > threshold:
            return False
    return True


def run(args: argparse.Namespace) -> None:
    dependency_authority = validate_dependency_authority()
    inputs = resolve_inputs(args.selection.resolve(), args.scorecard.resolve())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"extension output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = execution_provenance.begin(
        output_dir=output,
        primary_source=Path(__file__),
        input_paths=list(inputs.values()),
        code_paths=[Path(record["path"]) for record in dependency_authority.values()],
    )

    scorecard = pd.read_csv(inputs["scorecard"])
    candidate_names = sorted(scorecard["candidate"].astype(str).unique())
    if len(candidate_names) != 57 or DYNAMIC not in candidate_names or RICH_CLEAN not in candidate_names:
        raise ValueError("V5 eligible extension family changed")
    requested = candidate_names + [BASELINE]
    prediction_columns = [
        "candidate",
        "event_key",
        "dg_id_key",
        "event_start_date",
        "prediction",
        "prediction_asof",
        "chronology_ok",
        "training_max_event_end",
    ]
    predictions = pd.read_parquet(
        inputs["predictions"],
        columns=prediction_columns,
        filters=[
            ("candidate", "in", requested),
            ("event_start_date", ">=", WARMUP_START),
            ("event_start_date", "<=", DISCOVERY_END),
        ],
    )
    predictions["event_start_date"] = pd.to_datetime(predictions["event_start_date"]).dt.normalize()
    predictions["prediction_asof"] = pd.to_datetime(predictions["prediction_asof"]).dt.normalize()
    predictions["training_max_event_end"] = pd.to_datetime(
        predictions["training_max_event_end"], errors="coerce"
    ).dt.normalize()
    if predictions["event_start_date"].gt(DISCOVERY_END).any():
        raise ValueError("sealed prediction retained")
    validate_exact_prediction_panel(
        predictions, requested, label="V5 extension parents"
    )

    snapshots = pd.read_parquet(
        inputs["snapshots"],
        columns=[
            "event_key",
            "dg_id_key",
            "event_start_date",
            "event_end_date",
            "label_dg_sg_adj_r1_r2_mean",
            "label_dg_sg_adj_r1_r2_count",
        ],
        filters=[
            ("event_start_date", ">=", WARMUP_START),
            ("event_start_date", "<=", DISCOVERY_END),
        ],
    )
    snapshots["event_start_date"] = pd.to_datetime(snapshots["event_start_date"]).dt.normalize()
    snapshots["event_end_date"] = pd.to_datetime(snapshots["event_end_date"]).dt.normalize()
    if snapshots["event_start_date"].gt(DISCOVERY_END).any():
        raise ValueError("sealed label retained")
    if snapshots.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("augmented snapshot keys are duplicated")
    snapshot_membership = predictions[
        ["event_key", "dg_id_key", "event_start_date"]
    ].drop_duplicates().merge(
        snapshots[["event_key", "dg_id_key", "event_start_date"]].rename(
            columns={"event_start_date": "snapshot_event_start_date"}
        ),
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    if snapshot_membership["snapshot_event_start_date"].isna().any():
        raise ValueError("a V5 prediction key is absent from augmented snapshots")
    if not snapshot_membership["event_start_date"].eq(
        snapshot_membership["snapshot_event_start_date"]
    ).all():
        raise ValueError("prediction and snapshot event dates disagree")
    bridge = pd.read_parquet(
        inputs["bridge"],
        columns=["event_start", "dg_id", *COMPONENTS],
        filters=[
            ("event_start", ">=", WARMUP_START),
            ("event_start", "<=", DISCOVERY_END),
        ],
    )
    bridge["event_start_date"] = pd.to_datetime(bridge.pop("event_start")).dt.normalize()
    bridge["dg_id_key"] = bridge.pop("dg_id").map(market_eval._stable_id)
    if bridge["dg_id_key"].eq("").any():
        raise ValueError("bridge contains a blank normalized player ID")
    if bridge.duplicated(["event_start_date", "dg_id_key"]).any():
        raise ValueError("bridge event/player keys are duplicated")

    base = predictions.merge(
        snapshots.drop(columns="event_start_date"),
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
    ).merge(
        bridge,
        on=["event_start_date", "dg_id_key"],
        how="left",
        validate="many_to_one",
        indicator="_bridge",
    )
    base["bridge_available"] = base["_bridge"].eq("both")
    base.drop(columns="_bridge", inplace=True)
    base.rename(columns={"prediction": "base_prediction"}, inplace=True)
    base.loc[:, COMPONENTS] = base.loc[:, COMPONENTS].fillna(0.0)
    if not np.isfinite(base.loc[:, COMPONENTS].to_numpy(float)).all():
        raise ValueError("bridge components are non-finite")

    bridge_names = [BRIDGE_PREFIX + name for name in requested]
    extension_parts: list[pd.DataFrame] = []
    fit_logs: list[pd.DataFrame] = []
    for candidate in requested:
        extension, log = build_bridge_extension(
            base.loc[base["candidate"].eq(candidate)].copy(), candidate
        )
        extension_parts.append(extension)
        fit_logs.append(log)

    key_columns = [
        "event_key",
        "dg_id_key",
        "event_start_date",
        "prediction_asof",
        "chronology_ok",
    ]
    wide = predictions.loc[predictions["candidate"].isin([BASELINE, DYNAMIC, RICH_CLEAN])].pivot(
        index=key_columns,
        columns="candidate",
        values="prediction",
    ).reset_index()
    wide = wide.merge(
        snapshots.drop(columns="event_start_date"),
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    control_blend, control_log = build_blend_extension(
        wide,
        name=CONTROL_BLEND,
        anchor_column=BASELINE,
        challenger_column=DYNAMIC,
        prior_weight=0.0,
        selection_eligible=False,
    )
    clean_blend, clean_log = build_blend_extension(
        wide,
        name=CLEAN_BLEND,
        anchor_column=DYNAMIC,
        challenger_column=RICH_CLEAN,
        prior_weight=0.5,
        selection_eligible=True,
    )
    extension_parts.extend([control_blend, clean_blend])
    fit_logs.extend([control_log, clean_log])
    extensions = pd.concat(extension_parts, ignore_index=True)
    logs = pd.concat(fit_logs, ignore_index=True, sort=False)
    if extensions["event_start_date"].gt(DISCOVERY_END).any():
        raise ValueError("extension retained a sealed row")
    extension_names = [*bridge_names, CONTROL_BLEND, CLEAN_BLEND]
    validate_exact_prediction_panel(
        extensions, extension_names, label="generated discovery extensions"
    )

    quotes = pd.read_parquet(
        inputs["quotes"],
        filters=[
            ("event_start_date", ">=", pd.Timestamp("2020-01-01")),
            ("event_start_date", "<=", DISCOVERY_END),
        ],
    )
    quotes["event_start_date"] = pd.to_datetime(quotes["event_start_date"]).dt.normalize()
    if quotes["event_start_date"].gt(DISCOVERY_END).any():
        raise ValueError("sealed quote retained")
    market_eval.validate_normalized_matchup_quotes(quotes)

    bridge_predictions = extensions.loc[
        extensions["candidate"].isin(bridge_names),
        ["candidate", "event_key", "dg_id_key", "prediction"],
    ]
    raw_bridge_parents = predictions.loc[
        predictions["candidate"].isin(requested),
        ["candidate", "event_key", "dg_id_key", "prediction"],
    ]
    bridge_comparison_predictions = pd.concat(
        [raw_bridge_parents, bridge_predictions], ignore_index=True
    )
    (
        bridge_scored,
        bridge_logical,
        bridge_vs_raw_proper,
        bridge_vs_raw_aggregate,
    ) = score_prediction_set(quotes, bridge_comparison_predictions, BASELINE)
    if bridge_scored["probability_calibration_source"].astype(str).str.contains(
        "fallback", case=False, regex=False
    ).any():
        raise ValueError("bridge comparison unexpectedly used a probability fallback")
    bridge_vs_rebuilt_proper = discovery.paired_common_proper_scores(
        bridge_logical, BRIDGE_BASELINE
    )
    bridge_vs_rebuilt_aggregate = discovery.aggregate_candidate_scores(
        bridge_vs_rebuilt_proper
    )
    bridge_own_parent_parts: list[pd.DataFrame] = []
    for parent in candidate_names:
        extension_name = BRIDGE_PREFIX + parent
        pair = bridge_logical.loc[
            bridge_logical["candidate"].isin([parent, extension_name])
        ]
        proper = discovery.paired_common_proper_scores(pair, parent)
        proper = proper.loc[proper["candidate"].eq(extension_name)].copy()
        proper["comparison_baseline"] = parent
        bridge_own_parent_parts.append(proper)
    bridge_vs_own_parent_proper = pd.concat(
        bridge_own_parent_parts, ignore_index=True
    )
    raw_parents = predictions.loc[
        predictions["candidate"].isin([BASELINE, DYNAMIC, RICH_CLEAN]),
        ["candidate", "event_key", "dg_id_key", "prediction"],
    ]
    blend_predictions = pd.concat(
        [
            raw_parents,
            extensions.loc[extensions["candidate"].isin([CONTROL_BLEND, CLEAN_BLEND]), [
                "candidate", "event_key", "dg_id_key", "prediction"
            ]],
        ],
        ignore_index=True,
    )
    blend_scored, blend_logical, blend_proper, blend_aggregate = score_prediction_set(
        quotes, blend_predictions, BASELINE
    )
    if blend_scored["probability_calibration_source"].astype(str).str.contains(
        "fallback", case=False, regex=False
    ).any():
        raise ValueError("clean blend comparison unexpectedly used a probability fallback")
    clean_vs_dynamic = discovery.paired_common_proper_scores(
        blend_logical.loc[
            blend_logical["candidate"].isin([DYNAMIC, CLEAN_BLEND])
        ],
        DYNAMIC,
    )
    clean_vs_rich = discovery.paired_common_proper_scores(
        blend_logical.loc[
            blend_logical["candidate"].isin([RICH_CLEAN, CLEAN_BLEND])
        ],
        RICH_CLEAN,
    )

    bridge_extensions = [name for name in bridge_names if name != BRIDGE_BASELINE]
    bridge_delta_parts: list[pd.DataFrame] = []
    rebuilt_control_deltas = event_loss_deltas(
        bridge_logical, bridge_extensions, BRIDGE_BASELINE
    )
    raw_control_deltas = event_loss_deltas(bridge_logical, bridge_names, BASELINE)
    for candidate in bridge_extensions:
        bridge_delta_parts.append(
            relabel_comparison_deltas(
                rebuilt_control_deltas.loc[
                    rebuilt_control_deltas["candidate"].eq(candidate)
                ],
                candidate=candidate,
                baseline=BRIDGE_BASELINE,
            )
        )
    for candidate in bridge_names:
        bridge_delta_parts.append(
            relabel_comparison_deltas(
                raw_control_deltas.loc[raw_control_deltas["candidate"].eq(candidate)],
                candidate=candidate,
                baseline=BASELINE,
            )
        )
    for parent in candidate_names:
        candidate = BRIDGE_PREFIX + parent
        own = event_loss_deltas(
            bridge_logical.loc[
                bridge_logical["candidate"].isin([parent, candidate])
            ],
            [candidate],
            parent,
        )
        bridge_delta_parts.append(
            relabel_comparison_deltas(own, candidate=candidate, baseline=parent)
        )
    bridge_event_deltas = pd.concat(bridge_delta_parts, ignore_index=True)
    bridge_inference, bridge_inference_audit = search_correction.search_corrected_inference(
        bridge_event_deltas,
        draws=args.bootstrap_draws,
        seed=args.seed + 101,
        block_lengths=(1, 2, 3, 4),
    )
    bridge_year_inference, bridge_year_inference_audit = year_stratified_search_inference(
        bridge_event_deltas, draws=args.bootstrap_draws, seed=args.seed + 102
    )

    clean_delta_specs = [
        (CONTROL_BLEND, BASELINE),
        (CLEAN_BLEND, BASELINE),
        (CLEAN_BLEND, DYNAMIC),
        (CLEAN_BLEND, RICH_CLEAN),
    ]
    clean_delta_parts: list[pd.DataFrame] = []
    for candidate, baseline in clean_delta_specs:
        pair = blend_logical.loc[
            blend_logical["candidate"].isin([baseline, candidate])
        ]
        delta = event_loss_deltas(pair, [candidate], baseline)
        clean_delta_parts.append(
            relabel_comparison_deltas(
                delta, candidate=candidate, baseline=baseline
            )
        )
    clean_event_deltas = pd.concat(clean_delta_parts, ignore_index=True)
    clean_inference, clean_inference_audit = search_correction.search_corrected_inference(
        clean_event_deltas,
        draws=args.bootstrap_draws,
        seed=args.seed,
        block_lengths=(1, 2, 3, 4),
    )
    clean_year_inference, clean_year_inference_audit = year_stratified_search_inference(
        clean_event_deltas, draws=args.bootstrap_draws, seed=args.seed + 1
    )

    money_rows: list[dict[str, Any]] = []
    for candidate in bridge_names:
        baselines = [BASELINE]
        if candidate != BRIDGE_BASELINE:
            baselines.extend([BRIDGE_BASELINE, candidate.removeprefix(BRIDGE_PREFIX)])
        for baseline in baselines:
            row = discovery.discovery_guardrails(bridge_scored, candidate, baseline)
            row["hypothesis"] = comparison_hypothesis(candidate, baseline)
            row["extension_family"] = "retrospective_bridge_refit_diagnostic"
            row["guardrail_gate_pass"] = _guardrail_gate(row)
            money_rows.append(row)
    for candidate, baseline in clean_delta_specs:
        row = discovery.discovery_guardrails(blend_scored, candidate, baseline)
        row["hypothesis"] = comparison_hypothesis(candidate, baseline)
        row["extension_family"] = (
            "clean_challenger_blend"
            if candidate == CLEAN_BLEND
            else "legacy_control_blend_diagnostic"
        )
        row["guardrail_gate_pass"] = _guardrail_gate(row)
        money_rows.append(row)
    money = pd.DataFrame(money_rows)

    def proper_for_baseline(
        proper: pd.DataFrame, candidate: str, baseline: str
    ) -> bool:
        subset = proper.loc[proper["baseline"].eq(baseline)]
        return _market_gate(subset, candidate)

    def year_certified(hypothesis: str, frame: pd.DataFrame) -> bool:
        row = frame.loc[frame["candidate"].eq(hypothesis)]
        return bool(
            len(row) == 1
            and bool(row.iloc[0]["familywise_improvement_certified"])
        )

    def money_certified(hypotheses: Sequence[str]) -> bool:
        rows = money.loc[money["hypothesis"].isin(hypotheses)]
        return bool(
            len(rows) == len(hypotheses)
            and rows["hypothesis"].nunique() == len(hypotheses)
            and rows["guardrail_gate_pass"].eq(True).all()
        )

    gate_rows: list[dict[str, Any]] = []
    for candidate in bridge_extensions:
        parent = candidate.removeprefix(BRIDGE_PREFIX)
        hypotheses = [
            comparison_hypothesis(candidate, BRIDGE_BASELINE),
            comparison_hypothesis(candidate, BASELINE),
            comparison_hypothesis(candidate, parent),
        ]
        proper_pass = (
            proper_for_baseline(bridge_vs_rebuilt_proper, candidate, BRIDGE_BASELINE)
            and proper_for_baseline(bridge_vs_raw_proper, candidate, BASELINE)
            and proper_for_baseline(bridge_vs_own_parent_proper, candidate, parent)
        )
        search_pass = all(
            inference_certified_under_all_resamples(
                bridge_inference, bridge_inference_audit, hypothesis
            )
            and year_certified(hypothesis, bridge_year_inference)
            for hypothesis in hypotheses
        )
        gate_rows.append(
            {
                "candidate": candidate,
                "extension_class": "retrospective_bridge_refit_diagnostic",
                "proper_gate_all_comparators": proper_pass,
                "search_corrected_all_comparators": search_pass,
                "money_guardrail_all_comparators": money_certified(hypotheses),
                "eligible_for_sealed_confirmation": False,
                "production_promotable": False,
                "reason": (
                    "full-archive bridge exposures embed indirect sealed-period information; "
                    "diagnostic cannot affect clean selection"
                ),
            }
        )

    control_hypothesis = comparison_hypothesis(CONTROL_BLEND, BASELINE)
    gate_rows.append(
        {
            "candidate": CONTROL_BLEND,
            "extension_class": "legacy_control_blend_diagnostic",
            "proper_gate_all_comparators": proper_for_baseline(
                blend_proper, CONTROL_BLEND, BASELINE
            ),
            "search_corrected_all_comparators": (
                inference_certified_under_all_resamples(
                    clean_inference, clean_inference_audit, control_hypothesis
                )
                and year_certified(control_hypothesis, clean_year_inference)
            ),
            "money_guardrail_all_comparators": money_certified([control_hypothesis]),
            "eligible_for_sealed_confirmation": False,
            "production_promotable": False,
            "reason": "blend reintroduces legacy skill_est through its control anchor",
        }
    )

    clean_hypotheses = [
        comparison_hypothesis(CLEAN_BLEND, baseline)
        for baseline in (BASELINE, DYNAMIC, RICH_CLEAN)
    ]
    clean_proper_pass = (
        proper_for_baseline(blend_proper, CLEAN_BLEND, BASELINE)
        and proper_for_baseline(clean_vs_dynamic, CLEAN_BLEND, DYNAMIC)
        and proper_for_baseline(clean_vs_rich, CLEAN_BLEND, RICH_CLEAN)
    )
    clean_search_pass = all(
        inference_certified_under_all_resamples(
            clean_inference, clean_inference_audit, hypothesis
        )
        and year_certified(hypothesis, clean_year_inference)
        for hypothesis in clean_hypotheses
    )
    clean_money_pass = money_certified(clean_hypotheses)
    clean_advances = bool(clean_proper_pass and clean_search_pass and clean_money_pass)
    gate_rows.append(
        {
            "candidate": CLEAN_BLEND,
            "extension_class": "clean_challenger_blend",
            "proper_gate_all_comparators": clean_proper_pass,
            "search_corrected_all_comparators": clean_search_pass,
            "money_guardrail_all_comparators": clean_money_pass,
            "eligible_for_sealed_confirmation": clean_advances,
            "production_promotable": False,
            "reason": (
                "advance only as one frozen candidate to one-shot sealed confirmation"
                if clean_advances
                else "failed a frozen proper-score, search, or market-money gate"
            ),
        }
    )
    gates = pd.DataFrame(gate_rows)
    clean_sealed_candidates = [CLEAN_BLEND] if clean_advances else []

    extensions.to_parquet(output / "extension_predictions_2020_2022.parquet", index=False)
    logs.to_csv(output / "extension_fit_log.csv", index=False)
    bridge_vs_raw_proper.to_csv(
        output / "bridge_refit_scores_vs_raw_control.csv", index=False
    )
    bridge_vs_raw_aggregate.to_csv(
        output / "bridge_refit_aggregate_vs_raw_control.csv", index=False
    )
    bridge_vs_rebuilt_proper.to_csv(
        output / "bridge_refit_scores_vs_rebuilt_control.csv", index=False
    )
    bridge_vs_rebuilt_aggregate.to_csv(
        output / "bridge_refit_aggregate_vs_rebuilt_control.csv", index=False
    )
    bridge_vs_own_parent_proper.to_csv(
        output / "bridge_refit_scores_vs_own_parent.csv", index=False
    )
    blend_proper.to_csv(output / "blend_proper_scores_vs_control.csv", index=False)
    blend_aggregate.to_csv(output / "blend_aggregate_scores_vs_control.csv", index=False)
    clean_vs_dynamic.to_csv(output / "clean_blend_scores_vs_dynamic_parent.csv", index=False)
    clean_vs_rich.to_csv(output / "clean_blend_scores_vs_rich_parent.csv", index=False)
    bridge_event_deltas.to_parquet(
        output / "bridge_event_loss_deltas_retrospective.parquet", index=False
    )
    bridge_inference.to_csv(
        output / "bridge_search_corrected_inference_retrospective.csv", index=False
    )
    bridge_year_inference.to_csv(
        output / "bridge_year_stratified_inference_retrospective.csv", index=False
    )
    clean_event_deltas.to_parquet(output / "clean_event_loss_deltas.parquet", index=False)
    clean_inference.to_csv(output / "clean_search_corrected_inference.csv", index=False)
    clean_year_inference.to_csv(
        output / "clean_year_stratified_inference.csv", index=False
    )
    gates.to_csv(output / "extension_gates.csv", index=False)
    money.to_csv(output / "extension_money_guardrails.csv", index=False)
    audit = {
        "research_only": True,
        "selection_period": "2021-01-01/2022-12-31",
        "warmup_period": "2020-01-01/2020-12-31",
        "direct_sealed_rows_read_retained_or_scored": 0,
        "indirect_sealed_information_embedded_in_bridge_diagnostics": True,
        "indirect_sealed_information_used_in_clean_selection": False,
        "retrospective_bridge_inference_isolated_from_clean_inference": True,
        "retrospective_bridge_diagnostics_can_affect_clean_or_sealed_selection": False,
        "closing_prices_used_as_features": False,
        "v5_selection_sha256": EXPECTED_SELECTION_SHA256,
        "dependency_authority": dependency_authority,
        "v5_eligible_candidates": len(candidate_names),
        "bridge_refit_candidates_including_control": len(bridge_names),
        "bridge_refit_production_promotable": False,
        "bridge_nonpromotion_reason": (
            "current bridge course/history exposures embed full-archive retrospective fits"
        ),
        "control_blend_production_promotable": False,
        "control_blend_nonpromotion_reason": "uses legacy skill_est control anchor",
        "clean_blend_pair_frozen_before_extension_scoring": [DYNAMIC, RICH_CLEAN],
        "clean_blend_pair_selected_posthoc_from_v5_discovery": True,
        "clean_blend_weight_uses_odds": False,
        "clean_blend_skill_prediction_shared_across_markets": True,
        "prediction_rows": int(len(extensions)),
        "prediction_candidates": int(extensions["candidate"].nunique()),
        "prediction_key_duplicates": int(
            extensions.duplicated(["candidate", "event_key", "dg_id_key"]).sum()
        ),
        "bridge_coverage_ratio": float(extensions.loc[
            extensions["extension_class"].eq("retrospective_bridge_refit_diagnostic"),
            "bridge_available",
        ].mean()),
        "clean_proper_gate_all_comparators": clean_proper_pass,
        "clean_search_corrected_all_comparators": clean_search_pass,
        "clean_money_guardrail_all_comparators": clean_money_pass,
        "clean_candidates_eligible_for_one_shot_sealed_confirmation": clean_sealed_candidates,
        "production_promotable_from_discovery": False,
        "money_guardrails_computed_for_all_extension_comparisons": True,
        "clean_search_correction": clean_inference_audit,
        "clean_year_stratified_search_correction": clean_year_inference_audit,
        "retrospective_bridge_search_correction": bridge_inference_audit,
        "retrospective_bridge_year_stratified_search_correction": bridge_year_inference_audit,
    }
    _atomic_json(output / "extension_audit.json", audit)
    lines = [
        "# Base-skill discovery extension",
        "",
        "Research only; no production or V5 artifact was changed. No row dated after 2022-12-31 was directly read into fitting or scoring.",
        "The bridge diagnostics nevertheless embed indirect 2023-2026 information because their legacy course/history exposures were fitted on the full archive. They are isolated from clean inference and cannot affect selection.",
        "",
        f"Clean blend pair: `{DYNAMIC}` + `{RICH_CLEAN}`.",
        f"Clean proper-score gate against control and both parents: {clean_proper_pass}.",
        f"Clean all-resample search gate: {clean_search_pass}.",
        f"Clean market-money/CLV/concentration gate: {clean_money_pass}.",
        f"Eligible for one-shot sealed confirmation: {clean_sealed_candidates or 'none'}.",
        f"Clean-family White reality-check p={clean_inference_audit['white_reality_check_null_centered_one_sided_p']:.4f}.",
        "",
        "The bridge refit is retrospective diagnostic only and cannot justify selection by itself. The control blend is also non-selectable because it reintroduces legacy skill_est.",
        "",
        "```text",
        gates.to_string(index=False),
        "```",
        "",
    ]
    (output / "DISCOVERY_EXTENSION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    execution_audit = execution_provenance.finalize(provenance)
    artifacts = []
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
        "contract": "base_skill_discovery_extension_v1_exploratory",
        "complete": True,
        "research_only": True,
        "cannot_modify_v5_or_production": True,
        "direct_sealed_rows_read_retained_or_scored": 0,
        "indirect_sealed_information_embedded_in_bridge_diagnostics": True,
        "indirect_sealed_information_used_in_clean_selection": False,
        "bridge_and_clean_inference_isolated": True,
        "production_promotable_from_discovery": False,
        "audit": audit,
        "inputs": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "execution_provenance": {
            "path": "execution_provenance/execution_provenance.json",
            "verified_unchanged_before_complete_manifest": execution_audit[
                "verified_unchanged_before_complete_manifest"
            ],
        },
        "artifacts": artifacts,
    }
    execution_provenance.assert_current(provenance)
    _atomic_json(output / "manifest.json", manifest)
    print(f"Completed discovery extension: {output}")
    print(
        "Eligible for one-shot sealed confirmation: "
        f"{clean_sealed_candidates if clean_sealed_candidates else 'none'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

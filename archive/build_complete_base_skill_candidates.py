"""Research-only completion layer for the neutral base-skill rebuild.

This module deliberately stops before betting-market evaluation.  It consumes
the frozen PGA player-event feature snapshots, raw all-tour DataGolf rounds,
and the clean IMG accounting rebuild, then emits point-in-time SG-per-round
predictions for the market and tournament evaluators.

The central invariants are intentionally executable:

* challenger features never include ``skill_est``, ``base_skill``, a cascade
  prediction, course fit, or a betting price;
* every historical observation satisfies ``source_round_date < event_start``;
* model coefficients are fitted only to completed earlier event batches;
* model/hyperparameter choice is delegated to the 2021-22 market evaluator;
* the 2023-26 sealed sample can update later walk-forward coefficients, but it
  can never select a family or freeze a finalist; and
* all outputs live below ``permanent_data/research`` and have no production
  consumer.

The current ``skill_estimates.py`` architecture is represented by an explicitly
named approximation.  Its established-PGA branch uses the available legacy
as-of estimate, while its under-20-round branch is reconstructed from a lagged
PGA low-data prior and partial history.  Exact current-code retrospective
controls are emitted separately and can never be mistaken for challengers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow
import scipy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime telemetry
    psutil = None


REPO = Path(__file__).resolve().parents[1]
PANEL_DIR = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_SNAPSHOTS = (
    PANEL_DIR / "model_eval_clean_dg_plumbing_v2" / "feature_snapshots.parquet"
)
DEFAULT_WEATHER_COEFFICIENTS = (
    PANEL_DIR / "model_eval_clean_dg_plumbing_v2" / "weather_coefficients.csv"
)
DEFAULT_ROUND_PANEL = PANEL_DIR / "player_round_panel.parquet"
DEFAULT_IMG = PANEL_DIR / "clean_img_accounting" / "clean_img_player_round.parquet"
DEFAULT_CORE_BRIDGE = PANEL_DIR / "core_bridge" / "legacy_core_bridge.parquet"
DEFAULT_ALL_TOUR_DB = Path(
    r"C:\Users\McKinley Slade\dev\sim_prep\work\dg_historical.db"
)
DEFAULT_OUTPUT = PANEL_DIR / "model_completion"
EXPECTED_FULL_CATALOG_SHA256 = (
    "d355481e8aaa7a23c7e43470ab6d683f2f5017d9ac89c611e04adae8a6597d23"
)
ELASTIC_CD_MAX_ITER = 50_000
ELASTIC_CD_TOL = 1e-3
ELASTIC_CD_MAX_ABS_COEFFICIENT = 100.0
ELASTIC_CD_MAX_TRAIN_WMSE = 1_000.0
REGRESSION_FAMILIES = frozenset({"ridge", "elastic_cd", "robust_irls"})
PREDICTION_ENVELOPE_METHOD = (
    "strict_prior_event_equal_weighted_fitted_prediction_median_mad"
)
PREDICTION_ENVELOPE_SCALE_FACTOR = 1.4826
PREDICTION_ENVELOPE_MULTIPLIER = 6.0
REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX = 0.01
REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX = 0.05
REGRESSION_ENVELOPE_GATE_EVIDENCE_DIR = (
    PANEL_DIR / "model_completion_v3_regression_envelope_preflight"
)
REGRESSION_ENVELOPE_GATE_AUDIT_PATH = (
    REGRESSION_ENVELOPE_GATE_EVIDENCE_DIR / "numerical_preflight_audit.json"
)
REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH = (
    REGRESSION_ENVELOPE_GATE_EVIDENCE_DIR / "envelope_admission_by_candidate.csv"
)
REGRESSION_ENVELOPE_GATE_FOLD_PATH = (
    REGRESSION_ENVELOPE_GATE_EVIDENCE_DIR / "envelope_admission_by_fold.csv"
)
REGRESSION_ENVELOPE_GATE_AUDIT_SHA256 = (
    "37edc6ab44975ae50eddf0faa1503ed01e820be35b529e087e1ff4e1a8dc7fac"
)
REGRESSION_ENVELOPE_GATE_CANDIDATE_SHA256 = (
    "5e532080f0614a4b892c5783e8ada918f11bbc99eb5a3b6269c6c836dba7b171"
)
REGRESSION_ENVELOPE_GATE_FOLD_SHA256 = (
    "c1b074b12ec43d8591a99f95540bb98baaa369ce3e5a6e22953a0e3be4cefbbb"
)
REGRESSION_ENVELOPE_GATE_INELIGIBILITY_REASON = (
    "Fails the predeclared no-market regression-envelope numerical admission "
    "gate (overall clip rate >1% or any refit-fold clip rate >5%); retained "
    "unchanged as a diagnostic-only sensitivity."
)
NUMERICAL_GATE_INELIGIBLE_CANDIDATES = frozenset(
    {
        "complete__ridge__img_primary__w0__quarterly__a1",
        "complete__ridge__img_primary__w0__quarterly__a10",
        "complete__ridge__img_primary__w0__quarterly__a100",
        "complete__ridge__img_primary__w1_feature_only__quarterly__a1",
        "complete__ridge__img_primary__w1_feature_only__quarterly__a10",
        "complete__ridge__img_primary__w1_feature_only__quarterly__a100",
        "complete__ridge__img_primary__w1_neutral_target__quarterly__a1",
        "complete__ridge__img_primary__w1_neutral_target__quarterly__a10",
        "complete__ridge__img_primary__w1_neutral_target__quarterly__a100",
        "complete__ridge__long_game_img__w0__quarterly__a1",
        "complete__ridge__long_game_img__w0__quarterly__a10",
        "complete__ridge__long_game_img__w1_feature_only__quarterly__a1",
        "complete__ridge__long_game_img__w1_neutral_target__quarterly__a1",
        "complete__ridge__total_cat_img__w0__quarterly__a1",
        "complete__ridge__total_cat_img__w0__quarterly__a10",
        "complete__ridge__total_cat_img__w1_feature_only__quarterly__a1",
        "complete__ridge__total_cat_img__w1_feature_only__quarterly__a10",
        "complete__ridge__total_cat_img__w1_neutral_target__quarterly__a1",
        "complete__ridge__total_cat_img__w1_neutral_target__quarterly__a10",
        "complete__ridge__total_cat_img__w1_robust_covariate_feature_only__quarterly__a1",
        "complete__ridge__total_cat_img__w1_robust_covariate_feature_only__quarterly__a10",
        "complete__ridge__total_cat_img__w1_robust_covariate_neutral_target__quarterly__a1",
        "complete__ridge__total_cat_img__w1_robust_covariate_neutral_target__quarterly__a10",
        "complete__ridge__total_img__w0__quarterly__a1",
        "complete__ridge__total_img__w1_feature_only__quarterly__a1",
        "complete__ridge__total_img__w1_neutral_target__quarterly__a1",
        "complete__ridge_group_penalty__total_cat_img__w0__quarterly__a10__sgp2",
        "complete__ridge_group_penalty__total_cat_img__w0__quarterly__a10__sgp4",
        "complete__ridge_group_penalty__total_cat_img__w1_feature_only__quarterly__a10__sgp2",
        "complete__ridge_group_penalty__total_cat_img__w1_feature_only__quarterly__a10__sgp4",
        "complete__ridge_group_penalty__total_cat_img__w1_neutral_target__quarterly__a10__sgp2",
        "complete__ridge_group_penalty__total_cat_img__w1_neutral_target__quarterly__a10__sgp4",
        "complete__robust_irls__total_cat_img__w0__quarterly__a1",
        "complete__robust_irls__total_cat_img__w0__quarterly__a10",
    }
)

DISCOVERY_START = pd.Timestamp("2021-01-01")
DISCOVERY_END = pd.Timestamp("2022-12-31")
SEALED_START = pd.Timestamp("2023-01-01")
SEALED_END = pd.Timestamp("2026-08-22")
PREDICTION_START = pd.Timestamp("2020-01-01")
IMG_START = pd.Timestamp("2020-01-01")
PRIMARY_LABEL = "label_dg_sg_adj_r1_r2_mean"
NEUTRAL_LABEL = "label_dg_sg_neutral_r1_r2_mean"
TOURS = ("pga", "euro", "kft", "liv")
CATEGORIES = ("ott", "app", "arg", "putt")
CATEGORY_SHARES = {"ott": 0.20, "app": 0.33, "arg": 0.14, "putt": 0.33}

FORBIDDEN_CHALLENGER_TOKENS = (
    "legacy",
    "skill_est",
    "base_skill",
    "cascade_prediction",
    "pred_step",
    "odds",
    "price",
    "market",
    "course_fit",
    "course_history",
)

SNAPSHOT_META_COLUMNS = (
    "event_key",
    "dg_id_key",
    "dg_id",
    "event_start_date",
    "event_end_date",
    "year",
    "event_id",
    "player_name",
    PRIMARY_LABEL,
    "label_dg_sg_adj_r1_r2_count",
    NEUTRAL_LABEL,
    "label_dg_sg_neutral_r1_r2_count",
    "history_rounds",
    "max_source_round_date",
    "days_since_last_round",
    "prior_event_withdrawal",
    "legacy_skill_est_asof",
)

MODEL_SIGNAL_SUFFIXES = (
    "ewm_hl10",
    "ess_hl10",
    "ewm_hl20",
    "ess_hl20",
    "ewm_hl40",
    "ess_hl40",
    "ewm_hl80",
    "ess_hl80",
    "ewm_hl160",
    "ess_hl160",
    "mean_d60",
    "n_d60",
    "mean_d180",
    "n_d180",
    "mean_d365",
    "n_d365",
    "mean_d730",
    "n_d730",
    "mean_all",
    "n_all",
)

ALL_TOUR_QUERY_COLUMNS = (
    "player_name",
    "dg_id",
    "event_id",
    "tour",
    "year",
    "round_num",
    "round_date",
    "age",
    "pro_time",
    "sg_total_adj",
    "post_cut",
)


@dataclass(frozen=True)
class CandidateSpec:
    """A frozen challenger or comparator specification."""

    name: str
    family: str
    source: str
    weather_arm: str
    target_arm: str
    refit_schedule: str
    alpha: float = 0.0
    l1_ratio: float = 0.0
    huber_epsilon: float = 1.35
    dynamic_half_life: float = 80.0
    prior_rounds: float = 20.0
    process_variance_per_day: float = 0.0
    short_game_penalty: float = 1.0
    selection_eligible: bool = True
    sensitivity_only: bool = False
    control: bool = False
    prediction_envelope_applied: bool = False
    prediction_envelope_method: str = "none"
    prediction_envelope_scale_factor: float = 0.0
    prediction_envelope_multiplier: float = 0.0
    raw_regression_component_selection_eligible: bool = False
    selection_ineligibility_reason: str = ""
    description: str = ""


CONTROL_SPECS = (
    CandidateSpec(
        name="current_architecture_pt_v2_approx_control",
        family="control",
        source="current_skill_v2_architecture_proxy",
        weather_arm="none",
        target_arm="control",
        refit_schedule="event_batch",
        selection_eligible=False,
        sensitivity_only=True,
        control=True,
        description="Strict-point-in-time operational-v2 architecture approximation.",
    ),
    CandidateSpec(
        name="legacy_skill_est_pt_current_coefficients_control",
        family="control",
        source="legacy_skill_est",
        weather_arm="none",
        target_arm="control",
        refit_schedule="event_batch",
        selection_eligible=False,
        sensitivity_only=True,
        control=True,
        description="Current-coefficient legacy as-of control; retrospective coefficients.",
    ),
    CandidateSpec(
        name="legacy_pred_step2_exact_retrospective_control",
        family="control",
        source="legacy_pred_step2_bridge",
        weather_arm="none",
        target_arm="control",
        refit_schedule="event_batch",
        selection_eligible=False,
        sensitivity_only=True,
        control=True,
        description="Exact current-code pred_step2 bridge; not historical production replay.",
    ),
)


class HuberIRLS:
    """Deterministic ridge-regularized Huber regression.

    ``sklearn.HuberRegressor`` frequently exhausted its iteration budget in the
    first plumbing run.  This small IRLS implementation has an explicit
    convergence test and raises on a failed solve instead of silently accepting
    a non-converged model.
    """

    def __init__(
        self,
        alpha: float = 10.0,
        epsilon: float = 1.35,
        max_iter: int = 80,
        tol: float = 1e-7,
    ) -> None:
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    @staticmethod
    def _solve(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        root_w = np.sqrt(np.clip(weights, 1e-12, None))
        weighted_design = design * root_w[:, None]
        weighted_y = y * root_w
        penalty = np.eye(design.shape[1], dtype=float) * alpha
        penalty[0, 0] = 0.0
        lhs = weighted_design.T @ weighted_design + penalty
        rhs = weighted_design.T @ weighted_y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HuberIRLS":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        base_weight = (
            np.ones(len(y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        beta = self._solve(x, y, base_weight, self.alpha)
        for iteration in range(1, self.max_iter + 1):
            residual = y - (beta[0] + x @ beta[1:])
            center = np.median(residual)
            mad = np.median(np.abs(residual - center))
            scale = max(1.4826 * mad, 1e-8)
            absolute = np.abs(residual) / scale
            robust_weight = np.ones(len(y), dtype=float)
            tail = absolute > self.epsilon
            robust_weight[tail] = self.epsilon / absolute[tail]
            updated = self._solve(x, y, base_weight * robust_weight, self.alpha)
            denominator = max(float(np.linalg.norm(beta)), 1.0)
            change = float(np.linalg.norm(updated - beta) / denominator)
            beta = updated
            self.n_iter_ = iteration
            if change <= self.tol:
                self.converged_ = True
                break
        if not self.converged_:
            raise RuntimeError(
                f"Huber IRLS did not converge in {self.max_iter} iterations "
                f"(alpha={self.alpha}, epsilon={self.epsilon})"
            )
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("HuberIRLS must be fit before predict")
        return self.intercept_ + np.asarray(x, dtype=float) @ self.coef_


class PeakMemoryTracker:
    """Low-overhead process RSS sampler for research compute accounting."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = float(interval_seconds)
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if psutil is None:
            return
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self.peak_rss_bytes = max(
                        self.peak_rss_bytes, int(process.memory_info().rss)
                    )
                except (psutil.Error, OSError):
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return int(self.peak_rss_bytes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    normalized = values.astype("string").str.casefold()
    mapped = normalized.map({"true": True, "false": False})
    if mapped.isna().any():
        bad = sorted(normalized.loc[mapped.isna()].dropna().unique().tolist())
        raise AssertionError(f"numerical-gate evidence has invalid {label}: {bad}")
    return mapped.astype(bool)


def _load_regression_envelope_gate_evidence() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the frozen no-market numerical-admission evidence."""

    expected_hashes = {
        REGRESSION_ENVELOPE_GATE_AUDIT_PATH: REGRESSION_ENVELOPE_GATE_AUDIT_SHA256,
        REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH: (
            REGRESSION_ENVELOPE_GATE_CANDIDATE_SHA256
        ),
        REGRESSION_ENVELOPE_GATE_FOLD_PATH: REGRESSION_ENVELOPE_GATE_FOLD_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"required frozen numerical-gate evidence is missing: {path}"
            )
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise AssertionError(
                "frozen numerical-gate evidence hash changed: "
                f"path={path}, expected={expected_hash}, observed={observed_hash}"
            )
    evidence = pd.read_csv(REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH)
    audit = json.loads(
        REGRESSION_ENVELOPE_GATE_AUDIT_PATH.read_text(encoding="utf-8")
    )
    return evidence, audit


def _validate_regression_envelope_gate_evidence(
    candidates: Sequence[CandidateSpec],
    evidence: pd.DataFrame,
    audit: Mapping[str, Any],
) -> pd.DataFrame:
    """Fail closed unless the frozen preflight proves the exact 34-name OR gate."""

    required = {
        "candidate",
        "overall_clip_rate",
        "worst_fold",
        "worst_fold_clip_rate",
        "fails_overall_gate",
        "fails_fold_gate",
        "guard_dependent",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise AssertionError(
            f"numerical-gate candidate evidence is missing columns: {missing}"
        )
    frame = evidence.loc[:, sorted(required)].copy()
    frame["candidate"] = frame["candidate"].astype(str)
    if frame["candidate"].duplicated().any():
        duplicate_names = sorted(
            frame.loc[frame["candidate"].duplicated(False), "candidate"].unique()
        )
        raise AssertionError(
            f"numerical-gate evidence has duplicate candidates: {duplicate_names}"
        )
    for column in ("overall_clip_rate", "worst_fold_clip_rate"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise AssertionError(
                f"numerical-gate evidence has nonfinite values in {column}"
            )
        if frame[column].lt(0.0).any() or frame[column].gt(1.0).any():
            raise AssertionError(
                f"numerical-gate evidence has out-of-range values in {column}"
            )
    recorded_overall = _strict_bool_series(
        frame["fails_overall_gate"], "fails_overall_gate"
    )
    recorded_fold = _strict_bool_series(frame["fails_fold_gate"], "fails_fold_gate")
    recorded_union = _strict_bool_series(
        frame["guard_dependent"], "guard_dependent"
    )
    computed_overall = frame["overall_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
    )
    computed_fold = frame["worst_fold_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
    )
    computed_union = computed_overall | computed_fold
    if not recorded_overall.equals(computed_overall):
        raise AssertionError(
            "recorded overall numerical-gate flags do not equal the strict >1% gate"
        )
    if not recorded_fold.equals(computed_fold):
        raise AssertionError(
            "recorded fold numerical-gate flags do not equal the strict >5% gate"
        )
    if not recorded_union.equals(computed_union):
        raise AssertionError(
            "recorded numerical OR-gate flags do not equal overall OR any-fold failures"
        )

    regression_names = {
        candidate.name
        for candidate in candidates
        if candidate.family in REGRESSION_FAMILIES
    }
    evidence_names = set(frame["candidate"])
    if evidence_names != regression_names:
        raise AssertionError(
            "numerical-gate evidence does not cover the exact regression catalog: "
            f"missing={sorted(regression_names - evidence_names)}, "
            f"extra={sorted(evidence_names - regression_names)}"
        )
    computed_failures = set(frame.loc[computed_union, "candidate"])
    if computed_failures != set(NUMERICAL_GATE_INELIGIBLE_CANDIDATES):
        raise AssertionError(
            "numerical OR-gate failure set changed from the frozen 34-name set: "
            f"missing={sorted(NUMERICAL_GATE_INELIGIBLE_CANDIDATES - computed_failures)}, "
            f"extra={sorted(computed_failures - NUMERICAL_GATE_INELIGIBLE_CANDIDATES)}"
        )
    remaining = frame.loc[~computed_union]
    if remaining["overall_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
    ).any() or remaining["worst_fold_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
    ).any():
        raise AssertionError(
            "a regression candidate outside the frozen exclusion set fails a gate"
        )
    eligible_names = {
        candidate.name
        for candidate in candidates
        if candidate.family in REGRESSION_FAMILIES and candidate.selection_eligible
    }
    unexpected_eligible_failure = eligible_names & computed_failures
    if unexpected_eligible_failure - set(NUMERICAL_GATE_INELIGIBLE_CANDIDATES):
        raise AssertionError(
            "an unlisted eligible regression fails the numerical gate: "
            f"{sorted(unexpected_eligible_failure)}"
        )

    thresholds = audit.get("predeclared_gates", {})
    if (
        audit.get("market_data_read") is not False
        or audit.get("selection_performed") is not False
        or float(thresholds.get("overall_clip_rate_max", math.nan))
        != REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
        or float(thresholds.get("any_fold_clip_rate_max", math.nan))
        != REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
        or set(audit.get("guard_dependent_candidates", []))
        != set(NUMERICAL_GATE_INELIGIBLE_CANDIDATES)
        or int(audit.get("guard_dependent_candidate_count", -1))
        != len(NUMERICAL_GATE_INELIGIBLE_CANDIDATES)
        or int(audit.get("overall_gate_failure_count", -1))
        != int(computed_overall.sum())
        or int(audit.get("fold_gate_failure_count", -1))
        != int(computed_fold.sum())
    ):
        raise AssertionError(
            "numerical-gate audit metadata disagrees with the frozen evidence"
        )
    output_hashes = audit.get("outputs", {})
    if (
        output_hashes.get(REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH.name)
        != REGRESSION_ENVELOPE_GATE_CANDIDATE_SHA256
        or output_hashes.get(REGRESSION_ENVELOPE_GATE_FOLD_PATH.name)
        != REGRESSION_ENVELOPE_GATE_FOLD_SHA256
    ):
        raise AssertionError(
            "numerical-gate audit does not bind the expected evidence hashes"
        )
    return frame.set_index("candidate", verify_integrity=True)


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": pd.Timestamp(stat.st_mtime_ns, unit="ns", tz="UTC").isoformat(),
    }


def preserve_execution_source(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Atomically preserve the exact builder bytes inspected at process entry."""

    source_path = source_path.resolve()
    source_before = file_fingerprint(source_path)
    source_bytes = source_path.read_bytes()
    preserved_path = output_dir.resolve() / "executed_builder_source.py"
    preserved_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=preserved_path.name, suffix=".tmp", dir=preserved_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, preserved_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    preserved = file_fingerprint(preserved_path)
    if (
        preserved["sha256"] != source_before["sha256"]
        or preserved["size_bytes"] != source_before["size_bytes"]
    ):
        raise RuntimeError(
            "executed builder source changed while preserving entry bytes; "
            "refusing to start"
        )
    return {
        "source_before": source_before,
        "executed_source_copy": preserved,
        "source_copy_matches": True,
    }


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "pyarrow_version": pyarrow.__version__,
    }


def _serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def capture_execution_start(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    preserved_source = preserve_execution_source(source_path, args.output_dir)
    input_paths: dict[str, Path] = {
        "snapshots": args.snapshots,
        "all_tour_db": args.all_tour_db,
        "clean_img": args.clean_img,
        "round_panel": args.round_panel,
        "weather_coefficients": args.weather_coefficients,
        "regression_envelope_gate_audit": REGRESSION_ENVELOPE_GATE_AUDIT_PATH,
        "regression_envelope_gate_candidates": (
            REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH
        ),
        "regression_envelope_gate_folds": REGRESSION_ENVELOPE_GATE_FOLD_PATH,
    }
    if args.core_bridge.exists():
        input_paths["core_bridge"] = args.core_bridge
    if args.reuse_augmented is not None:
        input_paths["reuse_augmented"] = args.reuse_augmented
    return {
        "contract": "base_skill_execution_provenance_v1",
        "process_id": os.getpid(),
        "process_start_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "working_directory": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "parsed_args": _serialized_args(args),
        "runtime": runtime_fingerprint(),
        **preserved_source,
        "inputs_before": {
            name: file_fingerprint(path) for name, path in input_paths.items()
        },
    }


def verify_execution_unchanged(start: Mapping[str, Any]) -> dict[str, Any]:
    source_after = file_fingerprint(Path(start["source_before"]["path"]))
    preserved_before = start.get("executed_source_copy")
    preserved_after = (
        file_fingerprint(Path(preserved_before["path"]))
        if preserved_before is not None
        else None
    )
    inputs_after = {
        name: file_fingerprint(Path(fingerprint["path"]))
        for name, fingerprint in start["inputs_before"].items()
    }
    fields = ("sha256", "size_bytes", "mtime_ns")
    source_unchanged = all(
        source_after[field] == start["source_before"][field] for field in fields
    )
    source_copy_unchanged = bool(
        preserved_before is not None
        and preserved_after is not None
        and all(preserved_after[field] == preserved_before[field] for field in fields)
    )
    source_copy_matches = bool(
        preserved_after is not None
        and preserved_after["sha256"] == start["source_before"]["sha256"]
        and preserved_after["size_bytes"] == start["source_before"]["size_bytes"]
    )
    input_unchanged = {
        name: all(
            inputs_after[name][field] == before[field] for field in fields
        )
        for name, before in start["inputs_before"].items()
    }
    result = {
        "process_id": start["process_id"],
        "process_start_utc": start["process_start_utc"],
        "process_end_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_before": start["source_before"],
        "source_after": source_after,
        "source_unchanged": source_unchanged,
        "executed_source_copy_before": preserved_before,
        "executed_source_copy_after": preserved_after,
        "source_copy_unchanged": source_copy_unchanged,
        "source_copy_matches": source_copy_matches,
        "inputs_before": start["inputs_before"],
        "inputs_after": inputs_after,
        "input_unchanged": input_unchanged,
        "all_unchanged": bool(
            source_unchanged
            and source_copy_unchanged
            and source_copy_matches
            and all(input_unchanged.values())
        ),
        "argv": start["argv"],
        "parsed_args": start["parsed_args"],
        "runtime": start["runtime"],
        "working_directory": start["working_directory"],
    }
    if not result["all_unchanged"]:
        changed = [name for name, unchanged in input_unchanged.items() if not unchanged]
        if not source_unchanged:
            changed.insert(0, "executed_builder_source")
        if not source_copy_unchanged or not source_copy_matches:
            changed.insert(0, "executed_builder_source_copy")
        raise RuntimeError(
            "execution provenance changed during run; refusing completion: "
            + ", ".join(changed)
        )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _stable_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip().casefold()
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value).strip().casefold()


def _normalize_tour(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.replace(
        {
            "dpwt": "euro",
            "dp_world": "euro",
            "dp world": "euro",
            "european": "euro",
        }
    )


def _split_for_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce").dt.normalize()
    return pd.Series(
        np.select(
            [
                dates.lt(PREDICTION_START),
                dates.lt(DISCOVERY_START),
                dates.le(DISCOVERY_END),
                dates.le(SEALED_END),
            ],
            ["warmup", "calibration_warmup", "discovery", "sealed"],
            default="post_contract",
        ),
        index=values.index,
        dtype="string",
    )


def _canonical_event_key(year: object, event_id: object) -> str:
    try:
        return f"pga:R{int(year):04d}{int(event_id):03d}"
    except (TypeError, ValueError):
        return ""


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_key", observed=True)["event_key"].transform("size")
    raw = 1.0 / counts.to_numpy(dtype=float)
    return raw * (len(raw) / max(raw.sum(), 1e-12))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Return a deterministic weighted quantile after strict finite validation."""

    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(values) != len(weights) or not len(values):
        raise ValueError("weighted quantile requires nonempty equal-length arrays")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"weighted quantile outside [0, 1]: {quantile}")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.all():
        raise ValueError("weighted quantile received nonfinite/nonpositive values or weights")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = float(quantile) * float(ordered_weights.sum())
    position = int(
        np.searchsorted(np.cumsum(ordered_weights), threshold, side="left")
    )
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def _fit_prediction_envelope(
    training_prediction: np.ndarray,
    event_equal_weights: np.ndarray,
    candidate: CandidateSpec,
) -> dict[str, float | str]:
    """Fit the regression extrapolation envelope from strict-prior fits only."""

    if (
        candidate.family not in REGRESSION_FAMILIES
        or not candidate.prediction_envelope_applied
        or candidate.prediction_envelope_method != PREDICTION_ENVELOPE_METHOD
    ):
        raise ValueError(
            f"regression candidate has no bound prediction envelope: {candidate.name}"
        )
    prediction = np.asarray(training_prediction, dtype=float).reshape(-1)
    weights = np.asarray(event_equal_weights, dtype=float).reshape(-1)
    center = _weighted_quantile(prediction, weights, 0.5)
    mad = _weighted_quantile(np.abs(prediction - center), weights, 0.5)
    robust_scale = float(candidate.prediction_envelope_scale_factor) * mad
    multiplier = float(candidate.prediction_envelope_multiplier)
    lower = center - multiplier * robust_scale
    upper = center + multiplier * robust_scale
    values = np.asarray([center, mad, robust_scale, multiplier, lower, upper])
    if (
        not np.isfinite(values).all()
        or mad <= 0.0
        or robust_scale <= 0.0
        or multiplier <= 0.0
        or not lower < upper
    ):
        raise RuntimeError(
            "strict-prior fitted-prediction envelope has zero/nonfinite scale or "
            f"bounds for {candidate.name}: center={center}, mad={mad}, "
            f"scale={robust_scale}, multiplier={multiplier}, bounds=({lower}, {upper})"
        )
    return {
        "method": candidate.prediction_envelope_method,
        "center": center,
        "weighted_mad": mad,
        "robust_scale": robust_scale,
        "scale_factor": float(candidate.prediction_envelope_scale_factor),
        "multiplier": multiplier,
        "lower": lower,
        "upper": upper,
    }


def _apply_prediction_envelope(
    raw_prediction: np.ndarray,
    envelope: Mapping[str, float | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply fixed training-fold bounds without consulting test outcomes."""

    raw = np.asarray(raw_prediction, dtype=float).reshape(-1)
    lower = float(envelope["lower"])
    upper = float(envelope["upper"])
    if not np.isfinite(raw).all() or not np.isfinite([lower, upper]).all() or lower >= upper:
        raise RuntimeError("prediction envelope application received invalid values/bounds")
    guarded = np.clip(raw, lower, upper)
    distance = np.abs(raw - guarded)
    clipped = distance > 0.0
    if not np.isfinite(guarded).all() or not np.isfinite(distance).all():
        raise RuntimeError("prediction envelope emitted nonfinite guarded values")
    return guarded, clipped, distance


def _aggregate_values(
    values: np.ndarray,
    dates: np.ndarray,
    cutoff: pd.Timestamp,
    prefix: str,
    half_lives: Sequence[int] = (20, 80),
    day_windows: Sequence[int] = (60, 180, 365, 730),
) -> dict[str, Any]:
    """Aggregate already sliced strict-prior observations."""

    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    valid = np.isfinite(values) & ~np.isnat(dates)
    values = values[valid]
    dates = dates[valid]
    result: dict[str, Any] = {
        f"{prefix}__n_all": int(len(values)),
        f"{prefix}__mean_all": float(np.mean(values)) if len(values) else np.nan,
        f"{prefix}__max_source_date": pd.Timestamp(dates.max()) if len(values) else pd.NaT,
    }
    if not len(values):
        for half_life in half_lives:
            result[f"{prefix}__ewm_hl{half_life}"] = np.nan
            result[f"{prefix}__ess_hl{half_life}"] = 0.0
        for days in day_windows:
            result[f"{prefix}__mean_d{days}"] = np.nan
            result[f"{prefix}__n_d{days}"] = 0
        return result

    recency = np.arange(len(values) - 1, -1, -1, dtype=float)
    for half_life in half_lives:
        weights = np.exp2(-recency / float(half_life))
        result[f"{prefix}__ewm_hl{half_life}"] = _weighted_mean(values, weights)
        result[f"{prefix}__ess_hl{half_life}"] = float(
            weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12)
        )
    cutoff_np = np.datetime64(pd.Timestamp(cutoff).normalize())
    ages = (cutoff_np - dates).astype("timedelta64[D]").astype(float)
    for days in day_windows:
        inside = (ages >= 0) & (ages <= days)
        result[f"{prefix}__mean_d{days}"] = (
            float(np.mean(values[inside])) if inside.any() else np.nan
        )
        result[f"{prefix}__n_d{days}"] = int(inside.sum())
    return result


def _read_snapshot_columns(path: Path) -> list[str]:
    available = pq.ParquetFile(path).schema.names
    selected = [column for column in SNAPSHOT_META_COLUMNS if column in available]
    required = {
        "event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "event_end_date",
        PRIMARY_LABEL,
        "label_dg_sg_adj_r1_r2_count",
    }
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"frozen snapshots missing required columns: {missing}")
    for column in available:
        if column.startswith(("dg_precut_total_w0__", "dg_precut_total_w1__")):
            if column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES:
                selected.append(column)
        elif column.startswith(("dg_precut_cat_",)):
            if column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES:
                selected.append(column)
        elif (
            column.startswith("dg_precut_")
            and any(token in column.casefold() for token in ("robust", "covariate", "clipped"))
            and column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES
        ):
            # Preserve forthcoming covariate-only robust-W1 variants.  The
            # feature layer never reads weather_excluded/outcome flags.
            selected.append(column)
    return list(dict.fromkeys(selected))


def load_frozen_snapshots(path: Path) -> pd.DataFrame:
    columns = _read_snapshot_columns(path)
    frame = pd.read_parquet(path, columns=columns)
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["event_end_date"] = pd.to_datetime(
        frame["event_end_date"], errors="coerce"
    ).dt.normalize()
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
    frame["dg_id_key"] = frame["dg_id_key"].map(_stable_id)
    if frame.duplicated(["event_key", "dg_id_key"]).any():
        raise AssertionError("frozen snapshots have duplicate player-event keys")
    if NEUTRAL_LABEL not in frame:
        frame[NEUTRAL_LABEL] = np.nan
        frame["label_dg_sg_neutral_r1_r2_count"] = 0
    return frame.sort_values(
        ["event_start_date", "event_key", "dg_id_key"], kind="mergesort"
    ).reset_index(drop=True)


def load_all_tour_rounds(path: Path) -> pd.DataFrame:
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        info = pd.read_sql_query("PRAGMA table_info(player_rounds)", connection)
        available = set(info["name"].astype(str))
        columns = [column for column in ALL_TOUR_QUERY_COLUMNS if column in available]
        missing = sorted({"dg_id", "tour", "round_num", "round_date", "sg_total_adj"} - set(columns))
        if missing:
            raise ValueError(f"all-tour database missing columns: {missing}")
        query = (
            "SELECT "
            + ", ".join(f'"{column}"' for column in columns)
            + " FROM player_rounds WHERE dg_id IS NOT NULL "
            + "AND round_date IS NOT NULL AND sg_total_adj IS NOT NULL"
        )
        frame = pd.read_sql_query(query, connection)
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce").dt.normalize()
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame["sg_total_adj"] = pd.to_numeric(frame["sg_total_adj"], errors="coerce")
    frame["tour"] = _normalize_tour(frame["tour"])
    for column in ("age", "pro_time"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["tour"].isin(TOURS)].dropna(
        subset=["dg_id", "round_date", "sg_total_adj"]
    )
    return frame.sort_values(
        ["dg_id", "round_date", "tour", "event_id", "round_num"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_dg_weather_rounds(path: Path) -> pd.DataFrame:
    """Load only covariates/outcomes needed for robust W1 reconstruction."""

    columns = [
        "event_key",
        "year",
        "event_id",
        "dg_id",
        "round_num",
        "round_date",
        "course_num",
        "weather_delta_wind",
        "weather_delta_dew",
        "dg_sg_total_field_adjusted",
        "dg_sg_ott_field_adjusted",
        "dg_sg_app_field_adjusted",
        "dg_sg_arg_field_adjusted",
        "dg_sg_putt_field_adjusted",
        "img_accounting_sg_total_field_adjusted",
        "img_accounting_available",
        "img_v2_sg_total_signal",
        "img_v2_available",
        "img_v2_future_validated_transform",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")
    frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce")
    frame["course_num"] = pd.to_numeric(frame["course_num"], errors="coerce").astype("Int64")
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce").dt.normalize()
    numeric_columns = [
        "weather_delta_wind",
        "weather_delta_dew",
        "dg_sg_total_field_adjusted",
        "dg_sg_ott_field_adjusted",
        "dg_sg_app_field_adjusted",
        "dg_sg_arg_field_adjusted",
        "dg_sg_putt_field_adjusted",
        "img_accounting_sg_total_field_adjusted",
        "img_v2_sg_total_signal",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["dg_id", "round_date"]).sort_values(
        ["dg_id", "round_date", "event_key", "round_num"], kind="mergesort"
    )


def fit_covariate_only_weather_bounds(
    rounds: pd.DataFrame,
) -> dict[str, dict[str, float | int | str]]:
    """Freeze robust weather bounds from pre-2021 covariates only.

    The 0.1%/99.9% winsor limits are deliberately outcome-blind.  They operate
    on course/round-centered exposure deltas and never read the repository's
    retrospectively outcome-selected weather exclusion fields.
    """

    warmup = rounds.loc[rounds["round_date"].lt(DISCOVERY_START)]
    result: dict[str, dict[str, float | int | str]] = {}
    for short, column in (
        ("wind", "weather_delta_wind"),
        ("dew", "weather_delta_dew"),
    ):
        values = pd.to_numeric(warmup[column], errors="coerce").dropna()
        if len(values) < 1000:
            raise ValueError(f"insufficient pre-2021 {short} observations for robust bounds")
        lower = float(values.quantile(0.001))
        upper = float(values.quantile(0.999))
        if not lower < upper:
            raise ValueError(f"invalid robust weather bounds for {short}: {lower}, {upper}")
        result[short] = {
            "column": column,
            "fit_before": DISCOVERY_START.date().isoformat(),
            "fit_rows": int(len(values)),
            "lower": lower,
            "upper": upper,
            "warmup_min": float(values.min()),
            "warmup_max": float(values.max()),
            "method": "pre_2021_covariate_only_q001_q999_winsor",
        }
    return result


def _clip_weather(
    values: pd.Series | np.ndarray,
    bound: Mapping[str, float | int | str],
) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return np.clip(numeric, float(bound["lower"]), float(bound["upper"]))


def _robust_weather_values(
    frame: pd.DataFrame,
    weather_row: Mapping[str, Any],
    bounds: Mapping[str, Mapping[str, float | int | str]],
) -> dict[str, np.ndarray]:
    wind = _clip_weather(frame["weather_delta_wind"], bounds["wind"])
    dew = _clip_weather(frame["weather_delta_dew"], bounds["dew"])
    values: dict[str, np.ndarray] = {}
    for category in (*CATEGORIES, "total"):
        observed = pd.to_numeric(
            frame[f"dg_sg_{category}_field_adjusted"], errors="coerce"
        ).to_numpy(float)
        values[category] = (
            observed
            + float(weather_row[f"{category}_beta_wind"]) * np.nan_to_num(wind, nan=0.0)
            + float(weather_row[f"{category}_beta_dew"]) * np.nan_to_num(dew, nan=0.0)
        )
        values[category][~np.isfinite(observed)] = np.nan
    return values


def build_robust_w1_features(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, float | int | str]]]:
    """Rebuild R1-R2 DG histories with outcome-blind clipped weather deltas."""

    bounds = fit_covariate_only_weather_bounds(rounds)
    lookup = _coefficient_lookup(weather_coefficients)
    by_player = rounds.groupby("dg_id", sort=False, observed=True).indices
    current_keys = rounds.groupby(["event_key", "dg_id"], sort=False, observed=True).indices
    queries = snapshots.groupby("dg_id", sort=False, observed=True).indices
    output: list[dict[str, Any]] = [dict() for _ in range(len(snapshots))]
    coefficient_violations = 0
    for dg_id, positions_raw in queries.items():
        positions = np.asarray(positions_raw, dtype=int)
        history_positions = by_player.get(dg_id)
        history = (
            rounds.iloc[np.asarray(history_positions, dtype=int)].sort_values(
                "round_date", kind="mergesort"
            )
            if history_positions is not None
            else rounds.iloc[:0]
        )
        dates = history["round_date"].to_numpy(dtype="datetime64[ns]")
        for position in positions:
            target = pd.Timestamp(snapshots.at[position, "event_start_date"]).normalize()
            stop = int(np.searchsorted(dates, np.datetime64(target), side="left"))
            prior = history.iloc[:stop]
            prior = prior.loc[prior["round_num"].isin([1, 2])]
            prior_dates = prior["round_date"].to_numpy(dtype="datetime64[ns]")
            weather_row = _weather_row_for_target(lookup, target)
            fit_max = pd.to_datetime(weather_row.get("fit_max_round_date"), errors="coerce")
            if pd.notna(fit_max) and fit_max >= target:
                coefficient_violations += 1
            row: dict[str, Any] = {
                "robust_w1_max_source_round_date": (
                    pd.Timestamp(prior_dates.max()) if len(prior_dates) else pd.NaT
                ),
                "robust_w1_coefficient_fit_max_date": fit_max,
            }
            values = (
                _robust_weather_values(prior, weather_row, bounds)
                if len(prior)
                else {category: np.array([], dtype=float) for category in (*CATEGORIES, "total")}
            )
            row.update(
                _aggregate_values(
                    values["total"],
                    prior_dates,
                    target,
                    "dg_precut_total_w1_robust_covariate",
                    half_lives=(10, 20, 40, 80, 160),
                    day_windows=(60, 180, 365, 730),
                )
            )
            for category in CATEGORIES:
                residual = values[category] - CATEGORY_SHARES[category] * values["total"]
                row.update(
                    _aggregate_values(
                        residual,
                        prior_dates,
                        target,
                        f"dg_precut_cat_{category}_w1_robust_covariate",
                        half_lives=(10, 20, 40, 80, 160),
                        day_windows=(60, 180, 365, 730),
                    )
                )

            current_positions = current_keys.get((snapshots.at[position, "event_key"], dg_id))
            current = (
                rounds.iloc[np.asarray(current_positions, dtype=int)]
                if current_positions is not None
                else rounds.iloc[:0]
            )
            current = current.loc[current["round_num"].isin([1, 2])]
            current_values = (
                _robust_weather_values(current, weather_row, bounds)["total"]
                if len(current)
                else np.array([], dtype=float)
            )
            row["label_dg_sg_neutral_robust_r1_r2_mean"] = (
                float(np.nanmean(current_values))
                if np.isfinite(current_values).any()
                else np.nan
            )
            row["label_dg_sg_neutral_robust_r1_r2_count"] = int(
                np.isfinite(current_values).sum()
            )
            output[position] = row
    features = pd.DataFrame(output, index=snapshots.index)
    target = pd.to_datetime(snapshots["event_start_date"], errors="coerce")
    source = pd.to_datetime(features["robust_w1_max_source_round_date"], errors="coerce")
    fit_max = pd.to_datetime(features["robust_w1_coefficient_fit_max_date"], errors="coerce")
    source_violations = source.notna() & source.ge(target)
    fit_violations = fit_max.notna() & fit_max.ge(target)
    if source_violations.any() or fit_violations.any() or coefficient_violations:
        raise AssertionError("robust W1 chronology audit failed")
    audit: dict[str, Any] = {
        "method": "pre_2021_covariate_only_q001_q999_winsor",
        "strict_prior_violations": int(source_violations.sum()),
        "coefficient_strict_prior_violations": int(fit_violations.sum()),
        "outcome_selected_weather_flags_read": False,
        "bounds": bounds,
    }
    for short, column in (("wind", "weather_delta_wind"), ("dew", "weather_delta_dew")):
        raw = pd.to_numeric(rounds[column], errors="coerce")
        lower = float(bounds[short]["lower"])
        upper = float(bounds[short]["upper"])
        audit[f"{short}_clipped_rows"] = int((raw.lt(lower) | raw.gt(upper)).sum())
        audit[f"{short}_raw_min"] = float(raw.min())
        audit[f"{short}_raw_max"] = float(raw.max())
        audit[f"{short}_used_min"] = float(raw.clip(lower, upper).min())
        audit[f"{short}_used_max"] = float(raw.clip(lower, upper).max())
    return features, audit, bounds


def _fallback_reason(tour_counts: Mapping[str, int]) -> str:
    positive = [tour for tour, count in tour_counts.items() if count > 0]
    if not positive:
        return "cold_start_pga_prior"
    if positive == ["pga"]:
        return "pga_history_only"
    if "pga" in positive:
        return "pga_plus_cross_tour"
    if len(positive) == 1:
        return f"{positive[0]}_translation_only"
    return "multi_non_pga_translation"


def build_postcut_nuisance_events(rounds: pd.DataFrame) -> pd.DataFrame:
    """Create strictly ordered player-event evidence for survivor uplift.

    Each source event receives its history/skill bucket from R1-R2 observations
    in earlier events on the same tour.  The uplift label is the within-player,
    within-event R3-R4 mean minus R1-R2 mean.  No full-sample uplift table or
    outcome-selected exclusion is consumed.
    """

    keys = ["tour", "year", "event_id", "dg_id"]
    source = rounds.copy()
    source["_pre_value"] = source["sg_total_adj"].where(source["round_num"].isin([1, 2]))
    source["_post_value"] = source["sg_total_adj"].where(source["round_num"].isin([3, 4]))
    events = (
        source.groupby(keys, as_index=False, observed=True)
        .agg(
            event_start=("round_date", "min"),
            event_end=("round_date", "max"),
            pre_mean=("_pre_value", "mean"),
            pre_count=("_pre_value", "count"),
            post_mean=("_post_value", "mean"),
            post_count=("_post_value", "count"),
        )
        .sort_values(["dg_id", "tour", "event_start", "year", "event_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped = events.groupby(["dg_id", "tour"], sort=False, observed=True)
    events["prior_pre_count"] = grouped["pre_count"].transform(
        lambda values: values.cumsum().shift(1, fill_value=0)
    )
    pre_sum = events["pre_mean"].fillna(0.0) * events["pre_count"]
    events["_pre_sum"] = pre_sum
    events["prior_pre_sum"] = events.groupby(
        ["dg_id", "tour"], sort=False, observed=True
    )["_pre_sum"].transform(lambda values: values.cumsum().shift(1, fill_value=0.0))
    events["prior_pre_mean"] = np.divide(
        events["prior_pre_sum"],
        events["prior_pre_count"],
        out=np.full(len(events), np.nan, dtype=float),
        where=events["prior_pre_count"].to_numpy(float) > 0,
    )
    events["history_bucket"] = pd.cut(
        events["prior_pre_count"],
        bins=[-np.inf, 19, 79, np.inf],
        labels=["h00_19", "h20_79", "h80_plus"],
    ).astype("string")
    events["skill_bucket"] = pd.cut(
        events["prior_pre_mean"],
        bins=[-np.inf, -0.75, 0.0, 0.75, np.inf],
        labels=["s_low", "s_below_avg", "s_above_avg", "s_high"],
    ).astype("string").fillna("s_unknown")
    events["survivor_uplift"] = events["post_mean"] - events["pre_mean"]
    events.loc[
        events["pre_count"].lt(2) | events["post_count"].lt(1),
        "survivor_uplift",
    ] = np.nan
    return events.drop(columns=["_pre_sum"])


def _robust_unweighted_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return 0.0
    median = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - median)))
    if mad > 1e-8:
        numeric = np.clip(numeric, median - 4.4478 * mad, median + 4.4478 * mad)
    return float(np.mean(numeric))


def estimate_postcut_uplifts(
    nuisance_events: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[dict[tuple[str, str, str], float], dict[str, float], dict[str, Any]]:
    """Fit partially pooled post-cut corrections before one target batch."""

    training = nuisance_events.loc[
        nuisance_events["event_end"].lt(cutoff)
        & nuisance_events["survivor_uplift"].notna()
    ].copy()
    global_mean = _robust_unweighted_mean(training["survivor_uplift"])
    tour_means: dict[str, float] = {}
    tour_counts: dict[str, int] = {}
    for tour in TOURS:
        group = training.loc[training["tour"].eq(tour), "survivor_uplift"]
        count = int(group.notna().sum())
        raw = _robust_unweighted_mean(group)
        weight = count / (count + 200.0)
        tour_means[tour] = float(global_mean + weight * (raw - global_mean))
        tour_counts[tour] = count
    cells: dict[tuple[str, str, str], float] = {}
    for key, group in training.groupby(
        ["tour", "history_bucket", "skill_bucket"], observed=True
    ):
        key = tuple(str(value) for value in key)
        count = int(group["survivor_uplift"].notna().sum())
        raw = _robust_unweighted_mean(group["survivor_uplift"])
        parent = tour_means.get(key[0], global_mean)
        weight = count / (count + 50.0)
        cells[key] = float(parent + weight * (raw - parent))
    audit = {
        "cutoff": cutoff,
        "training_rows": int(len(training)),
        "training_max_event_end": training["event_end"].max() if len(training) else pd.NaT,
        "global_uplift": global_mean,
        **{f"tour_uplift_{tour}": tour_means[tour] for tour in TOURS},
        **{f"tour_rows_{tour}": tour_counts[tour] for tour in TOURS},
        "cells": int(len(cells)),
    }
    return cells, tour_means, audit


def _attach_postcut_source_buckets(
    rounds: pd.DataFrame,
    nuisance_events: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "tour",
        "year",
        "event_id",
        "dg_id",
        "history_bucket",
        "skill_bucket",
    ]
    output = rounds.merge(
        nuisance_events[columns],
        on=["tour", "year", "event_id", "dg_id"],
        how="left",
        validate="many_to_one",
    )
    output["history_bucket"] = output["history_bucket"].fillna("h00_19")
    output["skill_bucket"] = output["skill_bucket"].fillna("s_unknown")
    return output


def build_cross_tour_features(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build strict-prior per-tour and combined R1-R2 histories."""

    nuisance_events = build_postcut_nuisance_events(rounds)
    rounds = _attach_postcut_source_buckets(rounds, nuisance_events)
    nuisance_cache: dict[pd.Timestamp, tuple[dict[tuple[str, str, str], float], dict[str, float], dict[str, Any]]] = {}
    for cutoff in snapshots["event_start_date"].drop_duplicates().sort_values():
        target = pd.Timestamp(cutoff).normalize()
        nuisance_cache[target] = estimate_postcut_uplifts(nuisance_events, target)
    query_by_player = snapshots.groupby("dg_id", sort=False, observed=True).indices
    history_by_player = rounds.groupby("dg_id", sort=False, observed=True).indices
    rows: list[dict[str, Any]] = [dict() for _ in range(len(snapshots))]
    for dg_id, query_positions_raw in query_by_player.items():
        query_positions = np.asarray(query_positions_raw, dtype=int)
        history_positions = history_by_player.get(dg_id)
        history = (
            rounds.iloc[np.asarray(history_positions, dtype=int)]
            if history_positions is not None
            else rounds.iloc[:0]
        )
        history = history.sort_values("round_date", kind="mergesort")
        all_dates = history["round_date"].to_numpy(dtype="datetime64[ns]")
        all_values = history["sg_total_adj"].to_numpy(dtype=float)
        all_tours = history["tour"].astype(str).to_numpy()
        all_round_nums = history["round_num"].to_numpy(dtype=float)
        all_history_buckets = history["history_bucket"].astype(str).to_numpy()
        all_skill_buckets = history["skill_bucket"].astype(str).to_numpy()
        for position in query_positions:
            target = pd.Timestamp(snapshots.at[position, "event_start_date"]).normalize()
            stop = int(
                np.searchsorted(all_dates, np.datetime64(target), side="left")
            )
            prior_dates = all_dates[:stop]
            prior_values = all_values[:stop]
            prior_tours = all_tours[:stop]
            prior_round_nums = all_round_nums[:stop]
            prior_history_buckets = all_history_buckets[:stop]
            prior_skill_buckets = all_skill_buckets[:stop]
            r12 = np.isin(prior_round_nums, [1.0, 2.0])
            row: dict[str, Any] = {}
            tour_counts: dict[str, int] = {}
            nuisance_cells, nuisance_tours, nuisance_audit = nuisance_cache[target]
            adjusted_all = prior_values.copy()
            postcut = np.isin(prior_round_nums, [3.0, 4.0])
            if postcut.any():
                corrections = np.zeros(len(prior_values), dtype=float)
                for index in np.flatnonzero(postcut):
                    key = (
                        str(prior_tours[index]),
                        str(prior_history_buckets[index]),
                        str(prior_skill_buckets[index]),
                    )
                    corrections[index] = nuisance_cells.get(
                        key, nuisance_tours.get(key[0], 0.0)
                    )
                adjusted_all[postcut] -= corrections[postcut]
            for tour in TOURS:
                mask = r12 & (prior_tours == tour)
                aggregate = _aggregate_values(
                    prior_values[mask],
                    prior_dates[mask],
                    target,
                    f"xtour_{tour}_r12",
                )
                row.update(aggregate)
                tour_counts[tour] = int(aggregate[f"xtour_{tour}_r12__n_all"])
                all_mask = prior_tours == tour
                row[f"xtour_{tour}_all_round_count"] = int(all_mask.sum())
                row[f"xtour_{tour}_r12_since_img_start"] = int(
                    (
                        mask
                        & (prior_dates >= np.datetime64(IMG_START))
                    ).sum()
                )
                row[f"xtour_{tour}_all_since_img_start"] = int(
                    (
                        all_mask
                        & (prior_dates >= np.datetime64(IMG_START))
                    ).sum()
                )
                row.update(
                    _aggregate_values(
                        adjusted_all[all_mask],
                        prior_dates[all_mask],
                        target,
                        f"xtour_{tour}_postcut_adj",
                    )
                )
            row.update(
                _aggregate_values(
                    prior_values[r12],
                    prior_dates[r12],
                    target,
                    "xtour_combined_r12",
                )
            )
            row["xtour_distinct_tours"] = int(sum(count > 0 for count in tour_counts.values()))
            row["xtour_has_pga_history"] = bool(tour_counts["pga"] > 0)
            row["xtour_only_non_pga_history"] = bool(
                tour_counts["pga"] == 0 and sum(tour_counts.values()) > 0
            )
            row["cross_tour_fallback_reason"] = _fallback_reason(tour_counts)
            row["postcut_nuisance_fit_max_event_end"] = nuisance_audit[
                "training_max_event_end"
            ]
            row["postcut_nuisance_training_rows"] = nuisance_audit["training_rows"]
            row["postcut_nuisance_global_uplift"] = nuisance_audit["global_uplift"]
            row["xtour_max_source_round_date"] = (
                pd.Timestamp(prior_dates.max()) if len(prior_dates) else pd.NaT
            )
            row["xtour_days_since_last_round"] = (
                float((np.datetime64(target) - prior_dates.max()) / np.timedelta64(1, "D"))
                if len(prior_dates)
                else np.nan
            )
            if stop:
                prior = history.iloc[:stop]
                for source, destination in (
                    ("age", "age_asof"),
                    ("pro_time", "pro_time_asof"),
                ):
                    available = prior.dropna(subset=[source])
                    row[destination] = (
                        float(available.iloc[-1][source]) if len(available) else np.nan
                    )
            else:
                row["age_asof"] = np.nan
                row["pro_time_asof"] = np.nan
            rows[position] = row
    features = pd.DataFrame(rows, index=snapshots.index)
    source = pd.to_datetime(features["xtour_max_source_round_date"], errors="coerce")
    target = pd.to_datetime(snapshots["event_start_date"], errors="coerce")
    violations = source.notna() & source.ge(target)
    nuisance_source = pd.to_datetime(
        features["postcut_nuisance_fit_max_event_end"], errors="coerce"
    )
    nuisance_violations = nuisance_source.notna() & nuisance_source.ge(target)
    audit = {
        "round_rows": int(len(rounds)),
        "players": int(rounds["dg_id"].nunique()),
        "snapshot_rows": int(len(snapshots)),
        "strict_prior_violations": int(violations.sum()),
        "postcut_nuisance_strict_prior_violations": int(nuisance_violations.sum()),
        "fallback_reason_counts": {
            str(key): int(value)
            for key, value in features["cross_tour_fallback_reason"].value_counts().items()
        },
        "tour_round_counts": {
            str(key): int(value)
            for key, value in rounds["tour"].value_counts().items()
        },
        "postcut_nuisance_event_rows": int(len(nuisance_events)),
        "postcut_nuisance_labeled_rows": int(
            nuisance_events["survivor_uplift"].notna().sum()
        ),
    }
    if violations.any() or nuisance_violations.any():
        raise AssertionError("cross-tour feature history reaches target event start")
    return features, audit


def _load_img_with_weather(
    img_path: Path,
    round_panel_path: Path,
    weather_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    img = pd.read_parquet(img_path)
    keep_weather = [
        "year",
        "event_id",
        "dg_id",
        "round_num",
        "round_date",
        "course_num",
        "weather_delta_wind",
        "weather_delta_dew",
    ]
    weather = (
        weather_frame[keep_weather].copy()
        if weather_frame is not None
        else pd.read_parquet(round_panel_path, columns=keep_weather)
    )
    for frame in (img, weather):
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")
        frame["dg_id"] = pd.to_numeric(frame["dg_id"], errors="coerce").astype("Int64")
        frame["round_num"] = pd.to_numeric(frame["round_num"], errors="coerce").astype("Int64")
        frame["course_num"] = pd.to_numeric(frame["course_num"], errors="coerce").astype("Int64")
        frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce").dt.normalize()
    keys = ["year", "event_id", "dg_id", "round_num", "round_date", "course_num"]
    missing_img_identity = int(img[keys].isna().any(axis=1).sum())
    missing_weather_identity = int(weather[keys].isna().any(axis=1).sum())
    if missing_img_identity or missing_weather_identity:
        raise AssertionError(
            "IMG/weather identity contains missing values; "
            f"img={missing_img_identity}, weather={missing_weather_identity}"
        )
    duplicate_img = int(img.duplicated(keys, keep=False).sum())
    duplicate_weather = int(weather.duplicated(keys, keep=False).sum())
    if duplicate_img or duplicate_weather:
        raise AssertionError(
            "ambiguous IMG/weather round identity; "
            f"img_duplicate_rows={duplicate_img}, "
            f"weather_duplicate_rows={duplicate_weather}"
        )
    source_weather_missing = int(
        weather[["weather_delta_wind", "weather_delta_dew"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    img = img.merge(
        weather,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_weather_join",
    )
    unmatched = int(img["_weather_join"].ne("both").sum())
    if unmatched:
        raise AssertionError(
            "clean IMG rows lack an exact weather-panel identity: "
            f"{unmatched}"
        )
    joined_weather_missing = int(
        img[["weather_delta_wind", "weather_delta_dew"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    nonzero_weather = (
        pd.to_numeric(img["weather_delta_wind"], errors="coerce").fillna(0).abs()
        + pd.to_numeric(img["weather_delta_dew"], errors="coerce").fillna(0).abs()
    ).gt(0)
    img = img.drop(columns="_weather_join")
    audit = {
        "img_rows": int(len(img)),
        "weather_join_identity": keys,
        "img_duplicate_identity_rows": duplicate_img,
        "weather_duplicate_input_rows": duplicate_weather,
        "weather_join_matched_rows": int(len(img) - unmatched),
        "weather_join_unmatched_rows": unmatched,
        "weather_source_missing_rows": source_weather_missing,
        "weather_missing_rows": joined_weather_missing,
        "weather_nonzero_rows": int(nonzero_weather.sum()),
        "weather_missing_explanation": (
            "Missing deltas are present in the exactly matched source round panel; "
            "no missing row is caused by join-key failure."
        ),
    }
    return img.sort_values(["dg_id", "round_date", "event_key", "round_num"]), audit


def _coefficient_lookup(weather_coefficients: pd.DataFrame) -> dict[pd.Timestamp, dict[str, Any]]:
    frame = weather_coefficients.copy()
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["fit_max_round_date"] = pd.to_datetime(
        frame["fit_max_round_date"], errors="coerce"
    ).dt.normalize()
    return {
        pd.Timestamp(row["event_start_date"]): row
        for row in frame.sort_values("event_start_date").to_dict(orient="records")
    }


def _weather_row_for_target(
    lookup: Mapping[pd.Timestamp, Mapping[str, Any]], target: pd.Timestamp
) -> Mapping[str, Any]:
    if target in lookup:
        return lookup[target]
    earlier = [date for date in lookup if date < target]
    if not earlier:
        return {
            "total_beta_wind": 0.155,
            "total_beta_dew": -0.026,
            "ott_beta_wind": 0.031,
            "ott_beta_dew": -0.0052,
            "app_beta_wind": 0.05115,
            "app_beta_dew": -0.00858,
            "arg_beta_wind": 0.0217,
            "arg_beta_dew": -0.00364,
            "putt_beta_wind": 0.05115,
            "putt_beta_dew": -0.00858,
            "fit_max_round_date": pd.NaT,
            "coefficient_source": "frozen_prior_fallback",
        }
    return lookup[max(earlier)]


def _img_signal_arrays(
    prior: pd.DataFrame,
    measurement: str,
    weather_arm: str,
    weather_row: Mapping[str, Any],
    robust_bounds: Mapping[str, Mapping[str, float | int | str]] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    prefix = f"img_clean_{measurement}"
    available = prior[f"{prefix}_signal_available"].fillna(False).astype(bool).to_numpy()
    wind_series = pd.to_numeric(prior["weather_delta_wind"], errors="coerce")
    dew_series = pd.to_numeric(prior["weather_delta_dew"], errors="coerce")
    if weather_arm == "w1_robust_covariate":
        if robust_bounds is None:
            raise ValueError("robust IMG W1 requested without frozen covariate bounds")
        wind = _clip_weather(wind_series, robust_bounds["wind"])
        dew = _clip_weather(dew_series, robust_bounds["dew"])
        wind = np.nan_to_num(wind, nan=0.0)
        dew = np.nan_to_num(dew, nan=0.0)
    else:
        wind = wind_series.fillna(0.0).to_numpy(float)
        dew = dew_series.fillna(0.0).to_numpy(float)
    values: dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        raw = pd.to_numeric(
            prior[f"{prefix}_sg_{category}_field_adjusted_per18"], errors="coerce"
        ).to_numpy(float)
        if weather_arm.startswith("w1"):
            raw = (
                raw
                + float(weather_row[f"{category}_beta_wind"]) * wind
                + float(weather_row[f"{category}_beta_dew"]) * dew
            )
        values[category] = raw
    total = pd.to_numeric(
        prior[f"{prefix}_sg_total_field_adjusted_per18"], errors="coerce"
    ).to_numpy(float)
    if weather_arm.startswith("w1"):
        total = (
            total
            + float(weather_row["total_beta_wind"]) * wind
            + float(weather_row["total_beta_dew"]) * dew
        )
    values["total"] = total
    available &= np.isfinite(total)
    return values, available


def validate_img_history_coverage(features: pd.DataFrame) -> dict[str, Any]:
    """Fail closed unless each IMG coverage fraction has a consistent denominator."""

    audits: dict[str, Any] = {}
    total_violations = 0
    for measurement in ("warmup", "unambiguous"):
        numerator = pd.to_numeric(
            features[f"clean_img_{measurement}_history_rounds"], errors="coerce"
        )
        denominator = pd.to_numeric(
            features[f"clean_img_{measurement}_history_coverage_denominator"],
            errors="coerce",
        )
        coverage = pd.to_numeric(
            features[f"clean_img_{measurement}_history_coverage"], errors="coerce"
        )
        invalid = (
            numerator.isna()
            | denominator.isna()
            | coverage.isna()
            | numerator.lt(0)
            | denominator.lt(0)
            | numerator.gt(denominator)
            | coverage.lt(0)
            | coverage.gt(1)
        )
        violations = int(invalid.sum())
        total_violations += violations
        audits[f"{measurement}_history_coverage_violation_count"] = violations
        audits[f"{measurement}_history_coverage_max"] = float(
            coverage.max() if len(coverage) else 0.0
        )
        audits[f"{measurement}_history_coverage_min"] = float(
            coverage.min() if len(coverage) else 0.0
        )
        audits[f"{measurement}_signal_without_denominator_count"] = int(
            (numerator.gt(0) & denominator.eq(0)).sum()
        )
    audits["history_coverage_violation_count"] = int(total_violations)
    if total_violations:
        raise AssertionError(f"clean IMG history coverage audit failed: {audits}")
    return audits


def validate_img_w1_effect(features: pd.DataFrame) -> dict[str, Any]:
    """Verify weather-normalized IMG histories are not an identity copy of W0."""

    audits: dict[str, Any] = {}
    exposed_total = 0
    changed_total = 0
    for measurement in ("warmup", "unambiguous"):
        exposure = pd.to_numeric(
            features[f"clean_img_{measurement}_prior_nonzero_weather_rounds"],
            errors="coerce",
        ).fillna(0)
        w0 = pd.to_numeric(
            features[f"clean_img_{measurement}_w0_total__ewm_hl80"],
            errors="coerce",
        )
        w1 = pd.to_numeric(
            features[f"clean_img_{measurement}_w1_total__ewm_hl80"],
            errors="coerce",
        )
        comparable = exposure.gt(0) & w0.notna() & w1.notna()
        delta = (w1 - w0).abs()
        changed = comparable & delta.gt(1e-12)
        exposed_count = int(comparable.sum())
        changed_count = int(changed.sum())
        exposed_total += exposed_count
        changed_total += changed_count
        audits[f"{measurement}_w1_exposed_snapshot_rows"] = exposed_count
        audits[f"{measurement}_w1_changed_snapshot_rows"] = changed_count
        audits[f"{measurement}_w1_unchanged_exposed_snapshot_rows"] = int(
            (comparable & ~changed).sum()
        )
        audits[f"{measurement}_w1_max_abs_delta"] = float(
            delta.loc[comparable].max() if comparable.any() else 0.0
        )
    audits["w1_exposed_snapshot_rows"] = int(exposed_total)
    audits["w1_changed_snapshot_rows"] = int(changed_total)
    if exposed_total and not changed_total:
        raise AssertionError(
            "clean IMG W1 histories equal W0 despite nonzero joined weather: "
            f"{audits}"
        )
    return audits


def build_clean_img_features(
    snapshots: pd.DataFrame,
    cross_tour_features: pd.DataFrame,
    img: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
    robust_bounds: Mapping[str, Mapping[str, float | int | str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build clean warm-up and literal/unambiguous IMG histories."""

    lookup = _coefficient_lookup(weather_coefficients)
    by_player = img.groupby("dg_id", sort=False, observed=True).indices
    queries = snapshots.groupby("dg_id", sort=False, observed=True).indices
    output: list[dict[str, Any]] = [dict() for _ in range(len(snapshots))]
    coefficient_violations = 0
    for dg_id, positions_raw in queries.items():
        positions = np.asarray(positions_raw, dtype=int)
        history_positions = by_player.get(dg_id)
        history = (
            img.iloc[np.asarray(history_positions, dtype=int)].sort_values(
                "round_date", kind="mergesort"
            )
            if history_positions is not None
            else img.iloc[:0]
        )
        dates = history["round_date"].to_numpy(dtype="datetime64[ns]")
        for position in positions:
            target = pd.Timestamp(snapshots.at[position, "event_start_date"]).normalize()
            stop = int(np.searchsorted(dates, np.datetime64(target), side="left"))
            prior = history.iloc[:stop]
            prior_dates = dates[:stop]
            weather_row = _weather_row_for_target(lookup, target)
            fit_max = pd.to_datetime(weather_row.get("fit_max_round_date"), errors="coerce")
            if pd.notna(fit_max) and fit_max >= target:
                coefficient_violations += 1
            row: dict[str, Any] = {
                "clean_img_max_source_round_date": (
                    pd.Timestamp(prior_dates.max()) if len(prior_dates) else pd.NaT
                ),
                "clean_img_coefficient_fit_max_date": fit_max,
                "clean_img_coefficient_source": weather_row.get("coefficient_source", "unknown"),
            }
            for measurement in ("warmup", "unambiguous"):
                prefix = f"img_clean_{measurement}"
                for weather_arm in ("w0", "w1", "w1_robust_covariate"):
                    if prior.empty:
                        values = {key: np.array([], dtype=float) for key in (*CATEGORIES, "total")}
                        available = np.array([], dtype=bool)
                    else:
                        values, available = _img_signal_arrays(
                            prior,
                            measurement,
                            weather_arm,
                            weather_row,
                            robust_bounds,
                        )
                    row.update(
                        _aggregate_values(
                            values["total"][available],
                            prior_dates[available],
                            target,
                            f"clean_img_{measurement}_{weather_arm}_total",
                        )
                    )
                    for category in CATEGORIES:
                        residual = (
                            values[category]
                            - CATEGORY_SHARES[category] * values["total"]
                        )
                        row.update(
                            _aggregate_values(
                                residual[available],
                                prior_dates[available],
                                target,
                                f"clean_img_{measurement}_{weather_arm}_cat_{category}",
                            )
                        )
                if prior.empty:
                    signal_available = np.array([], dtype=bool)
                    complete = np.array([], dtype=bool)
                    holes = np.array([], dtype=float)
                else:
                    signal_available = prior[f"{prefix}_signal_available"].fillna(False).astype(bool).to_numpy()
                    complete = prior[f"{prefix}_strict_complete_18"].fillna(False).astype(bool).to_numpy()
                    holes = pd.to_numeric(prior[f"{prefix}_clean_holes"], errors="coerce").fillna(0).to_numpy(float)
                n_signal = int(signal_available.sum())
                if prior.empty:
                    nonzero_weather = np.array([], dtype=bool)
                else:
                    wind = pd.to_numeric(
                        prior["weather_delta_wind"], errors="coerce"
                    ).fillna(0)
                    dew = pd.to_numeric(
                        prior["weather_delta_dew"], errors="coerce"
                    ).fillna(0)
                    nonzero_weather = wind.abs().add(dew.abs()).gt(0).to_numpy()
                denominator = int(
                    cross_tour_features.at[position, "xtour_pga_all_since_img_start"]
                )
                row[f"clean_img_{measurement}_history_rounds"] = n_signal
                row[f"clean_img_{measurement}_prior_nonzero_weather_rounds"] = int(
                    (signal_available & nonzero_weather).sum()
                )
                row[f"clean_img_{measurement}_complete_rounds"] = int(complete.sum())
                row[f"clean_img_{measurement}_history_holes"] = float(holes[signal_available].sum())
                row[f"clean_img_{measurement}_history_coverage_denominator"] = denominator
                row[f"clean_img_{measurement}_history_coverage"] = float(
                    n_signal / denominator if denominator > 0 else 0.0
                )
                if n_signal:
                    row[f"clean_img_{measurement}_days_since_last"] = float(
                        (np.datetime64(target) - prior_dates[signal_available].max())
                        / np.timedelta64(1, "D")
                    )
                else:
                    row[f"clean_img_{measurement}_days_since_last"] = np.nan
            output[position] = row
    features = pd.DataFrame(output, index=snapshots.index)
    source = pd.to_datetime(features["clean_img_max_source_round_date"], errors="coerce")
    target = pd.to_datetime(snapshots["event_start_date"], errors="coerce")
    source_violations = source.notna() & source.ge(target)
    if source_violations.any() or coefficient_violations:
        raise AssertionError(
            "clean IMG strict-prior audit failed: "
            f"sources={int(source_violations.sum())}, coefficients={coefficient_violations}"
        )
    coverage_audit = validate_img_history_coverage(features)
    weather_effect_audit = validate_img_w1_effect(features)
    audit = {
        "img_rows": int(len(img)),
        "img_players": int(img["dg_id"].nunique()),
        "strict_prior_violations": int(source_violations.sum()),
        "coefficient_strict_prior_violations": int(coefficient_violations),
        "warmup_any_history": int(
            features["clean_img_warmup_history_rounds"].gt(0).sum()
        ),
        "literal_any_history": int(
            features["clean_img_unambiguous_history_rounds"].gt(0).sum()
        ),
        **coverage_audit,
        **weather_effect_audit,
    }
    return features, audit


def augment_feature_snapshots(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
    dg_weather_rounds: pd.DataFrame,
    img: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cross_tour, cross_audit = build_cross_tour_features(snapshots, rounds)
    robust_w1, robust_w1_audit, robust_bounds = build_robust_w1_features(
        snapshots, dg_weather_rounds, weather_coefficients
    )
    clean_img, img_audit = build_clean_img_features(
        snapshots, cross_tour, img, weather_coefficients, robust_bounds
    )
    output = pd.concat(
        [snapshots.reset_index(drop=True), cross_tour, robust_w1, clean_img], axis=1
    )
    output["age_sq_asof"] = pd.to_numeric(output["age_asof"], errors="coerce") ** 2
    output["log_xtour_rounds"] = np.log1p(
        pd.to_numeric(output["xtour_combined_r12__n_all"], errors="coerce").fillna(0)
    )
    output["split"] = _split_for_date(output["event_start_date"])
    output["canonical_event_key"] = [
        _canonical_event_key(year, event_id)
        for year, event_id in zip(output["year"], output["event_id"])
    ]
    return output, {
        "cross_tour": cross_audit,
        "robust_w1": robust_w1_audit,
        "clean_img": img_audit,
    }


def _selected_signal_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(prefix)
        and column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES
        and not column.endswith("max_source_date")
    ]


def candidate_feature_columns(frame: pd.DataFrame, candidate: CandidateSpec) -> list[str]:
    if candidate.control or candidate.family in {"dynamic_eb", "dynamic_kalman"}:
        return []
    base = [
        "age_asof",
        "age_sq_asof",
        "pro_time_asof",
        "log_xtour_rounds",
        "xtour_distinct_tours",
        "xtour_has_pga_history",
        "xtour_only_non_pga_history",
        "xtour_days_since_last_round",
        "prior_event_withdrawal",
    ]
    columns = [column for column in base if column in frame]
    img_primary = candidate.source == "img_primary"
    weather_neutral = candidate.weather_arm.startswith("w1")
    cross_tour_arm = (
        "postcut_adj" if candidate.source == "postcut_nuisance" else "r12"
    )
    for tour in TOURS:
        if not img_primary and not (weather_neutral and tour == "pga"):
            columns.extend(
                _selected_signal_columns(frame, f"xtour_{tour}_{cross_tour_arm}__")
            )
        for column in (
            f"xtour_{tour}_all_round_count",
            f"xtour_{tour}_r12_since_img_start",
        ):
            if column in frame:
                columns.append(column)
    weather = candidate.weather_arm
    if candidate.source not in {"postcut_nuisance", "img_primary"}:
        columns.extend(_selected_signal_columns(frame, f"dg_precut_total_{weather}__"))
    if candidate.source in {"total_cat", "total_cat_img", "long_game_img", "literal_full"}:
        categories = ("ott", "app") if candidate.source == "long_game_img" else CATEGORIES
        for category in categories:
            columns.extend(
                _selected_signal_columns(
                    frame, f"dg_precut_cat_{category}_{weather}__"
                )
            )
    measurement = "unambiguous" if candidate.source == "literal_full" else "warmup"
    if candidate.source in {
        "total_img",
        "total_cat_img",
        "long_game_img",
        "literal_full",
        "img_primary",
    }:
        columns.extend(
            _selected_signal_columns(
                frame, f"clean_img_{measurement}_{weather}_total__"
            )
        )
        columns.extend(
            column
            for column in (
                f"clean_img_{measurement}_history_rounds",
                f"clean_img_{measurement}_complete_rounds",
                f"clean_img_{measurement}_history_holes",
                f"clean_img_{measurement}_history_coverage",
                f"clean_img_{measurement}_days_since_last",
            )
            if column in frame
        )
    if candidate.source in {
        "total_cat_img",
        "long_game_img",
        "literal_full",
        "img_primary",
    }:
        categories = ("ott", "app") if candidate.source == "long_game_img" else CATEGORIES
        for category in categories:
            columns.extend(
                _selected_signal_columns(
                    frame,
                    f"clean_img_{measurement}_{weather}_cat_{category}__",
                )
            )
    return list(dict.fromkeys(columns))


def assert_no_legacy_challenger_features(
    frame: pd.DataFrame, candidates: Sequence[CandidateSpec]
) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.control:
            continue
        columns = candidate_feature_columns(frame, candidate)
        forbidden = [
            column
            for column in columns
            if any(token in column.casefold() for token in FORBIDDEN_CHALLENGER_TOKENS)
        ]
        if forbidden:
            usage[candidate.name] = forbidden
    if usage:
        raise AssertionError(f"challengers selected forbidden features: {usage}")
    return usage


def assert_weather_feature_policy(
    frame: pd.DataFrame, candidates: Sequence[CandidateSpec]
) -> dict[str, Any]:
    """Enforce the predeclared weather-neutral feature boundary for W1 arms."""

    raw_pga_usage: dict[str, list[str]] = {}
    allowed_non_pga_usage: dict[str, int] = {}
    cleaned_pga_usage: dict[str, int] = {}
    missing_cleaned_pga: list[str] = []
    w1_candidates = 0
    for candidate in candidates:
        if candidate.control or not candidate.weather_arm.startswith("w1"):
            continue
        w1_candidates += 1
        columns = candidate_feature_columns(frame, candidate)
        raw_pga = [
            column
            for column in columns
            if column.startswith("xtour_pga_")
            and "__" in column
            and column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES
        ]
        if raw_pga:
            raw_pga_usage[candidate.name] = raw_pga
        allowed_non_pga_usage[candidate.name] = sum(
            column.startswith(("xtour_euro_", "xtour_kft_", "xtour_liv_"))
            and "__" in column
            and column.rsplit("__", 1)[-1] in MODEL_SIGNAL_SUFFIXES
            for column in columns
        )
        cleaned_pga_usage[candidate.name] = sum(
            column.startswith(f"dg_precut_total_{candidate.weather_arm}__")
            for column in columns
        )
        if (
            candidate.source != "img_primary"
            and cleaned_pga_usage[candidate.name] == 0
        ):
            missing_cleaned_pga.append(candidate.name)
    if raw_pga_usage:
        raise AssertionError(
            "W1 candidates selected raw PGA cross-tour SG features: "
            f"{raw_pga_usage}"
        )
    if missing_cleaned_pga:
        raise AssertionError(
            "W1 candidates are missing cleaned PGA DG histories: "
            f"{missing_cleaned_pga}"
        )
    return {
        "w1_candidates_audited": int(w1_candidates),
        "raw_pga_performance_feature_violations": 0,
        "missing_cleaned_pga_candidate_count": 0,
        "allowed_raw_non_pga_performance_feature_counts": allowed_non_pga_usage,
        "cleaned_pga_performance_feature_counts": cleaned_pga_usage,
        "non_pga_weather_limitation": (
            "Euro/KFT/LIV histories retain raw adjusted SG because no strict-prior "
            "weather archive exists; PGA performance signals use DG W1 histories."
        ),
        "img_primary_w1_fallback_policy": (
            "DG W1 PGA hl80/ESS plus translated raw Euro/KFT/LIV histories and a "
            "weather-neutral PGA population prior."
        ),
    }


def assert_recency_window_policy(
    frame: pd.DataFrame, candidates: Sequence[CandidateSpec]
) -> dict[str, Any]:
    """Require every fitted candidate to expose 60/180/365/730-day signals."""

    required_windows = (60, 180, 365, 730)
    missing_by_candidate: dict[str, list[int]] = {}
    feature_counts_by_window = {str(days): 0 for days in required_windows}
    candidates_audited = 0
    for candidate in candidates:
        if candidate.control or candidate.family in {"dynamic_eb", "dynamic_kalman"}:
            continue
        candidates_audited += 1
        columns = candidate_feature_columns(frame, candidate)
        present = {
            days
            for days in required_windows
            if any(
                column.endswith((f"__mean_d{days}", f"__n_d{days}"))
                for column in columns
            )
        }
        missing = sorted(set(required_windows) - present)
        if missing:
            missing_by_candidate[candidate.name] = missing
        for days in required_windows:
            feature_counts_by_window[str(days)] += sum(
                column.endswith((f"__mean_d{days}", f"__n_d{days}"))
                for column in columns
            )
    if missing_by_candidate:
        raise AssertionError(
            "candidate recency windows are incomplete: "
            f"{missing_by_candidate}"
        )
    return {
        "required_day_windows": list(required_windows),
        "candidates_audited": int(candidates_audited),
        "missing_candidate_windows": {},
        "selected_feature_counts_by_window": feature_counts_by_window,
    }


def _bind_regression_prediction_envelope(candidate: CandidateSpec) -> CandidateSpec:
    if candidate.family not in REGRESSION_FAMILIES:
        return candidate
    return replace(
        candidate,
        prediction_envelope_applied=True,
        prediction_envelope_method=PREDICTION_ENVELOPE_METHOD,
        prediction_envelope_scale_factor=PREDICTION_ENVELOPE_SCALE_FACTOR,
        prediction_envelope_multiplier=PREDICTION_ENVELOPE_MULTIPLIER,
        raw_regression_component_selection_eligible=False,
    )


def _apply_regression_envelope_admission_gate(
    candidates: Sequence[CandidateSpec],
) -> list[CandidateSpec]:
    evidence, audit = _load_regression_envelope_gate_evidence()
    validated = _validate_regression_envelope_gate_evidence(
        candidates, evidence, audit
    )
    output: list[CandidateSpec] = []
    for candidate in candidates:
        if candidate.name not in NUMERICAL_GATE_INELIGIBLE_CANDIDATES:
            output.append(candidate)
            continue
        if candidate.family not in REGRESSION_FAMILIES:
            raise AssertionError(
                f"numerical-gate exclusion is not a regression: {candidate.name}"
            )
        row = validated.loc[candidate.name]
        reason = (
            f"{REGRESSION_ENVELOPE_GATE_INELIGIBILITY_REASON} "
            f"Frozen evidence: overall_clip_rate={float(row['overall_clip_rate']):.17g}; "
            f"worst_fold={row['worst_fold']}; "
            f"worst_fold_clip_rate={float(row['worst_fold_clip_rate']):.17g}."
        )
        output.append(
            replace(
                candidate,
                selection_eligible=False,
                sensitivity_only=True,
                selection_ineligibility_reason=reason,
            )
        )
    excluded = {
        candidate.name
        for candidate in output
        if candidate.selection_ineligibility_reason.startswith(
            REGRESSION_ENVELOPE_GATE_INELIGIBILITY_REASON
        )
    }
    if excluded != set(NUMERICAL_GATE_INELIGIBLE_CANDIDATES):
        raise AssertionError(
            "catalog did not apply the exact frozen numerical-gate exclusion set"
        )
    remaining_eligible_regressions = [
        candidate
        for candidate in output
        if candidate.family in REGRESSION_FAMILIES and candidate.selection_eligible
    ]
    remaining_evidence = validated.loc[
        [candidate.name for candidate in remaining_eligible_regressions]
    ]
    if remaining_evidence["overall_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
    ).any() or remaining_evidence["worst_fold_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
    ).any():
        raise AssertionError(
            "a remaining selection-eligible regression fails the numerical gate"
        )
    return output


def candidate_catalog() -> list[CandidateSpec]:
    """Return the pre-sealed candidate grid.

    Hyperparameter variants remain separate candidates.  Only the discovery
    market scorecard may choose between them.
    """

    candidates: list[CandidateSpec] = []
    clean_sources = (
        "total",
        "total_cat",
        "total_img",
        "total_cat_img",
        "long_game_img",
        "img_primary",
    )
    for source in clean_sources:
        for alpha in (1.0, 10.0, 100.0):
            candidates.append(
                CandidateSpec(
                    name=f"complete__ridge__{source}__w0__quarterly__a{alpha:g}",
                    family="ridge",
                    source=source,
                    weather_arm="w0",
                    target_arm="observed",
                    refit_schedule="quarterly",
                    alpha=alpha,
                    description=(
                        "Cross-tour R1-R2 adjusted SG with clean source layers. "
                        "For img_primary, IMG is the only fitted performance signal and "
                        "thin coverage blends explicitly to a translated cross-tour prior."
                    ),
                )
            )
    for source in clean_sources:
        for target_arm in ("feature_only", "neutral_target"):
            for alpha in (1.0, 10.0, 100.0):
                candidates.append(
                    CandidateSpec(
                        name=(
                            f"complete__ridge__{source}__w1_{target_arm}"
                            f"__quarterly__a{alpha:g}"
                        ),
                        family="ridge",
                        source=source,
                        weather_arm="w1",
                        target_arm=target_arm,
                        refit_schedule="quarterly",
                        alpha=alpha,
                        description="Weather-neutralized historical features; target arm explicit.",
                    )
                )
    for source in ("total_cat", "total_cat_img"):
        for target_arm in ("feature_only", "neutral_target"):
            for alpha in (1.0, 10.0):
                candidates.append(
                    CandidateSpec(
                        name=(
                            f"complete__ridge__{source}__w1_robust_covariate_{target_arm}"
                            f"__quarterly__a{alpha:g}"
                        ),
                        family="ridge",
                        source=source,
                        weather_arm="w1_robust_covariate",
                        target_arm=target_arm,
                        refit_schedule="quarterly",
                        alpha=alpha,
                        description=(
                            "Pre-2021 covariate-only 0.1%/99.9% weather winsorization; "
                            "never uses outcome-selected weather exclusions."
                        ),
                    )
                )
    for source in ("total", "total_cat_img"):
        for alpha in (0.001, 0.01):
            candidates.append(
                CandidateSpec(
                    name=f"complete__elastic_cd__{source}__w0__quarterly__a{alpha:g}_l10.25",
                    family="elastic_cd",
                    source=source,
                    weather_arm="w0",
                    target_arm="observed",
                    refit_schedule="quarterly",
                    alpha=alpha,
                    l1_ratio=0.25,
                    description=(
                        "Deterministic cyclic coordinate-descent elastic-net sensitivity "
                        "without legacy skill; bounded convergence and stability checks."
                    ),
                )
            )
        for alpha in (1.0, 10.0):
            candidates.append(
                CandidateSpec(
                    name=f"complete__robust_irls__{source}__w0__quarterly__a{alpha:g}",
                    family="robust_irls",
                    source=source,
                    weather_arm="w0",
                    target_arm="observed",
                    refit_schedule="quarterly",
                    alpha=alpha,
                    description="Convergence-enforced deterministic Huber IRLS.",
                )
            )
    for source in ("total", "total_cat_img"):
        candidates.append(
            CandidateSpec(
                name=f"complete__ridge__{source}__w0__annual__a10",
                family="ridge",
                source=source,
                weather_arm="w0",
                target_arm="observed",
                refit_schedule="annual",
                alpha=10.0,
                description="Conservative annual coefficient-refit audit arm.",
            )
        )
    for weather_arm, target_arms in (
        ("w0", ("observed",)),
        ("w1", ("feature_only", "neutral_target")),
    ):
        for source in ("total_cat", "total_cat_img"):
            for target_arm in target_arms:
                for penalty in (2.0, 4.0):
                    target_token = (
                        "" if target_arm == "observed" else f"_{target_arm}"
                    )
                    candidates.append(
                        CandidateSpec(
                            name=(
                                f"complete__ridge_group_penalty__{source}__{weather_arm}"
                                f"{target_token}__quarterly__a10__sgp{penalty:g}"
                            ),
                            family="ridge",
                            source=source,
                            weather_arm=weather_arm,
                            target_arm=target_arm,
                            refit_schedule="quarterly",
                            alpha=10.0,
                            short_game_penalty=penalty,
                            description=(
                                f"Broadie sensitivity: ARG/PUTT coefficients receive "
                                f"{penalty:g}x the ridge penalty of total/OTT/APP features."
                            ),
                        )
                    )
    for alpha in (1.0, 10.0, 100.0):
        candidates.append(
            CandidateSpec(
                name=f"complete__ridge__postcut_nuisance__w0__quarterly__a{alpha:g}",
                family="ridge",
                source="postcut_nuisance",
                weather_arm="w0",
                target_arm="observed",
                refit_schedule="quarterly",
                alpha=alpha,
                description=(
                    "All-round cross-tour totals with strict-prior, partially pooled "
                    "R3-R4 survivor-uplift subtraction."
                ),
            )
        )
    candidates.append(
        CandidateSpec(
            name="complete__ridge__literal_full__w0__quarterly__a10__sensitivity",
            family="ridge",
            source="literal_full",
            weather_arm="w0",
            target_arm="observed",
            refit_schedule="quarterly",
            alpha=10.0,
            selection_eligible=False,
            sensitivity_only=True,
            description="Strict literal/unambiguous IMG accounting sensitivity.",
        )
    )
    for half_life in (40.0, 80.0):
        candidates.append(
            CandidateSpec(
                name=f"complete__dynamic_eb__xtour__w0__event_batch__hl{half_life:g}_p20",
                family="dynamic_eb",
                source="total",
                weather_arm="w0",
                target_arm="observed",
                refit_schedule="event_batch",
                dynamic_half_life=half_life,
                prior_rounds=20.0,
                description="Translated cross-tour empirical-Bayes recency state.",
            )
        )
    for process_variance in (0.0005, 0.0015):
        candidates.append(
            CandidateSpec(
                name=(
                    "complete__dynamic_kalman__xtour__w0__event_batch"
                    f"__q{process_variance:g}"
                ),
                family="dynamic_kalman",
                source="total",
                weather_arm="w0",
                target_arm="observed",
                refit_schedule="event_batch",
                process_variance_per_day=process_variance,
                prior_rounds=20.0,
                description="Translated cross-tour local-level state filter.",
            )
        )
    bound = [_bind_regression_prediction_envelope(candidate) for candidate in candidates]
    return _apply_regression_envelope_admission_gate(bound)


def add_available_robust_weather_candidates(
    candidates: Sequence[CandidateSpec], frame: pd.DataFrame
) -> list[CandidateSpec]:
    """Add feature-only robust-W1 arms when their columns are materialized.

    Robust columns must have been constructed upstream from weather covariates
    alone.  This function merely detects/preserves them and never inspects
    ``weather_excluded`` or any result-conditioned flag.
    """

    arms: set[str] = set()
    pattern = re.compile(r"^dg_precut_total_(.+?)__")
    for column in frame.columns:
        match = pattern.match(column)
        if not match:
            continue
        arm = match.group(1)
        if arm not in {"w0", "w1"} and any(
            token in arm.casefold() for token in ("robust", "covariate", "clipped")
        ):
            arms.add(arm)
    output = list(candidates)
    existing = {candidate.name for candidate in output}
    for arm in sorted(arms):
        for source in ("total_cat",):
            for alpha in (10.0, 100.0):
                candidate = CandidateSpec(
                        name=(
                            f"complete__ridge__{source}__{arm}_feature_only"
                            f"__quarterly__a{alpha:g}"
                        ),
                        family="ridge",
                        source=source,
                        weather_arm=arm,
                        target_arm="feature_only",
                        refit_schedule="quarterly",
                        alpha=alpha,
                        description=(
                            "Covariate-only robust weather normalization supplied by "
                            "the frozen upstream snapshot builder."
                        ),
                    )
                if candidate.name not in existing:
                    output.append(candidate)
                    existing.add(candidate.name)
    return [_bind_regression_prediction_envelope(candidate) for candidate in output]


def _period_cutoff(dates: pd.Series, schedule: str) -> pd.Series:
    dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if schedule == "annual":
        return dates.dt.to_period("Y").dt.start_time
    if schedule == "quarterly":
        return dates.dt.to_period("Q").dt.start_time
    if schedule == "event_batch":
        return dates
    raise ValueError(f"unknown refit schedule {schedule!r}")


def _target_columns(candidate: CandidateSpec) -> tuple[str, str]:
    if candidate.target_arm == "neutral_target":
        if candidate.weather_arm == "w1_robust_covariate":
            return (
                "label_dg_sg_neutral_robust_r1_r2_mean",
                "label_dg_sg_neutral_robust_r1_r2_count",
            )
        return NEUTRAL_LABEL, "label_dg_sg_neutral_r1_r2_count"
    return PRIMARY_LABEL, "label_dg_sg_adj_r1_r2_count"


def _feature_penalty_scales(
    feature_columns: Sequence[str], candidate: CandidateSpec
) -> np.ndarray:
    """Scale columns so ridge imposes an explicit group-specific penalty."""

    scales = np.ones(len(feature_columns), dtype=float)
    if candidate.short_game_penalty <= 1.0:
        return scales
    for index, column in enumerate(feature_columns):
        normalized = column.casefold()
        if "cat_arg" in normalized or "cat_putt" in normalized:
            scales[index] = 1.0 / math.sqrt(float(candidate.short_game_penalty))
    return scales


def _fit_regression(
    candidate: CandidateSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> Any:
    convergence_warnings: list[warnings.WarningMessage] = []
    if candidate.family == "ridge":
        model: Any = Ridge(alpha=candidate.alpha)
    elif candidate.family == "elastic_cd":
        model = ElasticNet(
            alpha=candidate.alpha,
            l1_ratio=candidate.l1_ratio,
            fit_intercept=True,
            precompute=True,
            max_iter=ELASTIC_CD_MAX_ITER,
            tol=ELASTIC_CD_TOL,
            selection="cyclic",
        )
    elif candidate.family == "robust_irls":
        model = HuberIRLS(
            alpha=candidate.alpha,
            epsilon=candidate.huber_epsilon,
            max_iter=80,
            tol=1e-7,
        )
    else:
        raise ValueError(f"not a regression family: {candidate.family}")
    if candidate.family == "elastic_cd":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train, sample_weight=sample_weight)
        convergence_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ]
        if convergence_warnings:
            messages = [str(warning.message) for warning in convergence_warnings]
            raise RuntimeError(
                "ElasticNet coordinate descent emitted a convergence warning for "
                f"{candidate.name}; refusing fit: {messages}"
            )
        iterations = _iteration_count(model)
        if iterations >= ELASTIC_CD_MAX_ITER:
            raise RuntimeError(
                "ElasticNet coordinate descent exhausted the explicit convergence "
                f"budget for {candidate.name}: {iterations}"
            )
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    training_prediction = np.asarray(model.predict(x_train), dtype=float)
    training_wmse = float(
        np.average(np.square(training_prediction - y_train), weights=sample_weight)
    )
    raw_dual_gap = getattr(model, "dual_gap_", None)
    dual_gap = (
        np.asarray(raw_dual_gap, dtype=float).reshape(-1)
        if raw_dual_gap is not None
        else np.asarray([], dtype=float)
    )
    dual_gap_max = (
        float(np.max(np.abs(dual_gap))) if dual_gap.size else float("nan")
    )
    max_abs_coefficient = float(
        max(
            np.max(np.abs(coefficients)) if coefficients.size else 0.0,
            np.max(np.abs(intercept)) if intercept.size else 0.0,
        )
    )
    finite_parts = [coefficients, intercept, training_prediction]
    if dual_gap.size:
        finite_parts.append(dual_gap)
    numerical_finite = bool(
        all(np.isfinite(part).all() for part in finite_parts)
        and np.isfinite(training_wmse)
        and np.isfinite(max_abs_coefficient)
        and (not dual_gap.size or np.isfinite(dual_gap_max))
    )
    if not numerical_finite:
        raise RuntimeError(
            f"regression fit emitted nonfinite numerical state for {candidate.name}"
        )
    if candidate.family == "elastic_cd" and (
        max_abs_coefficient > ELASTIC_CD_MAX_ABS_COEFFICIENT
        or training_wmse > ELASTIC_CD_MAX_TRAIN_WMSE
    ):
        raise RuntimeError(
            "ElasticNet coordinate descent failed the numerical stability gate "
            f"for {candidate.name}: wmse={training_wmse}, "
            f"max_abs_coefficient={max_abs_coefficient}, dual_gap={dual_gap_max}"
        )
    if candidate.family == "elastic_cd":
        model.converged_ = True
    model.training_wmse_ = training_wmse
    model.max_abs_coefficient_ = max_abs_coefficient
    model.dual_gap_max_ = dual_gap_max
    model.numerical_finite_ = numerical_finite
    model.convergence_warning_count_ = len(convergence_warnings)
    return model


def _iteration_count(model: Any) -> int:
    value = getattr(model, "n_iter_", 0)
    if value is None:
        return 0
    array = np.asarray(value)
    return int(array.max()) if array.size else 0


def _prediction_base_columns(test: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_key",
        "canonical_event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "split",
        "history_rounds",
        "days_since_last_round",
        "prior_event_withdrawal",
        "cross_tour_fallback_reason",
        "clean_img_warmup_history_rounds",
        "clean_img_warmup_history_coverage",
    ]
    output = test[[column for column in columns if column in test]].copy()
    output = output.rename(
        columns={
            "clean_img_warmup_history_rounds": "img_history_rounds",
            "clean_img_warmup_history_coverage": "img_history_coverage",
        }
    )
    output["event_start"] = output["event_start_date"]
    output["prediction_asof"] = output["event_start_date"]
    output["chronology_ok"] = True
    return output


def build_retrospective_img_sensitivities(
    snapshots: pd.DataFrame,
    round_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expose future-transform-contaminated IMG ceilings in a separate file."""

    tests = snapshots.loc[snapshots["event_start_date"].ge(PREDICTION_START)].copy()
    by_player = round_panel.groupby("dg_id", sort=False, observed=True).indices
    specifications = {
        "retrospective_img_accounting_v4_full_archive_sensitivity": (
            "img_accounting_sg_total_field_adjusted",
            "full_archive_surface_mapping",
        ),
        "retrospective_img_validated_v2_future_transform_sensitivity": (
            "img_v2_sg_total_signal",
            "future_validated_predictive_transform",
        ),
    }
    rows: dict[str, list[float]] = {name: [] for name in specifications}
    counts: dict[str, list[int]] = {name: [] for name in specifications}
    max_dates: dict[str, list[pd.Timestamp | pd.NaT]] = {
        name: [] for name in specifications
    }
    for event in tests.itertuples(index=False):
        positions = by_player.get(event.dg_id)
        history = (
            round_panel.iloc[np.asarray(positions, dtype=int)].sort_values(
                "round_date", kind="mergesort"
            )
            if positions is not None
            else round_panel.iloc[:0]
        )
        dates = history["round_date"].to_numpy(dtype="datetime64[ns]")
        stop = int(
            np.searchsorted(
                dates, np.datetime64(pd.Timestamp(event.event_start_date)), side="left"
            )
        )
        prior = history.iloc[:stop]
        fallback = pd.to_numeric(
            pd.Series([getattr(event, "xtour_combined_r12__ewm_hl80")]),
            errors="coerce",
        ).iloc[0]
        fallback = float(fallback) if pd.notna(fallback) else 0.0
        for name, (column, _) in specifications.items():
            values = pd.to_numeric(prior[column], errors="coerce")
            valid = values.notna()
            numeric = values[valid].to_numpy(float)
            if len(numeric):
                recency = np.arange(len(numeric) - 1, -1, -1, dtype=float)
                weights = np.exp2(-recency / 80.0)
                prediction = _weighted_mean(numeric, weights)
                source_max = prior.loc[valid, "round_date"].max()
            else:
                prediction = fallback
                source_max = pd.NaT
            rows[name].append(float(prediction))
            counts[name].append(int(len(numeric)))
            max_dates[name].append(source_max)
    pieces: list[pd.DataFrame] = []
    for name, (_, contamination) in specifications.items():
        piece = _prediction_base_columns(tests)
        piece["candidate"] = name
        piece["prediction"] = rows[name]
        piece["prediction_units"] = "sg_per_round_retrospective_sensitivity"
        piece["retrospective_img_history_rounds"] = counts[name]
        piece["retrospective_img_max_source_round_date"] = max_dates[name]
        piece["selection_eligible"] = False
        piece["future_transform_contaminated"] = True
        piece["contamination_reason"] = contamination
        pieces.append(piece)
    output = pd.concat(pieces, ignore_index=True)
    source = pd.to_datetime(
        output["retrospective_img_max_source_round_date"], errors="coerce"
    )
    target = pd.to_datetime(output["event_start_date"], errors="coerce")
    violations = source.notna() & source.ge(target)
    if violations.any():
        raise AssertionError("retrospective IMG sensitivity violates source chronology")
    audit = {
        "selection_eligible": False,
        "future_transform_contaminated": True,
        "strict_source_chronology_violations": int(violations.sum()),
        "rows": int(len(output)),
        "candidates": list(specifications),
        "warning": (
            "Transforms were selected using full-archive/future evidence. These rows "
            "are a retrospective ceiling and cannot win finalist status."
        ),
    }
    return output, audit


def _reliability_flag(frame: pd.DataFrame, uses_img: bool) -> np.ndarray:
    history = pd.to_numeric(frame["xtour_combined_r12__n_all"], errors="coerce").fillna(0)
    stale = pd.to_numeric(frame["xtour_days_since_last_round"], errors="coerce")
    withdrawal = frame["prior_event_withdrawal"].fillna(False).astype(bool)
    reliable = history.ge(20) & stale.fillna(np.inf).le(180) & ~withdrawal
    if uses_img:
        reliable &= (
            pd.to_numeric(frame["clean_img_warmup_history_rounds"], errors="coerce")
            .fillna(0)
            .ge(8)
            & pd.to_numeric(
                frame["clean_img_warmup_history_coverage"], errors="coerce"
            )
            .fillna(0)
            .ge(0.25)
        )
    return reliable.to_numpy(dtype=bool)


def _cross_tour_hierarchical_fallback(
    test: pd.DataFrame,
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
    weather_arm: str,
    prior_rounds: float = 20.0,
) -> np.ndarray:
    """Transparent PGA-scale fallback for thin/missing IMG histories."""

    weather_neutral = weather_arm.startswith("w1")
    offsets, _ = estimate_tour_translations(
        snapshots,
        cutoff,
        pga_weather_arm=weather_arm if weather_neutral else None,
    )
    if weather_neutral:
        pga_value_column = f"dg_precut_total_{weather_arm}__ewm_hl80"
        pga_effective_column = f"dg_precut_total_{weather_arm}__ess_hl80"
        required = {pga_value_column, pga_effective_column}
        missing = sorted(required - set(test.columns))
        if missing:
            raise ValueError(
                "weather-neutral IMG-primary fallback is missing cleaned PGA "
                f"features: {missing}"
            )
        prior_mean = _dynamic_prior_mean(
            snapshots,
            cutoff,
            target_column=NEUTRAL_LABEL,
            count_column="label_dg_sg_neutral_r1_r2_count",
        )
    else:
        pga_value_column = "xtour_pga_r12__ewm_hl80"
        pga_effective_column = "xtour_pga_r12__ess_hl80"
        prior_mean = _dynamic_prior_mean(snapshots, cutoff)
    output = np.full(len(test), prior_mean, dtype=float)
    for output_index, (_, row) in enumerate(test.iterrows()):
        values: list[float] = []
        weights: list[float] = []
        for tour in TOURS:
            value_column = (
                pga_value_column
                if tour == "pga"
                else f"xtour_{tour}_r12__ewm_hl80"
            )
            effective_column = (
                pga_effective_column
                if tour == "pga"
                else f"xtour_{tour}_r12__ess_hl80"
            )
            value = pd.to_numeric(
                pd.Series([row.get(value_column)]), errors="coerce"
            ).iloc[0]
            effective = pd.to_numeric(
                pd.Series([row.get(effective_column)]), errors="coerce"
            ).iloc[0]
            if pd.notna(value) and pd.notna(effective) and effective > 0:
                values.append(float(value) - float(offsets.get(tour, 0.0)))
                weights.append(float(effective))
        if weights:
            output[output_index] = float(
                (prior_rounds * prior_mean + np.dot(values, weights))
                / (prior_rounds + np.sum(weights))
            )
    return output


def _blend_img_primary_prediction(
    raw_prediction: np.ndarray,
    test: pd.DataFrame,
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
    weather_arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use IMG as the primary signal and an explicit cross-tour fallback."""

    history = pd.to_numeric(
        test["clean_img_warmup_history_rounds"], errors="coerce"
    ).fillna(0)
    coverage = pd.to_numeric(
        test["clean_img_warmup_history_coverage"], errors="coerce"
    ).fillna(0)
    complete = pd.to_numeric(
        test["clean_img_warmup_complete_rounds"], errors="coerce"
    ).fillna(0)
    weight = np.minimum.reduce(
        [
            (history / 20.0).clip(0, 1).to_numpy(float),
            (coverage / 0.50).clip(0, 1).to_numpy(float),
            (complete / 8.0).clip(0, 1).to_numpy(float),
        ]
    )
    fallback = _cross_tour_hierarchical_fallback(
        test, snapshots, cutoff, weather_arm=weather_arm
    )
    blended = weight * np.asarray(raw_prediction, dtype=float) + (1.0 - weight) * fallback
    return blended, weight, fallback


def _guard_regression_prediction(
    raw_regression_component: np.ndarray,
    training_prediction: np.ndarray,
    event_equal_weights: np.ndarray,
    candidate: CandidateSpec,
    test: pd.DataFrame,
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | str],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Guard the fitted component, then apply IMG-primary fallback if required."""

    envelope = _fit_prediction_envelope(
        training_prediction, event_equal_weights, candidate
    )
    guarded, clipped, distance = _apply_prediction_envelope(
        raw_regression_component, envelope
    )
    final = guarded.copy()
    img_primary_weight: np.ndarray | None = None
    img_primary_fallback: np.ndarray | None = None
    if candidate.source == "img_primary":
        final, img_primary_weight, img_primary_fallback = _blend_img_primary_prediction(
            guarded,
            test,
            snapshots,
            cutoff,
            weather_arm=candidate.weather_arm,
        )
    if not np.isfinite(final).all():
        raise RuntimeError(
            f"nonfinite final guarded prediction for {candidate.name} at {cutoff}"
        )
    return (
        final,
        guarded,
        clipped,
        distance,
        envelope,
        img_primary_weight,
        img_primary_fallback,
    )


def walk_forward_regression_predictions(
    snapshots: pd.DataFrame,
    candidate: CandidateSpec,
    minimum_training_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = candidate_feature_columns(snapshots, candidate)
    if not feature_columns:
        raise ValueError(f"no features for {candidate.name}")
    target_column, count_column = _target_columns(candidate)
    eligible_test = snapshots.loc[snapshots["event_start_date"].ge(PREDICTION_START)].copy()
    eligible_test["_refit_cutoff"] = _period_cutoff(
        eligible_test["event_start_date"], candidate.refit_schedule
    )
    pieces: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for cutoff, test in eligible_test.groupby("_refit_cutoff", sort=True, observed=True):
        cutoff = pd.Timestamp(cutoff).normalize()
        train = snapshots.loc[
            snapshots["event_end_date"].lt(cutoff)
            & snapshots[target_column].notna()
            & pd.to_numeric(snapshots[count_column], errors="coerce").ge(2)
        ].copy()
        if len(train) < minimum_training_rows:
            logs.append(
                {
                    "candidate": candidate.name,
                    "refit_cutoff": cutoff,
                    "status": "insufficient_training",
                    "training_rows": int(len(train)),
                    "test_rows": int(len(test)),
                }
            )
            continue
        usable = [column for column in feature_columns if train[column].notna().any()]
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        scaler = StandardScaler()
        x_train = imputer.fit_transform(train[usable])
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(imputer.transform(test[usable]))
        penalty_scales = _feature_penalty_scales(usable, candidate)
        x_train = x_train * penalty_scales
        x_test = x_test * penalty_scales
        y_train = pd.to_numeric(train[target_column], errors="coerce").to_numpy(float)
        weights = _event_equal_weights(train)
        started = time.perf_counter()
        model = _fit_regression(candidate, x_train, y_train, weights)
        elapsed = time.perf_counter() - started
        training_prediction = np.asarray(model.predict(x_train), dtype=float)
        raw_regression_component = np.asarray(model.predict(x_test), dtype=float)
        (
            prediction,
            guarded_regression_component,
            prediction_envelope_clipped,
            prediction_envelope_clip_distance,
            prediction_envelope,
            img_primary_weight,
            img_primary_fallback,
        ) = _guard_regression_prediction(
            raw_regression_component,
            training_prediction,
            weights,
            candidate,
            test,
            snapshots,
            cutoff,
        )
        piece = _prediction_base_columns(test)
        piece["candidate"] = candidate.name
        piece["prediction"] = prediction
        piece["finalist_core"] = prediction
        piece["raw_regression_component"] = raw_regression_component
        piece["guarded_regression_component"] = guarded_regression_component
        piece["raw_regression_component_selection_eligible"] = False
        piece["prediction_envelope_method"] = prediction_envelope["method"]
        piece["prediction_envelope_center"] = prediction_envelope["center"]
        piece["prediction_envelope_weighted_mad"] = prediction_envelope[
            "weighted_mad"
        ]
        piece["prediction_envelope_robust_scale"] = prediction_envelope[
            "robust_scale"
        ]
        piece["prediction_envelope_lower"] = prediction_envelope["lower"]
        piece["prediction_envelope_upper"] = prediction_envelope["upper"]
        piece["prediction_envelope_clipped"] = prediction_envelope_clipped
        piece["prediction_envelope_clip_distance"] = (
            prediction_envelope_clip_distance
        )
        piece["prediction_units"] = "sg_per_round"
        piece["training_target_column"] = target_column
        piece["refit_cutoff"] = cutoff
        piece["training_max_event_end"] = train["event_end_date"].max()
        piece["n_features"] = int(len(usable))
        piece["reliability_ok"] = _reliability_flag(
            test,
            uses_img=candidate.source in {
                "total_img",
                "total_cat_img",
                "long_game_img",
                "literal_full",
                "img_primary",
            },
        )
        if img_primary_weight is not None:
            piece["img_primary_weight"] = img_primary_weight
            piece["img_primary_fallback_used"] = img_primary_weight < 1.0
            piece["img_primary_fallback_component"] = img_primary_fallback
        pieces.append(piece)
        logs.append(
            {
                "candidate": candidate.name,
                "refit_cutoff": cutoff,
                "status": "fit",
                "training_rows": int(len(train)),
                "training_events": int(train["event_key"].nunique()),
                "training_max_event_end": train["event_end_date"].max(),
                "test_rows": int(len(test)),
                "features": int(len(usable)),
                "short_game_penalty": candidate.short_game_penalty,
                "short_game_penalized_features": int(
                    np.count_nonzero(penalty_scales < 1.0)
                ),
                "fit_seconds": elapsed,
                "iterations": _iteration_count(model),
                "converged": bool(getattr(model, "converged_", True)),
                "training_wmse": getattr(model, "training_wmse_", np.nan),
                "max_abs_coefficient": getattr(
                    model, "max_abs_coefficient_", np.nan
                ),
                "dual_gap_max": getattr(model, "dual_gap_max_", np.nan),
                "numerical_finite": bool(
                    getattr(model, "numerical_finite_", False)
                ),
                "convergence_warning_count": int(
                    getattr(model, "convergence_warning_count_", 0)
                ),
                "prediction_envelope_method": prediction_envelope["method"],
                "prediction_envelope_center": prediction_envelope["center"],
                "prediction_envelope_weighted_mad": prediction_envelope[
                    "weighted_mad"
                ],
                "prediction_envelope_robust_scale": prediction_envelope[
                    "robust_scale"
                ],
                "prediction_envelope_lower": prediction_envelope["lower"],
                "prediction_envelope_upper": prediction_envelope["upper"],
                "prediction_envelope_clip_count": int(
                    prediction_envelope_clipped.sum()
                ),
                "prediction_envelope_clip_rate": float(
                    prediction_envelope_clipped.mean()
                ),
                "prediction_envelope_max_overshoot": float(
                    prediction_envelope_clip_distance.max(initial=0.0)
                ),
                "raw_regression_component_min": float(
                    raw_regression_component.min()
                ),
                "raw_regression_component_max": float(
                    raw_regression_component.max()
                ),
                "guarded_regression_component_min": float(
                    guarded_regression_component.min()
                ),
                "guarded_regression_component_max": float(
                    guarded_regression_component.max()
                ),
                "final_prediction_min": float(prediction.min()),
                "final_prediction_max": float(prediction.max()),
                "img_primary_fallback_min": (
                    float(img_primary_fallback.min())
                    if img_primary_fallback is not None
                    else np.nan
                ),
                "img_primary_fallback_max": (
                    float(img_primary_fallback.max())
                    if img_primary_fallback is not None
                    else np.nan
                ),
            }
        )
    predictions = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return predictions, pd.DataFrame(logs), feature_columns


def _weighted_robust_location(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    ordered = values[order]
    ordered_weights = weights[order]
    midpoint = 0.5 * ordered_weights.sum()
    median = float(ordered[np.searchsorted(np.cumsum(ordered_weights), midpoint)])
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-8:
        values = np.clip(values, median - 4.4478 * mad, median + 4.4478 * mad)
    return float(np.average(values, weights=weights))


def estimate_tour_translations(
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
    pga_weather_arm: str | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Estimate PGA-scale tour offsets from strictly earlier transition players."""

    prior = snapshots.loc[snapshots["event_start_date"].lt(cutoff)].copy()
    prior = prior.sort_values("event_start_date", kind="mergesort").drop_duplicates(
        "dg_id_key", keep="last"
    )
    pga_prefix = (
        f"dg_precut_total_{pga_weather_arm}"
        if pga_weather_arm is not None and pga_weather_arm.startswith("w1")
        else "xtour_pga_r12"
    )
    required_pga_columns = {f"{pga_prefix}__mean_all", f"{pga_prefix}__n_all"}
    missing_pga_columns = sorted(required_pga_columns - set(prior.columns))
    if missing_pga_columns:
        raise ValueError(
            "tour translation is missing its PGA reference features: "
            f"{missing_pga_columns}"
        )
    pga_mean = pd.to_numeric(prior[f"{pga_prefix}__mean_all"], errors="coerce")
    pga_count = pd.to_numeric(prior[f"{pga_prefix}__n_all"], errors="coerce").fillna(0)
    offsets = {"pga": 0.0}
    transition_counts = {"pga": int(pga_count.gt(0).sum())}
    for tour in TOURS[1:]:
        tour_mean = pd.to_numeric(prior[f"xtour_{tour}_r12__mean_all"], errors="coerce")
        tour_count = pd.to_numeric(prior[f"xtour_{tour}_r12__n_all"], errors="coerce").fillna(0)
        valid = pga_mean.notna() & tour_mean.notna() & pga_count.ge(4) & tour_count.ge(4)
        differences = (tour_mean[valid] - pga_mean[valid]).to_numpy(float)
        weights = np.minimum(pga_count[valid], tour_count[valid]).clip(upper=80).to_numpy(float)
        raw = _weighted_robust_location(differences, weights)
        players = int(valid.sum())
        shrinkage = players / (players + 20.0)
        offsets[tour] = float(raw * shrinkage)
        transition_counts[tour] = players
    return offsets, transition_counts


class ExpandingTourMoments:
    def __init__(self, rounds: pd.DataFrame) -> None:
        self.by_tour: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for tour in TOURS:
            group = rounds.loc[
                rounds["tour"].eq(tour) & rounds["round_num"].isin([1, 2])
            ].sort_values("round_date")
            dates = group["round_date"].to_numpy(dtype="datetime64[ns]")
            values = group["sg_total_adj"].to_numpy(dtype=float)
            self.by_tour[tour] = (
                dates,
                np.concatenate(([0.0], np.cumsum(values))),
                np.concatenate(([0.0], np.cumsum(np.square(values)))),
            )

    def variance(self, tour: str, cutoff: pd.Timestamp) -> float:
        dates, cumulative, cumulative_sq = self.by_tour[tour]
        stop = int(np.searchsorted(dates, np.datetime64(cutoff), side="left"))
        if stop < 100:
            return 5.0
        mean = cumulative[stop] / stop
        variance = cumulative_sq[stop] / stop - mean * mean
        return float(np.clip(variance, 1.0, 12.0))


def _dynamic_prior_mean(
    snapshots: pd.DataFrame,
    cutoff: pd.Timestamp,
    target_column: str = PRIMARY_LABEL,
    count_column: str = "label_dg_sg_adj_r1_r2_count",
) -> float:
    train = snapshots.loc[
        snapshots["event_end_date"].lt(cutoff)
        & snapshots[target_column].notna()
        & pd.to_numeric(snapshots[count_column], errors="coerce").ge(2)
    ]
    if train.empty:
        return 0.0
    values = pd.to_numeric(train[target_column], errors="coerce")
    return float(np.clip(values.mean(), -0.5, 0.5))


def _dynamic_player_history(
    player_history: pd.DataFrame,
    target: pd.Timestamp,
) -> pd.DataFrame:
    dates = player_history["round_date"].to_numpy(dtype="datetime64[ns]")
    stop = int(np.searchsorted(dates, np.datetime64(target), side="left"))
    prior = player_history.iloc[:stop]
    return prior.loc[prior["round_num"].isin([1, 2]) & prior["sg_total_adj"].notna()]


def _empirical_bayes_state(
    history: pd.DataFrame,
    offsets: Mapping[str, float],
    variances: Mapping[str, float],
    prior_mean: float,
    prior_rounds: float,
    half_life: float,
) -> float:
    if history.empty:
        return prior_mean
    values = history["sg_total_adj"].to_numpy(float) - history["tour"].map(offsets).fillna(0).to_numpy(float)
    recency = np.arange(len(values) - 1, -1, -1, dtype=float)
    decay = np.exp2(-recency / half_life)
    noise = history["tour"].map(variances).fillna(5.0).to_numpy(float)
    precision = decay / np.clip(noise, 1e-8, None)
    pga_variance = max(float(variances.get("pga", 5.0)), 1e-8)
    prior_precision = prior_rounds / pga_variance
    return float(
        (np.sum(precision * values) + prior_precision * prior_mean)
        / max(float(precision.sum() + prior_precision), 1e-12)
    )


def _kalman_state(
    history: pd.DataFrame,
    offsets: Mapping[str, float],
    variances: Mapping[str, float],
    prior_mean: float,
    prior_rounds: float,
    process_variance_per_day: float,
) -> float:
    if history.empty:
        return prior_mean
    state = float(prior_mean)
    pga_variance = max(float(variances.get("pga", 5.0)), 1e-8)
    variance = pga_variance / max(prior_rounds, 1e-8)
    previous_date: pd.Timestamp | None = None
    for row in history.itertuples(index=False):
        date = pd.Timestamp(row.round_date).normalize()
        if previous_date is not None:
            variance += process_variance_per_day * max((date - previous_date).days, 0)
        tour = str(row.tour)
        observation = float(row.sg_total_adj) - float(offsets.get(tour, 0.0))
        observation_variance = max(float(variances.get(tour, 5.0)), 1e-8)
        gain = variance / (variance + observation_variance)
        state += gain * (observation - state)
        variance = max((1.0 - gain) * variance, 1e-8)
        previous_date = date
    return float(state)


def dynamic_walk_forward_predictions(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
    candidate: CandidateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests = snapshots.loc[snapshots["event_start_date"].ge(PREDICTION_START)].copy()
    tests["_refit_cutoff"] = _period_cutoff(
        tests["event_start_date"], candidate.refit_schedule
    )
    history_map = {
        dg_id: rounds.iloc[np.asarray(positions, dtype=int)].sort_values(
            "round_date", kind="mergesort"
        )
        for dg_id, positions in rounds.groupby("dg_id", sort=False, observed=True).indices.items()
    }
    moments = ExpandingTourMoments(rounds)
    pieces: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    for cutoff, test in tests.groupby("_refit_cutoff", sort=True, observed=True):
        cutoff = pd.Timestamp(cutoff).normalize()
        offsets, transition_counts = estimate_tour_translations(snapshots, cutoff)
        variances = {tour: moments.variance(tour, cutoff) for tour in TOURS}
        prior_mean = _dynamic_prior_mean(snapshots, cutoff)
        predictions: list[float] = []
        for row in test.itertuples(index=False):
            player_history = history_map.get(row.dg_id, rounds.iloc[:0])
            prior = _dynamic_player_history(player_history, pd.Timestamp(row.event_start_date))
            if candidate.family == "dynamic_eb":
                prediction = _empirical_bayes_state(
                    prior,
                    offsets,
                    variances,
                    prior_mean,
                    candidate.prior_rounds,
                    candidate.dynamic_half_life,
                )
            else:
                prediction = _kalman_state(
                    prior,
                    offsets,
                    variances,
                    prior_mean,
                    candidate.prior_rounds,
                    candidate.process_variance_per_day,
                )
            predictions.append(prediction)
        piece = _prediction_base_columns(test)
        piece["candidate"] = candidate.name
        piece["prediction"] = np.asarray(predictions, dtype=float)
        piece["finalist_core"] = piece["prediction"]
        piece["prediction_units"] = "sg_per_round"
        piece["training_target_column"] = "dynamic_state_no_target_fit"
        piece["refit_cutoff"] = cutoff
        prior_train = snapshots.loc[snapshots["event_end_date"].lt(cutoff)]
        piece["training_max_event_end"] = prior_train["event_end_date"].max()
        piece["n_features"] = 0
        piece["reliability_ok"] = _reliability_flag(test, uses_img=False)
        pieces.append(piece)
        logs.append(
            {
                "candidate": candidate.name,
                "refit_cutoff": cutoff,
                "status": "dynamic_state",
                "training_rows": int(len(prior_train)),
                "training_max_event_end": prior_train["event_end_date"].max(),
                "test_rows": int(len(test)),
                "pga_prior_mean": prior_mean,
                **{f"offset_{tour}": offsets[tour] for tour in TOURS},
                **{f"transition_players_{tour}": transition_counts[tour] for tour in TOURS},
                **{f"observation_variance_{tour}": variances[tour] for tour in TOURS},
            }
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(logs)


def _lagged_low_data_priors(snapshots: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=snapshots.index, dtype=float)
    eligible = snapshots.loc[
        snapshots[PRIMARY_LABEL].notna()
        & pd.to_numeric(snapshots["label_dg_sg_adj_r1_r2_count"], errors="coerce").ge(2)
        & pd.to_numeric(snapshots["xtour_combined_r12__n_all"], errors="coerce").lt(20)
    ].copy()
    for date, positions in snapshots.groupby("event_start_date", sort=True).indices.items():
        date = pd.Timestamp(date).normalize()
        prior = eligible.loc[
            eligible["event_end_date"].lt(date)
            & eligible["event_end_date"].ge(date - pd.Timedelta(days=1095))
        ]
        output.iloc[np.asarray(positions, dtype=int)] = (
            float(prior[PRIMARY_LABEL].mean()) if len(prior) else -2.5
        )
    return output


def current_architecture_controls(
    snapshots: pd.DataFrame,
    core_bridge_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Emit point-in-time v2 approximation and separate retrospective controls."""

    tests = snapshots.loc[snapshots["event_start_date"].ge(PREDICTION_START)].copy()
    prior = _lagged_low_data_priors(snapshots).loc[tests.index]
    count = pd.to_numeric(tests["xtour_combined_r12__n_all"], errors="coerce").fillna(0)
    observed = pd.to_numeric(
        tests["xtour_combined_r12__ewm_hl20"], errors="coerce"
    ).fillna(prior)
    partial = prior + (count / 20.0).clip(0, 1) * (observed - prior)
    legacy = pd.to_numeric(tests.get("legacy_skill_est_asof"), errors="coerce")
    operational = legacy.where(count.ge(20) & legacy.notna(), partial)
    source = np.select(
        [
            count.ge(20) & legacy.notna(),
            count.eq(0),
            count.lt(20),
        ],
        [
            "weekly_pga_core",
            "fixed_cold_start_proxy",
            "tour_prior_shrunk_partial_proxy",
        ],
        default="established_v2_proxy_no_legacy",
    )
    base = _prediction_base_columns(tests)
    v2 = base.copy()
    v2["candidate"] = "current_architecture_pt_v2_approx_control"
    v2["prediction"] = operational.to_numpy(float)
    v2["finalist_core"] = v2["prediction"]
    v2["prediction_units"] = "sg_per_round_control"
    v2["operational_skill_source"] = source
    v2["training_target_column"] = "control_architecture_proxy"
    v2["refit_cutoff"] = v2["event_start_date"]
    v2["training_max_event_end"] = pd.NaT
    v2["n_features"] = 0
    v2["reliability_ok"] = _reliability_flag(tests, uses_img=False)

    controls = [v2]
    if legacy.notna().any():
        legacy_piece = base.loc[legacy.notna()].copy()
        legacy_piece["candidate"] = "legacy_skill_est_pt_current_coefficients_control"
        legacy_piece["prediction"] = legacy.loc[legacy.notna()].to_numpy(float)
        legacy_piece["finalist_core"] = legacy_piece["prediction"]
        legacy_piece["prediction_units"] = "sg_per_round_retrospective_control"
        legacy_piece["training_target_column"] = "retrospective_control"
        legacy_piece["refit_cutoff"] = legacy_piece["event_start_date"]
        legacy_piece["training_max_event_end"] = pd.NaT
        legacy_piece["n_features"] = 0
        legacy_piece["reliability_ok"] = _reliability_flag(
            tests.loc[legacy.notna()], uses_img=False
        )
        controls.append(legacy_piece)

    bridge_rows = 0
    if core_bridge_path.exists():
        bridge_columns = [
            "year",
            "event_id",
            "dg_id",
            "event_start",
            "legacy_core_pred_step2",
        ]
        bridge = pd.read_parquet(core_bridge_path, columns=bridge_columns)
        bridge["event_start"] = pd.to_datetime(bridge["event_start"], errors="coerce").dt.normalize()
        bridge["dg_id"] = pd.to_numeric(bridge["dg_id"], errors="coerce").astype("Int64")
        bridge = bridge.drop_duplicates(["year", "event_id", "dg_id", "event_start"])
        keyed = tests.reset_index().merge(
            bridge,
            left_on=["year", "event_id", "dg_id", "event_start_date"],
            right_on=["year", "event_id", "dg_id", "event_start"],
            how="left",
            validate="one_to_one",
        )
        available = keyed["legacy_core_pred_step2"].notna()
        if available.any():
            bridge_test = tests.loc[keyed.loc[available, "index"].to_numpy()].copy()
            bridge_piece = _prediction_base_columns(bridge_test)
            bridge_piece["candidate"] = "legacy_pred_step2_exact_retrospective_control"
            bridge_piece["prediction"] = keyed.loc[available, "legacy_core_pred_step2"].to_numpy(float)
            bridge_piece["finalist_core"] = bridge_piece["prediction"]
            bridge_piece["prediction_units"] = "sg_per_round_retrospective_control"
            bridge_piece["training_target_column"] = "retrospective_control"
            bridge_piece["refit_cutoff"] = bridge_piece["event_start_date"]
            bridge_piece["training_max_event_end"] = pd.NaT
            bridge_piece["n_features"] = 0
            bridge_piece["reliability_ok"] = _reliability_flag(
                bridge_test, uses_img=False
            )
            controls.append(bridge_piece)
            bridge_rows = int(len(bridge_piece))
    return pd.concat(controls, ignore_index=True), {
        "v2_comparator_status": "strict_point_in_time_architecture_approximation",
        "v2_exact_historical_replay": False,
        "reason": (
            "Contemporaneous weekly v2 fit/acceptance decisions and production snapshots "
            "were not archived. Established PGA uses the available as-of legacy core; "
            "under-20 players use a lagged low-data prior and partial history."
        ),
        "retrospective_exact_bridge_rows": bridge_rows,
    }


def validate_prediction_output(
    predictions: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    expected_keys: pd.DataFrame | Iterable[tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    required = {
        "candidate",
        "event_key",
        "canonical_event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "prediction",
        "prediction_units",
        "prediction_asof",
        "chronology_ok",
        "split",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise AssertionError(f"prediction output missing columns: {missing}")
    duplicate = predictions.duplicated(["candidate", "event_key", "dg_id_key"])
    if duplicate.any():
        raise AssertionError(f"duplicate prediction keys: {int(duplicate.sum())}")
    if not np.isfinite(pd.to_numeric(predictions["prediction"], errors="coerce")).all():
        raise AssertionError("predictions contain missing/non-finite values")
    chronology_flags = predictions["chronology_ok"].fillna(False).astype(bool)
    if not chronology_flags.all():
        raise AssertionError(
            "prediction chronology flags contain failures: "
            f"{int((~chronology_flags).sum())}"
        )
    prediction_asof = pd.to_datetime(predictions["prediction_asof"], errors="coerce")
    event_start = pd.to_datetime(predictions["event_start_date"], errors="coerce")
    asof_violations = prediction_asof.isna() | event_start.isna() | prediction_asof.gt(
        event_start
    )
    if asof_violations.any():
        raise AssertionError(
            "prediction as-of date exceeds event start: "
            f"{int(asof_violations.sum())}"
        )
    fitted = predictions["training_max_event_end"].notna()
    chronology = (
        pd.to_datetime(predictions.loc[fitted, "training_max_event_end"])
        < pd.to_datetime(predictions.loc[fitted, "refit_cutoff"])
    )
    if not chronology.all():
        raise AssertionError("prediction coefficients include an incomplete/current batch")
    refit_cutoff = pd.to_datetime(predictions["refit_cutoff"], errors="coerce")
    refit_violations = refit_cutoff.isna() | refit_cutoff.gt(event_start)
    if refit_violations.any():
        raise AssertionError(
            "prediction refit cutoff exceeds event start: "
            f"{int(refit_violations.sum())}"
        )
    catalog_names = {candidate.name for candidate in candidates if not candidate.control}
    allowed_controls = {candidate.name for candidate in CONTROL_SPECS}
    emitted_names = set(predictions["candidate"].astype(str).unique())
    emitted_challengers = emitted_names - allowed_controls
    missing_challengers = sorted(catalog_names - emitted_challengers)
    extra_challengers = sorted(emitted_challengers - catalog_names)
    if missing_challengers or extra_challengers:
        raise AssertionError(
            "emitted challenger set does not equal frozen catalog; "
            f"missing={missing_challengers}, extra={extra_challengers}"
        )

    spec_by_name = {candidate.name: candidate for candidate in candidates}
    regression_names = {
        candidate.name
        for candidate in candidates
        if candidate.family in REGRESSION_FAMILIES
    }
    envelope_required = {
        "raw_regression_component",
        "guarded_regression_component",
        "raw_regression_component_selection_eligible",
        "prediction_envelope_method",
        "prediction_envelope_center",
        "prediction_envelope_weighted_mad",
        "prediction_envelope_robust_scale",
        "prediction_envelope_lower",
        "prediction_envelope_upper",
        "prediction_envelope_clipped",
        "prediction_envelope_clip_distance",
    }
    envelope_missing = sorted(envelope_required - set(predictions.columns))
    if regression_names and envelope_missing:
        raise AssertionError(
            "regression predictions missing extrapolation-envelope audit columns: "
            f"{envelope_missing}"
        )
    envelope_rows = predictions.loc[
        predictions["candidate"].isin(regression_names)
    ].copy()
    envelope_clip_count = 0
    envelope_max_overshoot = 0.0
    if regression_names:
        numeric_envelope_columns = [
            "raw_regression_component",
            "guarded_regression_component",
            "prediction_envelope_center",
            "prediction_envelope_weighted_mad",
            "prediction_envelope_robust_scale",
            "prediction_envelope_lower",
            "prediction_envelope_upper",
            "prediction_envelope_clip_distance",
        ]
        numeric = envelope_rows[numeric_envelope_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise AssertionError("regression prediction-envelope audit contains nonfinite values")
        invalid_bounds = (
            numeric["prediction_envelope_weighted_mad"].le(0)
            | numeric["prediction_envelope_robust_scale"].le(0)
            | numeric["prediction_envelope_lower"].ge(
                numeric["prediction_envelope_upper"]
            )
            | numeric["prediction_envelope_clip_distance"].lt(0)
        )
        if invalid_bounds.any():
            raise AssertionError(
                "regression prediction-envelope audit has invalid scale/bounds"
            )
        expected_methods = envelope_rows["candidate"].map(
            lambda name: spec_by_name[str(name)].prediction_envelope_method
        )
        if not envelope_rows["prediction_envelope_method"].eq(expected_methods).all():
            raise AssertionError("regression prediction-envelope method differs from catalog")
        raw_eligibility = envelope_rows[
            "raw_regression_component_selection_eligible"
        ].fillna(True).astype(bool)
        if raw_eligibility.any():
            raise AssertionError("raw regression component was marked selection eligible")
        expected_scale_factor = envelope_rows["candidate"].map(
            lambda name: spec_by_name[str(name)].prediction_envelope_scale_factor
        ).to_numpy(float)
        expected_multiplier = envelope_rows["candidate"].map(
            lambda name: spec_by_name[str(name)].prediction_envelope_multiplier
        ).to_numpy(float)
        if not np.allclose(
            numeric["prediction_envelope_robust_scale"],
            numeric["prediction_envelope_weighted_mad"] * expected_scale_factor,
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("prediction-envelope robust scale differs from catalog")
        if not np.allclose(
            numeric["prediction_envelope_lower"],
            numeric["prediction_envelope_center"]
            - expected_multiplier * numeric["prediction_envelope_robust_scale"],
            rtol=0,
            atol=1e-12,
        ) or not np.allclose(
            numeric["prediction_envelope_upper"],
            numeric["prediction_envelope_center"]
            + expected_multiplier * numeric["prediction_envelope_robust_scale"],
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("prediction-envelope bounds differ from catalog")
        constant_columns = [
            "prediction_envelope_method",
            "prediction_envelope_center",
            "prediction_envelope_weighted_mad",
            "prediction_envelope_robust_scale",
            "prediction_envelope_lower",
            "prediction_envelope_upper",
            "raw_regression_component_selection_eligible",
        ]
        varying = (
            envelope_rows.groupby(
                ["candidate", "refit_cutoff"], observed=True, dropna=False
            )[constant_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if varying.any():
            raise AssertionError(
                "prediction-envelope parameters vary inside a candidate/refit fold"
            )
        expected_guarded = np.clip(
            numeric["raw_regression_component"].to_numpy(float),
            numeric["prediction_envelope_lower"].to_numpy(float),
            numeric["prediction_envelope_upper"].to_numpy(float),
        )
        if not np.allclose(
            numeric["guarded_regression_component"],
            expected_guarded,
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("guarded regression component is not clip(raw, lower, upper)")
        guarded_inside = (
            numeric["guarded_regression_component"].ge(
                numeric["prediction_envelope_lower"]
            )
            & numeric["guarded_regression_component"].le(
                numeric["prediction_envelope_upper"]
            )
        )
        if not guarded_inside.all():
            raise AssertionError("guarded regression component lies outside its bounds")
        computed_distance = np.abs(
            numeric["raw_regression_component"]
            - numeric["guarded_regression_component"]
        )
        if not np.allclose(
            computed_distance,
            numeric["prediction_envelope_clip_distance"],
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError("prediction-envelope clip distances are inconsistent")
        clipped = envelope_rows["prediction_envelope_clipped"].fillna(False).astype(bool)
        if not np.array_equal(
            clipped.to_numpy(), computed_distance.gt(0).to_numpy()
        ):
            raise AssertionError("prediction-envelope clip flags are inconsistent")
        non_img = envelope_rows["candidate"].map(
            lambda name: spec_by_name[str(name)].source != "img_primary"
        )
        if not np.allclose(
            pd.to_numeric(envelope_rows.loc[non_img, "prediction"], errors="coerce"),
            numeric.loc[non_img, "guarded_regression_component"],
            rtol=0,
            atol=1e-12,
        ):
            raise AssertionError(
                "non-IMG final prediction differs from guarded regression component"
            )
        img_primary = ~non_img
        if img_primary.any():
            img_required = {
                "img_primary_weight",
                "img_primary_fallback_component",
                "img_primary_fallback_used",
            }
            missing_img = sorted(img_required - set(envelope_rows.columns))
            if missing_img:
                raise AssertionError(
                    "IMG-primary envelope audit is missing blend columns: "
                    f"{missing_img}"
                )
            img_weight = pd.to_numeric(
                envelope_rows.loc[img_primary, "img_primary_weight"], errors="coerce"
            )
            img_fallback = pd.to_numeric(
                envelope_rows.loc[img_primary, "img_primary_fallback_component"],
                errors="coerce",
            )
            img_guarded = numeric.loc[img_primary, "guarded_regression_component"]
            if (
                not np.isfinite(img_weight).all()
                or not np.isfinite(img_fallback).all()
                or not img_weight.between(0.0, 1.0, inclusive="both").all()
            ):
                raise AssertionError("IMG-primary blend weight/fallback is invalid")
            expected_final = (
                img_weight.to_numpy(float) * img_guarded.to_numpy(float)
                + (1.0 - img_weight.to_numpy(float))
                * img_fallback.to_numpy(float)
            )
            if not np.allclose(
                pd.to_numeric(
                    envelope_rows.loc[img_primary, "prediction"], errors="coerce"
                ),
                expected_final,
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError(
                    "IMG-primary final prediction does not equal guard-then-fallback blend"
                )
            fallback_used = envelope_rows.loc[
                img_primary, "img_primary_fallback_used"
            ].fillna(False).astype(bool)
            if not np.array_equal(
                fallback_used.to_numpy(), img_weight.lt(1.0).to_numpy()
            ):
                raise AssertionError("IMG-primary fallback-used flag differs from weight")
        envelope_clip_count = int(clipped.sum())
        envelope_max_overshoot = float(
            numeric["prediction_envelope_clip_distance"].max()
        )

    expected_key_set: set[tuple[str, str]] | None = None
    expected_duplicate_keys = 0
    if expected_keys is not None:
        if isinstance(expected_keys, pd.DataFrame):
            expected_missing_columns = sorted(
                {"event_key", "dg_id_key"} - set(expected_keys.columns)
            )
            if expected_missing_columns:
                raise AssertionError(
                    "expected prediction keys missing columns: "
                    f"{expected_missing_columns}"
                )
            expected_duplicate_keys = int(
                expected_keys.duplicated(["event_key", "dg_id_key"]).sum()
            )
            if expected_duplicate_keys:
                raise AssertionError(
                    "requested test snapshots contain duplicate prediction keys: "
                    f"{expected_duplicate_keys}"
                )
            expected_iterable = expected_keys[["event_key", "dg_id_key"]].itertuples(
                index=False, name=None
            )
        else:
            expected_iterable = expected_keys
        expected_key_set = {
            (str(event_key), str(dg_id_key))
            for event_key, dg_id_key in expected_iterable
        }
        if not expected_key_set:
            raise AssertionError("requested test snapshot key set is empty")

    coverage_by_challenger: dict[str, dict[str, Any]] = {}
    total_missing_keys = 0
    total_extra_keys = 0
    if expected_key_set is not None:
        for candidate_name in sorted(catalog_names):
            group = predictions.loc[predictions["candidate"].eq(candidate_name)]
            actual_key_set = {
                (str(event_key), str(dg_id_key))
                for event_key, dg_id_key in group[
                    ["event_key", "dg_id_key"]
                ].itertuples(index=False, name=None)
            }
            missing_keys = sorted(expected_key_set - actual_key_set)
            extra_keys = sorted(actual_key_set - expected_key_set)
            total_missing_keys += len(missing_keys)
            total_extra_keys += len(extra_keys)
            coverage_by_challenger[candidate_name] = {
                "expected_key_count": int(len(expected_key_set)),
                "emitted_key_count": int(len(actual_key_set)),
                "missing_key_count": int(len(missing_keys)),
                "extra_key_count": int(len(extra_keys)),
                "missing_key_sample": [list(key) for key in missing_keys[:10]],
                "extra_key_sample": [list(key) for key in extra_keys[:10]],
            }
        if total_missing_keys or total_extra_keys:
            offenders = {
                name: audit
                for name, audit in coverage_by_challenger.items()
                if audit["missing_key_count"] or audit["extra_key_count"]
            }
            raise AssertionError(
                "challenger prediction key coverage does not equal requested test "
                f"snapshots; total_missing={total_missing_keys}, "
                f"total_extra={total_extra_keys}, offenders={offenders}"
            )
    return {
        "rows": int(len(predictions)),
        "candidates": int(predictions["candidate"].nunique()),
        "expected_challenger_count": int(len(catalog_names)),
        "emitted_challenger_count": int(len(emitted_challengers)),
        "missing_challengers": missing_challengers,
        "extra_challengers": extra_challengers,
        "expected_keys_per_challenger": (
            int(len(expected_key_set)) if expected_key_set is not None else None
        ),
        "expected_snapshot_duplicate_keys": expected_duplicate_keys,
        "total_missing_challenger_keys": int(total_missing_keys),
        "total_extra_challenger_keys": int(total_extra_keys),
        "coverage_by_challenger": coverage_by_challenger,
        "duplicate_keys": int(duplicate.sum()),
        "nonfinite_predictions": 0,
        "chronology_flag_violations": 0,
        "prediction_asof_violations": int(asof_violations.sum()),
        "refit_cutoff_violations": int(refit_violations.sum()),
        "coefficient_chronology_violations": int((~chronology).sum()),
        "prediction_envelope_rows": int(len(envelope_rows)),
        "prediction_envelope_clip_count": envelope_clip_count,
        "prediction_envelope_clip_rate": (
            float(envelope_clip_count / len(envelope_rows))
            if len(envelope_rows)
            else 0.0
        ),
        "prediction_envelope_max_overshoot": envelope_max_overshoot,
        "raw_regression_component_selection_eligible_rows": 0,
    }


def audit_current_regression_envelope_admission(
    predictions: pd.DataFrame,
    fit_log: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    require_exact_frozen_failure_set: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Recompute envelope admission from this run and fail closed on eligibility."""

    regression_specs = {
        candidate.name: candidate
        for candidate in candidates
        if candidate.family in REGRESSION_FAMILIES
    }
    if not regression_specs:
        return (
            {
                "applies": False,
                "regression_candidate_count": 0,
                "guard_dependent_candidate_count": 0,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )
    required_prediction_columns = {
        "candidate",
        "refit_cutoff",
        "prediction",
        "raw_regression_component",
        "guarded_regression_component",
        "prediction_envelope_clipped",
        "prediction_envelope_clip_distance",
    }
    missing_prediction_columns = sorted(
        required_prediction_columns - set(predictions.columns)
    )
    if missing_prediction_columns:
        raise AssertionError(
            "current-run numerical admission is missing prediction columns: "
            f"{missing_prediction_columns}"
        )
    panel = predictions.loc[
        predictions["candidate"].isin(regression_specs)
    ].copy()
    if panel.empty:
        raise AssertionError("current-run numerical admission has no regression rows")
    if set(panel["candidate"].astype(str)) != set(regression_specs):
        raise AssertionError(
            "current-run numerical admission is missing a selected regression"
        )
    panel["refit_cutoff"] = pd.to_datetime(panel["refit_cutoff"], errors="coerce")
    if panel["refit_cutoff"].isna().any():
        raise AssertionError("current-run numerical admission has invalid refit cutoffs")
    panel["prediction_envelope_clipped"] = _strict_bool_series(
        panel["prediction_envelope_clipped"], "prediction_envelope_clipped"
    )
    for column in (
        "prediction",
        "raw_regression_component",
        "guarded_regression_component",
        "prediction_envelope_clip_distance",
    ):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        if not np.isfinite(panel[column]).all():
            raise AssertionError(
                f"current-run numerical admission has nonfinite {column}"
            )

    folds = (
        panel.groupby(["candidate", "refit_cutoff"], observed=True)
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
        panel.groupby("candidate", observed=True)
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
    overall["fails_overall_gate"] = overall["overall_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
    )
    overall["fails_fold_gate"] = overall["worst_fold_clip_rate"].gt(
        REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
    )
    overall["guard_dependent"] = overall[
        ["fails_overall_gate", "fails_fold_gate"]
    ].any(axis=1)
    overall["selection_eligible"] = overall["candidate"].map(
        {
            name: candidate.selection_eligible
            for name, candidate in regression_specs.items()
        }
    )
    overall["sensitivity_only"] = overall["candidate"].map(
        {
            name: candidate.sensitivity_only
            for name, candidate in regression_specs.items()
        }
    )
    overall["selection_ineligibility_reason"] = overall["candidate"].map(
        {
            name: candidate.selection_ineligibility_reason
            for name, candidate in regression_specs.items()
        }
    )

    required_fit_columns = {
        "candidate",
        "refit_cutoff",
        "test_rows",
        "prediction_envelope_clip_count",
        "prediction_envelope_clip_rate",
    }
    missing_fit_columns = sorted(required_fit_columns - set(fit_log.columns))
    if missing_fit_columns:
        raise AssertionError(
            "current-run numerical admission is missing fit-log columns: "
            f"{missing_fit_columns}"
        )
    logs = fit_log.loc[fit_log["candidate"].isin(regression_specs)].copy()
    logs["refit_cutoff"] = pd.to_datetime(logs["refit_cutoff"], errors="coerce")
    for column in (
        "test_rows",
        "prediction_envelope_clip_count",
        "prediction_envelope_clip_rate",
    ):
        logs[column] = pd.to_numeric(logs[column], errors="coerce")
    if logs[list(required_fit_columns - {"candidate", "refit_cutoff"})].isna().any().any():
        raise AssertionError("current-run fit-log envelope telemetry is incomplete")
    if logs.duplicated(["candidate", "refit_cutoff"]).any():
        raise AssertionError(
            "current-run fit log has duplicate candidate/refit envelope telemetry"
        )
    compare = folds.merge(
        logs[
            [
                "candidate",
                "refit_cutoff",
                "test_rows",
                "prediction_envelope_clip_count",
                "prediction_envelope_clip_rate",
            ]
        ],
        on=["candidate", "refit_cutoff"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compare["_merge"].eq("both").all():
        raise AssertionError(
            "current-run prediction folds and fit-log folds do not match exactly"
        )
    if (
        not compare["rows"].eq(compare["test_rows"]).all()
        or not compare["clips"].eq(compare["prediction_envelope_clip_count"]).all()
        or not np.allclose(
            compare["clip_rate"],
            compare["prediction_envelope_clip_rate"],
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise AssertionError(
            "current-run prediction and fit-log envelope rates disagree"
        )

    observed_failures = set(
        overall.loc[overall["guard_dependent"], "candidate"].astype(str)
    )
    unexpected_failures = observed_failures - set(
        NUMERICAL_GATE_INELIGIBLE_CANDIDATES
    )
    if unexpected_failures:
        raise AssertionError(
            "a current-run regression outside the frozen diagnostic set fails the "
            f"numerical gate: {sorted(unexpected_failures)}"
        )
    eligible_failures = set(
        overall.loc[
            overall["guard_dependent"] & overall["selection_eligible"], "candidate"
        ].astype(str)
    )
    if eligible_failures:
        raise AssertionError(
            "a current-run selection-eligible regression fails the numerical gate: "
            f"{sorted(eligible_failures)}"
        )
    if require_exact_frozen_failure_set and observed_failures != set(
        NUMERICAL_GATE_INELIGIBLE_CANDIDATES
    ):
        raise AssertionError(
            "full current-run numerical failure set differs from the frozen exact "
            f"34-name set: missing={sorted(NUMERICAL_GATE_INELIGIBLE_CANDIDATES - observed_failures)}, "
            f"extra={sorted(observed_failures - NUMERICAL_GATE_INELIGIBLE_CANDIDATES)}"
        )
    summary = {
        "applies": True,
        "market_data_read": False,
        "selection_performed": False,
        "overall_clip_rate_max": REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX,
        "any_refit_fold_clip_rate_max": (
            REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
        ),
        "regression_candidate_count": int(len(regression_specs)),
        "guard_dependent_candidate_count": int(len(observed_failures)),
        "guard_dependent_candidates": sorted(observed_failures),
        "eligible_gate_failure_count": int(len(eligible_failures)),
        "exact_frozen_failure_set_required": bool(
            require_exact_frozen_failure_set
        ),
        "exact_frozen_failure_set_matched": bool(
            observed_failures == set(NUMERICAL_GATE_INELIGIBLE_CANDIDATES)
        ),
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
        "prediction_fit_log_fold_match": True,
    }
    return (
        summary,
        overall.sort_values("candidate").reset_index(drop=True),
        folds.sort_values(["candidate", "refit_cutoff"]).reset_index(drop=True),
    )


def _catalog_frame(
    candidates: Sequence[CandidateSpec],
    feature_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        row = asdict(candidate)
        row["feature_columns"] = json.dumps(list(feature_map.get(candidate.name, [])))
        row["selection_period_start"] = DISCOVERY_START
        row["selection_period_end"] = DISCOVERY_END
        row["sealed_selection_allowed"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_catalog_payload(candidates: Sequence[CandidateSpec]) -> dict[str, Any]:
    return {
        "contract": "base_skill_complete_candidate_catalog_v1",
        "research_only": True,
        "frozen_before_complete_candidate_scoring": True,
        "selection_window_start": DISCOVERY_START.date().isoformat(),
        "selection_window_end": DISCOVERY_END.date().isoformat(),
        "sealed_data_used_for_catalog": False,
        "closing_odds_used_as_features": False,
        "alpha_1_provenance": (
            "Added after the discovery-only DG plumbing screen selected alpha=1 for "
            "total families and before any complete-candidate score; no sealed result "
            "informed this grid."
        ),
        "robust_w1_method": "pre_2021_covariate_only_q001_q999_winsor",
        "group_penalty_parameterization": (
            "After within-fold standardization, ARG/PUTT columns are divided by the "
            "square root of short_game_penalty before ridge fitting and prediction; "
            "alpha/s^2 therefore equals the stated coefficient-penalty multiplier."
        ),
        "post_preflight_regression_envelope_admission_gate": {
            "date": "2026-08-23",
            "market_data_read": False,
            "selection_performed": False,
            "comparison": "strict_greater_than",
            "overall_clip_rate_max": (
                REGRESSION_ENVELOPE_GATE_OVERALL_CLIP_RATE_MAX
            ),
            "any_refit_fold_clip_rate_max": (
                REGRESSION_ENVELOPE_GATE_ANY_FOLD_CLIP_RATE_MAX
            ),
            "audited_regression_candidate_count": 88,
            "overall_gate_failure_count": 9,
            "fold_gate_failure_count": 31,
            "or_gate_failure_count": len(NUMERICAL_GATE_INELIGIBLE_CANDIDATES),
            "non_failure_regression_candidate_count": 54,
            "remaining_selection_eligible_regression_candidate_count": 53,
            "action": (
                "Retain predictions and candidate identities unchanged; mark exact "
                "OR-gate failures selection_eligible=false and sensitivity_only=true."
            ),
            "guard_dependent_candidates": sorted(
                NUMERICAL_GATE_INELIGIBLE_CANDIDATES
            ),
            "evidence": {
                "directory": str(
                    REGRESSION_ENVELOPE_GATE_EVIDENCE_DIR.relative_to(REPO)
                ).replace("\\", "/"),
                "audit_file": REGRESSION_ENVELOPE_GATE_AUDIT_PATH.name,
                "audit_sha256": REGRESSION_ENVELOPE_GATE_AUDIT_SHA256,
                "candidate_file": REGRESSION_ENVELOPE_GATE_CANDIDATE_PATH.name,
                "candidate_sha256": REGRESSION_ENVELOPE_GATE_CANDIDATE_SHA256,
                "fold_file": REGRESSION_ENVELOPE_GATE_FOLD_PATH.name,
                "fold_sha256": REGRESSION_ENVELOPE_GATE_FOLD_SHA256,
                "preflight_builder_source_sha256": (
                    "e15f0e83b740a05ac3ee445b20f512d721a533054c2aaa6f6d14e231477e3dfe"
                ),
                "preflight_prediction_sha256": (
                    "b944e5fc76d72d40544a9e960fa39cbe67632679357e330459282bc7fb85ef87"
                ),
                "preflight_fit_log_sha256": (
                    "627c54a04f51a31c6bd4645dbaf5260f72fe5ee19a2d9bdb50a55ab614c52183"
                ),
                "preflight_manifest_sha256": (
                    "1f94a02d5b926a83d825ea13ad569cde42086216d32cf91ee06360f51714c18e"
                ),
            },
        },
        "pre_score_numerical_protocol_amendment": {
            "date": "2026-08-23",
            "selection_or_market_scores_read": False,
            "superseded_catalog_sha256": (
                "3356629482d178f71029ce66f84c15660212e2fa1457ca9c519c8377dd770f1f"
            ),
            "failed_attempt": "model_completion_v2_provenance",
            "failed_attempt_output_path": (
                "permanent_data/research/base_skill_rebuild_20200101_20260822/"
                "model_completion_v2_provenance"
            ),
            "failed_executed_builder_source_sha256": (
                "2be3785b0469d30bf3ff35f80fbdc8bd82f5205c235ada61284421f69ecbfaa3"
            ),
            "failed_attempt_progress": (
                "Reached candidate 63/92 and failed on the first elastic_sgd arm; "
                "no final prediction panel, manifest, or market score was emitted."
            ),
            "failed_family": "elastic_sgd",
            "reason": (
                "A numerical-only chronological preflight found catastrophic SGD "
                "coefficient/prediction divergence and unreliable convergence status."
            ),
            "replacement_family": "elastic_cd",
            "scope": (
                "Only the four elastic-net sensitivity solver identities changed; "
                "source, alpha, l1_ratio, refit schedule, and candidate count are fixed."
            ),
            "coordinate_descent_preflight": {
                "chronological_fits": 108,
                "all_converged": True,
                "max_iterations": 33201,
                "weighted_mse_range": [4.2494, 4.4374],
                "max_abs_coefficient": 1.3901,
                "aggregate_fit_seconds_approx": 27.0,
                "end_to_end_diagnostic_seconds": 154.55,
            },
            "remaining_nonridge_preflight": {
                "path": (
                    "permanent_data/research/base_skill_rebuild_20200101_20260822/"
                    "model_completion_v3_nonridge_numerical_preflight"
                ),
                "market_data_read": False,
                "candidate_count": 8,
                "expected_keys_per_candidate": 38409,
                "all_keys_complete": True,
                "all_finite": True,
                "chronology_violations": 0,
                "elapsed_seconds": 1392.847,
                "peak_rss_gib": 2.83347,
                "executed_builder_source_sha256": (
                    "6e4a39129f79c13a00a53da421e35402ae4159820728a610021923188af83285"
                ),
                "manifest_sha256": (
                    "5f2b2847fc704de397eda6c36ad5b39abc016d01a091406460dedd8ec06d395a"
                ),
                "prediction_sha256": (
                    "cf2d09b6ea36e620375927cc2c7b9aed40568a37b9a9c5ff1aecc078274ee97f"
                ),
                "fit_log_sha256": (
                    "28b273a4da558a99f2871e2292c710447c0915badbc22e9638a1e43abf334e4e"
                ),
            },
            "regression_extrapolation_amendment": {
                "reason": (
                    "Converged low-alpha ridge and Huber fits produced unsupported "
                    "test-fold tails when sparse/correlated recency features left "
                    "their strict-prior fitted-prediction support."
                ),
                "applies_to": sorted(REGRESSION_FAMILIES),
                "method": PREDICTION_ENVELOPE_METHOD,
                "center": "event_equal_weighted_median_training_fitted_prediction",
                "scale": "1.4826_times_event_equal_weighted_MAD",
                "multiplier": PREDICTION_ENVELOPE_MULTIPLIER,
                "raw_component_selection_eligible": False,
                "img_primary_order": "guard_regression_component_then_blend_fallback",
                "market_or_test_outcomes_used_for_bounds": False,
                "ordinary_support_diagnostic": {
                    "path": (
                        "permanent_data/research/base_skill_rebuild_20200101_20260822/"
                        "model_completion_v3_numerical_support_diagnostic"
                    ),
                    "diagnostic_source_sha256": (
                        "95b6f21fe71bf0a6d26dc09f2d91d9bbf9c7b88676bb4ae61500253054de2901"
                    ),
                    "manifest_sha256": (
                        "2965a29453e4be1031cd9e0b1e1f3cc13156dd0752aa55a19f67d8d0bcda4ed2"
                    ),
                    "summary_sha256": (
                        "e230c68081c3917f276c1ea74502a8be8e9db2c577a40adb079189119792b7ef"
                    ),
                    "elapsed_seconds": 401.500062799998,
                    "python": "3.10.11",
                    "platform": "Windows-10-10.0.22000-SP0",
                    "decision": "rejected_marginal_and_raw_L2_support_projection",
                },
                "mahalanobis_support_diagnostic": {
                    "path": (
                        "permanent_data/research/base_skill_rebuild_20200101_20260822/"
                        "model_completion_v3_mahalanobis_support_diagnostic"
                    ),
                    "diagnostic_source_sha256": (
                        "760160560e8f2891af988c91360f96922c7e27c13ce888da74bba5574b3a1d0a"
                    ),
                    "manifest_sha256": (
                        "ccf132d163e43900f7bf8769b6d268de35c93f19270e2c50d313798bd5411007"
                    ),
                    "summary_sha256": (
                        "2094bfff0df7fb607a8f83a6e5c272bc415d6d4c31656fee8fdf1a6395de0621"
                    ),
                    "elapsed_seconds": 118.41964520000329,
                    "python": "3.10.11",
                    "platform": "Windows-10-10.0.22000-SP0",
                    "decision": (
                        "rejected_as_too_invasive_despite_bounded_numerical_tails"
                    ),
                },
            },
        },
        "estimator_runtime_config": {
            "regression_prediction_envelope": {
                "method": PREDICTION_ENVELOPE_METHOD,
                "center": "event_equal_weighted_median",
                "scale": "event_equal_weighted_MAD",
                "scale_factor": PREDICTION_ENVELOPE_SCALE_FACTOR,
                "multiplier": PREDICTION_ENVELOPE_MULTIPLIER,
                "fit_scope": "strict_prior_training_fitted_predictions_only",
                "zero_or_nonfinite_scale_policy": "raise_and_disqualify",
                "raw_component_selection_eligible": False,
            },
            "elastic_cd": {
                "class": "sklearn.linear_model.ElasticNet",
                "penalty": "elasticnet",
                "selection": "cyclic",
                "precompute": True,
                "tol": ELASTIC_CD_TOL,
                "max_iter": ELASTIC_CD_MAX_ITER,
                "max_abs_coefficient": ELASTIC_CD_MAX_ABS_COEFFICIENT,
                "max_training_wmse": ELASTIC_CD_MAX_TRAIN_WMSE,
                "nonconvergence_policy": "raise_on_warning_or_budget_exhaustion",
                "numerical_instability_policy": "raise_and_disqualify",
            },
            "robust_irls": {
                "loss": "huber",
                "epsilon": 1.35,
                "tol": 1e-7,
                "max_iter": 80,
                "nonconvergence_policy": "raise_and_disqualify",
            },
        },
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def write_frozen_catalog(
    candidates: Sequence[CandidateSpec], output_dir: Path
) -> tuple[Path, str]:
    path = output_dir / "candidate_catalog_frozen.json"
    _atomic_json(path, frozen_catalog_payload(candidates))
    digest = _sha256(path)
    full_catalog_names = {candidate.name for candidate in candidate_catalog()}
    emitted_names = {candidate.name for candidate in candidates}
    if emitted_names == full_catalog_names and digest != EXPECTED_FULL_CATALOG_SHA256:
        raise AssertionError(
            "full frozen candidate catalog hash changed from the pre-score contract; "
            f"expected={EXPECTED_FULL_CATALOG_SHA256}, observed={digest}"
        )
    return path, digest


def estimate_compute(
    snapshot_path: Path,
    img_path: Path,
    db_path: Path,
    candidates: Sequence[CandidateSpec],
) -> dict[str, Any]:
    snapshot_rows = pq.ParquetFile(snapshot_path).metadata.num_rows
    img_rows = pq.ParquetFile(img_path).metadata.num_rows
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        all_tour_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM player_rounds WHERE dg_id IS NOT NULL "
                "AND round_date IS NOT NULL AND sg_total_adj IS NOT NULL"
            ).fetchone()[0]
        )
        pga_events = int(
            connection.execute(
                "SELECT COUNT(DISTINCT year || ':' || event_id) FROM player_rounds "
                "WHERE tour='pga' AND round_date >= '2021-01-01'"
            ).fetchone()[0]
        )
    regression = [
        candidate for candidate in candidates if candidate.family in {"ridge", "elastic_cd", "robust_irls"}
    ]
    approximate_fits = 0
    for candidate in regression:
        periods_per_year = 1 if candidate.refit_schedule == "annual" else 4
        approximate_fits += periods_per_year * 6
    event_batch_regression_fits = len(regression) * pga_events
    return {
        "snapshot_rows": snapshot_rows,
        "all_tour_rounds": all_tour_rows,
        "clean_img_rounds": img_rows,
        "candidate_count": len(candidates),
        "approximate_default_regression_fits": approximate_fits,
        "event_batch_all_regressions_fits": event_batch_regression_fits,
        "estimated_feature_materialization_minutes": "1-4",
        "estimated_default_model_minutes": "10-30",
        "estimated_event_batch_all_regressions": "hours; implemented but not default",
        "estimated_prediction_rows": snapshot_rows * len(candidates),
    }


def _filter_candidates(
    candidates: Sequence[CandidateSpec],
    pattern: str | None,
    schedules: set[str],
) -> list[CandidateSpec]:
    selected = [candidate for candidate in candidates if candidate.refit_schedule in schedules]
    if pattern:
        compiled = re.compile(pattern)
        selected = [candidate for candidate in selected if compiled.search(candidate.name)]
    if not selected:
        raise ValueError("candidate filter selected no models")
    return selected


def run_completion(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution_start: dict[str, Any] | None = None
    execution_start_path = args.output_dir / "execution_start.json"
    if not args.estimate_only and not args.freeze_catalog_only:
        execution_start = capture_execution_start(args)
        _atomic_json(execution_start_path, execution_start)
        print(
            "[completion] execution source sha256="
            f"{execution_start['source_before']['sha256']} pid={execution_start['process_id']}"
        )
    catalog = candidate_catalog()
    estimate = estimate_compute(args.snapshots, args.clean_img, args.all_tour_db, catalog)
    print(json.dumps(estimate, indent=2, sort_keys=True))
    if args.estimate_only:
        return
    memory_tracker = PeakMemoryTracker()
    memory_tracker.start()

    schedules = {value.strip() for value in args.refit_schedules.split(",") if value.strip()}
    selected = _filter_candidates(catalog, args.candidate_pattern, schedules)
    frozen_catalog_path, frozen_catalog_sha = write_frozen_catalog(
        selected, args.output_dir
    )
    print(
        f"[completion] frozen candidate catalog: {len(selected)} candidates, "
        f"sha256={frozen_catalog_sha}"
    )
    if args.freeze_catalog_only:
        return
    snapshots = load_frozen_snapshots(args.snapshots)
    prediction_end = pd.to_datetime(args.prediction_end, errors="coerce")
    if pd.notna(prediction_end):
        prediction_end = pd.Timestamp(prediction_end).normalize()
        snapshots = snapshots.loc[
            snapshots["event_start_date"].le(prediction_end)
        ].copy()
        if snapshots.empty:
            raise ValueError(f"no snapshots on/before prediction end {prediction_end}")
    if args.smoke_events:
        event_dates = (
            snapshots.loc[snapshots["event_start_date"].ge(PREDICTION_START), "event_start_date"]
            .drop_duplicates()
            .sort_values()
            .head(args.smoke_events)
        )
        last_date = event_dates.max()
        snapshots = snapshots.loc[snapshots["event_start_date"].le(last_date)].copy()
        print(f"[completion] bounded smoke through {last_date.date()} ({len(event_dates)} batches)")
    rounds = load_all_tour_rounds(args.all_tour_db)
    max_target = snapshots["event_start_date"].max()
    rounds = rounds.loc[rounds["round_date"].lt(max_target + pd.Timedelta(days=1))].copy()
    if args.reuse_augmented is not None:
        augmented = pd.read_parquet(args.reuse_augmented)
        expected = set(zip(snapshots["event_key"], snapshots["dg_id_key"]))
        observed = set(zip(augmented["event_key"], augmented["dg_id_key"]))
        if expected != observed:
            raise ValueError(
                "reused augmented snapshot keys do not match the requested frozen snapshot slice"
            )
        feature_audit = {
            "reused": True,
            "path": str(args.reuse_augmented.resolve()),
            "sha256": _sha256(args.reuse_augmented),
        }
        img_load_audit = {"reused_augmented_features": True}
        print(f"[completion] reusing augmented features from {args.reuse_augmented}")
        dg_weather_rounds = load_dg_weather_rounds(args.round_panel)
    else:
        dg_weather_rounds = load_dg_weather_rounds(args.round_panel)
        img, img_load_audit = _load_img_with_weather(
            args.clean_img, args.round_panel, dg_weather_rounds
        )
        weather = pd.read_csv(args.weather_coefficients)
        print(
            f"[completion] augmenting {len(snapshots):,} snapshots from "
            f"{len(rounds):,} all-tour and {len(img):,} clean IMG rounds"
        )
        augmented, feature_audit = augment_feature_snapshots(
            snapshots, rounds, dg_weather_rounds, img, weather
        )
    assert_no_legacy_challenger_features(augmented, selected)
    weather_feature_policy_audit = assert_weather_feature_policy(augmented, selected)
    recency_window_audit = assert_recency_window_policy(augmented, selected)
    feature_audit = {
        **feature_audit,
        "weather_feature_policy": weather_feature_policy_audit,
        "recency_window_policy": recency_window_audit,
    }
    augmented_path = args.output_dir / "augmented_feature_snapshots.parquet"
    _atomic_parquet(augmented, augmented_path)
    retrospective_img, retrospective_img_audit = build_retrospective_img_sensitivities(
        augmented, dg_weather_rounds
    )
    retrospective_img_path = args.output_dir / "retrospective_img_sensitivities.parquet"
    retrospective_img_manifest_path = (
        args.output_dir / "retrospective_img_sensitivities_manifest.json"
    )
    _atomic_parquet(retrospective_img, retrospective_img_path)
    _atomic_json(
        retrospective_img_manifest_path,
        {
            **retrospective_img_audit,
            "prediction_file": retrospective_img_path.name,
            "prediction_sha256": _sha256(retrospective_img_path),
            "clean_catalog_sha256": frozen_catalog_sha,
        },
    )

    prediction_pieces: list[pd.DataFrame] = []
    fit_logs: list[pd.DataFrame] = []
    feature_map: dict[str, list[str]] = {}
    for index, candidate in enumerate(selected, start=1):
        print(f"[completion] {index}/{len(selected)} {candidate.name}")
        if candidate.family in {"dynamic_eb", "dynamic_kalman"}:
            predictions, fit_log = dynamic_walk_forward_predictions(
                augmented, rounds, candidate
            )
            feature_map[candidate.name] = []
        else:
            predictions, fit_log, features = walk_forward_regression_predictions(
                augmented,
                candidate,
                minimum_training_rows=args.minimum_training_rows,
            )
            feature_map[candidate.name] = features
        if predictions.empty:
            raise RuntimeError(f"candidate emitted no predictions: {candidate.name}")
        prediction_pieces.append(predictions)
        fit_logs.append(fit_log)

    controls, control_audit = current_architecture_controls(
        augmented, args.core_bridge
    )
    prediction_pieces.append(controls)
    predictions = pd.concat(prediction_pieces, ignore_index=True, sort=False)
    fit_log = pd.concat(fit_logs, ignore_index=True, sort=False) if fit_logs else pd.DataFrame()
    expected_prediction_keys = augmented.loc[
        augmented["event_start_date"].ge(PREDICTION_START),
        ["event_key", "dg_id_key"],
    ].copy()
    prediction_audit = validate_prediction_output(
        predictions, selected, expected_keys=expected_prediction_keys
    )
    selected_regression_names = {
        candidate.name
        for candidate in selected
        if candidate.family in REGRESSION_FAMILIES
    }
    full_regression_names = {
        candidate.name
        for candidate in catalog
        if candidate.family in REGRESSION_FAMILIES
    }
    require_exact_frozen_failure_set = bool(
        selected_regression_names == full_regression_names
        and args.smoke_events == 0
        and pd.notna(prediction_end)
        and prediction_end >= SEALED_END
    )
    (
        regression_envelope_admission_audit,
        regression_envelope_admission_by_candidate,
        regression_envelope_admission_by_fold,
    ) = audit_current_regression_envelope_admission(
        predictions,
        fit_log,
        selected,
        require_exact_frozen_failure_set=require_exact_frozen_failure_set,
    )

    prediction_path = args.output_dir / "player_event_predictions.parquet"
    fit_log_path = args.output_dir / "fit_log.csv"
    catalog_path = args.output_dir / "candidate_catalog.csv"
    coverage_path = args.output_dir / "coverage_by_candidate_year.csv"
    admission_candidate_path = (
        args.output_dir / "regression_envelope_admission_by_candidate.csv"
    )
    admission_fold_path = (
        args.output_dir / "regression_envelope_admission_by_fold.csv"
    )
    _atomic_parquet(predictions, prediction_path)
    _atomic_csv(fit_log, fit_log_path)
    _atomic_csv(_catalog_frame([*selected, *CONTROL_SPECS], feature_map), catalog_path)
    _atomic_csv(
        regression_envelope_admission_by_candidate, admission_candidate_path
    )
    _atomic_csv(regression_envelope_admission_by_fold, admission_fold_path)
    coverage = (
        predictions.assign(year=predictions["event_start_date"].dt.year)
        .groupby(["candidate", "year", "split"], observed=True)
        .agg(
            prediction_rows=("prediction", "size"),
            players=("dg_id_key", "nunique"),
            events=("event_key", "nunique"),
            reliable_rows=("reliability_ok", "sum"),
        )
        .reset_index()
    )
    _atomic_csv(coverage, coverage_path)

    chronology_columns = [
        "max_source_round_date",
        "xtour_max_source_round_date",
        "postcut_nuisance_fit_max_event_end",
        "clean_img_max_source_round_date",
        "clean_img_coefficient_fit_max_date",
    ]
    chronology: dict[str, int] = {}
    for column in chronology_columns:
        if column not in augmented:
            continue
        source = pd.to_datetime(augmented[column], errors="coerce")
        chronology[column] = int(
            (source.notna() & source.ge(augmented["event_start_date"])).sum()
        )
    if any(chronology.values()):
        raise AssertionError(f"final chronology audit failed: {chronology}")

    if execution_start is None:
        raise AssertionError("full completion run is missing entry provenance")
    execution_provenance = verify_execution_unchanged(execution_start)
    execution_end_path = args.output_dir / "execution_end.json"
    _atomic_json(execution_end_path, execution_provenance)

    peak_rss_bytes = memory_tracker.stop()
    elapsed_seconds = time.perf_counter() - started

    output_files = [
        Path(execution_start["executed_source_copy"]["path"]),
        augmented_path,
        prediction_path,
        fit_log_path,
        catalog_path,
        coverage_path,
        admission_candidate_path,
        admission_fold_path,
        frozen_catalog_path,
        retrospective_img_path,
        retrospective_img_manifest_path,
        execution_start_path,
        execution_end_path,
    ]
    manifest = {
        "complete": True,
        "research_only": True,
        "production_files_changed": False,
        "execution_provenance": execution_provenance,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": peak_rss_bytes / (1024**3),
        "contract": {
            "discovery_start": DISCOVERY_START.date().isoformat(),
            "discovery_end": DISCOVERY_END.date().isoformat(),
            "sealed_start": SEALED_START.date().isoformat(),
            "sealed_end": SEALED_END.date().isoformat(),
            "selection_uses_sealed": False,
            "closing_prices_used_as_features": False,
            "challengers_use_legacy_skill": False,
            "primary_target_requires_rounds": 2,
            "prediction_end": (
                prediction_end.date().isoformat()
                if pd.notna(prediction_end)
                else None
            ),
            "sealed_predictions_emitted": bool(
                predictions["event_start_date"].ge(SEALED_START).any()
            ),
        },
        "inputs": execution_provenance["inputs_after"],
        "compute_estimate": estimate,
        "feature_audit": feature_audit,
        "img_load_audit": img_load_audit,
        "control_audit": control_audit,
        "control_catalog": [asdict(candidate) for candidate in CONTROL_SPECS],
        "prediction_audit": prediction_audit,
        "regression_envelope_admission_audit": (
            regression_envelope_admission_audit
        ),
        "retrospective_img_audit": retrospective_img_audit,
        "chronology_audit": chronology,
        "candidates": [asdict(candidate) for candidate in selected],
        "frozen_candidate_catalog_sha256": frozen_catalog_sha,
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in output_files
        },
        "selection_interface": {
            "performed_here": False,
            "required_market_period": "2021-01-01/2022-12-31",
            "sealed_period_can_select": False,
            "freeze_command": (
                "python archive/build_complete_base_skill_candidates.py "
                "--freeze-candidate <DISCOVERY_WINNER> --selection-manifest <JSON>"
            ),
        },
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(
        f"[completion] wrote {len(predictions):,} predictions across "
        f"{predictions['candidate'].nunique()} candidates to {prediction_path}"
    )


def _selection_cutoff(payload: Mapping[str, Any]) -> pd.Timestamp:
    candidates = (
        payload.get("selection_window_end"),
        payload.get("selection_period_end"),
        payload.get("selection_cutoff"),
        payload.get("discovery_end"),
        payload.get("contract", {}).get("discovery_end")
        if isinstance(payload.get("contract"), Mapping)
        else None,
    )
    for value in candidates:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    raise ValueError("selection manifest has no parseable discovery cutoff")


def freeze_finalist(
    prediction_path: Path,
    candidate: str,
    selection_manifest_path: Path,
    output_dir: Path,
) -> None:
    payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    cutoff = _selection_cutoff(payload)
    if cutoff > DISCOVERY_END:
        raise ValueError(
            f"refusing sealed-selected finalist: cutoff {cutoff.date()} exceeds {DISCOVERY_END.date()}"
        )
    if (
        payload.get("selection_uses_sealed") is True
        or payload.get("sealed_data_used_for_selection") is True
    ):
        raise ValueError("refusing selection manifest that declares sealed-sample use")
    declared_finalist = payload.get("finalist_name") or payload.get("selected_candidate")
    if declared_finalist is not None and str(declared_finalist) != candidate:
        raise ValueError(
            f"selection manifest finalist {declared_finalist!r} does not match {candidate!r}"
        )
    predictions = pd.read_parquet(prediction_path)
    finalist_list = payload.get("finalists")
    promotion_gate_passed = bool(payload.get("promotion_gate_passed", True))
    selection_role = str(
        payload.get(
            "selection_role",
            "promoted_finalist" if promotion_gate_passed else "diagnostic_challenger",
        )
    )
    diagnostic_value = payload.get("diagnostic_challenger")
    diagnostic_name = (
        str(diagnostic_value.get("candidate") or diagnostic_value.get("name"))
        if isinstance(diagnostic_value, Mapping)
        else (str(diagnostic_value) if diagnostic_value is not None else None)
    )
    if selection_role == "diagnostic_challenger" or not promotion_gate_passed:
        if promotion_gate_passed:
            raise ValueError("diagnostic_challenger role cannot claim a passed promotion gate")
        if diagnostic_name is not None and diagnostic_name != candidate:
            raise ValueError(
                f"diagnostic challenger {diagnostic_name!r} does not match {candidate!r}"
            )
    elif finalist_list is not None:
        normalized_finalists = {
            str(value.get("candidate") or value.get("name"))
            if isinstance(value, Mapping)
            else str(value)
            for value in finalist_list
        }
        if candidate not in normalized_finalists:
            raise ValueError(f"candidate {candidate!r} is absent from selection finalists")
    selected = predictions.loc[predictions["candidate"].eq(candidate)].copy()
    if selected.empty:
        raise ValueError(f"candidate {candidate!r} not present in predictions")
    if not candidate.startswith("complete__") or "sensitivity" in candidate:
        raise ValueError("only a selection-eligible clean challenger may be frozen")
    control_name = "current_architecture_pt_v2_approx_control"
    control = (
        predictions.loc[
            predictions["candidate"].eq(control_name),
            ["event_key", "dg_id_key", "prediction"],
        ]
        .rename(columns={"prediction": "control_core"})
        .drop_duplicates(["event_key", "dg_id_key"])
    )
    selected = selected.merge(
        control,
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    if not np.isfinite(pd.to_numeric(selected["control_core"], errors="coerce")).all():
        raise ValueError("point-in-time control core is missing for one or more finalist rows")
    selected["event_key"] = selected["canonical_event_key"]
    if selected["event_key"].eq("").any():
        raise ValueError("candidate predictions contain an empty canonical event key")
    columns = [
        "event_key",
        "dg_id",
        "event_start",
        "candidate",
        "finalist_core",
        "control_core",
        "split",
        "prediction_asof",
        "chronology_ok",
    ]
    selected = selected[[column for column in columns if column in selected]]
    if selected.duplicated(["event_key", "dg_id"]).any():
        raise AssertionError("frozen finalist has duplicate player-event keys")
    output_dir.mkdir(parents=True, exist_ok=True)
    finalist_path = output_dir / "player_event_predictions.parquet"
    _atomic_parquet(selected, finalist_path)
    manifest = {
        "contract": "base_skill_frozen_finalist_v1",
        "frozen": True,
        "research_only": True,
        "finalist_name": candidate if promotion_gate_passed else None,
        "selected_model_name": candidate,
        "diagnostic_challenger": candidate if not promotion_gate_passed else None,
        "selection_role": selection_role,
        "promotion_gate_passed": promotion_gate_passed,
        "promotion_gate_failed_reasons": payload.get(
            "promotion_gate_failed_reasons", payload.get("failed_reasons", [])
        ),
        "finalist_column": "finalist_core",
        "prediction_file": finalist_path.name,
        "prediction_sha256": _sha256(finalist_path),
        "prediction_units": "sg_per_round",
        "control_name": control_name,
        "control_units": "sg_per_round",
        "control_core_point_in_time": True,
        "control_core_uses_sealed_future": False,
        "source_prediction_file": str(prediction_path.resolve()),
        "source_prediction_sha256": _sha256(prediction_path),
        "selection_manifest": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": _sha256(selection_manifest_path),
        "selection_window_end": cutoff.date().isoformat(),
        "sealed_data_used_for_selection": False,
        "frozen_before_sealed_scoring": True,
        "uses_closing_odds_as_feature": False,
        "rows": int(len(selected)),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[completion] froze {candidate} to {finalist_path}")


def repair_control_catalog(output_dir: Path) -> None:
    """Backfill explicit audited control rows into an already completed run."""

    catalog_path = output_dir / "candidate_catalog.csv"
    prediction_path = output_dir / "player_event_predictions.parquet"
    manifest_path = output_dir / "manifest.json"
    if not all(path.exists() for path in (catalog_path, prediction_path, manifest_path)):
        raise FileNotFoundError("completed prediction, catalog, and manifest files are required")
    catalog = pd.read_csv(catalog_path)
    existing = set(catalog["name"].astype(str))
    missing = [candidate for candidate in CONTROL_SPECS if candidate.name not in existing]
    if missing:
        catalog = pd.concat(
            [catalog, _catalog_frame(missing, {})], ignore_index=True, sort=False
        )
        _atomic_csv(catalog, catalog_path)
    predictions = pd.read_parquet(prediction_path, columns=["candidate"])
    prediction_names = set(predictions["candidate"].astype(str))
    catalog_names = set(catalog["name"].astype(str))
    unknown = sorted(prediction_names - catalog_names)
    if unknown:
        raise AssertionError(f"prediction candidates remain absent from catalog: {unknown}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["control_catalog"] = [asdict(candidate) for candidate in CONTROL_SPECS]
    manifest.setdefault("outputs", {})[catalog_path.name] = {
        "path": str(catalog_path.resolve()),
        "sha256": _sha256(catalog_path),
    }
    manifest["control_catalog_repaired_post_run"] = bool(missing)
    _atomic_json(manifest_path, manifest)
    print(
        f"[completion] control catalog verified: {len(CONTROL_SPECS)} controls, "
        f"repaired={bool(missing)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--all-tour-db", type=Path, default=DEFAULT_ALL_TOUR_DB)
    parser.add_argument("--clean-img", type=Path, default=DEFAULT_IMG)
    parser.add_argument("--round-panel", type=Path, default=DEFAULT_ROUND_PANEL)
    parser.add_argument(
        "--weather-coefficients", type=Path, default=DEFAULT_WEATHER_COEFFICIENTS
    )
    parser.add_argument("--core-bridge", type=Path, default=DEFAULT_CORE_BRIDGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-augmented",
        type=Path,
        help="Reuse an already-audited augmented snapshot with exactly matching keys.",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--freeze-catalog-only", action="store_true")
    parser.add_argument("--candidate-pattern")
    parser.add_argument(
        "--refit-schedules",
        default="annual,quarterly,event_batch",
        help="Comma-separated schedules. Event-batch regression is supported only by "
        "editing/expanding the frozen catalog; dynamic candidates use it by default.",
    )
    parser.add_argument("--minimum-training-rows", type=int, default=500)
    parser.add_argument(
        "--prediction-end",
        default=SEALED_END.date().isoformat(),
        help="Latest target event start to emit. Use 2022-12-31 for discovery-only build.",
    )
    parser.add_argument(
        "--smoke-events",
        type=int,
        default=0,
        help="Bounded first N discovery event-date batches; 0 means full chronology.",
    )
    parser.add_argument("--freeze-candidate")
    parser.add_argument("--repair-control-catalog", action="store_true")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=DEFAULT_OUTPUT / "player_event_predictions.parquet",
    )
    parser.add_argument(
        "--frozen-finalist-output",
        type=Path,
        default=DEFAULT_OUTPUT / "frozen_finalist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repair_control_catalog:
        repair_control_catalog(args.output_dir)
        return
    if args.freeze_candidate:
        if args.selection_manifest is None:
            raise ValueError("--freeze-candidate requires --selection-manifest")
        freeze_finalist(
            args.prediction_file,
            args.freeze_candidate,
            args.selection_manifest,
            args.frozen_finalist_output,
        )
        return
    run_completion(args)


if __name__ == "__main__":
    main()

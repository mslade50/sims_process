"""Research-only runner for the frozen 2023 all-round base-skill rebuild.

This file is deliberately isolated from the weekly simulation pipeline.  It has
two phases:

``train``
    Build the R1--R4 target arms, make strictly chronological 2023 predictions
    (January--March warm-up, April onward out-of-event), and write the frozen
    candidate catalog and target diagnostics used by the market selector.

``evaluate``
    Read a selected-specification JSON, fit that *one* specification on 2023,
    and predict 2024--2026 from label-free feature/context projections.  The
    evaluation phase never loads a 2024--2026 outcome column.

The governing contract is ``ALL_ROUND_BASE_SKILL_REBUILD_CONTRACT_2026_08.md``.
No output path defaults to a production directory and this module is not
imported by production code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive.all_round_target_design import (  # noqa: E402
    OUTCOME_COLUMN,
    build_all_round_targets,
    build_mar_round_rows,
)


RESEARCH_ROOT = (
    REPO_ROOT
    / "permanent_data"
    / "research"
    / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_FEATURES = (
    RESEARCH_ROOT / "model_eval_clean_dg_plumbing_v4" / "feature_snapshots.parquet"
)
DEFAULT_AUGMENTED_FEATURES = (
    RESEARCH_ROOT
    / "model_completion_v3_provenance"
    / "augmented_feature_snapshots.parquet"
)
DEFAULT_ROUNDS = RESEARCH_ROOT / "player_round_panel.parquet"
DEFAULT_WEATHER = (
    RESEARCH_ROOT / "model_eval_clean_dg_plumbing_v4" / "weather_coefficients.csv"
)
DEFAULT_BRIDGE = RESEARCH_ROOT / "core_bridge" / "legacy_core_bridge.parquet"
DEFAULT_CASCADE = (
    RESEARCH_ROOT
    / "model_eval_round_cascade_v2_full_20260823"
    / "cascade_player_round_predictions.parquet"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "all_round_base_skill_rebuild_2023_train"
CONTRACT_PATH = REPO_ROOT / "ALL_ROUND_BASE_SKILL_REBUILD_CONTRACT_2026_08.md"
EXPECTED_CONTRACT_SHA256 = (
    "4091a0192a959640a9dc88ff6b348759cf3052ed7aed3a116b5989019111374c"
)

TRAIN_YEAR = 2023
EVALUATION_YEARS = (2024, 2025, 2026)
CHRONOLOGICAL_PREDICTION_START = pd.Timestamp("2023-04-01")
CALIBRATION_END = pd.Timestamp("2023-06-30")
SELECTION_START = pd.Timestamp("2023-07-01")
SELECTION_END = pd.Timestamp("2023-12-31")
EXCLUDED_TARGET_EVENT_DATE = pd.Timestamp("2023-12-14")

DG_SIGNAL_SUFFIXES = (
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
IMG_SIGNAL_SUFFIXES = (
    "ewm_hl20",
    "ess_hl20",
    "ewm_hl80",
    "ess_hl80",
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
COMMON_CONTEXT = (
    "history_rounds",
    "days_since_last_round",
    "prior_event_withdrawal",
)
IMG_CONTEXT = (
    "clean_img_warmup_history_rounds",
    "clean_img_warmup_complete_rounds",
    "clean_img_warmup_history_holes",
    "clean_img_warmup_history_coverage",
    "clean_img_warmup_days_since_last",
)

FEATURE_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "T": ("total",),
    "T_APP": ("total", "app"),
    "T_BALL": ("total", "ott", "app"),
    "T_FULL": ("total", "ott", "app", "arg", "putt"),
    "T_IMG": ("total",),
    "T_BALL_IMG": ("total", "ott", "app"),
    "T_FULL_IMG": ("total", "ott", "app", "arg", "putt"),
}
IMG_FAMILIES = frozenset({"T_IMG", "T_BALL_IMG", "T_FULL_IMG"})
TARGET_LABELS = {"T0": "raw_obs4_target", "T1": "survcomp_obs4_target"}
TARGET_COUNTS = {"T0": "raw_obs4_count", "T1": "survcomp_obs4_count"}
TARGET_WEIGHTS = {"T0": "raw_obs4_weight", "T1": "survcomp_obs4_weight"}


@dataclass(frozen=True)
class CandidateSpec:
    """One immutable regression specification in the locked menu."""

    candidate: str
    target_arm: str
    weather_arm: str
    feature_family: str
    estimator: str
    alpha: float
    selectable: bool
    diagnostic_only: bool = False
    epsilon: float | None = None
    l1_ratio: float | None = None
    max_leaf_nodes: int | None = None
    learning_rate: float | None = None


@dataclass
class FittedRegressor:
    imputer: SimpleImputer
    scaler: StandardScaler
    estimator: Any
    target_arm: str
    prediction_offset: float
    feature_columns: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_name(
    target: str,
    weather: str,
    family: str,
    estimator: str,
    alpha: float,
) -> str:
    alpha_text = str(int(alpha)) if float(alpha).is_integer() else f"{alpha:g}"
    return f"allr__{target.lower()}__{weather.lower()}__{family.lower()}__{estimator}__a{alpha_text}"


def build_candidate_catalog(*, include_diagnostics: bool = True) -> list[CandidateSpec]:
    """Return the exact revised-contract menu (34 selectable, two T0 checks)."""

    specs: list[CandidateSpec] = []

    def add(
        target: str,
        weather: str,
        family: str,
        estimator: str,
        alpha: float,
        *,
        selectable: bool = True,
        epsilon: float | None = None,
    ) -> None:
        specs.append(
            CandidateSpec(
                candidate=_candidate_name(target, weather, family, estimator, alpha),
                target_arm=target,
                weather_arm=weather,
                feature_family=family,
                estimator=estimator,
                alpha=float(alpha),
                selectable=selectable,
                diagnostic_only=not selectable,
                epsilon=epsilon,
            )
        )

    for family in FEATURE_FAMILY_SIGNALS:
        for alpha in (100.0, 1000.0, 3000.0):
            add("T2", "W0", family, "ridge", alpha)
    for family in ("T", "T_BALL", "T_FULL"):
        add("T2", "W1", family, "ridge", 1000.0)
    for family in ("T", "T_BALL", "T_FULL"):
        add("T2", "W0", family, "huber_irls", 1000.0, epsilon=1.35)
    for family in ("T", "T_BALL", "T_FULL", "T_FULL_IMG"):
        add("T1", "W0", family, "ridge", 1000.0)
    for family in ("T", "T_BALL", "T_FULL"):
        add("T1", "W1", family, "ridge", 1000.0)
    if include_diagnostics:
        for family in ("T", "T_FULL"):
            add("T0", "W0", family, "ridge", 1000.0, selectable=False)

    names = [spec.candidate for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("candidate catalog contains duplicate names")
    selectable = sum(spec.selectable for spec in specs)
    expected = 34
    if selectable != expected:
        raise AssertionError(f"locked catalog must contain {expected} selectable candidates")
    return specs


def _stable_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(number)) if np.isfinite(number) and number.is_integer() else text


def _parquet_columns(path: str | Path) -> list[str]:
    return list(pq.ParquetFile(Path(path)).schema.names)


def _safe_feature_projection(path: str | Path, *, augmented: bool) -> list[str]:
    available = set(_parquet_columns(path))
    keys = [
        "event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        "event_end_date",
        "event_id",
        "year",
        "player_name",
        "event_name",
        "max_source_round_date",
    ]
    if not augmented:
        prefixes = ("dg_total_w0__", "dg_total_w1__", "dg_cat_")
        context = list(COMMON_CONTEXT)
        columns = keys + context + [
            column
            for column in available
            if column.startswith(prefixes)
            and not "precut" in column.casefold()
        ]
    else:
        prefixes = ("clean_img_warmup_w0_", "clean_img_warmup_w1_")
        chronology = (
            "clean_img_max_source_round_date",
            "clean_img_coefficient_fit_max_date",
        )
        columns = keys + list(IMG_CONTEXT) + list(chronology) + [
            column for column in available if column.startswith(prefixes)
        ]
    columns = [column for column in dict.fromkeys(columns) if column in available]
    forbidden = ("label", "legacy", "cascade", "odds", "outcome", "skill_est")
    leaked = [column for column in columns if any(token in column.casefold() for token in forbidden)]
    if leaked:
        raise AssertionError(f"feature projection contains prohibited columns: {leaked}")
    return columns


def load_feature_frame(
    feature_path: str | Path = DEFAULT_FEATURES,
    augmented_path: str | Path = DEFAULT_AUGMENTED_FEATURES,
    *,
    years: Iterable[int] = (TRAIN_YEAR, *EVALUATION_YEARS),
) -> pd.DataFrame:
    """Load only point-in-time predictor columns, never cached labels/controls."""

    years_set = {int(year) for year in years}
    base_columns = _safe_feature_projection(feature_path, augmented=False)
    base = pd.read_parquet(
        feature_path,
        columns=base_columns,
        filters=[("year", "in", sorted(years_set))],
    )
    base["event_start_date"] = pd.to_datetime(base["event_start_date"], errors="coerce").dt.normalize()
    derived_year = base["event_start_date"].dt.year
    base = base.loc[derived_year.isin(years_set)].copy()
    base["dg_id_key"] = base["dg_id_key"].map(_stable_id)
    if base.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("base feature snapshot has duplicate player-event keys")

    augmented_columns = _safe_feature_projection(augmented_path, augmented=True)
    augmented = pd.read_parquet(
        augmented_path,
        columns=augmented_columns,
        filters=[("year", "in", sorted(years_set))],
    )
    augmented["event_start_date"] = pd.to_datetime(
        augmented["event_start_date"], errors="coerce"
    ).dt.normalize()
    augmented = augmented.loc[augmented["event_start_date"].dt.year.isin(years_set)].copy()
    augmented["dg_id_key"] = augmented["dg_id_key"].map(_stable_id)
    join_keys = ["event_key", "dg_id_key"]
    if augmented.duplicated(join_keys).any():
        raise ValueError("augmented feature snapshot has duplicate player-event keys")
    extra_columns = [column for column in augmented.columns if column not in base.columns]
    frame = base.merge(
        augmented[join_keys + extra_columns],
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    _assert_feature_chronology(frame)
    return frame.sort_values(
        ["event_start_date", "event_key", "dg_id_key"], kind="mergesort"
    ).reset_index(drop=True)


def _assert_feature_chronology(frame: pd.DataFrame) -> None:
    target = pd.to_datetime(frame["event_start_date"], errors="coerce")
    date_columns = [
        column
        for column in (
            "max_source_round_date",
            "clean_img_max_source_round_date",
            "clean_img_coefficient_fit_max_date",
        )
        if column in frame
    ]
    violations: dict[str, int] = {}
    for column in date_columns:
        source = pd.to_datetime(frame[column], errors="coerce")
        bad = source.notna() & source.ge(target)
        if bad.any():
            violations[column] = int(bad.sum())
    if violations:
        raise AssertionError(f"feature chronology violations: {violations}")


def feature_columns_for_spec(frame: pd.DataFrame, spec: CandidateSpec) -> list[str]:
    """Resolve, and strictly validate, the declared feature-family columns."""

    if spec.feature_family not in FEATURE_FAMILY_SIGNALS:
        raise ValueError(f"unknown feature family {spec.feature_family!r}")
    weather = spec.weather_arm.casefold()
    signals = FEATURE_FAMILY_SIGNALS[spec.feature_family]
    columns = list(COMMON_CONTEXT)
    for signal in signals:
        prefix = f"dg_total_{weather}__" if signal == "total" else f"dg_cat_{signal}_{weather}__"
        columns.extend(f"{prefix}{suffix}" for suffix in DG_SIGNAL_SUFFIXES)
    if spec.feature_family in IMG_FAMILIES:
        columns.extend(column for column in IMG_CONTEXT if column in frame)
        img_signals = signals
        for signal in img_signals:
            prefix = f"clean_img_warmup_{weather}_{'total' if signal == 'total' else 'cat_' + signal}__"
            columns.extend(f"{prefix}{suffix}" for suffix in IMG_SIGNAL_SUFFIXES)
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(
            f"{spec.candidate} cannot satisfy its declared feature columns: {missing[:20]}"
        )
    forbidden = ("legacy", "cascade", "odds", "label", "outcome", "precut")
    leaked = [column for column in columns if any(token in column.casefold() for token in forbidden)]
    if leaked:
        raise AssertionError(f"prohibited challenger features: {leaked}")
    return columns


ROUND_TARGET_COLUMNS = (
    "event_key",
    "event_start_date",
    "event_end_date",
    "tour",
    "year",
    "event_id",
    "dg_id",
    "round_num",
    "round_date",
    OUTCOME_COLUMN,
    "weather_delta_wind",
    "weather_delta_dew",
)


def load_training_rounds(
    path: str | Path = DEFAULT_ROUNDS,
    *,
    year: int = TRAIN_YEAR,
) -> pd.DataFrame:
    """Load source rounds through ``year`` for strict-prior target anchors.

    Coefficients are still fit only to 2023 targets.  Earlier rounds are read
    solely because T1's candidate-independent anchor at a 2023 event must have
    the player's genuine pre-2023 history.
    """

    available = set(_parquet_columns(path))
    missing = [column for column in ROUND_TARGET_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"round panel missing target columns: {missing}")
    frame = pd.read_parquet(
        path,
        columns=list(ROUND_TARGET_COLUMNS),
        filters=[("year", "<=", int(year))],
    )
    observed_year = pd.to_numeric(frame["year"], errors="coerce")
    if not observed_year.le(int(year)).all():
        raise AssertionError("target-source predicate admitted a future year")
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["event_end_date"] = pd.to_datetime(
        frame["event_end_date"], errors="coerce"
    ).dt.normalize()
    frame["round_date"] = pd.to_datetime(frame["round_date"], errors="coerce").dt.normalize()
    frame["dg_id_key"] = frame["dg_id"].map(_stable_id)
    return frame


def load_frozen_weather_coefficients(
    path: str | Path = DEFAULT_WEATHER,
) -> pd.DataFrame:
    required = {
        "event_start_date",
        "fit_max_round_date",
        "coefficient_source",
        "total_beta_wind",
        "total_beta_dew",
    }
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weather coefficient table missing: {missing}")
    frame["event_start_date"] = pd.to_datetime(
        frame["event_start_date"], errors="coerce"
    ).dt.normalize()
    frame["fit_max_round_date"] = pd.to_datetime(
        frame["fit_max_round_date"], errors="coerce"
    ).dt.normalize()
    frame = frame.loc[frame["event_start_date"].dt.year.le(TRAIN_YEAR)].copy()
    if frame.duplicated("event_start_date").any():
        raise ValueError("weather coefficient table has duplicate event dates")
    return frame


def neutralize_training_rounds(
    rounds: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Neutralize target SG with the exact pre-2021-frozen transform."""

    coefficients = weather_coefficients[
        [
            "event_start_date",
            "fit_max_round_date",
            "coefficient_source",
            "total_beta_wind",
            "total_beta_dew",
        ]
    ].copy()
    output = rounds.merge(
        coefficients,
        on="event_start_date",
        how="left",
        validate="many_to_one",
    )
    missing = output["total_beta_wind"].isna() | output["total_beta_dew"].isna()
    if missing.any():
        dates = sorted(output.loc[missing, "event_start_date"].dropna().unique())
        raise ValueError(f"missing frozen weather coefficients for dates: {dates[:10]}")
    used = output.loc[output["event_start_date"].dt.year.ge(2021)]
    if not used["coefficient_source"].eq("frozen_pre_2021").all():
        values = sorted(used["coefficient_source"].dropna().astype(str).unique())
        raise AssertionError(f"W1 used non-frozen coefficient sources: {values}")
    fit_date = pd.to_datetime(output["fit_max_round_date"], errors="coerce")
    chronology_bad = fit_date.notna() & fit_date.ge(pd.Timestamp("2021-01-01"))
    if chronology_bad.any():
        raise AssertionError("W1 coefficient fit reaches 2021 or later")
    observed = pd.to_numeric(output[OUTCOME_COLUMN], errors="coerce")
    wind = pd.to_numeric(output["weather_delta_wind"], errors="coerce").fillna(0.0)
    dew = pd.to_numeric(output["weather_delta_dew"], errors="coerce").fillna(0.0)
    output[OUTCOME_COLUMN] = (
        observed
        + pd.to_numeric(output["total_beta_wind"], errors="coerce") * wind
        + pd.to_numeric(output["total_beta_dew"], errors="coerce") * dew
    )
    output.loc[observed.isna(), OUTCOME_COLUMN] = np.nan
    return output, {
        "weather_rows": int(len(output)),
        "weather_rows_with_observed_sg": int(observed.notna().sum()),
        "weather_rows_missing_both_covariates": int(
            (
                pd.to_numeric(output["weather_delta_wind"], errors="coerce").isna()
                & pd.to_numeric(output["weather_delta_dew"], errors="coerce").isna()
            ).sum()
        ),
        "coefficient_sources": sorted(output["coefficient_source"].dropna().unique()),
        "coefficient_fit_max": str(fit_date.max().date()) if fit_date.notna().any() else None,
        "positive_wind_delta_definition": "player_wind_minus_field_wind",
        "neutralization_formula": "adjusted_sg + beta_wind*delta_wind + beta_dew*delta_dew",
    }


def build_training_targets(
    rounds: pd.DataFrame,
    weather_coefficients: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    """Build W0/W1 event targets and MAR round rows from 2023 only."""

    target_frames: dict[str, pd.DataFrame] = {}
    mar_frames: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for weather_arm, source in (
        ("W0", rounds.copy()),
        ("W1", neutralize_training_rounds(rounds, weather_coefficients)[0]),
    ):
        event_targets, target_audit = build_all_round_targets(source)
        mar_rows, mar_audit = build_mar_round_rows(source)
        event_targets["dg_id_key"] = event_targets["dg_id_key"].map(_stable_id)
        mar_rows["dg_id_key"] = mar_rows["dg_id_key"].map(_stable_id)
        # Earlier rows were anchor/history sources only.  Regression labels are
        # physically reduced to calendar 2023 here.
        event_targets = event_targets.loc[
            pd.to_datetime(event_targets["event_start_date"], errors="coerce")
            .dt.year.eq(TRAIN_YEAR)
        ].copy()
        mar_rows = mar_rows.loc[
            pd.to_datetime(mar_rows["event_start_date"], errors="coerce")
            .dt.year.eq(TRAIN_YEAR)
        ].copy()
        target_frames[weather_arm] = event_targets.reset_index(drop=True)
        mar_frames[weather_arm] = mar_rows.reset_index(drop=True)
        audits[weather_arm] = {
            "event_target": target_audit,
            "mar_rows": mar_audit,
        }
    _, weather_audit = neutralize_training_rounds(rounds, weather_coefficients)
    audits["weather_neutralization"] = weather_audit
    return target_frames, mar_frames, audits


def _training_event_metadata(
    features: pd.DataFrame,
    target_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    event_meta = (
        features.loc[features["event_start_date"].dt.year.eq(TRAIN_YEAR)]
        .groupby(["event_key", "event_start_date"], as_index=False, observed=True)
        .agg(event_end_date=("event_end_date", "max"), feature_rows=("dg_id_key", "nunique"))
    )
    event_meta["event_end_date"] = pd.to_datetime(
        event_meta["event_end_date"], errors="coerce"
    ).dt.normalize()
    if event_meta["event_end_date"].isna().any():
        # A missing cached end date is resolved only from the already-loaded
        # 2023 target construction, never from evaluation outcomes.
        fallback = (
            target_frames["W0"]
            .groupby("event_key", observed=True)["event_start_date"]
            .first()
            + pd.Timedelta(days=3)
        )
        event_meta["event_end_date"] = event_meta["event_end_date"].fillna(
            event_meta["event_key"].map(fallback)
        )
    return event_meta.sort_values(
        ["event_start_date", "event_key"], kind="mergesort"
    ).reset_index(drop=True)


def chronological_batches(event_meta: pd.DataFrame) -> list[tuple[pd.Timestamp, tuple[str, ...]]]:
    """Return April--December event-start batches in deterministic order."""

    eligible = event_meta.loc[
        event_meta["event_start_date"].between(
            CHRONOLOGICAL_PREDICTION_START, SELECTION_END
        )
    ]
    batches: list[tuple[pd.Timestamp, tuple[str, ...]]] = []
    for start_date, group in eligible.groupby("event_start_date", sort=True, observed=True):
        batches.append((pd.Timestamp(start_date), tuple(sorted(group["event_key"].astype(str)))))
    return batches


class HuberRidgeIRLS:
    """Small deterministic Huber IRLS solver with an L2 coefficient penalty."""

    def __init__(
        self,
        *,
        epsilon: float = 1.35,
        alpha: float = 1000.0,
        max_iter: int = 30,
        tolerance: float = 1e-7,
        fit_intercept: bool = False,
    ) -> None:
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.fit_intercept = bool(fit_intercept)
        self.coef_: np.ndarray | None = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HuberRidgeIRLS":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        base_weight = (
            np.ones(len(y), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if self.fit_intercept:
            design = np.column_stack([np.ones(len(x)), x])
            penalty = np.eye(design.shape[1]) * self.alpha
            penalty[0, 0] = 0.0
        else:
            design = x
            penalty = np.eye(design.shape[1]) * self.alpha
        robust = np.ones(len(y), dtype=float)
        coefficients = np.zeros(design.shape[1], dtype=float)
        for iteration in range(self.max_iter):
            weights = np.clip(base_weight * robust, 0.0, np.inf)
            lhs = design.T @ (weights[:, None] * design) + penalty
            rhs = design.T @ (weights * y)
            try:
                updated = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            residual = y - design @ updated
            center = float(np.median(residual))
            scale = float(1.4826 * np.median(np.abs(residual - center)))
            scale = max(scale, 1e-8)
            absolute = np.abs(residual - center)
            cutoff = self.epsilon * scale
            new_robust = np.minimum(1.0, cutoff / np.maximum(absolute, 1e-12))
            delta = float(np.max(np.abs(updated - coefficients)))
            coefficients = updated
            robust = new_robust
            self.n_iter_ = iteration + 1
            if delta <= self.tolerance:
                break
        if self.fit_intercept:
            self.intercept_ = float(coefficients[0])
            self.coef_ = coefficients[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = coefficients.copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("HuberRidgeIRLS is not fitted")
        return np.asarray(x, dtype=float) @ self.coef_ + self.intercept_


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    numeric = frame.loc[:, list(columns)].copy()
    for column in numeric.columns:
        if pd.api.types.is_bool_dtype(numeric[column]):
            numeric[column] = numeric[column].astype(float)
        else:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric.to_numpy(float)


def _fit_preprocessor(
    frame: pd.DataFrame,
    columns: Sequence[str],
    sample_weight: np.ndarray,
) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    raw = _numeric_matrix(frame, columns)
    all_missing = [columns[index] for index in np.flatnonzero(np.isnan(raw).all(axis=0))]
    if all_missing:
        raise ValueError(f"training batch has all-missing declared columns: {all_missing[:20]}")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed = imputer.fit_transform(raw)
    scaler = StandardScaler()
    try:
        transformed = scaler.fit_transform(imputed, sample_weight=sample_weight)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        transformed = scaler.fit_transform(imputed)
    if not np.isfinite(transformed).all():
        raise AssertionError("nonfinite processed feature matrix")
    return imputer, scaler, transformed


def _transform_features(model: FittedRegressor, frame: pd.DataFrame) -> np.ndarray:
    raw = _numeric_matrix(frame, model.feature_columns)
    transformed = model.scaler.transform(model.imputer.transform(raw))
    if not np.isfinite(transformed).all():
        raise AssertionError("nonfinite prediction feature matrix")
    return transformed


def _event_round_center(
    matrix: np.ndarray,
    groups: pd.Series,
) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    group_codes, _ = pd.factorize(groups.astype(str), sort=False)
    for code in np.unique(group_codes):
        positions = np.flatnonzero(group_codes == code)
        result[positions] -= result[positions].mean(axis=0)
    return result


def _make_estimator(spec: CandidateSpec, *, fit_intercept: bool) -> Any:
    if spec.estimator == "ridge":
        return Ridge(alpha=spec.alpha, fit_intercept=fit_intercept, solver="lsqr")
    if spec.estimator == "huber_irls":
        return HuberRidgeIRLS(
            epsilon=float(spec.epsilon or 1.35),
            alpha=spec.alpha,
            fit_intercept=fit_intercept,
        )
    if spec.estimator == "elastic_net":
        return ElasticNet(
            alpha=spec.alpha,
            l1_ratio=float(spec.l1_ratio or 0.1),
            fit_intercept=fit_intercept,
            max_iter=50_000,
            tol=1e-4,
            random_state=20260823,
        )
    if spec.estimator == "hist_gb":
        return HistGradientBoostingRegressor(
            max_leaf_nodes=int(spec.max_leaf_nodes or 7),
            learning_rate=float(spec.learning_rate or 0.03),
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260823,
        )
    raise ValueError(f"unknown estimator {spec.estimator!r}")


def _fit_candidate(
    spec: CandidateSpec,
    fit_rows: pd.DataFrame,
    feature_columns: Sequence[str],
) -> FittedRegressor:
    weights = pd.to_numeric(fit_rows["_sample_weight"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(fit_rows["_target"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        raise ValueError(f"only {int(valid.sum())} valid training rows")
    train = fit_rows.loc[valid].reset_index(drop=True)
    y = y[valid]
    weights = weights[valid]
    imputer, scaler, processed = _fit_preprocessor(train, feature_columns, weights)
    t2 = spec.target_arm == "T2"
    if t2:
        if "event_round_key" not in train:
            raise ValueError("T2 fit rows are missing event_round_key")
        processed_fit = _event_round_center(processed, train["event_round_key"])
        # Recompute the nuisance removal instead of trusting a cached center.
        y_frame = pd.DataFrame({"y": y, "group": train["event_round_key"].astype(str)})
        y_fit = y - y_frame.groupby("group", observed=True)["y"].transform("mean").to_numpy()
    else:
        processed_fit = processed
        y_fit = y
    estimator = _make_estimator(spec, fit_intercept=not t2)
    estimator.fit(processed_fit, y_fit, sample_weight=weights)
    raw_train_prediction = np.asarray(estimator.predict(processed), dtype=float)
    prediction_offset = 0.0
    if t2:
        # Within-event-round fitting identifies slopes.  Recover the absolute
        # SG level under the same event-round-equal measure from uncentered X.
        prediction_offset = float(
            np.average(y - raw_train_prediction, weights=weights)
        )
    return FittedRegressor(
        imputer=imputer,
        scaler=scaler,
        estimator=estimator,
        target_arm=spec.target_arm,
        prediction_offset=prediction_offset,
        feature_columns=tuple(feature_columns),
    )


def predict_candidate(model: FittedRegressor, frame: pd.DataFrame) -> np.ndarray:
    transformed = _transform_features(model, frame)
    prediction = np.asarray(model.estimator.predict(transformed), dtype=float)
    prediction += float(model.prediction_offset)
    return prediction


def _candidate_fit_rows(
    features: pd.DataFrame,
    spec: CandidateSpec,
    target_frames: Mapping[str, pd.DataFrame],
    mar_frames: Mapping[str, pd.DataFrame],
    train_event_keys: set[str],
) -> pd.DataFrame:
    keys = ["event_key", "event_start_date", "dg_id_key"]
    source_features = features.loc[features["event_key"].astype(str).isin(train_event_keys)].copy()
    if spec.target_arm in TARGET_LABELS:
        targets = target_frames[spec.weather_arm]
        target_column = TARGET_LABELS[spec.target_arm]
        count_column = TARGET_COUNTS[spec.target_arm]
        weight_column = TARGET_WEIGHTS[spec.target_arm]
        use = targets.loc[pd.to_numeric(targets[count_column], errors="coerce").ge(2)].copy()
        fit = source_features.merge(
            use[keys + [target_column, weight_column]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        fit["_target"] = pd.to_numeric(fit[target_column], errors="coerce")
        raw_weight = pd.to_numeric(fit[weight_column], errors="coerce")
        event_total = raw_weight.groupby(fit["event_key"], observed=True).transform("sum")
        event_equal = raw_weight / event_total.replace(0.0, np.nan)
        fit["_sample_weight"] = event_equal / event_equal.mean()
        return fit
    if spec.target_arm != "T2":
        raise ValueError(f"unsupported target arm {spec.target_arm}")
    rounds = mar_frames[spec.weather_arm].copy()
    rounds = rounds.loc[
        rounds["event_key"].astype(str).isin(train_event_keys)
    ]
    fit = rounds.merge(source_features, on=keys, how="inner", validate="many_to_one")
    fit["_target"] = pd.to_numeric(fit["raw_y"], errors="coerce")
    group_size = fit.groupby("event_round_key", observed=True)["dg_id_key"].transform("count")
    event_round_equal = 1.0 / group_size.clip(lower=1).astype(float)
    # Preserve equal total mass per event-round while fixing the arbitrary
    # global scale at mean one, so the frozen ridge-alpha grid has its intended
    # meaning independent of row count.
    fit["_sample_weight"] = event_round_equal / event_round_equal.mean()
    return fit


def _period_for_date(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates, errors="coerce")
    return pd.Series(
        np.select(
            [
                dates.between(CHRONOLOGICAL_PREDICTION_START, CALIBRATION_END),
                dates.between(SELECTION_START, SELECTION_END),
            ],
            ["calibration_apr_jun", "selection_jul_dec"],
            default="outside_scored_window",
        ),
        index=dates.index,
    )


def train_chronological_candidates(
    features: pd.DataFrame,
    target_frames: Mapping[str, pd.DataFrame],
    mar_frames: Mapping[str, pd.DataFrame],
    specs: Sequence[CandidateSpec],
    *,
    smoke: bool = False,
    smoke_max_candidates: int = 3,
    smoke_max_batches: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate strictly earlier-event predictions from April onward."""

    features_2023 = features.loc[
        features["event_start_date"].dt.year.eq(TRAIN_YEAR)
    ].copy()
    event_meta = _training_event_metadata(features_2023, target_frames)
    batches = chronological_batches(event_meta)
    run_specs = list(specs)
    if smoke:
        # A smoke artifact is explicitly nonselectable in the manifest; the
        # subset exists only to exercise every data boundary quickly.
        run_specs = run_specs[: max(1, int(smoke_max_candidates))]
        batches = batches[: max(1, int(smoke_max_batches))]
    predictions: list[pd.DataFrame] = []
    catalog_rows: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(run_specs, start=1):
        status = "ok"
        error = ""
        prediction_rows = 0
        batch_count = 0
        feature_columns: list[str] = []
        print(f"[{spec_index}/{len(run_specs)}] chronological {spec.candidate}")
        try:
            feature_columns = feature_columns_for_spec(features_2023, spec)
            for start_date, event_keys_tuple in batches:
                train_events = event_meta.loc[event_meta["event_end_date"].lt(start_date)]
                train_event_keys = set(train_events["event_key"].astype(str))
                if not train_event_keys:
                    raise ValueError(f"no completed 2023 warm-up events before {start_date.date()}")
                fit_rows = _candidate_fit_rows(
                    features_2023,
                    spec,
                    target_frames,
                    mar_frames,
                    train_event_keys,
                )
                model = _fit_candidate(spec, fit_rows, feature_columns)
                validation = features_2023.loc[
                    features_2023["event_key"].astype(str).isin(event_keys_tuple)
                ].copy()
                if validation.empty:
                    raise ValueError(f"prediction batch {start_date.date()} has no feature rows")
                validation["prediction"] = predict_candidate(model, validation)
                validation["candidate"] = spec.candidate
                validation["player_id"] = validation["dg_id_key"].map(_stable_id)
                validation["prediction_batch_start"] = start_date
                validation["training_max_event_end"] = train_events["event_end_date"].max()
                validation["training_events"] = int(len(train_event_keys))
                validation["training_rows"] = int(len(fit_rows))
                validation["period"] = _period_for_date(validation["event_start_date"])
                validation["target_arm"] = spec.target_arm
                validation["weather_arm"] = spec.weather_arm
                validation["feature_family"] = spec.feature_family
                validation["estimator"] = spec.estimator
                validation["alpha"] = spec.alpha
                keep = [
                    "candidate",
                    "event_key",
                    "event_start_date",
                    "dg_id_key",
                    "player_id",
                    "prediction",
                    "prediction_batch_start",
                    "training_max_event_end",
                    "training_events",
                    "training_rows",
                    "period",
                    "target_arm",
                    "weather_arm",
                    "feature_family",
                    "estimator",
                    "alpha",
                ]
                predictions.append(validation[keep])
                prediction_rows += int(len(validation))
                batch_count += 1
        except Exception as exc:
            status = "rejected"
            error = f"{type(exc).__name__}: {exc}"
            # A candidate is atomic.  Discard partial OOF predictions if any
            # declared batch/feature cannot be satisfied.
            predictions = [
                frame for frame in predictions if not frame["candidate"].eq(spec.candidate).any()
            ]
            prediction_rows = 0
            batch_count = 0
            print(f"  rejected: {error}")
        catalog_rows.append(
            {
                **asdict(spec),
                "feature_count": int(len(feature_columns)),
                "chronological_batches": batch_count,
                "oof_prediction_rows": prediction_rows,
                "run_status": status,
                "rejection_reason": error,
                "smoke_run": bool(smoke),
            }
        )

    if smoke:
        run_names = {spec.candidate for spec in run_specs}
        for spec in specs:
            if spec.candidate in run_names:
                continue
            try:
                feature_count = len(feature_columns_for_spec(features_2023, spec))
                reason = "not run by explicit smoke candidate limit"
            except Exception as exc:
                feature_count = 0
                reason = f"feature validation failed during smoke catalog: {type(exc).__name__}: {exc}"
            catalog_rows.append(
                {
                    **asdict(spec),
                    "feature_count": int(feature_count),
                    "chronological_batches": 0,
                    "oof_prediction_rows": 0,
                    "run_status": "not_run_smoke",
                    "rejection_reason": reason,
                    "smoke_run": True,
                }
            )

    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not oof.empty:
        key = ["candidate", "event_key", "player_id"]
        if oof.duplicated(key).any():
            raise AssertionError(f"chronological predictions duplicate {key}")
        chronology = pd.to_datetime(oof["training_max_event_end"], errors="coerce").lt(
            pd.to_datetime(oof["event_start_date"], errors="coerce")
        )
        if not chronology.all():
            raise AssertionError("chronological prediction used an uncompleted target/later event")
        if pd.to_datetime(oof["event_start_date"]).dt.year.ne(TRAIN_YEAR).any():
            raise AssertionError("OOF predictions escaped calendar 2023")
    catalog = pd.DataFrame(catalog_rows).sort_values("candidate", kind="mergesort").reset_index(drop=True)
    return oof, catalog


def _weighted_metrics(
    frame: pd.DataFrame,
    *,
    actual: str,
    predicted: str,
    weight: str,
) -> dict[str, float | int]:
    y = pd.to_numeric(frame[actual], errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame[predicted], errors="coerce").to_numpy(float)
    w = pd.to_numeric(frame[weight], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return {"rows": 0, "weighted_rmse": np.nan, "weighted_mae": np.nan, "correlation": np.nan}
    y, p, w = y[valid], p[valid], w[valid]
    error = p - y
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else np.nan
    return {
        "rows": int(len(y)),
        "weighted_rmse": float(np.sqrt(np.average(error * error, weights=w))),
        "weighted_mae": float(np.average(np.abs(error), weights=w)),
        "correlation": correlation,
    }


def score_oof_targets(
    oof: pd.DataFrame,
    target_frames: Mapping[str, pd.DataFrame],
    mar_frames: Mapping[str, pd.DataFrame],
    specs: Sequence[CandidateSpec],
) -> pd.DataFrame:
    """Score target fit without using this diagnostic for market selection."""

    if oof.empty:
        return pd.DataFrame()
    spec_map = {spec.candidate: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    periods = ("calibration_apr_jun", "selection_jul_dec", "chronological_apr_dec")
    keys = ["event_key", "event_start_date", "dg_id_key"]
    for candidate, predictions in oof.groupby("candidate", sort=True, observed=True):
        spec = spec_map[candidate]
        if spec.target_arm in TARGET_LABELS:
            target_column = TARGET_LABELS[spec.target_arm]
            count_column = TARGET_COUNTS[spec.target_arm]
            weight_column = TARGET_WEIGHTS[spec.target_arm]
            targets = target_frames[spec.weather_arm]
            targets = targets.loc[pd.to_numeric(targets[count_column], errors="coerce").ge(2)]
            scored = predictions.merge(
                targets[keys + [target_column, weight_column]],
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            scored["_event_equal_weight"] = pd.to_numeric(
                scored[weight_column], errors="coerce"
            )
            scored["_event_equal_weight"] /= scored.groupby(
                "event_key", observed=True
            )["_event_equal_weight"].transform("sum")
            actual, weight = target_column, "_event_equal_weight"
        else:
            target_rounds = mar_frames[spec.weather_arm].copy()
            scored = target_rounds.merge(
                predictions[keys + ["candidate", "prediction", "period"]],
                on=keys,
                how="inner",
                validate="many_to_one",
            )
            scored["_prediction_centered"] = scored["prediction"] - scored.groupby(
                "event_round_key", observed=True
            )["prediction"].transform("mean")
            group_size = scored.groupby("event_round_key", observed=True)["dg_id_key"].transform("count")
            scored["_event_round_weight"] = 1.0 / group_size.clip(lower=1).astype(float)
            actual, weight = "raw_y_event_round_centered", "_event_round_weight"
            scored["prediction"] = scored["_prediction_centered"]
        # The target-less Q-school event is excluded explicitly even if a
        # future cache happens to populate a partial row.
        scored = scored.loc[
            ~pd.to_datetime(scored["event_start_date"]).eq(EXCLUDED_TARGET_EVENT_DATE)
        ]
        for period in periods:
            group = scored if period == "chronological_apr_dec" else scored.loc[scored["period"].eq(period)]
            metrics = _weighted_metrics(group, actual=actual, predicted="prediction", weight=weight)
            rows.append(
                {
                    "candidate": candidate,
                    "target_arm": spec.target_arm,
                    "weather_arm": spec.weather_arm,
                    "feature_family": spec.feature_family,
                    "estimator": spec.estimator,
                    "alpha": spec.alpha,
                    "period": period,
                    "events": int(group["event_key"].nunique()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def split_market_selection_oof(
    oof: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep nonselectable sanity checks out of every market-selection input."""

    if oof.empty:
        return oof.copy(), oof.copy()
    required = {"candidate", "selectable", "run_status"}
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise ValueError(f"candidate catalog missing OOF eligibility fields: {missing}")
    eligibility = catalog[["candidate", "selectable", "run_status"]].copy()
    if not pd.api.types.is_bool_dtype(eligibility["selectable"]):
        normalized = eligibility["selectable"].astype(str).str.strip().str.casefold()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("candidate catalog selectable field is not boolean")
        eligibility["selectable"] = normalized.eq("true")
    eligibility["market_selection_eligible"] = (
        eligibility["selectable"] & eligibility["run_status"].eq("ok")
    )
    tagged = oof.merge(eligibility, on="candidate", how="left", validate="many_to_one")
    if tagged["market_selection_eligible"].isna().any():
        raise AssertionError("an OOF candidate is absent from the frozen catalog")
    market = tagged.loc[tagged["market_selection_eligible"]].drop(
        columns=["selectable", "run_status", "market_selection_eligible"]
    )
    diagnostics = tagged.loc[~tagged["market_selection_eligible"]].drop(
        columns=["selectable", "run_status", "market_selection_eligible"]
    )
    if not market.empty:
        admitted = set(market["candidate"])
        declared = set(
            eligibility.loc[eligibility["market_selection_eligible"], "candidate"]
        )
        if admitted != declared:
            raise AssertionError("market OOF does not close to successful selectable catalog rows")
    return market.reset_index(drop=True), diagnostics.reset_index(drop=True)


def build_market_context(
    features: pd.DataFrame,
    *,
    years: Iterable[int],
    bridge_path: str | Path = DEFAULT_BRIDGE,
    cascade_path: str | Path = DEFAULT_CASCADE,
) -> pd.DataFrame:
    """Load only prediction/downstream context; never round outcomes."""

    years_set = {int(year) for year in years}
    if not years_set:
        raise ValueError("market context requires at least one year")
    io_start = pd.Timestamp(year=min(years_set), month=1, day=1)
    io_end = pd.Timestamp(year=max(years_set) + 1, month=1, day=1)
    event_lookup_columns = [
        "event_key",
        "event_start_date",
        "year",
        "event_id",
    ]
    event_lookup = features[event_lookup_columns].drop_duplicates().copy()
    event_lookup["year_key"] = pd.to_numeric(event_lookup["year"], errors="coerce").astype("Int64")
    event_lookup["event_id_key"] = event_lookup["event_id"].map(_stable_id)
    event_lookup["event_start_date"] = pd.to_datetime(
        event_lookup["event_start_date"], errors="coerce"
    ).dt.normalize()
    event_lookup = event_lookup[
        ["event_key", "event_start_date", "year_key", "event_id_key"]
    ].drop_duplicates()
    if event_lookup.duplicated(["year_key", "event_id_key", "event_start_date"]).any():
        raise ValueError("feature event lookup is ambiguous")

    bridge_columns = [
        "year",
        "event_id",
        "event_start",
        "dg_id",
        "downstream_preserved_total",
        "legacy_core_pred_step2",
        "baseline_r1_prediction",
    ]
    bridge = pd.read_parquet(
        bridge_path,
        columns=bridge_columns,
        filters=[("year", "in", sorted(years_set))],
    )
    bridge["event_start_date"] = pd.to_datetime(bridge.pop("event_start"), errors="coerce").dt.normalize()
    bridge["year_key"] = pd.to_numeric(bridge["year"], errors="coerce").astype("Int64")
    bridge["event_id_key"] = bridge["event_id"].map(_stable_id)
    bridge["dg_id_key"] = bridge["dg_id"].map(_stable_id)
    bridge = bridge.loc[bridge["year_key"].isin(years_set)].merge(
        event_lookup,
        on=["year_key", "event_id_key", "event_start_date"],
        how="inner",
        validate="many_to_one",
    )
    bridge = bridge[
        [
            "event_key",
            "dg_id_key",
            "downstream_preserved_total",
            "legacy_core_pred_step2",
            "baseline_r1_prediction",
        ]
    ]
    if bridge.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("bridge context is ambiguous by player-event")

    control_columns = [
        "event_key",
        "event_start_date",
        "dg_id_key",
        "round_num",
        "prediction",
    ]
    control = pd.read_parquet(
        cascade_path,
        columns=control_columns + ["model"],
        filters=[
            ("model", "==", "C0_current_cascade_retrospective"),
            ("event_start_date", ">=", io_start),
            ("event_start_date", "<", io_end),
        ],
    ).drop(columns="model")
    control["event_start_date"] = pd.to_datetime(
        control["event_start_date"], errors="coerce"
    ).dt.normalize()
    control = control.loc[control["event_start_date"].dt.year.isin(years_set)].copy()
    control["dg_id_key"] = control["dg_id_key"].map(_stable_id)
    control = control.rename(columns={"prediction": "current_control_prediction"})

    stage_columns = [
        "event_key",
        "event_start_date",
        "dg_id_key",
        "round_num",
        "legacy_stage_adjustment",
    ]
    stage = pd.read_parquet(
        cascade_path,
        columns=stage_columns + ["variant", "candidate"],
        filters=[
            ("variant", "==", "C1"),
            ("candidate", "==", "current_architecture_pt_v2_approx_control"),
            ("event_start_date", ">=", io_start),
            ("event_start_date", "<", io_end),
        ],
    ).drop(columns=["variant", "candidate"])
    stage["event_start_date"] = pd.to_datetime(
        stage["event_start_date"], errors="coerce"
    ).dt.normalize()
    stage = stage.loc[stage["event_start_date"].dt.year.isin(years_set)].copy()
    stage["dg_id_key"] = stage["dg_id_key"].map(_stable_id)
    round_keys = ["event_key", "event_start_date", "dg_id_key", "round_num"]
    if control.duplicated(round_keys).any() or stage.duplicated(round_keys).any():
        raise ValueError("cascade context contains duplicate player-round keys")
    context = control.merge(stage, on=round_keys, how="left", validate="one_to_one")
    context = context.merge(
        bridge,
        on=["event_key", "dg_id_key"],
        how="left",
        validate="many_to_one",
    )
    context["player_id"] = context["dg_id_key"]
    context["downstream_context_missing"] = context["downstream_preserved_total"].isna()
    context["cascade_stage_context_missing"] = context["legacy_stage_adjustment"].isna()
    return context.sort_values(round_keys, kind="mergesort").reset_index(drop=True)


def _assert_contract(contract_path: str | Path = CONTRACT_PATH) -> str:
    observed = sha256_file(contract_path)
    if observed.casefold() != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            "all-round contract hash changed after freeze: "
            f"expected {EXPECTED_CONTRACT_SHA256}, observed {observed}"
        )
    return observed


def _catalog_hash(catalog: pd.DataFrame) -> str:
    columns = [
        "candidate",
        "target_arm",
        "weather_arm",
        "feature_family",
        "estimator",
        "alpha",
        "selectable",
    ]
    payload = catalog[columns].sort_values("candidate").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_training_phase(
    *,
    output_dir: str | Path,
    feature_path: str | Path = DEFAULT_FEATURES,
    augmented_path: str | Path = DEFAULT_AUGMENTED_FEATURES,
    round_path: str | Path = DEFAULT_ROUNDS,
    weather_path: str | Path = DEFAULT_WEATHER,
    bridge_path: str | Path = DEFAULT_BRIDGE,
    cascade_path: str | Path = DEFAULT_CASCADE,
    smoke: bool = False,
    smoke_max_candidates: int = 3,
    smoke_max_batches: int = 2,
) -> dict[str, Any]:
    """Execute the pre-evaluation 2023 phase and write research artifacts."""

    contract_hash = _assert_contract()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = build_candidate_catalog(include_diagnostics=True)
    features = load_feature_frame(
        feature_path,
        augmented_path,
        years=(TRAIN_YEAR,),
    )
    source_rounds = load_training_rounds(round_path, year=TRAIN_YEAR)
    weather = load_frozen_weather_coefficients(weather_path)
    target_frames, mar_frames, target_audit = build_training_targets(source_rounds, weather)
    oof, catalog = train_chronological_candidates(
        features,
        target_frames,
        mar_frames,
        specs,
        smoke=smoke,
        smoke_max_candidates=smoke_max_candidates,
        smoke_max_batches=smoke_max_batches,
    )
    metrics = score_oof_targets(oof, target_frames, mar_frames, specs)
    market_oof, diagnostic_oof = split_market_selection_oof(oof, catalog)
    context = build_market_context(
        features,
        years=(TRAIN_YEAR,),
        bridge_path=bridge_path,
        cascade_path=cascade_path,
    )

    catalog.to_csv(output / "candidate_catalog.csv", index=False)
    market_oof.to_parquet(output / "chronological_oof_predictions.parquet", index=False)
    diagnostic_oof.to_parquet(
        output / "nonselectable_diagnostic_oof_predictions.parquet", index=False
    )
    metrics.to_csv(output / "target_metrics.csv", index=False)
    context.to_parquet(output / "market_context_2023.parquet", index=False)
    combined_targets = pd.concat(
        [frame.assign(weather_arm=arm) for arm, frame in target_frames.items()],
        ignore_index=True,
    )
    combined_targets.to_parquet(output / "all_round_targets_2023.parquet", index=False)
    (output / "target_construction_audit.json").write_text(
        json.dumps(target_audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    catalog_sha = _catalog_hash(catalog)
    schema = {
        "candidate": "one successful selectable candidate from candidate_catalog.csv",
        "downstream_lambda": [0.0, 0.5, 1.0],
        "live_update_lambda": [0.0, 0.5, 1.0],
        "catalog_sha256": catalog_sha,
        "contract_sha256": contract_hash,
        "selection_source": "2023-07-01 through 2023-12-31 only",
    }
    (output / "selected_spec_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "research_only": True,
        "production_files_written": False,
        "phase": "train",
        "smoke": bool(smoke),
        "smoke_outputs_selection_eligible": False if smoke else True,
        "contract_path": str(CONTRACT_PATH),
        "contract_sha256": contract_hash,
        "candidate_catalog_sha256": catalog_sha,
        "calendar_2023_only_coefficients": True,
        "pre_2023_rounds_role": "strict_prior_history_and_T1_anchor_only",
        "chronological_prediction_start": str(CHRONOLOGICAL_PREDICTION_START.date()),
        "probability_calibration_period": "2023-04-01/2023-06-30",
        "architecture_selection_period": "2023-07-01/2023-12-31",
        "evaluation_outcomes_read": False,
        "rows": {
            "features_2023": int(len(features)),
            "target_source_rounds_through_2023": int(len(source_rounds)),
            "market_selection_oof_predictions": int(len(market_oof)),
            "nonselectable_diagnostic_oof_predictions": int(len(diagnostic_oof)),
            "all_oof_predictions_scored_for_target_diagnostics": int(len(oof)),
            "market_context": int(len(context)),
        },
        "inputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "augmented_features": {
                "path": str(augmented_path),
                "sha256": sha256_file(augmented_path),
            },
            "rounds": {"path": str(round_path), "sha256": sha256_file(round_path)},
            "weather": {"path": str(weather_path), "sha256": sha256_file(weather_path)},
            "bridge": {"path": str(bridge_path), "sha256": sha256_file(bridge_path)},
            "cascade": {"path": str(cascade_path), "sha256": sha256_file(cascade_path)},
        },
    }
    (output / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def load_selected_spec(
    path: str | Path,
    catalog: Sequence[CandidateSpec],
    *,
    expected_catalog_sha256: str | None = None,
) -> tuple[CandidateSpec, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "live_update_lambda" not in payload and "live_lambda" in payload:
        payload["live_update_lambda"] = payload["live_lambda"]
    required = {
        "candidate",
        "downstream_lambda",
        "live_update_lambda",
        "catalog_sha256",
        "contract_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"selected spec missing required fields: {missing}")
    if str(payload["contract_sha256"]).casefold() != EXPECTED_CONTRACT_SHA256:
        raise ValueError("selected spec was not frozen under the current contract")
    if expected_catalog_sha256 is not None and str(payload["catalog_sha256"]) != expected_catalog_sha256:
        raise ValueError("selected spec catalog hash does not match training catalog")
    by_name = {spec.candidate: spec for spec in catalog}
    candidate = str(payload["candidate"])
    if candidate not in by_name:
        raise ValueError(f"selected candidate is outside the frozen catalog: {candidate}")
    spec = by_name[candidate]
    if not spec.selectable:
        raise ValueError("a nonselectable diagnostic cannot enter evaluation")
    downstream = float(payload["downstream_lambda"])
    live = float(payload["live_update_lambda"])
    if downstream not in {0.0, 0.5, 1.0}:
        raise ValueError("downstream_lambda is outside the frozen menu")
    if live not in {0.0, 0.5, 1.0}:
        raise ValueError("live_update_lambda is outside the frozen menu")
    return spec, payload


def validate_selected_spec_sidecar(path: str | Path) -> str:
    """Require the immutable handoff sidecar before any evaluation data read."""

    source = Path(path).resolve()
    sidecar = source.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError("selected runner spec lacks its SHA-256 handoff sidecar")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError("selected runner spec sidecar is empty")
    observed = sha256_file(source)
    if tokens[0].casefold() != observed.casefold():
        raise ValueError("selected runner spec hash differs from its handoff sidecar")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("research_only") is not True:
        raise ValueError("selected runner spec is not marked research-only")
    if payload.get("production_files_written") is not False:
        raise ValueError("selected runner spec reports a production write")
    if str(payload.get("selected_arm")) != str(payload.get("candidate")):
        raise ValueError("selected runner spec arm name differs from its candidate")
    if not isinstance(payload.get("promotion_eligible"), bool):
        raise ValueError("selected runner spec does not preserve promotion eligibility")
    return observed


def assert_prediction_artifact_roles(
    raw_core: pd.DataFrame,
    composed_pre_event: pd.DataFrame,
    composed_round: pd.DataFrame,
) -> None:
    """Prevent a downstream adapter from confusing raw and scalar-applied files."""

    checks = (
        (
            raw_core,
            "raw_base_core_before_downstream_and_live",
            False,
            {False},
            "raw core",
        ),
        (
            composed_pre_event,
            "composed_pre_event_after_downstream",
            True,
            {False},
            "composed pre-event",
        ),
        (
            composed_round,
            "composed_round_after_downstream_and_live",
            True,
            {False, True},
            "composed round",
        ),
    )
    for frame, stage, downstream_applied, allowed_live, label in checks:
        required = {
            "prediction_stage",
            "downstream_lambda_applied",
            "live_update_lambda_applied",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise AssertionError(f"{label} artifact lacks stage fields: {missing}")
        if set(frame["prediction_stage"].dropna().astype(str)) != {stage}:
            raise AssertionError(f"{label} artifact has the wrong prediction stage")
        if set(frame["downstream_lambda_applied"].dropna().astype(bool)) != {
            downstream_applied
        }:
            raise AssertionError(f"{label} downstream-applied flag is inconsistent")
        live_values = set(frame["live_update_lambda_applied"].dropna().astype(bool))
        if not live_values.issubset(allowed_live) or not live_values:
            raise AssertionError(f"{label} live-applied flag is inconsistent")


def refit_selected_2023(
    features: pd.DataFrame,
    spec: CandidateSpec,
    target_frames: Mapping[str, pd.DataFrame],
    mar_frames: Mapping[str, pd.DataFrame],
) -> tuple[FittedRegressor, dict[str, Any]]:
    train_features = features.loc[
        features["event_start_date"].dt.year.eq(TRAIN_YEAR)
    ].copy()
    train_event_keys = set(train_features["event_key"].astype(str))
    feature_columns = feature_columns_for_spec(train_features, spec)
    fit_rows = _candidate_fit_rows(
        train_features,
        spec,
        target_frames,
        mar_frames,
        train_event_keys,
    )
    model = _fit_candidate(spec, fit_rows, feature_columns)
    training_max_event = pd.to_datetime(fit_rows["event_start_date"], errors="coerce").max()
    training_max_event_end = pd.to_datetime(
        fit_rows.get("event_end_date"), errors="coerce"
    ).max()
    if pd.isna(training_max_event) or training_max_event.year != TRAIN_YEAR:
        raise AssertionError("final refit did not use calendar-2023 labels")
    if pd.isna(training_max_event_end) or training_max_event_end.year != TRAIN_YEAR:
        raise AssertionError("final refit has no valid calendar-2023 completion boundary")
    return model, {
        "training_rows": int(len(fit_rows)),
        "training_events": int(fit_rows["event_key"].nunique()),
        "training_max_event_start": str(training_max_event.date()),
        "training_max_event_end": str(training_max_event_end.date()),
        "feature_count": int(len(feature_columns)),
        "prediction_offset": float(model.prediction_offset),
    }


def run_evaluation_prediction_phase(
    *,
    selected_spec_path: str | Path,
    output_dir: str | Path,
    feature_path: str | Path = DEFAULT_FEATURES,
    augmented_path: str | Path = DEFAULT_AUGMENTED_FEATURES,
    round_path: str | Path = DEFAULT_ROUNDS,
    weather_path: str | Path = DEFAULT_WEATHER,
    bridge_path: str | Path = DEFAULT_BRIDGE,
    cascade_path: str | Path = DEFAULT_CASCADE,
) -> dict[str, Any]:
    """Refit once on 2023 and make label-blind 2024--2026 predictions."""

    contract_hash = _assert_contract()
    training_round_path = Path(round_path)
    # This validation occurs before feature, target, or 2024-26 context reads.
    validate_selected_spec_sidecar(selected_spec_path)
    specs = build_candidate_catalog(include_diagnostics=True)
    catalog_frame = pd.DataFrame(asdict(spec) for spec in specs)
    catalog_sha = _catalog_hash(catalog_frame)
    spec, selected = load_selected_spec(
        selected_spec_path,
        specs,
        expected_catalog_sha256=catalog_sha,
    )
    # Explicit column projection in load_feature_frame excludes every cached
    # label/control column for both training and evaluation rows.
    features = load_feature_frame(
        feature_path,
        augmented_path,
        years=(TRAIN_YEAR, *EVALUATION_YEARS),
    )
    source_rounds = load_training_rounds(training_round_path, year=TRAIN_YEAR)
    if pd.to_numeric(source_rounds["year"], errors="coerce").gt(TRAIN_YEAR).any():
        raise AssertionError("evaluation phase read a post-2023 target round")
    weather = load_frozen_weather_coefficients(weather_path)
    target_frames, mar_frames, target_audit = build_training_targets(source_rounds, weather)
    model, fit_audit = refit_selected_2023(
        features,
        spec,
        target_frames,
        mar_frames,
    )
    evaluation = features.loc[
        features["event_start_date"].dt.year.isin(EVALUATION_YEARS)
    ].copy()
    if evaluation.empty:
        raise ValueError("no 2024--2026 feature rows available")
    evaluation["base_core_prediction"] = predict_candidate(model, evaluation)
    context = build_market_context(
        features,
        years=EVALUATION_YEARS,
        bridge_path=bridge_path,
        cascade_path=cascade_path,
    )
    downstream_lookup = context[
        ["event_key", "dg_id_key", "downstream_preserved_total", "downstream_context_missing"]
    ].drop_duplicates(["event_key", "dg_id_key"])
    evaluation = evaluation.merge(
        downstream_lookup,
        on=["event_key", "dg_id_key"],
        how="left",
        validate="one_to_one",
    )
    evaluation["downstream_context_missing"] = evaluation[
        "downstream_context_missing"
    ].fillna(True)
    evaluation["downstream_preserved_total"] = pd.to_numeric(
        evaluation["downstream_preserved_total"], errors="coerce"
    ).fillna(0.0)
    downstream_lambda = float(selected["downstream_lambda"])
    live_lambda = float(selected["live_update_lambda"])
    evaluation["base_event_prediction"] = (
        evaluation["base_core_prediction"]
        + downstream_lambda * evaluation["downstream_preserved_total"]
    )
    winner_name = str(
        selected.get(
            "selected_arm",
            f"{spec.candidate}__downstream_{downstream_lambda:g}__live_{live_lambda:g}",
        )
    )
    evaluation["candidate"] = winner_name
    evaluation["player_id"] = evaluation["dg_id_key"].map(_stable_id)
    evaluation["prediction"] = evaluation["base_event_prediction"]
    evaluation["prediction_stage"] = "composed_pre_event_after_downstream"
    evaluation["downstream_lambda_applied"] = True
    evaluation["live_update_lambda_applied"] = False
    evaluation["target_arm"] = spec.target_arm
    evaluation["weather_arm"] = spec.weather_arm
    evaluation["feature_family"] = spec.feature_family
    evaluation["estimator"] = spec.estimator
    evaluation["alpha"] = spec.alpha
    evaluation["downstream_lambda"] = downstream_lambda
    evaluation["training_max_event_end"] = pd.Timestamp(
        fit_audit["training_max_event_end"]
    )
    base_columns = [
        "candidate",
        "event_key",
        "event_start_date",
        "dg_id_key",
        "player_id",
        "prediction",
        "base_core_prediction",
        "downstream_preserved_total",
        "downstream_context_missing",
        "target_arm",
        "weather_arm",
        "feature_family",
        "estimator",
        "alpha",
        "downstream_lambda",
        "training_max_event_end",
        "prediction_stage",
        "downstream_lambda_applied",
        "live_update_lambda_applied",
    ]
    base_predictions = evaluation[base_columns].copy()
    if base_predictions.duplicated(["candidate", "event_key", "player_id"]).any():
        raise AssertionError("winner base predictions are ambiguous")
    raw_core_predictions = evaluation[
        [
            "event_key",
            "event_start_date",
            "dg_id_key",
            "player_id",
            "base_core_prediction",
            "training_max_event_end",
        ]
    ].copy()
    raw_core_predictions.insert(0, "candidate", spec.candidate)
    raw_core_predictions = raw_core_predictions.rename(
        columns={"base_core_prediction": "prediction"}
    )
    raw_core_predictions["prediction_stage"] = "raw_base_core_before_downstream_and_live"
    raw_core_predictions["downstream_lambda_applied"] = False
    raw_core_predictions["live_update_lambda_applied"] = False

    round_predictions = context.merge(
        evaluation[
            [
                "event_key",
                "dg_id_key",
                "base_core_prediction",
                "base_event_prediction",
            ]
        ],
        on=["event_key", "dg_id_key"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_winner"),
    )
    stage = pd.to_numeric(round_predictions["legacy_stage_adjustment"], errors="coerce")
    is_r1 = pd.to_numeric(round_predictions["round_num"], errors="coerce").eq(1)
    # R1 legitimately has no completed-round update.  A missing R2--R4 stage
    # remains missing and is excluded downstream rather than silently zeroed.
    round_predictions["prediction"] = np.where(
        is_r1,
        round_predictions["base_event_prediction"],
        round_predictions["base_event_prediction"] + live_lambda * stage,
    )
    round_predictions["candidate"] = winner_name
    round_predictions["prediction_stage"] = "composed_round_after_downstream_and_live"
    round_predictions["downstream_lambda_applied"] = True
    round_predictions["live_update_lambda_applied"] = ~is_r1
    round_predictions["live_update_lambda"] = live_lambda
    round_predictions["downstream_lambda"] = downstream_lambda
    round_predictions["player_id"] = round_predictions["dg_id_key"].map(_stable_id)
    round_predictions = round_predictions.loc[
        np.isfinite(pd.to_numeric(round_predictions["prediction"], errors="coerce"))
    ].copy()
    round_key = ["candidate", "event_key", "round_num", "player_id"]
    if round_predictions.duplicated(round_key).any():
        raise AssertionError("winner round predictions are ambiguous")
    assert_prediction_artifact_roles(
        raw_core_predictions, base_predictions, round_predictions
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_core_path = output / "winner_raw_core_predictions_2024_2026.parquet"
    base_path = output / "winner_composed_pre_event_predictions_2024_2026.parquet"
    round_prediction_path = output / "winner_composed_round_predictions_2024_2026.parquet"
    context_path = output / "market_context_2024_2026.parquet"
    raw_core_predictions.to_parquet(raw_core_path, index=False)
    base_predictions.to_parquet(base_path, index=False)
    round_predictions.to_parquet(round_prediction_path, index=False)
    context.to_parquet(context_path, index=False)
    selected_hash = sha256_file(selected_spec_path)
    manifest = {
        "research_only": True,
        "production_files_written": False,
        "phase": "evaluate_prediction_only",
        "selected_candidate": spec.candidate,
        "selected_arm": winner_name,
        "promotion_eligible": bool(selected["promotion_eligible"]),
        "diagnostic_fallback": bool(selected.get("diagnostic_fallback", False)),
        "selection_mode": str(selected.get("selection_mode", "primary_contract_selection")),
        "downstream_lambda": downstream_lambda,
        "live_update_lambda": live_lambda,
        "contract_sha256": contract_hash,
        "candidate_catalog_sha256": catalog_sha,
        "selected_spec_path": str(selected_spec_path),
        "selected_spec_sha256": selected_hash,
        "coefficient_training_years": [TRAIN_YEAR],
        "prediction_years": list(EVALUATION_YEARS),
        "post_2023_target_columns_projected": [],
        "post_2023_outcomes_read": False,
        "prediction_artifact_roles": {
            "winner_raw_core_predictions_2024_2026.parquet": (
                "adapter input; downstream/live lambdas not applied"
            ),
            "winner_composed_pre_event_predictions_2024_2026.parquet": (
                "selected downstream lambda already applied"
            ),
            "winner_composed_round_predictions_2024_2026.parquet": (
                "selected downstream and common live lambdas already applied"
            ),
        },
        "final_refit": fit_audit,
        "target_construction_audit_2023_only": target_audit,
        "rows": {
            "raw_core_predictions": int(len(raw_core_predictions)),
            "base_predictions": int(len(base_predictions)),
            "round_predictions": int(len(round_predictions)),
            "round_context": int(len(context)),
            "missing_downstream_base_rows": int(
                base_predictions["downstream_context_missing"].sum()
            ),
            "missing_non_r1_stage_context_rows": int(
                (
                    pd.to_numeric(context["round_num"], errors="coerce").gt(1)
                    & context["legacy_stage_adjustment"].isna()
                ).sum()
            ),
        },
        "inputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "augmented_features": {
                "path": str(augmented_path),
                "sha256": sha256_file(augmented_path),
            },
            "training_rounds": {
                "path": str(training_round_path),
                "sha256": sha256_file(training_round_path),
            },
            "weather": {"path": str(weather_path), "sha256": sha256_file(weather_path)},
            "bridge": {"path": str(bridge_path), "sha256": sha256_file(bridge_path)},
            "cascade": {"path": str(cascade_path), "sha256": sha256_file(cascade_path)},
        },
        "outputs": {
            "raw_core_predictions": {
                "path": raw_core_path.name,
                "sha256": sha256_file(raw_core_path),
                "rows": int(len(raw_core_predictions)),
            },
            "composed_pre_event_predictions": {
                "path": base_path.name,
                "sha256": sha256_file(base_path),
                "rows": int(len(base_predictions)),
            },
            "composed_round_predictions": {
                "path": round_prediction_path.name,
                "sha256": sha256_file(round_prediction_path),
                "rows": int(len(round_predictions)),
            },
            "market_context": {
                "path": context_path.name,
                "sha256": sha256_file(context_path),
                "rows": int(len(context)),
            },
        },
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest_path = output / "evaluation_prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output / "evaluation_prediction_manifest.sha256").write_text(
        sha256_file(manifest_path) + f"  {manifest_path.name}\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    def common_paths(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
        subparser.add_argument(
            "--augmented-features", type=Path, default=DEFAULT_AUGMENTED_FEATURES
        )
        subparser.add_argument("--rounds", type=Path, default=DEFAULT_ROUNDS)
        subparser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
        subparser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
        subparser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE)
        subparser.add_argument("--output-dir", type=Path, required=True)

    train = subparsers.add_parser("train", help="build chronological 2023 predictions")
    common_paths(train)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--smoke-max-candidates", type=int, default=3)
    train.add_argument("--smoke-max-batches", type=int, default=2)

    evaluate = subparsers.add_parser(
        "evaluate", help="refit the selected 2023 spec and predict 2024-26"
    )
    common_paths(evaluate)
    evaluate.add_argument("--selected-spec", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    common = {
        "output_dir": args.output_dir,
        "feature_path": args.features,
        "augmented_path": args.augmented_features,
        "round_path": args.rounds,
        "weather_path": args.weather,
        "bridge_path": args.bridge,
        "cascade_path": args.cascade,
    }
    if args.phase == "train":
        manifest = run_training_phase(
            **common,
            smoke=args.smoke,
            smoke_max_candidates=args.smoke_max_candidates,
            smoke_max_batches=args.smoke_max_batches,
        )
    else:
        manifest = run_evaluation_prediction_phase(
            **common,
            selected_spec_path=args.selected_spec,
        )
    print(json.dumps({"phase": args.phase, "rows": manifest["rows"]}, indent=2))


if __name__ == "__main__":
    main()

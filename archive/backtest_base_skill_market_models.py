"""Research-only, market-first screen of point-in-time base-skill models.

This script consumes the immutable player-round panel produced by
``archive/build_base_skill_market_panel.py``.  It never writes to a production
input or imports a closing price as a feature.  Its jobs are deliberately
limited to the first rung of the frozen base-skill rebuild contract:

* build strict-prior player/event feature snapshots;
* select ridge strength on the 2021-22 discovery sample;
* make expanding, calendar-year walk-forward predictions for 2023-2026;
* calibrate an analytic head-to-head variance on discovery matchups; and
* score 72-hole and R1 closing matchups at BetOnline, Betcris, and Pinnacle.

Neither IMG ``validated_v2`` nor the v4 accounting archive is used in a clean
arm: both inherit transforms selected after the discovery cutoff.  Weather arm
W1 estimates nuisance coefficients from
earlier R1-R2 rounds, then rebuilds every historical aggregate using neutralized
observations.  Realized target-round weather is never an input.

The calendar-year refit schedule is conservative: every prediction in a year
uses coefficients fitted before January 1 of that year.  Feature snapshots and
weather nuisance estimates are still rebuilt at each event date with a strict
``source_round_date < event_start`` boundary.  Thus the run is point-in-time,
though less adaptive than a weekly refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent
for _path in (str(REPO), str(ARCHIVE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from historical_odds_cache import HistoricalOddsCacheReader  # noqa: E402
from archive.market_backtest_scoring import (  # noqa: E402
    american_decimal_odds,
    american_implied_probability,
    settle_bets,
    ties_lose,
    valid_american_odds_mask,
)


DEFAULT_PANEL_DIR = (
    REPO / "permanent_data" / "research" / "base_skill_rebuild_20200101_20260822"
)
DEFAULT_ODDS_CACHE = (
    REPO / "permanent_data" / "research" / "dg_matchup_cache_20200101_20260822"
)
DEFAULT_OUTPUT = DEFAULT_PANEL_DIR / "model_eval"
DEFAULT_CORE_BRIDGE = DEFAULT_PANEL_DIR / "core_bridge" / "legacy_core_bridge.parquet"

CATEGORY_ORDER = ("ott", "app", "arg", "putt")
CATEGORY_SHARES = {"ott": 0.20, "app": 0.33, "arg": 0.14, "putt": 0.33}
HALF_LIVES = (10, 20, 40, 80, 160)
DAY_WINDOWS = (60, 180, 365, 730)
BOOKS = ("betonline", "betcris", "pinnacle")
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
DISCOVERY_START = pd.Timestamp("2021-01-01")
DISCOVERY_END = pd.Timestamp("2022-12-31")
SEALED_START = pd.Timestamp("2023-01-01")
SEALED_END = pd.Timestamp("2026-08-22")
COMPARABLE_START = pd.Timestamp("2023-08-22")
PRIMARY_LABEL = "label_dg_sg_adj_r1_r2_mean"
EV_GATE = 0.03
EPS = 1e-8
BASELINE_CANDIDATE = "legacy_skill_refit_pt"
RAW_LEGACY_CANDIDATE = "legacy_skill_est_pt"
RETROSPECTIVE_CORE_CANDIDATE = "legacy_pred_step2_retrospective"


@dataclass(frozen=True)
class Candidate:
    name: str
    weather_arm: str
    groups: tuple[str, ...]
    estimator: str = "ridge"
    alphas: tuple[float, ...] = ALPHAS
    short_game_scale: float = 0.5


CANDIDATES = (
    Candidate("dg_total_w0", "w0", ("dg_total",)),
    Candidate("dg_total_w1", "w1", ("dg_total",)),
    Candidate("dg_total_w1_feature_only", "w1", ("dg_total",)),
    Candidate("dg_precut_total_w0", "w0", ("dg_precut_total",)),
    Candidate("dg_precut_total_w1", "w1", ("dg_precut_total",)),
    Candidate(
        "dg_precut_total_cat_w0", "w0", ("dg_precut_total", "dg_precut_cat")
    ),
    Candidate(
        "dg_precut_total_cat_w1", "w1", ("dg_precut_total", "dg_precut_cat")
    ),
    Candidate(
        "dg_precut_total_cat_w1_feature_only",
        "w1",
        ("dg_precut_total", "dg_precut_cat"),
    ),
    Candidate("dg_total_cat_w0", "w0", ("dg_total", "dg_cat")),
    Candidate("dg_total_cat_w1", "w1", ("dg_total", "dg_cat")),
    Candidate("dg_total_cat_w1_feature_only", "w1", ("dg_total", "dg_cat")),
    Candidate(
        "long_game_w0", "w0", ("dg_total", "dg_long")
    ),
    Candidate(
        "long_game_w1", "w1", ("dg_total", "dg_long")
    ),
)

PAIRWISE_CANDIDATES = (
    Candidate(
        "pairwise_dg_precut_total_cat_w0",
        "w0",
        ("dg_precut_total", "dg_precut_cat"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
    Candidate(
        "pairwise_dg_precut_total_cat_w1",
        "w1",
        ("dg_precut_total", "dg_precut_cat"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
    Candidate(
        "pairwise_dg_total_cat_w0",
        "w0",
        ("dg_total", "dg_cat"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
    Candidate(
        "pairwise_dg_total_cat_w1",
        "w1",
        ("dg_total", "dg_cat"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
    Candidate(
        "pairwise_long_game_w0",
        "w0",
        ("dg_total", "dg_long"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
    Candidate(
        "pairwise_long_game_w1",
        "w1",
        ("dg_total", "dg_long"),
        estimator="pairwise_logistic",
        alphas=(0.01, 0.1, 1.0, 10.0),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _normalize_text(value: object) -> str:
    text = str(value or "").casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def _market_name(value: object) -> str | None:
    text = _normalize_text(value)
    if text in {"72 hole match", "72 hole matchup", "tournament matchup"}:
        return "72h"
    if text in {"r1 match up", "r1 matchup", "round 1 matchup"}:
        return "r1"
    return None


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_panels(panel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_path = panel_dir / "player_round_panel.parquet"
    event_path = panel_dir / "player_event_outcomes.parquet"
    if not round_path.exists() or not event_path.exists():
        raise FileNotFoundError(
            "Base-skill panel is incomplete; expected player_round_panel.parquet "
            "and player_event_outcomes.parquet"
        )
    rounds = pd.read_parquet(round_path)
    events = pd.read_parquet(event_path)
    _require(
        rounds,
        ["event_key", "dg_id", "event_start_date", "round_date"],
        "player-round panel",
    )
    _require(
        events,
        ["event_key", "dg_id", "event_start_date", PRIMARY_LABEL],
        "player-event outcomes",
    )
    for frame in (rounds, events):
        frame["event_start_date"] = pd.to_datetime(
            frame["event_start_date"], errors="coerce"
        ).dt.normalize()
        frame["dg_id_key"] = frame["dg_id"].map(_stable_id)
    rounds["round_date"] = pd.to_datetime(rounds["round_date"], errors="coerce").dt.normalize()
    rounds = rounds.dropna(subset=["event_start_date", "round_date"])
    events = events.dropna(subset=["event_start_date"])
    if events.duplicated(["event_key", "dg_id_key"]).any():
        raise ValueError("player-event outcomes contain duplicate event/player keys")
    if rounds.duplicated("player_round_key").any() if "player_round_key" in rounds else False:
        raise ValueError("player-round panel contains duplicate player_round_key values")
    events[PRIMARY_LABEL] = pd.to_numeric(events[PRIMARY_LABEL], errors="coerce")
    event_end = rounds.groupby("event_key", observed=True)["round_date"].max()
    events["event_end_date"] = events["event_key"].map(event_end)
    return rounds, events


def _weather_design(rounds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wind = _safe_numeric(rounds, "weather_delta_wind").fillna(0.0).to_numpy(float)
    dew = _safe_numeric(rounds, "weather_delta_dew").fillna(0.0).to_numpy(float)
    observed = (
        _safe_numeric(rounds, "weather_delta_wind").notna()
        | _safe_numeric(rounds, "weather_delta_dew").notna()
    ).to_numpy()
    return wind, dew, observed


def _within_player_weather_beta(
    history: pd.DataFrame,
    outcome_column: str,
    prior: np.ndarray,
    prior_weight: float = 250.0,
) -> np.ndarray:
    """Estimate neutralization beta after player fixed-effect demeaning.

    ``observed SG = player level - beta @ weather_delta + noise``.  The ridge
    penalty is centered on the predeclared literature/reanalysis prior rather
    than on zero.  It stabilizes early folds without silently backfilling a
    full-sample coefficient.
    """

    if outcome_column not in history:
        return prior.copy()
    use = history.copy()
    if "round_num" in use:
        round_num = pd.to_numeric(use["round_num"], errors="coerce")
        use = use.loc[round_num.isin([1, 2])].copy()
    y = _safe_numeric(use, outcome_column)
    wind, dew, observed = _weather_design(use)
    valid = y.notna().to_numpy() & observed
    if valid.sum() < 500:
        return prior.copy()
    use = use.loc[valid].copy()
    y = y.loc[valid].to_numpy(float)
    x = np.column_stack([wind[valid], dew[valid]])
    player = use["dg_id_key"].to_numpy()
    temp = pd.DataFrame({"player": player, "y": y, "x0": x[:, 0], "x1": x[:, 1]})
    means = temp.groupby("player", observed=True)[["y", "x0", "x1"]].transform("mean")
    yc = temp["y"].to_numpy() - means["y"].to_numpy()
    xc = temp[["x0", "x1"]].to_numpy() - means[["x0", "x1"]].to_numpy()
    # Regress observed outcome on weather; neutralization coefficient has the
    # opposite sign.  Penalty target is therefore -prior in regression space.
    lhs = xc.T @ xc + prior_weight * np.eye(2)
    rhs = xc.T @ yc + prior_weight * (-prior)
    try:
        slope = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return prior.copy()
    beta = -slope
    if not np.isfinite(beta).all():
        return prior.copy()
    # Guard only against obvious unit/join failures, not normal sampling motion.
    return np.clip(beta, np.array([-0.25, -0.10]), np.array([0.50, 0.10]))


def estimate_weather_coefficients(rounds: pd.DataFrame, target_dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prior_total = np.array([0.155, -0.026], dtype=float)
    sorted_rounds = rounds.sort_values("round_date", kind="mergesort")
    round_dates = sorted_rounds["round_date"].to_numpy(dtype="datetime64[ns]")
    freeze_date = pd.Timestamp("2021-01-01")
    freeze_stop = int(np.searchsorted(round_dates, np.datetime64(freeze_date), side="left"))
    frozen_history = sorted_rounds.iloc[:freeze_stop]

    def fit_from(history: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        total_fit = _within_player_weather_beta(
            history, "dg_sg_total_field_adjusted", prior_total
        )
        category_fit: dict[str, np.ndarray] = {}
        for category_name in CATEGORY_ORDER:
            category_fit[category_name] = _within_player_weather_beta(
                history,
                f"dg_sg_{category_name}_field_adjusted",
                CATEGORY_SHARES[category_name] * total_fit,
                prior_weight=400.0,
            )
        for axis_index in range(2):
            raw = np.array(
                [category_fit[category_name][axis_index] for category_name in CATEGORY_ORDER]
            )
            shares = np.array([CATEGORY_SHARES[category_name] for category_name in CATEGORY_ORDER])
            pooled = 0.5 * raw + 0.5 * shares * total_fit[axis_index]
            pooled += shares * (total_fit[axis_index] - pooled.sum())
            for category_index, category_name in enumerate(CATEGORY_ORDER):
                category_fit[category_name][axis_index] = pooled[category_index]
        return total_fit, category_fit

    frozen_total, frozen_categories = fit_from(frozen_history)
    for target_date in sorted(pd.Timestamp(value).normalize() for value in target_dates):
        stop = int(np.searchsorted(round_dates, np.datetime64(target_date), side="left"))
        history = sorted_rounds.iloc[:stop]
        # Pre-2021 snapshots use coefficients available at their own date.  The
        # discovery and sealed eras share one nuisance fit frozen before 2021,
        # so no 2021-26 result can alter the weather transformation.
        if target_date < freeze_date:
            total, category_raw = fit_from(history)
            coefficient_source = "expanding_pre_2021"
            coefficient_fit_max = history["round_date"].max() if stop else pd.NaT
            coefficient_fit_rounds = int(stop)
        else:
            total = frozen_total.copy()
            category_raw = {
                category: values.copy() for category, values in frozen_categories.items()
            }
            coefficient_source = "frozen_pre_2021"
            coefficient_fit_max = frozen_history["round_date"].max() if freeze_stop else pd.NaT
            coefficient_fit_rounds = int(freeze_stop)
        row: dict[str, Any] = {
            "event_start_date": target_date,
            "fit_rounds": coefficient_fit_rounds,
            "fit_max_round_date": coefficient_fit_max,
            "coefficient_source": coefficient_source,
            "total_beta_wind": float(total[0]),
            "total_beta_dew": float(total[1]),
        }
        for category in CATEGORY_ORDER:
            row[f"{category}_beta_wind"] = float(category_raw[category][0])
            row[f"{category}_beta_dew"] = float(category_raw[category][1])
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, 0.0, 0
    weights = weights[valid]
    values = values[valid]
    total = weights.sum()
    mean = float(np.dot(values, weights) / total)
    ess = float(total * total / np.dot(weights, weights))
    return mean, ess, int(valid.sum())


def _aggregate_signal(
    values: np.ndarray,
    ages_days: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    n = len(values)
    recency = np.arange(n - 1, -1, -1, dtype=float)
    for half_life in HALF_LIVES:
        weights = np.power(0.5, recency / float(half_life))
        mean, ess, count = _weighted_mean(values, weights)
        result[f"{prefix}__ewm_hl{half_life}"] = mean
        result[f"{prefix}__ess_hl{half_life}"] = ess
        result[f"{prefix}__n_hl{half_life}"] = float(count)
    for days in DAY_WINDOWS:
        include = ages_days <= days
        selected = values[include]
        valid = np.isfinite(selected)
        result[f"{prefix}__mean_d{days}"] = (
            float(np.mean(selected[valid])) if valid.any() else np.nan
        )
        result[f"{prefix}__n_d{days}"] = float(valid.sum())
    valid_all = np.isfinite(values)
    result[f"{prefix}__mean_all"] = (
        float(np.mean(values[valid_all])) if valid_all.any() else np.nan
    )
    result[f"{prefix}__n_all"] = float(valid_all.sum())
    return result


def _neutralized_values(
    history: pd.DataFrame,
    source_prefix: str,
    category: str,
    weather_arm: str,
    weather_row: Mapping[str, Any],
) -> np.ndarray:
    column = f"{source_prefix}_sg_{category}_field_adjusted"
    values = _safe_numeric(history, column).to_numpy(float)
    if weather_arm == "w0":
        return values
    wind = _safe_numeric(history, "weather_delta_wind").fillna(0.0).to_numpy(float)
    dew = _safe_numeric(history, "weather_delta_dew").fillna(0.0).to_numpy(float)
    coefficient_prefix = "total" if category == "total" else category
    return (
        values
        + float(weather_row[f"{coefficient_prefix}_beta_wind"]) * wind
        + float(weather_row[f"{coefficient_prefix}_beta_dew"]) * dew
    )


def _prior_event_withdrawal(history: pd.DataFrame) -> bool:
    if history.empty:
        return False
    latest_event = history.iloc[-1]["event_key"]
    latest = history.loc[history["event_key"] == latest_event]
    status_columns = [
        column
        for column in ("fin_text", "finish_text", "status", "event_status", "result")
        if column in latest
    ]
    for column in status_columns:
        text = " ".join(latest[column].dropna().astype(str)).casefold()
        if "wd" in text or "withdraw" in text or "dq" in text:
            return True
    return False


def build_feature_snapshots(
    rounds: pd.DataFrame,
    events: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build W0/W1 features with a verifiable strict-prior boundary."""

    by_player: dict[str, pd.DataFrame] = {
        player: group.sort_values("round_date", kind="mergesort").reset_index(drop=True)
        for player, group in rounds.groupby("dg_id_key", sort=False, observed=True)
    }
    weather_map = {
        pd.Timestamp(row["event_start_date"]).normalize(): row
        for row in weather.to_dict(orient="records")
    }
    output: list[dict[str, Any]] = []
    ordered = events.sort_values(["event_start_date", "event_key", "dg_id_key"], kind="mergesort")
    for event in ordered.to_dict(orient="records"):
        target_date = pd.Timestamp(event["event_start_date"]).normalize()
        history_all = by_player.get(event["dg_id_key"])
        if history_all is None:
            history = pd.DataFrame(columns=rounds.columns)
        else:
            stop = int(
                np.searchsorted(
                    history_all["round_date"].to_numpy(dtype="datetime64[ns]"),
                    np.datetime64(target_date),
                    side="left",
                )
            )
            history = history_all.iloc[:stop]
        row: dict[str, Any] = {
            "event_key": event["event_key"],
            "dg_id_key": event["dg_id_key"],
            "dg_id": event["dg_id"],
            "event_start_date": target_date,
            PRIMARY_LABEL: event.get(PRIMARY_LABEL),
            "label_dg_sg_adj_r1_r2_count": event.get(
                "label_dg_sg_adj_r1_r2_count"
            ),
            "event_end_date": event.get("event_end_date"),
            "history_rounds": int(len(history)),
            "max_source_round_date": history["round_date"].max() if len(history) else pd.NaT,
            "days_since_last_round": (
                float((target_date - history["round_date"].max()).days)
                if len(history)
                else np.nan
            ),
            "prior_event_withdrawal": _prior_event_withdrawal(history),
        }
        for optional in (
            "player_name",
            "event_name",
            "year",
            "event_id",
            "legacy_skill_est_asof",
            "r1_cascade_benchmark_prediction",
            "cascade_prediction_r1",
        ):
            if optional in event:
                row[optional] = event.get(optional)
        # Prefer the exact pre-event/R1 legacy snapshot from the event table;
        # otherwise use the target event's R1 round row, never a future event.
        if pd.isna(row.get("legacy_skill_est_asof", np.nan)):
            current = rounds.loc[
                (rounds["event_key"] == event["event_key"])
                & (rounds["dg_id_key"] == event["dg_id_key"])
            ]
            if "round_num" in current:
                current = current.loc[pd.to_numeric(current["round_num"], errors="coerce") == 1]
            if not current.empty and "legacy_skill_est_asof" in current:
                row["legacy_skill_est_asof"] = pd.to_numeric(
                    current["legacy_skill_est_asof"], errors="coerce"
                ).iloc[0]
        if history.empty:
            output.append(row)
            continue
        ages = (target_date - history["round_date"]).dt.days.to_numpy(float)
        for weather_arm in ("w0", "w1"):
            weather_row = weather_map[target_date]
            dg_total = _neutralized_values(
                history, "dg", "total", weather_arm, weather_row
            )
            row.update(_aggregate_signal(dg_total, ages, f"dg_total_{weather_arm}"))
            dg_categories: dict[str, np.ndarray] = {}
            for category in CATEGORY_ORDER:
                category_values = _neutralized_values(
                    history, "dg", category, weather_arm, weather_row
                )
                dg_categories[category] = category_values - CATEGORY_SHARES[category] * dg_total
                row.update(
                    _aggregate_signal(
                        dg_categories[category], ages, f"dg_cat_{category}_{weather_arm}"
                    )
                )
            if "round_num" in history:
                precut_mask = (
                    pd.to_numeric(history["round_num"], errors="coerce")
                    .isin([1, 2])
                    .to_numpy()
                )
            else:
                precut_mask = np.ones(len(history), dtype=bool)
            row.update(
                _aggregate_signal(
                    dg_total[precut_mask],
                    ages[precut_mask],
                    f"dg_precut_total_{weather_arm}",
                )
            )
            for category in CATEGORY_ORDER:
                row.update(
                    _aggregate_signal(
                        dg_categories[category][precut_mask],
                        ages[precut_mask],
                        f"dg_precut_cat_{category}_{weather_arm}",
                    )
                )
        img_available = (
            _safe_numeric(history, "img_accounting_sg_ott_field_adjusted").notna()
            & _safe_numeric(history, "img_accounting_sg_app_field_adjusted").notna()
        )
        dg_available = _safe_numeric(history, "dg_sg_total_field_adjusted").notna()
        row["img_history_rounds"] = int(img_available.sum())
        row["dg_history_rounds"] = int(dg_available.sum())
        row["img_history_coverage"] = float(img_available.sum() / max(dg_available.sum(), 1))
        if img_available.any():
            row["days_since_last_img_round"] = float(
                (target_date - history.loc[img_available, "round_date"].max()).days
            )
        else:
            row["days_since_last_img_round"] = np.nan
        output.append(row)
    snapshots = pd.DataFrame(output)
    if (
        pd.to_datetime(snapshots["max_source_round_date"], errors="coerce")
        >= snapshots["event_start_date"]
    ).fillna(False).any():
        raise AssertionError("feature snapshot contains a round on/after event start")
    return snapshots


def attach_weather_neutral_labels(
    snapshots: pd.DataFrame,
    rounds: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build W1 post-event labels without using target weather as a feature."""

    target = rounds.copy()
    if "round_num" in target:
        target = target.loc[
            pd.to_numeric(target["round_num"], errors="coerce").isin([1, 2])
        ].copy()
    target = target.merge(
        weather[
            [
                "event_start_date",
                "total_beta_wind",
                "total_beta_dew",
                "coefficient_source",
            ]
        ],
        on="event_start_date",
        how="left",
        validate="many_to_one",
    )
    observed = _safe_numeric(target, "dg_sg_total_field_adjusted")
    target["_neutral_label"] = (
        observed
        + target["total_beta_wind"].fillna(0.0)
        * _safe_numeric(target, "weather_delta_wind").fillna(0.0)
        + target["total_beta_dew"].fillna(0.0)
        * _safe_numeric(target, "weather_delta_dew").fillna(0.0)
    )
    target.loc[observed.isna(), "_neutral_label"] = np.nan
    labels = (
        target.groupby(["event_key", "dg_id_key"], observed=True)["_neutral_label"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "label_dg_sg_neutral_r1_r2_mean",
                "count": "label_dg_sg_neutral_r1_r2_count",
            }
        )
    )
    output = snapshots.drop(
        columns=[
            "label_dg_sg_neutral_r1_r2_mean",
            "label_dg_sg_neutral_r1_r2_count",
        ],
        errors="ignore",
    ).merge(labels, on=["event_key", "dg_id_key"], how="left", validate="one_to_one")
    output["label_dg_sg_neutral_r1_r2_count"] = output[
        "label_dg_sg_neutral_r1_r2_count"
    ].fillna(0).astype(int)
    return output


def candidate_feature_columns(frame: pd.DataFrame, candidate: Candidate) -> list[str]:
    columns = [
        "history_rounds",
        "days_since_last_round",
        "dg_history_rounds",
        "prior_event_withdrawal",
    ]
    arm = candidate.weather_arm
    prefixes: list[str] = []
    if "dg_total" in candidate.groups:
        prefixes.append(f"dg_total_{arm}__")
    if "dg_precut_total" in candidate.groups:
        prefixes.append(f"dg_precut_total_{arm}__")
    if "dg_cat" in candidate.groups:
        prefixes.extend(f"dg_cat_{category}_{arm}__" for category in CATEGORY_ORDER)
    if "dg_long" in candidate.groups:
        prefixes.extend(f"dg_cat_{category}_{arm}__" for category in ("ott", "app"))
    if "dg_precut_cat" in candidate.groups:
        prefixes.extend(
            f"dg_precut_cat_{category}_{arm}__" for category in CATEGORY_ORDER
        )
    if "img_total" in candidate.groups:
        prefixes.append(f"img_total_{arm}__")
    if "img_cat" in candidate.groups:
        prefixes.extend(f"img_cat_{category}_{arm}__" for category in CATEGORY_ORDER)
    if "img_long" in candidate.groups:
        prefixes.extend(f"img_clean_{category}_{arm}__" for category in ("ott", "app"))
    columns.extend(column for column in frame if any(column.startswith(prefix) for prefix in prefixes))
    return list(dict.fromkeys(column for column in columns if column in frame))


class _ColumnPenalty:
    """Multiply standardized columns to express prespecified ridge penalties."""

    def __init__(self, scales: np.ndarray):
        self.scales = np.asarray(scales, dtype=float)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "_ColumnPenalty":
        if x.shape[1] != len(self.scales):
            raise ValueError("column-penalty scale length does not match model matrix")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) * self.scales


def _model(alpha: float, feature_columns: Sequence[str], candidate: Candidate) -> Pipeline:
    penalty_scales = _feature_penalty_scales(feature_columns, candidate)
    if candidate.estimator == "ridge":
        estimator: Any = Ridge(alpha=float(alpha))
    elif candidate.estimator == "huber":
        estimator = HuberRegressor(
            alpha=float(alpha), epsilon=1.35, max_iter=500, tol=1e-5
        )
    else:
        raise ValueError(f"Unknown estimator {candidate.estimator!r}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            ("category_penalty", _ColumnPenalty(penalty_scales)),
            ("estimator", estimator),
        ]
    )


def annual_walk_forward_predictions(
    snapshots: pd.DataFrame,
    candidate: Candidate,
    alpha: float,
    years: Sequence[int],
) -> pd.DataFrame:
    feature_columns = candidate_feature_columns(snapshots, candidate)
    if not feature_columns:
        raise ValueError(f"No feature columns found for {candidate.name}")
    neutral_target = candidate.weather_arm == "w1" and "feature_only" not in candidate.name
    target_column = (
        "label_dg_sg_neutral_r1_r2_mean" if neutral_target else PRIMARY_LABEL
    )
    target_count_column = (
        "label_dg_sg_neutral_r1_r2_count"
        if neutral_target
        else "label_dg_sg_adj_r1_r2_count"
    )
    rows: list[pd.DataFrame] = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = snapshots.loc[
            (snapshots["event_end_date"] < cutoff)
            & snapshots[target_column].notna()
            & (pd.to_numeric(snapshots[target_count_column], errors="coerce") >= 2)
        ].copy()
        test = snapshots.loc[snapshots["event_start_date"].dt.year == year].copy()
        if test.empty:
            continue
        if len(train) < 500:
            raise ValueError(
                f"Only {len(train)} training rows before {year} for {candidate.name}"
            )
        usable_features = [column for column in feature_columns if train[column].notna().any()]
        pipeline = _model(alpha, usable_features, candidate)
        pipeline.fit(train[usable_features], train[target_column])
        prediction = pipeline.predict(test[usable_features])
        piece = test[
            ["event_key", "dg_id_key", "dg_id", "event_start_date", PRIMARY_LABEL]
        ].copy()
        piece["candidate"] = candidate.name
        piece["prediction"] = prediction
        piece["prediction_units"] = "sg_per_round"
        piece["training_target_column"] = target_column
        piece["ridge_alpha"] = float(alpha)
        piece["n_features"] = int(len(usable_features))
        piece["training_rows"] = int(len(train))
        piece["training_max_event_end"] = train["event_end_date"].max()
        piece["history_rounds"] = test["history_rounds"].to_numpy()
        piece["days_since_last_round"] = test["days_since_last_round"].to_numpy()
        piece["img_history_rounds"] = test["img_history_rounds"].to_numpy()
        piece["img_history_coverage"] = test["img_history_coverage"].to_numpy()
        piece["prior_event_withdrawal"] = test["prior_event_withdrawal"].to_numpy()
        uses_img = any(group.startswith("img") for group in candidate.groups)
        piece["reliability_ok"] = (
            (test["history_rounds"] >= 20)
            & (test["days_since_last_round"].fillna(np.inf) <= 180)
            & (~test["prior_event_withdrawal"].fillna(False).astype(bool))
            & (
                ((test["img_history_rounds"] >= 8) & (test["img_history_coverage"] >= 0.25))
                if uses_img
                else True
            )
        ).to_numpy()
        rows.append(piece)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pairwise_training_quotes(quotes: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    training = quotes.loc[
        (quotes["market"] == "72h")
        & (quotes["event_start_date"] < cutoff)
        & quotes["p1_outcome"].isin([0.0, 1.0])
    ].copy()
    # The outcome is common across books.  Each logical golfer pair/event gets
    # one vote so menu duplication and book coverage do not alter the target.
    training = training.sort_values(
        ["mapped_event_key", "pair_key", "book"], kind="mergesort"
    ).drop_duplicates(["mapped_event_key", "pair_key"], keep="first")
    return training


def _feature_penalty_scales(
    feature_columns: Sequence[str], candidate: Candidate
) -> np.ndarray:
    scales = np.ones(len(feature_columns), dtype=float)
    for index, column in enumerate(feature_columns):
        if any(token in column for token in ("_cat_arg_", "_cat_putt_")):
            scales[index] = candidate.short_game_scale
    return scales


def pairwise_player_scores(
    snapshots: pd.DataFrame,
    quotes: pd.DataFrame,
    candidate: Candidate,
    regularization_c: float,
    years: Sequence[int],
) -> pd.DataFrame:
    """Fit a close-free pairwise ranking model and emit neutral player scores."""

    feature_columns = candidate_feature_columns(snapshots, candidate)
    rows: list[pd.DataFrame] = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train_snapshots = snapshots.loc[snapshots["event_end_date"] < cutoff].copy()
        test = snapshots.loc[snapshots["event_start_date"].dt.year == year].copy()
        training_quotes = _pairwise_training_quotes(quotes, cutoff)
        if test.empty:
            continue
        if len(training_quotes) < 250 or len(train_snapshots) < 500:
            raise ValueError(
                f"Insufficient pairwise history before {year}: "
                f"{len(training_quotes)} pairs, {len(train_snapshots)} snapshots"
            )
        usable_features = [
            column for column in feature_columns if train_snapshots[column].notna().any()
        ]
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        scaler = StandardScaler()
        train_matrix = imputer.fit_transform(train_snapshots[usable_features])
        train_matrix = scaler.fit_transform(train_matrix)
        train_matrix *= _feature_penalty_scales(usable_features, candidate)
        train_keys = list(
            zip(train_snapshots["event_key"].astype(str), train_snapshots["dg_id_key"])
        )
        key_to_index = {key: index for index, key in enumerate(train_keys)}
        valid_quotes: list[dict[str, Any]] = []
        left_indices: list[int] = []
        right_indices: list[int] = []
        for quote in training_quotes.to_dict(orient="records"):
            left_key = (str(quote["mapped_event_key"]), quote["p1_id"])
            right_key = (str(quote["mapped_event_key"]), quote["p2_id"])
            if left_key not in key_to_index or right_key not in key_to_index:
                continue
            valid_quotes.append(quote)
            left_indices.append(key_to_index[left_key])
            right_indices.append(key_to_index[right_key])
        if len(valid_quotes) < 200:
            raise ValueError(f"Only {len(valid_quotes)} mapped pairwise rows before {year}")
        pair_matrix = train_matrix[np.asarray(left_indices)] - train_matrix[np.asarray(right_indices)]
        outcome = np.asarray([row["p1_outcome"] for row in valid_quotes], dtype=float)
        event_ids = pd.Series([row["mapped_event_key"] for row in valid_quotes])
        event_counts = event_ids.map(event_ids.value_counts()).to_numpy(float)
        weights = 1.0 / event_counts
        estimator = LogisticRegression(
            C=float(regularization_c),
            penalty="l2",
            fit_intercept=False,
            solver="lbfgs",
            max_iter=1000,
            tol=1e-7,
        )
        estimator.fit(pair_matrix, outcome, sample_weight=weights)
        test_matrix = scaler.transform(imputer.transform(test[usable_features]))
        test_matrix *= _feature_penalty_scales(usable_features, candidate)
        score = test_matrix @ estimator.coef_.ravel()
        piece = test[
            ["event_key", "dg_id_key", "dg_id", "event_start_date", PRIMARY_LABEL]
        ].copy()
        piece["candidate"] = candidate.name
        piece["prediction"] = score
        piece["prediction_units"] = "market_logit_score"
        piece["ridge_alpha"] = float(regularization_c)
        piece["n_features"] = int(len(usable_features))
        piece["training_rows"] = int(len(valid_quotes))
        piece["training_max_event_end"] = train_snapshots["event_end_date"].max()
        for column in (
            "history_rounds",
            "days_since_last_round",
            "img_history_rounds",
            "img_history_coverage",
            "prior_event_withdrawal",
        ):
            piece[column] = test[column].to_numpy()
        piece["reliability_ok"] = (
            (test["history_rounds"] >= 20)
            & (test["days_since_last_round"].fillna(np.inf) <= 180)
            & (~test["prior_event_withdrawal"].fillna(False).astype(bool))
            & (test["img_history_rounds"] >= 8)
            & (test["img_history_coverage"] >= 0.25)
        ).to_numpy()
        rows.append(piece)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def legacy_predictions(snapshots: pd.DataFrame) -> pd.DataFrame:
    legacy_column = "legacy_skill_est_asof"
    if legacy_column not in snapshots:
        raise ValueError(
            "Fair legacy control requires strict-prior legacy_skill_est_asof; "
            "a retrospective cascade prediction is not an acceptable substitute"
        )
    output = snapshots[
        ["event_key", "dg_id_key", "dg_id", "event_start_date", PRIMARY_LABEL]
    ].copy()
    output["candidate"] = RAW_LEGACY_CANDIDATE
    output["prediction"] = pd.to_numeric(snapshots[legacy_column], errors="coerce")
    output["prediction_units"] = "sg_per_round"
    output["ridge_alpha"] = np.nan
    output["n_features"] = 1
    output["training_rows"] = np.nan
    output["training_max_event_end"] = pd.NaT
    output["legacy_source_column"] = legacy_column
    for column in (
        "history_rounds",
        "days_since_last_round",
        "img_history_rounds",
        "img_history_coverage",
        "prior_event_withdrawal",
    ):
        output[column] = snapshots[column].to_numpy()
    output["reliability_ok"] = (
        (snapshots["history_rounds"] >= 20)
        & (snapshots["days_since_last_round"].fillna(np.inf) <= 180)
        & (~snapshots["prior_event_withdrawal"].fillna(False).astype(bool))
    ).to_numpy()
    return output


def legacy_refit_predictions(snapshots: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Fair point-in-time control: refit the old skill input to the new label."""

    if "legacy_skill_est_asof" not in snapshots:
        raise ValueError("legacy_skill_est_asof is required for the fair refit control")
    rows: list[pd.DataFrame] = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = snapshots.loc[
            (snapshots["event_end_date"] < cutoff)
            & snapshots[PRIMARY_LABEL].notna()
            & (pd.to_numeric(snapshots["label_dg_sg_adj_r1_r2_count"], errors="coerce") >= 2)
            & snapshots["legacy_skill_est_asof"].notna()
        ].copy()
        test = snapshots.loc[
            (snapshots["event_start_date"].dt.year == year)
            & snapshots["legacy_skill_est_asof"].notna()
        ].copy()
        if test.empty:
            continue
        if len(train) < 500:
            raise ValueError(f"Only {len(train)} legacy-control training rows before {year}")
        model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        )
        model.fit(train[["legacy_skill_est_asof"]], train[PRIMARY_LABEL])
        piece = test[
            ["event_key", "dg_id_key", "dg_id", "event_start_date", PRIMARY_LABEL]
        ].copy()
        piece["candidate"] = BASELINE_CANDIDATE
        piece["prediction"] = model.predict(test[["legacy_skill_est_asof"]])
        piece["prediction_units"] = "sg_per_round"
        piece["ridge_alpha"] = 1.0
        piece["n_features"] = 1
        piece["training_rows"] = int(len(train))
        piece["training_max_event_end"] = train["event_end_date"].max()
        for column in (
            "history_rounds",
            "days_since_last_round",
            "img_history_rounds",
            "img_history_coverage",
            "prior_event_withdrawal",
        ):
            piece[column] = test[column].to_numpy()
        piece["reliability_ok"] = (
            (test["history_rounds"] >= 20)
            & (test["days_since_last_round"].fillna(np.inf) <= 180)
            & (~test["prior_event_withdrawal"].fillna(False).astype(bool))
        ).to_numpy()
        rows.append(piece)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def retrospective_core_predictions(
    snapshots: pd.DataFrame, bridge_path: Path
) -> pd.DataFrame:
    """Attach exact current-code pred_step2 as a labeled retrospective control."""

    if not bridge_path.exists():
        raise FileNotFoundError(f"Missing retrospective core bridge: {bridge_path}")
    bridge = pd.read_parquet(bridge_path)
    _require(
        bridge,
        ["dg_id", "legacy_core_pred_step2"],
        "legacy core bridge",
    )
    date_column = _first_existing(bridge.columns, ("event_start", "event_start_date"))
    if date_column is None:
        raise ValueError("legacy core bridge has no event-start column")
    bridge = bridge.copy()
    bridge["event_start_date"] = pd.to_datetime(bridge[date_column], errors="coerce").dt.normalize()
    bridge["dg_id_key"] = bridge["dg_id"].map(_stable_id)
    if bridge.duplicated(["event_start_date", "dg_id_key"]).any():
        raise ValueError("legacy core bridge is ambiguous on event date/player")
    base_columns = [
        "event_key",
        "dg_id_key",
        "dg_id",
        "event_start_date",
        PRIMARY_LABEL,
        "history_rounds",
        "days_since_last_round",
        "img_history_rounds",
        "img_history_coverage",
        "prior_event_withdrawal",
    ]
    output = snapshots[base_columns].merge(
        bridge[["event_start_date", "dg_id_key", "legacy_core_pred_step2"]],
        on=["event_start_date", "dg_id_key"],
        how="inner",
        validate="one_to_one",
    )
    output["candidate"] = RETROSPECTIVE_CORE_CANDIDATE
    output["prediction"] = pd.to_numeric(output["legacy_core_pred_step2"], errors="coerce")
    output["prediction_units"] = "sg_per_round_retrospective"
    output["ridge_alpha"] = np.nan
    output["n_features"] = np.nan
    output["training_rows"] = np.nan
    output["training_max_event_end"] = pd.NaT
    output["reliability_ok"] = (
        (output["history_rounds"] >= 20)
        & (output["days_since_last_round"].fillna(np.inf) <= 180)
        & (~output["prior_event_withdrawal"].fillna(False).astype(bool))
    )
    return output


def _event_catalog(rounds: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    optional = [column for column in ("event_name", "year", "event_id") if column in events]
    catalog = events[["event_key", "event_start_date", *optional]].drop_duplicates("event_key")
    end = rounds.groupby("event_key", observed=True)["round_date"].max().rename("local_end_date")
    catalog = catalog.merge(end, on="event_key", how="left", validate="one_to_one")
    if "event_name" not in catalog:
        catalog["event_name"] = ""
    catalog["event_name_norm"] = catalog["event_name"].map(_normalize_text)
    return catalog


def _choose_event_mapping(odds_event: Mapping[str, Any], catalog: pd.DataFrame) -> dict[str, Any]:
    completed = pd.to_datetime(odds_event.get("event_completed"), errors="coerce")
    completed = completed.normalize() if pd.notna(completed) else pd.NaT
    query_year = int(odds_event["query_year"])
    name = _normalize_text(odds_event.get("event_name"))
    event_id = _stable_id(odds_event.get("event_id"))
    candidates = catalog.copy()
    method = ""
    if pd.notna(completed):
        exact_end = candidates.loc[candidates["local_end_date"] == completed]
        if not exact_end.empty:
            candidates = exact_end
            method = "exact_completion_date"
        else:
            expected_start = completed - pd.Timedelta(days=3)
            nearby = candidates.loc[
                (candidates["event_start_date"] - expected_start).abs() <= pd.Timedelta(days=2)
            ]
            if not nearby.empty:
                candidates = nearby
                method = "near_expected_start"
    if not method:
        same_year = candidates.loc[candidates["event_start_date"].dt.year == query_year]
        if "event_id" in same_year:
            exact_id = same_year.loc[same_year["event_id"].map(_stable_id) == event_id]
        else:
            exact_id = pd.DataFrame()
        if not exact_id.empty:
            candidates = exact_id
            method = "same_year_event_id_fallback"
        else:
            candidates = same_year
            method = "same_year_name_fallback"
    if candidates.empty:
        return {
            "mapped_event_key": None,
            "mapping_method": "unmapped",
            "mapping_name_similarity": np.nan,
            "mapping_candidates": 0,
        }
    scores = candidates["event_name_norm"].map(
        lambda local: SequenceMatcher(None, name, local).ratio() if name and local else 0.0
    )
    chosen_index = scores.idxmax() if len(candidates) > 1 else candidates.index[0]
    chosen = candidates.loc[chosen_index]
    similarity = float(scores.loc[chosen_index]) if len(scores) else 0.0
    ambiguous = len(candidates) > 1 and similarity < 0.45
    return {
        "mapped_event_key": None if ambiguous else chosen["event_key"],
        "mapping_method": "ambiguous" if ambiguous else method,
        "mapping_name_similarity": similarity,
        "mapping_candidates": int(len(candidates)),
        "mapped_event_start_date": chosen["event_start_date"],
        "mapped_local_end_date": chosen["local_end_date"],
        "mapped_local_event_name": chosen.get("event_name"),
    }


def load_matchup_quotes(
    cache_dir: Path,
    catalog: pd.DataFrame,
    audited_crosswalk_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(HistoricalOddsCacheReader(cache_dir).matchups())
    if raw.empty:
        raise ValueError("Historical matchup cache has no rows")
    raw["market"] = raw["bet_type"].map(_market_name)
    raw = raw.loc[raw["market"].isin(["72h", "r1"])].copy()
    raw["book"] = raw["book"].map(_normalize_text).str.replace(" ", "", regex=False)
    raw = raw.loc[raw["book"].isin(BOOKS)].copy()
    event_columns = ["query_year", "event_id", "event_name", "event_completed"]
    odds_events = raw[event_columns].drop_duplicates().reset_index(drop=True)
    if not audited_crosswalk_path.exists():
        raise FileNotFoundError(
            f"Audited odds event crosswalk is required: {audited_crosswalk_path}"
        )
    crosswalk = pd.read_parquet(audited_crosswalk_path).copy()
    _require(
        crosswalk,
        [
            "odds_calendar_year",
            "odds_event_id",
            "odds_event_name",
            "odds_event_completed_date",
            "local_event_key",
            "event_identity_status",
        ],
        "audited odds-event crosswalk",
    )
    crosswalk["query_year"] = pd.to_numeric(
        crosswalk["odds_calendar_year"], errors="coerce"
    ).astype("Int64")
    crosswalk["event_id_key"] = crosswalk["odds_event_id"].map(_stable_id)
    crosswalk["event_name_key"] = crosswalk["odds_event_name"].map(_normalize_text)
    crosswalk["event_completed_key"] = pd.to_datetime(
        crosswalk["odds_event_completed_date"], errors="coerce"
    ).dt.normalize()
    raw["query_year"] = pd.to_numeric(raw["query_year"], errors="coerce").astype("Int64")
    raw["event_id_key"] = raw["event_id"].map(_stable_id)
    raw["event_name_key"] = raw["event_name"].map(_normalize_text)
    raw["event_completed_key"] = pd.to_datetime(raw["event_completed"], errors="coerce").dt.normalize()
    join_keys = ["query_year", "event_id_key", "event_name_key", "event_completed_key"]
    if crosswalk.duplicated(join_keys).any():
        raise ValueError("audited odds-event crosswalk has duplicate odds identities")
    audit_columns = [
        column
        for column in crosswalk.columns
        if column
        not in {
            "odds_calendar_year",
            "odds_event_id",
            "odds_event_name",
            "odds_event_completed_date",
            *join_keys,
        }
    ]
    mapping = odds_events.copy()
    mapping["query_year"] = pd.to_numeric(mapping["query_year"], errors="coerce").astype("Int64")
    mapping["event_id_key"] = mapping["event_id"].map(_stable_id)
    mapping["event_name_key"] = mapping["event_name"].map(_normalize_text)
    mapping["event_completed_key"] = pd.to_datetime(
        mapping["event_completed"], errors="coerce"
    ).dt.normalize()
    mapping = mapping.merge(
        crosswalk[join_keys + audit_columns], on=join_keys, how="left", validate="one_to_one"
    )
    invalid = mapping["event_identity_status"].ne("matched") | mapping["local_event_key"].isna()
    explicit_bad = pd.Series(False, index=mapping.index)
    for column in ("event_identity_ambiguous", "local_event_has_multiple_odds_ids"):
        if column in mapping:
            explicit_bad |= mapping[column].fillna(False).astype(bool)
    if (invalid | explicit_bad).any():
        examples = mapping.loc[invalid | explicit_bad, event_columns + ["event_identity_status", "local_event_key"]]
        raise ValueError(
            "Audited odds-event crosswalk is unmatched/ambiguous for requested markets: "
            + examples.head(20).to_json(orient="records", date_format="iso")
        )
    mapping["mapped_event_key"] = mapping["local_event_key"]
    local_start_column = _first_existing(
        mapping.columns, ("local_event_start_date", "local_event_start")
    )
    if local_start_column is not None:
        mapping["mapped_event_start_date"] = pd.to_datetime(
            mapping[local_start_column], errors="coerce"
        ).dt.normalize()
    else:
        local_start = catalog.set_index("event_key")["event_start_date"]
        mapping["mapped_event_start_date"] = mapping["local_event_key"].map(local_start)
    mapping["mapping_method"] = "audited_crosswalk:" + mapping.get(
        "event_identity_method", pd.Series("matched", index=mapping.index)
    ).astype(str)
    similarity_column = _first_existing(
        mapping.columns,
        ("normalized_name_similarity", "event_name_similarity", "name_similarity"),
    )
    mapping["mapping_name_similarity"] = (
        pd.to_numeric(mapping[similarity_column], errors="coerce")
        if similarity_column is not None
        else np.nan
    )
    mapping["mapping_candidates"] = 1
    raw = raw.merge(
        mapping[event_columns + [
            "mapped_event_key",
            "mapped_event_start_date",
            "mapping_method",
            "mapping_name_similarity",
            "mapping_candidates",
        ]],
        on=event_columns,
        how="left",
        validate="many_to_one",
    )
    raw["p1_id"] = raw["p1_dg_id"].map(_stable_id)
    raw["p2_id"] = raw["p2_dg_id"].map(_stable_id)
    for side in ("p1", "p2"):
        raw[f"{side}_close"] = pd.to_numeric(raw[f"{side}_close"], errors="coerce")
        raw[f"{side}_outcome"] = pd.to_numeric(raw[f"{side}_outcome"], errors="coerce")
    valid_odds = lambda values: values.notna() & ((values <= -100) | (values >= 100))
    raw = raw.loc[
        raw["mapped_event_key"].notna()
        & raw["p1_id"].ne("")
        & raw["p2_id"].ne("")
        & valid_odds(raw["p1_close"])
        & valid_odds(raw["p2_close"])
    ].copy()
    raw["pair_key"] = raw.apply(
        lambda row: "|".join(sorted((row["p1_id"], row["p2_id"]))), axis=1
    )
    raw["close_time_sort"] = pd.to_datetime(raw.get("close_time"), errors="coerce", utc=True)
    raw = raw.sort_values("close_time_sort", ascending=False, kind="mergesort")
    quote_keys = ["mapped_event_key", "market", "book", "pair_key"]
    raw = raw.drop_duplicates(quote_keys, keep="first").reset_index(drop=True)
    raw["event_start_date"] = pd.to_datetime(raw["mapped_event_start_date"]).dt.normalize()
    raw["quote_id"] = (
        raw["mapped_event_key"].astype(str)
        + "|"
        + raw["market"]
        + "|"
        + raw["book"]
        + "|"
        + raw["pair_key"]
    )
    if raw.duplicated("quote_id").any():
        raise AssertionError("normalized matchup quote IDs are not unique")
    validate_normalized_matchup_quotes(raw)
    return raw, mapping


def augment_events_from_market_board(
    events: pd.DataFrame,
    quotes: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add quoted golfers without a target-event result row as unlabeled targets."""

    player_rows: list[dict[str, Any]] = []
    for side in ("p1", "p2"):
        player_rows.extend(
            {
                "event_key": row["mapped_event_key"],
                "event_start_date": row["event_start_date"],
                "dg_id_key": row[f"{side}_id"],
                "dg_id": row.get(f"{side}_dg_id"),
                "player_name": row.get(f"{side}_player_name"),
            }
            for row in quotes.to_dict(orient="records")
        )
    board = pd.DataFrame(player_rows).drop_duplicates(["event_key", "dg_id_key"])
    existing = events[["event_key", "dg_id_key"]].drop_duplicates()
    board = board.merge(
        existing.assign(_exists=True), on=["event_key", "dg_id_key"], how="left"
    )
    missing = board.loc[board["_exists"].isna()].drop(columns="_exists")
    if missing.empty:
        return events, missing
    catalog_lookup = catalog.set_index("event_key")
    additions = pd.DataFrame(np.nan, index=np.arange(len(missing)), columns=events.columns)
    for column in missing.columns:
        if column in additions:
            additions[column] = missing[column].to_numpy()
    additions["event_end_date"] = missing["event_key"].map(catalog_lookup["local_end_date"])
    for column in ("event_name", "year", "event_id"):
        if column in additions and column in catalog_lookup:
            additions[column] = missing["event_key"].map(catalog_lookup[column])
    if "market_evaluation_eligible" in additions:
        additions["market_evaluation_eligible"] = True
    augmented = pd.concat([events, additions], ignore_index=True)
    if augmented.duplicated(["event_key", "dg_id_key"]).any():
        raise AssertionError("market-board augmentation created duplicate event/player rows")
    return augmented, missing


def attach_skill_differences(quotes: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    reliability_columns = [
        column
        for column in (
            "reliability_ok",
            "history_rounds",
            "days_since_last_round",
            "img_history_rounds",
            "img_history_coverage",
            "prior_event_withdrawal",
        )
        if column in predictions
    ]
    lookup = predictions[
        ["candidate", "event_key", "dg_id_key", "prediction", *reliability_columns]
    ].drop_duplicates(["candidate", "event_key", "dg_id_key"])
    left = lookup.rename(
        columns={
            "dg_id_key": "p1_id",
            "prediction": "p1_skill",
            **{column: f"p1_{column}" for column in reliability_columns},
        }
    )
    right = lookup.rename(
        columns={
            "dg_id_key": "p2_id",
            "prediction": "p2_skill",
            **{column: f"p2_{column}" for column in reliability_columns},
        }
    )
    output = quotes.merge(
        left,
        left_on=["mapped_event_key", "p1_id"],
        right_on=["event_key", "p1_id"],
        how="inner",
    ).drop(columns=["event_key"])
    output = output.merge(
        right,
        left_on=["candidate", "mapped_event_key", "p2_id"],
        right_on=["candidate", "event_key", "p2_id"],
        how="inner",
    ).drop(columns=["event_key"])
    output["skill_difference"] = output["p1_skill"] - output["p2_skill"]
    if "p1_reliability_ok" in output and "p2_reliability_ok" in output:
        output["pair_reliability_ok"] = (
            output["p1_reliability_ok"].fillna(False).astype(bool)
            & output["p2_reliability_ok"].fillna(False).astype(bool)
        )
    output = output.loc[np.isfinite(output["skill_difference"])].copy()
    return output


def validate_normalized_matchup_quotes(frame: pd.DataFrame) -> dict[str, int]:
    """Fail closed on pair identity and the two cached outcome orientations."""

    required = {"p1_id", "p2_id", "pair_key", "p1_outcome"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"normalized matchup quotes lack integrity fields: {missing}")
    p1 = frame["p1_id"].map(_stable_id)
    p2 = frame["p2_id"].map(_stable_id)
    if p1.eq("").any() or p2.eq("").any() or p1.eq(p2).any():
        raise ValueError("normalized matchup quotes contain blank/self player pairs")
    expected_pair = pd.Series(
        ["|".join(sorted((left, right))) for left, right in zip(p1, p2)],
        index=frame.index,
        dtype="object",
    )
    observed_pair = frame["pair_key"].astype(str)
    bad_pair = observed_pair.ne(expected_pair)
    if bad_pair.any():
        raise ValueError(
            "normalized matchup pair_key disagrees with reconstructed player IDs: "
            + frame.loc[bad_pair, ["p1_id", "p2_id", "pair_key"]]
            .head(10)
            .to_json(orient="records")
        )

    outcome_pairs_checked = 0
    if "p2_outcome" in frame:
        left_outcome = pd.to_numeric(frame["p1_outcome"], errors="coerce")
        right_outcome = pd.to_numeric(frame["p2_outcome"], errors="coerce")
        one_missing = left_outcome.isna() ^ right_outcome.isna()
        graded = left_outcome.notna() & right_outcome.notna()
        allowed = {0.0, 0.5, 1.0}
        invalid_value = graded & (
            ~left_outcome.isin(allowed) | ~right_outcome.isin(allowed)
        )
        noncomplement = graded & ~np.isclose(
            left_outcome + right_outcome, 1.0, rtol=0.0, atol=1e-12
        )
        if one_missing.any() or invalid_value.any() or noncomplement.any():
            raise ValueError(
                "cached p1/p2 matchup outcomes are not exact complements"
            )
        outcome_pairs_checked = int(graded.sum())
    return {
        "quote_rows": int(len(frame)),
        "pair_keys_reconstructed": int(len(frame)),
        "complementary_outcome_rows": outcome_pairs_checked,
    }


def _resolved_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["p1_outcome"].isin([0.0, 1.0])].copy()


def _canonical_logical_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """One orientation and one row per logical event/market/player pair."""

    if frame.empty:
        return frame.copy()
    output = frame.copy()
    validate_normalized_matchup_quotes(output)
    p1 = output["p1_id"].map(_stable_id)
    p2 = output["p2_id"].map(_stable_id)
    output["p1_id"] = p1
    output["p2_id"] = p2
    output["pair_key"] = [
        "|".join(sorted((left, right))) for left, right in zip(p1, p2)
    ]
    canonical_left = output["pair_key"].str.split("|", regex=False).str[0]
    flip = output["p1_id"].ne(canonical_left)
    output["canonical_skill_difference"] = np.where(
        flip, -output["skill_difference"], output["skill_difference"]
    )
    outcome = pd.to_numeric(output["p1_outcome"], errors="coerce")
    output["canonical_outcome"] = np.where(
        flip & outcome.isin([0.0, 1.0]), 1.0 - outcome, outcome
    )
    output["canonical_left_close"] = np.where(
        flip, output["p2_close"], output["p1_close"]
    )
    output["canonical_right_close"] = np.where(
        flip, output["p1_close"], output["p2_close"]
    )
    if "p1_open" in output and "p2_open" in output:
        output["canonical_left_open"] = np.where(
            flip, output["p2_open"], output["p1_open"]
        )
        output["canonical_right_open"] = np.where(
            flip, output["p1_open"], output["p2_open"]
        )
    if "p1_probability" in output:
        push = pd.to_numeric(output.get("push_probability", 0.0), errors="coerce").fillna(0.0)
        output["canonical_probability"] = np.where(
            flip,
            1.0 - output["p1_probability"] - push,
            output["p1_probability"],
        )
        output["canonical_push_probability"] = push
    book_rank = {"pinnacle": 0, "betcris": 1, "betonline": 2}
    output["_book_rank"] = output["book"].map(book_rank).fillna(99)
    keys = ["candidate", "mapped_event_key", "market", "pair_key"]
    invariant_columns = [
        column
        for column in (
            "canonical_skill_difference",
            "canonical_outcome",
            "canonical_probability",
            "canonical_push_probability",
        )
        if column in output
    ]
    for column in invariant_columns:
        inconsistent = output.groupby(keys, observed=True)[column].nunique(
            dropna=False
        ).gt(1)
        if inconsistent.any():
            raise ValueError(
                f"logical matchup reconstruction disagrees across books for {column}"
            )
    output = output.sort_values(keys + ["_book_rank"], kind="mergesort").drop_duplicates(keys)
    return output.drop(columns="_book_rank")


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("mapped_event_key", observed=True)["quote_id"].transform("count")
    return 1.0 / counts.to_numpy(float)


def _win_push_probabilities(
    difference: np.ndarray, sigma: float, tie_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = norm.cdf((difference - tie_threshold) / sigma)
    loss = norm.cdf((-difference - tie_threshold) / sigma)
    push = np.maximum(1.0 - win - loss, 0.0)
    return win, push, loss


def fit_probability_params(discovery: pd.DataFrame) -> tuple[float, float]:
    logical = _canonical_logical_pairs(discovery)
    logical = logical.loc[logical["canonical_outcome"].isin([0.0, 0.5, 1.0])].copy()
    if len(logical) < 100:
        return np.nan, np.nan
    diff = logical["canonical_skill_difference"].to_numpy(float)
    outcome = logical["canonical_outcome"].to_numpy(float)
    weights = _event_equal_weights(logical)

    def objective(parameters: np.ndarray) -> float:
        sigma = math.exp(float(parameters[0]))
        threshold = math.exp(float(parameters[1]))
        win, push, loss = _win_push_probabilities(diff, sigma, threshold)
        probability = np.where(outcome == 1.0, win, np.where(outcome == 0.5, push, loss))
        return float(-np.sum(weights * np.log(np.clip(probability, EPS, 1.0))) / np.sum(weights))

    result = minimize(
        objective,
        np.array([math.log(1.5), math.log(0.10)]),
        method="L-BFGS-B",
        bounds=[(math.log(0.01), math.log(100.0)), (math.log(1e-5), math.log(5.0))],
    )
    if not result.success:
        return np.nan, np.nan
    return float(math.exp(result.x[0])), float(math.exp(result.x[1]))


def score_probabilities(
    frame: pd.DataFrame, sigma: float, tie_threshold: float
) -> pd.DataFrame:
    output = frame.copy()
    output["sigma"] = float(sigma)
    output["tie_threshold"] = float(tie_threshold)
    win, push, _ = _win_push_probabilities(
        output["skill_difference"].to_numpy(float), sigma, tie_threshold
    )
    output["p1_probability"] = win
    output["push_probability"] = push
    output["p1_probability"] = output["p1_probability"].clip(EPS, 1.0 - EPS)
    output["p1_pricing_probability"] = (
        output["p1_probability"] / (1.0 - output["push_probability"])
    ).clip(EPS, 1.0 - EPS)
    return output


def proper_score_summary(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    logical = _canonical_logical_pairs(frame)
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    for (candidate, market), group in resolved.groupby(["candidate", "market"], observed=True):
        y = group["canonical_outcome"].to_numpy(float)
        p = (
            group["canonical_probability"]
            / (1.0 - group["canonical_push_probability"])
        ).clip(EPS, 1.0 - EPS).to_numpy(float)
        group = group.assign(
            log_loss=-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)),
            brier=np.square(p - y),
        )
        event_score = group.groupby("mapped_event_key", observed=True)[["log_loss", "brier"]].mean()
        # De-vigged same-book close is an evaluation benchmark, never a feature.
        implied1 = np.asarray(
            american_implied_probability(group["canonical_left_close"]), dtype=float
        )
        implied2 = np.asarray(
            american_implied_probability(group["canonical_right_close"]), dtype=float
        )
        market_p = np.clip(implied1 / (implied1 + implied2), EPS, 1.0 - EPS)
        market_loss = -(y * np.log(market_p) + (1.0 - y) * np.log(1.0 - market_p))
        market_events = pd.DataFrame(
            {"event": group["mapped_event_key"].to_numpy(), "loss": market_loss}
        ).groupby("event", observed=True)["loss"].mean()
        rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "market": market,
                "events": int(event_score.index.nunique()),
                "resolved_quotes": int(len(group)),
                "event_equal_log_loss": float(event_score["log_loss"].mean()),
                "event_equal_brier": float(event_score["brier"].mean()),
                "event_equal_devig_close_log_loss": float(market_events.mean()),
                "mean_probability": float(p.mean()),
                "observed_win_rate": float(y.mean()),
                "sigma": float(group["sigma"].iloc[0]),
                "tie_threshold": float(group["tie_threshold"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    logical = _canonical_logical_pairs(frame)
    resolved = logical.loc[logical["canonical_outcome"].isin([0.0, 1.0])].copy()
    for (candidate, market), group in resolved.groupby(["candidate", "market"], observed=True):
        y = group["canonical_outcome"].to_numpy(float)
        conditional = (
            group["canonical_probability"]
            / (1.0 - group["canonical_push_probability"])
        ).clip(EPS, 1.0 - EPS)
        x = logit(conditional.to_numpy(float))
        weights = _event_equal_weights(group)

        def objective(parameters: np.ndarray) -> float:
            probability = np.clip(expit(parameters[0] + parameters[1] * x), EPS, 1.0 - EPS)
            loss = -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
            return float(np.sum(weights * loss) / np.sum(weights))

        fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
        intercept, slope = fit.x if fit.success else (np.nan, np.nan)
        bins = pd.qcut(conditional, q=10, duplicates="drop")
        calibration = group.assign(
            _bin=bins, _conditional_probability=conditional.to_numpy()
        ).groupby("_bin", observed=True).agg(
            predicted=("_conditional_probability", "mean"),
            actual=("canonical_outcome", "mean"),
            n=("pair_key", "count"),
        )
        ece = float(
            np.average(
                np.abs(calibration["predicted"] - calibration["actual"]),
                weights=calibration["n"],
            )
        )
        rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "market": market,
                "calibration_intercept": float(intercept),
                "calibration_slope": float(slope),
                "ece_decile": ece,
                "fit_success": bool(fit.success),
                "resolved_quotes": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def expand_selection_lines(scored_quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    line_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for row in scored_quotes.to_dict(orient="records"):
        for side, other, probability in (
            ("p1", "p2", row["p1_probability"]),
            (
                "p2",
                "p1",
                1.0 - row["p1_probability"] - row.get("push_probability", 0.0),
            ),
        ):
            selection_key = f"{row['candidate']}|{row['quote_id']}|{side}"
            line_rows.append(
                {
                    "selection_key": selection_key,
                    "candidate": row["candidate"],
                    "event_id": row["mapped_event_key"],
                    "event_key": row["mapped_event_key"],
                    "event_start_date": row["event_start_date"],
                    "market": "tournament_matchup",
                    "round_market": row["market"],
                    "selection_id": row[f"{side}_id"],
                    "selection_name": row.get(f"{side}_player_name"),
                    "opponent_id": row[f"{other}_id"],
                    "opponent_name": row.get(f"{other}_player_name"),
                    "book": row["book"],
                    "close_odds": row[f"{side}_close"],
                    "close_time": row.get("close_time"),
                    "tie_rule": row.get("tie_rule"),
                    "outcome_numeric": row.get(f"{side}_outcome"),
                    "outcome": row.get(f"{side}_outcome_text"),
                    "pair_key": row["pair_key"],
                    "pair_reliability_ok": bool(row.get("pair_reliability_ok", False)),
                }
            )
            probability_rows.append(
                {
                    "selection_key": selection_key,
                    "probability": float(probability),
                    "push_probability": float(row.get("push_probability", 0.0)),
                }
            )
    return pd.DataFrame(line_rows), pd.DataFrame(probability_rows)


def _best_lines(lines: pd.DataFrame, policy: str) -> pd.DataFrame:
    keys = [
        "candidate",
        "event_key",
        "round_market",
        "selection_id",
        "opponent_id",
    ]
    if policy in BOOKS:
        chosen = lines.loc[lines["book"] == policy].copy()
        chosen["line_policy"] = policy
        return chosen.drop_duplicates(keys, keep="first")
    if policy != "best_of_three":
        raise ValueError(f"Unknown line policy: {policy}")
    chosen = lines.sort_values(
        keys + ["close_odds", "book"],
        ascending=[True] * len(keys) + [False, True],
        kind="mergesort",
    ).drop_duplicates(keys, keep="first")
    chosen["line_policy"] = policy
    return chosen


def _ties_lose_text(value: object) -> bool:
    return ties_lose(value)


def select_and_settle_bets(
    scored: pd.DataFrame,
    policy: str,
    sizing: str,
    min_ev: float = EV_GATE,
) -> pd.DataFrame:
    lines, probabilities = expand_selection_lines(scored)
    if lines.empty:
        return pd.DataFrame()
    chosen = _best_lines(lines, policy)
    joined = chosen.merge(probabilities, on="selection_key", how="inner", validate="one_to_one")
    decimal = np.asarray(american_decimal_odds(joined["close_odds"]), dtype=float)
    implied = 1.0 / decimal
    effective_push = np.where(
        joined["tie_rule"].map(_ties_lose_text).to_numpy(),
        0.0,
        joined["push_probability"].to_numpy(float),
    )
    loss_probability = 1.0 - joined["probability"].to_numpy(float) - effective_push
    joined["implied_probability"] = implied
    joined["pricing_probability"] = joined["probability"] / (1.0 - effective_push)
    joined["probability_edge"] = joined["pricing_probability"] - implied
    joined["ev_per_unit"] = (
        joined["probability"] * (decimal - 1.0) - loss_probability
    )
    reliability_arm = joined["candidate"].str.endswith("__reliability_gate")
    unreliable_large_edge = (
        reliability_arm
        & (joined["ev_per_unit"] >= 0.10)
        & (~joined["pair_reliability_ok"].fillna(False).astype(bool))
    )
    joined = joined.loc[~unreliable_large_edge].copy()
    joined = joined.loc[joined["ev_per_unit"] >= min_ev].copy()
    # One direction per canonical matchup.  This also prevents a true
    # cross-book arbitrage from being credited as model alpha.
    direction_keys = ["candidate", "event_key", "round_market", "pair_key"]
    joined = joined.sort_values(
        direction_keys + ["ev_per_unit", "close_odds", "book"],
        ascending=[True] * len(direction_keys) + [False, False, True],
        kind="mergesort",
    ).drop_duplicates(direction_keys, keep="first")
    if sizing == "flat_1u":
        joined["stake"] = 1.0
    elif sizing == "production_risk_to_win":
        joined["stake"] = np.where(
            joined["close_odds"] < 0,
            joined["close_odds"].abs() / 100.0,
            1.0,
        )
    else:
        raise ValueError(f"Unknown sizing: {sizing}")
    joined["sizing"] = sizing
    return settle_bets(joined)


def _max_drawdown(bets: pd.DataFrame) -> float:
    if bets.empty:
        return 0.0
    by_event = (
        bets.groupby(["event_start_date", "event_key"], observed=True)["pnl"].sum().sort_index()
    )
    cumulative = by_event.cumsum().to_numpy(float)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])[:-1]
    return float(np.max(peak - cumulative)) if len(cumulative) else 0.0


def betting_performance(bets: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bets.empty:
        return pd.DataFrame()
    for keys, group in bets.groupby(
        ["candidate", "round_market", "line_policy", "sizing", "opportunity_mode"],
        observed=True,
    ):
        candidate, market, policy, sizing, opportunity_mode = keys
        stake = float(group["stake"].sum())
        pnl = float(group["pnl"].sum())
        rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "market": market,
                "line_policy": policy,
                "sizing": sizing,
                "opportunity_mode": opportunity_mode,
                "events": int(group["event_key"].nunique()),
                "bets": int(len(group)),
                "stake": stake,
                "pnl": pnl,
                "roi": pnl / stake if stake else np.nan,
                "max_drawdown": _max_drawdown(group),
                "mean_ev": float(group["ev_per_unit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def edge_bucket_performance(scored: pd.DataFrame, sample: str, policy: str) -> pd.DataFrame:
    lines, probabilities = expand_selection_lines(scored)
    chosen = _best_lines(lines, policy)
    joined = chosen.merge(probabilities, on="selection_key", how="inner", validate="one_to_one")
    decimal = np.asarray(american_decimal_odds(joined["close_odds"]), dtype=float)
    effective_push = np.where(
        joined["tie_rule"].map(_ties_lose_text).to_numpy(),
        0.0,
        joined["push_probability"].to_numpy(float),
    )
    loss_probability = 1.0 - joined["probability"].to_numpy(float) - effective_push
    joined["ev_per_unit"] = joined["probability"] * (decimal - 1.0) - loss_probability
    reliability_arm = joined["candidate"].str.endswith("__reliability_gate")
    joined = joined.loc[
        ~(
            reliability_arm
            & (joined["ev_per_unit"] >= 0.10)
            & (~joined["pair_reliability_ok"].fillna(False).astype(bool))
        )
    ].copy()
    direction_keys = ["candidate", "event_key", "round_market", "pair_key"]
    joined = joined.sort_values(
        direction_keys + ["ev_per_unit"], ascending=[True] * len(direction_keys) + [False]
    ).drop_duplicates(direction_keys)
    joined["edge_bucket"] = pd.cut(
        joined["ev_per_unit"],
        bins=[-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-10%", ">10%"],
        right=False,
    )
    joined["stake"] = 1.0
    settled = settle_bets(joined)
    rows: list[dict[str, Any]] = []
    for keys, group in settled.groupby(
        ["candidate", "round_market", "edge_bucket"], observed=True
    ):
        candidate, market, bucket = keys
        rows.append(
            {
                "sample": sample,
                "line_policy": policy,
                "candidate": candidate,
                "market": market,
                "edge_bucket": str(bucket),
                "bets": int(len(group)),
                "events": int(group["event_key"].nunique()),
                "pnl": float(group["pnl"].sum()),
                "roi": float(group["pnl"].sum() / group["stake"].sum()),
                "mean_model_ev": float(group["ev_per_unit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_event_bootstrap(
    scored: pd.DataFrame,
    bets: pd.DataFrame,
    sample: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Paired whole-event intervals versus legacy, including zero-bet events."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    resolved = _canonical_logical_pairs(scored)
    resolved = resolved.loc[resolved["canonical_outcome"].isin([0.0, 1.0])].copy()
    resolved["_conditional_probability"] = (
        resolved["canonical_probability"]
        / (1.0 - resolved["canonical_push_probability"])
    ).clip(EPS, 1.0 - EPS)
    resolved["log_loss"] = -(
        resolved["canonical_outcome"] * np.log(resolved["_conditional_probability"])
        + (1.0 - resolved["canonical_outcome"])
        * np.log(1.0 - resolved["_conditional_probability"])
    )
    for market in sorted(resolved["market"].unique()):
        baseline_name = BASELINE_CANDIDATE
        market_frame = resolved.loc[resolved["market"] == market]
        baseline = market_frame.loc[market_frame["candidate"] == baseline_name, [
            "pair_key", "mapped_event_key", "log_loss"
        ]].rename(columns={"log_loss": "baseline_loss"})
        if baseline.empty:
            continue
        for candidate in sorted(
            value for value in market_frame["candidate"].unique() if value != baseline_name
        ):
            challenger = market_frame.loc[
                market_frame["candidate"] == candidate,
                ["pair_key", "mapped_event_key", "log_loss"],
            ].rename(columns={"log_loss": "candidate_loss"})
            paired_quotes = baseline.merge(
                challenger,
                on=["pair_key", "mapped_event_key"],
                how="inner",
                validate="one_to_one",
            )
            paired = paired_quotes.assign(
                delta=paired_quotes["candidate_loss"] - paired_quotes["baseline_loss"]
            ).groupby("mapped_event_key", observed=True)["delta"].mean()
            if paired.empty:
                continue
            delta = paired.to_numpy(float)
            indices = rng.integers(0, len(delta), size=(draws, len(delta)))
            samples = delta[indices].mean(axis=1)
            rows.append(
                {
                    "sample": sample,
                    "metric": "event_equal_log_loss_delta",
                    "market": market,
                    "candidate": candidate,
                    "baseline": baseline_name,
                    "line_policy": "all_quotes",
                    "sizing": "none",
                    "events": int(len(delta)),
                    "common_quotes": int(len(paired_quotes)),
                    "estimate": float(delta.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "probability_better": float(np.mean(samples < 0.0)),
                    "draws": int(draws),
                }
            )
    if not bets.empty:
        group_keys = ["round_market", "line_policy", "sizing", "opportunity_mode"]
        for keys, slice_frame in bets.groupby(group_keys, observed=True):
            market, policy, sizing, opportunity_mode = keys
            all_events = sorted(
                scored.loc[scored["market"] == market, "mapped_event_key"].unique()
            )
            event_model = (
                slice_frame.groupby(["candidate", "event_key"], observed=True)["pnl"]
                .sum()
                .unstack("candidate")
                .reindex(all_events, fill_value=0.0)
                .fillna(0.0)
            )
            baseline_name = BASELINE_CANDIDATE
            if baseline_name not in event_model:
                continue
            for candidate in (column for column in event_model if column != baseline_name):
                delta = event_model[candidate].to_numpy() - event_model[baseline_name].to_numpy()
                indices = rng.integers(0, len(delta), size=(draws, len(delta)))
                # Sum P&L over an event-resampled portfolio of the same size.
                samples = delta[indices].sum(axis=1)
                rows.append(
                    {
                        "sample": sample,
                        "metric": "pnl_delta",
                        "market": market,
                        "candidate": candidate,
                        "baseline": baseline_name,
                        "line_policy": policy,
                        "sizing": sizing,
                        "opportunity_mode": opportunity_mode,
                        "events": int(len(delta)),
                        "estimate": float(delta.sum()),
                        "ci_low": float(np.quantile(samples, 0.025)),
                        "ci_high": float(np.quantile(samples, 0.975)),
                        "probability_better": float(np.mean(samples > 0.0)),
                        "draws": int(draws),
                    }
                )
    return pd.DataFrame(rows)


def fit_candidate_alphas(
    snapshots: pd.DataFrame,
    quotes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Choose ridge alpha by discovery market log loss, not target RMSE."""

    prediction_pieces = [legacy_predictions(snapshots)]
    for candidate in CANDIDATES:
        for alpha in candidate.alphas:
            piece = annual_walk_forward_predictions(
                snapshots, candidate, alpha, years=(2021, 2022)
            )
            piece["candidate"] = f"{candidate.name}__alpha_{alpha:g}"
            prediction_pieces.append(piece)
    for candidate in PAIRWISE_CANDIDATES:
        for regularization_c in candidate.alphas:
            piece = pairwise_player_scores(
                snapshots,
                quotes,
                candidate,
                regularization_c,
                years=(2021, 2022),
            )
            piece["candidate"] = f"{candidate.name}__c_{regularization_c:g}"
            prediction_pieces.append(piece)
    predictions = pd.concat(prediction_pieces, ignore_index=True)
    discovery_quotes = quotes.loc[
        quotes["event_start_date"].between(DISCOVERY_START, DISCOVERY_END)
    ]
    paired = attach_skill_differences(discovery_quotes, predictions)
    scored_pieces: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    for (candidate, market), group in paired.groupby(["candidate", "market"], observed=True):
        sigma, tie_threshold = fit_probability_params(group)
        if not np.isfinite(sigma) or not np.isfinite(tie_threshold):
            continue
        scored_group = score_probabilities(group, sigma, tie_threshold)
        scored_pieces.append(scored_group)
        summary = proper_score_summary(scored_group, "discovery")
        if not summary.empty:
            score_rows.extend(summary.to_dict(orient="records"))
    scores = pd.DataFrame(score_rows)
    selected: dict[str, float] = {}
    # Pooled event-equal market loss is the prespecified first criterion.  Ties
    # choose the stronger regularization.
    for candidate in CANDIDATES:
        prefix = f"{candidate.name}__alpha_"
        subset = scores.loc[scores["candidate"].str.startswith(prefix, na=False)].copy()
        if subset.empty:
            raise ValueError(f"No discovery market scores for {candidate.name}")
        pooled = subset.groupby("candidate", observed=True)["event_equal_log_loss"].mean()
        best_loss = pooled.min()
        tied = pooled.loc[pooled <= best_loss + 1e-10].index.tolist()
        alphas = {
            name: float(name.rsplit("_", 1)[-1])
            for name in tied
        }
        selected[candidate.name] = max(alphas.values())
    for candidate in PAIRWISE_CANDIDATES:
        prefix = f"{candidate.name}__c_"
        subset = scores.loc[
            scores["candidate"].str.startswith(prefix, na=False)
            & scores["market"].eq("72h")
        ].copy()
        if subset.empty:
            raise ValueError(f"No discovery pairwise market scores for {candidate.name}")
        pooled = subset.groupby("candidate", observed=True)["event_equal_log_loss"].mean()
        best_loss = pooled.min()
        tied = pooled.loc[pooled <= best_loss + 1e-10].index.tolist()
        strengths = {name: float(name.rsplit("_", 1)[-1]) for name in tied}
        # Smaller C is the more regularized tie-break.
        selected[candidate.name] = min(strengths.values())
    scored = pd.concat(scored_pieces, ignore_index=True) if scored_pieces else pd.DataFrame()
    return scores, scored, selected


def build_final_predictions(
    snapshots: pd.DataFrame,
    selected_alphas: Mapping[str, float],
    quotes: pd.DataFrame,
    core_bridge_path: Path,
) -> pd.DataFrame:
    years = tuple(range(2021, 2027))
    pieces = [
        legacy_predictions(snapshots),
        legacy_refit_predictions(snapshots, years),
        retrospective_core_predictions(snapshots, core_bridge_path),
    ]
    for candidate in CANDIDATES:
        pieces.append(
            annual_walk_forward_predictions(
                snapshots,
                candidate,
                selected_alphas[candidate.name],
                years=years,
            )
        )
    for candidate in PAIRWISE_CANDIDATES:
        pieces.append(
            pairwise_player_scores(
                snapshots,
                quotes,
                candidate,
                selected_alphas[candidate.name],
                years=years,
            )
        )
    output = pd.concat(pieces, ignore_index=True)
    # Keep evaluation years; warm-up predictions were never requested.
    return output.loc[output["event_start_date"].dt.year.isin(years)].copy()


def fit_discovery_sigmas(
    quotes: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_paired = attach_skill_differences(quotes, predictions)
    quote_counts = (
        all_paired.groupby("candidate", observed=True)["quote_id"].nunique().astype(int)
    )
    total_quotes = int(quotes["quote_id"].nunique())
    common_audit = {
        "candidate_count": int(predictions["candidate"].nunique()),
        "total_normalized_quotes": total_quotes,
        "quotes_by_candidate": {str(key): int(value) for key, value in quote_counts.items()},
    }
    paired = all_paired.loc[
        all_paired["event_start_date"].between(DISCOVERY_START, DISCOVERY_END)
    ].copy()
    sigma_rows: list[dict[str, Any]] = []
    for (candidate, market), group in paired.groupby(["candidate", "market"], observed=True):
        sigma, tie_threshold = fit_probability_params(group)
        sigma_rows.append(
            {
                "candidate": candidate,
                "market": market,
                "sigma": sigma,
                "tie_threshold": tie_threshold,
                "sigma_near_upper_bound": bool(
                    np.isfinite(sigma) and sigma >= 99.0
                ),
                "discovery_quotes": int(len(group)),
                "discovery_resolved_quotes": int(len(_resolved_quotes(group))),
                "discovery_events": int(group["mapped_event_key"].nunique()),
            }
        )
    sigmas = pd.DataFrame(sigma_rows)
    scored = all_paired.merge(
        sigmas[["candidate", "market", "sigma", "tie_threshold"]],
        on=["candidate", "market"],
    )
    win, push, _ = _win_push_probabilities(
        scored["skill_difference"].to_numpy(float),
        scored["sigma"].to_numpy(float),
        scored["tie_threshold"].to_numpy(float),
    )
    scored["p1_probability"] = np.clip(win, EPS, 1.0 - EPS)
    scored["push_probability"] = np.clip(push, 0.0, 1.0)
    scored["p1_pricing_probability"] = (
        scored["p1_probability"] / (1.0 - scored["push_probability"])
    ).clip(EPS, 1.0 - EPS)
    return sigmas, scored, common_audit


def _sample_frame(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "discovery":
        mask = frame["event_start_date"].between(DISCOVERY_START, DISCOVERY_END)
    elif sample == "sealed":
        mask = frame["event_start_date"].between(SEALED_START, SEALED_END)
    elif sample == "comparable_3y":
        mask = frame["event_start_date"].between(COMPARABLE_START, SEALED_END)
    else:
        raise ValueError(sample)
    return frame.loc[mask].copy()


def choose_discovery_shortlist(
    scored: pd.DataFrame,
    explicit: Sequence[str],
    shortlist_size: int,
) -> tuple[list[str], pd.DataFrame]:
    available = set(scored["candidate"].unique())
    if explicit:
        missing = sorted(set(explicit) - available)
        if missing:
            raise ValueError(f"Requested shortlist candidates are unavailable: {missing}")
        shortlist = list(dict.fromkeys(explicit))
    else:
        discovery = _sample_frame(scored, "discovery")
        summary = proper_score_summary(discovery, "discovery")
        eligible = summary.loc[
            ~summary["candidate"].isin(
                [BASELINE_CANDIDATE, RAW_LEGACY_CANDIDATE, RETROSPECTIVE_CORE_CANDIDATE]
            )
        ].copy()
        pooled = (
            eligible.groupby("candidate", observed=True)["event_equal_log_loss"]
            .mean()
            .sort_values()
        )
        sg_rank = [name for name in pooled.index if not name.startswith("pairwise_")]
        pair_rank = [name for name in pooled.index if name.startswith("pairwise_")]
        sg_slots = shortlist_size if shortlist_size <= 2 else shortlist_size - 1
        shortlist = sg_rank[:sg_slots]
        if shortlist_size > 2 and pair_rank:
            shortlist.append(pair_rank[0])
        for name in pooled.index:
            if len(shortlist) >= shortlist_size:
                break
            if name not in shortlist:
                shortlist.append(name)
    discovery_scores = proper_score_summary(
        _sample_frame(scored, "discovery"), "discovery"
    )
    discovery_scores["shortlisted_for_money"] = discovery_scores["candidate"].isin(shortlist)
    return shortlist, discovery_scores


def _write_report(
    output: Path,
    selected_alphas: Mapping[str, float],
    proper: pd.DataFrame,
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    def frame_block(frame: pd.DataFrame, *, index: bool = True) -> str:
        if frame.empty:
            return "No rows."
        return "```text\n" + frame.to_string(index=index, float_format=lambda value: f"{value:.6f}") + "\n```"

    sealed_scores = proper.loc[proper["sample"] == "sealed"].copy()
    score_table = sealed_scores.pivot_table(
        index="candidate", columns="market", values="event_equal_log_loss"
    ).sort_index()
    if performance.empty:
        primary_perf = pd.DataFrame()
        pnl_table = pd.DataFrame()
    else:
        primary_perf = performance.loc[
            (performance["sample"] == "sealed")
            & (performance["line_policy"] == "best_of_three")
            & (performance["sizing"] == "flat_1u")
            & (performance["opportunity_mode"] == "all_available")
        ].copy()
        pnl_table = primary_perf.pivot_table(
            index="candidate", columns="market", values=["bets", "pnl", "roi"]
        ).sort_index()
    if "opportunity_mode" not in bootstrap:
        boot = pd.DataFrame()
    else:
        boot = bootstrap.loc[
            (bootstrap["sample"] == "sealed")
            & (bootstrap["line_policy"] == "best_of_three")
            & (bootstrap["sizing"] == "flat_1u")
            & (bootstrap["metric"] == "pnl_delta")
            & bootstrap["opportunity_mode"].astype(str).str.startswith("paired_common::")
        ]
    lines = [
        "# Base-skill market model — analytic first-stage screen",
        "",
        "Research only. No production file or consumer was changed.",
        "",
        "## Protocol",
        "",
        "- Features use only rounds strictly before each event start.",
        "- Ridge strength was selected on 2021-22 matchup log loss; 2023-2026 is sealed.",
        "- Model coefficients expand annually and are frozen before each calendar year.",
        "- Head-to-head variance was calibrated on discovery only, separately by model and market.",
        "- Closing prices were used only for evaluation, EV gates, and settlement.",
        "- W1 uses fold-estimated correction of prior observations; it never uses target-round weather.",
        "- Clean arms are DG-only. All current IMG archives are excluded because their transforms were selected after the discovery cutoff.",
        "",
        "## Selected ridge penalties",
        "",
        "```json",
        json.dumps(dict(selected_alphas), indent=2, sort_keys=True),
        "```",
        "",
        "## Sealed proper scores",
        "",
        frame_block(score_table) if not score_table.empty else "No sealed scores.",
        "",
        "## Sealed best-of-three, flat-risk results",
        "",
        frame_block(pnl_table) if not pnl_table.empty else "No sealed bets.",
        "",
        "## Paired event-bootstrap P&L deltas versus legacy",
        "",
        boot[["candidate", "market", "estimate", "ci_low", "ci_high", "probability_better"]]
        .sort_values(["market", "estimate"], ascending=[True, False])
        .pipe(frame_block, index=False)
        if not boot.empty
        else "No bootstrap results.",
        "",
        "## Integrity audit",
        "",
        "```json",
        json.dumps(dict(audit), indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Scope limits",
        "",
        "This is an analytic base-only matchup screen, not a full tournament Monte Carlo. It does not yet retain/refit downstream course terms, evaluate finish markets, rebuild the R2-R4 cascade, extend history across Euro/KFT/LIV, or add strict-prior age/pro-experience. A leakage-safe stroke-level rebuild is also still required before IMG can re-enter. Calendar-year refits are point-in-time but intentionally less adaptive than production's weekly cadence. Market fills, limits, and simultaneous best prices are not archived.",
        "",
    ]
    (output / "MODEL_EVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    panel_dir = args.panel_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rounds, events = load_panels(panel_dir)
    print(f"Loaded {len(rounds):,} player-rounds and {len(events):,} player-events")
    catalog = _event_catalog(rounds, events)
    quotes, event_mapping = load_matchup_quotes(
        args.odds_cache.resolve(),
        catalog,
        panel_dir / "odds_event_crosswalk.parquet",
    )
    events, market_only_targets = augment_events_from_market_board(events, quotes, catalog)
    print(
        f"Added {len(market_only_targets):,} quoted event-player targets with no result row"
    )
    if args.reuse_snapshots_from is not None:
        reuse = args.reuse_snapshots_from.resolve()
        weather = pd.read_csv(reuse / "weather_coefficients.csv", parse_dates=[
            "event_start_date", "fit_max_round_date"
        ])
        snapshots = pd.read_parquet(reuse / "feature_snapshots.parquet")
        print(f"Reused strict-prior snapshots from {reuse}")
    else:
        target_dates = sorted(events["event_start_date"].dropna().unique())
        weather = estimate_weather_coefficients(rounds, target_dates)
        print(f"Estimated point-in-time weather coefficients for {len(weather):,} event dates")
        snapshots = build_feature_snapshots(rounds, events, weather)
    weather["event_start_date"] = pd.to_datetime(weather["event_start_date"]).dt.normalize()
    snapshots["event_start_date"] = pd.to_datetime(snapshots["event_start_date"]).dt.normalize()
    snapshots["event_end_date"] = pd.to_datetime(snapshots["event_end_date"], errors="coerce").dt.normalize()
    snapshots = attach_weather_neutral_labels(snapshots, rounds, weather)
    weather.to_csv(output / "weather_coefficients.csv", index=False)
    contaminated_feature_usage = {
        candidate.name: [
            column
            for column in candidate_feature_columns(snapshots, candidate)
            if column.startswith("img_")
        ]
        for candidate in (*CANDIDATES, *PAIRWISE_CANDIDATES)
    }
    contaminated_feature_usage = {
        candidate: columns
        for candidate, columns in contaminated_feature_usage.items()
        if columns
    }
    if contaminated_feature_usage:
        raise AssertionError(
            "Clean candidate selected future-contaminated IMG total/category features: "
            f"{contaminated_feature_usage}"
        )
    snapshots.to_parquet(output / "feature_snapshots.parquet", index=False)
    print(f"Built {len(snapshots):,} strict-prior feature snapshots")
    event_mapping.to_csv(output / "odds_event_mapping.csv", index=False)
    quotes.to_parquet(output / "normalized_matchup_quotes.parquet", index=False)
    print(
        f"Mapped {quotes['mapped_event_key'].nunique():,} odds events and "
        f"{len(quotes):,} 72-hole/R1 book quotes"
    )
    if args.reuse_models_from is not None:
        model_reuse = args.reuse_models_from.resolve()
        with (model_reuse / "selected_hyperparameters.json").open(
            "r", encoding="utf-8"
        ) as handle:
            selected_alphas = json.load(handle)
        discovery_scores = pd.read_csv(model_reuse / "discovery_alpha_scores.csv")
        valid_candidates = {
            BASELINE_CANDIDATE,
            RAW_LEGACY_CANDIDATE,
            RETROSPECTIVE_CORE_CANDIDATE,
            *(candidate.name for candidate in CANDIDATES),
            *(candidate.name for candidate in PAIRWISE_CANDIDATES),
        }
        predictions = pd.read_parquet(model_reuse / "player_event_predictions.parquet")
        predictions = predictions.loc[
            predictions["candidate"].isin(valid_candidates)
        ].copy()
        if args.shortlist:
            bounded_candidates = set(args.shortlist) | {
                BASELINE_CANDIDATE,
                RAW_LEGACY_CANDIDATE,
                RETROSPECTIVE_CORE_CANDIDATE,
            }
            predictions = predictions.loc[
                predictions["candidate"].isin(bounded_candidates)
            ].copy()
        print(f"Reused discovery-selected models from {model_reuse}")
    else:
        discovery_scores, discovery_scored, selected_alphas = fit_candidate_alphas(
            snapshots, quotes
        )
        discovery_scored.to_parquet(
            output / "discovery_alpha_market_predictions.parquet", index=False
        )
        predictions = build_final_predictions(
            snapshots,
            selected_alphas,
            quotes,
            args.core_bridge.resolve(),
        )
    discovery_scores.to_csv(output / "discovery_alpha_scores.csv", index=False)
    _atomic_json(output / "selected_hyperparameters.json", selected_alphas)
    predictions.to_parquet(output / "player_event_predictions.parquet", index=False)
    sigmas, scored, common_menu_audit = fit_discovery_sigmas(quotes, predictions)
    sigmas.to_csv(output / "discovery_probability_sigmas.csv", index=False)
    scored.to_parquet(output / "market_probabilities.parquet", index=False)

    shortlist, shortlist_scores = choose_discovery_shortlist(
        scored, args.shortlist, args.shortlist_size
    )
    shortlist_scores.to_csv(output / "discovery_shortlist_scores.csv", index=False)
    _atomic_json(
        output / "money_shortlist.json",
        {
            "selection_window_end": "2022-12-31",
            "sealed_data_used_for_selection": False,
            "shortlist": shortlist,
            "controls": [
                BASELINE_CANDIDATE,
                RAW_LEGACY_CANDIDATE,
                RETROSPECTIVE_CORE_CANDIDATE,
            ],
        },
    )
    money_candidates = set(shortlist) | {
        BASELINE_CANDIDATE,
        RAW_LEGACY_CANDIDATE,
        RETROSPECTIVE_CORE_CANDIDATE,
    }

    proper_pieces: list[pd.DataFrame] = []
    calibration_pieces: list[pd.DataFrame] = []
    bet_pieces: list[pd.DataFrame] = []
    edge_pieces: list[pd.DataFrame] = []
    bootstrap_pieces: list[pd.DataFrame] = []
    performance_pieces: list[pd.DataFrame] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_scored = _sample_frame(scored, sample)
        proper_pieces.append(proper_score_summary(sample_scored, sample))
        calibration_pieces.append(calibration_summary(sample_scored, sample))
        bootstrap_pieces.append(
            paired_event_bootstrap(
                sample_scored,
                pd.DataFrame(),
                sample,
                draws=args.bootstrap_draws,
                seed=args.seed + len(bootstrap_pieces) * 1009,
            )
        )
        if args.skip_betting:
            continue
        money_scored = sample_scored.loc[
            sample_scored["candidate"].isin(money_candidates)
        ].copy()
        gated = money_scored.loc[money_scored["candidate"].isin(shortlist)].copy()
        gated["candidate"] = gated["candidate"] + "__reliability_gate"
        betting_scored = pd.concat([money_scored, gated], ignore_index=True)
        all_available_parts: list[pd.DataFrame] = []
        for policy in args.policies:
            edge_pieces.append(edge_bucket_performance(betting_scored, sample, policy))
            for sizing in args.sizings:
                selected = select_and_settle_bets(betting_scored, policy, sizing)
                if not selected.empty:
                    selected["sample"] = sample
                    selected["line_policy"] = policy
                    selected["opportunity_mode"] = "all_available"
                    all_available_parts.append(selected)
                    bet_pieces.append(selected)
        if all_available_parts:
            combined_all = pd.concat(all_available_parts, ignore_index=True)
            performance_pieces.append(betting_performance(combined_all, sample))
            all_bootstrap = paired_event_bootstrap(
                money_scored,
                combined_all,
                sample,
                draws=args.bootstrap_draws,
                seed=args.seed + 5003 + len(bootstrap_pieces) * 1009,
            )
            bootstrap_pieces.append(
                all_bootstrap.loc[all_bootstrap["metric"] == "pnl_delta"].copy()
            )
        # Exact candidate-vs-baseline common-opportunity portfolios.  These are
        # summarized and discarded one challenger at a time.  Repeated baseline
        # ticket frames are never retained in the final bets artifact.
        challenger_names = sorted(
            candidate
            for candidate in betting_scored["candidate"].unique()
            if candidate != BASELINE_CANDIDATE
        )
        for challenger in challenger_names:
            pair = betting_scored.loc[
                betting_scored["candidate"].isin([BASELINE_CANDIDATE, challenger])
            ].copy()
            coverage = pair.groupby("quote_id", observed=True)["candidate"].nunique()
            common_ids = coverage.loc[coverage == 2].index
            pair = pair.loc[pair["quote_id"].isin(common_ids)].copy()
            opportunity_mode = f"paired_common::{challenger}"
            pair_parts: list[pd.DataFrame] = []
            for policy in args.policies:
                for sizing in args.sizings:
                    selected = select_and_settle_bets(pair, policy, sizing)
                    if selected.empty:
                        continue
                    selected["sample"] = sample
                    selected["line_policy"] = policy
                    selected["opportunity_mode"] = opportunity_mode
                    pair_parts.append(selected)
            if pair_parts:
                combined_pair = pd.concat(pair_parts, ignore_index=True)
                performance_pieces.append(betting_performance(combined_pair, sample))
                pair_bootstrap = paired_event_bootstrap(
                    pair,
                    combined_pair,
                    sample,
                    draws=args.bootstrap_draws,
                    seed=args.seed + 9001 + len(bootstrap_pieces) * 1009,
                )
                bootstrap_pieces.append(
                    pair_bootstrap.loc[pair_bootstrap["metric"] == "pnl_delta"].copy()
                )
    proper = pd.concat(proper_pieces, ignore_index=True)
    calibration = pd.concat(calibration_pieces, ignore_index=True)
    bets = pd.concat(bet_pieces, ignore_index=True) if bet_pieces else pd.DataFrame()
    performance = (
        pd.concat(performance_pieces, ignore_index=True)
        if performance_pieces
        else pd.DataFrame()
    )
    offered_rows: list[dict[str, Any]] = []
    for sample in ("discovery", "sealed", "comparable_3y"):
        sample_scored = _sample_frame(scored, sample)
        for market, group in sample_scored.groupby("market", observed=True):
            offered_rows.append(
                {
                    "sample": sample,
                    "market": market,
                    "offered_events": int(group["mapped_event_key"].nunique()),
                }
            )
    offered = pd.DataFrame(offered_rows)
    if not performance.empty:
        performance = performance.merge(
            offered,
            left_on=["sample", "market"],
            right_on=["sample", "market"],
            how="left",
            validate="many_to_one",
        )
        performance["zero_bet_events"] = performance["offered_events"] - performance["events"]
    edge = pd.concat(edge_pieces, ignore_index=True) if edge_pieces else pd.DataFrame()
    bootstrap = pd.concat(bootstrap_pieces, ignore_index=True)
    proper.to_csv(output / "proper_scores.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    if not bets.empty:
        bets.to_parquet(output / "bets.parquet", index=False)
    performance.to_csv(output / "performance.csv", index=False)
    edge.to_csv(output / "edge_bucket_performance.csv", index=False)
    bootstrap.to_csv(output / "event_cluster_bootstrap.csv", index=False)

    prediction_players = predictions.groupby("candidate", observed=True).size().to_dict()
    mapped_player_pairs = scored.groupby("candidate", observed=True)["quote_id"].nunique().to_dict()
    audit = {
        "round_rows": int(len(rounds)),
        "event_rows": int(len(events)),
        "market_only_event_player_targets": int(len(market_only_targets)),
        "feature_snapshots": int(len(snapshots)),
        "strict_prior_violations": int(
            (
                pd.to_datetime(snapshots["max_source_round_date"], errors="coerce")
                >= snapshots["event_start_date"]
            ).fillna(False).sum()
        ),
        "weather_fit_boundary_violations": int(
            (
                pd.to_datetime(weather["fit_max_round_date"], errors="coerce")
                >= weather["event_start_date"]
            ).fillna(False).sum()
        ),
        "odds_events_total": int(len(event_mapping)),
        "odds_events_unmapped_or_ambiguous": int(
            event_mapping["mapped_event_key"].isna().sum()
        ),
        "odds_event_mapping_methods": event_mapping["mapping_method"].value_counts(dropna=False).to_dict(),
        "normalized_quotes": int(len(quotes)),
        "normalized_quote_events": int(quotes["mapped_event_key"].nunique()),
        "player_event_prediction_rows": {str(k): int(v) for k, v in prediction_players.items()},
        "mapped_quote_rows_by_candidate": {str(k): int(v) for k, v in mapped_player_pairs.items()},
        "common_market_menu": common_menu_audit,
        "duplicate_prediction_keys": int(
            predictions.duplicated(["candidate", "event_key", "dg_id_key"]).sum()
        ),
        "retrospective_img_v2_used": False,
        "retrospective_img_accounting_used": False,
        "clean_img_components": [],
        "img_accounting_provenance_warning": (
            "All v4 accounting categories inherit a lie-code map selected on the "
            "full archive in 2026. They are excluded pending a pre-2021 frozen-curve rebuild."
        ),
        "closing_price_used_as_feature": False,
        "target_round_weather_used": False,
        "selection_period": "2021-01-01/2022-12-31",
        "sealed_period": "2023-01-01/2026-08-22",
        "refit_schedule": "calendar_year_expanding",
        "bootstrap_draws": int(args.bootstrap_draws),
        "broad_candidate_count": int(scored["candidate"].nunique()),
        "money_shortlist": shortlist,
        "skip_betting": bool(args.skip_betting),
        "money_line_policies": list(args.policies),
        "money_sizings": list(args.sizings),
        "sigma_near_upper_bound_count": int(
            sigmas["sigma_near_upper_bound"].sum()
        ),
        "bet_materialization": (
            "none"
            if args.skip_betting
            else "shortlist_all_available_only; paired_common_summaries_streamed"
        ),
        "seed": int(args.seed),
    }
    if audit["strict_prior_violations"] or audit["weather_fit_boundary_violations"]:
        raise AssertionError(f"Point-in-time audit failed: {audit}")
    _atomic_json(output / "audit.json", audit)
    _write_report(output, selected_alphas, proper, performance, bootstrap, audit)

    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    manifest = {
        "complete": True,
        "research_only": True,
        "contract": str(REPO / "BASE_SKILL_MARKET_REBUILD_CONTRACT_2026_08.md"),
        "inputs": {
            "round_panel": {
                "path": str(panel_dir / "player_round_panel.parquet"),
                "sha256": _sha256(panel_dir / "player_round_panel.parquet"),
            },
            "event_panel": {
                "path": str(panel_dir / "player_event_outcomes.parquet"),
                "sha256": _sha256(panel_dir / "player_event_outcomes.parquet"),
            },
            "odds_cache_manifest": {
                "path": str(args.odds_cache.resolve() / "manifest.json"),
                "sha256": _sha256(args.odds_cache.resolve() / "manifest.json"),
            },
            "legacy_core_bridge": {
                "path": str(args.core_bridge.resolve()),
                "sha256": _sha256(args.core_bridge.resolve()),
            },
            "odds_event_crosswalk": {
                "path": str(panel_dir / "odds_event_crosswalk.parquet"),
                "sha256": _sha256(panel_dir / "odds_event_crosswalk.parquet"),
            },
        },
        "selected_alphas": selected_alphas,
        "money_shortlist": shortlist,
        "run_type": "proper_score_screen" if args.skip_betting else "shortlist_money_screen",
        "artifacts": artifacts,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(f"Completed research screen: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--odds-cache", type=Path, default=DEFAULT_ODDS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--core-bridge", type=Path, default=DEFAULT_CORE_BRIDGE)
    parser.add_argument("--reuse-snapshots-from", type=Path)
    parser.add_argument("--reuse-models-from", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--shortlist", type=lambda value: tuple(
        item.strip() for item in value.split(",") if item.strip()
    ), default=())
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--skip-betting", action="store_true")
    parser.add_argument(
        "--policies",
        type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()),
        default=(*BOOKS, "best_of_three"),
    )
    parser.add_argument(
        "--sizings",
        type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()),
        default=("flat_1u", "production_risk_to_win"),
    )
    parser.add_argument("--seed", type=int, default=20260822)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
